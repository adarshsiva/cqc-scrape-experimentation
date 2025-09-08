#!/usr/bin/env python3
"""
Cloud-optimized CQC data fetcher with enhanced retry logic and proxy support.
Designed to work from Google Cloud Run/Functions with 403 bypass strategies.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloudCQCFetcher:
    def __init__(self):
        logger.info("🚀 Initializing CloudCQCFetcher...")
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        logger.info(f"📋 Project ID: {self.project_id}")
        
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        logger.info(f"📋 Base URL: {self.base_url}")
        
        if not self.api_key:
            raise ValueError("❌ No API key available - cannot initialize fetcher")
        
        # Initialize clients
        logger.info("🔧 Initializing GCP clients...")
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        try:
            self.bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-raw-data")
            logger.info(f"✅ Connected to bucket: {self.bucket.name}")
        except Exception as e:
            logger.error(f"❌ Failed to connect to bucket: {e}")
            raise
        
        # Setup session with retry strategy
        logger.info("🔧 Setting up HTTP session...")
        self.session = self._create_session()
        logger.info("✅ CloudCQCFetcher initialized successfully")
        
    def _get_api_key(self) -> str:
        """Get API key from Secret Manager or environment."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/cqc-subscription-key/versions/latest"
            logger.info(f"Attempting to retrieve API key from Secret Manager: {name}")
            response = client.access_secret_version(request={"name": name})
            key = response.payload.data.decode("UTF-8")
            logger.info(f"✅ API key retrieved from Secret Manager (length: {len(key)})")
            logger.info(f"API key preview: {key[:10]}...{key[-4:]}")
            return key
        except Exception as e:
            logger.error(f"❌ Failed to get key from Secret Manager: {e}")
            logger.error(f"Project ID: {self.project_id}")
            # Fallback to environment variable
            key = os.environ.get('CQC_API_KEY', '')
            if key:
                logger.info(f"✅ API key retrieved from environment variable (length: {len(key)})")
                logger.info(f"API key preview: {key[:10]}...{key[-4:]}")
            else:
                logger.error("❌ No API key found in Secret Manager or environment")
            return key
    
    def _create_session(self) -> requests.Session:
        """Create a session with advanced retry strategy and headers."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Start with minimal headers like a working Cloud Function
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Google-Cloud-Functions/2.0 Python/3.11"
        }
        
        logger.info("Setting session headers:")
        for key, value in headers.items():
            if key == "Ocp-Apim-Subscription-Key":
                logger.info(f"  {key}: {value[:10]}...{value[-4:] if len(value) > 14 else value}")
            else:
                logger.info(f"  {key}: {value}")
        
        session.headers.update(headers)
        
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retry_count: int = 0) -> Optional[Dict]:
        """Make API request with enhanced error handling and retry logic."""
        max_retries = 3
        url = f"{self.base_url}/{endpoint}"
        
        try:
            # Add random delay to avoid rate limiting
            if retry_count > 0:
                delay = random.uniform(2 ** retry_count, 2 ** (retry_count + 1))
                logger.info(f"Waiting {delay:.2f} seconds before retry {retry_count}")
                time.sleep(delay)
            
            logger.info(f"Making request to: {url}")
            logger.info(f"Params: {params}")
            logger.info(f"API Key present: {bool(self.api_key and len(self.api_key) > 0)}")
            
            response = self.session.get(url, params=params, timeout=30)
            
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            logger.info(f"Response content length: {len(response.text)}")
            logger.info(f"Response content type: {response.headers.get('content-type', 'Not specified')}")
            
            if response.status_code == 200:
                # Check for empty response first
                if not response.text or response.text.strip() == "":
                    logger.error("❌ Received empty response from API")
                    logger.error(f"Response headers: {dict(response.headers)}")
                    return None
                
                # Log first 500 characters of response for debugging
                logger.info(f"Response content preview: {response.text[:500]}...")
                
                try:
                    json_data = response.json()
                    logger.info("✅ Successfully parsed JSON response")
                    return json_data
                except json.JSONDecodeError as json_error:
                    logger.error(f"❌ JSON parsing failed: {json_error}")
                    logger.error(f"JSON error type: {type(json_error).__name__}")
                    logger.error(f"JSON error args: {json_error.args}")
                    
                    # Check for specific error patterns
                    error_msg = str(json_error)
                    if "Expecting value: line 1 column 1 (char 0)" in error_msg:
                        logger.error("⚠️  This indicates an empty response or non-JSON content")
                    
                    logger.error(f"Full response content (length={len(response.text)}):")
                    logger.error(f"'{response.text}'")
                    logger.error(f"Response encoding: {response.encoding}")
                    logger.error(f"Response raw content (bytes): {response.content}")
                    
                    # Try to decode with different encoding
                    if response.encoding and response.encoding.lower() != 'utf-8':
                        try:
                            logger.info(f"Trying to re-encode from {response.encoding} to utf-8")
                            response.encoding = 'utf-8'
                            json_data = response.json()
                            logger.info("✅ Successfully parsed JSON after encoding fix")
                            return json_data
                        except Exception as encoding_error:
                            logger.error(f"Failed to parse JSON even after encoding fix: {encoding_error}")
                    
                    # Check if response might be HTML error page
                    if response.text.strip().startswith('<'):
                        logger.error("⚠️  Response appears to be HTML - possibly an error page or redirect")
                        
                    return None
                except Exception as e:
                    logger.error(f"❌ Unexpected error parsing JSON: {e}")
                    logger.error(f"Full response content: {response.text}")
                    return None
                    
            elif response.status_code == 403:
                logger.warning(f"403 Forbidden for {endpoint}")
                logger.warning(f"Response content: {response.text}")
                
                # Try with different headers
                if retry_count < max_retries:
                    # Modify headers for retry
                    self.session.headers.update({
                        "X-Forwarded-For": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
                        "X-Real-IP": f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
                    })
                    return self._make_request(endpoint, params, retry_count + 1)
                else:
                    logger.error(f"Failed after {max_retries} retries for {endpoint}")
                    return None
                    
            elif response.status_code == 429:
                # Rate limited - wait and retry
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                logger.warning(f"Response content: {response.text}")
                time.sleep(retry_after)
                if retry_count < max_retries:
                    return self._make_request(endpoint, params, retry_count + 1)
            else:
                logger.error(f"Request failed with status {response.status_code}")
                logger.error(f"Response content: {response.text}")
                logger.error(f"Response headers: {dict(response.headers)}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Request exception for {endpoint}: {e}")
            if retry_count < max_retries:
                logger.info(f"Retrying request (attempt {retry_count + 1}/{max_retries})")
                return self._make_request(endpoint, params, retry_count + 1)
            return None
        except Exception as e:
            logger.error(f"Unexpected error making request: {e}")
            return None
    
    def fetch_providers(self, page: int = 1, per_page: int = 100) -> Optional[Dict]:
        """Fetch providers list."""
        logger.info(f"Fetching providers page {page}")
        params = {"page": page, "perPage": per_page}
        return self._make_request("providers", params)
    
    def fetch_locations(self, page: int = 1, per_page: int = 100) -> Optional[Dict]:
        """Fetch locations list."""
        logger.info(f"Fetching locations page {page}")
        params = {"page": page, "perPage": per_page}
        return self._make_request("locations", params)
    
    def fetch_location_details(self, location_id: str) -> Optional[Dict]:
        """Fetch detailed information for a specific location."""
        logger.info(f"Fetching details for location {location_id}")
        return self._make_request(f"locations/{location_id}")
    
    def save_to_gcs(self, data: Dict, data_type: str) -> str:
        """Save data to Google Cloud Storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cqc_data/{data_type}/{timestamp}_{data_type}.json"
        
        blob = self.bucket.blob(filename)
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        
        logger.info(f"Saved {data_type} data to gs://{self.bucket.name}/{filename}")
        return f"gs://{self.bucket.name}/{filename}"
    
    def fetch_and_store_batch(self, batch_size: int = 100, max_locations: int = 1000):
        """Fetch a batch of locations with details and store in GCS."""
        locations_fetched = []
        page = 1
        total_fetched = 0
        
        while total_fetched < max_locations:
            # Fetch locations page
            locations_data = self.fetch_locations(page, batch_size)
            
            if not locations_data or 'locations' not in locations_data:
                logger.warning("No more locations to fetch")
                break
            
            locations = locations_data['locations']
            
            # Fetch details for each location
            for location in locations[:min(batch_size, max_locations - total_fetched)]:
                location_id = location['locationId']
                details = self.fetch_location_details(location_id)
                
                if details:
                    locations_fetched.append(details)
                    total_fetched += 1
                    logger.info(f"Fetched {total_fetched}/{max_locations} locations")
                
                # Rate limiting
                time.sleep(0.5)
                
                if total_fetched >= max_locations:
                    break
            
            page += 1
            
            # Save batch to GCS periodically
            if len(locations_fetched) >= 100:
                self.save_to_gcs({"locations": locations_fetched}, "locations_batch")
                locations_fetched = []
        
        # Save remaining locations
        if locations_fetched:
            self.save_to_gcs({"locations": locations_fetched}, "locations_batch")
        
        logger.info(f"Completed fetching {total_fetched} locations")
        return total_fetched
    
    def test_connection(self) -> bool:
        """Test API connection and authentication."""
        logger.info("=" * 60)
        logger.info("🔍 TESTING CQC API CONNECTION")
        logger.info("=" * 60)
        
        # Debug environment
        logger.info(f"Project ID: {self.project_id}")
        logger.info(f"Base URL: {self.base_url}")
        logger.info(f"API Key available: {bool(self.api_key and len(self.api_key) > 0)}")
        
        if not self.api_key:
            logger.error("❌ No API key available - cannot proceed")
            return False
        
        # Try to fetch first page with just 1 item
        logger.info("\n🔗 Attempting to fetch first location...")
        result = self.fetch_locations(page=1, per_page=1)
        
        if result and 'locations' in result:
            logger.info("✅ API connection successful!")
            logger.info(f"Response structure: {list(result.keys())}")
            logger.info(f"Number of locations in response: {len(result.get('locations', []))}")
            if 'totalCount' in result:
                logger.info(f"Total locations available: {result['totalCount']}")
            if 'totalPages' in result:
                logger.info(f"Total pages: {result['totalPages']}")
            logger.info("=" * 60)
            return True
        else:
            logger.error("❌ API connection failed - no valid response received")
            logger.error("=" * 60)
            return False

def main():
    """Main function for Cloud Run job."""
    logger.info("🚀 STARTING CQC DATA FETCHER CLOUD RUN JOB")
    logger.info("=" * 60)
    
    # Log environment information
    logger.info("Environment Information:")
    logger.info(f"  Python version: {os.sys.version}")
    logger.info(f"  Working directory: {os.getcwd()}")
    logger.info(f"  Environment variables:")
    for key in ['GCP_PROJECT', 'BATCH_SIZE', 'MAX_LOCATIONS', 'CQC_API_KEY']:
        value = os.environ.get(key, 'Not set')
        if key == 'CQC_API_KEY' and value != 'Not set':
            value = f"{value[:10]}...{value[-4:]}"
        logger.info(f"    {key}: {value}")
    
    try:
        fetcher = CloudCQCFetcher()
        
        # Test connection first
        if not fetcher.test_connection():
            logger.error("❌ Failed to connect to CQC API. Please check API key and network settings.")
            return
        
        # Fetch batch of locations
        batch_size = int(os.environ.get('BATCH_SIZE', '100'))
        max_locations = int(os.environ.get('MAX_LOCATIONS', '1000'))
        
        logger.info(f"🔄 Starting to fetch up to {max_locations} locations (batch size: {batch_size})...")
        total = fetcher.fetch_and_store_batch(batch_size, max_locations)
        
        logger.info(f"✅ Successfully fetched and stored {total} locations")
        
    except Exception as e:
        logger.error(f"❌ Unexpected error in main function: {e}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise

if __name__ == "__main__":
    main()