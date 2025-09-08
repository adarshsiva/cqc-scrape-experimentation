#!/usr/bin/env python3
"""
Enhanced CQC data fetcher for complete dataset ingestion.
Fetches all 118,000 locations with care home filtering and batch processing.
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery
import random
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CompleteCQCFetcher:
    """Complete CQC dataset fetcher with optimized batch processing."""
    
    # Care home related service types and specialisms
    CARE_HOME_SERVICE_TYPES = {
        'Care home service with nursing',
        'Care home service without nursing',
        'Specialist college service',
        'Residential substance misuse treatment and/or rehabilitation service'
    }
    
    CARE_HOME_SPECIALISMS = {
        'Caring for adults over 65 yrs',
        'Caring for adults under 65 yrs',
        'Dementia',
        'Mental health conditions',
        'Physical disabilities',
        'Sensory impairments',
        'Learning disabilities or autistic spectrum disorder',
        'Substance misuse problems',
        'Rehabilitation services',
        'Acquired brain injury'
    }
    
    def __init__(self):
        logger.info("Initializing CompleteCQCFetcher...")
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        
        if not self.api_key:
            raise ValueError("No API key available - cannot initialize fetcher")
        
        # Initialize GCP clients
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        # Setup buckets
        self.raw_bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-raw-data")
        self.processed_bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-processed")
        
        # Setup session
        self.session = self._create_session()
        
        # Track progress
        self.total_locations_fetched = 0
        self.care_homes_found = 0
        self.processed_location_ids = self._load_processed_ids()
        
        logger.info(f"Previously processed locations: {len(self.processed_location_ids)}")
        logger.info("CompleteCQCFetcher initialized successfully")
    
    def _get_api_key(self) -> str:
        """Get API key from Secret Manager or environment."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/cqc-subscription-key/versions/latest"
            response = client.access_secret_version(request={"name": name})
            key = response.payload.data.decode("UTF-8")
            logger.info(f"API key retrieved from Secret Manager")
            return key
        except Exception as e:
            logger.error(f"Failed to get key from Secret Manager: {e}")
            key = os.environ.get('CQC_API_KEY', '')
            if key:
                logger.info("API key retrieved from environment variable")
            return key
    
    def _create_session(self) -> requests.Session:
        """Create a session with retry strategy."""
        session = requests.Session()
        
        retry_strategy = Retry(
            total=5,
            backoff_factor=2,
            status_forcelist=[403, 429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            raise_on_status=False
        )
        
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "Google-Cloud-Run/2.0 Python/3.11"
        }
        session.headers.update(headers)
        
        return session
    
    def _load_processed_ids(self) -> Set[str]:
        """Load already processed location IDs from storage."""
        processed_ids = set()
        
        try:
            blob = self.processed_bucket.blob("metadata/processed_location_ids.json")
            if blob.exists():
                content = blob.download_as_text()
                data = json.loads(content)
                processed_ids = set(data.get('location_ids', []))
                logger.info(f"Loaded {len(processed_ids)} processed location IDs")
        except Exception as e:
            logger.warning(f"Could not load processed IDs: {e}")
        
        return processed_ids
    
    def _save_processed_ids(self):
        """Save processed location IDs to storage."""
        try:
            blob = self.processed_bucket.blob("metadata/processed_location_ids.json")
            data = {
                'location_ids': list(self.processed_location_ids),
                'last_updated': datetime.now().isoformat(),
                'total_count': len(self.processed_location_ids)
            }
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
            logger.info(f"Saved {len(self.processed_location_ids)} processed IDs")
        except Exception as e:
            logger.error(f"Failed to save processed IDs: {e}")
    
    def _is_care_home(self, location: Dict) -> bool:
        """Check if a location is a care home based on service types and specialisms."""
        # Check service types
        service_types = location.get('gacServiceTypes', [])
        for service in service_types:
            if isinstance(service, dict):
                service_name = service.get('name', '')
                if any(care_type in service_name for care_type in self.CARE_HOME_SERVICE_TYPES):
                    return True
        
        # Check specialisms
        specialisms = location.get('specialisms', [])
        for specialism in specialisms:
            if isinstance(specialism, dict):
                spec_name = specialism.get('name', '')
                if spec_name in self.CARE_HOME_SPECIALISMS:
                    return True
        
        # Check regulated activities for care home indicators
        activities = location.get('regulatedActivities', [])
        care_home_activities = ['Accommodation for persons who require nursing or personal care']
        for activity in activities:
            if isinstance(activity, dict):
                activity_name = activity.get('name', '')
                if activity_name in care_home_activities:
                    return True
        
        return False
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None, retry_count: int = 0) -> Optional[Dict]:
        """Make API request with error handling."""
        max_retries = 3
        url = f"{self.base_url}/{endpoint}"
        
        try:
            if retry_count > 0:
                delay = min(60, 2 ** retry_count)
                time.sleep(delay)
            
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                retry_after = int(response.headers.get('Retry-After', 60))
                logger.warning(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
                if retry_count < max_retries:
                    return self._make_request(endpoint, params, retry_count + 1)
            elif response.status_code == 403 and retry_count < max_retries:
                logger.warning(f"403 error, retrying with modified headers...")
                self.session.headers['X-Forwarded-For'] = f"{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}"
                return self._make_request(endpoint, params, retry_count + 1)
            else:
                logger.error(f"Request failed with status {response.status_code}")
                
        except Exception as e:
            logger.error(f"Request exception: {e}")
            if retry_count < max_retries:
                return self._make_request(endpoint, params, retry_count + 1)
        
        return None
    
    def fetch_all_locations_summary(self):
        """Fetch summary of all locations (without details) to get IDs and basic info."""
        logger.info("Fetching all location summaries...")
        all_locations = []
        page = 1
        per_page = 1000  # Maximum allowed
        
        while True:
            logger.info(f"Fetching page {page} (locations {(page-1)*per_page + 1} to {page*per_page})")
            
            data = self._make_request("locations", {"page": page, "perPage": per_page})
            
            if not data or 'locations' not in data:
                logger.info("No more locations to fetch")
                break
            
            locations = data['locations']
            all_locations.extend(locations)
            
            total_count = data.get('totalCount', 0)
            logger.info(f"Progress: {len(all_locations)}/{total_count} locations")
            
            if len(locations) < per_page or len(all_locations) >= total_count:
                break
            
            page += 1
            time.sleep(0.5)  # Rate limiting
        
        logger.info(f"Fetched {len(all_locations)} location summaries")
        return all_locations
    
    def fetch_location_details_batch(self, location_ids: List[str], max_workers: int = 5):
        """Fetch details for multiple locations in parallel."""
        location_details = []
        
        def fetch_single(location_id):
            if location_id in self.processed_location_ids:
                return None
            
            details = self._make_request(f"locations/{location_id}")
            if details:
                self.processed_location_ids.add(location_id)
            return details
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_single, loc_id): loc_id for loc_id in location_ids}
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        location_details.append(result)
                        self.total_locations_fetched += 1
                        
                        if self._is_care_home(result):
                            self.care_homes_found += 1
                        
                        if self.total_locations_fetched % 100 == 0:
                            logger.info(f"Progress: {self.total_locations_fetched} locations fetched, {self.care_homes_found} care homes found")
                    
                except Exception as e:
                    logger.error(f"Error fetching location: {e}")
        
        return location_details
    
    def save_batch_to_storage(self, data: List[Dict], batch_num: int, data_type: str = "locations"):
        """Save a batch of data to Cloud Storage in partitioned format."""
        if not data:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        date_partition = datetime.now().strftime("%Y/%m/%d")
        
        # Save raw data
        raw_filename = f"raw/{data_type}/{date_partition}/batch_{batch_num:05d}_{timestamp}.json"
        raw_blob = self.raw_bucket.blob(raw_filename)
        raw_blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        logger.info(f"Saved {len(data)} items to gs://{self.raw_bucket.name}/{raw_filename}")
        
        # Filter and save care homes separately
        care_homes = [loc for loc in data if self._is_care_home(loc)]
        if care_homes:
            care_home_filename = f"processed/care_homes/{date_partition}/batch_{batch_num:05d}_{timestamp}.json"
            care_home_blob = self.processed_bucket.blob(care_home_filename)
            care_home_blob.upload_from_string(
                json.dumps(care_homes, indent=2),
                content_type='application/json'
            )
            logger.info(f"Saved {len(care_homes)} care homes to gs://{self.processed_bucket.name}/{care_home_filename}")
        
        return len(data), len(care_homes)
    
    def load_to_bigquery(self, data: List[Dict], table_id: str):
        """Load data directly to BigQuery."""
        if not data:
            return
        
        try:
            dataset_id = "cqc_dataset"
            table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
            
            # Configure load job
            job_config = bigquery.LoadJobConfig(
                source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
                autodetect=True,
                write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
                schema_update_options=[
                    bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
                    bigquery.SchemaUpdateOption.ALLOW_FIELD_RELAXATION
                ]
            )
            
            # Convert to newline-delimited JSON
            ndjson_data = '\n'.join([json.dumps(item) for item in data])
            
            # Load to BigQuery
            job = self.bigquery_client.load_table_from_json(
                data, table_ref, job_config=job_config
            )
            job.result()  # Wait for job to complete
            
            logger.info(f"Loaded {len(data)} records to BigQuery table {table_ref}")
            
        except Exception as e:
            logger.error(f"Failed to load to BigQuery: {e}")
    
    def fetch_complete_dataset(self, batch_size: int = 500, resume: bool = True):
        """
        Fetch the complete CQC dataset with efficient batching and storage.
        
        Args:
            batch_size: Number of locations to fetch in each batch
            resume: Whether to resume from previous progress
        """
        logger.info("=" * 80)
        logger.info("STARTING COMPLETE CQC DATASET INGESTION")
        logger.info("=" * 80)
        
        # First, get all location IDs
        logger.info("Phase 1: Fetching all location summaries...")
        all_locations = self.fetch_all_locations_summary()
        
        # Filter out already processed if resuming
        if resume:
            location_ids = [loc['locationId'] for loc in all_locations 
                          if loc['locationId'] not in self.processed_location_ids]
            logger.info(f"Resuming: {len(location_ids)} locations remaining to process")
        else:
            location_ids = [loc['locationId'] for loc in all_locations]
            logger.info(f"Processing all {len(location_ids)} locations")
        
        # Save location summary
        summary_blob = self.processed_bucket.blob("metadata/location_summary.json")
        summary_data = {
            'total_locations': len(all_locations),
            'locations_to_process': len(location_ids),
            'timestamp': datetime.now().isoformat(),
            'location_ids': location_ids[:1000]  # Save first 1000 for reference
        }
        summary_blob.upload_from_string(
            json.dumps(summary_data, indent=2),
            content_type='application/json'
        )
        
        # Process in batches
        logger.info(f"Phase 2: Fetching detailed information for {len(location_ids)} locations...")
        batch_num = 0
        
        for i in range(0, len(location_ids), batch_size):
            batch_ids = location_ids[i:i+batch_size]
            batch_num += 1
            
            logger.info(f"\nProcessing batch {batch_num} ({i+1} to {min(i+batch_size, len(location_ids))} of {len(location_ids)})")
            
            # Fetch details for batch
            location_details = self.fetch_location_details_batch(batch_ids, max_workers=5)
            
            if location_details:
                # Save to Cloud Storage
                total_saved, care_homes_saved = self.save_batch_to_storage(location_details, batch_num)
                
                # Load to BigQuery
                self.load_to_bigquery(location_details, "locations_complete")
                
                # Filter care homes and load separately
                care_homes = [loc for loc in location_details if self._is_care_home(loc)]
                if care_homes:
                    self.load_to_bigquery(care_homes, "care_homes")
            
            # Save progress periodically
            if batch_num % 10 == 0:
                self._save_processed_ids()
                logger.info(f"Progress checkpoint: {self.total_locations_fetched} total, {self.care_homes_found} care homes")
        
        # Final save
        self._save_processed_ids()
        
        # Generate final report
        self._generate_final_report()
        
        logger.info("=" * 80)
        logger.info("COMPLETE CQC DATASET INGESTION FINISHED")
        logger.info(f"Total locations fetched: {self.total_locations_fetched}")
        logger.info(f"Care homes identified: {self.care_homes_found}")
        logger.info("=" * 80)
    
    def _generate_final_report(self):
        """Generate a comprehensive report of the ingestion process."""
        report = {
            'ingestion_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_locations_fetched': self.total_locations_fetched,
                'care_homes_found': self.care_homes_found,
                'care_home_percentage': (self.care_homes_found / self.total_locations_fetched * 100) if self.total_locations_fetched > 0 else 0,
                'processed_location_ids_count': len(self.processed_location_ids)
            },
            'storage_locations': {
                'raw_data_bucket': f"gs://{self.raw_bucket.name}/raw/locations/",
                'processed_care_homes': f"gs://{self.processed_bucket.name}/processed/care_homes/",
                'bigquery_dataset': f"{self.project_id}.cqc_dataset"
            },
            'next_steps': [
                'Run feature engineering pipeline',
                'Train ML models on care home data',
                'Deploy prediction API'
            ]
        }
        
        # Save report
        report_blob = self.processed_bucket.blob(f"reports/ingestion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        report_blob.upload_from_string(
            json.dumps(report, indent=2),
            content_type='application/json'
        )
        
        logger.info("Final report generated and saved")

def main():
    """Main function for Cloud Run job."""
    logger.info("Starting Complete CQC Data Fetcher Cloud Run Job")
    
    # Get configuration from environment
    batch_size = int(os.environ.get('BATCH_SIZE', '500'))
    resume = os.environ.get('RESUME', 'true').lower() == 'true'
    
    try:
        fetcher = CompleteCQCFetcher()
        
        # Test connection first
        test_result = fetcher._make_request("locations", {"page": 1, "perPage": 1})
        if not test_result:
            logger.error("Failed to connect to CQC API")
            return
        
        logger.info("API connection successful, starting dataset ingestion...")
        
        # Fetch complete dataset
        fetcher.fetch_complete_dataset(batch_size=batch_size, resume=resume)
        
        logger.info("Job completed successfully")
        
    except Exception as e:
        logger.error(f"Job failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()