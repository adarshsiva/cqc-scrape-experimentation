#!/usr/bin/env python3
"""
Direct CQC API Data Fetcher - Fetches real data from CQC Syndication API
Focuses on care homes with current ratings for ML training.
"""

import os
import json
import time
import logging
import requests
from datetime import datetime
from typing import Dict, List, Optional
from google.cloud import bigquery
from google.cloud import secretmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DirectCQCDataFetcher:
    """Direct fetcher for real CQC data from Syndication API."""
    
    def __init__(self):
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.api_key = self._get_api_key()
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        
        # Create session with headers
        self.session = requests.Session()
        self.session.headers.update({
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "CQC-ML-Pipeline/1.0"
        })
        
        # BigQuery client
        self.bq_client = bigquery.Client(project=self.project_id)
        
        logger.info("Direct CQC Data Fetcher initialized")
        logger.info(f"API Key present: {bool(self.api_key)}")
    
    def _get_api_key(self) -> str:
        """Get API key from Secret Manager."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/machine-learning-exp-467008/secrets/cqc-subscription-key/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Could not get API key from Secret Manager: {e}")
            # Fallback to environment
            api_key = os.environ.get('CQC_API_KEY', '')
            if not api_key:
                raise ValueError("No CQC API key found in Secret Manager or environment")
            return api_key
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with error handling and rate limiting."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"✅ Successfully parsed JSON response")
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limited, waiting 60 seconds...")
                time.sleep(60)
                return self._make_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Request exception for {endpoint}: {e}")
            return None
    
    def fetch_care_home_locations(self, max_locations: int = 1000) -> List[Dict]:
        """Fetch care home locations from CQC API."""
        logger.info("Fetching care home locations from CQC Syndication API...")
        
        all_locations = []
        page = 1
        per_page = 500
        
        while len(all_locations) < max_locations:
            params = {
                "page": page,
                "perPage": per_page
            }
            
            data = self._make_request("locations", params)
            if not data or 'locations' not in data:
                break
            
            locations = data['locations']
            
            # Filter for care homes only
            care_homes = []
            for location in locations:
                location_type = location.get('type', '')
                if 'care home' in location_type.lower():
                    care_homes.append(location)
            
            all_locations.extend(care_homes)
            
            logger.info(f"Fetched {len(all_locations)}/{max_locations} care home locations")
            
            if len(locations) < per_page:
                break
            
            page += 1
            time.sleep(1)  # Rate limiting
        
        logger.info(f"✅ Found {len(all_locations)} total care home locations")
        return all_locations[:max_locations]
    
    def fetch_detailed_location(self, location_id: str) -> Optional[Dict]:
        """Fetch detailed information for a specific location."""
        return self._make_request(f"locations/{location_id}")
    
    def process_location_for_ml(self, location: Dict) -> Dict:
        """Process location data into ML-ready format."""
        
        # Extract basic info
        processed = {
            'locationId': location.get('locationId'),
            'locationName': location.get('name'),
            'providerId': location.get('providerId'),
            'type': location.get('type'),
            'registrationStatus': location.get('registrationStatus'),
            'numberOfBeds': location.get('numberOfBeds'),
            'region': location.get('region'),
            'localAuthority': location.get('localAuthority'),
            'constituency': location.get('constituency'), 
            'postalCode': location.get('postalCode'),
            'registrationDate': location.get('registrationDate'),
            'dormancy': location.get('dormancy', 'N')
        }
        
        # Extract current ratings
        if 'currentRatings' in location and location['currentRatings']:
            ratings = location['currentRatings']
            
            if 'overall' in ratings and ratings['overall']:
                processed['overall_rating'] = ratings['overall'].get('rating')
                processed['last_report_date'] = ratings['overall'].get('reportDate')
            
            # Domain ratings
            for domain in ['safe', 'effective', 'caring', 'responsive', 'wellLed']:
                if domain in ratings and ratings[domain]:
                    processed[f'{domain}_rating'] = ratings[domain].get('rating')
        
        # Extract regulated activities
        if 'regulatedActivities' in location and location['regulatedActivities']:
            activities = [activity.get('name') for activity in location['regulatedActivities'] if 'name' in activity]
            processed['regulatedActivities'] = activities
        else:
            processed['regulatedActivities'] = []
        
        # Extract GAC service types
        if 'gacServiceTypes' in location and location['gacServiceTypes']:
            services = [service.get('name') for service in location['gacServiceTypes'] if 'name' in service]
            processed['gacServiceTypes'] = services
        else:
            processed['gacServiceTypes'] = []
        
        # Extract specialisms
        if 'specialisms' in location and location['specialisms']:
            specs = [spec.get('name') for spec in location['specialisms'] if 'name' in spec]
            processed['specialisms'] = specs
        else:
            processed['specialisms'] = []
        
        # Add processing metadata
        processed['data_extraction_date'] = datetime.now().isoformat()
        processed['api_source'] = 'cqc_syndication_api'
        
        return processed
    
    def load_to_bigquery(self, locations: List[Dict]):
        """Load processed locations to BigQuery."""
        if not locations:
            logger.warning("No locations to load to BigQuery")
            return
        
        table_id = f"{self.project_id}.cqc_data.locations_real"
        
        # Configure load job
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,  # Append to existing data
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
                bigquery.SchemaUpdateOption.ALLOW_FIELD_RELAXATION
            ]
        )
        
        try:
            job = self.bq_client.load_table_from_json(locations, table_id, job_config=job_config)
            job.result()  # Wait for completion
            
            logger.info(f"✅ Successfully loaded {len(locations)} locations to {table_id}")
            
        except Exception as e:
            logger.error(f"Failed to load to BigQuery: {e}")
            raise
    
    def show_data_quality(self):
        """Show data quality summary."""
        table_id = f"{self.project_id}.cqc_data.locations_real"
        
        query = f"""
        SELECT 
            COUNT(*) as total_locations,
            COUNTIF(overall_rating IS NOT NULL) as locations_with_ratings,
            COUNTIF(overall_rating = 'Outstanding') as outstanding,
            COUNTIF(overall_rating = 'Good') as good,
            COUNTIF(overall_rating = 'Requires improvement') as requires_improvement,
            COUNTIF(overall_rating = 'Inadequate') as inadequate,
            COUNT(DISTINCT region) as unique_regions
        FROM `{table_id}`
        """
        
        try:
            results = list(self.bq_client.query(query))
            if results:
                row = results[0]
                logger.info("=== DATA QUALITY SUMMARY ===")
                logger.info(f"Total locations: {row.total_locations}")
                logger.info(f"With ratings: {row.locations_with_ratings}")
                logger.info(f"Outstanding: {row.outstanding}")
                logger.info(f"Good: {row.good}")
                logger.info(f"Requires improvement: {row.requires_improvement}")
                logger.info(f"Inadequate: {row.inadequate}")
                logger.info(f"Unique regions: {row.unique_regions}")
                logger.info("="*30)
        except Exception as e:
            logger.error(f"Could not show data quality: {e}")
    
    def run_extraction(self):
        """Run the full extraction pipeline."""
        max_locations = int(os.environ.get('MAX_LOCATIONS', 1000))
        batch_size = int(os.environ.get('BATCH_SIZE', 50))
        
        logger.info("="*70)
        logger.info("STARTING REAL CQC DATA EXTRACTION")
        logger.info(f"Target: {max_locations} care home locations")
        logger.info(f"Batch size: {batch_size}")
        logger.info("="*70)
        
        # Step 1: Fetch basic care home locations
        locations = self.fetch_care_home_locations(max_locations)
        
        if not locations:
            logger.error("No locations found!")
            return
        
        # Step 2: Fetch detailed data for each location in batches
        logger.info("Fetching detailed data for each location...")
        detailed_locations = []
        
        for i, location in enumerate(locations):
            location_id = location.get('locationId')
            if not location_id:
                continue
                
            # Fetch detailed data
            detailed = self.fetch_detailed_location(location_id)
            if detailed:
                # Process for ML
                processed = self.process_location_for_ml(detailed)
                detailed_locations.append(processed)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Processed {i+1}/{len(locations)}: {location_id}")
            
            # Rate limiting
            time.sleep(0.5)
            
            # Process in batches to avoid memory issues
            if len(detailed_locations) >= batch_size:
                logger.info(f"Loading batch of {len(detailed_locations)} to BigQuery...")
                self.load_to_bigquery(detailed_locations)
                detailed_locations = []
        
        # Load final batch
        if detailed_locations:
            logger.info(f"Loading final batch of {len(detailed_locations)} to BigQuery...")
            self.load_to_bigquery(detailed_locations)
        
        # Show final data quality
        self.show_data_quality()
        
        logger.info("="*70)
        logger.info("REAL CQC DATA EXTRACTION COMPLETE")
        logger.info("="*70)

def main():
    """Main execution function."""
    fetcher = DirectCQCDataFetcher()
    fetcher.run_extraction()

if __name__ == "__main__":
    main()