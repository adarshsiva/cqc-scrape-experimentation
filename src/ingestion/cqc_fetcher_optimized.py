#!/usr/bin/env python3
"""
Optimized CQC data fetcher using all available API endpoints.
Fetches locations, inspection areas, and reports for comprehensive ML training data.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedCQCFetcher:
    """Optimized fetcher using all CQC API endpoints for comprehensive data collection."""
    
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        
        # Initialize GCP clients
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        
        # Setup buckets
        self.raw_bucket = self.storage_client.bucket(f"{self.project_id}-cqc-raw-data")
        
        # Setup session
        self.session = self._create_session()
        
        logger.info("OptimizedCQCFetcher initialized successfully")
    
    def _get_api_key(self) -> str:
        """Get API key from Secret Manager or environment."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/cqc-subscription-key/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except:
            return os.environ.get('CQC_API_KEY', '')
    
    def _create_session(self) -> requests.Session:
        """Create session with proper headers."""
        session = requests.Session()
        session.headers.update({
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "CQC-ML-Pipeline/1.0"
        })
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with error handling."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                # Rate limited
                time.sleep(60)
                return self._make_request(endpoint, params)
        except Exception as e:
            logger.error(f"Request failed for {endpoint}: {e}")
        return None
    
    def fetch_all_locations(self, care_homes_only: bool = True) -> List[Dict]:
        """
        Fetch all locations using the Get Locations endpoint.
        This is more efficient than fetching details one by one.
        """
        logger.info("Fetching ALL locations from API...")
        all_locations = []
        page = 1
        per_page = 1000  # Maximum allowed
        total_count = None
        
        while True:
            data = self._make_request("locations", {"page": page, "perPage": per_page})
            
            if not data or 'locations' not in data:
                break
            
            # Get total count if available
            if 'totalCount' in data and total_count is None:
                total_count = data['totalCount']
                logger.info(f"Total locations available in API: {total_count}")
            
            locations = data['locations']
            
            # Filter for care homes if requested
            if care_homes_only:
                care_homes = [loc for loc in locations 
                             if self._is_care_home(loc)]
                all_locations.extend(care_homes)
                logger.info(f"Page {page}: {len(care_homes)} care homes out of {len(locations)} locations")
            else:
                all_locations.extend(locations)
                logger.info(f"Page {page}: {len(locations)} locations")
            
            logger.info(f"Progress: {len(all_locations)} collected so far...")
            
            # Check if we've fetched all pages
            if len(locations) < per_page:
                break
            
            page += 1
            time.sleep(0.3)  # Slightly faster rate limiting
        
        logger.info(f"✅ Total locations fetched: {len(all_locations)}")
        if care_homes_only:
            logger.info(f"📊 These are all care homes from the total {total_count} CQC locations")
        return all_locations
    
    def fetch_location_inspection_areas(self, location_id: str) -> Optional[Dict]:
        """
        Fetch inspection areas for a location.
        This provides detailed ratings for each CQC domain.
        """
        return self._make_request(f"locations/{location_id}/inspection-areas")
    
    def fetch_reports_for_location(self, location_id: str) -> Optional[List[Dict]]:
        """
        Fetch inspection reports for a location.
        Reports contain detailed findings and evidence.
        """
        data = self._make_request("reports", {"locationId": location_id})
        if data and 'reports' in data:
            return data['reports']
        return None
    
    def fetch_provider_details(self, provider_id: str) -> Optional[Dict]:
        """Fetch provider details for context."""
        return self._make_request(f"providers/{provider_id}")
    
    def fetch_changes_within_timeframe(self, days: int = 365) -> Optional[Dict]:
        """
        Fetch locations with rating changes in the last N days.
        Useful for tracking trends and model validation.
        """
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        return self._make_request("changes", {
            "startDate": start_date,
            "endDate": end_date
        })
    
    def _is_care_home(self, location: Dict) -> bool:
        """Check if a location is a care home."""
        care_home_types = [
            'Care home service with nursing',
            'Care home service without nursing'
        ]
        
        # Check GAC service types
        if 'gacServiceTypes' in location:
            for service in location['gacServiceTypes']:
                if isinstance(service, dict) and service.get('name') in care_home_types:
                    return True
        
        # Check type field
        if location.get('type') in care_home_types:
            return True
        
        return False
    
    def process_location_comprehensive(self, location: Dict) -> Dict:
        """
        Process a location with all additional data:
        - Inspection areas (detailed ratings)
        - Recent reports
        - Provider information
        """
        location_id = location['locationId']
        logger.info(f"Processing comprehensive data for {location_id}")
        
        # Add inspection areas
        inspection_areas = self.fetch_location_inspection_areas(location_id)
        if inspection_areas:
            location['inspectionAreas'] = inspection_areas
        
        # Add recent reports (limit to last 3)
        reports = self.fetch_reports_for_location(location_id)
        if reports:
            location['recentReports'] = reports[:3]
        
        # Add provider details if not already present
        if 'providerId' in location and 'providerDetails' not in location:
            provider_details = self.fetch_provider_details(location['providerId'])
            if provider_details:
                location['providerDetails'] = provider_details
        
        return location
    
    def save_to_bigquery(self, data: List[Dict], table_name: str):
        """Save data directly to BigQuery."""
        if not data:
            return
        
        dataset_id = "cqc_dataset"
        table_id = f"{self.project_id}.{dataset_id}.{table_name}"
        
        # Prepare data for BigQuery
        processed_data = []
        for item in data:
            record = self._flatten_for_bigquery(item)
            record['ingestion_timestamp'] = datetime.now().isoformat()
            record['processing_date'] = datetime.now().date().isoformat()
            processed_data.append(record)
        
        # Load to BigQuery
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=False,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
        )
        
        try:
            job = self.bigquery_client.load_table_from_json(
                processed_data, table_id, job_config=job_config
            )
            job.result()
            logger.info(f"Loaded {len(processed_data)} records to {table_id}")
        except Exception as e:
            logger.error(f"Failed to load to BigQuery: {e}")
            # Save to GCS as backup
            self._save_to_gcs_backup(data, table_name)
    
    def _flatten_for_bigquery(self, data: Dict) -> Dict:
        """Flatten nested structures for BigQuery."""
        flattened = {}
        
        # Basic fields
        flattened['locationId'] = data.get('locationId')
        flattened['locationName'] = data.get('locationName')
        flattened['providerId'] = data.get('providerId')
        flattened['providerName'] = data.get('providerName')
        flattened['type'] = data.get('type')
        flattened['registrationStatus'] = data.get('registrationStatus')
        flattened['registrationDate'] = data.get('registrationDate')
        flattened['numberOfBeds'] = data.get('numberOfBeds')
        
        # Address
        flattened['postalCode'] = data.get('postalCode')
        flattened['region'] = data.get('region')
        flattened['localAuthority'] = data.get('localAuthority')
        
        # Current ratings
        if 'currentRatings' in data and data['currentRatings']:
            ratings = data['currentRatings']
            if 'overall' in ratings:
                flattened['overall_rating'] = ratings['overall'].get('rating')
                flattened['last_report_date'] = ratings['overall'].get('reportDate')
            
            # Domain ratings
            for domain in ['safe', 'effective', 'caring', 'responsive', 'wellLed']:
                if domain in ratings:
                    flattened[f'{domain}_rating'] = ratings[domain].get('rating')
        
        # Care home specific
        flattened['care_home_type'] = data.get('type')
        flattened['has_nursing'] = 'nursing' in str(data.get('type', '')).lower()
        
        # Inspection areas summary
        if 'inspectionAreas' in data:
            areas = data['inspectionAreas']
            if isinstance(areas, dict) and 'inspectionAreas' in areas:
                flattened['inspection_areas_count'] = len(areas['inspectionAreas'])
        
        # Reports summary
        if 'recentReports' in data:
            flattened['recent_reports_count'] = len(data['recentReports'])
            if data['recentReports']:
                flattened['latest_report_date'] = data['recentReports'][0].get('reportDate')
        
        return flattened
    
    def _save_to_gcs_backup(self, data: List[Dict], prefix: str):
        """Save data to GCS as backup."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"backup/{prefix}/{timestamp}.json"
        
        blob = self.raw_bucket.blob(filename)
        blob.upload_from_string(
            json.dumps(data, indent=2),
            content_type='application/json'
        )
        logger.info(f"Saved backup to gs://{self.raw_bucket.name}/{filename}")
    
    def run_optimized_pipeline(self, max_locations: int = None):
        """
        Run the optimized data collection pipeline.
        
        Steps:
        1. Fetch all care home locations (bulk)
        2. Enrich with inspection areas and reports
        3. Load to BigQuery
        4. Track changes for model validation
        """
        logger.info("="*60)
        logger.info("STARTING OPTIMIZED CQC DATA PIPELINE")
        logger.info("="*60)
        
        # Step 1: Fetch all care home locations
        logger.info("Step 1: Fetching all care home locations...")
        locations = self.fetch_all_locations(care_homes_only=True)
        
        if max_locations:
            locations = locations[:max_locations]
        
        logger.info(f"Found {len(locations)} care homes to process")
        
        # Step 2: Enrich with additional data (in batches)
        logger.info("Step 2: Enriching with inspection data...")
        batch_size = 100
        enriched_locations = []
        
        for i in range(0, len(locations), batch_size):
            batch = locations[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1} to {min(i+batch_size, len(locations))})")
            
            # Process each location in the batch
            for location in batch:
                try:
                    enriched = self.process_location_comprehensive(location)
                    enriched_locations.append(enriched)
                    time.sleep(0.2)  # Rate limiting
                except Exception as e:
                    logger.error(f"Failed to process {location.get('locationId')}: {e}")
                    enriched_locations.append(location)  # Add basic data anyway
            
            # Save batch to BigQuery
            if enriched_locations:
                self.save_to_bigquery(enriched_locations, 'locations_complete')
                
                # Filter and save care homes
                care_homes = [loc for loc in enriched_locations if self._is_care_home(loc)]
                if care_homes:
                    self.save_to_bigquery(care_homes, 'care_homes')
                
                enriched_locations = []  # Clear for next batch
        
        # Step 3: Fetch recent changes for trend analysis
        logger.info("Step 3: Fetching recent rating changes...")
        changes = self.fetch_changes_within_timeframe(days=90)
        if changes:
            logger.info(f"Found {len(changes.get('changes', []))} recent changes")
            # Store changes for model validation
            self._save_to_gcs_backup(changes.get('changes', []), 'changes')
        
        logger.info("="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Processed {len(locations)} care homes")
        logger.info("="*60)

def main():
    """Main function for Cloud Run job."""
    # Set to None to fetch ALL available locations
    max_locations = os.environ.get('MAX_LOCATIONS')
    if max_locations and max_locations != 'ALL':
        max_locations = int(max_locations)
    else:
        max_locations = None  # Fetch everything
    
    logger.info(f"Starting fetcher with max_locations: {max_locations if max_locations else 'ALL AVAILABLE'}")
    
    fetcher = OptimizedCQCFetcher()
    fetcher.run_optimized_pipeline(max_locations=max_locations)

if __name__ == "__main__":
    main()