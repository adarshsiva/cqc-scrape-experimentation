"""
Enhanced CQC Data Fetcher
Fetches detailed location data with ratings history and compliance information
"""

import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import requests
from google.cloud import storage
from google.cloud import secretmanager
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DetailedCQCFetcher:
    """Fetches detailed CQC data including ratings and compliance information"""
    
    def __init__(self, project_id: str, bucket_name: str):
        self.project_id = project_id
        self.bucket_name = bucket_name
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.storage_client = storage.Client(project=project_id)
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Get API key from Secret Manager
        self.api_key = self._get_api_key()
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "CQC-ML-Predictor/1.0"
        }
        
        # Configure session with retry strategy
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Rate limiting
        self.rate_limit_delay = 0.5  # 500ms between requests
        
    def _get_api_key(self) -> str:
        """Retrieve API key from Secret Manager"""
        client = secretmanager.SecretManagerServiceClient()
        secret_name = f"projects/{self.project_id}/secrets/cqc-subscription-key/versions/latest"
        
        try:
            response = client.access_secret_version(request={"name": secret_name})
            return response.payload.data.decode("UTF-8").strip()
        except Exception as e:
            logger.error(f"Error retrieving API key: {e}")
            raise
        
    def fetch_locations_with_ratings(self, limit: int = 1000) -> List[Dict]:
        """Fetch locations that have ratings data"""
        locations = []
        page = 1
        
        while len(locations) < limit:
            try:
                url = f"{self.base_url}/locations"
                params = {
                    'page': page,
                    'perPage': 100
                }
                
                response = self.session.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                
                data = response.json()
                if not data.get('locations'):
                    break
                    
                # Filter locations that have been inspected and have ratings
                for loc in data['locations']:
                    # We'll get detailed info later to check for ratings
                    locations.append(loc)
                logger.info(f"Fetched page {page}, total locations: {len(locations)}")
                
                page += 1
                time.sleep(self.rate_limit_delay)
                
            except Exception as e:
                logger.error(f"Error fetching locations page {page}: {e}")
                break
                
        return locations[:limit]
    
    def fetch_location_details(self, location_id: str) -> Optional[Dict]:
        """Fetch detailed information for a specific location"""
        try:
            url = f"{self.base_url}/locations/{location_id}"
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            
            location_data = response.json()
            
            # Skip if no ratings
            if not location_data.get('currentRatings', {}).get('overall'):
                logger.info(f"Location {location_id} has no ratings, skipping")
                return None
            
            # Fetch additional details
            # Inspection areas (this endpoint exists)
            areas_url = f"{url}/inspection-areas"
            areas_response = self.session.get(areas_url, headers=self.headers)
            if areas_response.status_code == 200:
                location_data['inspectionAreas'] = areas_response.json()
            
            # Reports (if available)
            reports_url = f"{url}/reports"
            reports_response = self.session.get(reports_url, headers=self.headers)
            if reports_response.status_code == 200:
                location_data['reports'] = reports_response.json()
            
            # Extract key features for ML
            location_data['mlFeatures'] = self._extract_ml_features(location_data)
            
            time.sleep(self.rate_limit_delay)
            return location_data
            
        except Exception as e:
            logger.error(f"Error fetching details for location {location_id}: {e}")
            return None
    
    def save_to_gcs(self, data: Dict, blob_name: str) -> bool:
        """Save data to Google Cloud Storage"""
        try:
            blob = self.bucket.blob(blob_name)
            blob.upload_from_string(
                json.dumps(data, indent=2),
                content_type='application/json'
            )
            logger.info(f"Saved data to gs://{self.bucket_name}/{blob_name}")
            return True
        except Exception as e:
            logger.error(f"Error saving to GCS: {e}")
            return False
    
    def fetch_and_save_detailed_data(self, max_locations: int = 100, max_workers: int = 5):
        """Main method to fetch and save detailed CQC data"""
        logger.info(f"Starting detailed CQC data fetch for {max_locations} locations")
        
        # Fetch locations with ratings
        locations = self.fetch_locations_with_ratings(max_locations)
        logger.info(f"Found {len(locations)} locations with ratings")
        
        # Fetch detailed data for each location in parallel
        detailed_locations = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_location = {
                executor.submit(self.fetch_location_details, loc['locationId']): loc
                for loc in locations
            }
            
            # Process completed tasks
            for future in as_completed(future_to_location):
                location = future_to_location[future]
                try:
                    detailed_data = future.result()
                    if detailed_data:
                        detailed_locations.append(detailed_data)
                        logger.info(f"Fetched details for {location['locationName']}")
                except Exception as e:
                    logger.error(f"Error processing location {location['locationId']}: {e}")
        
        # Save to GCS
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save individual location files
        for location in detailed_locations:
            blob_name = f"detailed_locations/{location['locationId']}/{timestamp}.json"
            self.save_to_gcs(location, blob_name)
        
        # Save aggregated data
        aggregated_blob_name = f"detailed_locations/aggregated/{timestamp}_all_locations.json"
        aggregated_data = {
            'fetchTimestamp': timestamp,
            'totalLocations': len(detailed_locations),
            'locations': detailed_locations
        }
        self.save_to_gcs(aggregated_data, aggregated_blob_name)
        
        # Save summary statistics
        summary = self._generate_summary(detailed_locations)
        summary_blob_name = f"detailed_locations/summary/{timestamp}_summary.json"
        self.save_to_gcs(summary, summary_blob_name)
        
        logger.info(f"Completed fetching {len(detailed_locations)} detailed location records")
        return detailed_locations
    
    def _generate_summary(self, locations: List[Dict]) -> Dict:
        """Generate summary statistics from fetched data"""
        summary = {
            'totalLocations': len(locations),
            'fetchTimestamp': datetime.now().isoformat(),
            'ratingDistribution': {},
            'regionDistribution': {},
            'serviceTypeDistribution': {},
            'complianceStats': {
                'compliant': 0,
                'nonCompliant': 0,
                'unknown': 0
            }
        }
        
        for location in locations:
            # Overall rating distribution
            if 'currentRatings' in location and 'overall' in location['currentRatings']:
                rating = location['currentRatings']['overall']['rating']
                summary['ratingDistribution'][rating] = summary['ratingDistribution'].get(rating, 0) + 1
            
            # Region distribution
            if 'region' in location:
                region = location['region']['name']
                summary['regionDistribution'][region] = summary['regionDistribution'].get(region, 0) + 1
            
            # Service type distribution
            if 'type' in location:
                service_type = location['type']
                summary['serviceTypeDistribution'][service_type] = summary['serviceTypeDistribution'].get(service_type, 0) + 1
            
            # Compliance stats
            if 'lastInspection' in location:
                if location['lastInspection'].get('compliant'):
                    summary['complianceStats']['compliant'] += 1
                elif location['lastInspection'].get('compliant') is False:
                    summary['complianceStats']['nonCompliant'] += 1
                else:
                    summary['complianceStats']['unknown'] += 1
        
        return summary
    
    def _extract_ml_features(self, location: Dict) -> Dict:
        """Extract key features for ML model training"""
        features = {}
        
        # Basic info
        features['locationId'] = location.get('locationId')
        features['name'] = location.get('name')
        features['type'] = location.get('type')
        features['numberOfBeds'] = location.get('numberOfBeds', 0)
        
        # Registration info
        features['registrationDate'] = location.get('registrationDate')
        
        # Current ratings
        if 'currentRatings' in location:
            ratings = location['currentRatings']
            for domain in ['overall', 'safe', 'effective', 'caring', 'responsive', 'wellLed']:
                if domain in ratings and ratings[domain]:
                    features[f'{domain}Rating'] = ratings[domain].get('rating')
                    features[f'{domain}RatingDate'] = ratings[domain].get('reportDate')
        
        # Last inspection
        if 'lastInspection' in location:
            features['lastInspectionDate'] = location['lastInspection'].get('date')
        
        # Services and activities
        features['regulatedActivities'] = len(location.get('regulatedActivities', []))
        features['specialisms'] = len(location.get('specialisms', []))
        features['gacServiceTypes'] = len(location.get('gacServiceTypes', []))
        
        # Location
        features['region'] = location.get('region', {}).get('name')
        features['localAuthority'] = location.get('localAuthority')
        features['postalCode'] = location.get('postalCode')
        
        return features


def main():
    """Main execution function"""
    # Configuration
    project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
    bucket_name = os.environ.get('GCS_BUCKET', 'machine-learning-exp-467008-cqc-raw-data')
    max_locations = int(os.environ.get('MAX_LOCATIONS', '1000'))
    
    # Initialize fetcher
    fetcher = DetailedCQCFetcher(project_id, bucket_name)
    
    # Fetch and save data
    fetcher.fetch_and_save_detailed_data(max_locations=max_locations)


if __name__ == "__main__":
    main()