import json
import time
import logging
from datetime import datetime
import functions_framework
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery
import requests

logging.basicConfig(level=logging.INFO)

@functions_framework.http
def fetch_cqc_data(request):
    """HTTP Cloud Function to fetch CQC data."""
    
    # Parse request parameters
    request_json = request.get_json(silent=True)
    limit = request_json.get('limit', 100) if request_json else 100
    mode = request_json.get('mode', 'sample') if request_json else 'sample'
    
    try:
        # Initialize fetcher
        fetcher = CQCFetcher()
        
        if mode == 'sample':
            # Fetch a small sample for testing
            result = fetcher.fetch_sample_data(limit=min(limit, 10))
        else:
            # Fetch full data
            result = fetcher.fetch_locations_with_ratings(limit=limit)
            
        return {'status': 'success', 'result': result}
        
    except Exception as e:
        logging.error(f"Error in fetch_cqc_data: {e}")
        return {'status': 'error', 'message': str(e)}, 500


class CQCFetcher:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        self.storage_client = storage.Client()
        self.bigquery_client = bigquery.Client()
        
    def _get_api_key(self):
        """Get API key from Secret Manager."""
        client = secretmanager.SecretManagerServiceClient()
        name = client.secret_version_path(
            "machine-learning-exp-467008",
            "cqc-subscription-key",
            "latest"
        )
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
        
    def fetch_sample_data(self, limit=10):
        """Fetch a small sample of locations for testing."""
        logging.info(f"Fetching {limit} sample locations...")
        
        # Get locations list
        params = {"page": 1, "perPage": limit}
        
        response = requests.get(
            f"{self.base_url}/locations",
            params=params,
            headers=self.headers,
            timeout=30
        )
        
        if response.status_code != 200:
            logging.error(f"API Response: {response.text}")
            logging.error(f"Headers sent: {self.headers}")
            logging.error(f"URL: {self.base_url}/locations")
            logging.error(f"Params: {params}")
            raise Exception(f"API error: {response.status_code} - {response.text[:200]}")
            
        locations = response.json().get('locations', [])
        detailed_locations = []
        
        # Fetch details for each location
        for loc in locations[:min(limit, 5)]:
            try:
                detail_response = requests.get(
                    f"{self.base_url}/locations/{loc['locationId']}",
                    headers=self.headers
                )
                
                if detail_response.status_code == 200:
                    detailed_locations.append(detail_response.json())
                    
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                logging.error(f"Error fetching location {loc['locationId']}: {e}")
                
        # Save to BigQuery
        self._save_to_bigquery(detailed_locations)
        
        # Save to GCS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_path = self._save_to_gcs(
            detailed_locations,
            f"sample_data/locations_{timestamp}.json"
        )
        
        return {
            'locations_fetched': len(detailed_locations),
            'gcs_path': gcs_path,
            'timestamp': timestamp
        }
        
    def fetch_locations_with_ratings(self, limit=1000):
        """Fetch locations that have ratings for training."""
        logging.info(f"Fetching up to {limit} locations with ratings...")
        
        locations_with_ratings = []
        page = 1
        
        while len(locations_with_ratings) < limit:
            response = requests.get(
                f"{self.base_url}/locations",
                params={"page": page, "perPage": 100},
                headers=self.headers
            )
            
            if response.status_code != 200:
                break
                
            data = response.json()
            locations = data.get('locations', [])
            
            if not locations:
                break
                
            # Filter locations with ratings
            for loc in locations:
                if len(locations_with_ratings) >= limit:
                    break
                    
                # Check if location has ratings
                try:
                    detail_response = requests.get(
                        f"{self.base_url}/locations/{loc['locationId']}",
                        headers=self.headers
                    )
                    
                    if detail_response.status_code == 200:
                        details = detail_response.json()
                        if 'currentRatings' in details:
                            locations_with_ratings.append(details)
                            
                    time.sleep(0.1)  # Rate limiting
                    
                except Exception as e:
                    logging.error(f"Error: {e}")
                    
            page += 1
            
        # Save data
        self._save_to_bigquery(locations_with_ratings)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gcs_path = self._save_to_gcs(
            locations_with_ratings,
            f"full_data/locations_{timestamp}.json"
        )
        
        return {
            'locations_fetched': len(locations_with_ratings),
            'gcs_path': gcs_path,
            'timestamp': timestamp
        }
        
    def _save_to_gcs(self, data, filename):
        """Save data to Google Cloud Storage."""
        bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-raw-data")
        blob = bucket.blob(filename)
        blob.upload_from_string(json.dumps(data))
        return f"gs://machine-learning-exp-467008-cqc-raw-data/{filename}"
        
    def _save_to_bigquery(self, locations):
        """Save locations to BigQuery."""
        if not locations:
            return
            
        # Prepare rows for BigQuery
        rows = []
        for loc in locations:
            row = {
                'locationId': loc.get('locationId'),
                'name': loc.get('name'),
                'numberOfBeds': loc.get('numberOfBeds'),
                'registrationDate': loc.get('registrationDate'),
                'lastInspectionDate': loc.get('lastInspection', {}).get('date'),
                'postalCode': loc.get('postalCode'),
                'region': loc.get('region'),
                'localAuthority': loc.get('localAuthority'),
                'providerId': loc.get('providerId'),
                'locationType': loc.get('type'),
                'overallRating': loc.get('currentRatings', {}).get('overall', {}).get('rating'),
                'safeRating': loc.get('currentRatings', {}).get('safe', {}).get('rating'),
                'effectiveRating': loc.get('currentRatings', {}).get('effective', {}).get('rating'),
                'caringRating': loc.get('currentRatings', {}).get('caring', {}).get('rating'),
                'responsiveRating': loc.get('currentRatings', {}).get('responsive', {}).get('rating'),
                'wellLedRating': loc.get('currentRatings', {}).get('wellLed', {}).get('rating'),
                'regulatedActivitiesCount': len(loc.get('regulatedActivities', [])),
                'specialismsCount': len(loc.get('specialisms', [])),
                'serviceTypesCount': len(loc.get('gacServiceTypes', [])),
                'rawData': json.dumps(loc)
            }
            rows.append(row)
            
        # Insert into BigQuery
        table = self.bigquery_client.dataset('cqc_data').table('locations_staging')
        errors = self.bigquery_client.insert_rows_json(table, rows)
        
        if errors:
            logging.error(f"BigQuery insert errors: {errors}")
        else:
            logging.info(f"Inserted {len(rows)} rows to BigQuery")
