#!/usr/bin/env python3
"""
Cloud Run Job to process existing CQC data from GCS and load into BigQuery.
Enhanced version that fetches detailed location data for locations in GCS.
"""

import json
import logging
import os
import time
from typing import Dict, List
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import secretmanager
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CQCDataProcessor:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(f"{self.project_id}-cqc-raw-data")
        
        # API configuration
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.api_key = self._get_api_key()
        self.headers = {
            "Ocp-Apim-Subscription-Key": self.api_key,
            "User-Agent": "CQC-ML-Predictor/1.0"
        }
        
        # Session with retries
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
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
            
    def fetch_location_details(self, location_id: str) -> Dict:
        """Fetch detailed information for a specific location"""
        try:
            url = f"{self.base_url}/locations/{location_id}"
            response = self.session.get(url, timeout=30)
            
            if response.status_code == 200:
                location_data = response.json()
                
                # Extract ML features
                features = {
                    'locationId': location_data.get('locationId'),
                    'name': location_data.get('name'),
                    'type': location_data.get('type'),
                    'numberOfBeds': location_data.get('numberOfBeds', 0),
                    'registrationDate': location_data.get('registrationDate'),
                    'postalCode': location_data.get('postalCode'),
                    'region': location_data.get('region', {}).get('name'),
                    'localAuthority': location_data.get('localAuthority'),
                    'lastInspectionDate': location_data.get('lastInspection', {}).get('date'),
                    'providerId': location_data.get('providerId'),
                    'regulatedActivitiesCount': len(location_data.get('regulatedActivities', [])),
                    'specialismsCount': len(location_data.get('specialisms', [])),
                    'serviceTypesCount': len(location_data.get('gacServiceTypes', []))
                }
                
                # Extract ratings
                if 'currentRatings' in location_data:
                    ratings = location_data['currentRatings']
                    for domain in ['overall', 'safe', 'effective', 'caring', 'responsive', 'wellLed']:
                        if domain in ratings and ratings[domain]:
                            features[f'{domain}Rating'] = ratings[domain].get('rating')
                            features[f'{domain}RatingDate'] = ratings[domain].get('reportDate')
                
                # Store raw data too
                features['rawData'] = json.dumps(location_data)
                features['hasRatings'] = bool(features.get('overallRating'))
                
                return features
            else:
                logger.warning(f"Failed to fetch location {location_id}: Status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching location {location_id}: {e}")
            return None
    
    def process_existing_locations(self):
        """Process locations from existing GCS files"""
        logger.info("Processing existing location files from GCS...")
        
        # List all location files
        blobs = list(self.storage_client.list_blobs(
            self.bucket,
            prefix="real_data/locations/"
        ))
        
        all_location_ids = set()
        
        # Extract unique location IDs from all files
        for blob in blobs:
            if blob.name.endswith('.json'):
                try:
                    content = blob.download_as_text()
                    data = json.loads(content)
                    
                    if 'locations' in data:
                        # API response format
                        for loc in data['locations']:
                            all_location_ids.add(loc['locationId'])
                    elif isinstance(data, list):
                        # Array format
                        for loc in data:
                            if 'locationId' in loc:
                                all_location_ids.add(loc['locationId'])
                                
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {e}")
        
        logger.info(f"Found {len(all_location_ids)} unique location IDs")
        
        # Fetch detailed data for each location
        detailed_locations = []
        location_ids = list(all_location_ids)[:1000]  # Limit to 1000 for initial run
        
        # Use ThreadPoolExecutor for parallel fetching
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_location = {
                executor.submit(self.fetch_location_details, loc_id): loc_id
                for loc_id in location_ids
            }
            
            for i, future in enumerate(as_completed(future_to_location)):
                location_id = future_to_location[future]
                try:
                    details = future.result()
                    if details and details.get('hasRatings'):
                        detailed_locations.append(details)
                        logger.info(f"Fetched {i+1}/{len(location_ids)}: {location_id} - Has ratings")
                    elif details:
                        logger.info(f"Fetched {i+1}/{len(location_ids)}: {location_id} - No ratings")
                except Exception as e:
                    logger.error(f"Error processing {location_id}: {e}")
                
                # Rate limiting
                if i % 10 == 0:
                    time.sleep(1)
        
        logger.info(f"Successfully fetched {len(detailed_locations)} locations with ratings")
        
        # Save to BigQuery
        if detailed_locations:
            self.save_to_bigquery(detailed_locations)
            
        # Save to GCS for backup
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        blob_name = f"processed/detailed_locations_{timestamp}.json"
        blob = self.bucket.blob(blob_name)
        blob.upload_from_string(
            json.dumps(detailed_locations, indent=2),
            content_type='application/json'
        )
        logger.info(f"Saved to GCS: gs://{self.bucket.name}/{blob_name}")
        
    def save_to_bigquery(self, locations: List[Dict]):
        """Save detailed location data to BigQuery"""
        logger.info(f"Saving {len(locations)} locations to BigQuery...")
        
        # Create table if not exists
        dataset_id = "cqc_data"
        table_id = "locations_detailed"
        table_ref = f"{self.project_id}.{dataset_id}.{table_id}"
        
        # Define schema
        schema = [
            bigquery.SchemaField("locationId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("name", "STRING"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("numberOfBeds", "INTEGER"),
            bigquery.SchemaField("registrationDate", "DATE"),
            bigquery.SchemaField("postalCode", "STRING"),
            bigquery.SchemaField("region", "STRING"),
            bigquery.SchemaField("localAuthority", "STRING"),
            bigquery.SchemaField("lastInspectionDate", "DATE"),
            bigquery.SchemaField("providerId", "STRING"),
            bigquery.SchemaField("regulatedActivitiesCount", "INTEGER"),
            bigquery.SchemaField("specialismsCount", "INTEGER"),
            bigquery.SchemaField("serviceTypesCount", "INTEGER"),
            bigquery.SchemaField("overallRating", "STRING"),
            bigquery.SchemaField("overallRatingDate", "DATE"),
            bigquery.SchemaField("safeRating", "STRING"),
            bigquery.SchemaField("effectiveRating", "STRING"),
            bigquery.SchemaField("caringRating", "STRING"),
            bigquery.SchemaField("responsiveRating", "STRING"),
            bigquery.SchemaField("wellLedRating", "STRING"),
            bigquery.SchemaField("rawData", "STRING"),
            bigquery.SchemaField("fetchTimestamp", "TIMESTAMP"),
        ]
        
        # Create table if needed
        try:
            table = self.bigquery_client.get_table(table_ref)
        except:
            table = bigquery.Table(table_ref, schema=schema)
            table = self.bigquery_client.create_table(table)
            logger.info(f"Created table {table_ref}")
        
        # Prepare rows for insertion
        rows = []
        for loc in locations:
            row = loc.copy()
            row['fetchTimestamp'] = time.time()
            # Remove fields not in schema
            row.pop('hasRatings', None)
            rows.append(row)
        
        # Insert rows
        errors = self.bigquery_client.insert_rows_json(table, rows)
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Successfully inserted {len(rows)} rows to BigQuery")
            
        # Create ML features view
        self.create_ml_features_view()
        
    def create_ml_features_view(self):
        """Create ML features view from detailed locations"""
        logger.info("Creating ML features view...")
        
        query = """
        CREATE OR REPLACE VIEW `cqc_data.ml_features_proactive` AS
        SELECT
          locationId,
          name,
          IFNULL(numberOfBeds, 0) as number_of_beds,
          DATE_DIFF(CURRENT_DATE(), registrationDate, DAY) as days_since_registration,
          DATE_DIFF(CURRENT_DATE(), lastInspectionDate, DAY) as days_since_inspection,
          regulatedActivitiesCount,
          specialismsCount,
          serviceTypesCount,
          region,
          localAuthority,
          
          -- Risk indicators
          CASE 
            WHEN DATE_DIFF(CURRENT_DATE(), lastInspectionDate, DAY) > 730 THEN 1 
            ELSE 0 
          END as overdue_inspection,
          
          -- Current ratings (numeric encoding)
          CASE overallRating
            WHEN 'Outstanding' THEN 1
            WHEN 'Good' THEN 2
            WHEN 'Requires improvement' THEN 3
            WHEN 'Inadequate' THEN 4
            ELSE 0
          END as overall_rating_score,
          
          -- Domain-specific risks
          CASE WHEN safeRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as safe_at_risk,
          CASE WHEN effectiveRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as effective_at_risk,
          CASE WHEN caringRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as caring_at_risk,
          CASE WHEN responsiveRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as responsive_at_risk,
          CASE WHEN wellLedRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as well_led_at_risk,
          
          -- Composite risk score
          (
            CASE WHEN safeRating IN ('Requires improvement', 'Inadequate') THEN 0.25 ELSE 0 END +
            CASE WHEN effectiveRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END +
            CASE WHEN caringRating IN ('Requires improvement', 'Inadequate') THEN 0.15 ELSE 0 END +
            CASE WHEN responsiveRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END +
            CASE WHEN wellLedRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END
          ) * 100 as domain_risk_score,
          
          -- Target variable
          CASE 
            WHEN overallRating IN ('Requires improvement', 'Inadequate') THEN 1
            ELSE 0
          END as at_risk_label,
          
          overallRating,
          safeRating,
          effectiveRating,
          caringRating,
          responsiveRating,
          wellLedRating
          
        FROM `cqc_data.locations_detailed`
        WHERE overallRating IS NOT NULL
        """
        
        job = self.bigquery_client.query(query)
        job.result()
        
        logger.info("ML features view created successfully")
        
        # Check row count
        count_query = "SELECT COUNT(*) as count FROM `cqc_data.ml_features_proactive`"
        result = list(self.bigquery_client.query(count_query).result())
        count = result[0].count if result else 0
        logger.info(f"ML features view contains {count} rows")

def main():
    """Main execution function for Cloud Run Job"""
    logger.info("Starting CQC data processing job...")
    
    processor = CQCDataProcessor()
    
    # Check if we're running in Cloud Run (has K_SERVICE env var)
    if os.environ.get('K_SERVICE'):
        logger.info("Running in Cloud Run environment")
    else:
        logger.info("Running in local environment")
    
    try:
        processor.process_existing_locations()
        logger.info("Job completed successfully!")
    except Exception as e:
        logger.error(f"Job failed: {e}")
        raise

if __name__ == "__main__":
    main()