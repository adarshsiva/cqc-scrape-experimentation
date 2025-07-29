#!/usr/bin/env python3
"""
Process existing CQC data from GCS and load into BigQuery.
"""

import json
import logging
from google.cloud import storage
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self):
        self.project_id = "machine-learning-exp-467008"
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.bucket = self.storage_client.bucket(f"{self.project_id}-cqc-raw-data")
        
    def process_locations_file(self, blob_name: str):
        """Process a locations file from GCS."""
        logger.info(f"Processing {blob_name}...")
        
        # Download file content
        blob = self.bucket.blob(blob_name)
        content = blob.download_as_text()
        locations = json.loads(content)
        
        logger.info(f"Found {len(locations)} locations in file")
        
        # Process and insert into BigQuery
        rows = []
        locations_with_ratings = 0
        
        for loc in locations:
            # Check if location has ratings
            has_rating = False
            if isinstance(loc, dict):
                # Handle different data structures
                if 'overall' in loc:
                    # Direct rating structure
                    overall_rating = loc.get('overall', {}).get('rating')
                    has_rating = overall_rating is not None
                elif 'currentRatings' in loc:
                    # Nested rating structure
                    overall_rating = loc.get('currentRatings', {}).get('overall', {}).get('rating')
                    has_rating = overall_rating is not None
                else:
                    overall_rating = None
                    
                if has_rating:
                    locations_with_ratings += 1
                    
                # Create row for BigQuery
                row = {
                    'locationId': loc.get('locationId'),
                    'name': loc.get('locationName') or loc.get('name'),
                    'numberOfBeds': loc.get('numberOfBeds'),
                    'postalCode': loc.get('postalCode'),
                    'region': loc.get('region'),
                    'overallRating': overall_rating,
                    'rawData': json.dumps(loc)
                }
                rows.append(row)
                
        logger.info(f"Found {locations_with_ratings} locations with ratings")
        
        # Insert into BigQuery
        if rows:
            table_id = f"{self.project_id}.cqc_data.locations_staging"
            table = self.bigquery_client.get_table(table_id)
            
            errors = self.bigquery_client.insert_rows_json(table, rows)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Inserted {len(rows)} rows to BigQuery")
                
        return len(rows), locations_with_ratings
        
    def process_all_data(self):
        """Process all available real data files."""
        logger.info("Starting to process existing CQC data...")
        
        # List all real data files
        blobs = self.storage_client.list_blobs(
            self.bucket,
            prefix="real_data/locations/"
        )
        
        total_processed = 0
        total_with_ratings = 0
        
        for blob in blobs:
            if blob.name.endswith('.json'):
                try:
                    processed, with_ratings = self.process_locations_file(blob.name)
                    total_processed += processed
                    total_with_ratings += with_ratings
                except Exception as e:
                    logger.error(f"Error processing {blob.name}: {e}")
                    
        logger.info(f"\nProcessing complete!")
        logger.info(f"Total locations processed: {total_processed}")
        logger.info(f"Locations with ratings: {total_with_ratings}")
        
        # Create the ML features view
        self.create_features_view()
        
    def create_features_view(self):
        """Create or update the ML features view."""
        logger.info("Creating ML features view...")
        
        query = """
        CREATE OR REPLACE VIEW `cqc_data.ml_features_simple` AS
        SELECT DISTINCT
          locationId,
          FIRST_VALUE(name IGNORE NULLS) OVER (PARTITION BY locationId ORDER BY load_timestamp DESC) as name,
          FIRST_VALUE(region IGNORE NULLS) OVER (PARTITION BY locationId ORDER BY load_timestamp DESC) as region,
          FIRST_VALUE(overallRating IGNORE NULLS) OVER (PARTITION BY locationId ORDER BY load_timestamp DESC) as overallRating,
          CASE 
            WHEN FIRST_VALUE(overallRating IGNORE NULLS) OVER (PARTITION BY locationId ORDER BY load_timestamp DESC) 
                 IN ('Requires improvement', 'Inadequate') THEN 1
            ELSE 0
          END as at_risk_label
        FROM `cqc_data.locations_staging`
        WHERE overallRating IS NOT NULL
        """
        
        job = self.bigquery_client.query(query)
        job.result()
        
        logger.info("ML features view created successfully")

if __name__ == "__main__":
    processor = DataProcessor()
    processor.process_all_data()