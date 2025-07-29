import json
import logging
import functions_framework
from google.cloud import storage
from google.cloud import bigquery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@functions_framework.http
def process_existing_data(request):
    """Process existing CQC data from GCS."""
    
    project_id = "machine-learning-exp-467008"
    storage_client = storage.Client(project=project_id)
    bigquery_client = bigquery.Client(project=project_id)
    bucket = storage_client.bucket(f"{project_id}-cqc-raw-data")
    
    # List all real data files
    blobs = storage_client.list_blobs(
        bucket,
        prefix="real_data/locations/"
    )
    
    total_processed = 0
    total_with_ratings = 0
    
    for blob in blobs:
        if blob.name.endswith('.json'):
            try:
                logger.info(f"Processing {blob.name}...")
                
                # Download file content
                content = blob.download_as_text()
                locations = json.loads(content)
                
                logger.info(f"Found {len(locations)} locations in file")
                
                # Process locations
                rows = []
                locations_with_ratings = 0
                
                for loc in locations:
                    if isinstance(loc, dict):
                        # Extract rating
                        overall_rating = None
                        if 'overall' in loc:
                            overall_rating = loc.get('overall', {}).get('rating')
                        elif 'currentRatings' in loc:
                            overall_rating = loc.get('currentRatings', {}).get('overall', {}).get('rating')
                            
                        if overall_rating:
                            locations_with_ratings += 1
                            
                        # Create row
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
                        
                # Insert into BigQuery
                if rows:
                    table_id = f"{project_id}.cqc_data.locations_staging"
                    table = bigquery_client.get_table(table_id)
                    
                    errors = bigquery_client.insert_rows_json(table, rows)
                    if errors:
                        logger.error(f"BigQuery insert errors: {errors}")
                    else:
                        logger.info(f"Inserted {len(rows)} rows")
                        total_processed += len(rows)
                        total_with_ratings += locations_with_ratings
                        
            except Exception as e:
                logger.error(f"Error processing {blob.name}: {e}")
                
    # Create features view
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
    
    job = bigquery_client.query(query)
    job.result()
    
    return {
        'status': 'success',
        'total_processed': total_processed,
        'locations_with_ratings': total_with_ratings,
        'message': f'Processed {total_processed} locations, {total_with_ratings} with ratings'
    }