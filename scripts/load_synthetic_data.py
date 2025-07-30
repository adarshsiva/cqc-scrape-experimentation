#!/usr/bin/env python3
"""
Load synthetic CQC data into BigQuery for model training
"""

import json
import logging
from datetime import datetime
from google.cloud import bigquery
from google.cloud import storage
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SyntheticDataLoader:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        
    def load_synthetic_data(self):
        """Load synthetic data from GCS to BigQuery"""
        logger.info("Loading synthetic CQC data...")
        
        # Download the synthetic data
        bucket = self.storage_client.bucket(f"{self.project_id}-cqc-raw-data")
        blob = bucket.blob("raw/locations/20250728_191714_locations_sample.json")
        data = json.loads(blob.download_as_text())
        
        logger.info(f"Found {len(data)} synthetic locations")
        
        # Process and prepare for BigQuery
        rows = []
        for location in data:
            # Extract features
            row = {
                'locationId': location['locationId'],
                'name': location['name'],
                'type': location['type'],
                'numberOfBeds': location.get('numberOfBeds', 0),
                'registrationDate': location['registrationDate'][:10],  # Date only
                'postalCode': location['postalCode'],
                'region': location['region'],
                'localAuthority': location['localAuthority'],
                'lastInspectionDate': location['lastInspectionDate'][:10] if location.get('lastInspectionDate') else None,
                'providerId': location['providerId'],
                'regulatedActivitiesCount': len(location.get('regulatedActivities', [])),
                'specialismsCount': len(location.get('specialisms', [])),
                'serviceTypesCount': len(location.get('gacServiceTypes', [])),
                'overallRating': location['currentRatings']['overall']['rating'],
                'safeRating': location['currentRatings']['safe']['rating'],
                'effectiveRating': location['currentRatings']['effective']['rating'],
                'caringRating': location['currentRatings']['caring']['rating'],
                'responsiveRating': location['currentRatings']['responsive']['rating'],
                'wellLedRating': location['currentRatings']['wellLed']['rating'],
                'rawData': json.dumps(location),
                'fetchTimestamp': datetime.now().isoformat(),
                'dataSource': 'synthetic'
            }
            rows.append(row)
        
        # Load into BigQuery
        table_id = f"{self.project_id}.cqc_data.locations_detailed"
        
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
            bigquery.SchemaField("safeRating", "STRING"),
            bigquery.SchemaField("effectiveRating", "STRING"),
            bigquery.SchemaField("caringRating", "STRING"),
            bigquery.SchemaField("responsiveRating", "STRING"),
            bigquery.SchemaField("wellLedRating", "STRING"),
            bigquery.SchemaField("rawData", "STRING"),
            bigquery.SchemaField("fetchTimestamp", "TIMESTAMP"),
            bigquery.SchemaField("dataSource", "STRING"),
        ]
        
        # Create or replace table
        table = bigquery.Table(table_id, schema=schema)
        table = self.bigquery_client.create_table(table, exists_ok=True)
        
        # Insert data
        errors = self.bigquery_client.insert_rows_json(table, rows)
        if errors:
            logger.error(f"BigQuery insert errors: {errors}")
        else:
            logger.info(f"Successfully loaded {len(rows)} rows to BigQuery")
        
        # Create ML features view
        self.create_ml_features_view()
        
    def create_ml_features_view(self):
        """Create ML features view"""
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
    loader = SyntheticDataLoader()
    loader.load_synthetic_data()

if __name__ == "__main__":
    main()