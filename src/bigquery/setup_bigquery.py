#!/usr/bin/env python3
"""
BigQuery setup script for CQC data warehouse.
Creates datasets, tables, and views for the complete CQC ML pipeline.
Designed to run on Google Cloud Build.
"""

import os
import logging
from google.cloud import bigquery
from google.cloud.exceptions import Conflict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BigQuerySetup:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.client = bigquery.Client(project=self.project_id)
        self.dataset_id = 'cqc_dataset'
        self.location = 'europe-west2'
        
    def create_dataset(self):
        """Create the main CQC dataset."""
        dataset_id_full = f"{self.project_id}.{self.dataset_id}"
        dataset = bigquery.Dataset(dataset_id_full)
        dataset.location = self.location
        dataset.description = "CQC data warehouse for ML pipeline"
        
        try:
            dataset = self.client.create_dataset(dataset, timeout=30)
            logger.info(f"Created dataset {dataset_id_full}")
        except Conflict:
            logger.info(f"Dataset {dataset_id_full} already exists")
            
    def create_locations_complete_table(self):
        """Create table for all CQC locations."""
        table_id = f"{self.project_id}.{self.dataset_id}.locations_complete"
        
        schema = [
            bigquery.SchemaField("locationId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("locationName", "STRING"),
            bigquery.SchemaField("providerId", "STRING"),
            bigquery.SchemaField("providerName", "STRING"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("registrationStatus", "STRING"),
            bigquery.SchemaField("registrationDate", "DATE"),
            bigquery.SchemaField("deregistrationDate", "DATE"),
            bigquery.SchemaField("dormancy", "STRING"),
            bigquery.SchemaField("numberOfBeds", "INTEGER"),
            
            # Address fields
            bigquery.SchemaField("postalCode", "STRING"),
            bigquery.SchemaField("region", "STRING"),
            bigquery.SchemaField("localAuthority", "STRING"),
            bigquery.SchemaField("constituency", "STRING"),
            
            # Ratings
            bigquery.SchemaField("currentRatings", "RECORD", mode="NULLABLE", fields=[
                bigquery.SchemaField("overall", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                    bigquery.SchemaField("reportDate", "DATE"),
                    bigquery.SchemaField("reportLinkId", "STRING"),
                ]),
                bigquery.SchemaField("safe", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                ]),
                bigquery.SchemaField("effective", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                ]),
                bigquery.SchemaField("caring", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                ]),
                bigquery.SchemaField("responsive", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                ]),
                bigquery.SchemaField("wellLed", "RECORD", mode="NULLABLE", fields=[
                    bigquery.SchemaField("rating", "STRING"),
                ]),
            ]),
            
            # Service types
            bigquery.SchemaField("gacServiceTypes", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("description", "STRING"),
            ]),
            
            # Specialisms
            bigquery.SchemaField("specialisms", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("name", "STRING"),
            ]),
            
            # Regulated activities
            bigquery.SchemaField("regulatedActivities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("name", "STRING"),
                bigquery.SchemaField("code", "STRING"),
            ]),
            
            # Inspection categories
            bigquery.SchemaField("inspectionCategories", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("code", "STRING"),
                bigquery.SchemaField("primary", "STRING"),
                bigquery.SchemaField("name", "STRING"),
            ]),
            
            # Last inspection
            bigquery.SchemaField("lastInspection", "RECORD", mode="NULLABLE", fields=[
                bigquery.SchemaField("date", "DATE"),
            ]),
            
            # Metadata
            bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("processing_date", "DATE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="processing_date"
        )
        table.clustering_fields = ["region", "registrationStatus", "type"]
        
        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Conflict:
            logger.info(f"Table {table_id} already exists")
            
    def create_care_homes_table(self):
        """Create specialized table for care homes only."""
        table_id = f"{self.project_id}.{self.dataset_id}.care_homes"
        
        # Similar schema to locations_complete but with care home specific fields
        schema = [
            bigquery.SchemaField("locationId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("locationName", "STRING"),
            bigquery.SchemaField("providerId", "STRING"),
            bigquery.SchemaField("providerName", "STRING"),
            bigquery.SchemaField("type", "STRING"),
            bigquery.SchemaField("care_home_type", "STRING"),  # With/without nursing
            bigquery.SchemaField("registrationStatus", "STRING"),
            bigquery.SchemaField("registrationDate", "DATE"),
            bigquery.SchemaField("numberOfBeds", "INTEGER"),
            
            # Geographic
            bigquery.SchemaField("postalCode", "STRING"),
            bigquery.SchemaField("region", "STRING"),
            bigquery.SchemaField("localAuthority", "STRING"),
            
            # Ratings
            bigquery.SchemaField("overall_rating", "STRING"),
            bigquery.SchemaField("safe_rating", "STRING"),
            bigquery.SchemaField("effective_rating", "STRING"),
            bigquery.SchemaField("caring_rating", "STRING"),
            bigquery.SchemaField("responsive_rating", "STRING"),
            bigquery.SchemaField("wellLed_rating", "STRING"),
            bigquery.SchemaField("last_report_date", "DATE"),
            
            # Care home specialisms
            bigquery.SchemaField("cares_for_adults_over_65", "BOOLEAN"),
            bigquery.SchemaField("cares_for_adults_under_65", "BOOLEAN"),
            bigquery.SchemaField("dementia_care", "BOOLEAN"),
            bigquery.SchemaField("mental_health_care", "BOOLEAN"),
            bigquery.SchemaField("physical_disabilities_care", "BOOLEAN"),
            bigquery.SchemaField("learning_disabilities_care", "BOOLEAN"),
            
            # Quality indicators
            bigquery.SchemaField("has_nursing", "BOOLEAN"),
            bigquery.SchemaField("days_since_last_inspection", "INTEGER"),
            bigquery.SchemaField("rating_trend", "STRING"),  # improving/stable/declining
            
            # Metadata
            bigquery.SchemaField("ingestion_timestamp", "TIMESTAMP"),
            bigquery.SchemaField("processing_date", "DATE"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="processing_date"
        )
        table.clustering_fields = ["region", "care_home_type", "overall_rating"]
        
        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Conflict:
            logger.info(f"Table {table_id} already exists")
            
    def create_ml_features_table(self):
        """Create table for engineered ML features."""
        table_id = f"{self.project_id}.{self.dataset_id}.ml_features"
        
        schema = [
            bigquery.SchemaField("locationId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("feature_version", "STRING"),
            bigquery.SchemaField("feature_date", "DATE"),
            
            # Static features
            bigquery.SchemaField("number_of_beds", "FLOAT64"),
            bigquery.SchemaField("years_registered", "FLOAT64"),
            bigquery.SchemaField("is_dormant", "FLOAT64"),
            bigquery.SchemaField("has_nursing", "FLOAT64"),
            
            # Geographic features
            bigquery.SchemaField("region_encoded", "INTEGER"),
            bigquery.SchemaField("local_authority_encoded", "INTEGER"),
            bigquery.SchemaField("regional_avg_rating", "FLOAT64"),
            
            # Provider features
            bigquery.SchemaField("provider_location_count", "INTEGER"),
            bigquery.SchemaField("provider_avg_rating", "FLOAT64"),
            bigquery.SchemaField("provider_rating_std", "FLOAT64"),
            
            # Temporal features
            bigquery.SchemaField("days_since_last_inspection", "FLOAT64"),
            bigquery.SchemaField("inspection_frequency", "FLOAT64"),
            bigquery.SchemaField("seasonal_factor", "FLOAT64"),
            
            # Historical rating features
            bigquery.SchemaField("previous_overall_rating_encoded", "INTEGER"),
            bigquery.SchemaField("rating_change_indicator", "FLOAT64"),
            bigquery.SchemaField("consecutive_good_ratings", "INTEGER"),
            
            # Specialism features
            bigquery.SchemaField("num_specialisms", "INTEGER"),
            bigquery.SchemaField("high_risk_specialisms", "INTEGER"),
            
            # Target variable
            bigquery.SchemaField("target_overall_rating", "STRING"),
            bigquery.SchemaField("target_rating_encoded", "INTEGER"),
            
            # Metadata
            bigquery.SchemaField("created_timestamp", "TIMESTAMP"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.clustering_fields = ["feature_version", "feature_date"]
        
        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Conflict:
            logger.info(f"Table {table_id} already exists")
            
    def create_predictions_table(self):
        """Create table for model predictions."""
        table_id = f"{self.project_id}.{self.dataset_id}.predictions"
        
        schema = [
            bigquery.SchemaField("prediction_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("locationId", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("model_version", "STRING"),
            bigquery.SchemaField("prediction_date", "DATE"),
            
            # Predictions
            bigquery.SchemaField("predicted_overall_rating", "STRING"),
            bigquery.SchemaField("prediction_confidence", "FLOAT64"),
            bigquery.SchemaField("prediction_probabilities", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("rating", "STRING"),
                bigquery.SchemaField("probability", "FLOAT64"),
            ]),
            
            # Feature importance
            bigquery.SchemaField("top_features", "RECORD", mode="REPEATED", fields=[
                bigquery.SchemaField("feature_name", "STRING"),
                bigquery.SchemaField("importance_score", "FLOAT64"),
            ]),
            
            # Risk indicators
            bigquery.SchemaField("risk_score", "FLOAT64"),
            bigquery.SchemaField("risk_factors", "STRING", mode="REPEATED"),
            
            # Recommendations
            bigquery.SchemaField("improvement_areas", "STRING", mode="REPEATED"),
            bigquery.SchemaField("action_priority", "STRING"),
            
            # Actual outcome (for model evaluation)
            bigquery.SchemaField("actual_rating", "STRING"),
            bigquery.SchemaField("prediction_accuracy", "BOOLEAN"),
            
            # Metadata
            bigquery.SchemaField("created_timestamp", "TIMESTAMP"),
        ]
        
        table = bigquery.Table(table_id, schema=schema)
        table.time_partitioning = bigquery.TimePartitioning(
            type_=bigquery.TimePartitioningType.DAY,
            field="prediction_date"
        )
        table.clustering_fields = ["model_version", "predicted_overall_rating"]
        
        try:
            table = self.client.create_table(table)
            logger.info(f"Created table {table_id}")
        except Conflict:
            logger.info(f"Table {table_id} already exists")
            
    def create_views(self):
        """Create useful views for analysis."""
        
        # View for care homes with ratings
        view_id = f"{self.project_id}.{self.dataset_id}.v_care_homes_with_ratings"
        view_query = f"""
        CREATE OR REPLACE VIEW `{view_id}` AS
        SELECT 
            locationId,
            locationName,
            providerName,
            region,
            numberOfBeds,
            overall_rating,
            safe_rating,
            effective_rating,
            caring_rating,
            responsive_rating,
            wellLed_rating,
            last_report_date,
            CASE 
                WHEN overall_rating = 'Outstanding' THEN 4
                WHEN overall_rating = 'Good' THEN 3
                WHEN overall_rating = 'Requires improvement' THEN 2
                WHEN overall_rating = 'Inadequate' THEN 1
                ELSE 0
            END as rating_numeric,
            DATE_DIFF(CURRENT_DATE(), last_report_date, DAY) as days_since_inspection
        FROM `{self.project_id}.{self.dataset_id}.care_homes`
        WHERE registrationStatus = 'Registered'
        """
        
        try:
            self.client.query(view_query).result()
            logger.info(f"Created view {view_id}")
        except Exception as e:
            logger.warning(f"View {view_id} creation failed or already exists: {e}")
            
        # View for regional statistics
        view_id = f"{self.project_id}.{self.dataset_id}.v_regional_statistics"
        view_query = f"""
        CREATE OR REPLACE VIEW `{view_id}` AS
        SELECT 
            region,
            COUNT(*) as total_homes,
            AVG(CASE 
                WHEN overall_rating = 'Outstanding' THEN 4
                WHEN overall_rating = 'Good' THEN 3
                WHEN overall_rating = 'Requires improvement' THEN 2
                WHEN overall_rating = 'Inadequate' THEN 1
                ELSE NULL
            END) as avg_rating_score,
            COUNTIF(overall_rating = 'Outstanding') as outstanding_count,
            COUNTIF(overall_rating = 'Good') as good_count,
            COUNTIF(overall_rating = 'Requires improvement') as requires_improvement_count,
            COUNTIF(overall_rating = 'Inadequate') as inadequate_count,
            AVG(numberOfBeds) as avg_beds,
            AVG(DATE_DIFF(CURRENT_DATE(), last_report_date, DAY)) as avg_days_since_inspection
        FROM `{self.project_id}.{self.dataset_id}.care_homes`
        WHERE registrationStatus = 'Registered'
        GROUP BY region
        """
        
        try:
            self.client.query(view_query).result()
            logger.info(f"Created view {view_id}")
        except Exception as e:
            logger.warning(f"View {view_id} creation failed or already exists: {e}")
            
    def run_setup(self):
        """Run complete BigQuery setup."""
        logger.info("Starting BigQuery setup...")
        
        # Create dataset
        self.create_dataset()
        
        # Create tables
        self.create_locations_complete_table()
        self.create_care_homes_table()
        self.create_ml_features_table()
        self.create_predictions_table()
        
        # Create views
        self.create_views()
        
        logger.info("BigQuery setup completed successfully!")
        
        # Print summary
        dataset_ref = self.client.dataset(self.dataset_id)
        tables = list(self.client.list_tables(dataset_ref))
        
        logger.info(f"\nDataset: {self.project_id}.{self.dataset_id}")
        logger.info(f"Location: {self.location}")
        logger.info(f"Tables created: {len(tables)}")
        for table in tables:
            logger.info(f"  - {table.table_id}")

def main():
    """Main function to run on Cloud Build."""
    setup = BigQuerySetup()
    setup.run_setup()

if __name__ == "__main__":
    main()