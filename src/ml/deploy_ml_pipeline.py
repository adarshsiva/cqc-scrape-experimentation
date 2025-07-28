#!/usr/bin/env python3
"""
Deploy and run the CQC ML training pipeline on Vertex AI.

This script will:
1. Compile the ML pipeline
2. Submit it to Vertex AI Pipelines
3. Train XGBoost and LightGBM models
4. Evaluate models and deploy the best one
"""

import argparse
import sys
import os
from datetime import datetime
from google.cloud import aiplatform
from google.cloud import storage
import json

# Add the pipeline directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipeline'))


def create_bigquery_dataset(project_id: str, dataset_id: str = "cqc_data"):
    """Create BigQuery dataset if it doesn't exist."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    dataset_id_full = f"{project_id}.{dataset_id}"
    
    try:
        dataset = bigquery.Dataset(dataset_id_full)
        dataset.location = "US"
        dataset = client.create_dataset(dataset, timeout=30)
        print(f"Created dataset {dataset_id}")
    except Exception as e:
        if "Already Exists" in str(e):
            print(f"Dataset {dataset_id} already exists")
        else:
            raise e


def ensure_bucket_exists(bucket_name: str, project_id: str):
    """Ensure the ML pipeline bucket exists."""
    storage_client = storage.Client(project=project_id)
    
    try:
        bucket = storage_client.bucket(bucket_name)
        if not bucket.exists():
            bucket = storage_client.create_bucket(bucket_name, location="US-CENTRAL1")
            print(f"Created bucket: {bucket_name}")
        else:
            print(f"Bucket already exists: {bucket_name}")
    except Exception as e:
        print(f"Error checking/creating bucket: {e}")
        raise


def compile_and_upload_pipeline(project_id: str):
    """Compile the pipeline and upload to GCS."""
    from pipeline.pipeline import compile_pipeline
    
    # Compile pipeline
    local_pipeline_path = "cqc_ml_pipeline.json"
    compile_pipeline(local_pipeline_path)
    
    # Upload to GCS
    bucket_name = f"{project_id}-cqc-ml-pipeline"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gcs_pipeline_path = f"pipelines/cqc_ml_pipeline_{timestamp}.json"
    
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(gcs_pipeline_path)
    blob.upload_from_filename(local_pipeline_path)
    
    gcs_uri = f"gs://{bucket_name}/{gcs_pipeline_path}"
    print(f"Pipeline uploaded to: {gcs_uri}")
    
    # Clean up local file
    os.remove(local_pipeline_path)
    
    return gcs_uri


def create_custom_query(project_id: str):
    """Create a custom query to join locations with their ratings."""
    query = f"""
    SELECT 
        l.location_id,
        l.provider_id,
        l.name as location_name,
        l.type,
        l.postal_code,
        l.region,
        l.local_authority,
        l.registration_date,
        l.last_inspection_date,
        l.overall_rating as rating,
        l.regulated_activities,
        l.service_types,
        l.specialisms,
        l.days_since_last_inspection,
        l.days_since_registration,
        l.num_regulated_activities,
        l.num_service_types,
        l.has_specialisms,
        p.name as provider_name,
        p.type as provider_type,
        p.ownership_type
    FROM `{project_id}.cqc_data.locations` l
    LEFT JOIN `{project_id}.cqc_data.providers` p
        ON l.provider_id = p.provider_id
    WHERE l.overall_rating IS NOT NULL
        AND l.overall_rating IN ('Outstanding', 'Good', 'Requires improvement', 'Inadequate')
    """
    return query


def run_ml_pipeline(project_id: str, location: str = "us-central1"):
    """Run the ML training pipeline on Vertex AI."""
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Ensure bucket exists
    bucket_name = f"{project_id}-cqc-ml-pipeline"
    ensure_bucket_exists(bucket_name, project_id)
    
    # Ensure BigQuery dataset exists
    create_bigquery_dataset(project_id)
    
    # Compile and upload pipeline
    pipeline_uri = compile_and_upload_pipeline(project_id)
    
    # Set pipeline parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_params = {
        "project_id": project_id,
        "location": location,
        "dataset_id": "cqc_data",
        "table_id": "locations",  # We'll use a custom query instead
        "query": create_custom_query(project_id),
        "train_split": 0.7,
        "validation_split": 0.15,
        "test_split": 0.15,
        "target_column": "rating",
        "xgboost_hyperparameters": {
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 200,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "lightgbm_hyperparameters": {
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5
        },
        "enable_automl": False,  # Disable AutoML for faster training
        "endpoint_display_name": "cqc-rating-predictor",
        "machine_type": "n1-standard-4",
        "min_replicas": 1,
        "max_replicas": 2,
        "experiment_name": f"cqc-ml-experiment-{timestamp}"
    }
    
    # Create pipeline job
    job = aiplatform.PipelineJob(
        display_name=f"cqc-ml-training-{timestamp}",
        template_path=pipeline_uri,
        pipeline_root=f"gs://{bucket_name}/pipeline_runs/{timestamp}",
        parameter_values=pipeline_params,
        enable_caching=False
    )
    
    print("\n" + "="*50)
    print("Submitting ML Training Pipeline")
    print("="*50)
    print(f"\nPipeline: {job.display_name}")
    print(f"Project: {project_id}")
    print(f"Location: {location}")
    
    # Submit the pipeline
    job.submit()
    
    print("\n" + "="*50)
    print("ML Pipeline Submitted Successfully!")
    print("="*50)
    print(f"\nJob Name: {job.display_name}")
    print(f"\nView in console: {job._dashboard_uri()}")
    print("\nThe pipeline will:")
    print("1. Load processed data from BigQuery")
    print("2. Split data into train/validation/test sets")
    print("3. Perform feature engineering")
    print("4. Train XGBoost and LightGBM models")
    print("5. Evaluate models on test set")
    print("6. Deploy the best performing model to endpoint")
    print(f"\nEndpoint name: cqc-rating-predictor")
    
    return job


def check_etl_completion(project_id: str):
    """Check if ETL pipeline has completed by verifying BigQuery tables."""
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    
    # Check if tables exist and have data
    tables_to_check = ['locations', 'providers']
    dataset_id = f"{project_id}.cqc_data"
    
    for table_name in tables_to_check:
        table_id = f"{dataset_id}.{table_name}"
        try:
            table = client.get_table(table_id)
            rows_query = f"SELECT COUNT(*) as count FROM `{table_id}`"
            result = client.query(rows_query).result()
            count = list(result)[0].count
            print(f"Table {table_name}: {count} rows")
            
            if count == 0:
                print(f"WARNING: Table {table_name} exists but has no data")
                return False
                
        except Exception as e:
            print(f"Table {table_name} not found or error: {e}")
            return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Deploy CQC ML Training Pipeline')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--location', default='us-central1', help='GCP Region')
    parser.add_argument('--skip-etl-check', action='store_true', 
                       help='Skip checking if ETL has completed')
    args = parser.parse_args()
    
    try:
        # Check if ETL has completed
        if not args.skip_etl_check:
            print("Checking if ETL pipeline has completed...")
            if not check_etl_completion(args.project_id):
                print("\nERROR: ETL pipeline has not completed or tables are empty.")
                print("Please ensure the ETL pipeline has run successfully before training ML models.")
                print("You can skip this check with --skip-etl-check if you're sure the data is ready.")
                sys.exit(1)
            print("ETL data verified successfully!\n")
        
        # Run ML pipeline
        job = run_ml_pipeline(args.project_id, args.location)
        
        print("\nTo check pipeline status, run:")
        print(f"gcloud ai platform pipelines list --project={args.project_id} --region={args.location}")
        
    except Exception as e:
        print(f"\nError running ML pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()