"""
Configuration for CQC Airflow DAGs

This module provides centralized configuration for all DAGs in the CQC pipeline.
Update the values below based on your GCP project setup.
"""

import os
from datetime import timedelta

# GCP Configuration
GCP_PROJECT_ID = os.environ.get('GCP_PROJECT_ID', 'your-project-id')
GCP_REGION = os.environ.get('GCP_REGION', 'europe-west2')
GCP_ZONE = os.environ.get('GCP_ZONE', 'europe-west2-a')

# BigQuery Configuration
BQ_DATASET_ID = os.environ.get('BQ_DATASET_ID', 'cqc_data')
BQ_MONITORING_DATASET_ID = os.environ.get('BQ_MONITORING_DATASET_ID', 'cqc_monitoring')
BQ_LOCATION = os.environ.get('BQ_LOCATION', 'EU')

# Cloud Storage Configuration
GCS_BUCKET_NAME = os.environ.get('GCS_BUCKET_NAME', f'{GCP_PROJECT_ID}-cqc-data')
GCS_STAGING_BUCKET = os.environ.get('GCS_STAGING_BUCKET', f'{GCP_PROJECT_ID}-cqc-staging')
GCS_MODEL_BUCKET = os.environ.get('GCS_MODEL_BUCKET', f'{GCP_PROJECT_ID}-cqc-models')

# Cloud Function Configuration
CF_INGESTION_FUNCTION = os.environ.get('CF_INGESTION_FUNCTION', 'cqc-data-ingestion')
CF_PREDICTION_FUNCTION = os.environ.get('CF_PREDICTION_FUNCTION', 'cqc-prediction-api')

# Vertex AI Configuration
VERTEX_AI_ENDPOINT = os.environ.get('VERTEX_AI_ENDPOINT', '')
VERTEX_AI_MODEL_NAME = os.environ.get('VERTEX_AI_MODEL_NAME', 'cqc-risk-model')
VERTEX_AI_TRAINING_IMAGE = os.environ.get(
    'VERTEX_AI_TRAINING_IMAGE', 
    f'gcr.io/{GCP_PROJECT_ID}/cqc-training:latest'
)

# Notification Configuration
ALERT_EMAIL = os.environ.get('ALERT_EMAIL', 'alerts@example.com')
REPORT_EMAIL_LIST = os.environ.get('REPORT_EMAIL_LIST', 'reports@example.com').split(',')

# DAG Default Arguments
DEFAULT_DAG_ARGS = {
    'owner': 'cqc-pipeline',
    'depends_on_past': False,
    'email': [ALERT_EMAIL],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
    'retry_exponential_backoff': True,
    'max_retry_delay': timedelta(minutes=30),
}

# Monitoring Thresholds
MONITORING_THRESHOLDS = {
    'data_freshness_hours': 26,  # Alert if data is older than 26 hours
    'min_daily_records': 100,    # Minimum expected records per day
    'max_error_rate': 0.05,      # 5% error rate threshold
    'max_storage_gb': 1000,      # Alert if storage exceeds 1TB
    'max_daily_cost': 100,       # Alert if daily cost exceeds $100
    'max_endpoint_latency': 2.0, # Maximum acceptable endpoint latency in seconds
    'min_model_accuracy': 0.85,  # Minimum acceptable model accuracy
}

# Data Quality Rules
DATA_QUALITY_RULES = {
    'max_null_rate': 0.05,           # Maximum 5% null values allowed
    'max_duplicate_rate': 0.01,      # Maximum 1% duplicates allowed
    'required_fields': {
        'providers': ['provider_id', 'name', 'provider_type'],
        'locations': ['location_id', 'name', 'provider_id', 'type']
    }
}

# Model Training Configuration
TRAINING_CONFIG = {
    'machine_type': 'n1-standard-8',
    'accelerator_type': 'NVIDIA_TESLA_T4',
    'accelerator_count': 1,
    'training_image': VERTEX_AI_TRAINING_IMAGE,
    'max_training_hours': 4,
    'evaluation_metrics': ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'],
    'hyperparameter_tuning': {
        'max_trials': 20,
        'parallel_trials': 5,
        'metric_id': 'auc_roc',
        'goal': 'MAXIMIZE'
    }
}

# Batch Prediction Configuration
BATCH_PREDICTION_CONFIG = {
    'machine_type': 'n1-standard-4',
    'max_replica_count': 10,
    'batch_size': 1000,
    'format': 'bigquery'
}

# CQC API Configuration
CQC_API_CONFIG = {
    'base_url': 'https://api.cqc.org.uk/public/v1',
    'providers_endpoint': '/providers',
    'locations_endpoint': '/locations',
    'page_size': 500,
    'rate_limit_delay': 0.5,  # Seconds between API calls
    'max_retries': 5,
    'timeout': 30  # Seconds
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    'rating_mappings': {
        'Outstanding': 4,
        'Good': 3,
        'Requires improvement': 2,
        'Inadequate': 1,
        'No rating': 0
    },
    'risk_thresholds': {
        'critical': 0.8,
        'high': 0.6,
        'medium': 0.4,
        'low': 0.2
    },
    'feature_columns': [
        'location_type',
        'days_since_inspection',
        'num_regulated_activities',
        'num_service_types',
        'num_specialisms',
        'provider_type',
        'ownership_type',
        'provider_location_count',
        'previous_rating',
        'rating_trend'
    ]
}

# Alert Templates
ALERT_TEMPLATES = {
    'high_risk_subject': 'CQC High Risk Alerts - {date}',
    'monitoring_subject': 'CQC Pipeline Monitoring Alert - {date}',
    'data_quality_subject': 'CQC Data Quality Issues - {date}',
    'model_performance_subject': 'CQC Model Performance Alert - {date}'
}

# Useful helper functions
def get_table_id(dataset: str, table: str) -> str:
    """Get fully qualified BigQuery table ID"""
    return f"{GCP_PROJECT_ID}.{dataset}.{table}"

def get_gcs_uri(bucket: str, path: str) -> str:
    """Get GCS URI for a given path"""
    return f"gs://{bucket}/{path}"

def get_current_date_partition() -> str:
    """Get current date partition string for GCS paths"""
    from datetime import datetime
    return datetime.now().strftime("%Y/%m/%d")

# Validation function
def validate_config():
    """Validate that all required configuration is set"""
    required_vars = [
        'GCP_PROJECT_ID',
        'GCP_REGION',
        'BQ_DATASET_ID',
        'GCS_BUCKET_NAME',
        'CF_INGESTION_FUNCTION',
        'ALERT_EMAIL'
    ]
    
    missing_vars = []
    for var in required_vars:
        if globals()[var] in ['', 'your-project-id', None]:
            missing_vars.append(var)
    
    if missing_vars:
        raise ValueError(f"Missing required configuration: {', '.join(missing_vars)}")
    
    return True