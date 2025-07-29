#!/bin/bash

#######################################
# Deploy All Services Script
# Deploys all application services to GCP
#######################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check environment variables
    local required_vars=("GCP_PROJECT" "GCP_REGION")
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Environment variable $var is not set"
            exit 1
        fi
    done
    
    # Check if gcloud is authenticated
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &>/dev/null; then
        log_error "Not authenticated with gcloud. Please run 'gcloud auth login'"
        exit 1
    fi
    
    # Set defaults
    export GCS_RAW_BUCKET="${GCS_RAW_BUCKET:-${GCP_PROJECT}-cqc-raw}"
    export GCS_PROCESSED_BUCKET="${GCS_PROCESSED_BUCKET:-${GCP_PROJECT}-cqc-processed}"
    export GCS_MODELS_BUCKET="${GCS_MODELS_BUCKET:-${GCP_PROJECT}-cqc-models}"
    export BQ_DATASET="${BQ_DATASET:-cqc_data}"
    export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-cqc-service-account}"
    
    log_info "Prerequisites checked successfully"
}

# Deploy Cloud Functions
deploy_cloud_functions() {
    log_info "Deploying Cloud Functions..."
    
    # Deploy CQC API ingestion function
    log_info "Deploying CQC API ingestion function..."
    
    cd ../src/ingestion
    
    # Create requirements.txt if not exists
    if [[ ! -f requirements.txt ]]; then
        cat > requirements.txt <<EOF
google-cloud-storage==2.10.0
google-cloud-secret-manager==2.16.4
requests==2.31.0
EOF
    fi
    
    gcloud functions deploy cqc-api-ingestion \
        --runtime=python39 \
        --trigger-http \
        --entry-point=ingest_cqc_data \
        --source=. \
        --memory=512MB \
        --timeout=540s \
        --max-instances=10 \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --service-account="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCS_BUCKET=$GCS_RAW_BUCKET" \
        --allow-unauthenticated
    
    # Deploy prediction service function
    log_info "Deploying prediction service function..."
    
    cd ../ml/serving
    
    # Create requirements.txt if not exists
    if [[ ! -f requirements.txt ]]; then
        cat > requirements.txt <<EOF
google-cloud-storage==2.10.0
google-cloud-aiplatform==1.35.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
EOF
    fi
    
    gcloud functions deploy cqc-prediction-service \
        --runtime=python39 \
        --trigger-http \
        --entry-point=predict \
        --source=. \
        --memory=1024MB \
        --timeout=60s \
        --max-instances=50 \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --service-account="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        --set-env-vars="GCP_PROJECT=$GCP_PROJECT,GCS_MODELS_BUCKET=$GCS_MODELS_BUCKET" \
        --allow-unauthenticated
    
    cd ../../deployment
    
    log_info "Cloud Functions deployed successfully"
}

# Deploy Dataflow pipelines
deploy_dataflow_pipelines() {
    log_info "Deploying Dataflow pipeline templates..."
    
    cd ../src/pipeline
    
    # Build and upload Dataflow template
    log_info "Building Dataflow template..."
    
    # Create setup.py if not exists
    if [[ ! -f setup.py ]]; then
        cat > setup.py <<EOF
import setuptools

setuptools.setup(
    name='cqc-etl-pipeline',
    version='1.0.0',
    install_requires=[
        'apache-beam[gcp]==2.50.0',
        'pandas==2.0.3',
        'numpy==1.24.3',
    ],
    packages=setuptools.find_packages(),
)
EOF
    fi
    
    # Build template
    python -m etl_pipeline \
        --runner=DataflowRunner \
        --project="$GCP_PROJECT" \
        --region="$GCP_REGION" \
        --temp_location="gs://$GCS_PROCESSED_BUCKET/temp" \
        --template_location="gs://$GCS_PROCESSED_BUCKET/templates/cqc-etl-pipeline" \
        --staging_location="gs://$GCS_PROCESSED_BUCKET/staging" \
        --requirements_file=requirements.txt \
        --setup_file=./setup.py \
        --save_main_session
    
    cd ../../deployment
    
    log_info "Dataflow pipelines deployed successfully"
}

# Deploy ML models to Vertex AI
deploy_ml_models() {
    log_info "Deploying ML models to Vertex AI..."
    
    cd ../src/ml
    
    # Check if models exist in GCS
    if ! gsutil ls "gs://$GCS_MODELS_BUCKET/models/" &>/dev/null; then
        log_warn "No models found in gs://$GCS_MODELS_BUCKET/models/"
        log_warn "Please train models first using the training pipeline"
        return
    fi
    
    # Deploy models using Vertex AI
    log_info "Creating Vertex AI model endpoints..."
    
    # Create endpoint
    ENDPOINT_ID=$(gcloud ai endpoints create \
        --region="$GCP_REGION" \
        --display-name="cqc-rating-predictor" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    log_info "Created endpoint: $ENDPOINT_ID"
    
    # Upload and deploy model (example for XGBoost)
    MODEL_ID=$(gcloud ai models upload \
        --region="$GCP_REGION" \
        --display-name="cqc-xgboost-model" \
        --artifact-uri="gs://$GCS_MODELS_BUCKET/models/xgboost/" \
        --container-image-uri="gcr.io/cloud-aiplatform/prediction/xgboost-cpu.1-6:latest" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    log_info "Uploaded model: $MODEL_ID"
    
    # Deploy model to endpoint
    gcloud ai endpoints deploy-model "$ENDPOINT_ID" \
        --region="$GCP_REGION" \
        --model="$MODEL_ID" \
        --display-name="cqc-xgboost-v1" \
        --machine-type="n1-standard-4" \
        --min-replica-count=1 \
        --max-replica-count=3 \
        --traffic-split="0=100" \
        --project="$GCP_PROJECT"
    
    cd ../../deployment
    
    log_info "ML models deployed successfully"
}

# Deploy Cloud Composer DAGs
deploy_composer_dags() {
    log_info "Deploying Cloud Composer DAGs..."
    
    # Get Composer bucket
    COMPOSER_BUCKET=$(gcloud composer environments describe cqc-composer-env \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(config.dagGcsPrefix)" | sed 's|/dags||')
    
    if [[ -z "$COMPOSER_BUCKET" ]]; then
        log_warn "Could not find Composer environment bucket. Skipping DAG deployment."
        return
    fi
    
    # Copy DAGs to Composer bucket
    log_info "Copying DAGs to Composer bucket..."
    
    cd ../src/orchestration
    
    # Create sample DAG if not exists
    if [[ ! -f cqc_pipeline_dag.py ]]; then
        cat > cqc_pipeline_dag.py <<'EOF'
from datetime import datetime, timedelta
from airflow import DAG
from airflow.providers.google.cloud.operators.functions import CloudFunctionInvokeFunctionOperator
from airflow.providers.google.cloud.operators.dataflow import DataflowTemplatedJobStartOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryCheckOperator

default_args = {
    'owner': 'cqc-pipeline',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cqc_daily_pipeline',
    default_args=default_args,
    description='Daily CQC data pipeline',
    schedule_interval='0 2 * * *',  # Daily at 2 AM
    catchup=False,
)

# Task 1: Ingest data from CQC API
ingest_data = CloudFunctionInvokeFunctionOperator(
    task_id='ingest_cqc_data',
    location='{{ var.value.gcp_region }}',
    project_id='{{ var.value.gcp_project }}',
    function_id='cqc-api-ingestion',
    dag=dag,
)

# Task 2: Run ETL pipeline
run_etl = DataflowTemplatedJobStartOperator(
    task_id='run_etl_pipeline',
    template='gs://{{ var.value.gcs_processed_bucket }}/templates/cqc-etl-pipeline',
    project_id='{{ var.value.gcp_project }}',
    location='{{ var.value.gcp_region }}',
    dag=dag,
)

# Task 3: Check data quality
check_data_quality = BigQueryCheckOperator(
    task_id='check_data_quality',
    sql="""
    SELECT COUNT(*) as count
    FROM `{{ var.value.gcp_project }}.{{ var.value.bq_dataset }}.providers`
    WHERE DATE(last_updated) = CURRENT_DATE()
    """,
    gcp_conn_id='google_cloud_default',
    dag=dag,
)

# Set dependencies
ingest_data >> run_etl >> check_data_quality
EOF
    fi
    
    gsutil cp *.py "$COMPOSER_BUCKET/dags/"
    
    cd ../../deployment
    
    log_info "Composer DAGs deployed successfully"
}

# Deploy monitoring and alerting
setup_monitoring() {
    log_info "Setting up monitoring and alerting..."
    
    # Create log-based metrics
    log_info "Creating log-based metrics..."
    
    gcloud logging metrics create cqc_api_errors \
        --description="CQC API ingestion errors" \
        --project="$GCP_PROJECT" \
        --log-filter='resource.type="cloud_function"
        resource.labels.function_name="cqc-api-ingestion"
        severity=ERROR' || log_warn "Metric may already exist"
    
    gcloud logging metrics create cqc_prediction_latency \
        --description="CQC prediction service latency" \
        --project="$GCP_PROJECT" \
        --log-filter='resource.type="cloud_function"
        resource.labels.function_name="cqc-prediction-service"
        httpRequest.latency>1s' || log_warn "Metric may already exist"
    
    # Create alerting policies
    log_info "Creating alerting policies..."
    
    # Note: Alerting policies require more complex configuration
    log_info "Please configure alerting policies through the Cloud Console"
    
    log_info "Monitoring setup completed"
}

# Main execution
main() {
    log_info "Starting deployment of all services..."
    
    check_prerequisites
    
    # Deploy services
    deploy_cloud_functions
    deploy_dataflow_pipelines
    deploy_ml_models
    deploy_composer_dags
    setup_monitoring
    
    log_info "All services deployed successfully!"
    log_info "Next steps:"
    log_info "1. Run create_scheduler_jobs.sh to set up automated schedules"
    log_info "2. Run test_end_to_end.sh to test the complete system"
}

# Run main function
main "$@"