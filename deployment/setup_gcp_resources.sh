#!/bin/bash

#######################################
# GCP Resources Setup Script
# Creates all required GCP resources for CQC Rating Predictor
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

# Check if required environment variables are set
check_env_vars() {
    local required_vars=("GCP_PROJECT" "GCP_REGION")
    
    for var in "${required_vars[@]}"; do
        if [[ -z "${!var:-}" ]]; then
            log_error "Environment variable $var is not set"
            exit 1
        fi
    done
    
    log_info "Environment variables validated"
}

# Set default values
set_defaults() {
    export GCP_ZONE="${GCP_ZONE:-${GCP_REGION}-a}"
    export GCS_RAW_BUCKET="${GCS_RAW_BUCKET:-${GCP_PROJECT}-cqc-raw}"
    export GCS_PROCESSED_BUCKET="${GCS_PROCESSED_BUCKET:-${GCP_PROJECT}-cqc-processed}"
    export GCS_MODELS_BUCKET="${GCS_MODELS_BUCKET:-${GCP_PROJECT}-cqc-models}"
    export BQ_DATASET="${BQ_DATASET:-cqc_data}"
    export COMPOSER_ENV="${COMPOSER_ENV:-cqc-composer-env}"
    export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-cqc-service-account}"
    
    log_info "Default values set"
}

# Enable required APIs
enable_apis() {
    log_info "Enabling required GCP APIs..."
    
    local apis=(
        "compute.googleapis.com"
        "storage.googleapis.com"
        "bigquery.googleapis.com"
        "dataflow.googleapis.com"
        "cloudfunctions.googleapis.com"
        "cloudscheduler.googleapis.com"
        "appengine.googleapis.com"
        "secretmanager.googleapis.com"
        "aiplatform.googleapis.com"
        "composer.googleapis.com"
        "cloudresourcemanager.googleapis.com"
        "iam.googleapis.com"
        "cloudbuild.googleapis.com"
        "run.googleapis.com"
    )
    
    for api in "${apis[@]}"; do
        log_info "Enabling $api..."
        gcloud services enable "$api" --project="$GCP_PROJECT" || log_warn "Failed to enable $api (may already be enabled)"
    done
    
    log_info "APIs enabled successfully"
}

# Create service account
create_service_account() {
    log_info "Creating service account..."
    
    # Create service account
    gcloud iam service-accounts create "$SERVICE_ACCOUNT" \
        --display-name="CQC System Service Account" \
        --project="$GCP_PROJECT" 2>/dev/null || log_warn "Service account may already exist"
    
    # Grant necessary roles
    local roles=(
        "roles/bigquery.admin"
        "roles/storage.admin"
        "roles/dataflow.admin"
        "roles/cloudfunctions.admin"
        "roles/cloudscheduler.admin"
        "roles/secretmanager.admin"
        "roles/aiplatform.admin"
        "roles/composer.admin"
        "roles/iam.serviceAccountUser"
    )
    
    for role in "${roles[@]}"; do
        log_info "Granting $role to service account..."
        gcloud projects add-iam-policy-binding "$GCP_PROJECT" \
            --member="serviceAccount:${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
            --role="$role" \
            --quiet
    done
    
    log_info "Service account created and configured"
}

# Create storage buckets
create_storage_buckets() {
    log_info "Creating Cloud Storage buckets..."
    
    local buckets=("$GCS_RAW_BUCKET" "$GCS_PROCESSED_BUCKET" "$GCS_MODELS_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        log_info "Creating bucket: gs://$bucket"
        gsutil mb -p "$GCP_PROJECT" -l "$GCP_REGION" "gs://$bucket" 2>/dev/null || log_warn "Bucket $bucket may already exist"
        
        # Set lifecycle rules for raw and processed buckets
        if [[ "$bucket" == "$GCS_RAW_BUCKET" ]] || [[ "$bucket" == "$GCS_PROCESSED_BUCKET" ]]; then
            cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "NEARLINE"
        },
        "condition": {
          "age": 30
        }
      },
      {
        "action": {
          "type": "SetStorageClass",
          "storageClass": "COLDLINE"
        },
        "condition": {
          "age": 90
        }
      }
    ]
  }
}
EOF
            gsutil lifecycle set /tmp/lifecycle.json "gs://$bucket"
            rm /tmp/lifecycle.json
        fi
    done
    
    log_info "Storage buckets created successfully"
}

# Create BigQuery dataset
create_bigquery_dataset() {
    log_info "Creating BigQuery dataset..."
    
    bq mk --project_id="$GCP_PROJECT" \
        --location="$GCP_REGION" \
        --dataset \
        --description="CQC data warehouse" \
        "$BQ_DATASET" 2>/dev/null || log_warn "Dataset may already exist"
    
    # Create tables
    log_info "Creating BigQuery tables..."
    
    # Providers table
    bq mk --table \
        --project_id="$GCP_PROJECT" \
        "$BQ_DATASET.providers" \
        ../sql/bigquery_schema_providers.json 2>/dev/null || log_warn "Table providers may already exist"
    
    # Locations table
    bq mk --table \
        --project_id="$GCP_PROJECT" \
        "$BQ_DATASET.locations" \
        ../sql/bigquery_schema_locations.json 2>/dev/null || log_warn "Table locations may already exist"
    
    # ML features table
    bq mk --table \
        --project_id="$GCP_PROJECT" \
        "$BQ_DATASET.ml_features" \
        ../sql/bigquery_schema_ml_features.json 2>/dev/null || log_warn "Table ml_features may already exist"
    
    log_info "BigQuery dataset and tables created successfully"
}

# Create Secret Manager secrets
create_secrets() {
    log_info "Creating Secret Manager secrets..."
    
    # Create CQC API key secret (placeholder - user needs to update)
    echo -n "YOUR_CQC_API_KEY_HERE" | gcloud secrets create cqc-api-key \
        --project="$GCP_PROJECT" \
        --replication-policy="automatic" \
        --data-file=- 2>/dev/null || log_warn "Secret cqc-api-key may already exist"
    
    log_warn "Remember to update the cqc-api-key secret with your actual API key"
    
    log_info "Secrets created successfully"
}

# Create App Engine app (required for Cloud Scheduler)
create_app_engine() {
    log_info "Creating App Engine app for Cloud Scheduler..."
    
    gcloud app create --region="$GCP_REGION" --project="$GCP_PROJECT" 2>/dev/null || log_warn "App Engine app may already exist"
    
    log_info "App Engine app created successfully"
}

# Create Cloud Composer environment
create_composer_environment() {
    log_info "Creating Cloud Composer environment (this may take 20-30 minutes)..."
    
    gcloud composer environments create "$COMPOSER_ENV" \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --python-version=3 \
        --machine-type=n1-standard-2 \
        --node-count=3 \
        --disk-size=100 \
        --env-variables=GCP_PROJECT="$GCP_PROJECT",GCP_REGION="$GCP_REGION",GCS_RAW_BUCKET="$GCS_RAW_BUCKET",GCS_PROCESSED_BUCKET="$GCS_PROCESSED_BUCKET",BQ_DATASET="$BQ_DATASET" \
        2>/dev/null || log_warn "Composer environment may already exist"
    
    log_info "Cloud Composer environment created successfully"
}

# Create Vertex AI dataset
create_vertex_ai_resources() {
    log_info "Creating Vertex AI resources..."
    
    # Create dataset placeholder
    log_info "Vertex AI datasets will be created during model training"
    
    # Create model registry
    log_info "Vertex AI model registry will be populated during model deployment"
    
    log_info "Vertex AI resources setup completed"
}

# Main execution
main() {
    log_info "Starting GCP resources setup..."
    
    check_env_vars
    set_defaults
    
    # Authenticate with GCP
    log_info "Setting GCP project..."
    gcloud config set project "$GCP_PROJECT"
    
    # Create resources
    enable_apis
    create_service_account
    create_storage_buckets
    create_bigquery_dataset
    create_secrets
    create_app_engine
    create_composer_environment
    create_vertex_ai_resources
    
    log_info "GCP resources setup completed successfully!"
    log_info "Next steps:"
    log_info "1. Update the cqc-api-key secret with your actual API key"
    log_info "2. Run deploy_all_services.sh to deploy the application services"
}

# Run main function
main "$@"