#!/bin/bash

#######################################
# Cleanup GCP Resources Script
# Removes all CQC project resources (USE WITH CAUTION)
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

# Confirmation prompt
confirm_cleanup() {
    echo -e "${RED}WARNING: This script will DELETE all CQC project resources!${NC}"
    echo "This includes:"
    echo "  - Cloud Storage buckets and all data"
    echo "  - BigQuery datasets and tables"
    echo "  - Cloud Functions"
    echo "  - Cloud Scheduler jobs"
    echo "  - Vertex AI models and endpoints"
    echo "  - Cloud Composer environment"
    echo "  - Service accounts"
    echo
    read -p "Are you ABSOLUTELY SURE you want to continue? Type 'DELETE ALL' to confirm: " confirmation
    
    if [[ "$confirmation" != "DELETE ALL" ]]; then
        log_info "Cleanup cancelled"
        exit 0
    fi
    
    read -p "This action CANNOT be undone. Type your project ID ($GCP_PROJECT) to confirm: " project_confirm
    
    if [[ "$project_confirm" != "$GCP_PROJECT" ]]; then
        log_info "Project ID mismatch. Cleanup cancelled"
        exit 0
    fi
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
    
    # Set defaults
    export GCS_RAW_BUCKET="${GCS_RAW_BUCKET:-${GCP_PROJECT}-cqc-raw}"
    export GCS_PROCESSED_BUCKET="${GCS_PROCESSED_BUCKET:-${GCP_PROJECT}-cqc-processed}"
    export GCS_MODELS_BUCKET="${GCS_MODELS_BUCKET:-${GCP_PROJECT}-cqc-models}"
    export BQ_DATASET="${BQ_DATASET:-cqc_data}"
    export COMPOSER_ENV="${COMPOSER_ENV:-cqc-composer-env}"
    export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-cqc-service-account}"
}

# Delete Cloud Scheduler jobs
delete_scheduler_jobs() {
    log_info "Deleting Cloud Scheduler jobs..."
    
    jobs=$(gcloud scheduler jobs list \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    if [[ -n "$jobs" ]]; then
        while IFS= read -r job; do
            log_info "Deleting scheduler job: $job"
            gcloud scheduler jobs delete "$job" \
                --location="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --quiet || log_warn "Failed to delete job: $job"
        done <<< "$jobs"
    else
        log_info "No scheduler jobs found"
    fi
}

# Delete Cloud Functions
delete_cloud_functions() {
    log_info "Deleting Cloud Functions..."
    
    functions=("cqc-api-ingestion" "cqc-prediction-service")
    
    for func in "${functions[@]}"; do
        if gcloud functions describe "$func" \
            --region="$GCP_REGION" \
            --project="$GCP_PROJECT" &>/dev/null; then
            log_info "Deleting function: $func"
            gcloud functions delete "$func" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --quiet || log_warn "Failed to delete function: $func"
        fi
    done
}

# Delete Vertex AI resources
delete_vertex_ai_resources() {
    log_info "Deleting Vertex AI resources..."
    
    # Delete endpoints
    endpoints=$(gcloud ai endpoints list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    if [[ -n "$endpoints" ]]; then
        while IFS= read -r endpoint; do
            log_info "Undeploying models from endpoint: $endpoint"
            
            # Undeploy all models first
            deployed_models=$(gcloud ai endpoints describe "$endpoint" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --format="value(deployedModels[].id)")
            
            if [[ -n "$deployed_models" ]]; then
                while IFS= read -r model_id; do
                    gcloud ai endpoints undeploy-model "$endpoint" \
                        --deployed-model-id="$model_id" \
                        --region="$GCP_REGION" \
                        --project="$GCP_PROJECT" \
                        --quiet || log_warn "Failed to undeploy model: $model_id"
                done <<< "$deployed_models"
            fi
            
            log_info "Deleting endpoint: $endpoint"
            gcloud ai endpoints delete "$endpoint" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --quiet || log_warn "Failed to delete endpoint: $endpoint"
        done <<< "$endpoints"
    fi
    
    # Delete models
    models=$(gcloud ai models list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    if [[ -n "$models" ]]; then
        while IFS= read -r model; do
            log_info "Deleting model: $model"
            gcloud ai models delete "$model" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --quiet || log_warn "Failed to delete model: $model"
        done <<< "$models"
    fi
}

# Delete Cloud Composer environment
delete_composer_environment() {
    log_info "Deleting Cloud Composer environment..."
    
    if gcloud composer environments describe "$COMPOSER_ENV" \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" &>/dev/null; then
        log_info "Deleting Composer environment: $COMPOSER_ENV (this may take 20-30 minutes)"
        gcloud composer environments delete "$COMPOSER_ENV" \
            --location="$GCP_REGION" \
            --project="$GCP_PROJECT" \
            --quiet || log_warn "Failed to delete Composer environment"
    fi
}

# Delete BigQuery resources
delete_bigquery_resources() {
    log_info "Deleting BigQuery resources..."
    
    if bq ls -d --project_id="$GCP_PROJECT" | grep -q "$BQ_DATASET"; then
        log_info "Deleting BigQuery dataset: $BQ_DATASET"
        bq rm -r -f -d "$GCP_PROJECT:$BQ_DATASET" || log_warn "Failed to delete dataset"
    fi
}

# Delete Cloud Storage buckets
delete_storage_buckets() {
    log_info "Deleting Cloud Storage buckets..."
    
    buckets=("$GCS_RAW_BUCKET" "$GCS_PROCESSED_BUCKET" "$GCS_MODELS_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        if gsutil ls -b "gs://$bucket" &>/dev/null; then
            log_info "Deleting bucket: gs://$bucket"
            gsutil -m rm -r "gs://$bucket" || log_warn "Failed to delete bucket: $bucket"
        fi
    done
}

# Delete secrets
delete_secrets() {
    log_info "Deleting Secret Manager secrets..."
    
    secrets=$(gcloud secrets list \
        --project="$GCP_PROJECT" \
        --filter="name:cqc-" \
        --format="value(name)")
    
    if [[ -n "$secrets" ]]; then
        while IFS= read -r secret; do
            log_info "Deleting secret: $secret"
            gcloud secrets delete "$secret" \
                --project="$GCP_PROJECT" \
                --quiet || log_warn "Failed to delete secret: $secret"
        done <<< "$secrets"
    fi
}

# Delete service account
delete_service_account() {
    log_info "Deleting service account..."
    
    sa_email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
    
    if gcloud iam service-accounts describe "$sa_email" \
        --project="$GCP_PROJECT" &>/dev/null; then
        log_info "Deleting service account: $sa_email"
        gcloud iam service-accounts delete "$sa_email" \
            --project="$GCP_PROJECT" \
            --quiet || log_warn "Failed to delete service account"
    fi
}

# Main execution
main() {
    log_info "Starting resource cleanup..."
    
    check_prerequisites
    confirm_cleanup
    
    # Delete resources in reverse dependency order
    delete_scheduler_jobs
    delete_cloud_functions
    delete_vertex_ai_resources
    delete_composer_environment
    delete_bigquery_resources
    delete_storage_buckets
    delete_secrets
    delete_service_account
    
    log_info "Resource cleanup completed!"
    log_warn "Some APIs may still be enabled. Disable them manually if needed."
}

# Run main function
main "$@"