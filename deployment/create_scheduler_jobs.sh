#!/bin/bash

#######################################
# Create Cloud Scheduler Jobs Script
# Sets up automated schedules for the CQC pipeline
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
    
    # Set defaults
    export SERVICE_ACCOUNT="${SERVICE_ACCOUNT:-cqc-service-account}"
    
    log_info "Prerequisites checked successfully"
}

# Get Cloud Function URLs
get_function_urls() {
    log_info "Getting Cloud Function URLs..."
    
    # Get ingestion function URL
    INGESTION_URL=$(gcloud functions describe cqc-api-ingestion \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(httpsTrigger.url)" 2>/dev/null)
    
    if [[ -z "$INGESTION_URL" ]]; then
        log_error "Could not find cqc-api-ingestion function. Please deploy it first."
        exit 1
    fi
    
    log_info "Ingestion function URL: $INGESTION_URL"
}

# Create daily data ingestion job
create_daily_ingestion_job() {
    log_info "Creating daily data ingestion job..."
    
    gcloud scheduler jobs create http daily-cqc-ingestion \
        --location="$GCP_REGION" \
        --schedule="0 2 * * *" \
        --time-zone="Europe/London" \
        --uri="$INGESTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"trigger":"scheduled","mode":"full"}' \
        --attempt-deadline="10m" \
        --max-retry-attempts=3 \
        --min-backoff="1m" \
        --max-backoff="10m" \
        --max-doublings=2 \
        --description="Daily CQC data ingestion at 2 AM UK time" \
        --project="$GCP_PROJECT" \
        --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        2>/dev/null || {
            log_warn "Job daily-cqc-ingestion may already exist. Updating..."
            gcloud scheduler jobs update http daily-cqc-ingestion \
                --location="$GCP_REGION" \
                --schedule="0 2 * * *" \
                --time-zone="Europe/London" \
                --uri="$INGESTION_URL" \
                --http-method=POST \
                --headers="Content-Type=application/json" \
                --message-body='{"trigger":"scheduled","mode":"full"}' \
                --attempt-deadline="10m" \
                --max-retry-attempts=3 \
                --min-backoff="1m" \
                --max-backoff="10m" \
                --max-doublings=2 \
                --description="Daily CQC data ingestion at 2 AM UK time" \
                --project="$GCP_PROJECT" \
                --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
        }
    
    log_info "Daily ingestion job created successfully"
}

# Create weekly full refresh job
create_weekly_refresh_job() {
    log_info "Creating weekly full refresh job..."
    
    gcloud scheduler jobs create http weekly-cqc-refresh \
        --location="$GCP_REGION" \
        --schedule="0 3 * * 0" \
        --time-zone="Europe/London" \
        --uri="$INGESTION_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"trigger":"scheduled","mode":"full_refresh","include_historical":true}' \
        --attempt-deadline="30m" \
        --max-retry-attempts=3 \
        --min-backoff="5m" \
        --max-backoff="30m" \
        --max-doublings=2 \
        --description="Weekly full CQC data refresh on Sunday at 3 AM UK time" \
        --project="$GCP_PROJECT" \
        --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        2>/dev/null || {
            log_warn "Job weekly-cqc-refresh may already exist. Updating..."
            gcloud scheduler jobs update http weekly-cqc-refresh \
                --location="$GCP_REGION" \
                --schedule="0 3 * * 0" \
                --time-zone="Europe/London" \
                --uri="$INGESTION_URL" \
                --http-method=POST \
                --headers="Content-Type=application/json" \
                --message-body='{"trigger":"scheduled","mode":"full_refresh","include_historical":true}' \
                --attempt-deadline="30m" \
                --max-retry-attempts=3 \
                --min-backoff="5m" \
                --max-backoff="30m" \
                --max-doublings=2 \
                --description="Weekly full CQC data refresh on Sunday at 3 AM UK time" \
                --project="$GCP_PROJECT" \
                --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
        }
    
    log_info "Weekly refresh job created successfully"
}

# Create model retraining job
create_model_retraining_job() {
    log_info "Creating model retraining job..."
    
    # This job triggers Vertex AI pipeline for model retraining
    gcloud scheduler jobs create http monthly-model-retraining \
        --location="$GCP_REGION" \
        --schedule="0 4 1 * *" \
        --time-zone="Europe/London" \
        --uri="https://${GCP_REGION}-aiplatform.googleapis.com/v1/projects/${GCP_PROJECT}/locations/${GCP_REGION}/pipelineJobs" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body="{
            \"displayName\": \"cqc-model-retraining\",
            \"pipelineSpec\": {
                \"pipelineInfo\": {
                    \"name\": \"cqc-model-retraining-pipeline\"
                }
            }
        }" \
        --attempt-deadline="4h" \
        --max-retry-attempts=1 \
        --description="Monthly model retraining on the 1st at 4 AM UK time" \
        --project="$GCP_PROJECT" \
        --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        2>/dev/null || {
            log_warn "Job monthly-model-retraining may already exist. Updating..."
            gcloud scheduler jobs update http monthly-model-retraining \
                --location="$GCP_REGION" \
                --schedule="0 4 1 * *" \
                --time-zone="Europe/London" \
                --uri="https://${GCP_REGION}-aiplatform.googleapis.com/v1/projects/${GCP_PROJECT}/locations/${GCP_REGION}/pipelineJobs" \
                --http-method=POST \
                --headers="Content-Type=application/json" \
                --message-body="{
                    \"displayName\": \"cqc-model-retraining\",
                    \"pipelineSpec\": {
                        \"pipelineInfo\": {
                            \"name\": \"cqc-model-retraining-pipeline\"
                        }
                    }
                }" \
                --attempt-deadline="4h" \
                --max-retry-attempts=1 \
                --description="Monthly model retraining on the 1st at 4 AM UK time" \
                --project="$GCP_PROJECT" \
                --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
        }
    
    log_info "Model retraining job created successfully"
}

# Create data quality check job
create_data_quality_job() {
    log_info "Creating data quality check job..."
    
    # Create a Cloud Function URL for data quality checks (assuming it exists)
    DQ_URL="https://${GCP_REGION}-${GCP_PROJECT}.cloudfunctions.net/cqc-data-quality-check"
    
    gcloud scheduler jobs create http daily-data-quality-check \
        --location="$GCP_REGION" \
        --schedule="0 6 * * *" \
        --time-zone="Europe/London" \
        --uri="$DQ_URL" \
        --http-method=POST \
        --headers="Content-Type=application/json" \
        --message-body='{"check_type":"comprehensive","alert_on_failure":true}' \
        --attempt-deadline="30m" \
        --max-retry-attempts=2 \
        --min-backoff="5m" \
        --max-backoff="15m" \
        --description="Daily data quality check at 6 AM UK time" \
        --project="$GCP_PROJECT" \
        --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com" \
        2>/dev/null || {
            log_warn "Job daily-data-quality-check may already exist. Updating..."
            gcloud scheduler jobs update http daily-data-quality-check \
                --location="$GCP_REGION" \
                --schedule="0 6 * * *" \
                --time-zone="Europe/London" \
                --uri="$DQ_URL" \
                --http-method=POST \
                --headers="Content-Type=application/json" \
                --message-body='{"check_type":"comprehensive","alert_on_failure":true}' \
                --attempt-deadline="30m" \
                --max-retry-attempts=2 \
                --min-backoff="5m" \
                --max-backoff="15m" \
                --description="Daily data quality check at 6 AM UK time" \
                --project="$GCP_PROJECT" \
                --oidc-service-account-email="${SERVICE_ACCOUNT}@${GCP_PROJECT}.iam.gserviceaccount.com"
        }
    
    log_info "Data quality check job created successfully"
}

# List all scheduler jobs
list_scheduler_jobs() {
    log_info "Listing all Cloud Scheduler jobs..."
    
    gcloud scheduler jobs list \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="table(name,schedule,timeZone,state,lastAttemptTime)"
}

# Test run a job
test_job() {
    local job_name=$1
    log_info "Test running job: $job_name"
    
    gcloud scheduler jobs run "$job_name" \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT"
    
    log_info "Job $job_name triggered successfully. Check logs for execution status."
}

# Main execution
main() {
    log_info "Starting Cloud Scheduler jobs setup..."
    
    check_prerequisites
    get_function_urls
    
    # Create scheduler jobs
    create_daily_ingestion_job
    create_weekly_refresh_job
    create_model_retraining_job
    create_data_quality_job
    
    # List all jobs
    list_scheduler_jobs
    
    log_info "Cloud Scheduler jobs created successfully!"
    log_info "Next steps:"
    log_info "1. Test individual jobs using: gcloud scheduler jobs run JOB_NAME --location=$GCP_REGION"
    log_info "2. Monitor job execution in the Cloud Console"
    log_info "3. Set up alerting for job failures"
    
    # Optional: Test run daily ingestion job
    read -p "Would you like to test run the daily ingestion job now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        test_job "daily-cqc-ingestion"
    fi
}

# Run main function
main "$@"