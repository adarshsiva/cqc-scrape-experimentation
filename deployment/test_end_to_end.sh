#!/bin/bash

#######################################
# End-to-End System Test Script
# Tests the complete CQC pipeline from ingestion to prediction
#######################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TEST_RESULTS=()

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

log_test() {
    echo -e "${BLUE}[TEST]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Test result tracking
test_passed() {
    local test_name=$1
    ((TESTS_PASSED++))
    TEST_RESULTS+=("${GREEN}✓${NC} $test_name")
    log_info "Test passed: $test_name"
}

test_failed() {
    local test_name=$1
    local reason=$2
    ((TESTS_FAILED++))
    TEST_RESULTS+=("${RED}✗${NC} $test_name - $reason")
    log_error "Test failed: $test_name - $reason"
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
    
    # Check if required tools are installed
    command -v gcloud >/dev/null 2>&1 || { log_error "gcloud CLI is required but not installed."; exit 1; }
    command -v bq >/dev/null 2>&1 || { log_error "bq CLI is required but not installed."; exit 1; }
    command -v gsutil >/dev/null 2>&1 || { log_error "gsutil is required but not installed."; exit 1; }
    command -v curl >/dev/null 2>&1 || { log_error "curl is required but not installed."; exit 1; }
    command -v jq >/dev/null 2>&1 || { log_error "jq is required but not installed."; exit 1; }
    
    log_info "Prerequisites checked successfully"
}

# Test 1: GCP Resources
test_gcp_resources() {
    log_test "Testing GCP resources availability..."
    
    # Test Cloud Storage buckets
    for bucket in "$GCS_RAW_BUCKET" "$GCS_PROCESSED_BUCKET" "$GCS_MODELS_BUCKET"; do
        if gsutil ls -b "gs://$bucket" &>/dev/null; then
            test_passed "Storage bucket exists: $bucket"
        else
            test_failed "Storage bucket exists: $bucket" "Bucket not found"
        fi
    done
    
    # Test BigQuery dataset
    if bq ls -d --project_id="$GCP_PROJECT" | grep -q "$BQ_DATASET"; then
        test_passed "BigQuery dataset exists: $BQ_DATASET"
    else
        test_failed "BigQuery dataset exists: $BQ_DATASET" "Dataset not found"
    fi
    
    # Test Secret Manager
    if gcloud secrets describe cqc-api-key --project="$GCP_PROJECT" &>/dev/null; then
        test_passed "Secret Manager: cqc-api-key exists"
    else
        test_failed "Secret Manager: cqc-api-key exists" "Secret not found"
    fi
}

# Test 2: Cloud Functions
test_cloud_functions() {
    log_test "Testing Cloud Functions deployment..."
    
    # Test ingestion function
    INGESTION_URL=$(gcloud functions describe cqc-api-ingestion \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(httpsTrigger.url)" 2>/dev/null)
    
    if [[ -n "$INGESTION_URL" ]]; then
        test_passed "Cloud Function deployed: cqc-api-ingestion"
        
        # Test function with a simple health check
        response=$(curl -s -X POST "$INGESTION_URL" \
            -H "Content-Type: application/json" \
            -d '{"test": true}' \
            -w "\n%{http_code}" | tail -n1)
        
        if [[ "$response" == "200" ]] || [[ "$response" == "401" ]]; then
            test_passed "Cloud Function responding: cqc-api-ingestion"
        else
            test_failed "Cloud Function responding: cqc-api-ingestion" "HTTP $response"
        fi
    else
        test_failed "Cloud Function deployed: cqc-api-ingestion" "Function not found"
    fi
    
    # Test prediction service
    PREDICTION_URL=$(gcloud functions describe cqc-prediction-service \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(httpsTrigger.url)" 2>/dev/null)
    
    if [[ -n "$PREDICTION_URL" ]]; then
        test_passed "Cloud Function deployed: cqc-prediction-service"
    else
        test_failed "Cloud Function deployed: cqc-prediction-service" "Function not found"
    fi
}

# Test 3: API Ingestion
test_api_ingestion() {
    log_test "Testing API ingestion (limited test)..."
    
    # Create test request for a single provider
    test_request='{
        "mode": "test",
        "limit": 1,
        "endpoint": "providers"
    }'
    
    # Get function URL
    INGESTION_URL=$(gcloud functions describe cqc-api-ingestion \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(httpsTrigger.url)" 2>/dev/null)
    
    if [[ -z "$INGESTION_URL" ]]; then
        test_failed "API ingestion test" "Ingestion function not found"
        return
    fi
    
    # Invoke function
    response=$(curl -s -X POST "$INGESTION_URL" \
        -H "Content-Type: application/json" \
        -d "$test_request")
    
    if echo "$response" | jq -e '.success' &>/dev/null; then
        test_passed "API ingestion test request"
        
        # Check if data was written to GCS
        sleep 5  # Wait for data to be written
        if gsutil ls "gs://$GCS_RAW_BUCKET/providers/$(date +%Y/%m/%d)/" 2>/dev/null | grep -q ".json"; then
            test_passed "API ingestion data written to GCS"
        else
            test_failed "API ingestion data written to GCS" "No data found in bucket"
        fi
    else
        test_failed "API ingestion test request" "Function returned error"
    fi
}

# Test 4: BigQuery Tables
test_bigquery_tables() {
    log_test "Testing BigQuery tables..."
    
    # Test tables exist
    tables=("providers" "locations" "ml_features")
    for table in "${tables[@]}"; do
        if bq show --project_id="$GCP_PROJECT" "$BQ_DATASET.$table" &>/dev/null; then
            test_passed "BigQuery table exists: $table"
            
            # Check table schema
            schema=$(bq show --schema --format=prettyjson "$GCP_PROJECT:$BQ_DATASET.$table" 2>/dev/null)
            if [[ -n "$schema" ]]; then
                test_passed "BigQuery table has schema: $table"
            else
                test_failed "BigQuery table has schema: $table" "Schema is empty"
            fi
        else
            test_failed "BigQuery table exists: $table" "Table not found"
        fi
    done
}

# Test 5: Dataflow Pipeline
test_dataflow_pipeline() {
    log_test "Testing Dataflow pipeline template..."
    
    # Check if template exists
    if gsutil ls "gs://$GCS_PROCESSED_BUCKET/templates/cqc-etl-pipeline" &>/dev/null; then
        test_passed "Dataflow template exists"
        
        # Test launching a small job
        job_name="test-etl-$(date +%s)"
        gcloud dataflow jobs run "$job_name" \
            --gcs-location="gs://$GCS_PROCESSED_BUCKET/templates/cqc-etl-pipeline" \
            --region="$GCP_REGION" \
            --project="$GCP_PROJECT" \
            --parameters="testMode=true" \
            2>/dev/null || test_failed "Dataflow job launch" "Failed to start job"
        
        # Wait for job to start
        sleep 10
        
        # Check job status
        job_state=$(gcloud dataflow jobs list \
            --region="$GCP_REGION" \
            --project="$GCP_PROJECT" \
            --filter="name:$job_name" \
            --format="value(state)" \
            --limit=1)
        
        if [[ "$job_state" == "Running" ]] || [[ "$job_state" == "Done" ]]; then
            test_passed "Dataflow job started successfully"
            
            # Cancel test job if running
            if [[ "$job_state" == "Running" ]]; then
                gcloud dataflow jobs cancel "$job_name" \
                    --region="$GCP_REGION" \
                    --project="$GCP_PROJECT" &>/dev/null
            fi
        else
            test_failed "Dataflow job started successfully" "Job state: $job_state"
        fi
    else
        test_failed "Dataflow template exists" "Template not found"
    fi
}

# Test 6: ML Models
test_ml_models() {
    log_test "Testing ML models and endpoints..."
    
    # Check if models exist in GCS
    if gsutil ls "gs://$GCS_MODELS_BUCKET/models/" &>/dev/null; then
        test_passed "ML models bucket accessible"
        
        # List models
        model_count=$(gsutil ls "gs://$GCS_MODELS_BUCKET/models/" | wc -l)
        if [[ $model_count -gt 0 ]]; then
            test_passed "ML models found in bucket: $model_count"
        else
            test_failed "ML models found in bucket" "No models found"
        fi
    else
        test_failed "ML models bucket accessible" "Bucket or models directory not found"
    fi
    
    # Check Vertex AI endpoints
    endpoints=$(gcloud ai endpoints list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --filter="displayName:cqc-rating-predictor" \
        --format="value(name)" 2>/dev/null)
    
    if [[ -n "$endpoints" ]]; then
        test_passed "Vertex AI endpoint exists"
        
        # Check deployed models
        for endpoint in $endpoints; do
            deployed_models=$(gcloud ai endpoints describe "$endpoint" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --format="value(deployedModels[].id)" 2>/dev/null)
            
            if [[ -n "$deployed_models" ]]; then
                test_passed "Vertex AI endpoint has deployed models"
            else
                test_failed "Vertex AI endpoint has deployed models" "No models deployed"
            fi
        done
    else
        test_failed "Vertex AI endpoint exists" "No endpoints found"
    fi
}

# Test 7: Cloud Scheduler
test_cloud_scheduler() {
    log_test "Testing Cloud Scheduler jobs..."
    
    # List scheduler jobs
    jobs=$(gcloud scheduler jobs list \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(name)")
    
    expected_jobs=("daily-cqc-ingestion" "weekly-cqc-refresh" "monthly-model-retraining" "daily-data-quality-check")
    
    for job in "${expected_jobs[@]}"; do
        if echo "$jobs" | grep -q "$job"; then
            test_passed "Scheduler job exists: $job"
            
            # Check job state
            job_state=$(gcloud scheduler jobs describe "$job" \
                --location="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --format="value(state)" 2>/dev/null)
            
            if [[ "$job_state" == "ENABLED" ]]; then
                test_passed "Scheduler job enabled: $job"
            else
                test_failed "Scheduler job enabled: $job" "State: $job_state"
            fi
        else
            test_failed "Scheduler job exists: $job" "Job not found"
        fi
    done
}

# Test 8: End-to-End Prediction
test_end_to_end_prediction() {
    log_test "Testing end-to-end prediction flow..."
    
    # Get prediction service URL
    PREDICTION_URL=$(gcloud functions describe cqc-prediction-service \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(httpsTrigger.url)" 2>/dev/null)
    
    if [[ -z "$PREDICTION_URL" ]]; then
        test_failed "End-to-end prediction" "Prediction service not found"
        return
    fi
    
    # Create test prediction request
    test_prediction='{
        "provider_id": "1-1000000001",
        "features": {
            "number_of_locations": 5,
            "primary_inspection_category": "Adult social care",
            "region": "London",
            "local_authority": "Westminster",
            "total_beds": 50,
            "total_staff": 30
        }
    }'
    
    # Make prediction request
    response=$(curl -s -X POST "$PREDICTION_URL" \
        -H "Content-Type: application/json" \
        -d "$test_prediction" \
        -w "\n%{http_code}")
    
    http_code=$(echo "$response" | tail -n1)
    body=$(echo "$response" | head -n-1)
    
    if [[ "$http_code" == "200" ]]; then
        if echo "$body" | jq -e '.prediction' &>/dev/null; then
            test_passed "End-to-end prediction successful"
            
            # Check prediction format
            prediction=$(echo "$body" | jq -r '.prediction')
            if [[ "$prediction" =~ ^(Outstanding|Good|Requires improvement|Inadequate)$ ]]; then
                test_passed "Prediction format valid: $prediction"
            else
                test_failed "Prediction format valid" "Invalid prediction: $prediction"
            fi
        else
            test_failed "End-to-end prediction successful" "No prediction in response"
        fi
    else
        test_failed "End-to-end prediction successful" "HTTP $http_code"
    fi
}

# Generate test report
generate_report() {
    echo
    echo "========================================"
    echo "        TEST EXECUTION SUMMARY          "
    echo "========================================"
    echo
    
    for result in "${TEST_RESULTS[@]}"; do
        echo -e "  $result"
    done
    
    echo
    echo "========================================"
    echo -e "  Total Tests: $((TESTS_PASSED + TESTS_FAILED))"
    echo -e "  ${GREEN}Passed: $TESTS_PASSED${NC}"
    echo -e "  ${RED}Failed: $TESTS_FAILED${NC}"
    
    if [[ $TESTS_FAILED -eq 0 ]]; then
        echo -e "  ${GREEN}Result: ALL TESTS PASSED${NC}"
    else
        echo -e "  ${RED}Result: SOME TESTS FAILED${NC}"
    fi
    
    echo "========================================"
    echo
}

# Main execution
main() {
    log_info "Starting end-to-end system test..."
    
    check_prerequisites
    
    # Run all tests
    test_gcp_resources
    test_cloud_functions
    test_api_ingestion
    test_bigquery_tables
    test_dataflow_pipeline
    test_ml_models
    test_cloud_scheduler
    test_end_to_end_prediction
    
    # Generate report
    generate_report
    
    # Exit with appropriate code
    if [[ $TESTS_FAILED -eq 0 ]]; then
        log_info "All tests completed successfully!"
        exit 0
    else
        log_error "Some tests failed. Please check the report above."
        exit 1
    fi
}

# Run main function
main "$@"