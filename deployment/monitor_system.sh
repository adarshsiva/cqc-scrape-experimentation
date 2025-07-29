#!/bin/bash

#######################################
# System Monitoring Script
# Monitors the health of the CQC pipeline
#######################################

set -euo pipefail

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Monitoring interval (seconds)
MONITOR_INTERVAL="${MONITOR_INTERVAL:-60}"

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

log_metric() {
    echo -e "${BLUE}[METRIC]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# Check prerequisites
check_prerequisites() {
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
    export BQ_DATASET="${BQ_DATASET:-cqc_data}"
}

# Monitor Cloud Functions
monitor_cloud_functions() {
    log_info "Monitoring Cloud Functions..."
    
    functions=("cqc-api-ingestion" "cqc-prediction-service")
    
    for func in "${functions[@]}"; do
        # Get function status
        status=$(gcloud functions describe "$func" \
            --region="$GCP_REGION" \
            --project="$GCP_PROJECT" \
            --format="value(status)" 2>/dev/null || echo "NOT_FOUND")
        
        if [[ "$status" == "ACTIVE" ]]; then
            log_metric "$func: ${GREEN}ACTIVE${NC}"
            
            # Get recent errors (last hour)
            error_count=$(gcloud logging read \
                "resource.type=\"cloud_function\" \
                resource.labels.function_name=\"$func\" \
                severity>=ERROR \
                timestamp>=\"$(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S')Z\"" \
                --project="$GCP_PROJECT" \
                --format="value(timestamp)" | wc -l)
            
            if [[ $error_count -gt 0 ]]; then
                log_warn "$func: $error_count errors in last hour"
            fi
            
            # Get execution count (last hour)
            exec_count=$(gcloud logging read \
                "resource.type=\"cloud_function\" \
                resource.labels.function_name=\"$func\" \
                textPayload=\"Function execution started\" \
                timestamp>=\"$(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M:%S')Z\"" \
                --project="$GCP_PROJECT" \
                --format="value(timestamp)" | wc -l)
            
            log_metric "$func: $exec_count executions in last hour"
        else
            log_error "$func: ${RED}$status${NC}"
        fi
    done
}

# Monitor Storage Buckets
monitor_storage_buckets() {
    log_info "Monitoring Cloud Storage buckets..."
    
    buckets=("$GCS_RAW_BUCKET" "$GCS_PROCESSED_BUCKET")
    
    for bucket in "${buckets[@]}"; do
        # Get bucket size
        size=$(gsutil du -s "gs://$bucket" 2>/dev/null | awk '{print $1}' || echo "0")
        size_mb=$((size / 1024 / 1024))
        
        log_metric "$bucket: ${size_mb} MB"
        
        # Get object count
        object_count=$(gsutil ls -r "gs://$bucket/**" 2>/dev/null | wc -l || echo "0")
        log_metric "$bucket: $object_count objects"
        
        # Check recent uploads (last hour)
        recent_files=$(gsutil ls -l "gs://$bucket/**" 2>/dev/null | \
            grep -E "$(date -u '+%Y-%m-%d')" | wc -l || echo "0")
        
        if [[ $recent_files -gt 0 ]]; then
            log_metric "$bucket: $recent_files files uploaded today"
        fi
    done
}

# Monitor BigQuery
monitor_bigquery() {
    log_info "Monitoring BigQuery..."
    
    tables=("providers" "locations" "ml_features")
    
    for table in "${tables[@]}"; do
        # Get row count
        row_count=$(bq query --use_legacy_sql=false --format=csv \
            "SELECT COUNT(*) FROM \`$GCP_PROJECT.$BQ_DATASET.$table\`" \
            2>/dev/null | tail -n1 || echo "0")
        
        log_metric "$BQ_DATASET.$table: $row_count rows"
        
        # Check recent data (last 24 hours)
        if [[ "$table" != "ml_features" ]]; then
            recent_count=$(bq query --use_legacy_sql=false --format=csv \
                "SELECT COUNT(*) FROM \`$GCP_PROJECT.$BQ_DATASET.$table\` 
                WHERE last_updated >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)" \
                2>/dev/null | tail -n1 || echo "0")
            
            if [[ $recent_count -gt 0 ]]; then
                log_metric "$BQ_DATASET.$table: $recent_count records updated in last 24h"
            fi
        fi
    done
}

# Monitor Dataflow Jobs
monitor_dataflow_jobs() {
    log_info "Monitoring Dataflow jobs..."
    
    # Get running jobs
    running_jobs=$(gcloud dataflow jobs list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --status=active \
        --format="value(name)" | wc -l)
    
    log_metric "Dataflow: $running_jobs active jobs"
    
    # Get failed jobs (last 24 hours)
    failed_jobs=$(gcloud dataflow jobs list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --status=failed \
        --created-after="$(date -u -d '24 hours ago' '+%Y-%m-%dT%H:%M:%S')Z" \
        --format="value(name)" | wc -l)
    
    if [[ $failed_jobs -gt 0 ]]; then
        log_warn "Dataflow: $failed_jobs failed jobs in last 24h"
    fi
}

# Monitor Cloud Scheduler
monitor_scheduler_jobs() {
    log_info "Monitoring Cloud Scheduler jobs..."
    
    jobs=$(gcloud scheduler jobs list \
        --location="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --format="value(name,state,lastAttemptTime)")
    
    while IFS=$'\t' read -r name state last_attempt; do
        if [[ "$state" == "ENABLED" ]]; then
            log_metric "Scheduler $name: ${GREEN}ENABLED${NC}"
            
            # Check last execution time
            if [[ -n "$last_attempt" ]]; then
                last_attempt_epoch=$(date -d "$last_attempt" +%s 2>/dev/null || echo "0")
                current_epoch=$(date +%s)
                hours_ago=$(((current_epoch - last_attempt_epoch) / 3600))
                
                if [[ $hours_ago -lt 24 ]]; then
                    log_metric "Scheduler $name: Last run ${hours_ago}h ago"
                else
                    log_warn "Scheduler $name: Last run ${hours_ago}h ago"
                fi
            fi
        else
            log_warn "Scheduler $name: ${YELLOW}$state${NC}"
        fi
    done <<< "$jobs"
}

# Monitor Vertex AI
monitor_vertex_ai() {
    log_info "Monitoring Vertex AI endpoints..."
    
    endpoints=$(gcloud ai endpoints list \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" \
        --filter="displayName:cqc-rating-predictor" \
        --format="value(name)")
    
    if [[ -n "$endpoints" ]]; then
        for endpoint in $endpoints; do
            # Get deployed models count
            model_count=$(gcloud ai endpoints describe "$endpoint" \
                --region="$GCP_REGION" \
                --project="$GCP_PROJECT" \
                --format="value(deployedModels[].id)" | wc -l)
            
            if [[ $model_count -gt 0 ]]; then
                log_metric "Vertex AI endpoint: ${GREEN}$model_count models deployed${NC}"
            else
                log_warn "Vertex AI endpoint: ${YELLOW}No models deployed${NC}"
            fi
        done
    else
        log_warn "Vertex AI: No endpoints found"
    fi
}

# Generate summary report
generate_summary() {
    echo
    echo "========================================"
    echo "        SYSTEM HEALTH SUMMARY           "
    echo "========================================"
    echo
    
    # Quick health check
    all_good=true
    
    # Check critical services
    if ! gcloud functions describe cqc-api-ingestion \
        --region="$GCP_REGION" \
        --project="$GCP_PROJECT" &>/dev/null; then
        echo -e "  ${RED}✗${NC} API Ingestion Function"
        all_good=false
    else
        echo -e "  ${GREEN}✓${NC} API Ingestion Function"
    fi
    
    if ! bq show "$GCP_PROJECT:$BQ_DATASET" &>/dev/null; then
        echo -e "  ${RED}✗${NC} BigQuery Dataset"
        all_good=false
    else
        echo -e "  ${GREEN}✓${NC} BigQuery Dataset"
    fi
    
    if [[ "$all_good" == true ]]; then
        echo -e "\n  ${GREEN}Overall Status: HEALTHY${NC}"
    else
        echo -e "\n  ${RED}Overall Status: ISSUES DETECTED${NC}"
    fi
    
    echo "========================================"
    echo
}

# Continuous monitoring mode
continuous_monitor() {
    while true; do
        clear
        log_info "System Monitor - Press Ctrl+C to stop"
        echo
        
        monitor_cloud_functions
        echo
        monitor_storage_buckets
        echo
        monitor_bigquery
        echo
        monitor_dataflow_jobs
        echo
        monitor_scheduler_jobs
        echo
        monitor_vertex_ai
        
        generate_summary
        
        log_info "Next check in $MONITOR_INTERVAL seconds..."
        sleep "$MONITOR_INTERVAL"
    done
}

# Main execution
main() {
    check_prerequisites
    
    # Check if running in continuous mode
    if [[ "${1:-}" == "--continuous" ]] || [[ "${1:-}" == "-c" ]]; then
        continuous_monitor
    else
        # Single run
        log_info "Running system health check..."
        echo
        
        monitor_cloud_functions
        echo
        monitor_storage_buckets
        echo
        monitor_bigquery
        echo
        monitor_dataflow_jobs
        echo
        monitor_scheduler_jobs
        echo
        monitor_vertex_ai
        
        generate_summary
        
        log_info "Use --continuous or -c flag for continuous monitoring"
    fi
}

# Run main function
main "$@"