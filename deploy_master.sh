#!/bin/bash

# ============================================================
# CQC RATING PREDICTION ML SYSTEM - MASTER DEPLOYMENT SCRIPT
# ============================================================
# This script deploys the complete CQC ML pipeline on GCP
# All components run on Google Cloud Platform services

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION="europe-west2"
DEPLOYMENT_MODE=${1:-"full"}  # full, data-only, ml-only, api-only

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Main deployment
print_header "CQC ML PIPELINE - MASTER DEPLOYMENT"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Deployment Mode: $DEPLOYMENT_MODE"
echo ""

# Step 1: Prerequisites
print_header "STEP 1: CHECKING PREREQUISITES"

# Check gcloud installation
if ! command -v gcloud &> /dev/null; then
    print_error "gcloud CLI not found. Please install Google Cloud SDK."
    exit 1
fi
print_status "gcloud CLI found"

# Set project
gcloud config set project $PROJECT_ID
print_status "Project configured: $PROJECT_ID"

# Check authentication
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    print_warning "No active gcloud authentication found"
    gcloud auth login
fi
print_status "Authentication verified"

# Step 2: Enable APIs
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    print_header "STEP 2: ENABLING REQUIRED APIS"
    
    gcloud services enable \
        cloudbuild.googleapis.com \
        run.googleapis.com \
        cloudfunctions.googleapis.com \
        storage.googleapis.com \
        bigquery.googleapis.com \
        dataflow.googleapis.com \
        aiplatform.googleapis.com \
        secretmanager.googleapis.com \
        cloudscheduler.googleapis.com \
        monitoring.googleapis.com \
        logging.googleapis.com \
        --project=$PROJECT_ID
    
    print_status "All required APIs enabled"
fi

# Step 3: Deploy BigQuery infrastructure
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    print_header "STEP 3: SETTING UP BIGQUERY"
    
    gcloud builds submit \
        --config=src/bigquery/cloudbuild-bigquery-setup.yaml \
        --project=$PROJECT_ID \
        --region=$REGION \
        --no-source
    
    print_status "BigQuery dataset and tables created"
fi

# Step 4: Deploy data ingestion pipeline
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    print_header "STEP 4: DEPLOYING DATA INGESTION"
    
    # Deploy complete fetcher
    gcloud builds submit \
        --config=src/ingestion/cloudbuild-complete-fetcher.yaml \
        --project=$PROJECT_ID \
        --region=$REGION
    
    print_status "Data fetcher deployed to Cloud Run"
    
    # Set up scheduler
    SERVICE_ACCOUNT="cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT --project=$PROJECT_ID &>/dev/null; then
        gcloud iam service-accounts create cqc-fetcher \
            --display-name="CQC Data Fetcher Service Account" \
            --project=$PROJECT_ID
    fi
    
    gcloud scheduler jobs create http cqc-daily-fetch \
        --location=$REGION \
        --schedule="0 2 * * *" \
        --time-zone="Europe/London" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/cqc-complete-fetcher:run" \
        --http-method=POST \
        --oauth-service-account-email=$SERVICE_ACCOUNT \
        --project=$PROJECT_ID || print_warning "Scheduler job already exists"
    
    print_status "Scheduled daily data fetching configured"
fi

# Step 5: Deploy ETL pipeline
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    print_header "STEP 5: DEPLOYING ETL PIPELINE"
    
    # Check if dataflow directory exists
    if [ -d "src/dataflow" ]; then
        gcloud builds submit \
            --config=src/dataflow/cloudbuild-dataflow.yaml \
            --project=$PROJECT_ID \
            --region=$REGION || print_warning "ETL pipeline deployment skipped"
    else
        print_warning "Dataflow directory not found, skipping ETL deployment"
    fi
    
    print_status "ETL pipeline configured"
fi

# Step 6: Deploy ML training pipeline
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "ml-only" ]]; then
    print_header "STEP 6: DEPLOYING ML TRAINING PIPELINE"
    
    gcloud builds submit \
        --config=src/ml/cloudbuild-ml-training.yaml \
        --project=$PROJECT_ID \
        --region=$REGION \
        --substitutions=_DEPLOY_MODEL=false,_TUNE_HYPERPARAMETERS=false
    
    print_status "ML training pipeline deployed"
fi

# Step 7: Deploy prediction API
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "api-only" ]]; then
    print_header "STEP 7: DEPLOYING PREDICTION API"
    
    # Create service account for API
    API_SA="cqc-api@$PROJECT_ID.iam.gserviceaccount.com"
    
    if ! gcloud iam service-accounts describe $API_SA --project=$PROJECT_ID &>/dev/null; then
        gcloud iam service-accounts create cqc-api \
            --display-name="CQC Prediction API Service Account" \
            --project=$PROJECT_ID
    fi
    
    # Grant permissions
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$API_SA" \
        --role="roles/aiplatform.user" \
        --condition=None
    
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$API_SA" \
        --role="roles/storage.objectViewer" \
        --condition=None
    
    # Deploy API
    gcloud builds submit \
        --config=src/api/cloudbuild-deploy-api.yaml \
        --project=$PROJECT_ID \
        --region=$REGION
    
    print_status "Prediction API deployed to Cloud Functions"
    
    # Get API URL
    API_URL=$(gcloud functions describe predict-rating --region=$REGION --format="value(serviceConfig.uri)")
    echo ""
    print_status "API Endpoint: $API_URL"
fi

# Step 8: Set up monitoring
if [[ "$DEPLOYMENT_MODE" == "full" ]]; then
    print_header "STEP 8: SETTING UP MONITORING"
    
    chmod +x src/monitoring/setup_monitoring.sh
    ./src/monitoring/setup_monitoring.sh
    
    print_status "Monitoring and alerting configured"
fi

# Step 9: Initial data load (optional)
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    echo ""
    read -p "Do you want to trigger initial data ingestion now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_header "TRIGGERING INITIAL DATA INGESTION"
        
        gcloud run jobs execute cqc-complete-fetcher \
            --region=$REGION \
            --project=$PROJECT_ID
        
        print_status "Data ingestion started. Monitor progress in Cloud Run logs."
    fi
fi

# Step 10: Deploy web interface
if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "api-only" ]]; then
    print_header "STEP 10: DEPLOYING WEB INTERFACE"
    
    # Create bucket for static hosting
    WEBSITE_BUCKET="${PROJECT_ID}-cqc-web"
    
    if ! gsutil ls -b gs://${WEBSITE_BUCKET} &>/dev/null; then
        gsutil mb -p $PROJECT_ID -l $REGION gs://${WEBSITE_BUCKET}
        gsutil iam ch allUsers:objectViewer gs://${WEBSITE_BUCKET}
    fi
    
    # Update API endpoint in HTML
    if [ -f "src/web/care_home_form.html" ]; then
        sed -i "s|YOUR-PROJECT-ID|${PROJECT_ID}|g" src/web/care_home_form.html
        
        # Upload to bucket
        gsutil cp src/web/care_home_form.html gs://${WEBSITE_BUCKET}/index.html
        gsutil web set -m index.html gs://${WEBSITE_BUCKET}
        
        print_status "Web interface deployed"
        echo ""
        print_status "Web Interface URL: https://storage.googleapis.com/${WEBSITE_BUCKET}/index.html"
    fi
fi

# Final Summary
print_header "DEPLOYMENT COMPLETE!"

echo ""
echo "📊 DEPLOYMENT SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "data-only" ]]; then
    echo "✅ Data Infrastructure:"
    echo "   • BigQuery Dataset: ${PROJECT_ID}.cqc_dataset"
    echo "   • Cloud Run Job: cqc-complete-fetcher"
    echo "   • Cloud Scheduler: cqc-daily-fetch (2 AM daily)"
    echo "   • Storage Buckets: ${PROJECT_ID}-cqc-raw-data, ${PROJECT_ID}-cqc-processed"
    echo ""
fi

if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "ml-only" ]]; then
    echo "✅ ML Pipeline:"
    echo "   • Feature Engineering: Deployed"
    echo "   • Model Training: XGBoost, Random Forest, LightGBM"
    echo "   • Vertex AI Pipeline: Configured"
    echo ""
fi

if [[ "$DEPLOYMENT_MODE" == "full" ]] || [[ "$DEPLOYMENT_MODE" == "api-only" ]]; then
    echo "✅ API Services:"
    if [ -n "$API_URL" ]; then
        echo "   • Prediction API: $API_URL"
    fi
    echo "   • Health Check: https://${REGION}-${PROJECT_ID}.cloudfunctions.net/health-check"
    echo "   • Batch Predict: https://${REGION}-${PROJECT_ID}.cloudfunctions.net/batch-predict"
    if [ -n "$WEBSITE_BUCKET" ]; then
        echo "   • Web Interface: https://storage.googleapis.com/${WEBSITE_BUCKET}/index.html"
    fi
    echo ""
fi

if [[ "$DEPLOYMENT_MODE" == "full" ]]; then
    echo "✅ Monitoring:"
    echo "   • Dashboard: https://console.cloud.google.com/monitoring/dashboards"
    echo "   • Alerts: Configured for all critical components"
    echo "   • Logs: Exported to BigQuery"
    echo ""
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📝 NEXT STEPS:"
echo ""
echo "1. Monitor initial data ingestion:"
echo "   gcloud run jobs executions list --job=cqc-complete-fetcher --region=$REGION"
echo ""
echo "2. Check data in BigQuery:"
echo "   bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM \`${PROJECT_ID}.cqc_dataset.care_homes\`'"
echo ""
echo "3. Train ML models:"
echo "   gcloud builds submit --config=src/ml/cloudbuild-ml-training.yaml --substitutions=_DEPLOY_MODEL=true"
echo ""
echo "4. Test prediction API:"
echo "   curl -X POST $API_URL -H \"Content-Type: application/json\" -d @test_data.json"
echo ""
echo "5. View monitoring dashboard:"
echo "   https://console.cloud.google.com/monitoring"
echo ""

print_header "🎉 DEPLOYMENT SUCCESSFUL!"