#!/bin/bash

# CQC ML Pipeline Complete Deployment Script
# This script deploys the entire CQC data ingestion and ML pipeline on GCP

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION="europe-west2"
DATASET_ID="cqc_dataset"

echo "=========================================="
echo "CQC ML PIPELINE DEPLOYMENT"
echo "=========================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Set up GCP project
echo ""
echo "Step 1: Setting up GCP project..."
gcloud config set project $PROJECT_ID
print_status "Project configured: $PROJECT_ID"

# Step 2: Enable required APIs
echo ""
echo "Step 2: Enabling required GCP APIs..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    storage.googleapis.com \
    bigquery.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    dataflow.googleapis.com \
    aiplatform.googleapis.com \
    --project=$PROJECT_ID

print_status "APIs enabled"

# Step 3: Create service account for Cloud Run
echo ""
echo "Step 3: Setting up service account..."
SERVICE_ACCOUNT="cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT --project=$PROJECT_ID &>/dev/null; then
    gcloud iam service-accounts create cqc-fetcher \
        --display-name="CQC Data Fetcher Service Account" \
        --project=$PROJECT_ID
    print_status "Service account created"
else
    print_warning "Service account already exists"
fi

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/storage.admin" \
    --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/bigquery.dataEditor" \
    --condition=None

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$SERVICE_ACCOUNT" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

print_status "Service account permissions granted"

# Step 4: Create Cloud Storage buckets
echo ""
echo "Step 4: Creating Cloud Storage buckets..."

# Raw data bucket
if ! gsutil ls -b gs://${PROJECT_ID}-cqc-raw-data &>/dev/null; then
    gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-cqc-raw-data
    print_status "Created raw data bucket"
else
    print_warning "Raw data bucket already exists"
fi

# Processed data bucket
if ! gsutil ls -b gs://${PROJECT_ID}-cqc-processed &>/dev/null; then
    gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-cqc-processed
    print_status "Created processed data bucket"
else
    print_warning "Processed data bucket already exists"
fi

# Set lifecycle policies for cost optimization
cat > /tmp/lifecycle.json <<EOF
{
  "lifecycle": {
    "rule": [
      {
        "action": {"type": "Delete"},
        "condition": {
          "age": 90,
          "matchesPrefix": ["raw/"]
        }
      },
      {
        "action": {"type": "SetStorageClass", "storageClass": "NEARLINE"},
        "condition": {
          "age": 30,
          "matchesPrefix": ["raw/"]
        }
      }
    ]
  }
}
EOF

gsutil lifecycle set /tmp/lifecycle.json gs://${PROJECT_ID}-cqc-raw-data
gsutil lifecycle set /tmp/lifecycle.json gs://${PROJECT_ID}-cqc-processed
print_status "Lifecycle policies applied to buckets"

# Step 5: Set up BigQuery dataset and tables
echo ""
echo "Step 5: Setting up BigQuery..."
gcloud builds submit \
    --config=src/bigquery/cloudbuild-bigquery-setup.yaml \
    --project=$PROJECT_ID \
    --region=$REGION

print_status "BigQuery dataset and tables created"

# Step 6: Deploy the complete data fetcher to Cloud Run
echo ""
echo "Step 6: Deploying complete data fetcher to Cloud Run..."
gcloud builds submit \
    --config=src/ingestion/cloudbuild-complete-fetcher.yaml \
    --project=$PROJECT_ID \
    --region=$REGION

print_status "Complete data fetcher deployed to Cloud Run"

# Step 7: Set up Cloud Scheduler for automated runs
echo ""
echo "Step 7: Setting up Cloud Scheduler..."

# Create scheduler job for daily runs
if ! gcloud scheduler jobs describe cqc-daily-fetch --location=$REGION --project=$PROJECT_ID &>/dev/null; then
    gcloud scheduler jobs create http cqc-daily-fetch \
        --location=$REGION \
        --schedule="0 2 * * *" \
        --time-zone="Europe/London" \
        --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/cqc-complete-fetcher:run" \
        --http-method=POST \
        --oauth-service-account-email=$SERVICE_ACCOUNT \
        --project=$PROJECT_ID
    print_status "Cloud Scheduler job created for daily runs at 2 AM"
else
    print_warning "Cloud Scheduler job already exists"
fi

# Step 8: Run initial data fetch (optional)
echo ""
read -p "Do you want to run an initial data fetch now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Starting initial data fetch..."
    gcloud run jobs execute cqc-complete-fetcher \
        --region=$REGION \
        --project=$PROJECT_ID
    print_status "Initial data fetch started. Check Cloud Run logs for progress."
else
    print_warning "Skipping initial data fetch"
fi

# Step 9: Display deployment summary
echo ""
echo "=========================================="
echo "DEPLOYMENT SUMMARY"
echo "=========================================="
echo ""
print_status "Project: $PROJECT_ID"
print_status "Region: $REGION"
print_status "Service Account: $SERVICE_ACCOUNT"
print_status "Raw Data Bucket: gs://${PROJECT_ID}-cqc-raw-data"
print_status "Processed Bucket: gs://${PROJECT_ID}-cqc-processed"
print_status "BigQuery Dataset: ${PROJECT_ID}.${DATASET_ID}"
print_status "Cloud Run Job: cqc-complete-fetcher"
print_status "Scheduler Job: cqc-daily-fetch (daily at 2 AM)"
echo ""
echo "=========================================="
echo "NEXT STEPS"
echo "=========================================="
echo ""
echo "1. Monitor the data fetching job:"
echo "   gcloud run jobs executions list --job=cqc-complete-fetcher --region=$REGION"
echo ""
echo "2. Check BigQuery for ingested data:"
echo "   bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM \`${PROJECT_ID}.${DATASET_ID}.locations_complete\`'"
echo ""
echo "3. View Cloud Run logs:"
echo "   gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=cqc-complete-fetcher\" --limit=50"
echo ""
echo "4. Trigger manual run:"
echo "   gcloud run jobs execute cqc-complete-fetcher --region=$REGION"
echo ""
print_status "Deployment complete!"