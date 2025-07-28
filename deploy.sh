#!/bin/bash

# CQC Rating Predictor Deployment Script
# Project: machine-learning-exp-467008

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
CQC_SUBSCRIPTION_KEY="45bdb9898457429783644ff69da1b9c9"
CQC_PARTNER_CODE=""  # Optional - increases rate limit if provided

echo "======================================"
echo "CQC Rating Predictor Deployment"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "======================================"

# Verify gcloud authentication
echo "Checking authentication..."
ACTIVE_ACCOUNT=$(gcloud auth list --filter=status:ACTIVE --format="value(account)")
echo "Active account: $ACTIVE_ACCOUNT"

# Set project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Phase 1: Enable APIs
echo ""
echo "Phase 1: Enabling required APIs..."
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  storage-component.googleapis.com \
  bigquery.googleapis.com \
  dataflow.googleapis.com \
  composer.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  cloudresourcemanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  compute.googleapis.com \
  appengine.googleapis.com

echo "APIs enabled successfully!"

# Phase 2: Create Service Accounts
echo ""
echo "Phase 2: Creating service accounts..."

# Check if service accounts already exist
if ! gcloud iam service-accounts describe cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create cqc-cf-service-account \
      --display-name="CQC Cloud Functions Service Account"
fi

if ! gcloud iam service-accounts describe cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create cqc-dataflow-service-account \
      --display-name="CQC Dataflow Service Account"
fi

if ! gcloud iam service-accounts describe cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com &>/dev/null; then
    gcloud iam service-accounts create cqc-vertex-service-account \
      --display-name="CQC Vertex AI Service Account"
fi

echo "Service accounts created!"

# Assign IAM roles
echo "Assigning IAM roles..."

# Cloud Functions permissions
for role in "roles/secretmanager.secretAccessor" "roles/storage.admin" "roles/bigquery.dataEditor"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
      --role="$role" --quiet
done

# Dataflow permissions
for role in "roles/dataflow.worker" "roles/bigquery.dataEditor" "roles/storage.objectAdmin"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
      --role="$role" --quiet
done

# Vertex AI permissions
for role in "roles/aiplatform.user" "roles/bigquery.dataViewer" "roles/storage.admin"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
      --member="serviceAccount:cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
      --role="$role" --quiet
done

echo "IAM roles assigned!"

# Phase 3: Create Storage Buckets
echo ""
echo "Phase 3: Creating storage buckets..."

for bucket in "cqc-raw-data" "cqc-dataflow-temp" "cqc-ml-artifacts" "cqc-cf-source"; do
    if ! gsutil ls gs://$PROJECT_ID-$bucket &>/dev/null; then
        gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$PROJECT_ID-$bucket
    else
        echo "Bucket gs://$PROJECT_ID-$bucket already exists"
    fi
done

echo "Storage buckets created!"

# Phase 4: Store CQC API Credentials
echo ""
echo "Phase 4: Storing CQC API credentials..."

# Create or update secrets
if gcloud secrets describe cqc-subscription-key &>/dev/null; then
    echo -n "$CQC_SUBSCRIPTION_KEY" | gcloud secrets versions add cqc-subscription-key --data-file=-
else
    echo -n "$CQC_SUBSCRIPTION_KEY" | gcloud secrets create cqc-subscription-key --data-file=-
fi

# Only create partner code secret if provided
if [ ! -z "$CQC_PARTNER_CODE" ]; then
    if gcloud secrets describe cqc-partner-code &>/dev/null; then
        echo -n "$CQC_PARTNER_CODE" | gcloud secrets versions add cqc-partner-code --data-file=-
    else
        echo -n "$CQC_PARTNER_CODE" | gcloud secrets create cqc-partner-code --data-file=-
    fi
    
    gcloud secrets add-iam-policy-binding cqc-partner-code \
      --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
      --role="roles/secretmanager.secretAccessor" --quiet
else
    echo "No partner code provided - using standard rate limits"
fi

# Grant access to service accounts
gcloud secrets add-iam-policy-binding cqc-subscription-key \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --quiet

gcloud secrets add-iam-policy-binding cqc-partner-code \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor" --quiet

echo "Secrets stored successfully!"

# Phase 5: Create BigQuery Dataset and Tables
echo ""
echo "Phase 5: Creating BigQuery dataset and tables..."

# Create dataset if it doesn't exist
if ! bq ls -d $PROJECT_ID:cqc_data &>/dev/null; then
    bq mk --dataset \
      --location=$REGION \
      --description="CQC ratings data and predictions" \
      $PROJECT_ID:cqc_data
fi

# Create tables with schemas
echo "Creating BigQuery tables..."

# Create locations table
if ! bq ls $PROJECT_ID:cqc_data.locations &>/dev/null; then
    bq mk --table \
      --time_partitioning_field=ingestion_timestamp \
      --time_partitioning_type=DAY \
      --clustering_fields=region,overall_rating \
      $PROJECT_ID:cqc_data.locations \
      config/bigquery_schema_locations.json
fi

# Create providers table  
if ! bq ls $PROJECT_ID:cqc_data.providers &>/dev/null; then
    bq mk --table \
      --time_partitioning_field=ingestion_timestamp \
      --time_partitioning_type=DAY \
      $PROJECT_ID:cqc_data.providers \
      config/bigquery_schema_providers.json
fi

# Create predictions table
if ! bq ls $PROJECT_ID:cqc_data.predictions &>/dev/null; then
    bq mk --table \
      --time_partitioning_field=prediction_timestamp \
      --time_partitioning_type=DAY \
      $PROJECT_ID:cqc_data.predictions \
      config/bigquery_schema_predictions.json
fi

# Create ml_features table
if ! bq ls $PROJECT_ID:cqc_data.ml_features &>/dev/null; then
    bq mk --table \
      $PROJECT_ID:cqc_data.ml_features \
      config/bigquery_schema_ml_features.json
fi

echo "BigQuery tables created!"
echo ""
echo "======================================"
echo "Initial setup completed!"
echo ""
echo "Next steps:"
echo "1. Create BigQuery table schemas from the combined schema file"
echo "2. Deploy Cloud Functions"
echo "3. Run initial data ingestion"
echo "4. Execute ETL pipeline"
echo "5. Train ML models"
echo ""
echo "To continue deployment, run:"
echo "  ./deploy-functions.sh"
echo "======================================"