#!/bin/bash
# Deploy Comprehensive CQC Extractor as Cloud Run Job
# This script sets up the infrastructure and deploys the extractor

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION=${GCP_REGION:-"europe-west2"}
SERVICE_ACCOUNT_NAME="cqc-extraction-sa"
ARTIFACT_REGISTRY_REPO="cqc-ml"

echo "Deploying Comprehensive CQC Extractor..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# 1. Enable required APIs
echo "Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com --project=$PROJECT_ID
gcloud services enable run.googleapis.com --project=$PROJECT_ID
gcloud services enable artifactregistry.googleapis.com --project=$PROJECT_ID
gcloud services enable secretmanager.googleapis.com --project=$PROJECT_ID
gcloud services enable storage.googleapis.com --project=$PROJECT_ID
gcloud services enable bigquery.googleapis.com --project=$PROJECT_ID

# 2. Create Artifact Registry repository if it doesn't exist
echo "Setting up Artifact Registry repository..."
if ! gcloud artifacts repositories describe $ARTIFACT_REGISTRY_REPO \
    --location=$REGION --project=$PROJECT_ID >/dev/null 2>&1; then
    gcloud artifacts repositories create $ARTIFACT_REGISTRY_REPO \
        --repository-format=docker \
        --location=$REGION \
        --description="CQC ML Pipeline Docker images" \
        --project=$PROJECT_ID
    echo "Created Artifact Registry repository: $ARTIFACT_REGISTRY_REPO"
else
    echo "Artifact Registry repository already exists: $ARTIFACT_REGISTRY_REPO"
fi

# 3. Create service account if it doesn't exist
echo "Setting up service account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com \
    --project=$PROJECT_ID >/dev/null 2>&1; then
    gcloud iam service-accounts create $SERVICE_ACCOUNT_NAME \
        --display-name="CQC Data Extraction Service Account" \
        --description="Service account for CQC comprehensive data extraction" \
        --project=$PROJECT_ID
    echo "Created service account: $SERVICE_ACCOUNT_NAME"
else
    echo "Service account already exists: $SERVICE_ACCOUNT_NAME"
fi

# 4. Grant necessary IAM roles to service account
echo "Configuring IAM permissions..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/storage.admin" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.dataEditor" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/bigquery.jobUser" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

# 5. Create required Cloud Storage buckets
echo "Setting up Cloud Storage buckets..."
BUCKET_RAW="${PROJECT_ID}-cqc-raw-data"
BUCKET_PROCESSED="${PROJECT_ID}-cqc-processed"

# Raw data bucket
if ! gsutil ls -b gs://$BUCKET_RAW >/dev/null 2>&1; then
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_RAW
    echo "Created bucket: gs://$BUCKET_RAW"
else
    echo "Bucket already exists: gs://$BUCKET_RAW"
fi

# Processed data bucket  
if ! gsutil ls -b gs://$BUCKET_PROCESSED >/dev/null 2>&1; then
    gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://$BUCKET_PROCESSED
    echo "Created bucket: gs://$BUCKET_PROCESSED"
else
    echo "Bucket already exists: gs://$BUCKET_PROCESSED"
fi

# 6. Create BigQuery dataset
echo "Setting up BigQuery dataset..."
if ! bq show --dataset=true ${PROJECT_ID}:cqc_data >/dev/null 2>&1; then
    bq mk --dataset --location=$REGION ${PROJECT_ID}:cqc_data
    echo "Created BigQuery dataset: cqc_data"
else
    echo "BigQuery dataset already exists: cqc_data"
fi

# 7. Check if CQC API key secret exists
echo "Checking CQC API key secret..."
if ! gcloud secrets describe cqc-subscription-key --project=$PROJECT_ID >/dev/null 2>&1; then
    echo "WARNING: CQC API key secret 'cqc-subscription-key' not found!"
    echo "Please create it with: gcloud secrets create cqc-subscription-key --data-file=- --project=$PROJECT_ID"
    echo "Or set the CQC_API_KEY environment variable"
    exit 1
else
    echo "CQC API key secret found: cqc-subscription-key"
fi

# 8. Build and deploy using Cloud Build
echo "Building and deploying with Cloud Build..."
cd $(dirname "$0")

# Substitute PROJECT_ID in YAML files
sed "s/\${PROJECT_ID}/$PROJECT_ID/g" deploy-comprehensive-extractor.yaml > deploy-comprehensive-extractor-final.yaml

# Trigger Cloud Build
gcloud builds submit \
    --config=cloudbuild-comprehensive-extractor.yaml \
    --project=$PROJECT_ID \
    --substitutions=_REGION=$REGION

echo ""
echo "Deployment complete!"
echo ""
echo "To run the comprehensive extraction:"
echo "gcloud run jobs execute cqc-comprehensive-extractor \\"
echo "  --region=$REGION \\"
echo "  --project=$PROJECT_ID \\"
echo "  --wait"
echo ""
echo "To run with custom parameters:"
echo "gcloud run jobs execute cqc-comprehensive-extractor \\"
echo "  --region=$REGION \\"
echo "  --project=$PROJECT_ID \\"
echo "  --update-env-vars=\"ENDPOINTS=locations,providers,inspection_areas,reports,assessment_groups,MAX_LOCATIONS=10000,PARALLEL_WORKERS=5\" \\"
echo "  --wait"
echo ""
echo "Monitor execution:"
echo "gcloud run jobs executions list --job=cqc-comprehensive-extractor --region=$REGION --project=$PROJECT_ID"