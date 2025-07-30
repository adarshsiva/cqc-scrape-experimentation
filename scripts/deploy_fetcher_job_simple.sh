#!/bin/bash

# Simple deployment script for CQC Data Fetcher Cloud Run Job
# This assumes the image has already been built and pushed

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
IMAGE_NAME="gcr.io/$PROJECT_ID/cqc-data-fetcher:latest"
SERVICE_ACCOUNT="cqc-ml-processor@$PROJECT_ID.iam.gserviceaccount.com"

echo "Deploying CQC Data Fetcher Cloud Run Job..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"

# Check if the job already exists
if gcloud run jobs describe cqc-data-fetcher --region=$REGION --project=$PROJECT_ID &>/dev/null; then
    echo "Job already exists. Updating..."
    ACTION="update"
else
    echo "Creating new job..."
    ACTION="deploy"
fi

# Deploy or update the Cloud Run Job
gcloud run jobs $ACTION cqc-data-fetcher \
  --image=$IMAGE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --timeout=30m \
  --service-account=$SERVICE_ACCOUNT \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET=$PROJECT_ID-cqc-raw-data,MAX_LOCATIONS=1000" \
  --quiet

echo "Deployment completed!"

# Ask if user wants to execute the job now
read -p "Do you want to execute the job now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Executing the Cloud Run job..."
    gcloud run jobs execute cqc-data-fetcher \
      --project=$PROJECT_ID \
      --region=$REGION
    
    echo "Job execution started. Monitoring status..."
    
    # Wait a few seconds then show the execution status
    sleep 5
    gcloud run jobs executions list \
      --project=$PROJECT_ID \
      --region=$REGION \
      --job=cqc-data-fetcher \
      --limit=1
fi