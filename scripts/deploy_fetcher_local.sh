#!/bin/bash

# Deploy CQC Data Fetcher using local Docker build and push

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
IMAGE_NAME="gcr.io/$PROJECT_ID/cqc-data-fetcher:latest"

echo "Building and deploying CQC Data Fetcher..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Image: $IMAGE_NAME"

# Navigate to scripts directory
cd scripts

# Build the Docker image locally
echo "Building Docker image..."
docker build -t $IMAGE_NAME -f Dockerfile.fetcher .

# Configure Docker to use gcloud as credential helper
echo "Configuring Docker authentication..."
gcloud auth configure-docker gcr.io --quiet

# Push the image to Container Registry
echo "Pushing image to Container Registry..."
docker push $IMAGE_NAME

# Deploy to Cloud Run as a job
echo "Deploying to Cloud Run Job..."
gcloud run jobs deploy cqc-data-fetcher \
  --image=$IMAGE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --memory=2Gi \
  --cpu=2 \
  --max-retries=1 \
  --timeout=30m \
  --service-account=cqc-ml-processor@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET=$PROJECT_ID-cqc-raw-data,MAX_LOCATIONS=1000" \
  --quiet

echo "Deployment completed!"

# Execute the job
echo "Executing the Cloud Run job..."
gcloud run jobs execute cqc-data-fetcher \
  --project=$PROJECT_ID \
  --region=$REGION \
  --wait

echo "Job execution completed!"

# Show recent executions
echo "Recent job executions:"
gcloud run jobs executions list \
  --project=$PROJECT_ID \
  --region=$REGION \
  --job=cqc-data-fetcher \
  --limit=5