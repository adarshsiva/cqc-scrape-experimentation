#!/bin/bash

# Deploy CQC Data Fetcher as Cloud Run Job

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "Deploying CQC Data Fetcher to Cloud Run Job..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# Submit the Cloud Build job
gcloud builds submit \
  --project=$PROJECT_ID \
  --config=scripts/cloudbuild-fetcher.yaml \
  --substitutions=_REGION=$REGION \
  .

echo "Build and deployment completed!"

# Execute the job immediately
echo "Executing the Cloud Run job..."
gcloud run jobs execute cqc-data-fetcher \
  --project=$PROJECT_ID \
  --region=$REGION

echo "Job execution started. Check Cloud Run console for progress."

# Get job execution status
echo "Getting job execution status..."
gcloud run jobs executions list \
  --project=$PROJECT_ID \
  --region=$REGION \
  --job=cqc-data-fetcher \
  --limit=1