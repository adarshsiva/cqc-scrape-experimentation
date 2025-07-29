#!/bin/bash
# Deploy Proactive Risk Assessment API to Cloud Run

# Set variables
PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
SERVICE_NAME="proactive-risk-assessment"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# Ensure we're in the right directory
cd "$(dirname "$0")"

echo "Building Docker image for Proactive Risk Assessment API..."
# Build using the specific Dockerfile
docker build -f Dockerfile.proactive -t ${IMAGE_NAME} .

echo "Pushing image to Google Container Registry..."
docker push ${IMAGE_NAME}

echo "Deploying to Cloud Run..."
gcloud run deploy ${SERVICE_NAME} \
  --image ${IMAGE_NAME} \
  --platform managed \
  --region ${REGION} \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --concurrency 80 \
  --max-instances 10 \
  --set-env-vars "GCP_PROJECT=${PROJECT_ID},MODEL_BUCKET=${PROJECT_ID}-cqc-ml-artifacts,MODEL_PATH=models/proactive/model_package.pkl" \
  --allow-unauthenticated

echo "Deployment complete!"
echo "Service URL:"
gcloud run services describe ${SERVICE_NAME} --platform managed --region ${REGION} --format 'value(status.url)'