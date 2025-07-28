#!/bin/bash

# CQC Prediction API Cloud Function Deployment Script

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"your-project-id"}
REGION=${GCP_REGION:-"europe-west2"}
FUNCTION_NAME="cqc-prediction-api"
ENTRY_POINT="predict"
RUNTIME="python311"
MEMORY="512MB"
TIMEOUT="60s"
MAX_INSTANCES="100"
MIN_INSTANCES="1"
SERVICE_ACCOUNT="cqc-prediction-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Vertex AI Configuration
VERTEX_ENDPOINT_ID=${VERTEX_ENDPOINT_ID:-"your-endpoint-id"}

echo "Deploying CQC Prediction API Cloud Function..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Function: ${FUNCTION_NAME}"

# Check if required environment variables are set
if [[ -z "${VERTEX_ENDPOINT_ID}" ]] || [[ "${VERTEX_ENDPOINT_ID}" == "your-endpoint-id" ]]; then
    echo "Error: VERTEX_ENDPOINT_ID environment variable must be set"
    echo "Usage: VERTEX_ENDPOINT_ID=<your-endpoint-id> ./deploy.sh"
    exit 1
fi

# Create service account if it doesn't exist
echo "Checking service account..."
if ! gcloud iam service-accounts describe ${SERVICE_ACCOUNT} --project=${PROJECT_ID} &>/dev/null; then
    echo "Creating service account..."
    gcloud iam service-accounts create cqc-prediction-sa \
        --display-name="CQC Prediction API Service Account" \
        --project=${PROJECT_ID}
    
    # Grant necessary permissions
    echo "Granting permissions to service account..."
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/aiplatform.user"
    
    gcloud projects add-iam-policy-binding ${PROJECT_ID} \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/logging.logWriter"
fi

# Deploy the function
echo "Deploying Cloud Function..."
gcloud functions deploy ${FUNCTION_NAME} \
    --gen2 \
    --runtime=${RUNTIME} \
    --region=${REGION} \
    --source=. \
    --entry-point=${ENTRY_POINT} \
    --trigger-http \
    --allow-unauthenticated \
    --memory=${MEMORY} \
    --timeout=${TIMEOUT} \
    --max-instances=${MAX_INSTANCES} \
    --min-instances=${MIN_INSTANCES} \
    --service-account=${SERVICE_ACCOUNT} \
    --set-env-vars="GCP_PROJECT=${PROJECT_ID},GCP_REGION=${REGION},VERTEX_ENDPOINT_ID=${VERTEX_ENDPOINT_ID},MODEL_NAME=cqc-rating-predictor" \
    --project=${PROJECT_ID}

# Get the function URL
FUNCTION_URL=$(gcloud functions describe ${FUNCTION_NAME} \
    --region=${REGION} \
    --format="value(serviceConfig.uri)" \
    --project=${PROJECT_ID})

echo ""
echo "Deployment completed successfully!"
echo "Function URL: ${FUNCTION_URL}"
echo ""
echo "To test the function, update the URL in test_prediction.py and run:"
echo "python test_prediction.py"

# Deploy health check function (optional)
read -p "Deploy health check endpoint as well? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deploying health check endpoint..."
    gcloud functions deploy ${FUNCTION_NAME}-health \
        --gen2 \
        --runtime=${RUNTIME} \
        --region=${REGION} \
        --source=. \
        --entry-point=health \
        --trigger-http \
        --allow-unauthenticated \
        --memory=256MB \
        --timeout=10s \
        --max-instances=10 \
        --project=${PROJECT_ID}
fi