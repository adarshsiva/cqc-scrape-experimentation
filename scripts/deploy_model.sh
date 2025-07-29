#!/bin/bash

# Deploy model to Vertex AI

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
MODEL_NAME="cqc-rating-predictor"
MODEL_PATH="gs://machine-learning-exp-467008-cqc-ml-artifacts/models/20250729/model.pkl"
TIMESTAMP=$(date +%Y%m%d%H%M%S)

echo "Uploading model to Vertex AI..."

# Upload model
MODEL_UPLOAD_RESPONSE=$(gcloud ai models upload \
  --region=${REGION} \
  --display-name="${MODEL_NAME}-${TIMESTAMP}" \
  --container-image-uri="europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest" \
  --artifact-uri="$(dirname ${MODEL_PATH})" \
  --format=json)

MODEL_ID=$(echo $MODEL_UPLOAD_RESPONSE | jq -r '.name' | awk -F'/' '{print $NF}')

echo "Model uploaded with ID: ${MODEL_ID}"

# Check if endpoint exists
ENDPOINT_NAME="${MODEL_NAME}-endpoint"
ENDPOINT_LIST=$(gcloud ai endpoints list --region=${REGION} --filter="displayName:${ENDPOINT_NAME}" --format=json)

if [[ $(echo $ENDPOINT_LIST | jq '. | length') -eq 0 ]]; then
    echo "Creating new endpoint..."
    ENDPOINT_RESPONSE=$(gcloud ai endpoints create \
      --region=${REGION} \
      --display-name="${ENDPOINT_NAME}" \
      --format=json)
    ENDPOINT_ID=$(echo $ENDPOINT_RESPONSE | jq -r '.name' | awk -F'/' '{print $NF}')
else
    echo "Using existing endpoint..."
    ENDPOINT_ID=$(echo $ENDPOINT_LIST | jq -r '.[0].name' | awk -F'/' '{print $NF}')
fi

echo "Endpoint ID: ${ENDPOINT_ID}"

# Deploy model to endpoint
echo "Deploying model to endpoint..."
gcloud ai endpoints deploy-model ${ENDPOINT_ID} \
  --region=${REGION} \
  --model=${MODEL_ID} \
  --display-name="${MODEL_NAME}-deployment" \
  --machine-type="n1-standard-2" \
  --min-replica-count=1 \
  --max-replica-count=3 \
  --traffic-split=0=100

echo "Model deployed successfully!"
echo ""
echo "To update the prediction service with this endpoint:"
echo "gcloud run services update cqc-rating-prediction \\"
echo "  --region=${REGION} \\"
echo "  --update-env-vars=\"VERTEX_ENDPOINT_ID=${ENDPOINT_ID}\""