#!/bin/bash

# Submit Vertex AI training job

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
TRAINING_IMAGE="gcr.io/${PROJECT_ID}/cqc-ml-trainer:latest"
JOB_NAME="cqc-rating-predictor-training-$(date +%Y%m%d-%H%M%S)"

echo "Submitting training job: ${JOB_NAME}"

gcloud ai custom-jobs create \
  --region=${REGION} \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=${TRAINING_IMAGE} \
  --args="--project-id=${PROJECT_ID},--input-path=gs://${PROJECT_ID}-cqc-ml-artifacts/training_data/,--output-path=gs://${PROJECT_ID}-cqc-ml-artifacts/models/$(date +%Y%m%d)/,--model-type=xgboost" \
  --service-account=cqc-cf-service-account@${PROJECT_ID}.iam.gserviceaccount.com

echo "Training job submitted. Check progress in the Vertex AI console."