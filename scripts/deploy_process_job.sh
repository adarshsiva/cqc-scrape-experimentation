#!/bin/bash
# Deploy the data processing job to Cloud Run

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
JOB_NAME="cqc-data-processor"

echo "Building and deploying CQC data processor to Cloud Run..."

# Build and push the container
gcloud builds submit --tag gcr.io/${PROJECT_ID}/${JOB_NAME} \
  --project ${PROJECT_ID} \
  .

# Deploy as Cloud Run Job
gcloud run jobs create ${JOB_NAME} \
  --image gcr.io/${PROJECT_ID}/${JOB_NAME} \
  --region ${REGION} \
  --memory 2Gi \
  --cpu 2 \
  --task-timeout 3600 \
  --parallelism 1 \
  --max-retries 1 \
  --service-account 744974744548-compute@developer.gserviceaccount.com \
  --set-env-vars GCP_PROJECT=${PROJECT_ID} \
  --project ${PROJECT_ID}

echo "Cloud Run job deployed!"
echo "To execute the job run:"
echo "gcloud run jobs execute ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}"