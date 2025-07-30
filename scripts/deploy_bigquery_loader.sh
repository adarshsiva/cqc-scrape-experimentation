#!/bin/bash
# Deploy the BigQuery loader job to Cloud Run

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
JOB_NAME="cqc-bigquery-loader"
SERVICE_ACCOUNT="744974744548-compute@developer.gserviceaccount.com"

echo "Building and deploying CQC BigQuery loader to Cloud Run..."

# Navigate to scripts directory
cd "$(dirname "$0")"

# Build and push the container using the process Dockerfile
gcloud builds submit \
  --tag gcr.io/${PROJECT_ID}/${JOB_NAME} \
  --project ${PROJECT_ID} \
  -f Dockerfile.process \
  .

# Check if job already exists
if gcloud run jobs describe ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID} >/dev/null 2>&1; then
    echo "Updating existing Cloud Run job..."
    gcloud run jobs update ${JOB_NAME} \
      --image gcr.io/${PROJECT_ID}/${JOB_NAME} \
      --region ${REGION} \
      --memory 4Gi \
      --cpu 2 \
      --task-timeout 3600 \
      --parallelism 1 \
      --max-retries 1 \
      --service-account ${SERVICE_ACCOUNT} \
      --set-env-vars GCP_PROJECT=${PROJECT_ID} \
      --project ${PROJECT_ID}
else
    echo "Creating new Cloud Run job..."
    gcloud run jobs create ${JOB_NAME} \
      --image gcr.io/${PROJECT_ID}/${JOB_NAME} \
      --region ${REGION} \
      --memory 4Gi \
      --cpu 2 \
      --task-timeout 3600 \
      --parallelism 1 \
      --max-retries 1 \
      --service-account ${SERVICE_ACCOUNT} \
      --set-env-vars GCP_PROJECT=${PROJECT_ID} \
      --project ${PROJECT_ID}
fi

echo "Cloud Run job deployed successfully!"
echo ""
echo "To execute the job manually, run:"
echo "gcloud run jobs execute ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}"
echo ""
echo "To check job execution logs:"
echo "gcloud run jobs executions list --job ${JOB_NAME} --region ${REGION} --project ${PROJECT_ID}"