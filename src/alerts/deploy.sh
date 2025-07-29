#!/bin/bash

# Deploy CQC Alert Notification Service to Google Cloud Functions

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"your-project-id"}
REGION=${GCP_REGION:-"europe-west2"}
FUNCTION_NAME="cqc-alert-notifications"
ENTRY_POINT="main"
RUNTIME="python311"
MEMORY="512MB"
TIMEOUT="540s"
SERVICE_ACCOUNT="cqc-alerts@${PROJECT_ID}.iam.gserviceaccount.com"

# Environment variables for the function
ENV_VARS="GCP_PROJECT=${PROJECT_ID}"
ENV_VARS="${ENV_VARS},BIGQUERY_DATASET=cqc_data"
ENV_VARS="${ENV_VARS},ENABLE_EMAIL_NOTIFICATIONS=true"
ENV_VARS="${ENV_VARS},ENABLE_WEBHOOK_NOTIFICATIONS=true"
ENV_VARS="${ENV_VARS},ENABLE_SLACK_NOTIFICATIONS=false"
ENV_VARS="${ENV_VARS},ENABLE_TEAMS_NOTIFICATIONS=false"
ENV_VARS="${ENV_VARS},HIGH_RISK_THRESHOLD=70.0"

# Secrets to mount
SECRETS="smtp-password=smtp-password:latest"
SECRETS="${SECRETS},webhook-api-key=webhook-api-key:latest"

echo "Deploying CQC Alert Notification Service..."
echo "Project: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Function: ${FUNCTION_NAME}"

# Create service account if it doesn't exist
echo "Creating service account..."
gcloud iam service-accounts create cqc-alerts \
    --display-name="CQC Alert Notifications" \
    --project="${PROJECT_ID}" 2>/dev/null || echo "Service account already exists"

# Grant necessary permissions
echo "Granting permissions..."
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.dataViewer" \
    --condition=None

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/bigquery.jobUser" \
    --condition=None

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/secretmanager.secretAccessor" \
    --condition=None

# Deploy the function
echo "Deploying Cloud Function..."
gcloud functions deploy "${FUNCTION_NAME}" \
    --gen2 \
    --runtime="${RUNTIME}" \
    --region="${REGION}" \
    --source=. \
    --entry-point="${ENTRY_POINT}" \
    --trigger-http \
    --allow-unauthenticated \
    --memory="${MEMORY}" \
    --timeout="${TIMEOUT}" \
    --set-env-vars="${ENV_VARS}" \
    --set-secrets="${SECRETS}" \
    --service-account="${SERVICE_ACCOUNT}" \
    --project="${PROJECT_ID}"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --format="value(serviceConfig.uri)")

echo ""
echo "Deployment complete!"
echo "Function URL: ${FUNCTION_URL}"
echo ""
echo "To test the function:"
echo "  Health check: curl ${FUNCTION_URL}/health"
echo "  Dry run: curl -X POST ${FUNCTION_URL} -H 'Content-Type: application/json' -d '{\"dry_run\": true}'"
echo ""
echo "To schedule automatic alerts, create a Cloud Scheduler job:"
echo "  gcloud scheduler jobs create http cqc-daily-alerts \\"
echo "    --location=${REGION} \\"
echo "    --schedule='0 9 * * *' \\"
echo "    --uri=${FUNCTION_URL} \\"
echo "    --http-method=POST \\"
echo "    --headers='Content-Type=application/json' \\"
echo "    --message-body='{\"dry_run\": false}'"