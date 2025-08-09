#!/bin/bash

# Deploy enhanced CQC data fetcher to Cloud Run Job
# This script deploys a robust fetcher with retry logic and 403 bypass strategies

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
SERVICE_NAME="cqc-data-fetcher-enhanced"

echo "🚀 Deploying Enhanced CQC Data Fetcher..."
echo "Project: $PROJECT_ID"
echo "Region: $REGION"

# First, ensure you're authenticated
echo "ℹ️  Please ensure you're authenticated with the correct account"
echo "Run: gcloud auth login"
echo "Then: gcloud config set project $PROJECT_ID"

# Create or update the API key secret (if not exists)
echo "📝 Setting up API key in Secret Manager..."
echo "If you have a CQC API key, create the secret with:"
echo "echo -n 'YOUR_API_KEY' | gcloud secrets create cqc-subscription-key --data-file=- --project=$PROJECT_ID"

# Build and deploy using Cloud Build
echo "🔨 Building and deploying with Cloud Build..."
gcloud builds submit \
  --config=src/ingestion/cloudbuild-fetcher.yaml \
  --project=$PROJECT_ID \
  --region=$REGION

echo "✅ Deployment complete!"
echo ""
echo "📋 Next steps:"
echo "1. Ensure the API key is set in Secret Manager:"
echo "   gcloud secrets versions list cqc-subscription-key --project=$PROJECT_ID"
echo ""
echo "2. Test the connection:"
echo "   gcloud run jobs execute $SERVICE_NAME --region=$REGION --project=$PROJECT_ID"
echo ""
echo "3. View logs:"
echo "   gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=$SERVICE_NAME\" --project=$PROJECT_ID --limit=50"
echo ""
echo "4. Schedule regular fetches (optional):"
echo "   gcloud scheduler jobs create http daily-cqc-fetch \\"
echo "     --location=$REGION \\"
echo "     --schedule=\"0 2 * * *\" \\"
echo "     --uri=\"https://cloudrun.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/jobs/$SERVICE_NAME:run\" \\"
echo "     --http-method=POST \\"
echo "     --oauth-service-account-email=cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com"