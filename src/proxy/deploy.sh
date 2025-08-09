#!/bin/bash

# Deploy CQC API proxy as Cloud Function

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
FUNCTION_NAME="cqc-api-proxy"

echo "🚀 Deploying CQC API Proxy Function..."

gcloud functions deploy $FUNCTION_NAME \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=. \
  --entry-point=cqc_proxy \
  --trigger-http \
  --allow-unauthenticated \
  --memory=256MB \
  --timeout=60s \
  --max-instances=10 \
  --project=$PROJECT_ID \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID"

echo "✅ Deployment complete!"
echo ""
echo "Test the proxy with:"
echo "curl \"https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME?endpoint=locations&page=1&perPage=5\""
echo ""
echo "Or fetch specific location:"
echo "curl \"https://$REGION-$PROJECT_ID.cloudfunctions.net/$FUNCTION_NAME?locationId=1-000000001\""