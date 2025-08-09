#!/bin/bash

# Comprehensive script to test and deploy all components
# Run this after authentication: gcloud auth login

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "🚀 CQC Rating Predictor - Full Deployment Script"
echo "================================================"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Set project
echo "1️⃣ Setting GCP project..."
gcloud config set project $PROJECT_ID

# Step 2: Test model training
echo ""
echo "2️⃣ Testing model training pipeline..."
echo "Running model training job..."
gcloud run jobs execute cqc-model-trainer \
  --region=$REGION \
  --project=$PROJECT_ID \
  --wait

# Check if model was saved
echo "Checking if model was saved to GCS..."
gsutil ls gs://machine-learning-exp-467008-cqc-ml-artifacts/models/proactive/

# Step 3: Deploy enhanced CQC fetcher
echo ""
echo "3️⃣ Deploying enhanced CQC data fetcher..."

# Build and deploy fetcher
cd src/ingestion
gcloud builds submit \
  --config=cloudbuild-fetcher.yaml \
  --project=$PROJECT_ID \
  --region=$REGION
cd ../..

# Step 4: Test the fetcher
echo ""
echo "4️⃣ Testing CQC fetcher connection..."
echo "Note: This requires CQC API key in Secret Manager"
echo "To set API key: echo -n 'YOUR_KEY' | gcloud secrets create cqc-subscription-key --data-file=-"

# Create a test job with small batch
gcloud run jobs update cqc-data-fetcher-enhanced \
  --region=$REGION \
  --project=$PROJECT_ID \
  --set-env-vars="MAX_LOCATIONS=10,BATCH_SIZE=5"

# Execute test
gcloud run jobs execute cqc-data-fetcher-enhanced \
  --region=$REGION \
  --project=$PROJECT_ID \
  --wait

# Step 5: Check data in BigQuery
echo ""
echo "5️⃣ Checking data in BigQuery..."
bq query --use_legacy_sql=false --project_id=$PROJECT_ID \
  "SELECT COUNT(*) as total_locations FROM cqc_data.locations_detailed"

# Step 6: Deploy model to Vertex AI
echo ""
echo "6️⃣ Preparing model deployment to Vertex AI..."
echo "This step requires manual intervention:"
echo ""
echo "# Upload model to Vertex AI:"
echo "gcloud ai models upload \\"
echo "  --region=$REGION \\"
echo "  --display-name=cqc-risk-predictor \\"
echo "  --container-image-uri=gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest \\"
echo "  --artifact-uri=gs://machine-learning-exp-467008-cqc-ml-artifacts/models/proactive/"
echo ""
echo "# Create endpoint:"
echo "gcloud ai endpoints create \\"
echo "  --region=$REGION \\"
echo "  --display-name=cqc-risk-assessment"
echo ""

# Step 7: Check prediction API
echo ""
echo "7️⃣ Testing prediction API..."
curl -X GET https://proactive-risk-assessment-744974744548.europe-west2.run.app/health

echo ""
echo "✅ Deployment check complete!"
echo ""
echo "📊 Status Summary:"
echo "- Model Training: Check logs above"
echo "- CQC Fetcher: Deployed (needs API key)"
echo "- Prediction API: Running"
echo "- BigQuery Data: Check count above"
echo ""
echo "🔗 Useful commands:"
echo "- View training logs: gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=cqc-model-trainer\" --limit=50"
echo "- View fetcher logs: gcloud logging read \"resource.type=cloud_run_job AND resource.labels.job_name=cqc-data-fetcher-enhanced\" --limit=50"
echo "- Test prediction: curl -X POST https://proactive-risk-assessment-744974744548.europe-west2.run.app/assess-risk -H \"Content-Type: application/json\" -d '{\"locationId\": \"1-000000001\", \"numberOfBeds\": 25}'"