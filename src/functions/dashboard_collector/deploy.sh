#!/bin/bash

# Deploy Dashboard Metrics Collector Cloud Function
# Usage: ./deploy.sh

set -e

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
FUNCTION_NAME="dashboard-metrics-collector"

echo "🚀 Deploying Dashboard Metrics Collector Cloud Function..."

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo "❌ No active gcloud authentication found. Please run 'gcloud auth login'"
    exit 1
fi

# Set the project
echo "📋 Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Create Pub/Sub topic if it doesn't exist
echo "📢 Creating Pub/Sub topic 'dashboard-events'..."
gcloud pubsub topics create dashboard-events --project=$PROJECT_ID || echo "ℹ️  Topic already exists"

# Create BigQuery dataset and table if they don't exist
echo "🗄️  Setting up BigQuery resources..."
bq mk --dataset --location=$REGION $PROJECT_ID:cqc_data || echo "ℹ️  Dataset already exists"

bq mk --table \
    $PROJECT_ID:cqc_data.dashboard_metrics \
    metric_type:STRING,provider_id:STRING,location_id:STRING,collection_timestamp:TIMESTAMP,raw_metrics:JSON,derived_metrics:JSON \
    || echo "ℹ️  Table already exists"

# Deploy the Cloud Function
echo "☁️  Deploying Cloud Function..."
gcloud functions deploy $FUNCTION_NAME \
    --source=. \
    --entry-point=collect_dashboard_metrics \
    --runtime=python311 \
    --trigger=http \
    --allow-unauthenticated \
    --region=$REGION \
    --project=$PROJECT_ID \
    --memory=512MB \
    --timeout=300s \
    --max-instances=10 \
    --set-env-vars=PROJECT_ID=$PROJECT_ID,REGION=$REGION

# Set up IAM permissions
echo "🔐 Setting up IAM permissions..."

# Get the Cloud Function service account
FUNCTION_SA="$PROJECT_ID@appspot.gserviceaccount.com"

# Grant necessary permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/pubsub.publisher" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/bigquery.dataEditor" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/aiplatform.user" \
    --quiet

gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:$FUNCTION_SA" \
    --role="roles/secretmanager.secretAccessor" \
    --quiet

echo "✅ Deployment complete!"

# Get the function URL
FUNCTION_URL=$(gcloud functions describe $FUNCTION_NAME --region=$REGION --format="value(httpsTrigger.url)")
echo "🌐 Function URL: $FUNCTION_URL"

echo ""
echo "🧪 Test the function with:"
echo "curl -X POST $FUNCTION_URL \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"provider_id\": \"test-provider\", \"dashboard_type\": \"incidents\"}'"

echo ""
echo "📊 Monitor the function at:"
echo "https://console.cloud.google.com/functions/details/$REGION/$FUNCTION_NAME?project=$PROJECT_ID"