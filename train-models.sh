#!/bin/bash

# Train ML Models for CQC Rating Prediction
# Project: machine-learning-exp-467008

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "======================================"
echo "Training ML Models"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "======================================"

# Check if ETL data exists in BigQuery
echo "Checking for processed data in BigQuery..."
LOCATIONS_COUNT=$(bq query --use_legacy_sql=false --format=csv "SELECT COUNT(*) as count FROM \`$PROJECT_ID.cqc_data.locations\`" | tail -n 1)

if [ "$LOCATIONS_COUNT" -eq "0" ] || [ -z "$LOCATIONS_COUNT" ]; then
    echo "ERROR: No data found in BigQuery. Please run ETL pipeline first."
    exit 1
fi

echo "Found $LOCATIONS_COUNT records in locations table"

# Install required Python packages
echo ""
echo "Installing required Python packages..."
pip install -r src/ml/requirements.txt

# Create ML pipeline
echo ""
echo "Creating and running ML pipeline..."
cd src/ml/pipeline

# Generate unique pipeline name
PIPELINE_NAME="cqc-ml-pipeline-$(date +%Y%m%d-%H%M%S)"

python pipeline.py \
  --project-id=$PROJECT_ID \
  --pipeline-root=gs://$PROJECT_ID-cqc-ml-artifacts/pipelines \
  --display-name=$PIPELINE_NAME \
  --service-account=cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --region=$REGION

cd ../../..

echo ""
echo "======================================"
echo "ML pipeline submitted!"
echo ""
echo "Pipeline name: $PIPELINE_NAME"
echo ""
echo "Monitor progress in the Vertex AI console:"
echo "https://console.cloud.google.com/vertex-ai/pipelines/runs?project=$PROJECT_ID"
echo ""
echo "After training completes, the model will be automatically deployed."
echo "======================================