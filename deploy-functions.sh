#!/bin/bash

# Deploy Cloud Functions for CQC Rating Predictor
# Project: machine-learning-exp-467008

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "======================================"
echo "Deploying Cloud Functions"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "======================================"

# Deploy ingestion function
echo ""
echo "Deploying ingestion Cloud Function..."

# Package ingestion function
cd src/ingestion
zip -r ../../ingestion.zip . -x "*.pyc" -x "__pycache__/*"
cd ../..

# Upload to GCS
gsutil cp ingestion.zip gs://$PROJECT_ID-cqc-cf-source/

# Deploy function
gcloud functions deploy cqc-data-ingestion \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=gs://$PROJECT_ID-cqc-cf-source/ingestion.zip \
  --entry-point=ingest_cqc_data \
  --memory=512MB \
  --timeout=540s \
  --service-account=cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET=$PROJECT_ID-cqc-raw-data" \
  --trigger-http \
  --allow-unauthenticated

echo "Ingestion function deployed!"

# Deploy prediction function
echo ""
echo "Deploying prediction Cloud Function..."

# Package prediction function
cd src/prediction
zip -r ../../prediction.zip . -x "*.pyc" -x "__pycache__/*"
cd ../..

# Upload to GCS
gsutil cp prediction.zip gs://$PROJECT_ID-cqc-cf-source/

# Deploy function (without endpoint ID for now)
gcloud functions deploy cqc-rating-prediction \
  --gen2 \
  --runtime=python311 \
  --region=$REGION \
  --source=gs://$PROJECT_ID-cqc-cf-source/prediction.zip \
  --entry-point=predict_cqc_rating \
  --memory=1GB \
  --timeout=60s \
  --service-account=cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,VERTEX_ENDPOINT_ID=placeholder" \
  --trigger-http \
  --allow-unauthenticated

echo "Prediction function deployed!"

# Get function URLs
echo ""
echo "======================================"
echo "Cloud Functions deployed successfully!"
echo ""
echo "Function URLs:"
INGESTION_URL=$(gcloud functions describe cqc-data-ingestion --region=$REGION --format="value(url)")
PREDICTION_URL=$(gcloud functions describe cqc-rating-prediction --region=$REGION --format="value(url)")
echo "Ingestion: $INGESTION_URL"
echo "Prediction: $PREDICTION_URL"
echo ""
echo "To test ingestion, run:"
echo "  curl -X POST $INGESTION_URL"
echo "======================================"

# Clean up zip files
rm -f ingestion.zip prediction.zip