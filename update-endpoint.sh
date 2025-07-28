#!/bin/bash

# Update Prediction Function with Model Endpoint
# Project: machine-learning-exp-467008

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "======================================"
echo "Updating Prediction Function with Model Endpoint"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "======================================"

# Get the latest deployed endpoint
echo "Finding deployed model endpoints..."
ENDPOINT_ID=$(gcloud ai endpoints list \
  --region=$REGION \
  --format="value(name)" \
  --filter="displayName:cqc-rating-predictor*" \
  --limit=1 | cut -d'/' -f6)

if [ -z "$ENDPOINT_ID" ]; then
    echo "ERROR: No deployed model endpoint found."
    echo "Please ensure the ML pipeline has completed and deployed a model."
    exit 1
fi

echo "Found endpoint ID: $ENDPOINT_ID"

# Update the prediction function with the endpoint ID
echo ""
echo "Updating prediction function with endpoint ID..."
gcloud functions deploy cqc-rating-prediction \
  --gen2 \
  --region=$REGION \
  --update-env-vars="VERTEX_ENDPOINT_ID=$ENDPOINT_ID" \
  --quiet

echo ""
echo "======================================"
echo "Prediction function updated successfully!"
echo ""
echo "The system is now fully deployed and ready to use."
echo ""
echo "Test the prediction API with:"
echo "  curl -X POST $(gcloud functions describe cqc-rating-prediction --region=$REGION --format='value(url)') \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d @test-prediction.json"
echo "======================================