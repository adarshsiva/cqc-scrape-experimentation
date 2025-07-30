#!/bin/bash
# Quick deployment script for Proactive Risk Assessment API to Cloud Run

set -e

# Configuration
PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
SERVICE_NAME="proactive-risk-assessment"

echo "=========================================="
echo "Proactive Risk Assessment API Deployment"
echo "=========================================="
echo ""
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "Service: $SERVICE_NAME"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: You are not authenticated with Google Cloud."
    echo ""
    echo "Please run the following commands:"
    echo "  gcloud auth login"
    echo "  gcloud config set project $PROJECT_ID"
    echo ""
    exit 1
fi

# Check current project
CURRENT_PROJECT=$(gcloud config get-value project 2>/dev/null)
if [ "$CURRENT_PROJECT" != "$PROJECT_ID" ]; then
    echo "⚠️  Warning: Current project is '$CURRENT_PROJECT', expected '$PROJECT_ID'"
    echo "Setting project to $PROJECT_ID..."
    gcloud config set project $PROJECT_ID
fi

echo "✓ Authentication verified"
echo ""

# Change to script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting deployment using Cloud Build..."
echo ""

# Submit Cloud Build job
cd ../..  # Go to repository root
gcloud builds submit \
    --config=src/prediction/cloudbuild_proactive.yaml \
    --substitutions=COMMIT_SHA=$(date +%Y%m%d-%H%M%S) \
    --project=$PROJECT_ID

echo ""
echo "=========================================="
echo "Deployment completed successfully!"
echo "=========================================="
echo ""

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
    --region=$REGION \
    --format='value(status.url)')

echo "Service URL: $SERVICE_URL"
echo ""
echo "Test the deployment:"
echo "  curl $SERVICE_URL/health"
echo ""
echo "For detailed testing, run:"
echo "  cd src/prediction"
echo "  python test_proactive_api.py"
echo ""
echo "Note: Update BASE_URL in test_proactive_api.py to: $SERVICE_URL"