#!/bin/bash

# Deploy Dashboard Prediction API to Google Cloud Run
# This script automates the deployment process for the real-time CQC prediction service

set -e

# Configuration
PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION=${GCP_REGION:-"europe-west2"}
SERVICE_NAME="dashboard-prediction-api"
ENDPOINT_ID=${VERTEX_ENDPOINT_ID:-""}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying Dashboard Prediction API${NC}"
echo -e "Project: ${PROJECT_ID}"
echo -e "Region: ${REGION}"
echo -e "Service: ${SERVICE_NAME}"
echo -e "Endpoint ID: ${ENDPOINT_ID}"
echo ""

# Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI not found. Please install Google Cloud SDK.${NC}"
    exit 1
fi

if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker not found. Please install Docker.${NC}"
    exit 1
fi

# Set project
echo -e "${YELLOW}🔧 Setting up Google Cloud project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔌 Enabling required APIs...${NC}"
gcloud services enable run.googleapis.com
gcloud services enable cloudbuild.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable bigquery.googleapis.com
gcloud services enable secretmanager.googleapis.com

# Create API secret if it doesn't exist
echo -e "${YELLOW}🔐 Setting up API authentication...${NC}"
if ! gcloud secrets describe dashboard-api-key --quiet 2>/dev/null; then
    echo -e "${BLUE}Creating API secret...${NC}"
    # Generate a random API key
    API_KEY=$(openssl rand -hex 32)
    echo -n "$API_KEY" | gcloud secrets create dashboard-api-key --data-file=-
    echo -e "${GREEN}✅ API secret created: dashboard-api-key${NC}"
    echo -e "${YELLOW}⚠️  Save this API key for client authentication: $API_KEY${NC}"
else
    echo -e "${GREEN}✅ API secret already exists${NC}"
fi

# Build and deploy using Cloud Build
echo -e "${YELLOW}🏗️  Building and deploying with Cloud Build...${NC}"

# Update Cloud Build config with current endpoint ID
if [ ! -z "$ENDPOINT_ID" ]; then
    sed -i "s/_ENDPOINT_ID: ''/_ENDPOINT_ID: '$ENDPOINT_ID'/" src/api/cloudbuild-deploy-dashboard-api.yaml
fi

# Submit build
gcloud builds submit \
    --config=src/api/cloudbuild-deploy-dashboard-api.yaml \
    --substitutions=_REGION=$REGION,_ENDPOINT_ID=$ENDPOINT_ID \
    .

# Get service URL
echo -e "${YELLOW}🌐 Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo -e "${GREEN}🌐 Service URL: $SERVICE_URL${NC}"
echo ""
echo -e "${BLUE}📚 API Endpoints:${NC}"
echo -e "  Health Check: $SERVICE_URL/health"
echo -e "  Prediction:   $SERVICE_URL/api/cqc-prediction/dashboard/{care_home_id}"
echo ""
echo -e "${BLUE}🔧 Example Usage:${NC}"
echo -e "  # Health check"
echo -e "  curl $SERVICE_URL/health"
echo ""
echo -e "  # Make prediction (requires authentication)"
echo -e "  curl -H 'Authorization: Bearer YOUR_API_KEY' \\"
echo -e "       -H 'X-Client-ID: your_client_id' \\"
echo -e "       $SERVICE_URL/api/cqc-prediction/dashboard/care_home_123"
echo ""
echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo -e "  1. Set up your Vertex AI endpoint and update VERTEX_ENDPOINT_ID"
echo -e "  2. Configure your dashboard database connection"
echo -e "  3. Test the API with your client credentials"
echo -e "  4. Set up monitoring and alerting"
echo ""
echo -e "${GREEN}🎉 Dashboard Prediction API is ready!${NC}"