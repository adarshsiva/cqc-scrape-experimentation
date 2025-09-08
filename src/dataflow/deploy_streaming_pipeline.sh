#!/bin/bash

# Deploy CQC Streaming Feature Pipeline
# This script deploys the complete streaming infrastructure using Cloud Build

set -e  # Exit on any error

# Configuration
PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2" 
BUILD_CONFIG="cloudbuild-dataflow-streaming.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 CQC Streaming Feature Pipeline Deployment${NC}"
echo -e "${BLUE}===============================================${NC}"
echo ""

# Check if required tools are available
echo -e "${YELLOW}Checking prerequisites...${NC}"

if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}❌ gcloud CLI is not installed${NC}"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 is not installed${NC}" 
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Set project
echo -e "\n${YELLOW}Setting up GCP project...${NC}"
gcloud config set project $PROJECT_ID
echo -e "${GREEN}✅ Project set to: $PROJECT_ID${NC}"

# Check if we're in the right directory
if [ ! -f "$BUILD_CONFIG" ]; then
    echo -e "${RED}❌ Build configuration file not found: $BUILD_CONFIG${NC}"
    echo -e "${YELLOW}Please run this script from the src/dataflow directory${NC}"
    exit 1
fi

# Enable required APIs
echo -e "\n${YELLOW}Enabling required Google Cloud APIs...${NC}"
apis=(
    "cloudbuild.googleapis.com"
    "dataflow.googleapis.com"
    "pubsub.googleapis.com"
    "bigquery.googleapis.com"
    "aiplatform.googleapis.com"
    "storage.googleapis.com"
    "logging.googleapis.com"
    "monitoring.googleapis.com"
)

for api in "${apis[@]}"; do
    echo -e "Enabling $api..."
    gcloud services enable $api
done

echo -e "${GREEN}✅ APIs enabled successfully${NC}"

# Create service account if it doesn't exist
echo -e "\n${YELLOW}Setting up service accounts...${NC}"
SERVICE_ACCOUNT="cloudbuild-dataflow@${PROJECT_ID}.iam.gserviceaccount.com"

if ! gcloud iam service-accounts describe $SERVICE_ACCOUNT &>/dev/null; then
    echo -e "Creating service account: cloudbuild-dataflow"
    gcloud iam service-accounts create cloudbuild-dataflow \
        --display-name="Cloud Build Dataflow Service Account" \
        --description="Service account for Cloud Build to deploy Dataflow pipelines"
fi

# Grant required roles
echo -e "Granting required roles to service account..."
roles=(
    "roles/dataflow.admin"
    "roles/dataflow.worker" 
    "roles/pubsub.admin"
    "roles/bigquery.admin"
    "roles/storage.admin"
    "roles/aiplatform.admin"
    "roles/logging.logWriter"
    "roles/monitoring.metricWriter"
    "roles/cloudbuild.builds.editor"
)

for role in "${roles[@]}"; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:$SERVICE_ACCOUNT" \
        --role="$role" \
        --quiet
done

echo -e "${GREEN}✅ Service account configured${NC}"

# Show deployment plan
echo -e "\n${BLUE}📋 Deployment Plan${NC}"
echo -e "${BLUE}==================${NC}"
echo -e "Project ID: $PROJECT_ID"
echo -e "Region: $REGION"
echo -e "Build Config: $BUILD_CONFIG"
echo -e "Service Account: $SERVICE_ACCOUNT"
echo ""
echo -e "This deployment will create:"
echo -e "  • Pub/Sub topic: dashboard-events"
echo -e "  • BigQuery dataset: cqc_dataset"
echo -e "  • BigQuery tables: realtime_features, realtime_features_aggregated, streaming_errors"
echo -e "  • GCS buckets for Dataflow temp and staging"
echo -e "  • Vertex AI Feature Store: cqc_feature_store"
echo -e "  • Dataflow streaming job: cqc-streaming-features-[BUILD_ID]"
echo -e "  • BigQuery views for analytics"
echo -e "  • Monitoring and alerting setup"
echo ""

# Confirm deployment
read -p "Do you want to proceed with the deployment? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Deployment cancelled by user${NC}"
    exit 0
fi

# Start Cloud Build
echo -e "\n${YELLOW}Starting Cloud Build deployment...${NC}"
echo -e "${BLUE}This may take 10-30 minutes depending on resource creation${NC}"

BUILD_ID=$(gcloud builds submit \
    --config=$BUILD_CONFIG \
    --timeout=3600s \
    --format="value(id)")

if [ -z "$BUILD_ID" ]; then
    echo -e "${RED}❌ Failed to start Cloud Build${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Cloud Build started successfully${NC}"
echo -e "Build ID: $BUILD_ID"
echo -e "You can monitor the build at:"
echo -e "${BLUE}https://console.cloud.google.com/cloud-build/builds/$BUILD_ID?project=$PROJECT_ID${NC}"

# Wait for build completion with progress updates
echo -e "\n${YELLOW}Waiting for build completion...${NC}"

# Function to get build status
get_build_status() {
    gcloud builds describe $BUILD_ID --format="value(status)" 2>/dev/null || echo "UNKNOWN"
}

# Function to show build progress
show_progress() {
    local status=$(get_build_status)
    case $status in
        "WORKING")
            echo -e "${YELLOW}⏳ Build in progress...${NC}"
            ;;
        "SUCCESS") 
            echo -e "${GREEN}✅ Build completed successfully!${NC}"
            return 0
            ;;
        "FAILURE"|"CANCELLED"|"TIMEOUT")
            echo -e "${RED}❌ Build failed with status: $status${NC}"
            return 1
            ;;
        *)
            echo -e "${YELLOW}📊 Build status: $status${NC}"
            ;;
    esac
    return 2
}

# Poll build status every 30 seconds
while true; do
    show_progress
    case $? in
        0) break ;;  # Success
        1) exit 1 ;; # Failure
        2) sleep 30 ;; # Continue polling
    esac
done

# Show deployment results
echo -e "\n${GREEN}🎉 Deployment completed successfully!${NC}"
echo -e "${GREEN}=====================================${NC}"

# Get the Dataflow job name
DATAFLOW_JOB=$(gcloud dataflow jobs list \
    --region=$REGION \
    --filter="name~cqc-streaming-features" \
    --format="value(name)" \
    --limit=1)

echo -e "\n${BLUE}📊 Deployed Resources${NC}"
echo -e "Pub/Sub Topic: projects/$PROJECT_ID/topics/dashboard-events"
echo -e "BigQuery Dataset: $PROJECT_ID:cqc_dataset"
echo -e "Dataflow Job: $DATAFLOW_JOB"
echo -e "Feature Store: cqc_feature_store"

echo -e "\n${BLUE}🔗 Useful Links${NC}"
echo -e "Dataflow Console: https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
echo -e "BigQuery Console: https://console.cloud.google.com/bigquery?project=$PROJECT_ID"
echo -e "Pub/Sub Console: https://console.cloud.google.com/cloudpubsub?project=$PROJECT_ID"
echo -e "Vertex AI Console: https://console.cloud.google.com/vertex-ai?project=$PROJECT_ID"

echo -e "\n${BLUE}🧪 Test the Pipeline${NC}"
echo -e "Send a test event:"
echo -e "${YELLOW}gcloud pubsub topics publish dashboard-events --message='{\"location_id\":\"test-123\",\"event_type\":\"dashboard_interaction\",\"timestamp\":\"$(date -Iseconds)\",\"metrics\":{\"user_type\":\"public\",\"page_views\":1}}'${NC}"

echo -e "\nQuery processed data:"
echo -e "${YELLOW}bq query --use_legacy_sql=false 'SELECT * FROM \`$PROJECT_ID.cqc_dataset.realtime_features\` ORDER BY event_timestamp DESC LIMIT 10'${NC}"

echo -e "\n${GREEN}✨ Streaming pipeline is now active and processing events!${NC}"