#!/bin/bash
# Script to run ETL and ML pipelines on Google Cloud

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
LOCATION="us-central1"

echo "===================="
echo "CQC Pipeline Runner"
echo "===================="
echo "Project ID: $PROJECT_ID"
echo "Location: $LOCATION"
echo ""

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if gcloud is installed
if ! command_exists gcloud; then
    echo "ERROR: gcloud CLI is not installed."
    echo "Please install the Google Cloud SDK: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check authentication
echo "Checking Google Cloud authentication..."
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "ERROR: No active Google Cloud authentication found."
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo "Setting project to $PROJECT_ID..."
gcloud config set project $PROJECT_ID

# Check if required APIs are enabled
echo "Checking required APIs..."
REQUIRED_APIS=(
    "dataflow.googleapis.com"
    "bigquery.googleapis.com"
    "aiplatform.googleapis.com"
    "storage-component.googleapis.com"
)

for api in "${REQUIRED_APIS[@]}"; do
    if ! gcloud services list --enabled --filter="name:$api" --format="value(name)" | grep -q "$api"; then
        echo "Enabling $api..."
        gcloud services enable $api
    else
        echo "✓ $api is enabled"
    fi
done

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing ETL dependencies..."
pip install --upgrade pip
pip install -r src/etl/requirements.txt

echo "Installing ML dependencies..."
pip install -r src/ml/requirements.txt
pip install kfp==2.3.0  # Kubeflow Pipelines SDK

# Run ETL Pipeline
echo ""
echo "=============================="
echo "Step 1: Running ETL Pipeline"
echo "=============================="
cd src/etl
python deploy_etl_pipeline.py --project-id=$PROJECT_ID

# Wait for user confirmation
echo ""
echo "Please check the Dataflow console to ensure ETL pipelines have completed:"
echo "https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
echo ""
read -p "Press Enter when ETL pipelines have completed successfully..."

# Run ML Pipeline
echo ""
echo "=============================="
echo "Step 2: Running ML Pipeline"
echo "=============================="
cd ../ml
python deploy_ml_pipeline.py --project-id=$PROJECT_ID --location=$LOCATION

echo ""
echo "=============================="
echo "Pipeline Execution Complete!"
echo "=============================="
echo ""
echo "Monitor your pipelines:"
echo "- Dataflow: https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
echo "- Vertex AI: https://console.cloud.google.com/vertex-ai/pipelines?project=$PROJECT_ID"
echo "- BigQuery: https://console.cloud.google.com/bigquery?project=$PROJECT_ID"
echo ""
echo "Once the ML pipeline completes, your model will be available at:"
echo "Endpoint: cqc-rating-predictor"