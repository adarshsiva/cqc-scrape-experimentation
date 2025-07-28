#!/bin/bash

# Run ETL Pipeline for CQC Data
# Project: machine-learning-exp-467008

set -e  # Exit on error

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"

echo "======================================"
echo "Running ETL Pipeline"
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo "======================================"

# Check if data exists in GCS
echo "Checking for raw data in GCS..."
LOCATIONS_COUNT=$(gsutil ls gs://$PROJECT_ID-cqc-raw-data/raw/locations/*.json 2>/dev/null | wc -l || echo "0")
PROVIDERS_COUNT=$(gsutil ls gs://$PROJECT_ID-cqc-raw-data/raw/providers/*.json 2>/dev/null | wc -l || echo "0")

if [ "$LOCATIONS_COUNT" -eq "0" ] || [ "$PROVIDERS_COUNT" -eq "0" ]; then
    echo "ERROR: No raw data found in GCS. Please run data ingestion first."
    echo "Run: curl -X POST $(gcloud functions describe cqc-data-ingestion --region=$REGION --format='value(url)')"
    exit 1
fi

echo "Found $LOCATIONS_COUNT location files and $PROVIDERS_COUNT provider files"

# Install required Python packages
echo ""
echo "Installing required Python packages..."
pip install -r src/etl/requirements.txt

# Run ETL for locations
echo ""
echo "Running ETL pipeline for locations data..."
cd src/etl
python dataflow_pipeline.py \
  --project-id=$PROJECT_ID \
  --dataset-id=cqc_data \
  --temp-location=gs://$PROJECT_ID-cqc-dataflow-temp/temp \
  --input-path="gs://$PROJECT_ID-cqc-raw-data/raw/locations/*.json" \
  --data-type=locations \
  --runner=DataflowRunner \
  --region=$REGION \
  --job-name=cqc-locations-etl-$(date +%Y%m%d-%H%M%S) \
  --service-account-email=cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --max-num-workers=10

# Run ETL for providers  
echo ""
echo "Running ETL pipeline for providers data..."
python dataflow_pipeline.py \
  --project-id=$PROJECT_ID \
  --dataset-id=cqc_data \
  --temp-location=gs://$PROJECT_ID-cqc-dataflow-temp/temp \
  --input-path="gs://$PROJECT_ID-cqc-raw-data/raw/providers/*.json" \
  --data-type=providers \
  --runner=DataflowRunner \
  --region=$REGION \
  --job-name=cqc-providers-etl-$(date +%Y%m%d-%H%M%S) \
  --service-account-email=cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --max-num-workers=10

cd ../..

echo ""
echo "======================================"
echo "ETL pipelines submitted!"
echo ""
echo "Monitor progress in the Dataflow console:"
echo "https://console.cloud.google.com/dataflow/jobs?project=$PROJECT_ID"
echo ""
echo "To check BigQuery data after completion:"
echo "  bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM \`$PROJECT_ID.cqc_data.locations\`'"
echo "  bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM \`$PROJECT_ID.cqc_data.providers\`'"
echo "======================================