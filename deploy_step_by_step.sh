#!/bin/bash

# Step-by-step deployment script for CQC ML Pipeline
# Each step runs on Google Cloud Platform

set -e

PROJECT_ID=${GCP_PROJECT:-"machine-learning-exp-467008"}
REGION="europe-west2"

echo "=================================="
echo "CQC ML PIPELINE - STEP BY STEP DEPLOYMENT"
echo "=================================="
echo "Project: $PROJECT_ID"
echo "Region: $REGION"
echo ""

# Step 1: Enable APIs (runs on GCP)
echo "Step 1: Enabling APIs on GCP..."
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    cloudfunctions.googleapis.com \
    storage.googleapis.com \
    bigquery.googleapis.com \
    secretmanager.googleapis.com \
    cloudscheduler.googleapis.com \
    --project=$PROJECT_ID

echo "✅ APIs enabled"

# Step 2: Create service accounts (runs on GCP)
echo ""
echo "Step 2: Creating service accounts..."
gcloud iam service-accounts create cqc-fetcher \
    --display-name="CQC Data Fetcher" \
    --project=$PROJECT_ID 2>/dev/null || echo "Service account cqc-fetcher already exists"

gcloud iam service-accounts create cqc-api \
    --display-name="CQC Prediction API" \
    --project=$PROJECT_ID 2>/dev/null || echo "Service account cqc-api already exists"

# Grant permissions
echo "Granting permissions..."
for role in storage.admin bigquery.dataEditor secretmanager.secretAccessor; do
    gcloud projects add-iam-policy-binding $PROJECT_ID \
        --member="serviceAccount:cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com" \
        --role="roles/${role}" \
        --condition=None 2>/dev/null || true
done

echo "✅ Service accounts configured"

# Step 3: Create buckets (runs on GCP)
echo ""
echo "Step 3: Creating Cloud Storage buckets..."
gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-cqc-raw-data 2>/dev/null || echo "Bucket ${PROJECT_ID}-cqc-raw-data exists"
gsutil mb -p $PROJECT_ID -l $REGION gs://${PROJECT_ID}-cqc-processed 2>/dev/null || echo "Bucket ${PROJECT_ID}-cqc-processed exists"

echo "✅ Storage buckets ready"

# Step 4: Setup BigQuery using bq command (runs on GCP)
echo ""
echo "Step 4: Setting up BigQuery dataset and tables..."

# Create dataset
bq mk -d \
    --location=$REGION \
    --description="CQC ML Pipeline Dataset" \
    ${PROJECT_ID}:cqc_dataset 2>/dev/null || echo "Dataset cqc_dataset already exists"

# Create locations_complete table
bq mk -t \
    --location=$REGION \
    --description="Complete CQC locations data" \
    --time_partitioning_field=processing_date \
    --clustering_fields=region,registrationStatus,type \
    ${PROJECT_ID}:cqc_dataset.locations_complete \
    locationId:STRING,locationName:STRING,providerId:STRING,providerName:STRING,type:STRING,registrationStatus:STRING,numberOfBeds:INTEGER,postalCode:STRING,region:STRING,localAuthority:STRING,overall_rating:STRING,processing_date:DATE,ingestion_timestamp:TIMESTAMP 2>/dev/null || echo "Table locations_complete already exists"

# Create care_homes table
bq mk -t \
    --location=$REGION \
    --description="Care homes specific data" \
    --time_partitioning_field=processing_date \
    --clustering_fields=region,care_home_type,overall_rating \
    ${PROJECT_ID}:cqc_dataset.care_homes \
    locationId:STRING,locationName:STRING,care_home_type:STRING,numberOfBeds:INTEGER,region:STRING,overall_rating:STRING,has_nursing:BOOLEAN,dementia_care:BOOLEAN,processing_date:DATE 2>/dev/null || echo "Table care_homes already exists"

# Create ml_features table
bq mk -t \
    --location=$REGION \
    --description="ML features for training" \
    ${PROJECT_ID}:cqc_dataset.ml_features \
    locationId:STRING,feature_version:STRING,feature_date:DATE,target:STRING,created_timestamp:TIMESTAMP 2>/dev/null || echo "Table ml_features already exists"

# Create predictions table
bq mk -t \
    --location=$REGION \
    --description="Model predictions" \
    --time_partitioning_field=prediction_date \
    ${PROJECT_ID}:cqc_dataset.predictions \
    prediction_id:STRING,locationId:STRING,predicted_overall_rating:STRING,confidence:FLOAT64,prediction_date:DATE 2>/dev/null || echo "Table predictions already exists"

echo "✅ BigQuery dataset and tables created"

# Step 5: Deploy Cloud Run job for data fetching
echo ""
echo "Step 5: Building and deploying data fetcher to Cloud Run..."

# First check if we have the required files
if [ -f "src/ingestion/cqc_fetcher_complete.py" ]; then
    # Submit build to Cloud Build
    cat > /tmp/cloudbuild-fetcher-simple.yaml <<EOF
steps:
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/cqc-fetcher-simple', '-f', '-', '.']
    dir: 'src/ingestion'
    args:
      - 'build'
      - '-t'
      - 'gcr.io/$PROJECT_ID/cqc-fetcher-simple'
      - '-f'
      - '-'
      - '.'
    stdin: |
      FROM python:3.11-slim
      WORKDIR /app
      RUN pip install google-cloud-storage google-cloud-bigquery google-cloud-secret-manager requests pandas numpy
      COPY cqc_fetcher_complete.py .
      CMD ["python", "cqc_fetcher_complete.py"]

  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/cqc-fetcher-simple']

  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud run jobs create cqc-fetcher-simple \
          --image=gcr.io/$PROJECT_ID/cqc-fetcher-simple \
          --region=$REGION \
          --memory=2Gi \
          --task-timeout=3600 \
          --set-env-vars=GCP_PROJECT=$PROJECT_ID \
          --service-account=cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com \
          || gcloud run jobs update cqc-fetcher-simple \
          --image=gcr.io/$PROJECT_ID/cqc-fetcher-simple \
          --region=$REGION
EOF

    gcloud builds submit --config=/tmp/cloudbuild-fetcher-simple.yaml --project=$PROJECT_ID
    echo "✅ Data fetcher deployed to Cloud Run"
else
    echo "⚠️  Data fetcher source not found, skipping..."
fi

# Step 6: Deploy prediction API
echo ""
echo "Step 6: Deploying prediction API to Cloud Functions..."

if [ -f "src/api/prediction_api.py" ]; then
    gcloud functions deploy predict-rating \
        --gen2 \
        --runtime=python311 \
        --region=$REGION \
        --source=src/api \
        --entry-point=predict_rating \
        --trigger-http \
        --allow-unauthenticated \
        --memory=512MB \
        --timeout=60s \
        --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCP_REGION=$REGION" \
        --service-account=cqc-api@$PROJECT_ID.iam.gserviceaccount.com \
        --project=$PROJECT_ID
    
    echo "✅ Prediction API deployed"
    
    # Get the URL
    API_URL=$(gcloud functions describe predict-rating --region=$REGION --format="value(serviceConfig.uri)" 2>/dev/null)
    if [ -n "$API_URL" ]; then
        echo "API URL: $API_URL"
    fi
else
    echo "⚠️  API source not found, skipping..."
fi

# Step 7: Setup Cloud Scheduler
echo ""
echo "Step 7: Setting up Cloud Scheduler for daily runs..."

gcloud scheduler jobs create http cqc-daily-run \
    --location=$REGION \
    --schedule="0 2 * * *" \
    --time-zone="Europe/London" \
    --uri="https://${REGION}-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/$PROJECT_ID/jobs/cqc-fetcher-simple:run" \
    --http-method=POST \
    --oauth-service-account-email=cqc-fetcher@$PROJECT_ID.iam.gserviceaccount.com \
    --project=$PROJECT_ID 2>/dev/null || echo "Scheduler job already exists"

echo "✅ Cloud Scheduler configured"

# Summary
echo ""
echo "=================================="
echo "DEPLOYMENT SUMMARY"
echo "=================================="
echo ""
echo "✅ Components Deployed:"
echo "   • BigQuery Dataset: ${PROJECT_ID}.cqc_dataset"
echo "   • Storage Buckets: ${PROJECT_ID}-cqc-raw-data, ${PROJECT_ID}-cqc-processed"
echo "   • Cloud Run Job: cqc-fetcher-simple"
echo "   • Cloud Scheduler: cqc-daily-run (2 AM daily)"
if [ -n "$API_URL" ]; then
    echo "   • Prediction API: $API_URL"
fi
echo ""
echo "📊 Next Steps:"
echo "   1. Run initial data fetch:"
echo "      gcloud run jobs execute cqc-fetcher-simple --region=$REGION"
echo ""
echo "   2. Check data in BigQuery:"
echo "      bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM \`${PROJECT_ID}.cqc_dataset.locations_complete\`'"
echo ""
echo "   3. Monitor progress:"
echo "      gcloud run jobs executions list --job=cqc-fetcher-simple --region=$REGION"
echo ""
echo "✅ Deployment complete!"