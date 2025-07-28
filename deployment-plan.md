# CQC Rating Predictor - Complete Deployment Plan

## Project Details
- **GCP Project ID**: machine-learning-exp-467008
- **Active Account**: hello@ourtimehq.com
- **Region**: europe-west2 (London)
- **Service Name**: CQC Rating Predictor ML System

## Pre-Deployment Requirements

### 1. CQC API Credentials Required
Please provide the following CQC API credentials:
- **Subscription Key**: Your CQC API subscription key
- **Partner Code**: Your CQC partner code

These credentials are required to access the CQC API endpoints:
- Base URL: https://api.cqc.org.uk/public/v1/
- Endpoints: /providers and /locations
- Rate limit: 2000 requests/minute with partner code

### 2. Google Cloud Authentication
```bash
# Re-authenticate if needed
gcloud auth login
gcloud auth application-default login

# Verify project is set correctly
gcloud config set project machine-learning-exp-467008
```

## Deployment Steps

### Phase 1: Enable Required APIs (5 minutes)
```bash
# Enable all required Google Cloud APIs
gcloud services enable \
  cloudfunctions.googleapis.com \
  cloudbuild.googleapis.com \
  cloudscheduler.googleapis.com \
  storage-component.googleapis.com \
  bigquery.googleapis.com \
  dataflow.googleapis.com \
  composer.googleapis.com \
  aiplatform.googleapis.com \
  secretmanager.googleapis.com \
  cloudresourcemanager.googleapis.com \
  logging.googleapis.com \
  monitoring.googleapis.com \
  compute.googleapis.com \
  appengine.googleapis.com
```

### Phase 2: Create Service Accounts (5 minutes)
```bash
# Create service accounts
gcloud iam service-accounts create cqc-cf-service-account \
  --display-name="CQC Cloud Functions Service Account"

gcloud iam service-accounts create cqc-dataflow-service-account \
  --display-name="CQC Dataflow Service Account"

gcloud iam service-accounts create cqc-vertex-service-account \
  --display-name="CQC Vertex AI Service Account"

# Assign IAM roles
PROJECT_ID=machine-learning-exp-467008

# Cloud Functions service account permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# Dataflow service account permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/dataflow.worker"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.objectAdmin"

# Vertex AI service account permissions
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/aiplatform.user"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataViewer"

gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/storage.admin"
```

### Phase 3: Create Storage Buckets (3 minutes)
```bash
# Create required storage buckets
gsutil mb -p $PROJECT_ID -c STANDARD -l europe-west2 gs://$PROJECT_ID-cqc-raw-data
gsutil mb -p $PROJECT_ID -c STANDARD -l europe-west2 gs://$PROJECT_ID-cqc-dataflow-temp
gsutil mb -p $PROJECT_ID -c STANDARD -l europe-west2 gs://$PROJECT_ID-cqc-ml-artifacts
gsutil mb -p $PROJECT_ID -c STANDARD -l europe-west2 gs://$PROJECT_ID-cqc-cf-source

# Set lifecycle policies
gsutil lifecycle set config/gcs-lifecycle-raw-data.json gs://$PROJECT_ID-cqc-raw-data
gsutil lifecycle set config/gcs-lifecycle-temp.json gs://$PROJECT_ID-cqc-dataflow-temp
```

### Phase 4: Store CQC API Credentials (2 minutes)
```bash
# After you provide the credentials, we'll store them securely:
echo -n "YOUR_SUBSCRIPTION_KEY" | gcloud secrets create cqc-subscription-key --data-file=-
echo -n "YOUR_PARTNER_CODE" | gcloud secrets create cqc-partner-code --data-file=-

# Grant access to service accounts
gcloud secrets add-iam-policy-binding cqc-subscription-key \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding cqc-partner-code \
  --member="serviceAccount:cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com" \
  --role="roles/secretmanager.secretAccessor"
```

### Phase 5: Create BigQuery Dataset and Tables (3 minutes)
```bash
# Create BigQuery dataset
bq mk --dataset \
  --location=europe-west2 \
  --description="CQC ratings data and predictions" \
  $PROJECT_ID:cqc_data

# Create tables using schemas
bq mk --table \
  --time_partitioning_field=ingestion_timestamp \
  --time_partitioning_type=DAY \
  --clustering_fields=region,overall_rating \
  $PROJECT_ID:cqc_data.locations \
  config/bigquery_schema_locations.json

bq mk --table \
  --time_partitioning_field=ingestion_timestamp \
  --time_partitioning_type=DAY \
  $PROJECT_ID:cqc_data.providers \
  config/bigquery_schema_providers.json

bq mk --table \
  --time_partitioning_field=prediction_timestamp \
  --time_partitioning_type=DAY \
  $PROJECT_ID:cqc_data.predictions \
  config/bigquery_schema_predictions.json

bq mk --table \
  $PROJECT_ID:cqc_data.ml_features \
  config/bigquery_schema_ml_features.json
```

### Phase 6: Deploy Cloud Functions (10 minutes)
```bash
# Package and deploy ingestion function
cd src/ingestion
zip -r ingestion.zip .
gsutil cp ingestion.zip gs://$PROJECT_ID-cqc-cf-source/

gcloud functions deploy cqc-data-ingestion \
  --gen2 \
  --runtime=python311 \
  --region=europe-west2 \
  --source=gs://$PROJECT_ID-cqc-cf-source/ingestion.zip \
  --entry-point=ingest_cqc_data \
  --memory=512MB \
  --timeout=540s \
  --service-account=cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID,GCS_BUCKET=$PROJECT_ID-cqc-raw-data" \
  --trigger-http \
  --allow-unauthenticated

# Package and deploy prediction function
cd ../prediction
zip -r prediction.zip .
gsutil cp prediction.zip gs://$PROJECT_ID-cqc-cf-source/

gcloud functions deploy cqc-rating-prediction \
  --gen2 \
  --runtime=python311 \
  --region=europe-west2 \
  --source=gs://$PROJECT_ID-cqc-cf-source/prediction.zip \
  --entry-point=predict_cqc_rating \
  --memory=1GB \
  --timeout=60s \
  --service-account=cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=$PROJECT_ID" \
  --trigger-http \
  --allow-unauthenticated
```

### Phase 7: Set Up Cloud Scheduler (3 minutes)
```bash
# Create App Engine app (required for Cloud Scheduler)
gcloud app create --region=europe-west2

# Create weekly ingestion schedule
gcloud scheduler jobs create http cqc-weekly-ingestion \
  --location=europe-west2 \
  --schedule="0 2 * * 1" \
  --time-zone="Europe/London" \
  --uri="https://europe-west2-$PROJECT_ID.cloudfunctions.net/cqc-data-ingestion" \
  --http-method=POST \
  --oidc-service-account-email=cqc-cf-service-account@$PROJECT_ID.iam.gserviceaccount.com
```

### Phase 8: Initial Data Ingestion (15 minutes)
```bash
# Trigger initial data ingestion
INGESTION_URL=$(gcloud functions describe cqc-data-ingestion --region=europe-west2 --format="value(url)")
curl -X POST $INGESTION_URL

# Monitor ingestion progress
gcloud functions logs read cqc-data-ingestion --region=europe-west2 --limit=50
```

### Phase 9: Run ETL Pipeline (20 minutes)
```bash
cd src/etl

# Process locations data
python dataflow_pipeline.py \
  --project-id=$PROJECT_ID \
  --dataset-id=cqc_data \
  --temp-location=gs://$PROJECT_ID-cqc-dataflow-temp/temp \
  --input-path=gs://$PROJECT_ID-cqc-raw-data/raw/locations/*.json \
  --data-type=locations \
  --runner=DataflowRunner \
  --region=europe-west2 \
  --service-account-email=cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com

# Process providers data
python dataflow_pipeline.py \
  --project-id=$PROJECT_ID \
  --dataset-id=cqc_data \
  --temp-location=gs://$PROJECT_ID-cqc-dataflow-temp/temp \
  --input-path=gs://$PROJECT_ID-cqc-raw-data/raw/providers/*.json \
  --data-type=providers \
  --runner=DataflowRunner \
  --region=europe-west2 \
  --service-account-email=cqc-dataflow-service-account@$PROJECT_ID.iam.gserviceaccount.com
```

### Phase 10: Train ML Models (45 minutes)
```bash
cd src/ml/pipeline

# Create and run ML pipeline
python pipeline.py \
  --project-id=$PROJECT_ID \
  --pipeline-root=gs://$PROJECT_ID-cqc-ml-artifacts/pipelines \
  --display-name="cqc-ml-pipeline-$(date +%Y%m%d-%H%M%S)" \
  --service-account=cqc-vertex-service-account@$PROJECT_ID.iam.gserviceaccount.com
```

### Phase 11: Update Prediction Function with Model Endpoint (5 minutes)
```bash
# Get the deployed model endpoint ID from Vertex AI console or CLI
ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west2 --format="value(name)" | head -1)

# Update prediction function with endpoint ID
gcloud functions deploy cqc-rating-prediction \
  --gen2 \
  --update-env-vars="VERTEX_ENDPOINT_ID=$ENDPOINT_ID"
```

### Phase 12: Test the Complete System (5 minutes)
```bash
# Test prediction API
PREDICTION_URL=$(gcloud functions describe cqc-rating-prediction --region=europe-west2 --format="value(url)")

curl -X POST $PREDICTION_URL \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "days_since_last_inspection": 180,
      "num_regulated_activities": 5,
      "region": "London",
      "type": "Residential social care",
      "postal_code": "SW1A 1AA",
      "registration_status": "Registered",
      "ownership_type": "Private",
      "has_specialisms": true,
      "num_service_types": 3,
      "days_since_registration": 1825,
      "local_authority": "Westminster",
      "inspection_directorate": "Adult social care",
      "constituency": "Cities of London and Westminster",
      "regulated_activities": ["Accommodation for persons who require nursing or personal care"],
      "service_types": ["Care home service with nursing"],
      "specialisms": ["Dementia"]
    }]
  }'
```

## Total Deployment Time: ~2 hours

## Post-Deployment Monitoring

1. **Check Cloud Functions logs**:
   ```bash
   gcloud functions logs read --region=europe-west2
   ```

2. **Monitor Dataflow jobs**:
   ```bash
   gcloud dataflow jobs list --region=europe-west2
   ```

3. **View BigQuery data**:
   ```bash
   bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `machine-learning-exp-467008.cqc_data.locations`'
   ```

4. **Check Vertex AI pipeline**:
   - Visit: https://console.cloud.google.com/vertex-ai/pipelines

## Required Information from User

Before proceeding with deployment, please provide:

1. **CQC API Subscription Key**: _________________
2. **CQC API Partner Code**: _________________

Once you provide these credentials, we can proceed with the deployment following the steps above.

## Cost Estimates

Monthly costs (approximate):
- Cloud Functions: ~$5-10
- Cloud Storage: ~$20-50 (depending on data volume)
- BigQuery: ~$10-30 (storage + queries)
- Dataflow: ~$50-100 (weekly runs)
- Vertex AI: ~$100-200 (training + serving)
- **Total**: ~$185-390/month

## Troubleshooting

Common issues and solutions:

1. **API Enable Errors**: Wait 2-3 minutes after enabling APIs
2. **Permission Errors**: Ensure service accounts have correct roles
3. **Quota Errors**: Check project quotas in Cloud Console
4. **Dataflow Errors**: Check worker logs and increase machine type if needed
5. **Vertex AI Errors**: Ensure BigQuery permissions are correct