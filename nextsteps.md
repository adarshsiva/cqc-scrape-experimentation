# Next Steps - CQC Rating Predictor Deployment

## Current Status ✅

### Completed:
1. **Infrastructure Setup** ✅
   - Enabled all required GCP APIs
   - Created service accounts with proper permissions
   - Created storage buckets (raw data, ML artifacts, temp)
   - Stored CQC API subscription key in Secret Manager

2. **BigQuery Setup** ✅
   - Created `cqc_data` dataset
   - Created tables: locations, providers, predictions, ml_features
   - Set up partitioning and clustering

3. **Ingestion Service** ✅
   - Successfully deployed as Cloud Run service
   - URL: https://cqc-data-ingestion-744974744548.europe-west2.run.app
   - Note: Requires authenticated access due to org policies
   - **BLOCKED**: CQC API returns 403 Forbidden - subscription key needs activation

## Immediate Next Steps 🚀

### 1. Resolve CQC API Access (Priority: HIGH)
**Status**: BLOCKED - API returns 403 Forbidden
**Action Required**: 
- Contact CQC to activate subscription key: `45bdb9898457429783644ff69da1b9c9`
- Check if additional registration/approval is needed
- Verify if IP whitelisting is required
- Test endpoint configured with correct header: `Ocp-Apim-Subscription-Key`

### 2. Deploy Prediction Service (10 minutes)
**Status**: Ready to deploy
**Actions**:
1. Create Flask wrapper `app.py` for Cloud Run compatibility
2. Update requirements.txt to include Flask and gunicorn
3. Create Dockerfile for containerization
4. Deploy with:
```bash
cd src/prediction
gcloud run deploy cqc-rating-prediction \
  --source . \
  --platform managed \
  --region europe-west2 \
  --memory 2Gi \
  --timeout 60s \
  --service-account=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=machine-learning-exp-467008,VERTEX_ENDPOINT_ID=placeholder"
```

### 3. Set Up Cloud Scheduler (5 minutes)
**Status**: Ready once API access is resolved
**Note**: Postpone until CQC API access is working
```bash
# Create App Engine app (required for Cloud Scheduler)
gcloud app create --region=europe-west

# Create weekly schedule for data ingestion
gcloud scheduler jobs create http cqc-weekly-ingestion \
  --location=europe-west2 \
  --schedule="0 2 * * 1" \
  --time-zone="Europe/London" \
  --uri="https://cqc-data-ingestion-744974744548.europe-west2.run.app" \
  --http-method=POST \
  --oidc-service-account-email=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com
```

### 4. Generate Sample Data for Testing (15 minutes)
**Status**: Required due to API block
**Purpose**: Create synthetic CQC data to continue development/testing
**Actions**:
1. Create sample data generator script based on CQC schema
2. Generate test datasets (100-1000 records)
3. Upload to GCS buckets
4. Use for ETL pipeline and ML model development

### 5. Run ETL Pipeline (20-30 minutes)
**Status**: Ready once sample data is available
**Note**: Can proceed with sample data while waiting for API access
```bash
# Install dependencies
pip install -r src/etl/requirements.txt

# Run Dataflow pipeline for locations (with sample data)
cd src/etl
python dataflow_pipeline.py \
  --project-id=machine-learning-exp-467008 \
  --dataset-id=cqc_data \
  --temp-location=gs://machine-learning-exp-467008-cqc-dataflow-temp/temp \
  --input-path="gs://machine-learning-exp-467008-cqc-raw-data/raw/sample_locations/*.json" \
  --data-type=locations \
  --runner=DataflowRunner \
  --region=europe-west2

# Run for providers (with sample data)
python dataflow_pipeline.py \
  --project-id=machine-learning-exp-467008 \
  --dataset-id=cqc_data \
  --temp-location=gs://machine-learning-exp-467008-cqc-dataflow-temp/temp \
  --input-path="gs://machine-learning-exp-467008-cqc-raw-data/raw/sample_providers/*.json" \
  --data-type=providers \
  --runner=DataflowRunner \
  --region=europe-west2
```

### 6. Train ML Models (45-60 minutes)
**Status**: Ready once ETL pipeline populates BigQuery
**Note**: Can use sample data for initial model development
```bash
# Install ML dependencies
pip install -r src/ml/requirements.txt

# Run Vertex AI pipeline
cd src/ml/pipeline
python pipeline.py \
  --project-id=machine-learning-exp-467008 \
  --pipeline-root=gs://machine-learning-exp-467008-cqc-ml-artifacts/pipelines \
  --display-name="cqc-ml-pipeline-$(date +%Y%m%d-%H%M%S)" \
  --service-account=cqc-vertex-service-account@machine-learning-exp-467008.iam.gserviceaccount.com \
  --region=europe-west2 \
  --use-sample-data  # Flag to use sample data
```

### 7. Update Prediction Service with Model Endpoint (5 minutes)
**Status**: Ready after model training
```bash
# Get endpoint ID from Vertex AI
ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west2 --format="value(name)" | head -1 | cut -d'/' -f6)

# Update prediction service
gcloud run services update cqc-rating-prediction \
  --region=europe-west2 \
  --update-env-vars="VERTEX_ENDPOINT_ID=$ENDPOINT_ID"
```

### 8. End-to-End Testing (10 minutes)
**Status**: Ready after all components deployed
```bash
# Test prediction API
PREDICTION_URL=$(gcloud run services describe cqc-rating-prediction --region=europe-west2 --format="value(status.url)")

curl -X POST $PREDICTION_URL \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d @test-prediction.json
```

## Updated Timeline 📅

### Phase 1: Workaround Development (1-2 hours)
1. Deploy Prediction Service ✓
2. Generate Sample Data
3. Run ETL Pipeline with sample data
4. Train initial ML model

### Phase 2: API Resolution (Parallel)
1. Contact CQC for API activation
2. Test API access once activated
3. Set up Cloud Scheduler

### Phase 3: Production Deployment (2-3 hours)
1. Run full data ingestion
2. Process real data through ETL
3. Retrain models with real data
4. End-to-end testing

## Important Notes ⚠️

1. **CQC API Issue**: 
   - Subscription key: `45bdb9898457429783644ff69da1b9c9`
   - Returns 403 Forbidden - needs activation
   - Using correct header: `Ocp-Apim-Subscription-Key`

2. **Authentication**: All services require authentication:
   ```bash
   -H "Authorization: Bearer $(gcloud auth print-identity-token)"
   ```

3. **Monitoring**: Check logs regularly:
   - Cloud Run: `gcloud run services logs read SERVICE_NAME --region=europe-west2`
   - Dataflow: Console → Dataflow → Jobs
   - Vertex AI: Console → Vertex AI → Pipelines

4. **Cost Optimization**:
   - Use preemptible nodes for training
   - Monitor Dataflow and Vertex AI costs
   - Consider using DirectRunner for testing

## Troubleshooting 🔧

### Current Issues:
1. **CQC API 403 Error**: Contact CQC support for key activation
2. **Cloud Run Deployment**: Services need Flask wrapper instead of functions_framework

### Common Issues:
1. **Permission errors**: Check service account IAM roles
2. **No data in BigQuery**: Verify ingestion succeeded and ETL completed
3. **Model training fails**: Check BigQuery data quality and Vertex AI quotas
4. **Prediction errors**: Ensure model endpoint is deployed and accessible

## Total Estimated Time: 
- With sample data workaround: ~3-4 hours
- Waiting for API activation: Unknown (depends on CQC response time)

Once all steps are completed, the system will be fully operational with:
- Weekly automated data ingestion
- ETL processing pipeline
- Trained ML models
- Real-time prediction API