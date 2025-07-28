# Next Steps - CQC Rating Predictor Complete Deployment Plan

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

3. **Cloud Functions Deployed** ✅
   - **Sample Data Generator**: https://europe-west2-machine-learning-exp-467008.cloudfunctions.net/generate-sample-data
   - **Data Converter (JSON to NDJSON)**: https://europe-west2-machine-learning-exp-467008.cloudfunctions.net/convert-json-to-ndjson
   - **Ingestion Service**: https://cqc-data-ingestion-744974744548.europe-west2.run.app

4. **Prediction Service** ✅
   - Deployed to Cloud Run: https://cqc-rating-prediction-744974744548.europe-west2.run.app
   - Ready for ML model endpoint integration

5. **Sample Data Generated** ✅
   - 2000 locations and 200 providers in GCS
   - Files: raw/locations/20250728_191714_locations_sample.json

6. **GitHub Repository** ✅
   - Code pushed to: https://github.com/adarshsiva/cqc-scrape-experimentation

## Detailed Next Steps 🚀

### Phase 1: Complete Data Pipeline (1-2 hours)

#### 1. Convert Sample Data to NDJSON (5 minutes)
```bash
# Call the data converter function
TOKEN=$(gcloud auth print-identity-token)
curl -X POST https://europe-west2-machine-learning-exp-467008.cloudfunctions.net/convert-json-to-ndjson \
  -H "Authorization: Bearer ${TOKEN}" \
  -H "Content-Type: application/json" \
  -d '{
    "source_bucket": "machine-learning-exp-467008-cqc-raw-data",
    "source_prefix": "raw/",
    "dest_prefix": "processed/"
  }'
```

#### 2. Load Data into BigQuery (10 minutes)
```bash
# Load locations data
bq load \
  --source_format=NEWLINE_DELIMITED_JSON \
  --autodetect \
  --location=europe-west2 \
  machine-learning-exp-467008:cqc_data.locations_raw \
  gs://machine-learning-exp-467008-cqc-raw-data/processed/locations/*.ndjson

# Load providers data
bq load \
  --source_format=NEWLINE_DELIMITED_JSON \
  --autodetect \
  --location=europe-west2 \
  machine-learning-exp-467008:cqc_data.providers_raw \
  gs://machine-learning-exp-467008-cqc-raw-data/processed/providers/*.ndjson
```

#### 3. Create Feature Engineering Views (15 minutes)
```sql
-- Create ML features view in BigQuery
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_v1` AS
SELECT
  locationId,
  -- Basic features
  IFNULL(numberOfBeds, 0) as number_of_beds,
  1 as number_of_locations, -- Will be updated with JOIN to providers
  ARRAY_LENGTH(SPLIT(inspectionHistory, ',')) as inspection_history_length,
  DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_last_inspection,
  
  -- Categorical features
  IFNULL(ownershipType, 'Unknown') as ownership_type,
  ARRAY_LENGTH(serviceTypes) as service_types_count,
  ARRAY_LENGTH(specialisms) as specialisms_count,
  ARRAY_LENGTH(regulatedActivities) as regulated_activities_count,
  ARRAY_LENGTH(serviceUserBands) as service_user_groups_count,
  
  -- Location features
  region,
  localAuthority as local_authority,
  constituency,
  
  -- Rating features
  CASE WHEN currentRatings.overall.rating IS NOT NULL THEN TRUE ELSE FALSE END as has_previous_rating,
  IFNULL(currentRatings.overall.rating, 'No rating') as previous_rating,
  
  -- Additional features
  FALSE as ownership_changed_recently, -- Placeholder
  TRUE as nominated_individual_exists, -- Placeholder
  
  -- Target variable
  currentRatings.overall.rating as rating_label
FROM `machine-learning-exp-467008.cqc_data.locations_raw`
WHERE registrationStatus = 'Registered';
```

### Phase 2: Train ML Models (1-2 hours)

#### 1. Create Training Dataset (10 minutes)
```bash
# Export training data from BigQuery
bq extract \
  --destination_format=CSV \
  --field_delimiter=',' \
  machine-learning-exp-467008:cqc_data.ml_features_v1 \
  gs://machine-learning-exp-467008-cqc-ml-artifacts/training_data/features_*.csv
```

#### 2. Deploy ML Training Pipeline (45 minutes)
```bash
# Create Vertex AI training job
gcloud ai custom-jobs create \
  --region=europe-west2 \
  --display-name="cqc-rating-predictor-training" \
  --config=- <<EOF
{
  "jobSpec": {
    "workerPoolSpecs": [{
      "machineSpec": {
        "machineType": "n1-standard-4"
      },
      "replicaCount": 1,
      "pythonPackageSpec": {
        "executorImageUri": "europe-docker.pkg.dev/vertex-ai/training/scikit-learn-cpu.1-0:latest",
        "packageUris": ["gs://machine-learning-exp-467008-cqc-ml-artifacts/packages/trainer-0.1.tar.gz"],
        "pythonModule": "trainer.task",
        "args": [
          "--project-id=machine-learning-exp-467008",
          "--input-path=gs://machine-learning-exp-467008-cqc-ml-artifacts/training_data/",
          "--output-path=gs://machine-learning-exp-467008-cqc-ml-artifacts/models/",
          "--model-type=xgboost"
        ]
      }
    }]
  }
}
EOF
```

#### 3. Create and Deploy Model Endpoint (20 minutes)
```bash
# Create model
gcloud ai models upload \
  --region=europe-west2 \
  --display-name=cqc-rating-predictor \
  --container-image-uri=europe-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest \
  --artifact-uri=gs://machine-learning-exp-467008-cqc-ml-artifacts/models/model.pkl

# Create endpoint
gcloud ai endpoints create \
  --region=europe-west2 \
  --display-name=cqc-rating-predictor-endpoint

# Deploy model to endpoint
MODEL_ID=$(gcloud ai models list --region=europe-west2 --filter="displayName:cqc-rating-predictor" --format="value(name)" | head -1)
ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west2 --filter="displayName:cqc-rating-predictor-endpoint" --format="value(name)" | head -1)

gcloud ai endpoints deploy-model $ENDPOINT_ID \
  --region=europe-west2 \
  --model=$MODEL_ID \
  --display-name=cqc-rating-predictor-v1 \
  --machine-type=n1-standard-2 \
  --min-replica-count=1 \
  --max-replica-count=3
```

### Phase 3: CQC API Integration (When API Access is Granted)

#### 1. Test CQC API Access
```bash
# Test API with subscription key
curl -X GET "https://api.cqc.org.uk/public/v1/providers?page=1&perPage=10" \
  -H "Ocp-Apim-Subscription-Key: 45bdb9898457429783644ff69da1b9c9"
```

#### 2. Update Ingestion Service for Production
```bash
# Deploy production ingestion with proper error handling
cd src/ingestion
gcloud run deploy cqc-data-ingestion \
  --source . \
  --platform managed \
  --region europe-west2 \
  --memory 2Gi \
  --timeout 600s \
  --service-account=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com \
  --set-env-vars="GCP_PROJECT=machine-learning-exp-467008,GCS_BUCKET=machine-learning-exp-467008-cqc-raw-data" \
  --no-allow-unauthenticated
```

#### 3. Set Up Cloud Scheduler for Automated Ingestion
```bash
# Create App Engine app (required for Cloud Scheduler)
gcloud app create --region=europe-west

# Create daily ingestion schedule
gcloud scheduler jobs create http cqc-daily-ingestion \
  --location=europe-west2 \
  --schedule="0 2 * * *" \
  --time-zone="Europe/London" \
  --uri="https://cqc-data-ingestion-744974744548.europe-west2.run.app/ingest" \
  --http-method=POST \
  --oidc-service-account-email=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com \
  --body='{"full_sync": true}'

# Create weekly full sync
gcloud scheduler jobs create http cqc-weekly-full-sync \
  --location=europe-west2 \
  --schedule="0 3 * * 0" \
  --time-zone="Europe/London" \
  --uri="https://cqc-data-ingestion-744974744548.europe-west2.run.app/ingest" \
  --http-method=POST \
  --oidc-service-account-email=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com \
  --body='{"full_sync": true, "force_update": true}'
```

### Phase 4: Production Pipeline Automation

#### 1. Create Dataflow Flex Template (30 minutes)
```bash
# Build Dataflow template
cd src/etl
gcloud dataflow flex-template build \
  gs://machine-learning-exp-467008-cqc-dataflow-temp/templates/etl-pipeline.json \
  --image-gcr-path="gcr.io/machine-learning-exp-467008/cqc-etl-pipeline:latest" \
  --sdk-language="PYTHON" \
  --flex-template-base-image="PYTHON3" \
  --metadata-file="metadata.json" \
  --py-path="." \
  --env="FLEX_TEMPLATE_PYTHON_PY_FILE=dataflow_pipeline.py" \
  --env="FLEX_TEMPLATE_PYTHON_REQUIREMENTS_FILE=requirements.txt"
```

#### 2. Create Cloud Composer DAG for Orchestration (45 minutes)
```python
# Create DAG file: dags/cqc_ml_pipeline.py
from airflow import DAG
from airflow.providers.google.cloud.operators.dataflow import DataflowStartFlexTemplateOperator
from airflow.providers.google.cloud.operators.vertex_ai.custom_job import CreateCustomPythonPackageTrainingJobOperator
from airflow.providers.google.cloud.operators.vertex_ai.endpoint_service import DeployModelOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'cqc-ml-team',
    'depends_on_past': False,
    'email': ['alerts@example.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'cqc_ml_pipeline',
    default_args=default_args,
    description='CQC ML Pipeline',
    schedule_interval='@weekly',
    start_date=datetime(2024, 1, 1),
    catchup=False,
)

# ETL Task
run_etl = DataflowStartFlexTemplateOperator(
    task_id='run_etl_pipeline',
    template='gs://machine-learning-exp-467008-cqc-dataflow-temp/templates/etl-pipeline.json',
    parameters={
        'input_path': 'gs://machine-learning-exp-467008-cqc-raw-data/raw/latest/*',
        'output_dataset': 'cqc_data',
    },
    dag=dag,
)

# Training Task
train_model = CreateCustomPythonPackageTrainingJobOperator(
    task_id='train_ml_model',
    # ... training config
    dag=dag,
)

# Deployment Task
deploy_model = DeployModelOperator(
    task_id='deploy_model',
    # ... deployment config
    dag=dag,
)

run_etl >> train_model >> deploy_model
```

### Phase 5: Monitoring and Alerting Setup (30 minutes)

#### 1. Create Monitoring Dashboard
```bash
# Create custom dashboard
gcloud monitoring dashboards create --config-from-file=- <<EOF
{
  "displayName": "CQC ML Pipeline Dashboard",
  "widgets": [
    {
      "title": "API Ingestion Success Rate",
      "xyChart": {
        "dataSets": [{
          "timeSeriesQuery": {
            "timeSeriesFilter": {
              "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_count\""
            }
          }
        }]
      }
    },
    {
      "title": "Prediction Latency",
      "xyChart": {
        "dataSets": [{
          "timeSeriesQuery": {
            "timeSeriesFilter": {
              "filter": "resource.type=\"cloud_run_revision\" AND metric.type=\"run.googleapis.com/request_latencies\""
            }
          }
        }]
      }
    }
  ]
}
EOF
```

#### 2. Set Up Alerts
```bash
# Create alert for ingestion failures
gcloud alpha monitoring policies create \
  --notification-channels=CHANNEL_ID \
  --display-name="CQC Ingestion Failure Alert" \
  --condition-display-name="Ingestion Error Rate > 10%" \
  --condition-filter='resource.type="cloud_run_revision" AND metric.type="logging.googleapis.com/user/error_count"' \
  --condition-threshold-value=0.1 \
  --condition-threshold-duration=300s
```

### Phase 6: Testing and Validation

#### 1. Integration Tests
```bash
# Test prediction endpoint
ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west2 --format="value(name)" | grep cqc-rating-predictor)
PREDICTION_URL=$(gcloud run services describe cqc-rating-prediction --region=europe-west2 --format="value(status.url)")

# Update prediction service with endpoint
gcloud run services update cqc-rating-prediction \
  --region=europe-west2 \
  --update-env-vars="VERTEX_ENDPOINT_ID=${ENDPOINT_ID##*/}"

# Test prediction
curl -X POST $PREDICTION_URL \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d '{
    "number_of_beds": 50,
    "number_of_locations": 1,
    "inspection_history_length": 3,
    "days_since_last_inspection": 180,
    "ownership_type": "Organisation",
    "service_types": ["Care home service with nursing"],
    "specialisms": ["Dementia"],
    "region": "London",
    "local_authority": "LA-123",
    "constituency": "Constituency-456",
    "regulated_activities": ["Accommodation for persons who require nursing or personal care"],
    "service_user_groups": ["Older people"],
    "has_previous_rating": true,
    "previous_rating": "Good",
    "ownership_changed_recently": false,
    "nominated_individual_exists": true
  }'
```

#### 2. Load Testing
```bash
# Install hey for load testing
go install github.com/rakyll/hey@latest

# Run load test
hey -n 1000 -c 10 -m POST \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)" \
  -H "Content-Type: application/json" \
  -d @test-prediction.json \
  $PREDICTION_URL
```

## Timeline Summary 📅

### Immediate (Can do now with sample data):
1. **Hour 1**: Data pipeline completion (convert to NDJSON, load to BigQuery, create views)
2. **Hour 2-3**: ML model training and deployment
3. **Hour 3-4**: Integration testing and monitoring setup

### Once CQC API Access is Granted:
1. **Day 1**: 
   - Test API access
   - Update ingestion service
   - Set up schedulers
   - Run initial full data sync
2. **Day 2**:
   - Process real data through ETL
   - Retrain models with real data
   - Performance testing
3. **Day 3**:
   - Deploy Composer DAGs
   - Set up monitoring
   - Documentation updates

### Ongoing:
- Daily incremental data ingestion
- Weekly full data sync
- Monthly model retraining
- Continuous monitoring and optimization

## Success Criteria ✅

1. **Data Pipeline**:
   - [ ] Daily automated ingestion from CQC API
   - [ ] ETL processing < 30 minutes for full dataset
   - [ ] Data quality checks passing

2. **ML Models**:
   - [ ] Model accuracy > 80%
   - [ ] Prediction latency < 500ms
   - [ ] Auto-retraining pipeline operational

3. **Operations**:
   - [ ] 99.9% uptime for prediction service
   - [ ] Automated alerts for failures
   - [ ] Cost optimization implemented

4. **Documentation**:
   - [ ] API documentation complete
   - [ ] Runbooks for common issues
   - [ ] Architecture diagrams updated

## Repository Structure 📁
```
cqc-scrape-experimentation/
├── src/
│   ├── ingestion/         # CQC API data ingestion
│   ├── etl/              # Dataflow ETL pipelines
│   ├── ml/               # ML training pipelines
│   ├── prediction/       # Prediction service
│   ├── sample_data_generator/  # Test data generation
│   └── data_converter/   # JSON to NDJSON converter
├── terraform/            # Infrastructure as code
├── config/              # Configuration files
├── documentation/       # API docs and guides
└── scripts/            # Deployment scripts
```

## Contact for Issues 📧
- **CQC API Support**: For subscription key activation
- **GCP Support**: For quota increases or technical issues
- **Repository**: https://github.com/adarshsiva/cqc-scrape-experimentation

Remember to commit and push changes after each major milestone!