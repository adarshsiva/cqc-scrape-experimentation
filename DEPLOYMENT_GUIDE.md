# CQC Rating Predictor - Complete Deployment Guide

## Prerequisites
- Google Cloud SDK installed and configured
- Python 3.11+ installed
- Active GCP account with billing enabled
- CQC API subscription key: `45bdb9898457429783644ff69da1b9c9` (already configured)

## Deployment Steps

### Step 1: Initial Setup (10 minutes)
Run the initial deployment script to set up GCP resources:

```bash
cd /Users/adarsh/Documents/Dev/CQC_scrape
./deploy.sh
```

This will:
- Enable required GCP APIs
- Create service accounts with proper permissions
- Create storage buckets
- Store CQC API credentials in Secret Manager
- Create BigQuery dataset and tables

### Step 2: Deploy Cloud Functions (10 minutes)
Deploy the ingestion and prediction Cloud Functions:

```bash
./deploy-functions.sh
```

This will deploy:
- `cqc-data-ingestion`: Fetches data from CQC API
- `cqc-rating-prediction`: Serves ML predictions

### Step 3: Initial Data Ingestion (15 minutes)
Trigger the first data ingestion from CQC API:

```bash
# Get the ingestion function URL
INGESTION_URL=$(gcloud functions describe cqc-data-ingestion --region=europe-west2 --format="value(url)")

# Trigger ingestion
curl -X POST $INGESTION_URL
```

Monitor progress:
```bash
gcloud functions logs read cqc-data-ingestion --region=europe-west2 --limit=50
```

### Step 4: Run ETL Pipeline (20-30 minutes)
Process the raw data with Dataflow:

```bash
./run-etl.sh
```

This will:
- Process locations data
- Process providers data
- Extract features and load to BigQuery

Monitor in Dataflow console:
https://console.cloud.google.com/dataflow/jobs?project=machine-learning-exp-467008

### Step 5: Train ML Models (45-60 minutes)
Train and deploy ML models with Vertex AI:

```bash
./train-models.sh
```

This will:
- Train XGBoost and LightGBM models
- Evaluate model performance
- Automatically deploy the best model

Monitor in Vertex AI console:
https://console.cloud.google.com/vertex-ai/pipelines/runs?project=machine-learning-exp-467008

### Step 6: Update Prediction Function (2 minutes)
Update the prediction function with the deployed model endpoint:

```bash
./update-endpoint.sh
```

### Step 7: Set Up Scheduled Ingestion (5 minutes)
Create weekly scheduled ingestion:

```bash
# Create App Engine app (required for Cloud Scheduler)
gcloud app create --region=europe-west

# Create scheduler job
gcloud scheduler jobs create http cqc-weekly-ingestion \
  --location=europe-west2 \
  --schedule="0 2 * * 1" \
  --time-zone="Europe/London" \
  --uri="$(gcloud functions describe cqc-data-ingestion --region=europe-west2 --format='value(url)')" \
  --http-method=POST \
  --oidc-service-account-email=cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com
```

### Step 8: Test the System (5 minutes)
Test the prediction API:

```bash
# Get prediction URL
PREDICTION_URL=$(gcloud functions describe cqc-rating-prediction --region=europe-west2 --format="value(url)")

# Test with sample data
curl -X POST $PREDICTION_URL \
  -H "Content-Type: application/json" \
  -d @test-prediction.json
```

## Verification Checklist

1. **APIs Enabled**: Check in Cloud Console → APIs & Services
2. **Service Accounts**: Check in IAM & Admin → Service Accounts
3. **Storage Buckets**: `gsutil ls`
4. **BigQuery Tables**: 
   ```bash
   bq ls machine-learning-exp-467008:cqc_data
   ```
5. **Cloud Functions**: 
   ```bash
   gcloud functions list --region=europe-west2
   ```
6. **Data in BigQuery**:
   ```bash
   bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `machine-learning-exp-467008.cqc_data.locations`'
   ```
7. **Model Endpoint**:
   ```bash
   gcloud ai endpoints list --region=europe-west2
   ```

## Troubleshooting

### Common Issues:

1. **API not enabled error**:
   ```bash
   gcloud services enable <service-name>.googleapis.com
   ```

2. **Permission denied**:
   - Check service account permissions
   - Ensure you're authenticated: `gcloud auth login`

3. **Dataflow job fails**:
   - Check worker logs in Dataflow console
   - Verify BigQuery permissions

4. **No data in BigQuery**:
   - Check ingestion function logs
   - Verify CQC API key is correct

5. **Model training fails**:
   - Check Vertex AI pipeline logs
   - Ensure sufficient data in BigQuery

## Cost Management

- Enable budget alerts in Billing console
- Review storage lifecycle policies
- Monitor Dataflow and Vertex AI usage

## Next Steps

1. Set up monitoring dashboards
2. Configure alerting for failures
3. Implement data quality checks
4. Add more features for better predictions
5. Set up A/B testing for model improvements

## Support

For issues, check:
- Cloud Logging: https://console.cloud.google.com/logs
- Error Reporting: https://console.cloud.google.com/errors
- Cloud Monitoring: https://console.cloud.google.com/monitoring