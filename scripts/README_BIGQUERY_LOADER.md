# CQC BigQuery Loader

This Cloud Run job loads CQC data from Cloud Storage to BigQuery, enriching it with detailed location information fetched from the CQC API.

## Overview

The loader performs the following tasks:
1. Reads JSON data files from Cloud Storage (`gs://machine-learning-exp-467008-cqc-raw-data/real_data/locations/`)
2. Extracts unique location IDs from the files
3. Fetches detailed information for each location from the CQC API
4. Transforms the data into BigQuery-compatible format
5. Loads data into the `cqc_data.locations_detailed` table
6. Creates/updates the `cqc_data.ml_features_proactive` view for ML training

## Components

### Main Script: `process_data_job.py`
- Handles Cloud Storage operations
- Makes API calls to fetch detailed location data
- Performs data transformation and type conversions
- Manages BigQuery table creation and data insertion
- Creates ML feature engineering view

### Dockerfile: `Dockerfile.process`
- Based on Python 3.9 slim image
- Installs required dependencies from `requirements.txt`
- Runs the process_data_job.py script

### Dependencies (`requirements.txt`)
- `requests`: For CQC API calls
- `google-cloud-storage`: For reading from GCS
- `google-cloud-bigquery`: For BigQuery operations
- `google-cloud-secret-manager`: For secure API key storage

## Deployment

### Using the deployment script:
```bash
cd scripts
./deploy_bigquery_loader.sh
```

### Using Cloud Build:
```bash
gcloud builds submit --config scripts/cloudbuild-bigquery-loader.yaml \
  --project machine-learning-exp-467008
```

### Manual deployment:
```bash
# Build the image
gcloud builds submit --tag gcr.io/machine-learning-exp-467008/cqc-bigquery-loader \
  -f scripts/Dockerfile.process scripts/

# Deploy the job
gcloud run jobs create cqc-bigquery-loader \
  --image gcr.io/machine-learning-exp-467008/cqc-bigquery-loader \
  --region europe-west2 \
  --memory 4Gi \
  --cpu 2 \
  --task-timeout 3600 \
  --service-account 744974744548-compute@developer.gserviceaccount.com \
  --set-env-vars GCP_PROJECT=machine-learning-exp-467008
```

## Execution

### Execute the job:
```bash
./execute_bigquery_loader.sh
```

Or manually:
```bash
gcloud run jobs execute cqc-bigquery-loader \
  --region europe-west2 \
  --project machine-learning-exp-467008
```

### View logs:
```bash
# List recent executions
gcloud run jobs executions list \
  --job cqc-bigquery-loader \
  --region europe-west2 \
  --project machine-learning-exp-467008

# View logs for a specific execution
gcloud run jobs executions logs [EXECUTION_NAME] \
  --region europe-west2 \
  --project machine-learning-exp-467008
```

## Data Schema

### BigQuery Table: `cqc_data.locations_detailed`
- `locationId`: Unique identifier for the location
- `name`: Location name
- `type`: Type of care location
- `numberOfBeds`: Number of beds (0 if not applicable)
- `registrationDate`: Date location was registered
- `postalCode`: Postal code
- `region`: Geographic region
- `localAuthority`: Local authority area
- `lastInspectionDate`: Date of most recent inspection
- `providerId`: Associated provider ID
- `regulatedActivitiesCount`: Number of regulated activities
- `specialismsCount`: Number of specialisms
- `serviceTypesCount`: Number of service types
- Rating fields: `overallRating`, `safeRating`, `effectiveRating`, etc.
- `rawData`: Complete JSON response from API
- `fetchTimestamp`: When the data was fetched

### ML Features View: `cqc_data.ml_features_proactive`
Engineered features for ML training including:
- Numeric encodings of ratings
- Time-based features (days since registration/inspection)
- Risk indicators and composite risk scores
- Binary labels for at-risk classification

## Configuration

The job uses the following configuration:
- **Memory**: 4Gi (to handle large datasets)
- **CPU**: 2 cores
- **Timeout**: 3600 seconds (1 hour)
- **Parallelism**: 1 (sequential processing)
- **Max retries**: 1

## Rate Limiting

The script implements rate limiting when fetching from the CQC API:
- Uses ThreadPoolExecutor with 5 concurrent workers
- Includes 1-second delays every 10 requests
- Limits initial processing to 1000 locations

## Error Handling

- Graceful handling of API failures
- Continues processing even if individual location fetches fail
- Logs all errors for debugging
- Saves backup data to Cloud Storage

## Monitoring

Check job status and performance:
```bash
# View job details
gcloud run jobs describe cqc-bigquery-loader \
  --region europe-west2 \
  --project machine-learning-exp-467008

# Monitor in Cloud Console
https://console.cloud.google.com/run/jobs/details/europe-west2/cqc-bigquery-loader
```