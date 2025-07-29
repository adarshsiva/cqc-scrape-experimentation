# CQC Rating Predictor - Deployment Scripts

This directory contains comprehensive deployment and management scripts for the CQC Rating Predictor ML System on Google Cloud Platform.

## Prerequisites

Before running any deployment scripts, ensure you have:

1. **GCP Account and Project**: A Google Cloud Platform account with a project created
2. **Required Tools Installed**:
   - `gcloud` CLI (Google Cloud SDK)
   - `bq` CLI (BigQuery command-line tool)
   - `gsutil` (Cloud Storage utility)
   - `curl` and `jq` for testing
3. **Environment Variables Set**:
   ```bash
   export GCP_PROJECT="your-project-id"
   export GCP_REGION="your-region"  # e.g., "europe-west2"
   ```
4. **Authentication**: Run `gcloud auth login` and `gcloud auth application-default login`
5. **Billing**: Ensure billing is enabled for your GCP project

## Scripts Overview

### 1. `setup_gcp_resources.sh`
Creates all required GCP resources for the CQC system.

**What it does:**
- Enables required GCP APIs
- Creates service accounts with appropriate permissions
- Sets up Cloud Storage buckets with lifecycle policies
- Creates BigQuery dataset and tables
- Initializes Secret Manager secrets
- Creates App Engine app (required for Cloud Scheduler)
- Sets up Cloud Composer environment
- Prepares Vertex AI resources

**Usage:**
```bash
./setup_gcp_resources.sh
```

**Important Notes:**
- This script can take 20-30 minutes due to Cloud Composer setup
- After running, update the `cqc-api-key` secret with your actual CQC API key

### 2. `deploy_all_services.sh`
Deploys all application services to GCP.

**What it does:**
- Deploys Cloud Functions (API ingestion and prediction service)
- Builds and deploys Dataflow pipeline templates
- Deploys ML models to Vertex AI endpoints
- Uploads Cloud Composer DAGs
- Sets up monitoring and alerting

**Usage:**
```bash
./deploy_all_services.sh
```

**Prerequisites:**
- Run `setup_gcp_resources.sh` first
- Ensure ML models are trained and available in GCS

### 3. `create_scheduler_jobs.sh`
Sets up automated schedules for the pipeline.

**What it does:**
- Creates daily data ingestion job (2 AM UK time)
- Sets up weekly full refresh job (Sunday 3 AM UK time)
- Configures monthly model retraining job (1st of month, 4 AM UK time)
- Creates daily data quality check job (6 AM UK time)

**Usage:**
```bash
./create_scheduler_jobs.sh
```

**Interactive Features:**
- Option to test run the daily ingestion job immediately
- Lists all created scheduler jobs with their schedules

### 4. `test_end_to_end.sh`
Comprehensive testing script for the entire system.

**What it tests:**
- GCP resource availability (buckets, datasets, secrets)
- Cloud Function deployments and responses
- API ingestion functionality
- BigQuery table schemas and data
- Dataflow pipeline execution
- ML model deployment and predictions
- Cloud Scheduler job configurations
- End-to-end prediction flow

**Usage:**
```bash
./test_end_to_end.sh
```

**Output:**
- Detailed test results with pass/fail status
- Summary report with total passed/failed tests
- Exit code 0 if all tests pass, 1 if any fail

### 5. `monitor_system.sh`
Real-time monitoring of system health.

**What it monitors:**
- Cloud Function status and error rates
- Storage bucket sizes and recent uploads
- BigQuery table row counts and recent updates
- Dataflow job statuses
- Cloud Scheduler job execution times
- Vertex AI endpoint health

**Usage:**
```bash
# Single health check
./monitor_system.sh

# Continuous monitoring (updates every 60 seconds)
./monitor_system.sh --continuous

# Custom interval (e.g., every 30 seconds)
MONITOR_INTERVAL=30 ./monitor_system.sh -c
```

### 6. `cleanup_resources.sh`
Removes all CQC project resources (USE WITH EXTREME CAUTION).

**What it deletes:**
- All Cloud Scheduler jobs
- Cloud Functions
- Vertex AI models and endpoints
- Cloud Composer environment
- BigQuery datasets and tables
- Cloud Storage buckets and data
- Secret Manager secrets
- Service accounts

**Usage:**
```bash
./cleanup_resources.sh
```

**Safety Features:**
- Multiple confirmation prompts
- Requires typing "DELETE ALL" and project ID to proceed
- Cannot be undone

## Deployment Workflow

### Initial Setup
1. Set environment variables:
   ```bash
   export GCP_PROJECT="your-project-id"
   export GCP_REGION="europe-west2"  # or your preferred region
   ```

2. Run setup script:
   ```bash
   ./setup_gcp_resources.sh
   ```

3. Update CQC API key:
   ```bash
   echo -n "your-actual-api-key" | gcloud secrets versions add cqc-api-key --data-file=-
   ```

### Deploy Services
4. Deploy all services:
   ```bash
   ./deploy_all_services.sh
   ```

5. Create scheduled jobs:
   ```bash
   ./create_scheduler_jobs.sh
   ```

### Verify Deployment
6. Run end-to-end tests:
   ```bash
   ./test_end_to_end.sh
   ```

7. Monitor system health:
   ```bash
   ./monitor_system.sh --continuous
   ```

## Environment Variables

All scripts support the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GCP_PROJECT` | GCP project ID | Required |
| `GCP_REGION` | GCP region | Required |
| `GCP_ZONE` | GCP zone | `${GCP_REGION}-a` |
| `GCS_RAW_BUCKET` | Raw data bucket | `${GCP_PROJECT}-cqc-raw` |
| `GCS_PROCESSED_BUCKET` | Processed data bucket | `${GCP_PROJECT}-cqc-processed` |
| `GCS_MODELS_BUCKET` | ML models bucket | `${GCP_PROJECT}-cqc-models` |
| `BQ_DATASET` | BigQuery dataset name | `cqc_data` |
| `COMPOSER_ENV` | Composer environment name | `cqc-composer-env` |
| `SERVICE_ACCOUNT` | Service account name | `cqc-service-account` |

## Troubleshooting

### Common Issues

1. **API not enabled error**:
   ```bash
   gcloud services enable [API_NAME] --project=$GCP_PROJECT
   ```

2. **Permission denied**:
   - Ensure you have Owner or Editor role on the project
   - Check service account permissions

3. **Scheduler jobs not running**:
   - Verify App Engine is created in the correct region
   - Check service account has necessary permissions

4. **Function deployment fails**:
   - Check requirements.txt files exist
   - Verify source code is in correct directory

### Getting Help

1. Check script logs - all scripts provide detailed logging
2. Use `--help` flag on gcloud commands for more options
3. Check GCP Console for detailed error messages
4. Review Cloud Logging for runtime errors

## Cost Optimization

To minimize costs:

1. **Use lifecycle policies** (automatically set by setup script)
2. **Set appropriate machine types** for Composer and Vertex AI
3. **Configure autoscaling** for Cloud Functions and endpoints
4. **Schedule jobs during off-peak hours**
5. **Use cleanup script** to remove resources when not needed

## Security Best Practices

1. **Never commit API keys** - always use Secret Manager
2. **Use service accounts** with minimal required permissions
3. **Enable audit logging** for all services
4. **Restrict bucket access** to authorized users only
5. **Use VPC Service Controls** for additional security

## Next Steps

After successful deployment:

1. Configure monitoring dashboards in Cloud Console
2. Set up alerting policies for critical failures
3. Implement backup strategies for important data
4. Document any custom configurations
5. Plan for disaster recovery scenarios