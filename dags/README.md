# CQC Pipeline Airflow DAGs

This directory contains Apache Airflow DAGs for orchestrating the CQC (Care Quality Commission) data pipeline on Google Cloud Composer.

## DAG Overview

### 1. CQC Daily Pipeline (`cqc_daily_pipeline.py`)
**Schedule**: Daily at 2 AM UTC

This is the main ETL pipeline that:
- Fetches fresh data from the CQC API
- Loads data into BigQuery staging tables
- Performs data quality checks
- Merges data into production tables
- Updates feature engineering tables
- Triggers weekly model retraining (Sundays)
- Runs risk assessment predictions
- Sends alerts for high-risk locations

### 2. CQC Monitoring DAG (`cqc_monitoring_dag.py`)
**Schedule**: Every 4 hours

This DAG monitors system health:
- Checks data freshness
- Monitors Cloud Function execution logs
- Tests model endpoint health
- Tracks storage usage
- Monitors BigQuery costs
- Performs data quality checks
- Sends alerts for any anomalies

## Configuration

### Environment Variables

Set these variables in Cloud Composer:

```bash
# GCP Configuration
GCP_PROJECT_ID=your-project-id
GCP_REGION=europe-west2
BQ_DATASET_ID=cqc_data
GCS_BUCKET_NAME=your-bucket-name

# Function Names
CF_INGESTION_FUNCTION=cqc-data-ingestion
VERTEX_AI_ENDPOINT=projects/PROJECT/locations/REGION/endpoints/ENDPOINT_ID

# Notifications
ALERT_EMAIL=alerts@yourdomain.com
```

### Airflow Variables

Configure these in the Airflow UI:

- `gcp_project_id`: Your GCP project ID
- `gcp_region`: GCP region (e.g., europe-west2)
- `bq_dataset_id`: BigQuery dataset name
- `gcs_bucket_name`: Cloud Storage bucket name
- `cf_ingestion_function`: Cloud Function name for data ingestion
- `vertex_ai_endpoint`: Vertex AI endpoint ID
- `alert_email`: Email for alerts

## Directory Structure

```
dags/
├── cqc_daily_pipeline.py      # Main ETL pipeline
├── cqc_monitoring_dag.py      # System monitoring
├── config/
│   ├── __init__.py
│   └── dag_config.py          # Centralized configuration
└── utils/
    ├── __init__.py
    └── alerts.py              # Alert utility functions
```

## Key Features

### Error Handling
- All tasks have retry logic with exponential backoff
- Failed tasks trigger email alerts
- Critical tasks have custom error handling

### Data Quality
- Automatic validation of data completeness
- Duplicate detection
- Null value checks
- Schema validation

### Monitoring
- Real-time system health checks
- Cost monitoring and alerts
- Performance metrics tracking
- Automated issue detection

### Scalability
- Parallel task execution where possible
- Configurable resource allocation
- Batch processing for large datasets

## Deployment

1. Upload DAG files to Cloud Composer environment:
```bash
gcloud composer environments storage dags import \
    --environment ENVIRONMENT_NAME \
    --location LOCATION \
    --source dags/
```

2. Set Airflow variables:
```bash
gcloud composer environments run ENVIRONMENT_NAME \
    --location LOCATION \
    variables set -- gcp_project_id YOUR_PROJECT_ID
```

3. Verify DAGs are loaded correctly in the Airflow UI

## Monitoring and Alerts

### Alert Types
1. **High Risk Alerts**: Locations identified as high-risk
2. **System Alerts**: Infrastructure or pipeline issues
3. **Data Quality Alerts**: Data validation failures
4. **Cost Alerts**: Budget threshold exceeded

### Alert Channels
- Email notifications (configured via ALERT_EMAIL)
- BigQuery tables for historical tracking
- Optional Slack integration (configure webhook URL)

## Troubleshooting

### Common Issues

1. **Data Not Fresh**
   - Check Cloud Function logs
   - Verify CQC API is accessible
   - Check for rate limiting

2. **Model Endpoint Unhealthy**
   - Verify endpoint exists in Vertex AI
   - Check model deployment status
   - Review endpoint logs

3. **High Storage Costs**
   - Review lifecycle policies
   - Check for unnecessary data retention
   - Optimize query patterns

## Maintenance

### Weekly Tasks
- Review model performance metrics
- Check for any failed DAG runs
- Monitor storage growth trends

### Monthly Tasks
- Update DAG configurations if needed
- Review and optimize query performance
- Update documentation

## Support

For issues or questions:
1. Check Airflow logs in Cloud Composer
2. Review Cloud Logging for detailed errors
3. Contact the data team at alerts@yourdomain.com