# CQC Rating Predictor - Operational Runbook

## Table of Contents
1. [System Overview](#system-overview)
2. [Daily Operations](#daily-operations)
3. [Monitoring](#monitoring)
4. [Incident Response](#incident-response)
5. [Troubleshooting Guide](#troubleshooting-guide)
6. [Maintenance Procedures](#maintenance-procedures)
7. [Disaster Recovery](#disaster-recovery)
8. [Performance Tuning](#performance-tuning)

## System Overview

### Critical Components
- **Cloud Functions**: `cqc-data-ingestion`, `cqc-rating-prediction`
- **Cloud Storage Buckets**: `machine-learning-exp-467008-cqc-raw-data`, `machine-learning-exp-467008-cqc-ml-artifacts`
- **BigQuery Dataset**: `cqc_data` (tables: locations, providers, ml_features, predictions)
- **Vertex AI Endpoints**: ML model serving endpoints
- **Cloud Scheduler**: Weekly data ingestion job

### Service Dependencies
```
CQC API → Cloud Function → Cloud Storage → Dataflow → BigQuery → Vertex AI → Prediction API
```

## Daily Operations

### Health Checks (Daily)

1. **Check System Status**
   ```bash
   # Check Cloud Functions
   gcloud functions list --region=europe-west2 --format="table(name,state,updateTime)"
   
   # Check recent executions
   gcloud functions logs read cqc-data-ingestion --region=europe-west2 --limit=10
   gcloud functions logs read cqc-rating-prediction --region=europe-west2 --limit=10
   
   # Check BigQuery data freshness
   bq query --use_legacy_sql=false '
   SELECT 
     MAX(last_updated) as latest_update,
     DATE_DIFF(CURRENT_DATE(), DATE(MAX(last_updated)), DAY) as days_old
   FROM `machine-learning-exp-467008.cqc_data.locations`'
   ```

2. **Check Model Performance**
   ```bash
   # Get current endpoint
   ENDPOINT_ID=$(gcloud ai endpoints list --region=europe-west2 --format="value(name)" | head -n1)
   
   # Check endpoint status
   gcloud ai endpoints describe $ENDPOINT_ID --region=europe-west2
   ```

3. **Monitor Storage Usage**
   ```bash
   # Check bucket sizes
   gsutil du -s gs://machine-learning-exp-467008-cqc-raw-data
   gsutil du -s gs://machine-learning-exp-467008-cqc-ml-artifacts
   
   # Check BigQuery storage
   bq query --use_legacy_sql=false '
   SELECT 
     table_name,
     ROUND(size_bytes/1024/1024/1024, 2) as size_gb,
     row_count
   FROM `machine-learning-exp-467008.cqc_data.__TABLES__`'
   ```

### Weekly Operations

1. **Verify Scheduled Ingestion (Mondays)**
   ```bash
   # Check scheduler job status
   gcloud scheduler jobs describe cqc-weekly-ingestion --location=europe-west2
   
   # Verify last run
   gcloud scheduler jobs list --location=europe-west2 --format="table(name,state,lastAttemptTime)"
   ```

2. **Data Quality Checks**
   ```bash
   # Check for data anomalies
   bq query --use_legacy_sql=false '
   SELECT 
     DATE(last_updated) as update_date,
     COUNT(*) as record_count,
     COUNT(DISTINCT provider_id) as unique_providers
   FROM `machine-learning-exp-467008.cqc_data.locations`
   WHERE DATE(last_updated) >= DATE_SUB(CURRENT_DATE(), INTERVAL 7 DAY)
   GROUP BY update_date
   ORDER BY update_date DESC'
   ```

## Monitoring

### Key Metrics to Monitor

1. **Cloud Function Metrics**
   - Execution count
   - Error rate
   - Execution duration
   - Memory utilization

2. **Dataflow Pipeline Metrics**
   - Elements processed
   - System lag
   - Worker utilization
   - Failed elements

3. **Model Performance Metrics**
   - Prediction latency
   - Request count
   - Error rate
   - Model accuracy (from logged predictions)

### Setting Up Alerts

```bash
# Create error rate alert for Cloud Functions
gcloud alpha monitoring policies create \
  --notification-channels=[CHANNEL_ID] \
  --display-name="High Cloud Function Error Rate" \
  --condition-display-name="Error rate > 5%" \
  --condition-threshold-value=0.05 \
  --condition-threshold-duration=300s \
  --condition-threshold-filter='resource.type="cloud_function"
    AND resource.labels.function_name=starts_with("cqc-")
    AND metric.type="cloudfunctions.googleapis.com/function/error_count"'
```

### Dashboard Links
- [Cloud Functions Dashboard](https://console.cloud.google.com/functions/list?project=machine-learning-exp-467008)
- [Dataflow Jobs](https://console.cloud.google.com/dataflow/jobs?project=machine-learning-exp-467008)
- [BigQuery Console](https://console.cloud.google.com/bigquery?project=machine-learning-exp-467008)
- [Vertex AI Models](https://console.cloud.google.com/vertex-ai/models?project=machine-learning-exp-467008)
- [Cloud Logging](https://console.cloud.google.com/logs/query?project=machine-learning-exp-467008)

## Incident Response

### Severity Levels
- **P1 (Critical)**: Complete system outage, no predictions available
- **P2 (High)**: Degraded performance, >10% errors
- **P3 (Medium)**: Data ingestion issues, stale data
- **P4 (Low)**: Minor issues, cosmetic problems

### Response Procedures

#### P1: Complete System Outage
1. **Immediate Actions**
   ```bash
   # Check all Cloud Functions
   gcloud functions list --region=europe-west2
   
   # Check recent errors
   gcloud logging read "severity>=ERROR" --limit=50 --format=json
   
   # Check service health
   gcloud services list --enabled | grep -E "(cloudfunctions|aiplatform|bigquery)"
   ```

2. **Restart Services**
   ```bash
   # Redeploy prediction function
   cd src/prediction
   gcloud functions deploy cqc-rating-prediction \
     --runtime python311 \
     --trigger-http \
     --entry-point predict \
     --memory 1GB \
     --region europe-west2
   ```

#### P2: High Error Rate
1. **Identify Error Source**
   ```bash
   # Check function logs
   gcloud functions logs read cqc-rating-prediction \
     --region=europe-west2 \
     --filter="severity>=ERROR" \
     --limit=100
   
   # Check endpoint health
   gcloud ai endpoints list --region=europe-west2
   ```

2. **Mitigate**
   - Scale up Cloud Function memory/CPU
   - Check model endpoint quota
   - Verify BigQuery connectivity

#### P3: Data Ingestion Failure
1. **Check Ingestion Status**
   ```bash
   # Check scheduler job
   gcloud scheduler jobs describe cqc-weekly-ingestion --location=europe-west2
   
   # Manually trigger ingestion
   INGESTION_URL=$(gcloud functions describe cqc-data-ingestion --region=europe-west2 --format="value(url)")
   curl -X POST $INGESTION_URL
   ```

2. **Verify API Credentials**
   ```bash
   # Check secret exists
   gcloud secrets versions list cqc-subscription-key
   ```

## Troubleshooting Guide

### Common Issues and Solutions

#### 1. "API Key Invalid" Error
**Symptoms**: Ingestion function returns 401/403 errors
**Solution**:
```bash
# Update API key
echo -n "new-api-key" | gcloud secrets versions add cqc-subscription-key --data-file=-

# Restart function to pick up new key
gcloud functions deploy cqc-data-ingestion --gen2 --region=europe-west2
```

#### 2. "Model Endpoint Not Found" Error
**Symptoms**: Prediction function returns 404 errors
**Solution**:
```bash
# List available endpoints
gcloud ai endpoints list --region=europe-west2

# Update function with correct endpoint
cd src/prediction
# Edit main.py with correct endpoint ID
gcloud functions deploy cqc-rating-prediction --region=europe-west2
```

#### 3. BigQuery Permission Denied
**Symptoms**: ETL pipeline fails with permission errors
**Solution**:
```bash
# Grant necessary permissions
gcloud projects add-iam-policy-binding machine-learning-exp-467008 \
  --member="serviceAccount:cqc-cf-service-account@machine-learning-exp-467008.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

#### 4. Out of Memory Errors
**Symptoms**: Cloud Functions crash with memory errors
**Solution**:
```bash
# Increase memory allocation
gcloud functions deploy cqc-rating-prediction \
  --memory=2GB \
  --region=europe-west2
```

#### 5. Slow Predictions
**Symptoms**: Prediction latency > 5 seconds
**Solution**:
- Check model complexity
- Enable Cloud Function connection pooling
- Consider batch predictions for bulk requests

### Debug Commands

```bash
# Get detailed function logs
gcloud logging read "resource.type=cloud_function AND resource.labels.function_name=cqc-rating-prediction" \
  --limit=100 \
  --format="table(timestamp,severity,textPayload)"

# Check BigQuery job history
bq ls -j -a -n 50

# Monitor Dataflow job
gcloud dataflow jobs list --region=europe-west2 --filter="state=running"

# Check storage bucket permissions
gsutil iam get gs://machine-learning-exp-467008-cqc-raw-data
```

## Maintenance Procedures

### Monthly Maintenance

1. **Clean Up Old Data**
   ```bash
   # Delete old raw data files (older than 90 days)
   gsutil -m rm "gs://machine-learning-exp-467008-cqc-raw-data/raw/*/*2024-[01-09]*"
   
   # Archive old predictions
   bq query --use_legacy_sql=false '
   CREATE OR REPLACE TABLE `machine-learning-exp-467008.cqc_data.predictions_archive` AS
   SELECT * FROM `machine-learning-exp-467008.cqc_data.predictions`
   WHERE prediction_timestamp < DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)'
   ```

2. **Update Dependencies**
   ```bash
   # Update Cloud Function dependencies
   cd src/ingestion
   pip-compile requirements.in
   cd ../prediction
   pip-compile requirements.in
   ```

3. **Model Retraining**
   ```bash
   # Trigger model retraining pipeline
   ./train-models.sh
   ```

### Quarterly Maintenance

1. **Security Audit**
   - Review service account permissions
   - Rotate API keys
   - Check for unused resources

2. **Cost Optimization**
   - Review BigQuery slot usage
   - Optimize storage lifecycle policies
   - Right-size Cloud Function resources

3. **Performance Review**
   - Analyze prediction accuracy trends
   - Review system latency metrics
   - Plan capacity for growth

## Disaster Recovery

### Backup Procedures

1. **Daily Backups**
   ```bash
   # Export BigQuery tables
   bq extract --destination_format=AVRO \
     cqc_data.locations \
     gs://machine-learning-exp-467008-cqc-backups/bigquery/locations/$(date +%Y%m%d)/*.avro
   
   bq extract --destination_format=AVRO \
     cqc_data.providers \
     gs://machine-learning-exp-467008-cqc-backups/bigquery/providers/$(date +%Y%m%d)/*.avro
   ```

2. **Model Artifacts Backup**
   ```bash
   # Copy model artifacts
   gsutil -m cp -r gs://machine-learning-exp-467008-cqc-ml-artifacts/models/* \
     gs://machine-learning-exp-467008-cqc-backups/models/$(date +%Y%m%d)/
   ```

### Recovery Procedures

1. **BigQuery Table Recovery**
   ```bash
   # Restore from snapshot
   bq cp cqc_data.locations@1234567890000 cqc_data.locations_restored
   
   # Or restore from backup
   bq load --source_format=AVRO \
     cqc_data.locations_restored \
     gs://machine-learning-exp-467008-cqc-backups/bigquery/locations/20240115/*.avro
   ```

2. **Model Recovery**
   ```bash
   # List available model versions
   gcloud ai models list --region=europe-west2
   
   # Deploy previous version
   gcloud ai endpoints deploy-model [ENDPOINT_ID] \
     --region=europe-west2 \
     --model=[MODEL_ID] \
     --display-name="cqc-model-rollback"
   ```

### RTO/RPO Targets
- **RTO (Recovery Time Objective)**: 4 hours
- **RPO (Recovery Point Objective)**: 24 hours

## Performance Tuning

### Cloud Function Optimization

1. **Connection Pooling**
   ```python
   # Add to function initialization
   from google.cloud import bigquery
   
   # Global client initialization
   bq_client = bigquery.Client()
   ```

2. **Memory Settings**
   - Ingestion Function: 512MB-1GB
   - Prediction Function: 1GB-2GB

3. **Concurrency Settings**
   ```bash
   gcloud functions deploy cqc-rating-prediction \
     --max-instances=100 \
     --concurrency=10
   ```

### BigQuery Optimization

1. **Partitioning**
   ```sql
   -- Partition by date for time-based queries
   CREATE TABLE cqc_data.locations_partitioned
   PARTITION BY DATE(last_updated)
   AS SELECT * FROM cqc_data.locations;
   ```

2. **Clustering**
   ```sql
   -- Cluster by frequently queried fields
   CREATE TABLE cqc_data.locations_clustered
   PARTITION BY DATE(last_updated)
   CLUSTER BY region, type
   AS SELECT * FROM cqc_data.locations;
   ```

### Model Serving Optimization

1. **Auto-scaling Configuration**
   ```bash
   gcloud ai endpoints update [ENDPOINT_ID] \
     --region=europe-west2 \
     --min-replica-count=2 \
     --max-replica-count=10
   ```

2. **Batch Predictions**
   - Use for processing > 100 predictions
   - Submit via Vertex AI batch prediction jobs

## Appendix

### Useful Scripts

1. **System Health Check Script**
   ```bash
   #!/bin/bash
   echo "=== CQC System Health Check ==="
   echo "Cloud Functions:"
   gcloud functions list --region=europe-west2 --format="table(name,state)"
   echo -e "\nBigQuery Tables:"
   bq ls cqc_data
   echo -e "\nRecent Errors:"
   gcloud logging read "severity>=ERROR" --limit=10 --format="value(textPayload)"
   ```

2. **Quick Deployment Script**
   ```bash
   #!/bin/bash
   set -e
   echo "Deploying CQC system updates..."
   cd src/ingestion && gcloud functions deploy cqc-data-ingestion --quiet
   cd ../prediction && gcloud functions deploy cqc-rating-prediction --quiet
   echo "Deployment complete!"
   ```

### Emergency Contacts
- **GCP Support**: [Create support ticket](https://console.cloud.google.com/support)
- **On-call Engineer**: Check PagerDuty rotation
- **Escalation**: Team Lead → Engineering Manager → CTO

### Reference Links
- [GCP Status Page](https://status.cloud.google.com/)
- [CQC API Documentation](https://www.cqc.org.uk/about-us/transparency/using-cqc-data)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [Cloud Functions Documentation](https://cloud.google.com/functions/docs)