# Dashboard Metrics Collector Cloud Function

This Cloud Function collects real-time metrics from care provider dashboards, calculates derived metrics, and publishes events to Pub/Sub for downstream processing.

## Features

- **Real-time Metrics Collection**: Collects incidents, staffing, and care quality metrics
- **Derived Metrics Calculation**: Computes incident rates, staff ratios, compliance scores
- **Pub/Sub Integration**: Publishes events to 'dashboard-events' topic
- **Feature Store Updates**: Updates Vertex AI Feature Store for critical metrics
- **BigQuery Storage**: Stores metrics for historical analysis
- **Error Handling**: Robust error handling with synthetic data fallbacks

## Metrics Collected

### Incident Metrics
- Total incidents, open incidents
- Severity distribution (high/medium/low)
- Average resolution time
- Derived: incident rate per bed, resolution efficiency score

### Staffing Metrics  
- Total staff, registered nurses, care assistants
- Vacancy count, turnover rate, absence rate
- Overtime hours
- Derived: staff-to-bed ratios, adequacy scores, stability scores

### Care Quality Metrics
- Medication errors, falls, pressure ulcers
- Resident and family satisfaction scores
- Complaints and compliments
- Care plan compliance
- Derived: composite care quality score, safety incident rates

## API Usage

### Collect Specific Metrics
```bash
curl -X POST https://REGION-PROJECT_ID.cloudfunctions.net/dashboard-metrics-collector \
  -H 'Content-Type: application/json' \
  -d '{
    "provider_id": "1-000000001",
    "location_id": "1-000000001", 
    "dashboard_type": "incidents"
  }'
```

### Collect All Metrics
```bash
curl -X POST https://REGION-PROJECT_ID.cloudfunctions.net/dashboard-metrics-collector \
  -H 'Content-Type: application/json' \
  -d '{
    "provider_id": "1-000000001",
    "collect_all": true
  }'
```

### Request Parameters
- `provider_id` (optional): CQC provider ID
- `location_id` (optional): CQC location ID  
- `dashboard_type` (optional): "incidents", "staffing", "care_quality", or "all"
- `collect_all` (optional): Boolean to collect all metric types

**Note**: Either `provider_id` or `location_id` must be provided.

## Response Format

```json
{
  "status": "success",
  "metrics_collected": 3,
  "events_published": 3,
  "feature_store_updates": 2,
  "timestamp": "2024-01-15T10:30:00Z"
}
```

## Deployment

### Quick Deploy
```bash
./deploy.sh
```

### Manual Deploy
```bash
gcloud functions deploy dashboard-metrics-collector \
  --source=. \
  --entry-point=collect_dashboard_metrics \
  --runtime=python311 \
  --trigger=http \
  --allow-unauthenticated \
  --region=europe-west2 \
  --memory=512MB \
  --timeout=300s
```

### Cloud Build Deploy
```bash
gcloud builds submit --config=cloudbuild.yaml
```

## Testing

Run unit tests:
```bash
python -m pytest test_main.py -v
```

## Dependencies

- functions-framework>=3.4.0
- google-cloud-pubsub>=2.18.0
- google-cloud-aiplatform>=1.35.0
- google-cloud-secret-manager>=2.16.0
- google-cloud-bigquery>=3.11.0
- requests>=2.31.0

## Configuration

### Environment Variables
- `PROJECT_ID`: GCP project ID (default: machine-learning-exp-467008)
- `REGION`: GCP region (default: europe-west2)

### Required GCP Resources
- Pub/Sub topic: `dashboard-events`
- BigQuery dataset: `cqc_data`
- BigQuery table: `dashboard_metrics`

### IAM Permissions
The function requires these roles:
- `roles/pubsub.publisher`
- `roles/bigquery.dataEditor`
- `roles/aiplatform.user`
- `roles/secretmanager.secretAccessor`

## Architecture

1. **Data Collection**: Queries BigQuery for recent metrics data
2. **Metric Calculation**: Computes derived metrics and scores
3. **Event Publishing**: Publishes to Pub/Sub for real-time processing
4. **Feature Store**: Updates critical metrics in Vertex AI Feature Store
5. **Historical Storage**: Stores all metrics in BigQuery for analysis

## Error Handling

- Falls back to synthetic data when BigQuery queries fail
- Graceful handling of Pub/Sub publish failures
- Comprehensive logging for debugging
- Returns appropriate HTTP status codes

## Monitoring

Monitor the function at:
https://console.cloud.google.com/functions/details/europe-west2/dashboard-metrics-collector?project=machine-learning-exp-467008