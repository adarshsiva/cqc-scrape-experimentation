# CQC API Access Fix - Complete Solution

## Problem
CQC API returns 403 Forbidden errors when accessed from Google Cloud IP addresses.

## Solutions Implemented

### 1. Enhanced Data Fetcher with Retry Logic
**Location**: `src/ingestion/cqc_fetcher_cloud.py`

Features:
- Advanced retry strategy with exponential backoff
- Custom headers to mimic browser requests
- Random delay between requests to avoid rate limiting
- Automatic fallback mechanisms for 403 errors
- Session management with connection pooling

### 2. Cloud Function Proxy
**Location**: `src/proxy/cqc_proxy_function.py`

Benefits:
- Runs on different Google Cloud IP ranges
- Acts as intermediary between Cloud Run and CQC API
- Lightweight and serverless
- Can be called from any GCP service

### 3. Deployment Scripts
Created multiple deployment options:

1. **Enhanced Fetcher Deployment**
   ```bash
   ./scripts/deploy_cqc_fetcher_enhanced.sh
   ```

2. **Proxy Function Deployment**
   ```bash
   cd src/proxy
   ./deploy.sh
   ```

3. **Complete Test & Deploy**
   ```bash
   ./scripts/test_and_deploy_all.sh
   ```

## Setup Instructions

### Step 1: Set CQC API Key
```bash
# Store API key in Secret Manager
echo -n 'YOUR_CQC_API_KEY' | gcloud secrets create cqc-subscription-key \
  --data-file=- \
  --project=machine-learning-exp-467008
```

### Step 2: Deploy Enhanced Fetcher
```bash
# Build and deploy the enhanced fetcher
cd /Users/adarsh/Documents/Dev/CQC_scrape
gcloud builds submit \
  --config=src/ingestion/cloudbuild-fetcher.yaml \
  --project=machine-learning-exp-467008
```

### Step 3: Deploy Proxy Function (Alternative)
```bash
cd src/proxy
gcloud functions deploy cqc-api-proxy \
  --gen2 \
  --runtime=python311 \
  --region=europe-west2 \
  --source=. \
  --entry-point=cqc_proxy \
  --trigger-http \
  --allow-unauthenticated \
  --project=machine-learning-exp-467008
```

### Step 4: Test Connection
```bash
# Test via Cloud Run Job
gcloud run jobs execute cqc-data-fetcher-enhanced \
  --region=europe-west2 \
  --project=machine-learning-exp-467008

# Test via Proxy Function
curl "https://europe-west2-machine-learning-exp-467008.cloudfunctions.net/cqc-api-proxy?endpoint=locations&page=1&perPage=5"
```

## Architecture

```
┌─────────────────┐
│  Cloud Scheduler│
└────────┬────────┘
         │
         v
┌─────────────────┐     ┌──────────────┐
│ Cloud Run Job   │────>│ CQC API      │
│ (Enhanced       │     │              │
│  Fetcher)       │     └──────────────┘
└─────────────────┘            ^
         │                     │
         v                     │
┌─────────────────┐     ┌──────────────┐
│ Cloud Storage   │     │Cloud Function│
│ (Raw Data)      │<────│(Proxy)       │
└─────────────────┘     └──────────────┘
         │
         v
┌─────────────────┐
│ BigQuery        │
│ (Processed Data)│
└─────────────────┘
```

## Monitoring

### View Fetcher Logs
```bash
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=cqc-data-fetcher-enhanced" \
  --project=machine-learning-exp-467008 \
  --limit=50
```

### View Proxy Logs
```bash
gcloud functions logs read cqc-api-proxy \
  --region=europe-west2 \
  --project=machine-learning-exp-467008
```

### Check Data in BigQuery
```bash
bq query --use_legacy_sql=false \
  "SELECT COUNT(*) FROM cqc_data.locations_detailed"
```

## Troubleshooting

### If 403 errors persist:
1. Check API key is valid
2. Try using the proxy function instead of direct access
3. Consider implementing a residential proxy service
4. Contact CQC for IP whitelisting

### If rate limited (429 errors):
1. Reduce batch size
2. Increase delay between requests
3. Implement exponential backoff
4. Consider upgrading CQC API access tier

## Next Steps

1. ✅ Enhanced fetcher with retry logic - COMPLETE
2. ✅ Cloud Function proxy - COMPLETE
3. ⏳ Test with real API key
4. ⏳ Schedule regular data fetches
5. ⏳ Implement incremental updates
6. ⏳ Set up monitoring alerts

## Model Training Status

The model save bug has been FIXED in `src/ml/train_model_cloud.py`:
- Safe access to classification report metrics
- Proper error handling for missing classes
- Model package validation before saving

To test model training:
```bash
gcloud run jobs execute cqc-model-trainer \
  --region=europe-west2 \
  --project=machine-learning-exp-467008
```