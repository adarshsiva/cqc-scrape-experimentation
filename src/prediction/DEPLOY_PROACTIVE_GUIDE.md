# Proactive Risk Assessment API Deployment Guide

## Overview
This guide provides instructions for deploying the Proactive Risk Assessment API to Google Cloud Run.

## Prerequisites
1. Google Cloud SDK installed and authenticated
2. Docker installed (for local build option)
3. Access to the GCP project: `machine-learning-exp-467008`

## Deployment Options

### Option 1: Using Cloud Build (Recommended)

1. **Authenticate with Google Cloud:**
   ```bash
   gcloud auth login
   gcloud config set project machine-learning-exp-467008
   ```

2. **Submit the Cloud Build job:**
   ```bash
   cd /Users/adarsh/Documents/Dev/CQC_scrape
   gcloud builds submit --config=src/prediction/cloudbuild_proactive.yaml --substitutions=COMMIT_SHA=latest
   ```

3. **Verify deployment:**
   ```bash
   gcloud run services describe proactive-risk-assessment --region=europe-west2 --format='value(status.url)'
   ```

### Option 2: Using Local Docker Build

1. **Ensure Docker is running:**
   ```bash
   docker --version
   ```

2. **Run the deployment script:**
   ```bash
   cd /Users/adarsh/Documents/Dev/CQC_scrape/src/prediction
   ./deploy_proactive.sh
   ```

### Option 3: Manual Deployment

1. **Build the Docker image:**
   ```bash
   cd /Users/adarsh/Documents/Dev/CQC_scrape/src/prediction
   docker build -f Dockerfile.proactive -t gcr.io/machine-learning-exp-467008/proactive-risk-assessment .
   ```

2. **Push to Container Registry:**
   ```bash
   docker push gcr.io/machine-learning-exp-467008/proactive-risk-assessment
   ```

3. **Deploy to Cloud Run:**
   ```bash
   gcloud run deploy proactive-risk-assessment \
     --image gcr.io/machine-learning-exp-467008/proactive-risk-assessment \
     --platform managed \
     --region europe-west2 \
     --memory 2Gi \
     --cpu 2 \
     --timeout 300 \
     --concurrency 80 \
     --max-instances 10 \
     --set-env-vars "GCP_PROJECT=machine-learning-exp-467008,MODEL_BUCKET=machine-learning-exp-467008-cqc-ml-artifacts,MODEL_PATH=models/proactive/model_package.pkl" \
     --allow-unauthenticated
   ```

## Testing the Deployed API

1. **Get the service URL:**
   ```bash
   SERVICE_URL=$(gcloud run services describe proactive-risk-assessment --region=europe-west2 --format='value(status.url)')
   echo "Service URL: $SERVICE_URL"
   ```

2. **Test health endpoint:**
   ```bash
   curl "$SERVICE_URL/health"
   ```

3. **Test risk assessment endpoint:**
   ```bash
   curl -X POST "$SERVICE_URL/assess-risk" \
     -H "Content-Type: application/json" \
     -d '{
       "locationId": "1-123456789",
       "locationName": "Test Care Home",
       "staff_vacancy_rate": 0.25,
       "staff_turnover_rate": 0.35,
       "inspection_days_since_last": 450,
       "total_complaints": 2,
       "safe_key_questions_yes_ratio": 0.6,
       "effective_key_questions_yes_ratio": 0.7,
       "caring_key_questions_yes_ratio": 0.8,
       "responsive_key_questions_yes_ratio": 0.75,
       "well_led_key_questions_yes_ratio": 0.5
     }'
   ```

4. **Run comprehensive tests:**
   ```bash
   # Update the test script with your service URL
   sed -i '' "s|BASE_URL = .*|BASE_URL = \"$SERVICE_URL\"|" test_proactive_api.py
   python test_proactive_api.py
   ```

## API Endpoints

- **Health Check:** `GET /health`
- **Single Assessment:** `POST /assess-risk`
- **Batch Assessment:** `POST /batch-assess`
- **Risk Thresholds:** `GET /risk-thresholds`

## Monitoring

1. **View logs:**
   ```bash
   gcloud run services logs read proactive-risk-assessment --region=europe-west2
   ```

2. **View metrics:**
   ```bash
   gcloud run services describe proactive-risk-assessment --region=europe-west2
   ```

## Troubleshooting

1. **Model not loading:**
   - Verify the model exists in the GCS bucket:
     ```bash
     gsutil ls gs://machine-learning-exp-467008-cqc-ml-artifacts/models/proactive/
     ```

2. **Out of memory errors:**
   - The service is configured with 2Gi memory
   - Monitor memory usage in Cloud Console

3. **Timeout errors:**
   - Current timeout is 300 seconds
   - For large batch requests, consider reducing batch size

## Security Notes

- The service is currently configured with `--allow-unauthenticated`
- For production, consider implementing authentication:
  ```bash
  gcloud run services update proactive-risk-assessment \
    --no-allow-unauthenticated \
    --region=europe-west2
  ```

## Next Steps

1. Set up monitoring alerts for error rates and latency
2. Configure a custom domain if needed
3. Implement authentication for production use
4. Set up CI/CD pipeline for automatic deployments