# Deploy CQC Data Fetcher to Cloud Run Job

## Prerequisites
- Google Cloud SDK (gcloud) installed and authenticated
- Docker installed (for local build) OR use Cloud Shell
- Appropriate IAM permissions for Cloud Run and Container Registry

## Option 1: Deploy using Cloud Shell (Recommended)

1. Open Cloud Shell in your browser:
   ```
   https://console.cloud.google.com/cloudshell
   ```

2. Clone your repository:
   ```bash
   git clone <your-repo-url>
   cd CQC_scrape
   ```

3. Run the Cloud Build deployment:
   ```bash
   gcloud builds submit \
     --project=machine-learning-exp-467008 \
     --config=scripts/cloudbuild-fetcher.yaml \
     .
   ```

4. Execute the job:
   ```bash
   gcloud run jobs execute cqc-data-fetcher \
     --project=machine-learning-exp-467008 \
     --region=europe-west2
   ```

## Option 2: Deploy from Local Machine

1. Ensure Docker is running:
   ```bash
   docker info
   ```

2. Authenticate with Google Cloud:
   ```bash
   gcloud auth login
   gcloud config set project machine-learning-exp-467008
   ```

3. Run the deployment script:
   ```bash
   ./scripts/deploy_fetcher_local.sh
   ```

## Option 3: Manual Deployment Steps

1. Build the Docker image:
   ```bash
   cd scripts
   docker build -t gcr.io/machine-learning-exp-467008/cqc-data-fetcher:latest -f Dockerfile.fetcher .
   ```

2. Configure Docker authentication:
   ```bash
   gcloud auth configure-docker gcr.io
   ```

3. Push the image:
   ```bash
   docker push gcr.io/machine-learning-exp-467008/cqc-data-fetcher:latest
   ```

4. Deploy the Cloud Run Job:
   ```bash
   gcloud run jobs deploy cqc-data-fetcher \
     --image=gcr.io/machine-learning-exp-467008/cqc-data-fetcher:latest \
     --region=europe-west2 \
     --project=machine-learning-exp-467008 \
     --memory=2Gi \
     --cpu=2 \
     --max-retries=1 \
     --timeout=30m \
     --service-account=cqc-ml-processor@machine-learning-exp-467008.iam.gserviceaccount.com \
     --set-env-vars="GCP_PROJECT=machine-learning-exp-467008,GCS_BUCKET=machine-learning-exp-467008-cqc-raw-data,MAX_LOCATIONS=1000"
   ```

5. Execute the job:
   ```bash
   gcloud run jobs execute cqc-data-fetcher \
     --project=machine-learning-exp-467008 \
     --region=europe-west2
   ```

## Monitoring the Job

1. View job executions:
   ```bash
   gcloud run jobs executions list \
     --project=machine-learning-exp-467008 \
     --region=europe-west2 \
     --job=cqc-data-fetcher
   ```

2. View logs:
   ```bash
   gcloud run jobs executions logs <execution-name> \
     --project=machine-learning-exp-467008 \
     --region=europe-west2
   ```

3. Check Cloud Storage for fetched data:
   ```bash
   gsutil ls -r gs://machine-learning-exp-467008-cqc-raw-data/detailed_locations/
   ```

## Configuration

The job can be configured with environment variables:
- `GCP_PROJECT`: Google Cloud project ID (default: machine-learning-exp-467008)
- `GCS_BUCKET`: Cloud Storage bucket for data (default: machine-learning-exp-467008-cqc-raw-data)
- `MAX_LOCATIONS`: Maximum number of locations to fetch (default: 1000)

## Scheduling (Optional)

To run the fetcher on a schedule, create a Cloud Scheduler job:

```bash
gcloud scheduler jobs create http cqc-data-fetcher-schedule \
  --location=europe-west2 \
  --schedule="0 2 * * *" \
  --uri="https://europe-west2-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/machine-learning-exp-467008/jobs/cqc-data-fetcher:run" \
  --http-method=POST \
  --oauth-service-account-email=cqc-ml-processor@machine-learning-exp-467008.iam.gserviceaccount.com
```

This schedules the job to run daily at 2 AM.