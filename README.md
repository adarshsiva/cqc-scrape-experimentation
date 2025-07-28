# CQC Rating Predictor ML System

A comprehensive machine learning system built on Google Cloud Platform to predict Care Quality Commission (CQC) ratings for healthcare providers in the UK.

## Overview

This system implements an end-to-end ML pipeline that:
- Ingests data from CQC's public API
- Processes and transforms data using Apache Beam
- Trains multiple ML models using Vertex AI
- Serves predictions via a REST API

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│ Cloud Scheduler │────▶│Cloud Function│────▶│   CQC API       │
└─────────────────┘     └──────────────┘     └─────────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Cloud Storage │
                        │  (Raw Data)  │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Dataflow    │
                        │  (ETL)       │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  BigQuery    │
                        │(Data Warehouse)│
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │  Vertex AI   │
                        │(ML Training) │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Vertex AI     │
                        │Endpoint      │
                        └──────────────┘
                               │
                               ▼
                        ┌──────────────┐
                        │Cloud Function│
                        │(Prediction API)│
                        └──────────────┘
```

## Project Structure

```
cqc-rating-predictor/
├── src/
│   ├── ingestion/        # API data ingestion
│   ├── etl/              # Data transformation pipelines
│   ├── ml/               # ML pipeline and features
│   ├── prediction/       # Prediction API
│   └── utils/            # Shared utilities
├── terraform/            # Infrastructure as Code
├── config/               # Configuration files
├── tests/                # Test suites
└── docs/                 # Documentation
```

## Prerequisites

- Google Cloud Platform account with billing enabled
- Python 3.11+
- Terraform 1.0+
- gcloud CLI installed and configured

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd cqc-rating-predictor
```

### 2. Set Environment Variables

```bash
export GCP_PROJECT="your-project-id"
export GCP_REGION="europe-west2"
```

### 3. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install gcloud components
gcloud components install beta
```

### 4. Deploy Infrastructure

```bash
cd terraform
terraform init
terraform plan -var="project_id=${GCP_PROJECT}"
terraform apply -var="project_id=${GCP_PROJECT}"
```

### 5. Configure Secrets

Add your CQC API credentials to Secret Manager:

```bash
echo -n "your-subscription-key" | gcloud secrets create cqc-subscription-key --data-file=-
echo -n "your-partner-code" | gcloud secrets create cqc-partner-code --data-file=-
```

### 6. Deploy Cloud Functions

```bash
# Deploy ingestion function
cd src/ingestion
gcloud functions deploy cqc-data-ingestion \
  --runtime python311 \
  --trigger-http \
  --entry-point ingest_cqc_data \
  --memory 512MB \
  --set-env-vars GCP_PROJECT=${GCP_PROJECT},GCS_BUCKET=${GCP_PROJECT}-cqc-raw-data

# Deploy prediction function
cd ../prediction
gcloud functions deploy cqc-rating-prediction \
  --runtime python311 \
  --trigger-http \
  --entry-point predict_cqc_rating \
  --memory 1GB \
  --set-env-vars GCP_PROJECT=${GCP_PROJECT}
```

### 7. Run Initial Data Ingestion

Trigger the ingestion function manually:

```bash
curl -X POST https://[REGION]-[PROJECT_ID].cloudfunctions.net/cqc-data-ingestion \
  -H "Authorization: Bearer $(gcloud auth print-identity-token)"
```

### 8. Run ETL Pipeline

```bash
cd src/etl
python dataflow_pipeline.py \
  --project-id=${GCP_PROJECT} \
  --dataset-id=cqc_data \
  --temp-location=gs://${GCP_PROJECT}-cqc-dataflow-temp/temp \
  --input-path=gs://${GCP_PROJECT}-cqc-raw-data/raw/locations/*.json \
  --data-type=locations \
  --runner=DataflowRunner
```

### 9. Train ML Models

```bash
cd src/ml/pipeline
python pipeline.py \
  --project-id=${GCP_PROJECT} \
  --pipeline-root=gs://${GCP_PROJECT}-cqc-ml-artifacts/pipelines \
  --display-name="cqc-ml-pipeline-$(date +%Y%m%d-%H%M%S)"
```

## Usage

### Making Predictions

Single prediction:

```bash
curl -X POST https://[REGION]-[PROJECT_ID].cloudfunctions.net/cqc-rating-prediction \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [{
      "days_since_last_inspection": 180,
      "num_regulated_activities": 5,
      "region": "London",
      "type": "Residential social care",
      ...
    }]
  }'
```

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_ingestion.py
```

### Local Development

Use the provided docker-compose for local development:

```bash
docker-compose up -d
```

## Monitoring

- **Cloud Logging**: View logs in the GCP Console
- **Cloud Monitoring**: Set up dashboards for system metrics
- **Vertex AI Model Monitoring**: Track model performance over time

## Security Considerations

- All API keys stored in Secret Manager
- Service accounts with minimal required permissions
- VPC Service Controls for enhanced security
- Data encryption at rest and in transit

## Cost Optimization

- Lifecycle policies on Cloud Storage buckets
- Preemptible nodes for training
- Auto-scaling for Cloud Functions
- BigQuery partitioning and clustering

## License

[Your License Here]

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.