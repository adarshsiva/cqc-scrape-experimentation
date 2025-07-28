# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a CQC (Care Quality Commission) Rating Predictor ML System built on Google Cloud Platform. The project implements a comprehensive data pipeline and machine learning system to predict CQC ratings for healthcare providers.

## Project Documentation

Comprehensive documentation for all GCP services and tools used in this project is available in the `/documentation` directory:

### GCP Services Documentation
- **Cloud Functions** (`/documentation/gcp/cloud-functions.md`) - Serverless functions for API ingestion
- **Cloud Storage** (`/documentation/gcp/cloud-storage.md`) - Object storage for raw data and model artifacts
- **Cloud Dataflow** (`/documentation/gcp/cloud-dataflow.md`) - Apache Beam pipelines for ETL processing
- **BigQuery** (`/documentation/gcp/bigquery.md`) - Data warehouse for analytics and ML features
- **Vertex AI** (`/documentation/gcp/vertex-ai.md`) - ML platform for model training and serving
- **Cloud Composer** (`/documentation/gcp/cloud-composer.md`) - Workflow orchestration with Apache Airflow
- **Secret Manager** (`/documentation/gcp/secret-manager.md`) - Secure storage for API keys and credentials
- **Cloud Scheduler** (`/documentation/gcp/cloud-scheduler.md`) - Cron job scheduling for automated tasks

### Infrastructure as Code
- **Terraform GCP Provider** (`/documentation/terraform/gcp-provider.md`) - IaC configurations for all resources

## Architecture Overview

Based on the plan.md file, the system architecture includes:

### Data Flow
1. **Ingestion**: Cloud Scheduler → Cloud Functions → CQC API
2. **Storage**: Raw JSON data in Cloud Storage buckets
3. **Processing**: Cloud Dataflow pipelines for transformation
4. **Analytics**: BigQuery for data warehousing
5. **ML Pipeline**: Vertex AI for training and deployment
6. **Serving**: Vertex AI Endpoints for predictions
7. **Orchestration**: Cloud Composer for workflow management

### Key Components
- **API Integration**: Fetches data from CQC API endpoints (/providers and /locations)
- **ETL Pipeline**: Processes and transforms raw data for ML features
- **ML Models**: XGBoost, LightGBM, and AutoML implementations
- **Prediction API**: Cloud Function serving model predictions

## Development Setup

### Prerequisites
```bash
# Install required Python packages
pip install apache-beam[gcp]
pip install google-cloud-storage
pip install google-cloud-bigquery
pip install google-cloud-aiplatform
pip install google-cloud-secret-manager
pip install google-cloud-scheduler
```

### Environment Variables
```bash
export GCP_PROJECT=<your-project-id>
export GCP_REGION=<your-region>
export GCS_BUCKET=<your-bucket-name>
```

## Common Tasks

### Data Ingestion
- Review `/documentation/gcp/cloud-functions.md` for implementing CQC API ingestion
- Use Secret Manager for storing API keys securely

### ETL Pipeline Development
- Reference `/documentation/gcp/cloud-dataflow.md` for Apache Beam pipeline patterns
- See example ETL pipeline structure in plan.md

### ML Model Development
- Use `/documentation/gcp/vertex-ai.md` for training and deployment guidance
- Implement feature engineering as specified in plan.md

### Infrastructure Deployment
- Follow `/documentation/terraform/gcp-provider.md` for deploying all GCP resources
- Use provided Terraform templates for consistent infrastructure

## Best Practices

1. **Security**: Always use Secret Manager for sensitive data (API keys, passwords)
2. **Cost Optimization**: Enable lifecycle policies on Cloud Storage buckets
3. **Monitoring**: Set up Cloud Monitoring dashboards for all services
4. **Testing**: Implement unit tests for all data transformations
5. **Documentation**: Keep this file updated with new patterns and decisions
6. **Version Control**: After making significant codebase changes, always commit and push to git:
   - Use descriptive commit messages that explain what was changed and why
   - Include the 🤖 emoji and co-author attribution in commits
   - Push changes to the remote repository to ensure work is backed up
   - Example workflow:
     ```bash
     git add -A
     git commit -m "feat: Add new ML model training pipeline
     
     - Implemented XGBoost and LightGBM models
     - Added feature engineering module
     - Updated prediction service endpoint
     
     🤖 Generated with Claude Code
     
     Co-Authored-By: Claude <noreply@anthropic.com>"
     git push
     ```