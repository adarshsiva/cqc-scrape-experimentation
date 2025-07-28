# CQC Rating Predictor ML System on Google Cloud Platform

## Phase 1: System Architecture Design

### GCP Services Architecture:
- **Data Ingestion**: Cloud Scheduler → Cloud Functions
- **Raw Data Lake**: Cloud Storage (GCS) buckets
- **Data Transformation**: Cloud Dataflow (Apache Beam)
- **Data Warehouse**: BigQuery
- **ML Platform**: Vertex AI
- **Model Serving**: Vertex AI Endpoints
- **Orchestration**: Cloud Composer (Apache Airflow)
- **Security**: Secret Manager for API keys
- **Monitoring**: Cloud Monitoring & Logging

### Data Flow:
1. Cloud Scheduler triggers weekly ingestion
2. Cloud Function fetches from CQC API endpoints
3. Raw JSON stored in GCS bucket
4. Dataflow pipeline processes and transforms data
5. Cleaned data loaded to BigQuery
6. Vertex AI pipeline trains model
7. Model deployed to Vertex AI Endpoint
8. Prediction API via Cloud Function

## Phase 2: Project Structure

```
cqc-rating-predictor/
├── src/
│   ├── ingestion/
│   │   ├── main.py              # Cloud Function for API ingestion
│   │   └── requirements.txt
│   ├── etl/
│   │   ├── dataflow_pipeline.py # Apache Beam pipeline
│   │   ├── transforms.py        # Data transformation logic
│   │   └── requirements.txt
│   ├── ml/
│   │   ├── pipeline/
│   │   │   ├── components.py    # Vertex AI pipeline components
│   │   │   └── pipeline.py      # Main ML pipeline
│   │   ├── features.py          # Feature engineering
│   │   └── requirements.txt
│   ├── prediction/
│   │   ├── main.py              # Prediction API Cloud Function
│   │   └── requirements.txt
│   └── utils/
│       └── common.py            # Shared utilities
├── terraform/                   # Infrastructure as Code
│   ├── main.tf
│   ├── variables.tf
│   └── outputs.tf
├── config/
│   ├── bigquery_schema.json
│   └── vertex_ai_config.yaml
├── tests/
├── docs/
└── README.md
```

## Phase 3: Implementation Plan

### 1. Data Ingestion (Cloud Function)
- Authenticate using Secret Manager for API key
- Handle pagination for both providers and locations endpoints
- Store raw JSON responses in GCS with timestamp partitioning
- Implement retry logic and error handling

### 2. ETL Pipeline (Dataflow)
- Parse JSON and extract key fields:
  - locationId, providerId, name, type
  - currentRatings.overall.rating (target variable)
  - regulatedActivities, inspectionAreas
  - lastInspection.date, registrationDate
  - postalCode, region
- Feature engineering:
  - Time since last inspection
  - Number of regulated activities
  - Service type encodings
  - Regional statistics
- Load to BigQuery with proper schema

### 3. ML Pipeline (Vertex AI)
- Data preparation component
- Feature preprocessing (scaling, encoding)
- Model training with multiple algorithms:
  - XGBoost (baseline)
  - LightGBM
  - AutoML Tabular
- Model evaluation and comparison
- Conditional deployment based on performance

### 4. Prediction API
- Cloud Function to serve predictions
- Input validation
- Feature transformation to match training
- Call Vertex AI Endpoint
- Return predicted rating with confidence

### 5. Deployment & Monitoring
- Terraform scripts for all resources
- Cloud Monitoring dashboards
- Alerting for pipeline failures
- Model performance tracking

## Key Considerations

### API Details:
- Base URL: https://api.cqc.org.uk/public/v1/
- Main endpoints: /providers and /locations
- Requires partnerCode parameter
- Rate limit: 2000 requests/minute with partnerCode
- Authentication via subscription_key

### Target Variable:
- Field: currentRatings.overall.rating
- Values: "Outstanding", "Good", "Requires improvement", "Inadequate"
- Multi-class classification problem

### Feature Engineering Focus:
- Text analysis from inspection reports
- Temporal patterns (inspection frequency, trends)
- Geographic clustering
- Service type combinations
- Historical rating transitions

This comprehensive system will provide automated, scalable ML predictions for CQC ratings with proper MLOps practices on GCP.