# Proactive Risk Assessment API

This API provides risk assessment capabilities for CQC-registered healthcare locations, identifying those at risk of rating downgrades.

## Overview

The Proactive Risk Assessment API analyzes multiple factors to predict the likelihood of a location receiving a lower rating in their next CQC inspection. It provides:

- Risk scores (0-100%)
- Risk levels (HIGH/MEDIUM/LOW)
- Top risk factors contributing to the assessment
- Actionable recommendations for improvement
- Confidence levels for assessments

## API Endpoints

### Health Check
```
GET /health
```
Returns the health status of the service.

### Single Location Assessment
```
POST /assess-risk
Content-Type: application/json

{
    "locationId": "1-123456789",
    "locationName": "Care Home Name",
    "staff_vacancy_rate": 0.25,
    "staff_turnover_rate": 0.35,
    "inspection_days_since_last": 450,
    "total_complaints": 2,
    "safe_key_questions_yes_ratio": 0.6,
    "effective_key_questions_yes_ratio": 0.7,
    "caring_key_questions_yes_ratio": 0.8,
    "responsive_key_questions_yes_ratio": 0.75,
    "well_led_key_questions_yes_ratio": 0.5,
    // ... other features
}
```

Response:
```json
{
    "locationId": "1-123456789",
    "locationName": "Care Home Name",
    "riskScore": 72.5,
    "riskLevel": "HIGH",
    "topRiskFactors": [
        {
            "factor": "staff_vacancy_rate",
            "currentValue": 0.25,
            "riskContribution": 0.15
        }
    ],
    "recommendations": [
        "Reduce staff vacancy rate: Currently at 25.0%. Focus on recruitment and retention strategies."
    ],
    "assessmentDate": "2024-01-29T10:30:00",
    "confidence": "HIGH"
}
```

### Batch Assessment
```
POST /batch-assess
Content-Type: application/json

{
    "locations": [
        { /* location 1 data */ },
        { /* location 2 data */ },
        // ... up to 100 locations
    ]
}
```

Response includes summary statistics and individual assessments.

### Risk Thresholds
```
GET /risk-thresholds
```
Returns the current risk level definitions and thresholds.

## Risk Levels

- **HIGH (70-100%)**: Immediate attention required. High probability of rating downgrade.
- **MEDIUM (40-69%)**: Monitor closely and implement improvements.
- **LOW (0-39%)**: Continue current practices with regular monitoring.

## Local Development

1. Install dependencies:
```bash
pip install -r requirements_proactive.txt
```

2. Set environment variables:
```bash
export GCP_PROJECT=machine-learning-exp-467008
export MODEL_BUCKET=machine-learning-exp-467008-cqc-ml-artifacts
export MODEL_PATH=models/proactive/model_package.pkl
```

3. Run the application:
```bash
python proactive_predictor.py
```

4. Test the API:
```bash
python test_proactive_api.py
```

## Deployment

### Using Cloud Build (Recommended)
```bash
gcloud builds submit --config cloudbuild_proactive.yaml
```

### Manual Deployment
```bash
./deploy_proactive.sh
```

### Docker Build
```bash
docker build -f Dockerfile.proactive -t proactive-risk-assessment .
docker run -p 8080:8080 proactive-risk-assessment
```

## Model Information

The API uses an ensemble of machine learning models trained on historical CQC inspection data:
- XGBoost
- LightGBM
- Optional: AutoML models

The models are stored in Google Cloud Storage and loaded at startup.

## Feature Requirements

The following features are used for risk assessment:

### Staffing Metrics
- `staff_vacancy_rate`: Proportion of vacant positions
- `staff_turnover_rate`: Annual staff turnover rate

### Inspection History
- `inspection_days_since_last`: Days since last inspection
- `current_rating`: Current CQC rating
- `enforcement_actions`: Number of enforcement actions

### Quality Indicators
- `total_complaints`: Number of complaints
- `*_key_questions_yes_ratio`: Compliance ratios for each CQC domain
  - Safe
  - Effective
  - Caring
  - Responsive
  - Well-led

### Provider Context
- `provider_rating`: Overall provider rating
- `provider_good_outstanding_ratio`: Proportion of provider's locations rated Good/Outstanding

## Monitoring

Monitor the service using:
- Cloud Run metrics
- Cloud Logging for application logs
- Custom metrics for risk assessments

## Security

- API requires authentication in production
- Model artifacts are securely stored in Cloud Storage
- All data is processed in memory without persistence