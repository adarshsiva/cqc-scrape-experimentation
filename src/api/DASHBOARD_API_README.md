# Dashboard Prediction API Documentation

## Overview

The Dashboard Prediction API provides real-time CQC (Care Quality Commission) rating predictions using operational data from care home dashboards. This service implements Phase 3.2 of the CQC Rating Predictor system, enabling instant predictions based on current dashboard metrics.

## Architecture

```
Dashboard Data → Feature Extraction → Feature Alignment → ML Prediction → Recommendations
     (EAV)            (Operational)         (CQC Space)      (Vertex AI)       (Actionable)
```

### Key Components

1. **DashboardFeatureExtractor**: Extracts ML features from dashboard operational data
2. **FeatureAlignmentService**: Transforms dashboard features to match CQC training feature space
3. **ModelPredictionService**: Loads trained models and generates predictions with explanations
4. **Authentication Middleware**: Secures API access with token-based authentication

## API Endpoints

### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "service": "CQC Dashboard Prediction API",
  "timestamp": "2024-01-15T10:30:00Z",
  "version": "v1.0"
}
```

### Real-time CQC Prediction

```http
GET /api/cqc-prediction/dashboard/{care_home_id}
```

**Headers:**
- `Authorization: Bearer {api_token}` (required)
- `X-Client-ID: {client_id}` (required)

**Parameters:**
- `care_home_id`: Unique identifier for the care home entity

**Response:**
```json
{
  "care_home_id": "care_home_123",
  "prediction": {
    "predicted_rating": 3,
    "predicted_rating_text": "Good",
    "confidence_score": 0.85,
    "risk_level": "Low"
  },
  "contributing_factors": {
    "top_positive_factors": [
      {
        "factor": "Care quality indicator",
        "impact": 0.8,
        "value": 0.9,
        "interpretation": "Good care quality indicator (90%)"
      }
    ],
    "top_risk_factors": [
      {
        "factor": "Incident frequency",
        "impact": 0.6,
        "value": 0.2,
        "interpretation": "Low incident frequency (20%)"
      }
    ],
    "operational_score": 0.8,
    "quality_score": 0.7,
    "risk_score": 0.2
  },
  "recommendations": [
    {
      "category": "Continuous Improvement",
      "priority": "Low",
      "action": "Continue current good practices and consider expansion of successful programs",
      "timeline": "Ongoing"
    }
  ],
  "data_freshness": {
    "last_updated": "2024-01-15T10:30:00Z",
    "data_coverage": 0.85
  }
}
```

## Feature Mapping

The API transforms dashboard operational metrics into CQC training feature space:

### Operational Metrics
- `bed_capacity` → Direct mapping to facility size
- `occupancy_rate` → Direct mapping to occupancy
- `avg_care_complexity` → Service complexity score
- `facility_size_numeric` → Categorical facility size

### Risk Indicators
- `incident_frequency_risk` → Incident-based risk score
- `falls_risk` → Falls incident rate
- `medication_risk` → Medication error rate
- `safeguarding_risk` → Safeguarding concerns

### Care Quality
- `care_plan_compliance` → Care plan review compliance
- `care_goal_achievement` → Care goal success rate
- `staff_compliance_score` → Staff performance metrics

### Temporal Features
- `days_since_last_incident` → Incident timing
- `operational_stability` → Operational consistency
- `care_plan_review_frequency` → Review patterns

## Authentication

The API uses Bearer token authentication with API keys stored in Google Secret Manager.

### Setup Authentication

1. **Create API Secret:**
```bash
# Generate API key
API_KEY=$(openssl rand -hex 32)

# Store in Secret Manager
echo -n "$API_KEY" | gcloud secrets create dashboard-api-key --data-file=-
```

2. **Use in Requests:**
```bash
curl -H "Authorization: Bearer $API_KEY" \
     -H "X-Client-ID: your_client_id" \
     https://your-service-url/api/cqc-prediction/dashboard/care_home_123
```

## Deployment

### Prerequisites

- Google Cloud Project with enabled APIs:
  - Cloud Run
  - Cloud Build
  - Vertex AI
  - BigQuery
  - Secret Manager
- Docker installed locally
- gcloud CLI configured

### Quick Deploy

```bash
# Set environment variables
export GCP_PROJECT="your-project-id"
export GCP_REGION="europe-west2"
export VERTEX_ENDPOINT_ID="your-endpoint-id"

# Deploy using the provided script
./src/api/deploy-dashboard-api.sh
```

### Manual Deployment

```bash
# Build and deploy with Cloud Build
gcloud builds submit \
    --config=src/api/cloudbuild-deploy-dashboard-api.yaml \
    --substitutions=_REGION=europe-west2,_ENDPOINT_ID=your-endpoint-id \
    .
```

### Local Development

```bash
# Install dependencies
pip install -r src/api/requirements-dashboard.txt

# Set environment variables
export GCP_PROJECT="your-project-id"
export VERTEX_ENDPOINT_ID="your-endpoint-id"
export ENABLE_AUTH="false"  # Disable auth for local testing

# Run locally
python src/api/dashboard_prediction_service.py
```

## Configuration

### Environment Variables

- `GCP_PROJECT`: Google Cloud Project ID
- `GCP_REGION`: Deployment region (default: europe-west2)
- `VERTEX_ENDPOINT_ID`: Vertex AI endpoint for predictions
- `MODEL_BUCKET`: GCS bucket for model artifacts
- `API_SECRET_NAME`: Secret Manager secret name for API keys
- `ENABLE_AUTH`: Enable/disable authentication (default: true)

### Cloud Run Configuration

- **Memory**: 2Gi
- **CPU**: 1 vCPU
- **Timeout**: 300s
- **Concurrency**: 10 requests per instance
- **Max Instances**: 10

## Error Handling

### Error Response Format

```json
{
  "error": "Error type description",
  "error_type": "ERROR_CODE",
  "message": "Detailed error message",
  "care_home_id": "care_home_123",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

### Error Types

- `AUTHENTICATION_ERROR` (401): Invalid or missing API token
- `PREDICTION_ERROR` (500): Model prediction failed
- `FEATURE_EXTRACTION_ERROR` (500): Dashboard data extraction failed
- `VALIDATION_ERROR` (400): Invalid request parameters

## Monitoring and Logging

### Cloud Logging

All requests and errors are logged to Google Cloud Logging with structured logs:

```json
{
  "severity": "INFO",
  "message": "Prediction completed successfully",
  "care_home_id": "care_home_123",
  "client_id": "client_456",
  "prediction_rating": 3,
  "confidence": 0.85,
  "processing_time_ms": 1250
}
```

### Health Monitoring

- Health check endpoint: `/health`
- Monitoring dashboards available in Cloud Console
- Alerting configured for error rates and latency

## Data Privacy and Security

- All data transmission encrypted in transit (HTTPS)
- API tokens stored securely in Secret Manager
- No personal data logged or stored
- Compliant with data protection regulations

## Integration Examples

### Python Client

```python
import requests

def get_cqc_prediction(care_home_id, api_token, client_id):
    """Get CQC prediction for a care home."""
    
    headers = {
        'Authorization': f'Bearer {api_token}',
        'X-Client-ID': client_id
    }
    
    url = f'https://your-service-url/api/cqc-prediction/dashboard/{care_home_id}'
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f'API error: {response.status_code} - {response.text}')

# Usage
prediction = get_cqc_prediction('care_home_123', 'your_api_key', 'your_client_id')
print(f"Predicted rating: {prediction['prediction']['predicted_rating_text']}")
```

### JavaScript Client

```javascript
async function getCQCPrediction(careHomeId, apiToken, clientId) {
    const response = await fetch(
        `https://your-service-url/api/cqc-prediction/dashboard/${careHomeId}`,
        {
            headers: {
                'Authorization': `Bearer ${apiToken}`,
                'X-Client-ID': clientId
            }
        }
    );
    
    if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
    }
    
    return await response.json();
}

// Usage
getCQCPrediction('care_home_123', 'your_api_key', 'your_client_id')
    .then(prediction => {
        console.log(`Predicted rating: ${prediction.prediction.predicted_rating_text}`);
        console.log(`Confidence: ${prediction.prediction.confidence_score}`);
    })
    .catch(error => console.error('Prediction failed:', error));
```

## Performance Characteristics

- **Latency**: < 2 seconds for typical predictions
- **Throughput**: Up to 100 requests per second
- **Availability**: 99.9% SLA with automatic scaling
- **Data freshness**: Real-time dashboard data integration

## Support and Troubleshooting

### Common Issues

1. **Authentication Failed (401)**
   - Verify API token is correct
   - Check Secret Manager configuration
   - Ensure proper Authorization header format

2. **Feature Extraction Failed (500)**
   - Verify dashboard database connectivity
   - Check care home ID exists in system
   - Review BigQuery dataset permissions

3. **Model Prediction Failed (500)**
   - Verify Vertex AI endpoint is deployed
   - Check model artifacts in Cloud Storage
   - Review service account permissions

### Debug Mode

Enable detailed logging for troubleshooting:

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export ENABLE_DEBUG_LOGGING="true"
```

### Contact Information

For technical support or questions about the Dashboard Prediction API:
- Create an issue in the project repository
- Check the project documentation in `/documentation`
- Review Cloud Console logs and monitoring dashboards

## Changelog

### v1.0 (2024-01-15)
- Initial release with real-time predictions
- Dashboard feature extraction and alignment
- Authentication and security implementation
- Cloud Run deployment configuration
- Comprehensive error handling and logging