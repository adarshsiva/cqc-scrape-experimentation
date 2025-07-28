# CQC Rating Prediction API

This Cloud Function provides a REST API for predicting CQC ratings based on location features.

## Features

- **Single Prediction**: Predict rating for one location
- **Batch Prediction**: Predict ratings for up to 100 locations in a single request
- **Confidence Scores**: Returns probability distribution across all rating categories
- **Input Validation**: Comprehensive validation of all required features
- **Error Handling**: Detailed error messages for debugging
- **Health Check**: Endpoint to verify service availability

## API Endpoints

### POST /predict

Main prediction endpoint that accepts location features and returns predicted ratings.

#### Single Prediction Request
```json
{
    "location_id": "1-234567890",
    "number_of_beds": 45,
    "number_of_locations": 1,
    "inspection_history_length": 3,
    "days_since_last_inspection": 180,
    "ownership_type": "Organisation",
    "service_types": ["Care home service with nursing"],
    "specialisms": ["Dementia"],
    "region": "London",
    "local_authority": "Westminster",
    "constituency": "Westminster North",
    "regulated_activities": ["Accommodation for persons who require nursing or personal care"],
    "service_user_groups": ["Older people"],
    "has_previous_rating": true,
    "previous_rating": "Good",
    "ownership_changed_recently": false,
    "nominated_individual_exists": true
}
```

#### Single Prediction Response
```json
{
    "location_id": "1-234567890",
    "predicted_rating": "Good",
    "confidence_scores": {
        "Inadequate": 0.05,
        "Requires improvement": 0.15,
        "Good": 0.70,
        "Outstanding": 0.10
    },
    "confidence_level": "High",
    "prediction_timestamp": "2024-01-15T10:30:00",
    "model_version": "cqc-rating-predictor"
}
```

#### Batch Prediction Request
```json
{
    "instances": [
        { /* location 1 features */ },
        { /* location 2 features */ },
        // ... up to 100 locations
    ]
}
```

### GET /health

Health check endpoint for monitoring.

## Deployment

### Prerequisites

1. Google Cloud Project with enabled APIs:
   - Cloud Functions
   - Vertex AI
   - Cloud Logging

2. Vertex AI model endpoint deployed

3. Service account with required permissions

### Deploy with Script

```bash
# Set environment variables
export GCP_PROJECT="your-project-id"
export GCP_REGION="europe-west2"
export VERTEX_ENDPOINT_ID="your-endpoint-id"

# Make script executable
chmod +x deploy.sh

# Deploy the function
./deploy.sh
```

### Deploy with gcloud CLI

```bash
gcloud functions deploy cqc-prediction-api \
    --gen2 \
    --runtime=python311 \
    --region=europe-west2 \
    --source=. \
    --entry-point=predict \
    --trigger-http \
    --allow-unauthenticated \
    --memory=512MB \
    --timeout=60s \
    --set-env-vars="VERTEX_ENDPOINT_ID=your-endpoint-id"
```

## Testing

Use the provided test script to validate the deployment:

```bash
# Update FUNCTION_URL in test_prediction.py
python test_prediction.py
```

## Required Features

The API expects the following features for each location:

| Feature | Type | Description |
|---------|------|-------------|
| number_of_beds | int | Total bed capacity |
| number_of_locations | int | Number of locations under provider |
| inspection_history_length | int | Number of previous inspections |
| days_since_last_inspection | int | Days elapsed since last inspection |
| ownership_type | str | "Individual", "Organisation", or "Partnership" |
| service_types | list | Types of services provided |
| specialisms | list | Medical specialisms offered |
| region | str | Geographic region |
| local_authority | str | Local authority name |
| constituency | str | Parliamentary constituency |
| regulated_activities | list | CQC regulated activities |
| service_user_groups | list | Demographics served |
| has_previous_rating | bool | Whether location has been rated before |
| previous_rating | str | Previous CQC rating (if exists) |
| ownership_changed_recently | bool | Recent ownership change flag |
| nominated_individual_exists | bool | Presence of nominated individual |

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Successful prediction
- `400`: Invalid input data
- `405`: Method not allowed (only POST accepted)
- `500`: Server error (model unavailable, etc.)

Error responses include detailed messages:

```json
{
    "error": "Validation error: Missing required feature: number_of_beds"
}
```

## Monitoring

The function logs all requests and errors to Cloud Logging. Monitor via:

```bash
gcloud functions logs read cqc-prediction-api --region=europe-west2
```

## Performance

- Single prediction: ~100-200ms latency
- Batch prediction (100 instances): ~500-1000ms latency
- Auto-scales from 1 to 100 instances based on load
- 60-second timeout for complex requests