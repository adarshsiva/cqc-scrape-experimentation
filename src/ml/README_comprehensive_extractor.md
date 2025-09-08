# Comprehensive CQC API Data Extractor

This service implements the Enhanced CQC API Data Extraction as specified in **plan.md Phase 1.1**. It extracts data from all 12 CQC API endpoints with comprehensive feature engineering for ML model training.

## Overview

The Comprehensive CQC Extractor fetches data from all available CQC API endpoints:

1. **Get Location By Id** → Detailed facility information
2. **Get Locations** → Bulk location listings  
3. **Get Provider By Id** → Provider-level data
4. **Get Providers** → Bulk provider listings
5. **Get Location AssessmentServiceGroups** → Service complexity metrics
6. **Get Provider AssessmentServiceGroups** → Provider service patterns
7. **Get Location Inspection Areas** → Domain-specific ratings (Safe, Effective, Caring, etc.)
8. **Get Provider Inspection Areas By Location Id** → Historical inspection data
9. **Get Provider Inspection Areas By Provider Id** → Provider-level inspection patterns
10. **Get Inspection Areas** → Rating methodology data
11. **Get Reports** → Detailed inspection reports  
12. **Get Changes Within Timeframe** → Recent updates

## Features

- **Parallel Processing**: Configurable number of parallel workers for efficient data extraction
- **Rate Limiting**: Built-in rate limiting to respect API quotas (1800 requests/hour by default)
- **Comprehensive Feature Engineering**: Implements the feature extraction logic from plan.md SQL queries
- **Cloud Run Jobs Compatible**: Designed for scalable execution on Google Cloud Platform
- **Error Handling**: Robust error handling with retry logic and graceful degradation
- **Progress Tracking**: Tracks processed items to support resumable execution
- **Cloud Storage Integration**: Saves all data to structured Cloud Storage buckets
- **BigQuery Integration**: Exports processed features to BigQuery for ML training

## Environment Variables

Configure the extractor behavior using these environment variables (as specified in plan.md):

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `ENDPOINTS` | Comma-separated list of endpoints to fetch | `locations,providers,inspection_areas` | `locations,providers,inspection_areas,reports,assessment_groups` |
| `MAX_LOCATIONS` | Maximum number of locations to process | `50000` | `10000` |
| `INCLUDE_HISTORICAL` | Include historical inspection data | `true` | `true`/`false` |
| `FETCH_REPORTS` | Fetch detailed inspection reports | `true` | `true`/`false` |
| `RATE_LIMIT` | Requests per hour | `1800` | `1800` |
| `PARALLEL_WORKERS` | Number of parallel workers | `10` | `5` |
| `GCP_PROJECT` | Google Cloud Project ID | `machine-learning-exp-467008` | `your-project-id` |

## Deployment

### Prerequisites

1. **Google Cloud Project** with the following APIs enabled:
   - Cloud Run API
   - Cloud Build API
   - Cloud Storage API  
   - BigQuery API
   - Secret Manager API
   - Artifact Registry API

2. **CQC API Key** stored in Secret Manager as `cqc-subscription-key`

3. **Required IAM permissions** for the deployment service account

### Quick Deployment

```bash
# 1. Clone the repository and navigate to the ML directory
cd src/ml

# 2. Set your project ID
export GCP_PROJECT="your-project-id"

# 3. Run the deployment script
./deploy_comprehensive_extractor.sh
```

The deployment script will:
- Enable required APIs
- Create necessary service accounts and IAM bindings
- Set up Cloud Storage buckets
- Create BigQuery dataset
- Build and deploy the Docker image
- Create the Cloud Run Job

### Manual Deployment

If you prefer manual deployment:

```bash
# Build the image
gcloud builds submit --config cloudbuild-comprehensive-extractor.yaml

# Deploy the Cloud Run Job
gcloud run jobs replace deploy-comprehensive-extractor.yaml --region=europe-west2
```

## Usage

### Basic Execution

Execute the comprehensive extraction with default settings:

```bash
gcloud run jobs execute cqc-comprehensive-extractor \
  --region=europe-west2 \
  --project=your-project-id \
  --wait
```

### Custom Configuration

Execute with specific parameters (as shown in plan.md):

```bash
gcloud run jobs execute cqc-comprehensive-extractor \
  --region=europe-west2 \
  --project=your-project-id \
  --update-env-vars="
    ENDPOINTS=locations,providers,inspection_areas,reports,assessment_groups,
    MAX_LOCATIONS=50000,
    INCLUDE_HISTORICAL=true,
    FETCH_REPORTS=true,
    RATE_LIMIT=1800,
    PARALLEL_WORKERS=10" \
  --task-timeout=21600 \
  --wait
```

### Monitoring Execution

Check the status of running executions:

```bash
# List executions
gcloud run jobs executions list \
  --job=cqc-comprehensive-extractor \
  --region=europe-west2 \
  --project=your-project-id

# Get detailed execution info
gcloud run jobs executions describe EXECUTION_NAME \
  --region=europe-west2 \
  --project=your-project-id

# View logs
gcloud logging read "resource.type=cloud_run_job AND resource.labels.job_name=cqc-comprehensive-extractor" \
  --project=your-project-id \
  --limit=100
```

## Output Data Structure

The extractor saves data to Cloud Storage in the following structure:

```
{project-id}-cqc-raw-data/
├── comprehensive/
│   ├── locations/
│   │   ├── all_locations_YYYYMMDD_HHMMSS.json
│   │   └── details/YYYYMMDD/{locationId}.json
│   ├── providers/  
│   │   ├── all_providers_YYYYMMDD_HHMMSS.json
│   │   └── details/YYYYMMDD/{providerId}.json
│   ├── inspection_areas/
│   │   ├── locations/YYYYMMDD/{locationId}.json
│   │   └── metadata/all_inspection_areas_YYYYMMDD_HHMMSS.json
│   ├── assessment_groups/
│   │   ├── locations/YYYYMMDD/{locationId}.json
│   │   └── providers/YYYYMMDD/{providerId}.json
│   ├── provider_inspection_areas/
│   │   ├── by_location/YYYYMMDD/{locationId}.json
│   │   └── by_provider/YYYYMMDD/{providerId}.json
│   ├── reports/YYYYMMDD/{locationId}.json
│   └── changes/changes_YYYY-MM-DD_to_YYYY-MM-DD.json

{project-id}-cqc-processed/
├── metadata/
│   └── comprehensive_processed_items.json
└── extraction_summary_YYYYMMDD_HHMMSS.json
```

## Feature Engineering

The extractor implements comprehensive feature engineering as specified in the plan.md SQL queries, extracting:

### Temporal Features
- `days_since_inspection`: Days since last inspection
- `days_since_registration`: Days since registration  
- `inspection_overdue_risk`: Binary risk indicator

### Operational Features  
- `numberOfBeds`: Bed capacity
- `facility_size_numeric`: Categorized facility size
- `service_complexity`: Number of regulated activities
- `occupancy_rate`: Current occupancy percentage

### Quality Indicators
- `overall_rating_numeric`: Target variable (1-4 scale)
- `safe_rating`, `effective_rating`, `caring_rating`, `responsive_rating`, `well_led_rating`: Domain ratings
- `historical_avg_rating`: Historical performance
- `rating_volatility`: Performance consistency

### Provider Context
- `provider_location_count`: Multi-location providers
- `provider_avg_rating`: Provider-level performance
- `provider_rating_consistency`: Rating variance

### Geographic Features
- `region`: Geographic region
- `localAuthority`: Local authority area
- `regional_risk_rate`: Regional performance patterns

### Interaction Features
- `complexity_scale_interaction`: Service complexity × provider scale
- `inspection_regional_risk`: Inspection risk × regional patterns

## BigQuery Output

Processed features are saved to BigQuery table: `{project}.cqc_data.ml_training_features_comprehensive`

This table structure matches the comprehensive feature set required for ML model training as described in plan.md.

## Performance

- **Scalability**: Handles up to 50,000 locations efficiently
- **Rate Limiting**: Respects CQC API limits (1800 requests/hour)  
- **Parallel Processing**: Configurable workers (default: 10)
- **Resumable**: Tracks processed items to support interrupted runs
- **Resource Efficient**: Optimized memory usage and connection pooling

## Troubleshooting

### Common Issues

1. **API Key Missing**
   ```
   ERROR: No CQC API key available
   ```
   Solution: Create the secret in Secret Manager:
   ```bash
   echo "your-api-key" | gcloud secrets create cqc-subscription-key --data-file=-
   ```

2. **Rate Limiting**
   ```
   WARNING: Rate limit reached, waiting X seconds
   ```
   Solution: This is normal behavior. Adjust `RATE_LIMIT` environment variable if needed.

3. **Insufficient Permissions**
   ```
   ERROR: Permission denied for bucket operations
   ```
   Solution: Ensure the service account has required IAM roles (see deployment script).

4. **Job Timeout**
   ```
   ERROR: Job execution timed out
   ```  
   Solution: Increase `--task-timeout` or reduce `MAX_LOCATIONS` for shorter runs.

### Performance Tuning

- **Reduce `PARALLEL_WORKERS`** if hitting rate limits frequently
- **Increase `RATE_LIMIT`** if you have higher API quotas
- **Set `FETCH_REPORTS=false`** to speed up extraction (reports are large)
- **Use smaller `MAX_LOCATIONS`** for testing and development

### Monitoring

Monitor job execution through:
- **Cloud Console**: Cloud Run → Jobs → cqc-comprehensive-extractor
- **Logs**: Cloud Logging with filter `resource.type=cloud_run_job`
- **Storage**: Check bucket contents for progress tracking
- **BigQuery**: Query the output table for processed records

## Integration with ML Pipeline

This extractor provides the comprehensive training dataset required for the ML pipeline described in plan.md:

1. **Training Data**: Comprehensive CQC features → Real inspection ratings & outcomes
2. **Feature Compatibility**: Features align with dashboard prediction inputs
3. **Model Training**: Ready for XGBoost, LightGBM, and AutoML training
4. **Production Ready**: Scalable and robust for production data pipelines

The extracted features directly support the unified ML model training approach outlined in plan.md Phase 3.