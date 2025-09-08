# CQC Comprehensive ETL Pipeline

This directory contains an enhanced Apache Beam/Dataflow ETL pipeline for processing CQC (Care Quality Commission) data with advanced features for care home analysis and machine learning.

## Features

### Core Capabilities
- **Robust Data Processing**: Handles JSON data from CQC APIs with comprehensive error handling
- **Care Home Focus**: Specialized feature extraction for care home analysis
- **Data Quality**: Advanced validation with configurable rules and quality scoring
- **Dual Processing**: Both batch and streaming processing support
- **Multiple Outputs**: Writes to multiple BigQuery tables optimized for different use cases

### Enhanced Features
- **Care Home Classification**: Automatically identifies and classifies care homes
- **Risk Assessment**: Calculates risk scores based on ratings, inspection history, and compliance
- **Advanced Metrics**: Derives 30+ features for machine learning including temporal, geographic, and operational metrics
- **Data Quality Monitoring**: Tracks data issues and assigns quality scores
- **Dead Letter Queue**: Captures and stores failed records for analysis

## Architecture

```
Raw CQC Data (GCS) → Apache Beam Pipeline → BigQuery Tables
                              ↓
                    Error Records → DLQ Table
```

### Output Tables

1. **`locations_complete`** - All processed locations with full feature set
2. **`care_homes`** - Filtered care homes with specialized features
3. **`processing_errors`** - Dead letter queue for failed records

### Generated Views

1. **`care_homes_risk_view`** - Risk assessment and monitoring status
2. **`regional_care_home_stats`** - Regional statistics and aggregations
3. **`ml_features_view`** - Clean feature set for machine learning

## Files

- **`dataflow_etl_complete.py`** - Main enhanced ETL pipeline
- **`dataflow_pipeline.py`** - Original basic ETL pipeline
- **`transforms.py`** - Shared transformation functions
- **`cloudbuild-dataflow.yaml`** - Cloud Build deployment configuration
- **`setup.py`** - Package setup for Dataflow workers
- **`requirements.txt`** - Python dependencies

## Deployment

### Using Cloud Build (Recommended)

```bash
# Deploy the complete ETL pipeline
gcloud builds submit --config=cloudbuild-dataflow.yaml \
  --project=your-project-id \
  --region=europe-west2
```

This will:
1. Create necessary BigQuery datasets and GCS buckets
2. Install dependencies and package the pipeline
3. Run the ETL pipeline on Dataflow
4. Create analytical views in BigQuery
5. Set up monitoring and lifecycle policies

### Manual Deployment

```bash
# Install dependencies
pip install -r requirements.txt

# Run batch pipeline
python dataflow_etl_complete.py \
  --project-id=your-project-id \
  --dataset-id=cqc_processed \
  --temp-location=gs://your-project-temp/tmp \
  --input-pattern="gs://your-bucket/raw/locations/*.json" \
  --runner=DataflowRunner \
  --region=europe-west2

# Run streaming pipeline
python dataflow_etl_complete.py \
  --project-id=your-project-id \
  --dataset-id=cqc_processed \
  --temp-location=gs://your-project-temp/tmp \
  --pubsub-subscription=projects/your-project/subscriptions/cqc-updates \
  --streaming \
  --runner=DataflowRunner \
  --region=europe-west2
```

## Configuration Options

### Pipeline Arguments

| Argument | Required | Description | Default |
|----------|----------|-------------|---------|
| `--project-id` | Yes | GCP Project ID | - |
| `--dataset-id` | Yes | BigQuery dataset for output | - |
| `--temp-location` | Yes | GCS temp location | - |
| `--input-pattern` | Yes* | GCS input file pattern | - |
| `--pubsub-subscription` | Yes* | Pub/Sub subscription for streaming | - |
| `--region` | No | GCP region | `europe-west2` |
| `--runner` | No | Pipeline runner | `DataflowRunner` |
| `--streaming` | No | Enable streaming mode | `False` |
| `--job-name` | No | Custom job name | Auto-generated |
| `--num-workers` | No | Initial worker count | `2` |
| `--max-num-workers` | No | Maximum worker count | `10` |
| `--machine-type` | No | Worker machine type | `n1-standard-2` |
| `--disk-size-gb` | No | Worker disk size | `50` |

*Either `--input-pattern` (batch) or `--pubsub-subscription` (streaming) is required.

### Data Quality Configuration

The pipeline includes configurable data quality validation:

```python
validation_config = {
    'required_fields': ['location_id', 'provider_id'],
    'valid_rating_values': ['Outstanding', 'Good', 'Requires improvement', 'Inadequate'],
    'valid_registration_statuses': ['Registered', 'Deregistered', 'Application'],
    'postal_code_pattern': r'^[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][ABD-HJLNP-UW-Z]{2}$',
    'max_days_since_inspection': 3650,  # 10 years
    'min_registration_year': 1990
}
```

## Output Schema

### Core Fields
- `location_id`, `provider_id`, `location_name`, `provider_name`
- `postal_code`, `region`, `local_authority`
- `registration_status`, `registration_date`, `deregistration_date`
- `overall_rating`, `safe_rating`, `effective_rating`, `caring_rating`, `well_led_rating`, `responsive_rating`

### Care Home Features
- `is_care_home`, `care_home_type`, `bed_capacity_category`
- `specializes_in_dementia`, `specializes_in_nursing`, `provides_end_of_life_care`
- `accepts_mental_health`, `accepts_learning_disabilities`

### Risk and Compliance
- `risk_score`, `compliance_trend`, `inspection_frequency_category`
- `has_inadequate_rating`, `rating_consistency_score`

### Temporal Features
- `days_since_last_inspection`, `days_since_registration`
- `registration_year`, `last_inspection_year`, `inspection_to_registration_ratio`

### Advanced Metrics
- `service_complexity_score`, `regional_care_home_density`
- `urban_rural_indicator`, `theoretical_staff_ratio`

### Data Quality
- `data_quality_issues`, `data_quality_score`, `validation_timestamp`

## Monitoring and Alerting

### Pipeline Metrics

The pipeline emits custom metrics to Cloud Monitoring:

- **Processing Metrics**: `transforms/parse_success`, `transforms/parse_errors`
- **Quality Metrics**: `quality/validation_passed`, `quality/data_issues`
- **Feature Metrics**: `features/care_homes_processed`, `features/extraction_errors`

### Monitoring Queries

```sql
-- Check recent processing statistics
SELECT 
  COUNT(*) as total_records,
  COUNT(CASE WHEN error IS NOT NULL THEN 1 END) as error_count,
  AVG(data_quality_score) as avg_quality_score,
  COUNT(CASE WHEN is_care_home THEN 1 END) as care_home_count
FROM `project.dataset.locations_complete`
WHERE processing_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)

-- Identify data quality issues
SELECT 
  issue,
  COUNT(*) as count
FROM `project.dataset.locations_complete`,
UNNEST(data_quality_issues) as issue
WHERE processing_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 DAY)
GROUP BY issue
ORDER BY count DESC
```

## Best Practices

### Performance Optimization
1. **Worker Configuration**: Adjust `num-workers` and `machine-type` based on data volume
2. **Batch Size**: Process data in reasonable batch sizes (1000-10000 records per file)
3. **Partitioning**: Use BigQuery partitioning on `processing_timestamp`
4. **Clustering**: Cluster tables on frequently filtered columns (`region`, `overall_rating`)

### Data Quality
1. **Validation Rules**: Customize validation configuration for your data requirements
2. **Error Handling**: Monitor the dead letter queue for systematic issues
3. **Quality Scoring**: Use quality scores to filter data for machine learning
4. **Regular Audits**: Review data quality metrics regularly

### Cost Management
1. **Lifecycle Policies**: Set up GCS lifecycle policies for temp data (7 days)
2. **BigQuery Slots**: Use on-demand pricing for variable workloads
3. **Worker Autoscaling**: Enable autoscaling to optimize worker costs
4. **Resource Monitoring**: Monitor Dataflow job costs and optimize accordingly

## Troubleshooting

### Common Issues

1. **Pipeline Fails to Start**
   - Check GCS bucket permissions
   - Verify BigQuery dataset exists
   - Ensure Dataflow API is enabled

2. **High Error Rate**
   - Check input data format
   - Review validation configuration
   - Monitor dead letter queue

3. **Poor Performance**
   - Increase worker count
   - Use larger machine types
   - Check for data skew

4. **BigQuery Write Errors**
   - Verify schema compatibility
   - Check table permissions
   - Monitor quota limits

### Debug Commands

```bash
# Check pipeline status
gcloud dataflow jobs list --region=europe-west2

# View job details
gcloud dataflow jobs describe JOB_ID --region=europe-west2

# Check worker logs
gcloud logging read "resource.type=dataflow_job AND resource.labels.job_id=JOB_ID"

# Monitor BigQuery jobs
bq ls -j --max_results=10
```

## Development

### Local Testing

```bash
# Run with DirectRunner for testing
python dataflow_etl_complete.py \
  --project-id=your-project-id \
  --dataset-id=test_dataset \
  --temp-location=gs://your-temp-bucket/tmp \
  --input-pattern="gs://your-test-data/*.json" \
  --runner=DirectRunner
```

### Adding Custom Features

1. Create new DoFn classes in `transforms.py`
2. Add to pipeline in `dataflow_etl_complete.py`
3. Update BigQuery schema as needed
4. Test with small data samples

### Code Quality

```bash
# Run tests
pytest tests/

# Format code
black *.py

# Lint code
flake8 *.py
pylint *.py

# Type checking
mypy *.py
```

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review Dataflow job logs in Cloud Console
3. Monitor BigQuery job history
4. Contact the development team

## License

This project is licensed under the MIT License - see the LICENSE file for details.