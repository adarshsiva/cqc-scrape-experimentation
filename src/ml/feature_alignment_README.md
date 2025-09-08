# Feature Alignment & Transformation Service

## Overview

The Feature Alignment & Transformation Service is a critical component of the CQC Rating Predictor system that bridges the gap between live care home dashboard data and CQC training features. It transforms operational dashboard metrics into the feature space that CQC prediction models expect, enabling real-time predictions using live operational data.

## Purpose

**Training Data Source**: CQC Syndication API (historical inspection data, ratings, facility details)
**Prediction Data Source**: Care Home Dashboard (live operational metrics, incidents, care plans)
**Challenge**: These two data sources have different schemas and feature representations
**Solution**: This service transforms dashboard features to match the CQC training feature space

## Architecture

```
Dashboard Data → Feature Extraction → Feature Alignment → CQC-Compatible Features → ML Model
     ↓                    ↓                    ↓                        ↓               ↓
Operational         Individual         Transform to         Normalized        CQC Rating
Metrics            Features           CQC Schema           Features          Prediction
```

## Key Features

### 1. Direct Feature Mapping
- **bed_capacity**: Direct mapping from dashboard to CQC
- **occupancy_rate**: Direct mapping from dashboard to CQC
- **facility_size_numeric**: Categorized bed count classification

### 2. Derived Feature Calculations
- **service_complexity_score**: Combines care complexity + activity variety
- **inspection_overdue_risk**: Transforms incident/care plan risks to inspection risk
- **provider_avg_rating**: Estimates provider performance from operational metrics

### 3. Risk Indicator Transformations
- **incident_frequency_risk**: Direct from dashboard incident data
- **medication_risk**: Direct from dashboard medication error data
- **safeguarding_risk**: Direct from dashboard safeguarding incidents
- **falls_risk**: Direct from dashboard falls data

### 4. Quality Indicators
- **care_quality_indicator**: Combines care plans, engagement, safety metrics
- **operational_stability**: Stability score from multiple operational factors

### 5. Contextual Features
- **regional_risk_rate**: Postcode-based regional risk lookup
- **provider_location_count**: Defaults to 1 for single dashboard location
- **provider_rating_consistency**: Estimated consistency score

### 6. Interaction Features
- **complexity_scale_interaction**: Service complexity × provider scale
- **inspection_regional_risk**: Inspection risk × regional risk
- **capacity_complexity_interaction**: Bed capacity × service complexity
- **risk_quality_balance**: Quality score minus risk factors

## Usage Example

```python
from feature_alignment import FeatureAlignmentService

# Initialize service
service = FeatureAlignmentService(project_id='your-gcp-project')

# Dashboard features from your care home system
dashboard_features = {
    'bed_capacity': 45,
    'occupancy_rate': 0.88,
    'avg_care_complexity': 2.3,
    'incident_frequency_risk': 0.15,
    'medication_risk': 0.08,
    'care_plan_compliance': 0.92,
    'resident_engagement': 0.78,
    'staff_compliance_score': 0.94,
    'postcode': 'SW1A 1AA'
}

# Transform to CQC-compatible features
cqc_features = service.transform_dashboard_to_cqc_features(dashboard_features)

# Use for ML prediction
predicted_rating = your_cqc_model.predict(cqc_features)
```

## Feature Transformation Logic

### Service Complexity Calculation
```python
def _calculate_service_complexity(self, avg_care_complexity, activity_variety):
    # Normalize care complexity (1-3 scale) to 0-1
    care_complexity_norm = (avg_care_complexity - 1.0) / 2.0
    
    # Combine with activity variety (0-1 scale)
    service_complexity = (
        care_complexity_norm * 0.7 +  # Care complexity weighted higher
        activity_variety * 0.3
    )
    
    # Scale to CQC service complexity range (1-8)
    return 1.0 + (service_complexity * 7.0)
```

### Provider Rating Estimation
```python
def _estimate_provider_rating(self, dashboard_features):
    # Risk factors (negative)
    risk_score = (
        incident_frequency_risk * 0.3 +
        medication_risk * 0.3 +
        safeguarding_risk * 0.4
    )
    
    # Quality factors (positive)
    quality_score = (
        care_plan_compliance * 0.4 +
        resident_engagement * 0.3 +
        staff_compliance_score * 0.3
    )
    
    # Convert to CQC 1-4 scale
    overall_score = quality_score * 0.5 + (1.0 - risk_score) * 0.5
    return scale_to_cqc_rating(overall_score)
```

### Inspection Risk Transformation
```python
def _transform_to_inspection_risk(self, incident_risk, care_plan_risk):
    # Combine operational risks
    combined_risk = incident_risk * 0.6 + care_plan_risk * 0.4
    
    # Apply non-linear transformation for inspection risk pattern
    return min(1.0, np.tanh(combined_risk * 2.0))
```

## Feature Normalization

All features are normalized to match CQC training data distributions:

- **Ranges**: Features clipped to valid ranges (e.g., occupancy_rate: 0.0-1.0)
- **Scaling**: Features scaled to match typical CQC value ranges
- **Defaults**: Missing features filled with sensible defaults

## Regional Risk Mapping

Postcode-based regional risk rates based on CQC performance patterns:

```python
regional_risk_map = {
    'SW': 0.22,   # South West London - Lower risk
    'M': 0.35,    # Manchester - Higher risk  
    'B': 0.32,    # Birmingham - Moderate risk
    'E': 0.25,    # East of England - Lower risk
    # ... more regions
}
```

## Error Handling

- **Graceful Degradation**: Missing features use sensible defaults
- **Value Validation**: Extreme values are clipped to valid ranges
- **Logging**: Comprehensive logging for debugging and monitoring
- **Exception Handling**: Robust error handling prevents pipeline failures

## Testing

Run the test suite:

```bash
python src/ml/test_feature_alignment_minimal.py
```

Run examples:

```bash
python src/ml/feature_alignment_example.py
```

## Integration Points

### 1. Dashboard Feature Extractor
Receives features from `DashboardFeatureExtractor` and transforms them.

### 2. ML Model Pipeline
Outputs features in the format expected by CQC prediction models.

### 3. Prediction API
Used by the prediction service to prepare features for real-time inference.

## Production Considerations

### Performance
- **Caching**: Regional risk lookup caching for performance
- **Vectorization**: Batch processing support for multiple care homes
- **Memory**: Efficient feature computation without large data loading

### Monitoring
- **Feature Drift**: Monitor for changes in dashboard feature distributions
- **Quality Metrics**: Track transformation success rates and feature coverage
- **Alerting**: Alert on unusual feature values or transformation failures

### Scalability
- **Stateless Design**: Service can be scaled horizontally
- **BigQuery Integration**: Can query historical statistics for normalization
- **Cloud Ready**: Designed for Google Cloud Platform deployment

## Files Structure

```
src/ml/
├── feature_alignment.py                 # Main service class
├── feature_alignment_requirements.txt   # Dependencies  
├── feature_alignment_example.py         # Usage examples
├── feature_alignment_README.md          # This documentation
├── test_feature_alignment.py            # Full unit tests
└── test_feature_alignment_minimal.py    # Minimal tests (no dependencies)
```

## Dependencies

- `numpy>=1.21.0` - Numerical computations
- `pandas>=1.3.0` - Data manipulation (optional)
- `google-cloud-bigquery>=3.0.0` - CQC statistics lookup (production)

## Configuration

Environment variables:
- `GCP_PROJECT`: Google Cloud Project ID for BigQuery
- `CQC_FEATURE_STATS_TABLE`: BigQuery table with training feature statistics

## Next Steps

1. **Integration**: Integrate with `DashboardFeatureExtractor` 
2. **Model Training**: Use aligned features for CQC model training
3. **API Integration**: Deploy as part of prediction API service
4. **Monitoring**: Set up feature drift monitoring in production
5. **Optimization**: Fine-tune feature transformations based on model performance