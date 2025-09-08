# Unified CQC Model Trainer (Phase 3)

This directory contains the **Unified ML Pipeline** implementation as specified in `plan.md` Phase 3. The system trains comprehensive ensemble models on CQC Syndication API data while maintaining compatibility with dashboard prediction systems.

## 🎯 Overview

The `UnifiedCQCModelTrainer` class implements:

- **Comprehensive CQC Data Integration**: Loads data from all CQC API endpoints (locations, providers, inspection areas, assessments, reports)
- **Unified Feature Space**: Creates 30+ features compatible with both CQC training data and dashboard prediction data
- **Ensemble Models**: Trains XGBoost, LightGBM, and Random Forest with voting classifier
- **Dashboard Compatibility**: Validates feature alignment for real-time predictions
- **Production Deployment**: Full GCP integration with Vertex AI model serving

## 🏗️ Architecture

### Data Sources
- **Training**: CQC Syndication API (comprehensive historical data)
- **Prediction**: Care Home Dashboard (live operational metrics)
- **Output**: CQC rating predictions (1-4 scale)

### Unified Feature Space (30+ features)
```python
# Core operational (available in both CQC and dashboard)
'bed_capacity', 'facility_size_numeric', 'occupancy_rate'

# Risk indicators (CQC: historical, Dashboard: current)
'inspection_overdue_risk', 'incident_frequency_risk', 'medication_risk', 'safeguarding_risk'

# Quality metrics (CQC: ratings, Dashboard: compliance)
'service_complexity_score', 'care_quality_indicator'

# Temporal features
'days_since_inspection', 'operational_stability', 'days_since_registration'

# Provider context
'provider_location_count', 'provider_avg_rating', 'provider_rating_consistency'

# Regional context
'regional_risk_rate', 'regional_avg_beds', 'regional_good_rating_rate'

# Interaction features (engineered combinations)
'complexity_scale_interaction', 'inspection_regional_risk', 'provider_regional_performance'
```

### Models Included
1. **XGBoost**: Gradient boosting optimized for tabular data
2. **LightGBM**: Fast gradient boosting with memory efficiency
3. **Random Forest**: Robust ensemble method
4. **Voting Classifier**: Combines all models for final predictions

## 🚀 Deployment

### Option 1: Cloud Build (Recommended)
```bash
# Deploy complete training pipeline
gcloud builds submit \
  --config src/ml/cloudbuild-unified-trainer.yaml \
  --substitutions _REGION=europe-west2,_MACHINE_TYPE=n1-highmem-4

# With hyperparameter tuning
gcloud builds submit \
  --config src/ml/cloudbuild-unified-trainer.yaml \
  --substitutions _TUNE_HYPERPARAMETERS=--tune-hyperparameters
```

### Option 2: Direct Python Script
```bash
# Install dependencies
pip install -r src/ml/requirements_unified.txt

# Run training locally (requires GCP authentication)
export GCP_PROJECT=your-project-id
export GCP_REGION=europe-west2
python src/ml/unified_model_trainer.py

# Or deploy as Vertex AI job
python src/ml/deploy_unified_trainer.py \
  --project-id your-project-id \
  --tune-hyperparameters
```

### Option 3: Manual Vertex AI Job
```bash
# Create custom training job
gcloud ai custom-jobs create \
  --region=europe-west2 \
  --display-name=cqc-unified-training \
  --config=src/ml/vertex-job-config.yaml
```

## 📊 Training Pipeline

### 1. Data Loading
```sql
-- Comprehensive CQC data from BigQuery
-- Combines: locations, providers, inspection_areas, assessment_service_groups
-- Features: 50,000+ care home records with full regulatory history
```

### 2. Feature Engineering
- **Operational metrics**: Bed capacity, facility size, service complexity
- **Risk indicators**: Inspection overdue, incident rates, safeguarding concerns
- **Quality metrics**: Domain ratings, compliance scores, care indicators
- **Temporal patterns**: Time since inspection, registration age, stability
- **Provider context**: Multi-location patterns, reputation scores
- **Regional factors**: Geographic risk rates, local performance benchmarks

### 3. Model Training
- **XGBoost**: `max_depth=6, learning_rate=0.1, n_estimators=200`
- **LightGBM**: `max_depth=6, learning_rate=0.1, n_estimators=200`
- **Random Forest**: `n_estimators=200, max_depth=15`
- **Ensemble**: Soft voting across all models

### 4. Validation & Deployment
- Cross-validation assessment
- Feature alignment validation for dashboard compatibility
- Model artifact storage in Cloud Storage
- Vertex AI model registration and endpoint deployment

## 📈 Expected Performance

Based on plan.md success metrics:

- **Training Accuracy**: >75% on CQC validation data
- **Precision/Recall**: >70% for all rating classes
- **Dashboard Compatibility**: >80% feature coverage
- **API Latency**: <500ms for real-time predictions
- **Prediction Confidence**: Average >70%

## 🔧 Model Artifacts

After training, artifacts are saved to:
```
gs://{project-id}-cqc-models/models/unified/{version}/
├── xgboost_model.pkl          # XGBoost model
├── lightgbm_model.pkl         # LightGBM model  
├── random_forest_model.pkl    # Random Forest model
├── ensemble_model.pkl         # Voting classifier
├── feature_names.json         # Feature list
├── feature_mapping.json       # Dashboard compatibility mapping
└── metadata.json             # Model metadata
```

## 🎪 Integration with Dashboard

The unified feature space ensures seamless integration:

### Dashboard Feature Extraction
```python
from dashboard_feature_extractor import DashboardFeatureExtractor

extractor = DashboardFeatureExtractor(client_id)
features = extractor.extract_care_home_features(care_home_id)

# Features automatically align with training data
aligned_features = feature_alignment.transform_dashboard_to_cqc_features(features)
```

### Real-time Predictions
```python
# Load trained ensemble
model = joblib.load('gs://bucket/models/unified/v1/ensemble_model.pkl')

# Predict CQC rating
prediction = model.predict_proba([aligned_features])[0]
rating = model.classes_[prediction.argmax()]
confidence = prediction.max()
```

## 🏥 Business Value

- **Early Warning**: Identify rating decline risk 3-6 months ahead
- **Actionable Insights**: Specific recommendations for improvement areas
- **Regulatory Preparation**: Support CQC inspection readiness
- **Risk Management**: Proactive identification of quality concerns
- **Performance Benchmarking**: Compare against regional and provider peers

## 🔍 Monitoring

Model performance is tracked via:
- **Vertex AI Model Monitoring**: Distribution skew, prediction drift
- **BigQuery Analytics**: Prediction accuracy over time
- **Cloud Monitoring**: API latency and error rates
- **Dashboard Integration**: User engagement with predictions

## 📞 Support

For issues or questions:
1. Check Cloud Logging: `gcloud logging read "resource.type=gce_instance"`
2. Monitor training jobs: GCP Console → Vertex AI → Training
3. Validate features: Run `python src/ml/test_unified_trainer.py`
4. Review model artifacts in Cloud Storage

---

**Implementation Status**: ✅ Complete - Ready for Production Deployment  
**Plan.md Phase**: 3 (Unified ML Pipeline)  
**GCP Services**: BigQuery, Cloud Storage, Vertex AI, Cloud Build  
**Dashboard Compatible**: ✅ Yes