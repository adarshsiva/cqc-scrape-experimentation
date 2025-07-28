# CQC ML Pipeline

This directory contains the machine learning pipeline components for the CQC Rating Predictor system using Google Cloud Vertex AI.

## Directory Structure

```
ml/
├── features.py              # Feature engineering module
├── pipeline/
│   ├── components.py       # Vertex AI pipeline components
│   └── pipeline.py         # Main pipeline orchestration
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Components

### 1. Feature Engineering (`features.py`)

The `CQCFeatureEngineer` class provides comprehensive feature engineering capabilities:

- **Temporal Features**: Days/months/years since last inspection, registration age
- **Categorical Encoding**: One-hot and label encoding for service types, regions, etc.
- **Text Features**: TF-IDF extraction from inspection areas and regulated activities
- **Numerical Scaling**: StandardScaler for numerical features
- **Aggregated Features**: Count of regulated activities, regional statistics
- **Feature Selection**: Statistical methods (chi2, f_classif) for selecting top features

### 2. Pipeline Components (`pipeline/components.py`)

Reusable Vertex AI pipeline components:

- **data_preparation_component**: Loads data from BigQuery and splits into train/val/test
- **feature_preprocessing_component**: Applies feature engineering to datasets
- **train_xgboost_component**: Trains XGBoost classifier
- **train_lightgbm_component**: Trains LightGBM classifier
- **train_automl_component**: Trains AutoML Tabular model
- **model_evaluation_component**: Evaluates models and makes deployment decisions
- **model_deployment_component**: Deploys models to Vertex AI Endpoints

### 3. Main Pipeline (`pipeline/pipeline.py`)

The `cqc_ml_pipeline` orchestrates the entire ML workflow:

1. Data preparation from BigQuery
2. Feature preprocessing
3. Parallel model training (XGBoost, LightGBM, optional AutoML)
4. Model evaluation on test set
5. Conditional deployment based on performance criteria
6. Model versioning and tracking

## Usage

### Running the Pipeline

```python
from src.ml.pipeline import run_pipeline

# Run the pipeline
job = run_pipeline(
    project_id="your-gcp-project",
    location="us-central1",
    dataset_id="cqc_data",
    table_id="cqc_ratings",
    enable_automl=False,  # Set to True to include AutoML
    service_account="your-sa@project.iam.gserviceaccount.com"
)
```

### Command Line Interface

```bash
# Compile pipeline only
python -m src.ml.pipeline.pipeline --project-id YOUR_PROJECT --action compile

# Run pipeline
python -m src.ml.pipeline.pipeline \
    --project-id YOUR_PROJECT \
    --action run \
    --service-account your-sa@project.iam.gserviceaccount.com

# Schedule weekly pipeline runs
python -m src.ml.pipeline.pipeline \
    --project-id YOUR_PROJECT \
    --action schedule \
    --service-account your-sa@project.iam.gserviceaccount.com
```

### Using Feature Engineering Standalone

```python
from src.ml.features import CQCFeatureEngineer
import pandas as pd

# Initialize feature engineer
engineer = CQCFeatureEngineer()

# Load your data
df = pd.read_csv("cqc_data.csv")

# Apply feature engineering
X, y = engineer.create_feature_pipeline(
    df, 
    target_column='rating',
    is_training=True
)
```

## Configuration

### Pipeline Parameters

- **train_split**: Fraction of data for training (default: 0.7)
- **validation_split**: Fraction for validation (default: 0.15)
- **test_split**: Fraction for testing (default: 0.15)
- **xgboost_hyperparameters**: Dict of XGBoost parameters
- **lightgbm_hyperparameters**: Dict of LightGBM parameters
- **automl_budget_hours**: Training budget for AutoML (default: 1.0)
- **machine_type**: Instance type for serving (default: "n1-standard-4")
- **min_replicas**: Minimum serving replicas (default: 1)
- **max_replicas**: Maximum serving replicas (default: 3)

### Deployment Criteria

Models are automatically deployed if they meet these criteria:
- Test accuracy > 70%
- All class-wise recall scores > 50%

## Prerequisites

1. **Google Cloud Setup**:
   - Enable Vertex AI API
   - Create a GCS bucket for pipeline artifacts
   - Set up a service account with appropriate permissions

2. **Permissions Required**:
   - BigQuery Data Viewer
   - Storage Object Admin
   - Vertex AI User
   - Service Account Token Creator

3. **Install Dependencies**:
   ```bash
   pip install -r src/ml/requirements.txt
   ```

## Model Performance Tracking

The pipeline automatically logs metrics to Vertex AI Experiments:
- Per-class precision, recall, F1 scores
- Confusion matrices
- Feature importance rankings
- Model hyperparameters

View results in the Vertex AI console under Experiments.

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed and PYTHONPATH includes the project root
2. **Permission Denied**: Check service account has all required permissions
3. **Pipeline Timeout**: Increase timeout in pipeline configuration (default: 2 hours)
4. **Out of Memory**: Use larger machine types or reduce batch sizes

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- Hyperparameter tuning with Vertex AI Vizier
- Model monitoring and drift detection
- A/B testing framework for model comparison
- Integration with Feature Store
- Real-time feature serving