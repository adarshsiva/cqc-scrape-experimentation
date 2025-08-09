# CQC Rating Predictor - Project Structure

## 📁 Core Directories

### `/src` - Source Code
- **`/ml`** - Machine learning pipeline
  - `train_model_cloud.py` - Main training script
  - `Dockerfile.train` - Container for training
  - `cloudbuild-train.yaml` - Build configuration
- **`/prediction`** - Prediction API
  - `proactive_predictor.py` - Risk assessment API
  - `Dockerfile.proactive` - Container for API
- **`/alerts`** - Notification service
- **`/etl`** - Data transformation pipeline
- **`/ingestion`** - Data ingestion service

### `/scripts` - Data Processing Scripts
- `fetch_detailed_cqc_data.py` - CQC API data fetcher
- `process_data_job.py` - BigQuery data processor
- `load_synthetic_data.py` - Synthetic data loader
- `generate_synthetic_data.py` - Test data generator

### `/sql` - Database Scripts
- `create_ml_features_view.sql` - ML feature engineering
- `load_synthetic_data.sql` - Data loading queries
- `risk_monitoring.sql` - Monitoring queries

### `/config` - Configuration Files
- BigQuery schemas
- Vertex AI configuration
- GCS lifecycle policies

### `/dags` - Airflow DAGs
- `cqc_daily_pipeline.py` - Daily orchestration
- `cqc_monitoring_dag.py` - Monitoring workflow

### `/deployment` - Deployment Scripts
- `deploy_all_services.sh` - Master deployment
- `setup_gcp_resources.sh` - Infrastructure setup
- `monitor_system.sh` - System monitoring

### `/documentation` - Technical Docs
- **`/gcp`** - Google Cloud service guides
- **`/terraform`** - Infrastructure as Code
- **`/cqc`** - CQC API documentation

## 📄 Key Files

### Root Level
- `CLAUDE.md` - AI assistant instructions
- `README.md` - Project overview
- `DEPLOYMENT_GUIDE.md` - Deployment instructions
- `RUNBOOK.md` - Operational procedures
- `ARCHITECTURE.md` - System architecture
- `CURRENT_STATUS.md` - Current deployment status
- `work.md` - Implementation summary
- `nextsteps.md` - Remaining tasks

### Build Files
- `cloudbuild.yaml` - Main build configuration
- `Dockerfile.*` - Container definitions
- `requirements.txt` - Python dependencies

## 🚀 Deployment Artifacts

### Cloud Run Jobs
- `cqc-data-processor`
- `cqc-model-trainer`
- `cqc-synthetic-loader`
- `fetch-cqc-data-job`

### Cloud Run Services
- `proactive-risk-assessment`
- `cqc-rating-prediction`
- `cqc-data-ingestion`

### Container Images
- `gcr.io/machine-learning-exp-467008/cqc-data-processor`
- `gcr.io/machine-learning-exp-467008/cqc-model-trainer`
- `gcr.io/machine-learning-exp-467008/cqc-data-loader`

## 🧹 Cleanup Summary

Removed:
- Test files (`test_*.py`, `test_*.sh`)
- Temporary scripts (`simple_bq_load.sh`, etc.)
- Duplicate deployment scripts
- Extra README files
- Virtual environment (`venv/`)
- Empty directories

The project is now organized with clear separation between:
- Production code (`/src`)
- Data processing (`/scripts`)
- Infrastructure (`/deployment`, `/terraform`)
- Documentation (`/documentation`, root docs)