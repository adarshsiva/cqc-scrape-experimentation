# CQC Rating Predictor - Current Status
*Last Updated: 2025-07-29*

## 🏗️ Infrastructure Status

### ✅ Created Resources
1. **Cloud Run Jobs**
   - `cqc-data-processor` - Processes CQC data from API
   - `cqc-synthetic-loader` - Loads synthetic data to BigQuery
   - `cqc-model-trainer` - (Ready to deploy) Trains ML models

2. **Container Images**
   - `gcr.io/machine-learning-exp-467008/cqc-data-processor`
   - `gcr.io/machine-learning-exp-467008/cqc-data-loader`
   - `gcr.io/machine-learning-exp-467008/cqc-model-trainer` (Ready to build)

3. **BigQuery Resources**
   - Dataset: `cqc_data`
   - Tables: `locations_detailed`, `locations_staging`, `ml_features_simple`
   - View: `ml_features_proactive` (Ready to create)

4. **Google Cloud Storage**
   - Bucket: `machine-learning-exp-467008-cqc-raw-data`
   - Bucket: `machine-learning-exp-467008-cqc-ml-artifacts`

## 🚧 Current Blockers

1. **CQC API Access**
   - API returns 403 errors from Google Cloud IPs
   - Need alternative approach (proxy, VM, or API whitelisting)

2. **Authentication**
   - gcloud CLI needs re-authentication
   - Run: `gcloud auth login` and `gcloud auth application-default login`

3. **Data Loading**
   - Synthetic data ready but not loaded to BigQuery
   - Need to execute SQL script: `sql/load_synthetic_data.sql`

## 📊 Data Status

- **Synthetic Data**: Available at `gs://machine-learning-exp-467008-cqc-raw-data/raw/locations/20250728_191714_locations_sample.json`
- **Real Data**: Limited samples in `gs://machine-learning-exp-467008-cqc-raw-data/real_data/`
- **Processed Data**: None yet

## 🔄 Next Immediate Actions

1. **Fix Authentication**
   ```bash
   gcloud auth login
   gcloud config set project machine-learning-exp-467008
   ```

2. **Load Synthetic Data**
   ```bash
   bq query --use_legacy_sql=false --project_id=machine-learning-exp-467008 < sql/load_synthetic_data.sql
   ```

3. **Train Model**
   ```bash
   gcloud run jobs execute cqc-model-trainer --region europe-west2
   ```

4. **Deploy API**
   ```bash
   cd src/prediction
   gcloud run deploy proactive-risk-assessment --source . --region europe-west2
   ```

## 🎯 Project Goals

Build a proactive early warning system that:
- Identifies healthcare locations at risk of poor CQC ratings
- Provides 3-6 month advance warning
- Offers actionable recommendations for improvement
- Achieves >85% AUC score in predictions

## 📁 Key Files

- **Training Script**: `src/ml/train_model_cloud.py`
- **Prediction API**: `src/prediction/proactive_predictor.py`
- **Data Loader**: `scripts/load_synthetic_data.py`
- **SQL Scripts**: `sql/load_synthetic_data.sql`
- **Next Steps**: `nextsteps.md`

## 🔗 Useful Commands

```bash
# Check job status
gcloud run jobs list --region europe-west2

# View logs
gcloud logging read "resource.type=cloud_run_job" --limit 50 --format "table(timestamp,textPayload)"

# List BigQuery tables
bq ls cqc_data

# Check GCS buckets
gsutil ls gs://machine-learning-exp-467008-cqc-raw-data/
```