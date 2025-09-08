# CQC Rating Predictor - Complete Implementation Plan

*Last Updated: 2025-01-09*

## 📊 Current System Status Assessment (CORRECTED)

### ✅ DEPLOYED INFRASTRUCTURE (100% Complete)
- **Cloud Run Services**: 8 services deployed and operational
- **Cloud Run Jobs**: 7 jobs deployed for data processing
- **BigQuery**: Full schema with tables and views
- **Storage**: Raw data buckets configured

### ⚠️ DATA STATUS (Real CQC API Data Assessment)
**Real CQC Syndication API Data Available:**
- ✅ **117,781 location IDs** from `/locations` endpoint
- ✅ **Basic location info**: ID, name, postal code
- ❌ **No detailed data**: Missing ratings, inspection dates, features needed for ML

**Current Dummy Data:**
- ❌ **2,000 synthetic locations** in BigQuery with dummy names ("Care Location 17")
- ❌ **Not suitable for ML training** - synthetic data

### 🚨 CRITICAL GAPS (Requires Immediate Action)
1. **No ML Training Data**: We have location IDs but need detailed data with ratings
2. **Missing CQC Details**: Need to fetch individual location details from `/locations/{locationId}`
3. **API Rate Limits**: 117,781 detailed requests needed - requires careful rate limiting
4. **No Trained Models**: Cannot train without real rating data

---

## 🎯 REVISED Implementation Roadmap

### PHASE 1: Fetch Real Detailed CQC Data ✅ **CRITICAL PRIORITY**

#### 1.1 Fetch Detailed Location Data from CQC Syndication API

**Goal**: Transform 117,781 basic location IDs into detailed records with ratings

**Current Status**: 
- Have: 117,781 location IDs from real CQC API
- Need: Detailed data for each location from `/locations/{locationId}` endpoint

**Required Data Fields for ML Training**:
```json
{
  "locationId": "1-123456789",
  "name": "Real Care Home Name",
  "currentRatings": {
    "overall": {"rating": "Good"},
    "safe": {"rating": "Good"},
    "effective": {"rating": "Good"},
    "caring": {"rating": "Outstanding"},
    "responsive": {"rating": "Good"},
    "wellLed": {"rating": "Good"},
    "reportDate": "2024-01-15",
    "reportUri": "/locations/1-123456789/reports/12345"
  },
  "lastInspection": {
    "date": "2024-01-10"
  },
  "registrationDate": "2019-03-15",
  "numberOfBeds": 45,
  "regulatedActivities": [...],
  "region": "London",
  "localAuthority": "Camden",
  "type": "Care home service with nursing",
  "specialisms": [...]
}
```

**Implementation Strategy**:

1. **Priority Location Selection** (Start with Care Homes Only):
   ```bash
   # Filter to care home locations from the 117,781 locations
   # Target: ~40,000 care home locations
   # Focus on: "Care home service with nursing" and "Care home service without nursing"
   ```

2. **Batch Processing with Rate Limiting**:
   ```bash
   # Execute enhanced ML data extractor
   gcloud run jobs execute cqc-ml-data-extractor \
     --region europe-west2 \
     --set-env-vars="
       SOURCE_DATA=real_locations,
       MAX_LOCATIONS=40000,
       LOCATION_TYPES=care_homes_only,
       BATCH_SIZE=100,
       RATE_LIMIT_PER_MINUTE=1800,
       PARALLEL_WORKERS=5,
       RETRY_FAILED=true" \
     --task-timeout=14400 \
     --wait
   ```

3. **Data Quality Requirements**:
   - Minimum 20,000 locations with current ratings
   - At least 500 examples per rating category
   - Complete inspection date information
   - Geographic distribution across UK regions

#### 1.2 Load Real Data to BigQuery

**Target Schema**: Replace dummy data with real CQC data
```sql
-- Clear dummy data and create real data table
DROP TABLE IF EXISTS `machine-learning-exp-467008.cqc_data.locations_detailed`;

CREATE TABLE `machine-learning-exp-467008.cqc_data.locations_real` (
  locationId STRING,
  locationName STRING,
  providerId STRING,
  
  -- Rating Information (CRITICAL for ML)
  overall_rating STRING,
  safe_rating STRING,
  effective_rating STRING,
  caring_rating STRING,
  responsive_rating STRING,
  well_led_rating STRING,
  last_report_date DATE,
  
  -- Operational Features
  numberOfBeds INT64,
  dormancy STRING,
  registrationStatus STRING,
  registrationDate DATETIME,
  lastInspectionDate DATETIME,
  
  -- Geographic Features  
  region STRING,
  localAuthority STRING,
  constituency STRING,
  postalCode STRING,
  
  -- Service Features
  organisationType STRING,
  locationType STRING,
  regulatedActivities ARRAY<STRING>,
  gacServiceTypes ARRAY<STRING>,
  specialisms ARRAY<STRING>,
  
  -- Metadata
  data_extraction_date TIMESTAMP,
  api_source STRING
)
PARTITION BY DATE(data_extraction_date)
CLUSTER BY region, overall_rating;
```

### PHASE 2: Data Quality Validation ✅ **PRIORITY 1**

#### 2.1 Validate Real CQC Data Quality

**Minimum Requirements for ML Training**:
```sql
-- Data quality validation query
WITH data_quality AS (
  SELECT 
    COUNT(*) as total_locations,
    COUNTIF(overall_rating IS NOT NULL AND overall_rating != '') as locations_with_ratings,
    COUNTIF(lastInspectionDate IS NOT NULL) as locations_with_inspection_dates,
    COUNTIF(numberOfBeds > 0) as locations_with_bed_count,
    COUNTIF(region IS NOT NULL) as locations_with_region,
    
    -- Rating distribution
    COUNTIF(overall_rating = 'Outstanding') as outstanding_count,
    COUNTIF(overall_rating = 'Good') as good_count,  
    COUNTIF(overall_rating = 'Requires improvement') as requires_improvement_count,
    COUNTIF(overall_rating = 'Inadequate') as inadequate_count,
    
    -- Geographic distribution
    COUNT(DISTINCT region) as unique_regions,
    COUNT(DISTINCT localAuthority) as unique_authorities
    
  FROM `machine-learning-exp-467008.cqc_data.locations_real`
)

SELECT 
  *,
  SAFE_DIVIDE(locations_with_ratings, total_locations) * 100 as rating_coverage_percent,
  CASE 
    WHEN locations_with_ratings >= 15000 
         AND outstanding_count >= 500 
         AND good_count >= 8000
         AND requires_improvement_count >= 3000 
         AND inadequate_count >= 500 
         AND unique_regions >= 8 
    THEN 'READY_FOR_TRAINING'
    ELSE 'INSUFFICIENT_DATA'
  END as ml_readiness_status
FROM data_quality;
```

**Success Criteria**:
- ✅ Total locations: ≥20,000 care homes
- ✅ Rating coverage: ≥75% locations have ratings
- ✅ Outstanding: ≥500 examples
- ✅ Good: ≥8,000 examples  
- ✅ Requires improvement: ≥3,000 examples
- ✅ Inadequate: ≥500 examples
- ✅ Geographic coverage: All UK regions represented

### PHASE 3: Feature Engineering with Real Data ✅ **PRIORITY 2**

#### 3.1 Create ML Training Features from Real CQC Data

**Feature Engineering Strategy** (Using Real Data Only):

```sql
CREATE OR REPLACE TABLE `machine-learning-exp-467008.cqc_data.ml_training_features_real` AS
WITH real_location_features AS (
  SELECT 
    locationId,
    locationName,
    providerId,
    
    -- === TEMPORAL FEATURES (High Predictive Power) ===
    DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_inspection,
    DATE_DIFF(CURRENT_DATE(), DATE(registrationDate), DAY) as days_since_registration,
    
    CASE 
      WHEN DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) > 730 THEN 1  -- Over 2 years
      WHEN DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) > 365 THEN 0.5 -- Over 1 year
      ELSE 0
    END as inspection_overdue_risk,
    
    -- === OPERATIONAL FEATURES ===
    COALESCE(numberOfBeds, 0) as bed_capacity,
    CASE 
      WHEN numberOfBeds >= 60 THEN 'Very_Large'
      WHEN numberOfBeds >= 40 THEN 'Large'
      WHEN numberOfBeds >= 20 THEN 'Medium'
      WHEN numberOfBeds > 0 THEN 'Small'
      ELSE 'Unknown'
    END as facility_size,
    
    -- === GEOGRAPHIC FEATURES (Regional Risk Patterns) ===
    region,
    localAuthority,
    constituency,
    SUBSTR(postalCode, 1, 2) as postcode_district,
    
    -- === SERVICE COMPLEXITY FEATURES ===
    ARRAY_LENGTH(regulatedActivities) as regulated_activities_count,
    ARRAY_LENGTH(gacServiceTypes) as service_types_count,
    ARRAY_LENGTH(specialisms) as specialisms_count,
    
    -- Service complexity score
    (ARRAY_LENGTH(regulatedActivities) * 0.4) + 
    (ARRAY_LENGTH(gacServiceTypes) * 0.3) + 
    (ARRAY_LENGTH(specialisms) * 0.3) as service_complexity_score,
    
    -- === ORGANIZATIONAL FEATURES ===
    organisationType,
    locationType,
    registrationStatus,
    dormancy,
    
    -- === TARGET VARIABLES ===
    overall_rating,
    safe_rating,
    effective_rating,
    caring_rating,
    responsive_rating,
    well_led_rating,
    
    -- === RISK INDICATORS ===
    CASE WHEN dormancy = 'Y' THEN 1 ELSE 0 END as is_dormant,
    CASE WHEN registrationStatus != 'Registered' THEN 1 ELSE 0 END as registration_issues,
    
    -- === METADATA ===
    last_report_date,
    data_extraction_date
    
  FROM `machine-learning-exp-467008.cqc_data.locations_real`
  WHERE overall_rating IS NOT NULL 
    AND overall_rating IN ('Outstanding', 'Good', 'Requires improvement', 'Inadequate')
    AND lastInspectionDate IS NOT NULL
),

-- Add derived features and provider-level aggregations
provider_stats AS (
  SELECT 
    providerId,
    COUNT(*) as provider_location_count,
    AVG(CASE 
      WHEN overall_rating = 'Outstanding' THEN 4
      WHEN overall_rating = 'Good' THEN 3
      WHEN overall_rating = 'Requires improvement' THEN 2
      WHEN overall_rating = 'Inadequate' THEN 1
    END) as provider_avg_rating_numeric,
    
    STDDEV(CASE 
      WHEN overall_rating = 'Outstanding' THEN 4
      WHEN overall_rating = 'Good' THEN 3
      WHEN overall_rating = 'Requires improvement' THEN 2
      WHEN overall_rating = 'Inadequate' THEN 1
    END) as provider_rating_volatility
    
  FROM real_location_features
  WHERE providerId IS NOT NULL
  GROUP BY providerId
),

-- Regional risk patterns
regional_stats AS (
  SELECT 
    region,
    COUNT(*) as region_location_count,
    COUNTIF(overall_rating IN ('Requires improvement', 'Inadequate')) / COUNT(*) as region_risk_rate,
    AVG(bed_capacity) as region_avg_bed_capacity
  FROM real_location_features
  WHERE region IS NOT NULL
  GROUP BY region
)

SELECT 
  f.*,
  
  -- === DERIVED NUMERIC TARGET ===
  CASE 
    WHEN overall_rating = 'Outstanding' THEN 4
    WHEN overall_rating = 'Good' THEN 3
    WHEN overall_rating = 'Requires improvement' THEN 2
    WHEN overall_rating = 'Inadequate' THEN 1
    ELSE NULL
  END as overall_rating_numeric,
  
  -- === PROVIDER-LEVEL FEATURES ===
  COALESCE(p.provider_location_count, 1) as provider_location_count,
  COALESCE(p.provider_avg_rating_numeric, 3) as provider_avg_rating,
  COALESCE(p.provider_rating_volatility, 0) as provider_rating_consistency,
  
  -- === REGIONAL RISK FEATURES ===  
  COALESCE(r.region_risk_rate, 0.2) as regional_risk_rate,
  COALESCE(r.region_avg_bed_capacity, 30) as regional_avg_beds,
  
  -- === INTERACTION FEATURES ===
  service_complexity_score * COALESCE(p.provider_location_count, 1) as complexity_scale_interaction,
  inspection_overdue_risk * COALESCE(r.region_risk_rate, 0.2) as inspection_regional_risk,
  
  -- === HIGH-RISK INDICATORS ===
  CASE 
    WHEN overall_rating IN ('Requires improvement', 'Inadequate') THEN 1 
    ELSE 0 
  END as high_risk_current,
  
  CASE 
    WHEN days_since_inspection > 730 
         AND COALESCE(r.region_risk_rate, 0) > 0.25 
         AND service_complexity_score > 5 
    THEN 1 ELSE 0 
  END as high_risk_predicted_features

FROM real_location_features f
LEFT JOIN provider_stats p USING(providerId)  
LEFT JOIN regional_stats r USING(region)
WHERE overall_rating_numeric IS NOT NULL;
```

### PHASE 4: ML Model Training with Real Data ✅ **PRIORITY 3**

#### 4.1 Train Production ML Models

**Training Configuration** (Real Data Only):
```bash
gcloud run jobs execute cqc-model-trainer \
  --region europe-west2 \
  --set-env-vars="
    TRAINING_TABLE=cqc_data.ml_training_features_real,
    TARGET_COLUMN=overall_rating_numeric,
    FEATURE_COLUMNS=days_since_inspection,bed_capacity,regulated_activities_count,service_complexity_score,provider_avg_rating,regional_risk_rate,
    MIN_SAMPLES_PER_CLASS=500,
    VALIDATION_SPLIT=0.2,
    ALGORITHMS=xgboost,lightgbm,random_forest,
    TARGET_ACCURACY=0.80,
    CROSS_VALIDATION=true,
    EARLY_STOPPING=true,
    HYPERPARAMETER_TUNING=true" \
  --task-timeout=3600 \
  --wait
```

**Expected Results with Real Data**:
- **Accuracy Target**: >80% (realistic with real-world data)
- **Class Balance**: Handle imbalanced dataset (more "Good" than "Inadequate")
- **Feature Importance**: Identify key predictors from real CQC data
- **Model Interpretability**: SHAP values for regulatory compliance

#### 4.2 Deploy Models to Vertex AI

**Production Deployment**:
```bash
# Deploy best performing model to Vertex AI endpoint
gcloud run jobs execute cqc-model-trainer \
  --region europe-west2 \
  --set-env-vars="
    MODE=deploy_to_vertex,
    MODEL_NAME=cqc-rating-predictor-real,
    ENDPOINT_MACHINE_TYPE=n1-standard-4,
    MIN_REPLICAS=1,
    MAX_REPLICAS=20" \
  --wait
```

### PHASE 5: API Integration & Production Testing ✅ **PRIORITY 4**

#### 5.1 Configure Production APIs with Real Models

**API Services Configuration**:
```bash
# Enable APIs for production use
gcloud run services add-iam-policy-binding cqc-rating-prediction \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region europe-west2

gcloud run services add-iam-policy-binding proactive-risk-assessment \
  --member="allUsers" \
  --role="roles/run.invoker" \
  --region europe-west2
```

#### 5.2 End-to-End Testing with Real Location IDs

**Test with Real CQC Location IDs**:
```bash
# Test with actual location ID from our real data
curl -X POST https://cqc-rating-prediction-744974744548.europe-west2.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "location_id": "1-10000302982",
    "features": {
      "days_since_inspection": 180,
      "bed_capacity": 45,
      "service_complexity_score": 5.2,
      "regional_risk_rate": 0.15
    }
  }'
```

---

## 🔧 IMMEDIATE Action Plan (Execute in Order)

### Step 1: Fetch Real Detailed CQC Data (2-3 hours) ✅ **START HERE**
```bash
# Execute ML data extractor to get detailed location data from CQC Syndication API
gcloud run jobs execute cqc-ml-data-extractor \
  --region europe-west2 \
  --set-env-vars="
    SOURCE_DATA=real_cqc_api,
    MAX_LOCATIONS=25000,
    FOCUS_CARE_HOMES=true,
    BATCH_SIZE=50,
    RATE_LIMIT=1800" \
  --task-timeout=14400 \
  --wait
```

### Step 2: Validate Real Data Quality (15 minutes)
```bash
# Check if we have sufficient real data for ML training
bq query --use_legacy_sql=false --project_id=machine-learning-exp-467008 \
"SELECT 
  COUNT(*) as total,
  COUNTIF(overall_rating IS NOT NULL) as with_ratings,
  COUNTIF(overall_rating = 'Outstanding') as outstanding,
  COUNTIF(overall_rating = 'Good') as good,
  COUNTIF(overall_rating = 'Requires improvement') as requires_improvement,
  COUNTIF(overall_rating = 'Inadequate') as inadequate,
  COUNT(DISTINCT region) as regions
FROM \`cqc_data.locations_real\`"
```

### Step 3: Create ML Features from Real Data (30 minutes)
```bash  
# Create training features from real CQC data
bq query --use_legacy_sql=false --project_id=machine-learning-exp-467008 < sql/create_real_ml_features.sql
```

### Step 4: Train Models on Real Data (45 minutes)
```bash
# Train production models using real CQC data
gcloud run jobs execute cqc-model-trainer \
  --region europe-west2 \
  --set-env-vars="TRAINING_TABLE=cqc_data.ml_training_features_real,REAL_DATA=true" \
  --wait
```

### Step 5: Test Production System (15 minutes)
```bash
# Test with real location ID
curl -X POST https://cqc-rating-prediction-744974744548.europe-west2.run.app/predict \
  -H "Content-Type: application/json" \
  -d '{"location_id": "1-10000302982"}'
```

---

## 📈 Success Metrics (Real Data)

### Data Requirements (Must Have)
- **Real Locations**: ≥20,000 care home locations from CQC Syndication API
- **Rating Coverage**: ≥15,000 locations with current ratings (75%)
- **Class Distribution**: Minimum 500 examples per rating class
- **Geographic Coverage**: All UK regions represented
- **Data Freshness**: Extracted within last 30 days

### ML Performance (Realistic Targets)
- **Overall Accuracy**: ≥75% (realistic for real-world healthcare data)
- **Precision per Class**: ≥70% for all rating categories
- **Recall for Risk Cases**: ≥80% for "Requires improvement" + "Inadequate"
- **Model Confidence**: Provide confidence scores for predictions
- **Feature Importance**: Clear attribution for regulatory compliance

### System Performance
- **API Latency**: <500ms per prediction (real model complexity)
- **Throughput**: ≥50 predictions/second
- **Availability**: ≥99% uptime
- **Error Rate**: <2% prediction failures

---

## ⚠️ Critical Dependencies & Risks

### API Rate Limiting Risk
- **Challenge**: Need ~25,000 individual API calls to get detailed location data
- **CQC Limit**: 2,000 requests/minute with Partner Code
- **Time Required**: ~15-20 minutes for data extraction
- **Mitigation**: Batch processing with exponential backoff

### Data Quality Risk  
- **Challenge**: Unknown percentage of real locations have current ratings
- **Impact**: May have insufficient training data
- **Mitigation**: Prioritize recently inspected care homes
- **Contingency**: Accept lower accuracy with limited data

### Model Performance Risk
- **Challenge**: Real healthcare data is noisy and imbalanced  
- **Reality**: May achieve 75-80% accuracy vs 85%+ with perfect data
- **Mitigation**: Focus on high-confidence predictions
- **Value**: Still provides significant business value at 75% accuracy

---

## ⏱️ Revised Timeline

| Phase | Duration | Status | Dependency |
|-------|----------|---------|------------|
| **Fetch Real CQC Data** | **3 hours** | **CRITICAL** | API rate limits |
| **Data Validation** | **15 min** | **Ready** | Phase 1 complete |
| **Feature Engineering** | **30 min** | **Ready** | Phase 1 complete |
| **Model Training** | **45 min** | **Ready** | Real data available |
| **Testing & Deployment** | **30 min** | **Ready** | Models trained |
| **TOTAL REAL SYSTEM** | **4.5 hours** | **Achievable** | Start immediately |

---

## 🎯 Final System with Real Data

### What We'll Achieve
- **Real CQC Predictions**: Based on actual Syndication API data
- **Production Accuracy**: 75-80% (realistic for healthcare data)
- **Regulatory Compliance**: Explainable predictions using real features  
- **Live Integration**: APIs ready for care home dashboard integration
- **Scalable Architecture**: Can handle all UK care home locations

### Business Value
- **Proactive Risk Management**: Identify care homes likely to decline in ratings
- **Data-Driven Insights**: Based on real CQC inspection patterns
- **Regulatory Support**: Assist care providers in preparation for inspections
- **Quality Improvement**: Enable targeted interventions before problems occur

**The key insight: We need to fetch detailed data from the CQC Syndication API first, then we can build a robust ML system. The infrastructure is ready - we just need the real data to power it.**