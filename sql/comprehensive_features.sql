-- Comprehensive Feature Engineering SQL for CQC ML Training
-- Based on plan.md Section 1.2 - Creates ml_training_features_comprehensive table
-- Implements comprehensive feature engineering from all available CQC data sources

CREATE OR REPLACE TABLE `machine-learning-exp-467008.cqc_data.ml_training_features_comprehensive` AS
WITH location_data AS (
  -- Core location features from CQC locations data
  SELECT 
    locationId,
    name,
    providerId,
    numberOfBeds,
    registrationDate,
    lastInspectionDate,
    -- Extract ratings from nested structure
    currentRatings.overall.rating as overall_rating,
    currentRatings.safe.rating as safe_rating,
    currentRatings.effective.rating as effective_rating,
    currentRatings.caring.rating as caring_rating,
    currentRatings.responsive.rating as responsive_rating,
    currentRatings.wellLed.rating as well_led_rating,
    -- Geographic and organizational context
    region,
    localAuthority,
    organisationType,
    type as locationType,
    postalCode,
    -- Service complexity metrics
    ARRAY_LENGTH(regulatedActivities) as regulated_activities_count,
    ARRAY_LENGTH(gacServiceTypes) as gac_service_types_count,
    ARRAY_LENGTH(specialisms) as specialisms_count,
    ARRAY_LENGTH(serviceUserBands) as service_user_bands_count,
    -- Calculate days since key dates
    DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_inspection,
    DATE_DIFF(CURRENT_DATE(), DATE(registrationDate), DAY) as days_since_registration
  FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
),

inspection_history AS (
  -- Historical inspection patterns from nested inspection history
  SELECT 
    locationId,
    COUNT(*) as inspection_count,
    -- Calculate average historical rating (convert to numeric for calculation)
    AVG(CASE 
      WHEN ih.rating = 'Outstanding' THEN 4
      WHEN ih.rating = 'Good' THEN 3
      WHEN ih.rating = 'Requires improvement' THEN 2
      WHEN ih.rating = 'Inadequate' THEN 1
      ELSE NULL
    END) as historical_avg_rating_numeric,
    -- Calculate rating volatility
    STDDEV(CASE 
      WHEN ih.rating = 'Outstanding' THEN 4
      WHEN ih.rating = 'Good' THEN 3
      WHEN ih.rating = 'Requires improvement' THEN 2
      WHEN ih.rating = 'Inadequate' THEN 1
      ELSE NULL
    END) as rating_volatility,
    -- Most recent historical inspection date
    MAX(ih.inspectionDate) as most_recent_historical_inspection,
    -- Count unique inspection dates
    COUNT(DISTINCT DATE(ih.inspectionDate)) as unique_inspection_dates
  FROM `machine-learning-exp-467008.cqc_data.locations_detailed`,
  UNNEST(inspectionHistory) as ih
  WHERE ih.rating IS NOT NULL
  GROUP BY locationId
),

provider_context AS (
  -- Provider-level aggregated patterns
  SELECT
    providerId,
    COUNT(DISTINCT locationId) as provider_location_count,
    -- Calculate provider average rating
    AVG(CASE 
      WHEN currentRatings.overall.rating = 'Outstanding' THEN 4
      WHEN currentRatings.overall.rating = 'Good' THEN 3
      WHEN currentRatings.overall.rating = 'Requires improvement' THEN 2
      WHEN currentRatings.overall.rating = 'Inadequate' THEN 1
      ELSE NULL
    END) as provider_avg_rating,
    -- Provider geographic spread
    COUNT(DISTINCT region) as provider_geographic_spread,
    -- Provider capacity metrics
    AVG(numberOfBeds) as provider_avg_capacity,
    SUM(numberOfBeds) as provider_total_capacity,
    -- Rating consistency across provider locations
    STDDEV(CASE 
      WHEN currentRatings.overall.rating = 'Outstanding' THEN 4
      WHEN currentRatings.overall.rating = 'Good' THEN 3
      WHEN currentRatings.overall.rating = 'Requires improvement' THEN 2
      WHEN currentRatings.overall.rating = 'Inadequate' THEN 1
      ELSE NULL
    END) as provider_rating_consistency
  FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
  WHERE currentRatings.overall.rating IS NOT NULL
  GROUP BY providerId
),

regional_context AS (
  -- Regional risk assessment patterns
  SELECT
    region,
    COUNT(*) as regional_location_count,
    -- Regional rating distribution
    AVG(CASE 
      WHEN currentRatings.overall.rating = 'Outstanding' THEN 4
      WHEN currentRatings.overall.rating = 'Good' THEN 3
      WHEN currentRatings.overall.rating = 'Requires improvement' THEN 2
      WHEN currentRatings.overall.rating = 'Inadequate' THEN 1
      ELSE NULL
    END) as regional_avg_rating,
    -- Regional risk rate (proportion of poor ratings)
    COUNTIF(currentRatings.overall.rating IN ('Inadequate', 'Requires improvement')) / 
    NULLIF(COUNTIF(currentRatings.overall.rating IS NOT NULL), 0) as regional_risk_rate,
    -- Regional capacity metrics
    AVG(numberOfBeds) as regional_avg_beds,
    -- Regional service complexity
    AVG(ARRAY_LENGTH(regulatedActivities)) as regional_avg_service_complexity
  FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
  WHERE currentRatings.overall.rating IS NOT NULL
  GROUP BY region
),

service_complexity_metrics AS (
  -- Advanced service complexity analysis
  SELECT
    locationId,
    -- Service diversity score
    (COALESCE(ARRAY_LENGTH(regulatedActivities), 0) + 
     COALESCE(ARRAY_LENGTH(gacServiceTypes), 0) + 
     COALESCE(ARRAY_LENGTH(specialisms), 0) + 
     COALESCE(ARRAY_LENGTH(serviceUserBands), 0)) as total_service_complexity,
    -- Weighted complexity score
    (COALESCE(ARRAY_LENGTH(regulatedActivities), 0) * 1.0 + 
     COALESCE(ARRAY_LENGTH(gacServiceTypes), 0) * 0.8 + 
     COALESCE(ARRAY_LENGTH(specialisms), 0) * 1.2 + 
     COALESCE(ARRAY_LENGTH(serviceUserBands), 0) * 0.6) as weighted_service_complexity
  FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
)

-- Main feature selection and final transformations
SELECT 
  -- Primary identifiers
  l.locationId,
  l.name,
  l.providerId,
  
  -- Core operational features
  COALESCE(l.numberOfBeds, 0) as bed_capacity,
  CASE 
    WHEN l.numberOfBeds >= 60 THEN 4  -- Very Large
    WHEN l.numberOfBeds >= 40 THEN 3  -- Large  
    WHEN l.numberOfBeds >= 20 THEN 2  -- Medium
    WHEN l.numberOfBeds > 0 THEN 1    -- Small
    ELSE 0  -- Unknown
  END as facility_size_numeric,
  
  -- Temporal risk indicators
  COALESCE(l.days_since_inspection, 9999) as days_since_inspection,
  COALESCE(l.days_since_registration, 0) as days_since_registration,
  -- Inspection overdue risk (>2 years)
  CASE WHEN l.days_since_inspection > 730 THEN 1 ELSE 0 END as inspection_overdue,
  
  -- Historical performance context
  COALESCE(ih.inspection_count, 1) as inspection_history_count,
  COALESCE(ih.historical_avg_rating_numeric, 3.0) as historical_performance,
  COALESCE(ih.rating_volatility, 0.5) as performance_consistency,
  COALESCE(ih.unique_inspection_dates, 1) as inspection_frequency,
  -- Performance stability indicator
  CASE WHEN ih.rating_volatility > 1.0 THEN 1 ELSE 0 END as performance_unstable,
  
  -- Service complexity features
  COALESCE(scm.total_service_complexity, 3) as service_complexity_score,
  COALESCE(scm.weighted_service_complexity, 3.0) as weighted_service_complexity,
  COALESCE(l.regulated_activities_count, 1) as regulated_activities_count,
  COALESCE(l.specialisms_count, 0) as specialisms_count,
  
  -- Provider context features
  COALESCE(pc.provider_location_count, 1) as provider_scale,
  COALESCE(pc.provider_avg_rating, 3.0) as provider_reputation,
  COALESCE(pc.provider_geographic_spread, 1) as provider_diversity,
  COALESCE(pc.provider_total_capacity, l.numberOfBeds) as provider_total_capacity,
  COALESCE(pc.provider_rating_consistency, 0.5) as provider_rating_consistency,
  
  -- Regional context features  
  COALESCE(rc.regional_avg_rating, 3.0) as regional_avg_rating,
  COALESCE(rc.regional_risk_rate, 0.3) as regional_risk_rate,
  COALESCE(rc.regional_avg_beds, 30.0) as regional_avg_beds,
  COALESCE(rc.regional_avg_service_complexity, 3.0) as regional_avg_service_complexity,
  
  -- Geographic identifiers
  l.region,
  l.localAuthority,
  l.organisationType,
  l.locationType,
  
  -- Derived interaction features
  COALESCE(scm.total_service_complexity, 3) * COALESCE(pc.provider_location_count, 1) as complexity_scale_interaction,
  CASE WHEN l.days_since_inspection > 730 THEN 1 ELSE 0 END * COALESCE(rc.regional_risk_rate, 0.3) as inspection_regional_risk,
  
  -- Risk scoring composite features
  CASE 
    WHEN l.days_since_inspection > 1095 THEN 0.9  -- >3 years
    WHEN l.days_since_inspection > 730 THEN 0.6   -- >2 years
    WHEN l.days_since_inspection > 365 THEN 0.3   -- >1 year
    ELSE 0.1
  END as inspection_overdue_risk_score,
  
  -- Quality indicators from current ratings
  l.safe_rating,
  l.effective_rating,
  l.caring_rating,
  l.responsive_rating,
  l.well_led_rating,
  
  -- Target variables (numeric encoding)
  CASE 
    WHEN l.overall_rating = 'Outstanding' THEN 4
    WHEN l.overall_rating = 'Good' THEN 3  
    WHEN l.overall_rating = 'Requires improvement' THEN 2
    WHEN l.overall_rating = 'Inadequate' THEN 1
    ELSE NULL
  END as overall_rating_numeric,
  
  l.overall_rating as overall_rating_text,
  
  -- Data quality indicators
  CASE WHEN l.overall_rating IS NOT NULL THEN 1 ELSE 0 END as has_rating,
  CASE WHEN l.numberOfBeds IS NOT NULL AND l.numberOfBeds > 0 THEN 1 ELSE 0 END as has_capacity,
  CASE WHEN l.lastInspectionDate IS NOT NULL THEN 1 ELSE 0 END as has_inspection_date,
  
  -- Feature completeness score
  (CASE WHEN l.overall_rating IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN l.numberOfBeds IS NOT NULL AND l.numberOfBeds > 0 THEN 1 ELSE 0 END +
   CASE WHEN l.lastInspectionDate IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN l.region IS NOT NULL THEN 1 ELSE 0 END +
   CASE WHEN l.localAuthority IS NOT NULL THEN 1 ELSE 0 END) / 5.0 as feature_completeness_score,
  
  -- Processing metadata
  CURRENT_TIMESTAMP() as feature_extraction_timestamp

FROM location_data l
LEFT JOIN inspection_history ih ON l.locationId = ih.locationId
LEFT JOIN provider_context pc ON l.providerId = pc.providerId
LEFT JOIN regional_context rc ON l.region = rc.region
LEFT JOIN service_complexity_metrics scm ON l.locationId = scm.locationId

-- Filter for locations with minimum required data for ML training
WHERE l.overall_rating IS NOT NULL  -- Must have target variable
  AND l.region IS NOT NULL          -- Geographic context required
  AND (l.numberOfBeds IS NULL OR l.numberOfBeds >= 0)  -- Valid capacity data

-- Order for consistent processing
ORDER BY l.locationId;

-- Create additional indexes for query performance
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_training_ready` AS
SELECT * FROM `machine-learning-exp-467008.cqc_data.ml_training_features_comprehensive`
WHERE overall_rating_numeric IS NOT NULL
  AND feature_completeness_score >= 0.6  -- At least 60% feature completeness
  AND bed_capacity > 0;  -- Valid operational data

-- Summary statistics view for monitoring
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_summary` AS
SELECT
  COUNT(*) as total_locations,
  COUNTIF(overall_rating_numeric IS NOT NULL) as locations_with_rating,
  COUNTIF(feature_completeness_score >= 0.8) as high_quality_locations,
  AVG(feature_completeness_score) as avg_completeness_score,
  COUNT(DISTINCT region) as regions_covered,
  COUNT(DISTINCT providerId) as providers_covered,
  MIN(feature_extraction_timestamp) as extraction_start,
  MAX(feature_extraction_timestamp) as extraction_end
FROM `machine-learning-exp-467008.cqc_data.ml_training_features_comprehensive`;