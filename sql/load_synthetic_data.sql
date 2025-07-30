-- Load synthetic data from GCS directly into BigQuery

-- First, create or replace the external table pointing to our synthetic data
CREATE OR REPLACE EXTERNAL TABLE `machine-learning-exp-467008.cqc_data.synthetic_data_external`
OPTIONS (
  format = 'JSON',
  uris = ['gs://machine-learning-exp-467008-cqc-raw-data/raw/locations/20250728_191714_locations_sample.json']
);

-- Create the main locations_detailed table with proper schema
CREATE OR REPLACE TABLE `machine-learning-exp-467008.cqc_data.locations_detailed` AS
SELECT
  locationId,
  name,
  type,
  CAST(numberOfBeds AS INT64) as numberOfBeds,
  PARSE_DATE('%Y-%m-%d', SUBSTR(registrationDate, 1, 10)) as registrationDate,
  postalCode,
  region,
  localAuthority,
  PARSE_DATE('%Y-%m-%d', SUBSTR(lastInspectionDate, 1, 10)) as lastInspectionDate,
  providerId,
  ARRAY_LENGTH(regulatedActivities) as regulatedActivitiesCount,
  ARRAY_LENGTH(specialisms) as specialismsCount,
  ARRAY_LENGTH(gacServiceTypes) as serviceTypesCount,
  currentRatings.overall.rating as overallRating,
  currentRatings.safe.rating as safeRating,
  currentRatings.effective.rating as effectiveRating,
  currentRatings.caring.rating as caringRating,
  currentRatings.responsive.rating as responsiveRating,
  currentRatings.wellLed.rating as wellLedRating,
  TO_JSON_STRING(STRUCT(
    locationId,
    name,
    type,
    numberOfBeds,
    registrationDate,
    postalCode,
    region,
    localAuthority,
    lastInspectionDate,
    providerId,
    regulatedActivities,
    specialisms,
    gacServiceTypes,
    currentRatings,
    inspectionHistory
  )) as rawData,
  CURRENT_TIMESTAMP() as fetchTimestamp,
  'synthetic' as dataSource
FROM `machine-learning-exp-467008.cqc_data.synthetic_data_external`;

-- Check the count
SELECT COUNT(*) as total_locations FROM `machine-learning-exp-467008.cqc_data.locations_detailed`;

-- Create the ML features view
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_proactive` AS
SELECT
  locationId,
  name,
  IFNULL(numberOfBeds, 0) as number_of_beds,
  DATE_DIFF(CURRENT_DATE(), registrationDate, DAY) as days_since_registration,
  DATE_DIFF(CURRENT_DATE(), lastInspectionDate, DAY) as days_since_inspection,
  regulatedActivitiesCount,
  specialismsCount,
  serviceTypesCount,
  region,
  localAuthority,
  
  -- Risk indicators
  CASE 
    WHEN DATE_DIFF(CURRENT_DATE(), lastInspectionDate, DAY) > 730 THEN 1 
    ELSE 0 
  END as overdue_inspection,
  
  -- Current ratings (numeric encoding)
  CASE overallRating
    WHEN 'Outstanding' THEN 1
    WHEN 'Good' THEN 2
    WHEN 'Requires improvement' THEN 3
    WHEN 'Inadequate' THEN 4
    ELSE 0
  END as overall_rating_score,
  
  -- Domain-specific risks
  CASE WHEN safeRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as safe_at_risk,
  CASE WHEN effectiveRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as effective_at_risk,
  CASE WHEN caringRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as caring_at_risk,
  CASE WHEN responsiveRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as responsive_at_risk,
  CASE WHEN wellLedRating IN ('Requires improvement', 'Inadequate') THEN 1 ELSE 0 END as well_led_at_risk,
  
  -- Composite risk score
  (
    CASE WHEN safeRating IN ('Requires improvement', 'Inadequate') THEN 0.25 ELSE 0 END +
    CASE WHEN effectiveRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END +
    CASE WHEN caringRating IN ('Requires improvement', 'Inadequate') THEN 0.15 ELSE 0 END +
    CASE WHEN responsiveRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END +
    CASE WHEN wellLedRating IN ('Requires improvement', 'Inadequate') THEN 0.20 ELSE 0 END
  ) * 100 as domain_risk_score,
  
  -- Target variable
  CASE 
    WHEN overallRating IN ('Requires improvement', 'Inadequate') THEN 1
    ELSE 0
  END as at_risk_label,
  
  overallRating,
  safeRating,
  effectiveRating,
  caringRating,
  responsiveRating,
  wellLedRating
  
FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
WHERE overallRating IS NOT NULL;

-- Check the ML features view
SELECT 
  COUNT(*) as total_records,
  SUM(at_risk_label) as at_risk_count,
  AVG(domain_risk_score) as avg_risk_score
FROM `machine-learning-exp-467008.cqc_data.ml_features_proactive`;