-- Load CQC data from Cloud Storage directly into BigQuery

-- First, create the dataset if it doesn't exist
CREATE SCHEMA IF NOT EXISTS `machine-learning-exp-467008.cqc_data`;

-- Create external tables pointing to our CQC data in Cloud Storage
CREATE OR REPLACE EXTERNAL TABLE `machine-learning-exp-467008.cqc_data.cqc_data_external_1`
OPTIONS (
  format = 'NEWLINE_DELIMITED_JSON',
  uris = ['gs://machine-learning-exp-467008-cqc-raw-data/cloud_build/719d6c95-323c-4944-829c-cccf2ce6f94c_locations_detailed.json']
);

CREATE OR REPLACE EXTERNAL TABLE `machine-learning-exp-467008.cqc_data.cqc_data_external_2`
OPTIONS (
  format = 'NEWLINE_DELIMITED_JSON',
  uris = ['gs://machine-learning-exp-467008-cqc-raw-data/processed/detailed_locations_20250729_094046.json']
);

-- Create the main locations_detailed table by combining data from both sources
CREATE OR REPLACE TABLE `machine-learning-exp-467008.cqc_data.locations_detailed` AS
WITH combined_data AS (
  SELECT * FROM `machine-learning-exp-467008.cqc_data.cqc_data_external_1`
  UNION ALL
  SELECT * FROM `machine-learning-exp-467008.cqc_data.cqc_data_external_2`
)
SELECT DISTINCT
  locationId,
  name,
  type,
  CAST(numberOfBeds AS INT64) as numberOfBeds,
  PARSE_DATE('%Y-%m-%d', SUBSTR(registrationDate, 1, 10)) as registrationDate,
  postalCode,
  region,
  localAuthority,
  PARSE_DATE('%Y-%m-%d', SUBSTR(IFNULL(lastInspectionDate, registrationDate), 1, 10)) as lastInspectionDate,
  providerId,
  ARRAY_LENGTH(IFNULL(regulatedActivities, [])) as regulatedActivitiesCount,
  ARRAY_LENGTH(IFNULL(specialisms, [])) as specialismsCount,
  ARRAY_LENGTH(IFNULL(gacServiceTypes, [])) as serviceTypesCount,
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
  'cqc_api' as dataSource
FROM combined_data
WHERE locationId IS NOT NULL;

-- Check the count
SELECT 
  COUNT(*) as total_locations,
  COUNT(DISTINCT locationId) as unique_locations 
FROM `machine-learning-exp-467008.cqc_data.locations_detailed`;

-- Create the ML features view for proactive rating prediction
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_proactive` AS
SELECT
  locationId,
  name,
  type,
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
  
  -- Additional features for ML
  CASE 
    WHEN numberOfBeds > 50 THEN 'Large'
    WHEN numberOfBeds > 20 THEN 'Medium'
    WHEN numberOfBeds > 0 THEN 'Small'
    ELSE 'No Beds'
  END as size_category,
  
  overallRating,
  safeRating,
  effectiveRating,
  caringRating,
  responsiveRating,
  wellLedRating
  
FROM `machine-learning-exp-467008.cqc_data.locations_detailed`
WHERE overallRating IS NOT NULL;

-- Verify the ML features view
SELECT 
  COUNT(*) as total_records,
  SUM(at_risk_label) as at_risk_count,
  ROUND(AVG(domain_risk_score), 2) as avg_risk_score,
  COUNT(DISTINCT region) as regions
FROM `machine-learning-exp-467008.cqc_data.ml_features_proactive`;

-- Show rating distribution
SELECT 
  overallRating,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
FROM `machine-learning-exp-467008.cqc_data.ml_features_proactive`
GROUP BY overallRating
ORDER BY 
  CASE overallRating
    WHEN 'Outstanding' THEN 1
    WHEN 'Good' THEN 2
    WHEN 'Requires improvement' THEN 3
    WHEN 'Inadequate' THEN 4
    ELSE 5
  END;

-- Clean up external tables (optional)
-- DROP EXTERNAL TABLE IF EXISTS `machine-learning-exp-467008.cqc_data.cqc_data_external_1`;
-- DROP EXTERNAL TABLE IF EXISTS `machine-learning-exp-467008.cqc_data.cqc_data_external_2`;