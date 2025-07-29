-- Create table for detailed location data
CREATE OR REPLACE TABLE `cqc_data.locations_detailed` AS
SELECT
  JSON_EXTRACT_SCALAR(data, '$.location.locationId') as locationId,
  JSON_EXTRACT_SCALAR(data, '$.location.name') as name,
  CAST(JSON_EXTRACT_SCALAR(data, '$.location.numberOfBeds') AS INT64) as numberOfBeds,
  DATE(JSON_EXTRACT_SCALAR(data, '$.location.registrationDate')) as registrationDate,
  DATE(JSON_EXTRACT_SCALAR(data, '$.location.lastInspectionDate')) as lastInspectionDate,
  JSON_EXTRACT_SCALAR(data, '$.location.postalCode') as postalCode,
  JSON_EXTRACT_SCALAR(data, '$.location.region') as region,
  JSON_EXTRACT_SCALAR(data, '$.location.localAuthority') as localAuthority,
  JSON_EXTRACT_SCALAR(data, '$.location.providerId') as providerId,
  JSON_EXTRACT_SCALAR(data, '$.location.type') as locationType,
  
  -- Ratings
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.overall.rating') as overallRating,
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.safe.rating') as safeRating,
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.effective.rating') as effectiveRating,
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.caring.rating') as caringRating,
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.responsive.rating') as responsiveRating,
  JSON_EXTRACT_SCALAR(data, '$.location.currentRatings.wellLed.rating') as wellLedRating,
  
  -- Service details
  ARRAY_LENGTH(JSON_EXTRACT_ARRAY(data, '$.location.regulatedActivities')) as regulatedActivitiesCount,
  ARRAY_LENGTH(JSON_EXTRACT_ARRAY(data, '$.location.specialisms')) as specialismsCount,
  ARRAY_LENGTH(JSON_EXTRACT_ARRAY(data, '$.location.gacServiceTypes')) as serviceTypesCount,
  
  -- Raw JSON for additional processing
  data as rawData
FROM (
  SELECT JSON_EXTRACT(content, '$[*]') as data
  FROM `cqc_data.raw_detailed_locations`
);

-- Create ML features view for proactive detection
CREATE OR REPLACE VIEW `cqc_data.ml_features_proactive` AS
SELECT
  locationId,
  
  -- Basic features
  IFNULL(numberOfBeds, 0) as number_of_beds,
  DATE_DIFF(CURRENT_DATE(), registrationDate, DAY) as days_since_registration,
  DATE_DIFF(CURRENT_DATE(), lastInspectionDate, DAY) as days_since_inspection,
  
  -- Service complexity
  regulatedActivitiesCount,
  specialismsCount,
  serviceTypesCount,
  
  -- Location features
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
  END as at_risk_label
  
FROM `cqc_data.locations_detailed`
WHERE overallRating IS NOT NULL;