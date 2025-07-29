-- Create ML features view in BigQuery
CREATE OR REPLACE VIEW `machine-learning-exp-467008.cqc_data.ml_features_v1` AS
SELECT
  locationId,
  -- Basic features
  IFNULL(numberOfBeds, 0) as number_of_beds,
  1 as number_of_locations, -- Will be updated with JOIN to providers
  ARRAY_LENGTH(inspectionHistory) as inspection_history_length,
  DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_last_inspection,
  
  -- Categorical features (Note: ownershipType doesn't exist in schema, using type instead)
  IFNULL(type, 'Unknown') as ownership_type,
  ARRAY_LENGTH(gacServiceTypes) as service_types_count,
  ARRAY_LENGTH(specialisms) as specialisms_count,
  ARRAY_LENGTH(regulatedActivities) as regulated_activities_count,
  ARRAY_LENGTH(serviceUserBands) as service_user_groups_count,
  
  -- Location features
  region,
  localAuthority as local_authority,
  constituency,
  
  -- Rating features
  CASE WHEN currentRatings.overall.rating IS NOT NULL THEN TRUE ELSE FALSE END as has_previous_rating,
  IFNULL(currentRatings.overall.rating, 'No rating') as previous_rating,
  
  -- Additional features
  FALSE as ownership_changed_recently, -- Placeholder
  TRUE as nominated_individual_exists, -- Placeholder
  
  -- Target variable
  currentRatings.overall.rating as rating_label
FROM `machine-learning-exp-467008.cqc_data.locations_raw`
WHERE registrationStatus = 'Registered';