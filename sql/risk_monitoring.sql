-- Create risk monitoring table for storing assessment results
CREATE OR REPLACE TABLE `cqc_data.risk_assessments` (
  locationId STRING,
  locationName STRING,
  assessmentDate DATE,
  riskScore FLOAT64,
  riskLevel STRING,
  topRiskFactors ARRAY<STRING>,
  recommendations ARRAY<STRING>
)
PARTITION BY assessmentDate;

-- Daily risk summary view for monitoring dashboard
CREATE OR REPLACE VIEW `cqc_data.daily_risk_summary` AS
SELECT
  assessmentDate,
  riskLevel,
  COUNT(*) as location_count,
  AVG(riskScore) as avg_risk_score,
  ARRAY_AGG(DISTINCT region IGNORE NULLS) as regions_affected
FROM `cqc_data.risk_assessments` r
JOIN `cqc_data.locations_detailed` l ON r.locationId = l.locationId
WHERE assessmentDate = CURRENT_DATE()
GROUP BY assessmentDate, riskLevel;

-- Additional monitoring queries

-- High risk locations requiring immediate attention
CREATE OR REPLACE VIEW `cqc_data.high_risk_locations` AS
SELECT
  r.locationId,
  l.name as locationName,
  l.region,
  l.localAuthority,
  r.riskScore,
  r.topRiskFactors,
  r.recommendations,
  l.lastInspectionDate,
  DATE_DIFF(CURRENT_DATE(), l.lastInspectionDate, DAY) as days_since_inspection
FROM `cqc_data.risk_assessments` r
JOIN `cqc_data.locations_detailed` l ON r.locationId = l.locationId
WHERE r.assessmentDate = CURRENT_DATE()
  AND r.riskLevel = 'HIGH'
ORDER BY r.riskScore DESC;

-- Risk trend analysis by region
CREATE OR REPLACE VIEW `cqc_data.regional_risk_trends` AS
SELECT
  l.region,
  r.assessmentDate,
  COUNT(DISTINCT r.locationId) as total_locations,
  COUNTIF(r.riskLevel = 'HIGH') as high_risk_count,
  COUNTIF(r.riskLevel = 'MEDIUM') as medium_risk_count,
  COUNTIF(r.riskLevel = 'LOW') as low_risk_count,
  AVG(r.riskScore) as avg_risk_score,
  ROUND(COUNTIF(r.riskLevel = 'HIGH') / COUNT(*) * 100, 2) as high_risk_percentage
FROM `cqc_data.risk_assessments` r
JOIN `cqc_data.locations_detailed` l ON r.locationId = l.locationId
WHERE r.assessmentDate >= DATE_SUB(CURRENT_DATE(), INTERVAL 30 DAY)
GROUP BY l.region, r.assessmentDate
ORDER BY l.region, r.assessmentDate DESC;

-- Risk factor frequency analysis
CREATE OR REPLACE VIEW `cqc_data.risk_factor_analysis` AS
WITH risk_factors_unnested AS (
  SELECT
    assessmentDate,
    risk_factor
  FROM `cqc_data.risk_assessments`,
  UNNEST(topRiskFactors) as risk_factor
  WHERE assessmentDate = CURRENT_DATE()
)
SELECT
  risk_factor,
  COUNT(*) as frequency,
  ROUND(COUNT(*) / (SELECT COUNT(*) FROM `cqc_data.risk_assessments` WHERE assessmentDate = CURRENT_DATE()) * 100, 2) as percentage_of_locations
FROM risk_factors_unnested
GROUP BY risk_factor
ORDER BY frequency DESC;

-- Weekly risk assessment summary for reporting
CREATE OR REPLACE VIEW `cqc_data.weekly_risk_report` AS
SELECT
  DATE_TRUNC(assessmentDate, WEEK) as week_start,
  COUNT(DISTINCT locationId) as locations_assessed,
  COUNTIF(riskLevel = 'HIGH') as high_risk_count,
  COUNTIF(riskLevel = 'MEDIUM') as medium_risk_count,
  COUNTIF(riskLevel = 'LOW') as low_risk_count,
  AVG(riskScore) as avg_risk_score,
  MIN(riskScore) as min_risk_score,
  MAX(riskScore) as max_risk_score,
  APPROX_QUANTILES(riskScore, 100)[OFFSET(50)] as median_risk_score
FROM `cqc_data.risk_assessments`
WHERE assessmentDate >= DATE_SUB(CURRENT_DATE(), INTERVAL 90 DAY)
GROUP BY week_start
ORDER BY week_start DESC;