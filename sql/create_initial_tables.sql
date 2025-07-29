-- Create initial empty tables for CQC data

-- Create raw data table for detailed locations
CREATE OR REPLACE TABLE `cqc_data.raw_detailed_locations` (
  content STRING,
  load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Create staging table for locations
CREATE OR REPLACE TABLE `cqc_data.locations_staging` (
  locationId STRING,
  name STRING,
  numberOfBeds INT64,
  registrationDate DATE,
  lastInspectionDate DATE,
  postalCode STRING,
  region STRING,
  localAuthority STRING,
  providerId STRING,
  locationType STRING,
  overallRating STRING,
  safeRating STRING,
  effectiveRating STRING,
  caringRating STRING,
  responsiveRating STRING,
  wellLedRating STRING,
  regulatedActivitiesCount INT64,
  specialismsCount INT64,
  serviceTypesCount INT64,
  rawData STRING,
  load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Create provider information table
CREATE OR REPLACE TABLE `cqc_data.providers` (
  providerId STRING,
  providerName STRING,
  providerType STRING,
  locationsCount INT64,
  load_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Create locations_detailed table
CREATE OR REPLACE TABLE `cqc_data.locations_detailed` (
  locationId STRING,
  name STRING,
  numberOfBeds INT64,
  registrationDate DATE,
  lastInspectionDate DATE,
  postalCode STRING,
  region STRING,
  localAuthority STRING,
  providerId STRING,
  locationType STRING,
  overallRating STRING,
  safeRating STRING,
  effectiveRating STRING,
  caringRating STRING,
  responsiveRating STRING,
  wellLedRating STRING,
  regulatedActivitiesCount INT64,
  specialismsCount INT64,
  serviceTypesCount INT64,
  rawData STRING
);