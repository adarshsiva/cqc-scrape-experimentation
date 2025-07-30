#!/bin/bash
# Run BigQuery SQL to load synthetic data

PROJECT_ID="machine-learning-exp-467008"

echo "Loading synthetic data into BigQuery..."

# Run the SQL file
bq query --use_legacy_sql=false --project_id=${PROJECT_ID} < ../sql/load_synthetic_data.sql

echo "Data loading complete!"