#!/bin/bash

# Fetch real CQC data using the API

PROJECT_ID="machine-learning-exp-467008"
REGION="europe-west2"
API_KEY=$(gcloud secrets versions access latest --secret="cqc-subscription-key")

echo "Fetching real CQC data..."
echo "========================"

# Test API access first
echo "1. Testing API access..."
RESPONSE=$(curl -s -w "\nHTTP_STATUS:%{http_code}" \
  -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "https://api.service.cqc.org.uk/public/v1/providers?page=1&perPage=1")

HTTP_STATUS=$(echo "$RESPONSE" | grep "HTTP_STATUS:" | cut -d: -f2)
BODY=$(echo "$RESPONSE" | sed -n '1,/HTTP_STATUS:/p' | sed '$d')

if [ "$HTTP_STATUS" != "200" ]; then
    echo "❌ API access failed with status: $HTTP_STATUS"
    echo "Response: $BODY"
    exit 1
fi

echo "✅ API access successful!"
TOTAL_PROVIDERS=$(echo "$BODY" | jq -r '.total')
echo "Total providers available: $TOTAL_PROVIDERS"

# Fetch sample data for testing
echo
echo "2. Fetching sample data (100 providers and locations)..."

# Fetch providers
echo "Fetching providers..."
curl -s -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "https://api.service.cqc.org.uk/public/v1/providers?page=1&perPage=100" \
  -o /tmp/providers_sample.json

PROVIDER_COUNT=$(jq '.providers | length' /tmp/providers_sample.json)
echo "✅ Fetched $PROVIDER_COUNT providers"

# Fetch locations
echo "Fetching locations..."
curl -s -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "https://api.service.cqc.org.uk/public/v1/locations?page=1&perPage=100" \
  -o /tmp/locations_sample.json

LOCATION_COUNT=$(jq '.locations | length' /tmp/locations_sample.json)
echo "✅ Fetched $LOCATION_COUNT locations"

# Upload to GCS
echo
echo "3. Uploading to Cloud Storage..."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
gsutil cp /tmp/providers_sample.json "gs://${PROJECT_ID}-cqc-raw-data/real_data/providers/${TIMESTAMP}_providers_real.json"
gsutil cp /tmp/locations_sample.json "gs://${PROJECT_ID}-cqc-raw-data/real_data/locations/${TIMESTAMP}_locations_real.json"

echo "✅ Data uploaded to GCS"

# Show sample of the data
echo
echo "4. Sample of fetched data:"
echo "Providers sample:"
jq '.providers[0]' /tmp/providers_sample.json | head -20

echo
echo "Locations sample:"
jq '.locations[0]' /tmp/locations_sample.json | head -20

# Clean up
rm /tmp/providers_sample.json /tmp/locations_sample.json

echo
echo "========================"
echo "✅ Real CQC data fetched successfully!"
echo "Files saved to: gs://${PROJECT_ID}-cqc-raw-data/real_data/"
echo
echo "Next steps:"
echo "1. Convert to NDJSON format"
echo "2. Load into BigQuery"
echo "3. Retrain ML model with real data"