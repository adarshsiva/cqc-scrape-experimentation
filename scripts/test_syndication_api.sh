#!/bin/bash

echo "Testing CQC Syndication API Access"
echo "=================================="

# Get API key from Secret Manager
API_KEY=$(gcloud secrets versions access latest --secret="cqc-subscription-key")
echo "API Key (first 10 chars): ${API_KEY:0:10}..."
echo

# Base URLs to test
SYNDICATION_BASE="https://api.cqc.org.uk/syndication/v1"
PUBLIC_BASE="https://api.cqc.org.uk/public/v1"

echo "Testing Syndication API endpoints:"
echo "----------------------------------"

# Test syndication endpoints
echo "1. Testing /providers endpoint:"
curl -s -w "\nStatus: %{http_code}\n" \
  -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "$SYNDICATION_BASE/providers?page=1&perPage=1" | head -20

echo
echo "2. Testing /locations endpoint:"
curl -s -w "\nStatus: %{http_code}\n" \
  -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "$SYNDICATION_BASE/locations?page=1&perPage=1" | head -20

echo
echo "3. Testing with subscription-key header:"
curl -s -w "\nStatus: %{http_code}\n" \
  -H "subscription-key: $API_KEY" \
  "$SYNDICATION_BASE/providers?page=1&perPage=1" | head -20

echo
echo "Testing Public API endpoints for comparison:"
echo "-------------------------------------------"

echo "4. Public API /providers:"
curl -s -w "\nStatus: %{http_code}\n" \
  -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "$PUBLIC_BASE/providers?page=1&perPage=1" | head -20

echo
echo "5. Testing without any path (base URL):"
curl -s -w "\nStatus: %{http_code}\n" \
  -H "Ocp-Apim-Subscription-Key: $API_KEY" \
  "https://api.cqc.org.uk/" | head -20

echo
echo "=================================="
echo "If all tests return 403 Forbidden:"
echo "1. The API key may not be activated"
echo "2. Check your email for activation link"
echo "3. The syndication API may require special permissions"
echo "4. Contact CQC support for syndication API access"