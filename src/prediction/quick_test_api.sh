#!/bin/bash
# Quick test script for deployed Proactive Risk Assessment API

# Get the service URL
SERVICE_URL=$(gcloud run services describe proactive-risk-assessment \
    --region=europe-west2 \
    --format='value(status.url)' 2>/dev/null)

if [ -z "$SERVICE_URL" ]; then
    echo "❌ Error: Could not get service URL. Is the service deployed?"
    exit 1
fi

echo "Testing Proactive Risk Assessment API"
echo "Service URL: $SERVICE_URL"
echo "======================================"
echo ""

# Test 1: Health check
echo "1. Testing health endpoint..."
curl -s "$SERVICE_URL/health" | python -m json.tool
echo ""

# Test 2: Risk thresholds
echo "2. Testing risk thresholds endpoint..."
curl -s "$SERVICE_URL/risk-thresholds" | python -m json.tool
echo ""

# Test 3: Single location assessment
echo "3. Testing single location risk assessment..."
curl -s -X POST "$SERVICE_URL/assess-risk" \
    -H "Content-Type: application/json" \
    -d '{
        "locationId": "1-TEST123",
        "locationName": "Test Care Home",
        "region": "London",
        "staff_vacancy_rate": 0.25,
        "staff_turnover_rate": 0.35,
        "occupancy_rate": 0.85,
        "inspection_days_since_last": 450,
        "total_complaints": 2,
        "enforcement_actions": 1,
        "safe_key_questions_yes_ratio": 0.6,
        "effective_key_questions_yes_ratio": 0.7,
        "caring_key_questions_yes_ratio": 0.8,
        "responsive_key_questions_yes_ratio": 0.75,
        "well_led_key_questions_yes_ratio": 0.5,
        "current_rating": "Requires improvement",
        "provider_rating": "Requires improvement",
        "provider_total_locations": 5,
        "provider_good_outstanding_ratio": 0.4
    }' | python -m json.tool
echo ""

# Test 4: Batch assessment
echo "4. Testing batch assessment..."
curl -s -X POST "$SERVICE_URL/batch-assess" \
    -H "Content-Type: application/json" \
    -d '{
        "locations": [
            {
                "locationId": "1-BATCH001",
                "locationName": "Low Risk Home",
                "staff_vacancy_rate": 0.05,
                "staff_turnover_rate": 0.1,
                "inspection_days_since_last": 180,
                "total_complaints": 0,
                "safe_key_questions_yes_ratio": 0.95,
                "effective_key_questions_yes_ratio": 0.9,
                "caring_key_questions_yes_ratio": 0.95,
                "responsive_key_questions_yes_ratio": 0.9,
                "well_led_key_questions_yes_ratio": 0.85
            },
            {
                "locationId": "1-BATCH002",
                "locationName": "High Risk Home",
                "staff_vacancy_rate": 0.4,
                "staff_turnover_rate": 0.5,
                "inspection_days_since_last": 600,
                "total_complaints": 5,
                "safe_key_questions_yes_ratio": 0.4,
                "effective_key_questions_yes_ratio": 0.5,
                "caring_key_questions_yes_ratio": 0.6,
                "responsive_key_questions_yes_ratio": 0.5,
                "well_led_key_questions_yes_ratio": 0.3
            }
        ]
    }' | python -m json.tool

echo ""
echo "======================================"
echo "✓ All tests completed!"
echo ""
echo "If all tests passed, the API is working correctly."
echo "For more comprehensive testing, use: python test_proactive_api.py"