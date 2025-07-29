#!/usr/bin/env python3
"""
Test script for Proactive Risk Assessment API
"""

import requests
import json
import sys
from datetime import datetime

# API endpoint (update with your Cloud Run URL)
BASE_URL = "http://localhost:8080"  # For local testing
# BASE_URL = "https://proactive-risk-assessment-xxxxx-ey.a.run.app"  # Cloud Run URL

def test_health_check():
    """Test the health check endpoint"""
    print("Testing health check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def test_single_assessment():
    """Test single location risk assessment"""
    print("Testing single location assessment...")
    
    # Sample location data
    test_location = {
        "locationId": "1-123456789",
        "locationName": "Test Care Home",
        "region": "London",
        "localAuthority": "Westminster",
        "provider_type": "Residential social care",
        "primary_inspection_category": "Residential social care",
        
        # Operational metrics
        "number_of_beds": 50,
        "staff_vacancy_rate": 0.25,  # 25% vacancy
        "staff_turnover_rate": 0.35,  # 35% turnover
        "occupancy_rate": 0.85,
        
        # Recent inspection data
        "current_rating": "Requires improvement",
        "inspection_days_since_last": 450,  # Long time since inspection
        "total_reports": 3,
        "enforcement_actions": 1,
        "total_complaints": 2,
        
        # Key question ratings
        "safe_key_questions_yes_ratio": 0.6,
        "effective_key_questions_yes_ratio": 0.7,
        "caring_key_questions_yes_ratio": 0.8,
        "responsive_key_questions_yes_ratio": 0.75,
        "well_led_key_questions_yes_ratio": 0.5,
        
        # Provider metrics
        "provider_rating": "Requires improvement",
        "provider_total_locations": 5,
        "provider_good_outstanding_ratio": 0.4
    }
    
    response = requests.post(f"{BASE_URL}/assess-risk", json=test_location)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        # Print summary
        print("\n--- Risk Assessment Summary ---")
        print(f"Location: {result['locationName']}")
        print(f"Risk Score: {result['riskScore']:.1f}%")
        print(f"Risk Level: {result['riskLevel']}")
        print(f"Confidence: {result['confidence']}")
        
        print("\nTop Risk Factors:")
        for i, factor in enumerate(result['topRiskFactors'], 1):
            print(f"{i}. {factor['factor']}: {factor['currentValue']:.2f}")
        
        print("\nRecommendations:")
        for i, rec in enumerate(result['recommendations'], 1):
            print(f"{i}. {rec}")
    else:
        print(f"Error: {response.text}")
    print()

def test_batch_assessment():
    """Test batch location assessment"""
    print("Testing batch assessment...")
    
    # Create multiple test locations
    locations = [
        {
            "locationId": f"1-10000000{i}",
            "locationName": f"Test Location {i}",
            "staff_vacancy_rate": 0.1 + i * 0.1,
            "staff_turnover_rate": 0.2 + i * 0.05,
            "inspection_days_since_last": 200 + i * 100,
            "total_complaints": i,
            "safe_key_questions_yes_ratio": 0.9 - i * 0.1,
            "effective_key_questions_yes_ratio": 0.85 - i * 0.1,
            "caring_key_questions_yes_ratio": 0.9 - i * 0.05,
            "responsive_key_questions_yes_ratio": 0.8 - i * 0.05,
            "well_led_key_questions_yes_ratio": 0.7 - i * 0.1,
        }
        for i in range(5)
    ]
    
    payload = {"locations": locations}
    response = requests.post(f"{BASE_URL}/batch-assess", json=payload)
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"Summary: {json.dumps(result['summary'], indent=2)}")
        
        print("\n--- Individual Assessments ---")
        for assessment in result['assessments']:
            print(f"\nLocation: {assessment['locationName']}")
            print(f"  Risk Score: {assessment['riskScore']:.1f}%")
            print(f"  Risk Level: {assessment['riskLevel']}")
    else:
        print(f"Error: {response.text}")
    print()

def test_risk_thresholds():
    """Test risk thresholds endpoint"""
    print("Testing risk thresholds...")
    response = requests.get(f"{BASE_URL}/risk-thresholds")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    print()

def main():
    """Run all tests"""
    print(f"Testing Proactive Risk Assessment API at {BASE_URL}")
    print("=" * 50)
    
    try:
        test_health_check()
        test_risk_thresholds()
        test_single_assessment()
        test_batch_assessment()
        
        print("All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to API at {BASE_URL}")
        print("Make sure the API is running.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()