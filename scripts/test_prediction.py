#!/usr/bin/env python3
"""
Test script for CQC rating prediction service
"""
import json
import requests
import subprocess
import sys

def get_auth_token():
    """Get authentication token for Cloud Run service"""
    try:
        result = subprocess.run(
            ["gcloud", "auth", "print-identity-token"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get auth token: {e}")
        sys.exit(1)

def test_single_prediction(service_url, auth_token):
    """Test single prediction"""
    print("\n=== Testing Single Prediction ===")
    
    test_data = {
        "location_id": "test-001",
        "number_of_beds": 50,
        "number_of_locations": 1,
        "inspection_history_length": 3,
        "days_since_last_inspection": 180,
        "ownership_type": "Organisation",
        "service_types": ["Care home service with nursing"],
        "specialisms": ["Dementia"],
        "region": "London",
        "local_authority": "LA-123",
        "constituency": "Constituency-456",
        "regulated_activities": ["Accommodation for persons who require nursing or personal care"],
        "service_user_groups": ["Older people"],
        "has_previous_rating": True,
        "previous_rating": "Good",
        "ownership_changed_recently": False,
        "nominated_individual_exists": True
    }
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(service_url, json=test_data, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def test_batch_prediction(service_url, auth_token):
    """Test batch prediction"""
    print("\n=== Testing Batch Prediction ===")
    
    test_data = {
        "instances": [
            {
                "location_id": "batch-001",
                "number_of_beds": 30,
                "number_of_locations": 1,
                "inspection_history_length": 2,
                "days_since_last_inspection": 90,
                "ownership_type": "Individual",
                "service_types": ["Domiciliary care service"],
                "specialisms": [],
                "region": "North West",
                "local_authority": "LA-456",
                "constituency": "Constituency-789",
                "regulated_activities": ["Personal care"],
                "service_user_groups": ["Adults"],
                "has_previous_rating": True,
                "previous_rating": "Requires improvement",
                "ownership_changed_recently": True,
                "nominated_individual_exists": False
            },
            {
                "location_id": "batch-002",
                "number_of_beds": 100,
                "number_of_locations": 3,
                "inspection_history_length": 5,
                "days_since_last_inspection": 365,
                "ownership_type": "Organisation",
                "service_types": ["Care home service with nursing", "Rehabilitation services"],
                "specialisms": ["Dementia", "Mental health"],
                "region": "South East",
                "local_authority": "LA-789",
                "constituency": "Constituency-012",
                "regulated_activities": ["Accommodation for persons who require nursing or personal care", "Treatment of disease"],
                "service_user_groups": ["Older people", "People with mental health needs"],
                "has_previous_rating": True,
                "previous_rating": "Outstanding",
                "ownership_changed_recently": False,
                "nominated_individual_exists": True
            }
        ]
    }
    
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }
    
    response = requests.post(service_url, json=test_data, headers=headers)
    
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200

def main():
    # Get service URL
    try:
        result = subprocess.run(
            ["gcloud", "run", "services", "describe", "cqc-rating-prediction", 
             "--region=europe-west2", "--format=value(status.url)"],
            capture_output=True,
            text=True,
            check=True
        )
        service_url = result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get service URL: {e}")
        sys.exit(1)
    
    print(f"Service URL: {service_url}")
    
    # Get auth token
    auth_token = get_auth_token()
    
    # Run tests
    single_success = test_single_prediction(service_url, auth_token)
    batch_success = test_batch_prediction(service_url, auth_token)
    
    if single_success and batch_success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()