#!/usr/bin/env python3
"""
Integration test script for the Dashboard Metrics Collector Cloud Function.

This script tests the deployed function with various scenarios.
"""

import requests
import json
import time
import sys
import os

# Configuration
FUNCTION_URL = os.getenv('FUNCTION_URL', 'https://europe-west2-machine-learning-exp-467008.cloudfunctions.net/dashboard-metrics-collector')
TEST_PROVIDER_ID = "test-provider-123"
TEST_LOCATION_ID = "test-location-456"


def test_incident_collection():
    """Test collecting incident metrics."""
    print("🧪 Testing incident metrics collection...")
    
    payload = {
        "provider_id": TEST_PROVIDER_ID,
        "location_id": TEST_LOCATION_ID,
        "dashboard_type": "incidents"
    }
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Incident collection successful: {data}")
            return True
        else:
            print(f"❌ Incident collection failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during incident test: {e}")
        return False


def test_staffing_collection():
    """Test collecting staffing metrics."""
    print("🧪 Testing staffing metrics collection...")
    
    payload = {
        "provider_id": TEST_PROVIDER_ID,
        "dashboard_type": "staffing"
    }
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Staffing collection successful: {data}")
            return True
        else:
            print(f"❌ Staffing collection failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during staffing test: {e}")
        return False


def test_care_quality_collection():
    """Test collecting care quality metrics."""
    print("🧪 Testing care quality metrics collection...")
    
    payload = {
        "location_id": TEST_LOCATION_ID,
        "dashboard_type": "care_quality"
    }
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Care quality collection successful: {data}")
            return True
        else:
            print(f"❌ Care quality collection failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during care quality test: {e}")
        return False


def test_collect_all():
    """Test collecting all metric types."""
    print("🧪 Testing collection of all metrics...")
    
    payload = {
        "provider_id": TEST_PROVIDER_ID,
        "collect_all": True
    }
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60  # Longer timeout for all metrics
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Collect all successful: {data}")
            
            # Verify we got metrics for all types
            if data.get("metrics_collected", 0) >= 3:
                print("✅ All metric types collected")
                return True
            else:
                print("⚠️  Not all metric types collected")
                return False
        else:
            print(f"❌ Collect all failed: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during collect all test: {e}")
        return False


def test_error_handling():
    """Test error handling with invalid requests."""
    print("🧪 Testing error handling...")
    
    # Test missing required fields
    payload = {
        "dashboard_type": "incidents"
        # Missing both provider_id and location_id
    }
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        
        if response.status_code == 400:
            print("✅ Error handling successful: Correctly rejected invalid request")
            return True
        else:
            print(f"❌ Error handling failed: Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during error handling test: {e}")
        return False


def test_no_json_body():
    """Test handling of missing JSON body."""
    print("🧪 Testing no JSON body handling...")
    
    try:
        response = requests.post(
            FUNCTION_URL,
            headers={"Content-Type": "application/json"},
            timeout=30
            # No JSON body
        )
        
        if response.status_code == 400:
            print("✅ No JSON body handling successful")
            return True
        else:
            print(f"❌ No JSON body handling failed: Expected 400, got {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Exception during no JSON body test: {e}")
        return False


def main():
    """Run all integration tests."""
    print(f"🚀 Starting integration tests for Cloud Function: {FUNCTION_URL}")
    print("=" * 60)
    
    tests = [
        test_incident_collection,
        test_staffing_collection, 
        test_care_quality_collection,
        test_collect_all,
        test_error_handling,
        test_no_json_body
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
            print("-" * 40)
            time.sleep(1)  # Brief pause between tests
        except KeyboardInterrupt:
            print("\n🛑 Tests interrupted by user")
            break
    
    print("=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
        sys.exit(0)
    else:
        print(f"⚠️  {total - passed} tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()