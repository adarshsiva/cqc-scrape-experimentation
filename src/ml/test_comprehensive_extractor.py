#!/usr/bin/env python3
"""
Test script for Comprehensive CQC Extractor

This script allows you to test the extractor locally or validate
specific endpoints before running the full extraction.
"""

import os
import asyncio
import logging
from comprehensive_cqc_extractor import ComprehensiveCQCExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_api_connection():
    """Test basic API connectivity"""
    print("Testing CQC API connection...")
    
    # Set test environment variables
    os.environ.setdefault('GCP_PROJECT', 'machine-learning-exp-467008')
    os.environ.setdefault('ENDPOINTS', 'locations')
    os.environ.setdefault('MAX_LOCATIONS', '10')
    os.environ.setdefault('PARALLEL_WORKERS', '2')
    os.environ.setdefault('RATE_LIMIT', '100')
    
    try:
        extractor = ComprehensiveCQCExtractor()
        
        # Test fetching a small number of locations
        print("Fetching 10 locations for testing...")
        locations = await extractor.fetch_all_locations()
        
        if locations:
            print(f"✓ Successfully fetched {len(locations)} locations")
            
            # Test feature extraction on first location
            if len(locations) > 0:
                print("\nTesting feature extraction...")
                sample_location = locations[0]
                features = extractor.extract_comprehensive_features(sample_location)
                
                print(f"✓ Extracted {len(features)} features")
                print("Sample features:")
                for key, value in list(features.items())[:5]:
                    print(f"  {key}: {value}")
                
                return True
        else:
            print("✗ No locations fetched")
            return False
            
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

async def test_specific_endpoint(endpoint_name: str, location_id: str = None):
    """Test a specific endpoint"""
    print(f"Testing endpoint: {endpoint_name}")
    
    try:
        extractor = ComprehensiveCQCExtractor()
        
        if endpoint_name == "location_details" and location_id:
            data = await extractor.fetch_location_details(location_id)
        elif endpoint_name == "provider_details" and location_id:
            data = await extractor.fetch_provider_details(location_id)
        elif endpoint_name == "inspection_areas":
            data = await extractor.fetch_inspection_areas()
        else:
            print(f"Unknown endpoint: {endpoint_name}")
            return False
        
        if data:
            print(f"✓ Successfully fetched data from {endpoint_name}")
            print(f"  Data keys: {list(data.keys())}")
            return True
        else:
            print(f"✗ No data from {endpoint_name}")
            return False
            
    except Exception as e:
        print(f"✗ Test failed for {endpoint_name}: {e}")
        return False

def validate_configuration():
    """Validate the current configuration"""
    print("Validating configuration...")
    
    required_vars = ['GCP_PROJECT']
    optional_vars = ['CQC_API_KEY', 'ENDPOINTS', 'MAX_LOCATIONS']
    
    print("\nRequired environment variables:")
    for var in required_vars:
        value = os.environ.get(var)
        if value:
            print(f"  ✓ {var}: {value}")
        else:
            print(f"  ✗ {var}: Not set")
    
    print("\nOptional environment variables:")
    for var in optional_vars:
        value = os.environ.get(var)
        print(f"  {var}: {value or 'Not set (using default)'}")
    
    # Check for CQC API key
    try:
        from google.cloud import secretmanager
        project_id = os.environ.get('GCP_PROJECT')
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/cqc-subscription-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        print(f"  ✓ CQC API key found in Secret Manager")
    except:
        api_key = os.environ.get('CQC_API_KEY')
        if api_key:
            print(f"  ✓ CQC API key found in environment")
        else:
            print(f"  ✗ No CQC API key found in Secret Manager or environment")

async def main():
    """Main test function"""
    print("=" * 60)
    print("Comprehensive CQC Extractor - Test Suite")
    print("=" * 60)
    
    # 1. Validate configuration
    validate_configuration()
    print()
    
    # 2. Test API connection
    api_test_passed = await test_api_connection()
    print()
    
    if api_test_passed:
        print("✓ All tests passed! The extractor is ready to use.")
    else:
        print("✗ Some tests failed. Please check the configuration.")
    
    print("\n" + "=" * 60)
    print("Test complete.")

if __name__ == "__main__":
    asyncio.run(main())