"""
Test script for the DetailedCQCFetcher
"""

import os
import sys
import logging
from fetch_detailed_cqc_data import DetailedCQCFetcher

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_api_connection():
    """Test basic API connectivity"""
    import requests
    
    try:
        response = requests.get("https://api.cqc.org.uk/public/v1/locations?page=1&perPage=1")
        response.raise_for_status()
        logger.info("✓ API connection successful")
        return True
    except Exception as e:
        logger.error(f"✗ API connection failed: {e}")
        return False


def test_fetch_locations(fetcher: DetailedCQCFetcher):
    """Test fetching locations with ratings"""
    try:
        locations = fetcher.fetch_locations_with_ratings(limit=5)
        logger.info(f"✓ Fetched {len(locations)} locations")
        
        if locations:
            logger.info(f"  Sample location: {locations[0].get('locationName', 'Unknown')}")
        return len(locations) > 0
    except Exception as e:
        logger.error(f"✗ Failed to fetch locations: {e}")
        return False


def test_fetch_location_details(fetcher: DetailedCQCFetcher):
    """Test fetching detailed location data"""
    try:
        # First get a location ID
        locations = fetcher.fetch_locations_with_ratings(limit=1)
        if not locations:
            logger.error("✗ No locations available for testing")
            return False
        
        location_id = locations[0]['locationId']
        details = fetcher.fetch_location_details(location_id)
        
        if details:
            logger.info(f"✓ Fetched details for location: {details.get('locationName', 'Unknown')}")
            logger.info(f"  Has ratings history: {'ratingsHistory' in details}")
            logger.info(f"  Has inspection areas: {'inspectionAreas' in details}")
            logger.info(f"  Has reports: {'reports' in details}")
            return True
        else:
            logger.error("✗ Failed to fetch location details")
            return False
    except Exception as e:
        logger.error(f"✗ Error in detail fetching test: {e}")
        return False


def test_small_batch():
    """Test fetching a small batch of data"""
    project_id = os.environ.get('GCP_PROJECT', 'test-project')
    bucket_name = os.environ.get('GCS_BUCKET', 'test-bucket')
    
    logger.info("\n=== Testing DetailedCQCFetcher ===")
    logger.info(f"Project: {project_id}")
    logger.info(f"Bucket: {bucket_name}")
    
    # Test API connection first
    if not test_api_connection():
        logger.error("API connection test failed. Exiting.")
        return False
    
    # Initialize fetcher
    fetcher = DetailedCQCFetcher(project_id, bucket_name)
    
    # Run tests
    tests_passed = 0
    total_tests = 2
    
    if test_fetch_locations(fetcher):
        tests_passed += 1
    
    if test_fetch_location_details(fetcher):
        tests_passed += 1
    
    logger.info(f"\n=== Test Results: {tests_passed}/{total_tests} passed ===")
    
    if tests_passed == total_tests:
        logger.info("\nAll tests passed! You can now run the full fetcher with:")
        logger.info("  python fetch_detailed_cqc_data.py")
    
    return tests_passed == total_tests


if __name__ == "__main__":
    # Run tests
    test_small_batch()