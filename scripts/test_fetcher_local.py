#!/usr/bin/env python3
"""
Test the CQC fetcher locally with minimal data
"""

import os
import sys

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fetch_detailed_cqc_data import DetailedCQCFetcher
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_fetcher():
    """Test the fetcher with a small number of locations"""
    # Set up test configuration
    project_id = "machine-learning-exp-467008"
    bucket_name = "machine-learning-exp-467008-cqc-raw-data"
    
    print(f"Testing CQC Fetcher...")
    print(f"Project: {project_id}")
    print(f"Bucket: {bucket_name}")
    print("-" * 50)
    
    try:
        # Initialize fetcher
        fetcher = DetailedCQCFetcher(project_id, bucket_name)
        
        # Test with just 5 locations
        print("Fetching 5 locations for testing...")
        locations = fetcher.fetch_locations_with_ratings(limit=5)
        print(f"Found {len(locations)} locations")
        
        # Test fetching details for first location
        if locations:
            first_location = locations[0]
            print(f"\nFetching details for: {first_location['locationName']}")
            details = fetcher.fetch_location_details(first_location['locationId'])
            
            if details:
                print(f"Successfully fetched details!")
                print(f"Location type: {details.get('type')}")
                print(f"Has ratings: {'currentRatings' in details}")
                if 'currentRatings' in details and 'overall' in details['currentRatings']:
                    print(f"Overall rating: {details['currentRatings']['overall'].get('rating')}")
            else:
                print("No details fetched (location might not have ratings)")
        
        print("\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error during test: {e}")
        return False
    
    return True

if __name__ == "__main__":
    # Set environment variables for testing
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.expanduser('~/.config/gcloud/application_default_credentials.json')
    
    success = test_fetcher()
    sys.exit(0 if success else 1)