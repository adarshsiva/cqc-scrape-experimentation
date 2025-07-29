#!/usr/bin/env python3
"""Test script to fetch a small sample of CQC data locally"""

import os
import json
import time
import requests
from datetime import datetime
from google.cloud import storage

class LocalCQCFetcher:
    def __init__(self):
        # For local testing, we'll use environment variable
        self.api_key = os.environ.get('CQC_API_KEY')
        if not self.api_key:
            print("Please set CQC_API_KEY environment variable")
            print("You can get an API key from: https://api.service.cqc.org.uk/public/v1/swagger/index.html")
            exit(1)
            
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        
    def test_connection(self):
        """Test API connection"""
        print("Testing CQC API connection...")
        try:
            response = requests.get(
                f"{self.base_url}/providers",
                params={"page": 1, "perPage": 1},
                headers=self.headers
            )
            if response.status_code == 200:
                print("✓ API connection successful!")
                return True
            else:
                print(f"✗ API error: {response.status_code}")
                print(response.text)
                return False
        except Exception as e:
            print(f"✗ Connection error: {e}")
            return False
            
    def fetch_sample_locations(self, limit=10):
        """Fetch a small sample of locations for testing"""
        print(f"\nFetching {limit} sample locations...")
        locations = []
        
        try:
            response = requests.get(
                f"{self.base_url}/locations",
                params={"page": 1, "perPage": limit},
                headers=self.headers
            )
            
            if response.status_code == 200:
                data = response.json()
                locations = data.get('locations', [])
                print(f"✓ Found {len(locations)} locations")
                
                # Fetch details for first 3 locations
                detailed_locations = []
                for i, loc in enumerate(locations[:3]):
                    print(f"\nFetching details for location {i+1}/3: {loc.get('name', 'Unknown')}")
                    
                    # Get detailed info
                    detail_response = requests.get(
                        f"{self.base_url}/locations/{loc['locationId']}",
                        headers=self.headers
                    )
                    
                    if detail_response.status_code == 200:
                        details = detail_response.json()
                        detailed_locations.append(details)
                        
                        # Print summary
                        if 'currentRatings' in details:
                            overall = details['currentRatings'].get('overall', {}).get('rating', 'Not rated')
                            print(f"  - Overall Rating: {overall}")
                            print(f"  - Type: {details.get('type', 'Unknown')}")
                            print(f"  - Region: {details.get('region', 'Unknown')}")
                    
                    time.sleep(0.5)  # Rate limit
                    
                return detailed_locations
                
            else:
                print(f"✗ Failed to fetch locations: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"✗ Error fetching locations: {e}")
            return []
            
    def save_to_file(self, data, filename):
        """Save data to a local JSON file"""
        os.makedirs('data', exist_ok=True)
        filepath = f"data/{filename}"
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"\n✓ Data saved to {filepath}")
        return filepath
        
    def upload_to_gcs(self, filepath, bucket_name="machine-learning-exp-467008-cqc-raw-data"):
        """Upload file to Google Cloud Storage"""
        try:
            client = storage.Client(project="machine-learning-exp-467008")
            bucket = client.bucket(bucket_name)
            
            blob_name = f"test_data/{os.path.basename(filepath)}"
            blob = bucket.blob(blob_name)
            
            blob.upload_from_filename(filepath)
            print(f"✓ Uploaded to GCS: gs://{bucket_name}/{blob_name}")
            return True
            
        except Exception as e:
            print(f"✗ GCS upload failed: {e}")
            print("  (This is normal if running without GCP credentials)")
            return False

if __name__ == "__main__":
    print("CQC Data Fetcher - Local Test")
    print("="*50)
    
    fetcher = LocalCQCFetcher()
    
    # Test connection
    if not fetcher.test_connection():
        exit(1)
        
    # Fetch sample data
    locations = fetcher.fetch_sample_locations(limit=10)
    
    if locations:
        # Save locally
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cqc_sample_{timestamp}.json"
        filepath = fetcher.save_to_file(locations, filename)
        
        # Try to upload to GCS
        fetcher.upload_to_gcs(filepath)
        
        print("\n✓ Test completed successfully!")
        print(f"  - Fetched {len(locations)} detailed locations")
        print(f"  - Data saved to {filepath}")
    else:
        print("\n✗ No data fetched")
