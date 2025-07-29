#!/usr/bin/env python3
"""
Fetch full CQC data with ratings for ML training.
"""

import os
import json
import time
import logging
from datetime import datetime
import requests
from google.cloud import storage
from google.cloud import bigquery
from google.cloud import secretmanager
import concurrent.futures
from typing import List, Dict, Optional

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CQCDataFetcher:
    def __init__(self):
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        self.headers = {"Ocp-Apim-Subscription-Key": self.api_key}
        self.storage_client = storage.Client(project="machine-learning-exp-467008")
        self.bigquery_client = bigquery.Client(project="machine-learning-exp-467008")
        self.bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-raw-data")
        
    def _get_api_key(self):
        """Get API key from Secret Manager or environment."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = client.secret_version_path(
                "machine-learning-exp-467008",
                "cqc-subscription-key",
                "latest"
            )
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception:
            # Fallback to environment variable
            return os.environ.get('CQC_API_KEY', '')
    
    def fetch_locations_page(self, page: int, per_page: int = 100) -> Dict:
        """Fetch a page of locations."""
        params = {"page": page, "perPage": per_page}
        response = requests.get(
            f"{self.base_url}/locations",
            params=params,
            headers=self.headers,
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    
    def fetch_location_details(self, location_id: str) -> Optional[Dict]:
        """Fetch detailed information for a location."""
        try:
            response = requests.get(
                f"{self.base_url}/locations/{location_id}",
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Failed to fetch details for {location_id}: {e}")
            return None
    
    def fetch_all_locations_with_ratings(self, limit: int = 1000):
        """Fetch locations with ratings for ML training."""
        logger.info(f"Starting to fetch up to {limit} locations with ratings...")
        
        locations_with_ratings = []
        page = 1
        total_processed = 0
        
        while len(locations_with_ratings) < limit:
            try:
                # Fetch page of locations
                logger.info(f"Fetching page {page}...")
                data = self.fetch_locations_page(page)
                locations = data.get('locations', [])
                
                if not locations:
                    logger.info("No more locations to fetch")
                    break
                
                # Process locations in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    futures = []
                    for loc in locations:
                        if len(locations_with_ratings) >= limit:
                            break
                        future = executor.submit(self.fetch_location_details, loc['locationId'])
                        futures.append((loc['locationId'], future))
                        time.sleep(0.1)  # Rate limiting
                    
                    # Collect results
                    for location_id, future in futures:
                        try:
                            details = future.result(timeout=10)
                            if details and 'currentRatings' in details:
                                locations_with_ratings.append(details)
                                logger.info(f"✓ Location {location_id}: {details.get('name', 'Unknown')} - {details['currentRatings'].get('overall', {}).get('rating', 'No rating')}")
                        except Exception as e:
                            logger.warning(f"Failed to get details for {location_id}: {e}")
                
                total_processed += len(locations)
                logger.info(f"Progress: {len(locations_with_ratings)}/{limit} locations with ratings (processed {total_processed} total)")
                
                page += 1
                time.sleep(1)  # Rate limiting between pages
                
            except Exception as e:
                logger.error(f"Error on page {page}: {e}")
                break
        
        logger.info(f"\nFetched {len(locations_with_ratings)} locations with ratings")
        return locations_with_ratings
    
    def save_to_bigquery(self, locations: List[Dict]):
        """Save locations to BigQuery."""
        if not locations:
            logger.warning("No locations to save")
            return
            
        logger.info(f"Saving {len(locations)} locations to BigQuery...")
        
        # Prepare rows
        rows = []
        for loc in locations:
            try:
                row = {
                    'locationId': loc.get('locationId'),
                    'name': loc.get('name'),
                    'numberOfBeds': loc.get('numberOfBeds'),
                    'registrationDate': loc.get('registrationDate'),
                    'lastInspectionDate': loc.get('lastInspection', {}).get('date'),
                    'postalCode': loc.get('postalCode'),
                    'region': loc.get('region'),
                    'localAuthority': loc.get('localAuthority'),
                    'providerId': loc.get('providerId'),
                    'locationType': loc.get('type'),
                    'overallRating': loc.get('currentRatings', {}).get('overall', {}).get('rating'),
                    'safeRating': loc.get('currentRatings', {}).get('safe', {}).get('rating'),
                    'effectiveRating': loc.get('currentRatings', {}).get('effective', {}).get('rating'),
                    'caringRating': loc.get('currentRatings', {}).get('caring', {}).get('rating'),
                    'responsiveRating': loc.get('currentRatings', {}).get('responsive', {}).get('rating'),
                    'wellLedRating': loc.get('currentRatings', {}).get('wellLed', {}).get('rating'),
                    'regulatedActivitiesCount': len(loc.get('regulatedActivities', [])),
                    'specialismsCount': len(loc.get('specialisms', [])),
                    'serviceTypesCount': len(loc.get('gacServiceTypes', [])),
                    'rawData': json.dumps(loc)
                }
                rows.append(row)
            except Exception as e:
                logger.error(f"Error processing location {loc.get('locationId')}: {e}")
        
        # Insert into staging table
        table_id = "machine-learning-exp-467008.cqc_data.locations_staging"
        table = self.bigquery_client.get_table(table_id)
        
        # Batch insert
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            errors = self.bigquery_client.insert_rows_json(table, batch)
            if errors:
                logger.error(f"BigQuery insert errors: {errors}")
            else:
                logger.info(f"Inserted batch {i//batch_size + 1}/{(len(rows)-1)//batch_size + 1}")
        
        # Also insert into locations_detailed
        table_id = "machine-learning-exp-467008.cqc_data.locations_detailed"
        table = self.bigquery_client.get_table(table_id)
        
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            errors = self.bigquery_client.insert_rows_json(table, batch)
            if errors:
                logger.error(f"BigQuery insert errors (detailed): {errors}")
    
    def save_to_gcs(self, locations: List[Dict]):
        """Save locations to Cloud Storage."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"full_data/locations_with_ratings_{timestamp}.json"
        
        blob = self.bucket.blob(filename)
        blob.upload_from_string(json.dumps(locations, indent=2))
        
        logger.info(f"Saved to gs://machine-learning-exp-467008-cqc-raw-data/{filename}")
        return filename
    
    def generate_summary_report(self, locations: List[Dict]):
        """Generate summary statistics."""
        logger.info("\n" + "="*60)
        logger.info("SUMMARY REPORT")
        logger.info("="*60)
        
        # Rating distribution
        ratings = {}
        for loc in locations:
            rating = loc.get('currentRatings', {}).get('overall', {}).get('rating', 'Unknown')
            ratings[rating] = ratings.get(rating, 0) + 1
        
        logger.info("\nOverall Rating Distribution:")
        for rating, count in sorted(ratings.items()):
            percentage = (count / len(locations)) * 100
            logger.info(f"  {rating}: {count} ({percentage:.1f}%)")
        
        # Region distribution
        regions = {}
        for loc in locations:
            region = loc.get('region', 'Unknown')
            regions[region] = regions.get(region, 0) + 1
        
        logger.info("\nTop 5 Regions:")
        for region, count in sorted(regions.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.info(f"  {region}: {count}")
        
        # At-risk locations
        at_risk = sum(1 for loc in locations 
                     if loc.get('currentRatings', {}).get('overall', {}).get('rating') 
                     in ['Requires improvement', 'Inadequate'])
        logger.info(f"\nAt-risk locations: {at_risk} ({(at_risk/len(locations))*100:.1f}%)")
        logger.info("="*60)

if __name__ == "__main__":
    fetcher = CQCDataFetcher()
    
    # Fetch locations with ratings
    locations = fetcher.fetch_all_locations_with_ratings(limit=1000)
    
    if locations:
        # Save to Cloud Storage
        gcs_file = fetcher.save_to_gcs(locations)
        
        # Save to BigQuery
        fetcher.save_to_bigquery(locations)
        
        # Generate report
        fetcher.generate_summary_report(locations)
        
        logger.info("\n✅ Data fetching complete!")
        logger.info(f"Total locations with ratings: {len(locations)}")
        logger.info(f"Data saved to: gs://machine-learning-exp-467008-cqc-raw-data/{gcs_file}")
    else:
        logger.error("No locations fetched")