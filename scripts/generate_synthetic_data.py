#!/usr/bin/env python3
"""
Generate synthetic CQC data for testing the ML pipeline.
"""

import json
import random
import uuid
from datetime import datetime, timedelta
from google.cloud import bigquery
from google.cloud import storage
import numpy as np

# Configuration
NUM_LOCATIONS = 1000
REGIONS = ['London', 'South East', 'North West', 'Yorkshire and The Humber', 
           'East Midlands', 'West Midlands', 'South West', 'East of England', 
           'North East']
LOCATION_TYPES = ['Community based adult social care services', 'Residential social care',
                  'Community healthcare service', 'Hospital', 'Dental service',
                  'Primary Medical Services', 'Urgent care services']
RATINGS = ['Outstanding', 'Good', 'Requires improvement', 'Inadequate']
RATING_WEIGHTS = [0.05, 0.65, 0.25, 0.05]  # Realistic distribution

class SyntheticDataGenerator:
    def __init__(self):
        self.bigquery_client = bigquery.Client(project="machine-learning-exp-467008")
        self.storage_client = storage.Client(project="machine-learning-exp-467008")
        
    def generate_location(self, index):
        """Generate a synthetic location with realistic data."""
        location_id = f"LOC-{str(uuid.uuid4())[:8]}"
        provider_id = f"PROV-{random.randint(1000, 9999)}"
        
        # Generate correlated ratings (locations with good overall ratings likely have good domain ratings)
        overall_rating = np.random.choice(RATINGS, p=RATING_WEIGHTS)
        rating_index = RATINGS.index(overall_rating)
        
        # Add some variance to domain ratings
        domain_ratings = []
        for _ in range(5):
            variance = np.random.choice([-1, 0, 0, 0, 1])  # Mostly same, sometimes different
            domain_index = max(0, min(3, rating_index + variance))
            domain_ratings.append(RATINGS[domain_index])
        
        # Create features that correlate with ratings
        if overall_rating in ['Requires improvement', 'Inadequate']:
            # Poor ratings correlate with:
            days_since_inspection = random.randint(180, 730)  # Longer since last inspection
            number_of_beds = random.randint(50, 200)  # Larger facilities
            regulated_activities = random.randint(5, 10)  # More complex
        else:
            days_since_inspection = random.randint(30, 365)
            number_of_beds = random.randint(10, 100)
            regulated_activities = random.randint(2, 5)
            
        registration_date = datetime.now() - timedelta(days=random.randint(365, 3650))
        last_inspection_date = datetime.now() - timedelta(days=days_since_inspection)
        
        location = {
            'locationId': location_id,
            'name': f"Test Care Home {index}",
            'numberOfBeds': number_of_beds,
            'registrationDate': registration_date.isoformat(),
            'lastInspection': {
                'date': last_inspection_date.isoformat()
            },
            'postalCode': f"SW{random.randint(1, 20)} {random.randint(1, 9)}AA",
            'region': random.choice(REGIONS),
            'localAuthority': f"Borough {random.randint(1, 50)}",
            'providerId': provider_id,
            'type': random.choice(LOCATION_TYPES),
            'currentRatings': {
                'overall': {'rating': overall_rating},
                'safe': {'rating': domain_ratings[0]},
                'effective': {'rating': domain_ratings[1]},
                'caring': {'rating': domain_ratings[2]},
                'responsive': {'rating': domain_ratings[3]},
                'wellLed': {'rating': domain_ratings[4]}
            },
            'regulatedActivities': [f"Activity {i}" for i in range(regulated_activities)],
            'specialisms': [f"Specialism {i}" for i in range(random.randint(1, 5))],
            'gacServiceTypes': [f"Service {i}" for i in range(random.randint(1, 3))]
        }
        
        return location
    
    def generate_dataset(self):
        """Generate full synthetic dataset."""
        print(f"Generating {NUM_LOCATIONS} synthetic locations...")
        locations = []
        
        for i in range(NUM_LOCATIONS):
            if i % 100 == 0:
                print(f"Progress: {i}/{NUM_LOCATIONS}")
            locations.append(self.generate_location(i))
            
        return locations
    
    def save_to_bigquery(self, locations):
        """Save synthetic data to BigQuery."""
        print("\nSaving to BigQuery...")
        
        # Prepare rows for BigQuery
        rows = []
        for loc in locations:
            row = {
                'locationId': loc['locationId'],
                'name': loc['name'],
                'numberOfBeds': loc['numberOfBeds'],
                'registrationDate': loc['registrationDate'][:10],
                'lastInspectionDate': loc['lastInspection']['date'][:10],
                'postalCode': loc['postalCode'],
                'region': loc['region'],
                'localAuthority': loc['localAuthority'],
                'providerId': loc['providerId'],
                'locationType': loc['type'],
                'overallRating': loc['currentRatings']['overall']['rating'],
                'safeRating': loc['currentRatings']['safe']['rating'],
                'effectiveRating': loc['currentRatings']['effective']['rating'],
                'caringRating': loc['currentRatings']['caring']['rating'],
                'responsiveRating': loc['currentRatings']['responsive']['rating'],
                'wellLedRating': loc['currentRatings']['wellLed']['rating'],
                'regulatedActivitiesCount': len(loc['regulatedActivities']),
                'specialismsCount': len(loc['specialisms']),
                'serviceTypesCount': len(loc['gacServiceTypes']),
                'rawData': json.dumps(loc)
            }
            rows.append(row)
            
        # Insert into staging table
        table_id = "machine-learning-exp-467008.cqc_data.locations_staging"
        table = self.bigquery_client.get_table(table_id)
        
        errors = self.bigquery_client.insert_rows_json(table, rows)
        if errors:
            print(f"Error inserting rows: {errors}")
        else:
            print(f"✓ Inserted {len(rows)} rows to BigQuery")
            
        # Also insert into locations_detailed for the features
        table_id = "machine-learning-exp-467008.cqc_data.locations_detailed"
        table = self.bigquery_client.get_table(table_id)
        
        errors = self.bigquery_client.insert_rows_json(table, rows)
        if errors:
            print(f"Error inserting to locations_detailed: {errors}")
        else:
            print(f"✓ Inserted {len(rows)} rows to locations_detailed")
            
    def save_to_gcs(self, locations):
        """Save synthetic data to Cloud Storage."""
        print("\nSaving to Cloud Storage...")
        
        bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-raw-data")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        blob = bucket.blob(f"synthetic_data/locations_{timestamp}.json")
        blob.upload_from_string(json.dumps(locations, indent=2))
        
        print(f"✓ Saved to gs://machine-learning-exp-467008-cqc-raw-data/synthetic_data/locations_{timestamp}.json")
        
    def create_rating_distribution_report(self, locations):
        """Create a report of rating distributions."""
        print("\nRating Distribution:")
        print("="*50)
        
        overall_ratings = {}
        for loc in locations:
            rating = loc['currentRatings']['overall']['rating']
            overall_ratings[rating] = overall_ratings.get(rating, 0) + 1
            
        total = len(locations)
        for rating in RATINGS:
            count = overall_ratings.get(rating, 0)
            percentage = (count / total) * 100
            print(f"{rating}: {count} ({percentage:.1f}%)")
            
        # Count at-risk locations
        at_risk = sum(1 for loc in locations 
                     if loc['currentRatings']['overall']['rating'] in ['Requires improvement', 'Inadequate'])
        print(f"\nAt-risk locations: {at_risk} ({(at_risk/total)*100:.1f}%)")
        
if __name__ == "__main__":
    print("CQC Synthetic Data Generator")
    print("="*50)
    
    generator = SyntheticDataGenerator()
    
    # Generate data
    locations = generator.generate_dataset()
    
    # Save to BigQuery and GCS
    generator.save_to_bigquery(locations)
    generator.save_to_gcs(locations)
    
    # Print summary
    generator.create_rating_distribution_report(locations)
    
    print("\n✓ Synthetic data generation complete!")