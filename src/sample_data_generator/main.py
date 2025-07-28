import json
import random
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
import functions_framework
from google.cloud import storage
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERVICE_TYPES = [
    "Care home service with nursing",
    "Care home service without nursing", 
    "Community based services for people with a learning disability",
    "Hospital services for people with mental health needs",
    "Acute services",
    "Community health care services"
]

SPECIALISMS = [
    "Dementia",
    "Learning disabilities",
    "Mental health conditions",
    "Physical disabilities",
    "Sensory impairments",
    "Caring for adults over 65 yrs"
]

REGULATED_ACTIVITIES = [
    "Accommodation for persons who require nursing or personal care",
    "Treatment of disease, disorder or injury",
    "Diagnostic and screening procedures",
    "Personal care",
    "Nursing care"
]

SERVICE_USER_BANDS = [
    "Learning disabilities or autistic spectrum disorder",
    "Mental health",
    "Older people",
    "Physical disability",
    "Dementia"
]

REGIONS = [
    "London", "South East", "South West", "East of England",
    "West Midlands", "East Midlands", "Yorkshire and the Humber",
    "North West", "North East"
]

OWNERSHIP_TYPES = ["Individual", "Organisation", "Partnership"]

RATINGS = ["Outstanding", "Good", "Requires improvement", "Inadequate"]


def generate_provider(provider_id: int) -> Dict[str, Any]:
    """Generate a realistic provider record."""
    ownership_type = random.choice(OWNERSHIP_TYPES)
    
    provider = {
        "providerId": f"1-{provider_id:09d}",
        "name": f"{random.choice(['Care', 'Health', 'Medical', 'Wellness'])} {random.choice(['Group', 'Services', 'Partners', 'Trust'])} {provider_id}",
        "ownershipType": ownership_type,
        "type": "Provider",
        "registrationStatus": "Registered",
        "registrationDate": (datetime.now() - timedelta(days=random.randint(365, 3650))).isoformat(),
        "locationIds": []  # Will be populated later
    }
    
    if ownership_type == "Organisation":
        provider["companyNumber"] = f"{random.randint(10000000, 99999999)}"
    
    return provider


def generate_location(location_id: int, provider_id: str) -> Dict[str, Any]:
    """Generate a realistic location record."""
    has_rating = random.random() > 0.1  # 90% have ratings
    
    location = {
        "locationId": f"1-{location_id:09d}",
        "providerId": provider_id,
        "organisationType": random.choice(["Location", "NHS Trust Location"]),
        "type": "Social Care Org",
        "name": f"Care Location {location_id}",
        "brandName": f"Brand {random.randint(1, 20)}" if random.random() > 0.7 else None,
        "registrationStatus": "Registered",
        "registrationDate": (datetime.now() - timedelta(days=random.randint(180, 2500))).isoformat(),
        "numberOfBeds": random.choice([0, random.randint(10, 120)]),
        "postalCode": f"{random.choice(['SW', 'NW', 'SE', 'N', 'E', 'W'])}1 {random.randint(1,9)}{random.choice(['AA', 'BB', 'XY', 'ZZ'])}",
        "addressLine1": f"{random.randint(1, 999)} {random.choice(['High', 'Church', 'Park', 'London'])} Street",
        "addressLine2": random.choice([None, "Suite " + str(random.randint(1, 20))]),
        "townCity": random.choice(["London", "Manchester", "Birmingham", "Leeds", "Liverpool"]),
        "county": random.choice(["Greater London", "Greater Manchester", "West Midlands"]),
        "region": random.choice(REGIONS),
        "localAuthority": f"LA-{random.randint(100, 999)}",
        "constituency": f"Constituency-{random.randint(1, 650)}",
        "regulatedActivities": random.sample(REGULATED_ACTIVITIES, k=random.randint(1, 3)),
        "gacServiceTypes": random.sample(SERVICE_TYPES, k=random.randint(1, 3)),
        "specialisms": random.sample(SPECIALISMS, k=random.randint(0, 4)),
        "serviceUserBands": random.sample(SERVICE_USER_BANDS, k=random.randint(1, 3))
    }
    
    # Add inspection and rating data
    if has_rating:
        last_inspection = datetime.now() - timedelta(days=random.randint(30, 730))
        location["lastInspectionDate"] = last_inspection.isoformat()
        
        overall_rating = random.choices(RATINGS, weights=[5, 60, 25, 10])[0]
        location["currentRatings"] = {
            "overall": {"rating": overall_rating},
            "safe": {"rating": random.choice(RATINGS)},
            "effective": {"rating": random.choice(RATINGS)},
            "caring": {"rating": random.choice(RATINGS)},
            "wellLed": {"rating": random.choice(RATINGS)},
            "responsive": {"rating": random.choice(RATINGS)}
        }
        
        # Add inspection history
        inspection_count = random.randint(1, 5)
        location["inspectionHistory"] = []
        for i in range(inspection_count):
            inspection_date = last_inspection - timedelta(days=random.randint(180, 365) * (i + 1))
            location["inspectionHistory"].append({
                "inspectionDate": inspection_date.isoformat(),
                "rating": random.choice(RATINGS)
            })
    
    return location


def generate_sample_data(num_providers: int = 100, num_locations: int = 1000) -> tuple[List[Dict], List[Dict]]:
    """Generate sample provider and location data."""
    providers = []
    locations = []
    
    # Generate providers
    for i in range(num_providers):
        provider = generate_provider(i + 1)
        providers.append(provider)
    
    # Generate locations and assign to providers
    for i in range(num_locations):
        provider_idx = random.randint(0, num_providers - 1)
        provider_id = providers[provider_idx]["providerId"]
        
        location = generate_location(i + 1, provider_id)
        locations.append(location)
        
        # Add location to provider
        providers[provider_idx]["locationIds"].append(location["locationId"])
    
    return providers, locations


def upload_to_gcs(bucket_name: str, data: List[Dict], data_type: str) -> str:
    """Upload data to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    blob_name = f"raw/{data_type}/{timestamp}_{data_type}_sample.json"
    
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"Uploaded {len(data)} {data_type} to gs://{bucket_name}/{blob_name}")
    return blob_name


@functions_framework.http
def generate_sample(request):
    """Cloud Function to generate sample CQC data."""
    try:
        # Get configuration
        bucket_name = os.environ.get('GCS_BUCKET', 'cqc-data-landing')
        
        # Parse request for custom parameters
        request_json = request.get_json() if request.method == 'POST' else {}
        num_providers = request_json.get('num_providers', 100)
        num_locations = request_json.get('num_locations', 1000)
        
        logger.info(f"Generating sample data: {num_providers} providers, {num_locations} locations")
        
        # Generate sample data
        providers, locations = generate_sample_data(num_providers, num_locations)
        
        # Upload to GCS
        provider_blob = upload_to_gcs(bucket_name, providers, 'providers')
        location_blob = upload_to_gcs(bucket_name, locations, 'locations')
        
        return {
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'results': {
                'providers': {
                    'count': len(providers),
                    'blob_name': provider_blob
                },
                'locations': {
                    'count': len(locations),
                    'blob_name': location_blob
                }
            }
        }, 200
        
    except Exception as e:
        logger.error(f"Sample data generation failed: {str(e)}")
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }, 500