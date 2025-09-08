#!/usr/bin/env python3
"""
CQC ML Data Extractor - Maximizes useful data points for machine learning.
Extracts all features that can improve rating prediction accuracy.
"""

import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from google.cloud import storage
from google.cloud import secretmanager
from google.cloud import bigquery
import requests
import numpy as np
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CQCMLDataExtractor:
    """Extract maximum useful data points for ML model training."""
    
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.api_key = self._get_api_key()
        self.base_url = "https://api.service.cqc.org.uk/public/v1"
        
        # Initialize clients
        self.storage_client = storage.Client(project=self.project_id)
        self.bigquery_client = bigquery.Client(project=self.project_id)
        self.raw_bucket = self.storage_client.bucket(f"{self.project_id}-cqc-raw-data")
        
        # Setup session
        self.session = self._create_session()
        
        logger.info("CQC ML Data Extractor initialized")
    
    def _get_api_key(self) -> str:
        """Get API key from Secret Manager or environment."""
        try:
            client = secretmanager.SecretManagerServiceClient()
            name = f"projects/{self.project_id}/secrets/cqc-subscription-key/versions/latest"
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except:
            return os.environ.get('CQC_API_KEY', '')
    
    def _create_session(self) -> requests.Session:
        """Create session with proper headers."""
        session = requests.Session()
        session.headers.update({
            "Ocp-Apim-Subscription-Key": self.api_key,
            "Accept": "application/json",
            "User-Agent": "CQC-ML-Pipeline/1.0"
        })
        return session
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """Make API request with error handling."""
        url = f"{self.base_url}/{endpoint}"
        try:
            response = self.session.get(url, params=params, timeout=30)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(60)
                return self._make_request(endpoint, params)
        except Exception as e:
            logger.error(f"Request failed for {endpoint}: {e}")
        return None
    
    def extract_ml_features_from_location(self, location: Dict) -> Dict:
        """
        Extract all ML-relevant features from a location.
        Focus on features that have predictive power for ratings.
        """
        location_id = location.get('locationId')
        features = {
            # === IDENTIFIERS ===
            'locationId': location_id,
            'providerId': location.get('providerId'),
            
            # === OPERATIONAL FEATURES (Strong predictors) ===
            'numberOfBeds': location.get('numberOfBeds', 0),
            'dormancy': location.get('dormancy', 'N'),  # Y/N - dormant facilities often have issues
            'registrationStatus': location.get('registrationStatus'),
            
            # === TEMPORAL FEATURES (Critical for predictions) ===
            'registrationDate': location.get('registrationDate'),
            'deregistrationDate': location.get('deregistrationDate'),
            
            # === GEOGRAPHIC FEATURES (Regional patterns) ===
            'region': location.get('region'),
            'localAuthority': location.get('localAuthority'),
            'constituency': location.get('constituency'),
            'postalCode': location.get('postalCode'),
            
            # === SERVICE TYPE FEATURES (Different standards apply) ===
            'type': location.get('type'),
            'care_home_beds': location.get('careHome', {}).get('beds', 0) if 'careHome' in location else 0,
            
            # === CURRENT RATINGS (Target variables) ===
            'overall_rating': None,
            'safe_rating': None,
            'effective_rating': None,
            'caring_rating': None,
            'responsive_rating': None,
            'wellLed_rating': None,
            'last_report_date': None,
            'report_link_id': None,
            
            # === SPECIALISMS (Risk indicators) ===
            'specialisms': [],
            'specialism_count': 0,
            
            # === SERVICE TYPES (Complexity indicators) ===
            'gac_service_types': [],
            'service_type_count': 0,
            
            # === REGULATED ACTIVITIES (Compliance scope) ===
            'regulated_activities': [],
            'regulated_activity_count': 0,
            
            # === INSPECTION CATEGORIES ===
            'inspection_categories': [],
            'primary_inspection_category': None,
            
            # === OWNERSHIP TYPE (Management quality) ===
            'ownership_type': location.get('ownershipType'),
            
            # === NOMINATION RIGHTS (Public/Private indicator) ===
            'has_nomination_rights': location.get('nominationRights', 'N'),
            
            # === RELATIONSHIPS (Network effects) ===
            'relationships': location.get('relationships', []),
            'relationship_count': len(location.get('relationships', [])),
        }
        
        # Extract current ratings if available
        if 'currentRatings' in location and location['currentRatings']:
            ratings = location['currentRatings']
            
            # Overall rating
            if 'overall' in ratings and ratings['overall']:
                features['overall_rating'] = ratings['overall'].get('rating')
                features['last_report_date'] = ratings['overall'].get('reportDate')
                features['report_link_id'] = ratings['overall'].get('reportLinkId')
                
                # Use report date to calculate inspection recency
                if features['last_report_date']:
                    try:
                        report_date = datetime.strptime(features['last_report_date'], '%Y-%m-%d')
                        features['days_since_inspection'] = (datetime.now() - report_date).days
                    except:
                        features['days_since_inspection'] = None
            
            # Domain ratings
            for domain in ['safe', 'effective', 'caring', 'responsive', 'wellLed']:
                if domain in ratings and ratings[domain]:
                    features[f'{domain}_rating'] = ratings[domain].get('rating')
        
        # Extract specialisms
        if 'specialisms' in location and location['specialisms']:
            features['specialisms'] = [s.get('name') for s in location['specialisms'] if 'name' in s]
            features['specialism_count'] = len(features['specialisms'])
            
            # Create binary features for high-risk specialisms
            high_risk_specialisms = ['Dementia', 'Mental health conditions', 'Learning disabilities']
            for spec in high_risk_specialisms:
                feature_name = f"has_{spec.lower().replace(' ', '_')}"
                features[feature_name] = spec in features['specialisms']
        
        # Extract GAC service types
        if 'gacServiceTypes' in location and location['gacServiceTypes']:
            features['gac_service_types'] = [s.get('name') for s in location['gacServiceTypes'] if 'name' in s]
            features['service_type_count'] = len(features['gac_service_types'])
            
            # Binary feature for nursing
            features['has_nursing'] = any('nursing' in s.lower() for s in features['gac_service_types'])
        
        # Extract regulated activities
        if 'regulatedActivities' in location and location['regulatedActivities']:
            features['regulated_activities'] = [a.get('name') for a in location['regulatedActivities'] if 'name' in a]
            features['regulated_activity_count'] = len(features['regulated_activities'])
        
        # Extract inspection categories
        if 'inspectionCategories' in location and location['inspectionCategories']:
            for cat in location['inspectionCategories']:
                if cat.get('primary') == 'true':
                    features['primary_inspection_category'] = cat.get('name')
                features['inspection_categories'].append(cat.get('name'))
        
        # Calculate derived features
        features['is_care_home'] = self._is_care_home(location)
        features['registration_years'] = self._calculate_years_registered(features['registrationDate'])
        features['is_active'] = features['registrationStatus'] == 'Registered' and features['dormancy'] != 'Y'
        
        # Size categories
        beds = features['numberOfBeds'] or 0
        features['size_category'] = self._categorize_size(beds)
        
        # Rating quality score (how many domains are rated)
        rating_domains = ['safe_rating', 'effective_rating', 'caring_rating', 'responsive_rating', 'wellLed_rating']
        features['rated_domains_count'] = sum(1 for d in rating_domains if features[d] is not None)
        
        # Overall quality indicator
        features['has_inadequate_rating'] = any(
            features[d] == 'Inadequate' for d in rating_domains if features[d]
        )
        features['all_good_or_better'] = all(
            features[d] in ['Good', 'Outstanding'] for d in rating_domains if features[d]
        )
        
        return features
    
    def fetch_inspection_areas_features(self, location_id: str) -> Dict:
        """
        Fetch detailed inspection area scores.
        These provide granular rating breakdowns.
        """
        data = self._make_request(f"locations/{location_id}/inspection-areas")
        features = {}
        
        if data and 'inspectionAreas' in data:
            areas = data['inspectionAreas']
            features['inspection_area_count'] = len(areas)
            
            # Extract ratings for each area
            area_ratings = []
            for area in areas:
                if 'latest' in area and area['latest']:
                    latest = area['latest']
                    area_name = area.get('inspectionAreaName', '').replace(' ', '_').lower()
                    
                    # Store individual area ratings
                    features[f'area_{area_name}_rating'] = latest.get('rating')
                    
                    # Collect for statistics
                    rating_map = {'Outstanding': 4, 'Good': 3, 'Requires improvement': 2, 'Inadequate': 1}
                    if latest.get('rating') in rating_map:
                        area_ratings.append(rating_map[latest['rating']])
            
            # Calculate area statistics
            if area_ratings:
                features['inspection_areas_mean_score'] = np.mean(area_ratings)
                features['inspection_areas_std_score'] = np.std(area_ratings)
                features['inspection_areas_min_score'] = min(area_ratings)
                features['inspection_areas_max_score'] = max(area_ratings)
        
        return features
    
    def fetch_reports_features(self, location_id: str) -> Dict:
        """
        Extract features from inspection reports.
        Reports contain evidence and detailed findings.
        """
        data = self._make_request("reports", {"locationId": location_id})
        features = {}
        
        if data and 'reports' in data:
            reports = data['reports']
            features['total_reports_count'] = len(reports)
            
            if reports:
                # Most recent report
                latest_report = reports[0]
                features['latest_report_type'] = latest_report.get('reportType')
                features['latest_report_date'] = latest_report.get('reportDate')
                
                # Report history features
                if len(reports) > 1:
                    # Calculate inspection frequency
                    dates = []
                    for r in reports[:5]:  # Last 5 reports
                        try:
                            dates.append(datetime.strptime(r['reportDate'], '%Y-%m-%d'))
                        except:
                            pass
                    
                    if len(dates) > 1:
                        intervals = [(dates[i] - dates[i+1]).days for i in range(len(dates)-1)]
                        features['avg_inspection_interval_days'] = np.mean(intervals)
                        features['inspection_interval_std'] = np.std(intervals)
        
        return features
    
    def fetch_provider_features(self, provider_id: str) -> Dict:
        """
        Extract provider-level features.
        Provider quality affects location ratings.
        """
        data = self._make_request(f"providers/{provider_id}")
        features = {}
        
        if data:
            features['provider_name'] = data.get('name')
            features['provider_type'] = data.get('type')
            features['provider_ownership'] = data.get('ownershipType')
            
            # Provider also nominated services (indicator of public funding)
            features['provider_also_nominated'] = data.get('alsoNominated', 'N')
            
            # Get all provider locations for network effects
            locations_data = self._make_request(f"providers/{provider_id}/locations")
            if locations_data and 'locations' in locations_data:
                provider_locations = locations_data['locations']
                features['provider_location_count'] = len(provider_locations)
                
                # Calculate provider-wide statistics
                provider_ratings = []
                for loc in provider_locations:
                    if 'currentRatings' in loc and loc['currentRatings']:
                        if 'overall' in loc['currentRatings']:
                            rating = loc['currentRatings']['overall'].get('rating')
                            rating_map = {'Outstanding': 4, 'Good': 3, 'Requires improvement': 2, 'Inadequate': 1}
                            if rating in rating_map:
                                provider_ratings.append(rating_map[rating])
                
                if provider_ratings:
                    features['provider_avg_rating_score'] = np.mean(provider_ratings)
                    features['provider_rating_std'] = np.std(provider_ratings)
                    features['provider_min_rating'] = min(provider_ratings)
                    features['provider_has_inadequate'] = 1 in provider_ratings
        
        return features
    
    def _is_care_home(self, location: Dict) -> bool:
        """Check if location is a care home."""
        care_home_types = ['Care home service with nursing', 'Care home service without nursing']
        
        if location.get('type') in care_home_types:
            return True
        
        if 'gacServiceTypes' in location:
            for service in location['gacServiceTypes']:
                if isinstance(service, dict) and service.get('name') in care_home_types:
                    return True
        
        return False
    
    def _calculate_years_registered(self, registration_date: str) -> Optional[float]:
        """Calculate years since registration."""
        if not registration_date:
            return None
        try:
            reg_date = datetime.strptime(registration_date, '%Y-%m-%d')
            return (datetime.now() - reg_date).days / 365.25
        except:
            return None
    
    def _categorize_size(self, beds: int) -> str:
        """Categorize home size."""
        if beds == 0:
            return 'unknown'
        elif beds <= 10:
            return 'very_small'
        elif beds <= 25:
            return 'small'
        elif beds <= 50:
            return 'medium'
        elif beds <= 100:
            return 'large'
        else:
            return 'very_large'
    
    def extract_comprehensive_features(self, location_basic: Dict) -> Dict:
        """
        Extract ALL useful features from all available endpoints.
        This maximizes the data points for ML model training.
        """
        location_id = location_basic['locationId']
        logger.info(f"Extracting comprehensive features for {location_id}")
        
        # Start with basic location features
        features = self.extract_ml_features_from_location(location_basic)
        
        # Add inspection areas features
        try:
            inspection_features = self.fetch_inspection_areas_features(location_id)
            features.update(inspection_features)
        except Exception as e:
            logger.warning(f"Could not fetch inspection areas for {location_id}: {e}")
        
        # Add reports features
        try:
            reports_features = self.fetch_reports_features(location_id)
            features.update(reports_features)
        except Exception as e:
            logger.warning(f"Could not fetch reports for {location_id}: {e}")
        
        # Add provider features
        if 'providerId' in location_basic:
            try:
                provider_features = self.fetch_provider_features(location_basic['providerId'])
                features.update(provider_features)
            except Exception as e:
                logger.warning(f"Could not fetch provider data: {e}")
        
        # Add metadata
        features['extraction_timestamp'] = datetime.now().isoformat()
        features['feature_version'] = 'v2.0_comprehensive'
        
        return features
    
    def save_to_bigquery(self, features_list: List[Dict]):
        """Save extracted features to BigQuery ML-ready table."""
        if not features_list:
            return
        
        table_id = f"{self.project_id}.cqc_dataset.ml_features_comprehensive"
        
        # Prepare for BigQuery
        for features in features_list:
            # Convert lists to JSON strings for BigQuery
            for key, value in features.items():
                if isinstance(value, list):
                    features[key] = json.dumps(value) if value else None
                elif isinstance(value, bool):
                    features[key] = 1 if value else 0
        
        # Load to BigQuery
        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
            autodetect=True,
            write_disposition=bigquery.WriteDisposition.WRITE_APPEND,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION,
                bigquery.SchemaUpdateOption.ALLOW_FIELD_RELAXATION
            ]
        )
        
        try:
            job = self.bigquery_client.load_table_from_json(
                features_list, table_id, job_config=job_config
            )
            job.result()
            logger.info(f"✅ Loaded {len(features_list)} feature sets to {table_id}")
        except Exception as e:
            logger.error(f"Failed to load to BigQuery: {e}")
    
    def run_comprehensive_extraction(self):
        """
        Run the comprehensive feature extraction pipeline.
        Maximizes useful data points for ML model.
        """
        logger.info("="*70)
        logger.info("COMPREHENSIVE ML FEATURE EXTRACTION PIPELINE")
        logger.info("Extracting maximum useful data points for rating prediction")
        logger.info("="*70)
        
        # Fetch all care homes
        logger.info("Step 1: Fetching all care home locations...")
        all_locations = []
        page = 1
        per_page = 1000
        
        while True:
            data = self._make_request("locations", {"page": page, "perPage": per_page})
            if not data or 'locations' not in data:
                break
            
            locations = data['locations']
            # Filter for care homes only
            care_homes = [loc for loc in locations if self._is_care_home(loc)]
            all_locations.extend(care_homes)
            
            logger.info(f"Page {page}: Found {len(care_homes)} care homes")
            
            if len(locations) < per_page:
                break
            
            page += 1
            time.sleep(0.3)
        
        logger.info(f"✅ Found {len(all_locations)} total care homes")
        
        # Process in batches with comprehensive feature extraction
        logger.info("Step 2: Extracting comprehensive features...")
        batch_size = 50
        
        for i in range(0, len(all_locations), batch_size):
            batch = all_locations[i:i+batch_size]
            logger.info(f"Processing batch {i//batch_size + 1} ({i+1} to {min(i+batch_size, len(all_locations))})")
            
            features_batch = []
            for location in batch:
                try:
                    features = self.extract_comprehensive_features(location)
                    features_batch.append(features)
                    time.sleep(0.5)  # Rate limiting between locations
                except Exception as e:
                    logger.error(f"Failed to extract features for {location.get('locationId')}: {e}")
            
            # Save batch to BigQuery
            if features_batch:
                self.save_to_bigquery(features_batch)
                logger.info(f"Saved {len(features_batch)} feature sets to BigQuery")
        
        logger.info("="*70)
        logger.info("FEATURE EXTRACTION COMPLETE")
        logger.info(f"Processed {len(all_locations)} care homes with comprehensive features")
        logger.info("Data is ready for ML model training")
        logger.info("="*70)

def main():
    """Main function for Cloud Run job."""
    extractor = CQCMLDataExtractor()
    extractor.run_comprehensive_extraction()

if __name__ == "__main__":
    main()