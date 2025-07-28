"""
Apache Beam DoFn transforms for CQC data ETL pipeline.

This module contains transformation functions for processing CQC API data
including parsing, feature extraction, and data validation.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List

import apache_beam as beam
from apache_beam.metrics import Metrics


class ParseJsonFn(beam.DoFn):
    """Parse JSON strings into dictionaries."""
    
    def __init__(self):
        self.parse_errors = Metrics.counter('transforms', 'parse_errors')
        self.parse_success = Metrics.counter('transforms', 'parse_success')
    
    def process(self, element: str):
        """
        Parse JSON string into dictionary.
        
        Args:
            element: JSON string from CQC API
            
        Yields:
            Dict containing parsed JSON data
        """
        try:
            # Handle if element is already a dict (from previous transform)
            if isinstance(element, dict):
                self.parse_success.inc()
                yield element
                return
                
            # Parse JSON string
            parsed_data = json.loads(element)
            self.parse_success.inc()
            yield parsed_data
            
        except json.JSONDecodeError as e:
            self.parse_errors.inc()
            logging.error(f"Failed to parse JSON: {e}")
            # Optionally yield with error flag for downstream handling
            yield {
                'error': True,
                'error_message': str(e),
                'raw_data': element
            }
        except Exception as e:
            self.parse_errors.inc()
            logging.error(f"Unexpected error parsing JSON: {e}")


class ExtractLocationFeatures(beam.DoFn):
    """Extract and transform location-specific features from raw CQC data."""
    
    def __init__(self):
        self.processed = Metrics.counter('transforms', 'locations_processed')
        self.extraction_errors = Metrics.counter('transforms', 'extraction_errors')
    
    def process(self, element: Dict[str, Any]):
        """
        Extract location features from CQC API response.
        
        Args:
            element: Dictionary containing CQC location data
            
        Yields:
            Dict with extracted and transformed features
        """
        try:
            # Skip error records
            if element.get('error'):
                yield element
                return
            
            # Extract basic location information
            features = {
                'location_id': element.get('locationId'),
                'provider_id': element.get('providerId'),
                'location_name': element.get('locationName'),
                'brand_name': element.get('brandName'),
                
                # Address information
                'postal_code': element.get('postalCode'),
                'region': element.get('region'),
                'local_authority': element.get('localAuthority'),
                
                # Registration details
                'registration_status': element.get('registrationStatus'),
                'registration_date': element.get('registrationDate'),
                'deregistration_date': element.get('deregistrationDate'),
                
                # Type and category
                'type': element.get('type'),
                'main_service': element.get('mainService'),
                
                # Ratings
                'overall_rating': self._extract_rating(element, 'overall'),
                'safe_rating': self._extract_rating(element, 'safe'),
                'effective_rating': self._extract_rating(element, 'effective'),
                'caring_rating': self._extract_rating(element, 'caring'),
                'well_led_rating': self._extract_rating(element, 'wellLed'),
                'responsive_rating': self._extract_rating(element, 'responsive'),
                
                # Inspection information
                'last_inspection_date': self._extract_inspection_date(element),
                
                # Service users
                'service_user_bands': element.get('numberOfBeds', {}) if isinstance(element.get('numberOfBeds'), dict) else {},
                
                # Regulated activities (will be processed further)
                'regulated_activities': element.get('regulatedActivities', []),
                
                # Specialisms
                'specialisms': element.get('specialisms', []),
                
                # GAC service types
                'gac_service_types': element.get('gacServiceTypes', []),
                
                # Provider information
                'provider_name': element.get('provider', {}).get('name') if isinstance(element.get('provider'), dict) else None,
                
                # Timestamp for processing
                'processed_timestamp': datetime.now(timezone.utc).isoformat()
            }
            
            self.processed.inc()
            yield features
            
        except Exception as e:
            self.extraction_errors.inc()
            logging.error(f"Error extracting features: {e}")
            yield {
                'error': True,
                'error_message': f"Feature extraction failed: {str(e)}",
                'raw_data': element
            }
    
    def _extract_rating(self, element: Dict[str, Any], rating_type: str) -> Optional[str]:
        """Extract rating value from nested structure."""
        try:
            current_ratings = element.get('currentRatings', {})
            if isinstance(current_ratings, dict):
                rating_info = current_ratings.get(rating_type, {})
                if isinstance(rating_info, dict):
                    return rating_info.get('rating')
        except Exception:
            pass
        return None
    
    def _extract_inspection_date(self, element: Dict[str, Any]) -> Optional[str]:
        """Extract last inspection date."""
        try:
            last_inspection = element.get('lastInspection', {})
            if isinstance(last_inspection, dict):
                return last_inspection.get('date')
        except Exception:
            pass
        return None


class CalculateDerivedFeatures(beam.DoFn):
    """Calculate derived features from extracted base features."""
    
    def __init__(self):
        self.calculations_performed = Metrics.counter('transforms', 'calculations_performed')
        self.calculation_errors = Metrics.counter('transforms', 'calculation_errors')
    
    def process(self, element: Dict[str, Any]):
        """
        Calculate derived features like days since inspection, activity counts, etc.
        
        Args:
            element: Dictionary with extracted features
            
        Yields:
            Dict with original and derived features
        """
        try:
            # Skip error records
            if element.get('error'):
                yield element
                return
            
            # Copy original features
            features = element.copy()
            
            # Calculate days since last inspection
            features['days_since_last_inspection'] = self._calculate_days_since(
                features.get('last_inspection_date')
            )
            
            # Calculate days since registration
            features['days_since_registration'] = self._calculate_days_since(
                features.get('registration_date')
            )
            
            # Count regulated activities
            regulated_activities = features.get('regulated_activities', [])
            features['num_regulated_activities'] = (
                len(regulated_activities) if isinstance(regulated_activities, list) else 0
            )
            
            # Extract regulated activity names
            features['regulated_activity_names'] = self._extract_activity_names(regulated_activities)
            
            # Count specialisms
            specialisms = features.get('specialisms', [])
            features['num_specialisms'] = len(specialisms) if isinstance(specialisms, list) else 0
            
            # Extract specialism names
            features['specialism_names'] = self._extract_specialism_names(specialisms)
            
            # Count GAC service types
            gac_types = features.get('gac_service_types', [])
            features['num_gac_service_types'] = len(gac_types) if isinstance(gac_types, list) else 0
            
            # Extract GAC service type names
            features['gac_service_type_names'] = self._extract_gac_names(gac_types)
            
            # Calculate rating scores (convert to numeric)
            features['overall_rating_score'] = self._rating_to_score(features.get('overall_rating'))
            features['average_rating_score'] = self._calculate_average_rating_score(features)
            
            # Flag if any rating is inadequate
            features['has_inadequate_rating'] = self._has_inadequate_rating(features)
            
            # Calculate total bed capacity if available
            features['total_bed_capacity'] = self._calculate_total_beds(features.get('service_user_bands', {}))
            
            # Is currently active
            features['is_active'] = (
                features.get('registration_status') == 'Registered' and
                features.get('deregistration_date') is None
            )
            
            self.calculations_performed.inc()
            yield features
            
        except Exception as e:
            self.calculation_errors.inc()
            logging.error(f"Error calculating derived features: {e}")
            yield {
                'error': True,
                'error_message': f"Feature calculation failed: {str(e)}",
                'raw_data': element
            }
    
    def _calculate_days_since(self, date_str: Optional[str]) -> Optional[int]:
        """Calculate days between given date and today."""
        if not date_str:
            return None
        
        try:
            # Parse date (assuming ISO format YYYY-MM-DD)
            date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            today = datetime.now(timezone.utc)
            delta = today - date_obj
            return delta.days
        except Exception:
            return None
    
    def _extract_activity_names(self, activities: List[Dict[str, Any]]) -> List[str]:
        """Extract activity names from regulated activities list."""
        if not isinstance(activities, list):
            return []
        
        names = []
        for activity in activities:
            if isinstance(activity, dict) and 'name' in activity:
                names.append(activity['name'])
        return names
    
    def _extract_specialism_names(self, specialisms: List[Dict[str, Any]]) -> List[str]:
        """Extract specialism names."""
        if not isinstance(specialisms, list):
            return []
        
        names = []
        for spec in specialisms:
            if isinstance(spec, dict) and 'name' in spec:
                names.append(spec['name'])
        return names
    
    def _extract_gac_names(self, gac_types: List[Dict[str, Any]]) -> List[str]:
        """Extract GAC service type names."""
        if not isinstance(gac_types, list):
            return []
        
        names = []
        for gac in gac_types:
            if isinstance(gac, dict) and 'name' in gac:
                names.append(gac['name'])
        return names
    
    def _rating_to_score(self, rating: Optional[str]) -> Optional[int]:
        """Convert rating to numeric score."""
        rating_map = {
            'Outstanding': 4,
            'Good': 3,
            'Requires improvement': 2,
            'Inadequate': 1
        }
        return rating_map.get(rating) if rating else None
    
    def _calculate_average_rating_score(self, features: Dict[str, Any]) -> Optional[float]:
        """Calculate average score across all ratings."""
        rating_fields = ['overall_rating', 'safe_rating', 'effective_rating', 
                        'caring_rating', 'well_led_rating', 'responsive_rating']
        
        scores = []
        for field in rating_fields:
            score = self._rating_to_score(features.get(field))
            if score is not None:
                scores.append(score)
        
        return sum(scores) / len(scores) if scores else None
    
    def _has_inadequate_rating(self, features: Dict[str, Any]) -> bool:
        """Check if any rating is inadequate."""
        rating_fields = ['overall_rating', 'safe_rating', 'effective_rating', 
                        'caring_rating', 'well_led_rating', 'responsive_rating']
        
        for field in rating_fields:
            if features.get(field) == 'Inadequate':
                return True
        return False
    
    def _calculate_total_beds(self, bed_bands: Dict[str, Any]) -> Optional[int]:
        """Calculate total bed capacity from bands."""
        if not isinstance(bed_bands, dict):
            return None
        
        total = 0
        for key, value in bed_bands.items():
            if isinstance(value, (int, float)):
                total += int(value)
        
        return total if total > 0 else None


class FilterInvalidRecords(beam.DoFn):
    """Filter out records with missing critical fields."""
    
    def __init__(self, required_fields: Optional[List[str]] = None):
        """
        Initialize filter with required fields.
        
        Args:
            required_fields: List of field names that must be present and non-null
        """
        self.required_fields = required_fields or [
            'location_id',
            'provider_id',
            'overall_rating',
            'registration_status'
        ]
        self.valid_records = Metrics.counter('transforms', 'valid_records')
        self.invalid_records = Metrics.counter('transforms', 'invalid_records')
    
    def process(self, element: Dict[str, Any]):
        """
        Filter records based on required fields.
        
        Args:
            element: Dictionary with features
            
        Yields:
            Dict if all required fields are present, otherwise nothing
        """
        # Always filter out error records
        if element.get('error'):
            self.invalid_records.inc()
            logging.warning(f"Filtering error record: {element.get('error_message')}")
            return
        
        # Check required fields
        missing_fields = []
        for field in self.required_fields:
            value = element.get(field)
            if value is None or value == '':
                missing_fields.append(field)
        
        if missing_fields:
            self.invalid_records.inc()
            logging.warning(
                f"Record {element.get('location_id', 'unknown')} missing required fields: {missing_fields}"
            )
            # Optionally yield to a side output for analysis
            yield beam.pvalue.TaggedOutput('invalid_records', {
                'reason': 'missing_required_fields',
                'missing_fields': missing_fields,
                'location_id': element.get('location_id'),
                'provider_id': element.get('provider_id')
            })
        else:
            self.valid_records.inc()
            yield element