"""
Comprehensive Apache Beam/Dataflow ETL Pipeline for CQC Data Processing.

This enhanced pipeline provides:
- Robust data validation and cleaning
- Care home specific feature extraction
- Advanced derived metrics calculation
- Both batch and streaming processing support
- Comprehensive error handling and logging
- Multiple BigQuery table outputs (locations_complete, care_homes)
- Dead letter queue for failed records
- Data quality monitoring
"""

import argparse
import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple

import apache_beam as beam
from apache_beam.options.pipeline_options import (
    PipelineOptions, 
    GoogleCloudOptions, 
    StandardOptions,
    WorkerOptions,
    SetupOptions
)
from apache_beam.io import ReadFromText, WriteToBigQuery
from apache_beam.io.gcp.bigquery import BigQueryDisposition
from apache_beam.io.gcp.pubsub import ReadFromPubSub
from apache_beam.metrics import Metrics
from apache_beam.transforms.window import FixedWindows
from apache_beam.transforms.trigger import AfterProcessingTime, AccumulationMode

# Import existing transforms with fallback
try:
    from transforms import ParseJsonFn, ExtractLocationFeatures, CalculateDerivedFeatures, FilterInvalidRecords
except ImportError:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from transforms import ParseJsonFn, ExtractLocationFeatures, CalculateDerivedFeatures, FilterInvalidRecords

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataQualityValidation(beam.DoFn):
    """Advanced data quality validation with configurable rules."""
    
    def __init__(self, validation_config: Optional[Dict] = None):
        self.validation_config = validation_config or self._get_default_config()
        self.validation_passed = Metrics.counter('quality', 'validation_passed')
        self.validation_failed = Metrics.counter('quality', 'validation_failed')
        self.data_issues = Metrics.counter('quality', 'data_issues')
        
    def _get_default_config(self) -> Dict:
        """Default validation configuration."""
        return {
            'required_fields': ['location_id', 'provider_id'],
            'valid_rating_values': ['Outstanding', 'Good', 'Requires improvement', 'Inadequate'],
            'valid_registration_statuses': ['Registered', 'Deregistered', 'Application'],
            'postal_code_pattern': r'^[A-Z]{1,2}[0-9][A-Z0-9]?\s?[0-9][ABD-HJLNP-UW-Z]{2}$',
            'max_days_since_inspection': 3650,  # 10 years
            'min_registration_year': 1990
        }
    
    def process(self, element: Dict[str, Any]):
        """Validate data quality and flag issues."""
        try:
            if element.get('error'):
                yield element
                return
                
            issues = []
            
            # Check required fields
            for field in self.validation_config['required_fields']:
                if not element.get(field):
                    issues.append(f'missing_{field}')
            
            # Validate rating values
            for rating_field in ['overall_rating', 'safe_rating', 'effective_rating', 
                               'caring_rating', 'well_led_rating', 'responsive_rating']:
                rating = element.get(rating_field)
                if rating and rating not in self.validation_config['valid_rating_values']:
                    issues.append(f'invalid_{rating_field}')
            
            # Validate registration status
            reg_status = element.get('registration_status')
            if reg_status and reg_status not in self.validation_config['valid_registration_statuses']:
                issues.append('invalid_registration_status')
            
            # Validate postal code format
            postal_code = element.get('postal_code')
            if postal_code and not re.match(self.validation_config['postal_code_pattern'], postal_code.upper()):
                issues.append('invalid_postal_code')
            
            # Check inspection date reasonableness
            days_since_inspection = element.get('days_since_last_inspection')
            if (days_since_inspection and 
                days_since_inspection > self.validation_config['max_days_since_inspection']):
                issues.append('inspection_too_old')
            
            # Check registration date reasonableness
            reg_date = element.get('registration_date')
            if reg_date:
                try:
                    reg_year = datetime.fromisoformat(reg_date.replace('Z', '+00:00')).year
                    if reg_year < self.validation_config['min_registration_year']:
                        issues.append('registration_too_old')
                except Exception:
                    issues.append('invalid_registration_date_format')
            
            # Add validation results to element
            element['data_quality_issues'] = issues
            element['data_quality_score'] = max(0, 100 - len(issues) * 10)  # 10 points per issue
            element['validation_timestamp'] = datetime.now(timezone.utc).isoformat()
            
            if issues:
                self.data_issues.inc(len(issues))
                logger.warning(f"Data quality issues for {element.get('location_id')}: {issues}")
            
            self.validation_passed.inc()
            yield element
            
        except Exception as e:
            self.validation_failed.inc()
            logger.error(f"Validation failed: {e}")
            yield {
                **element,
                'error': True,
                'error_message': f"Validation failed: {str(e)}",
                'error_stage': 'data_quality_validation'
            }


class CareHomeFeatureExtractor(beam.DoFn):
    """Extract care home specific features and classifications."""
    
    def __init__(self):
        self.care_homes_processed = Metrics.counter('features', 'care_homes_processed')
        self.non_care_homes = Metrics.counter('features', 'non_care_homes')
        self.extraction_errors = Metrics.counter('features', 'extraction_errors')
    
    def process(self, element: Dict[str, Any]):
        """Extract care home specific features."""
        try:
            if element.get('error'):
                yield element
                return
            
            # Determine if this is a care home
            is_care_home = self._is_care_home(element)
            
            features = element.copy()
            features['is_care_home'] = is_care_home
            
            if is_care_home:
                self.care_homes_processed.inc()
                
                # Care home specific features
                features.update({
                    'care_home_type': self._classify_care_home_type(element),
                    'bed_capacity_category': self._categorize_bed_capacity(element),
                    'specializes_in_dementia': self._has_dementia_specialism(element),
                    'specializes_in_nursing': self._has_nursing_specialism(element),
                    'specializes_in_residential': self._has_residential_specialism(element),
                    'provides_end_of_life_care': self._provides_end_of_life_care(element),
                    'accepts_mental_health': self._accepts_mental_health(element),
                    'accepts_learning_disabilities': self._accepts_learning_disabilities(element),
                    'risk_score': self._calculate_risk_score(element),
                    'compliance_trend': self._assess_compliance_trend(element),
                    'inspection_frequency_category': self._categorize_inspection_frequency(element)
                })
            else:
                self.non_care_homes.inc()
                # For non-care homes, set care home specific fields to null
                features.update({
                    'care_home_type': None,
                    'bed_capacity_category': None,
                    'specializes_in_dementia': False,
                    'specializes_in_nursing': False,
                    'specializes_in_residential': False,
                    'provides_end_of_life_care': False,
                    'accepts_mental_health': False,
                    'accepts_learning_disabilities': False,
                    'risk_score': None,
                    'compliance_trend': None,
                    'inspection_frequency_category': None
                })
            
            yield features
            
        except Exception as e:
            self.extraction_errors.inc()
            logger.error(f"Care home feature extraction failed: {e}")
            yield {
                **element,
                'error': True,
                'error_message': f"Care home feature extraction failed: {str(e)}",
                'error_stage': 'care_home_features'
            }
    
    def _is_care_home(self, element: Dict[str, Any]) -> bool:
        """Determine if location is a care home."""
        care_home_indicators = [
            'care home', 'residential care', 'nursing home', 'care centre',
            'residential home', 'nursing care', 'elderly care'
        ]
        
        # Check main service
        main_service = (element.get('main_service') or '').lower()
        for indicator in care_home_indicators:
            if indicator in main_service:
                return True
        
        # Check location name
        location_name = (element.get('location_name') or '').lower()
        for indicator in care_home_indicators:
            if indicator in location_name:
                return True
        
        # Check regulated activities
        activities = element.get('regulated_activity_names', [])
        care_activities = ['accommodation for persons who require nursing or personal care']
        for activity in activities:
            if any(care_act.lower() in activity.lower() for care_act in care_activities):
                return True
        
        return False
    
    def _classify_care_home_type(self, element: Dict[str, Any]) -> Optional[str]:
        """Classify type of care home."""
        activities = element.get('regulated_activity_names', [])
        specialisms = element.get('specialism_names', [])
        
        # Check for nursing home
        nursing_indicators = ['nursing', 'registered nurse', 'clinical']
        if any(indicator in ' '.join(activities + specialisms).lower() 
               for indicator in nursing_indicators):
            return 'nursing_home'
        
        # Check for residential home
        residential_indicators = ['residential', 'personal care']
        if any(indicator in ' '.join(activities + specialisms).lower() 
               for indicator in residential_indicators):
            return 'residential_home'
        
        return 'mixed_care_home'
    
    def _categorize_bed_capacity(self, element: Dict[str, Any]) -> Optional[str]:
        """Categorize care home by bed capacity."""
        total_beds = element.get('total_bed_capacity')
        if not total_beds:
            return 'unknown'
        
        if total_beds <= 10:
            return 'small'
        elif total_beds <= 30:
            return 'medium'
        elif total_beds <= 60:
            return 'large'
        else:
            return 'extra_large'
    
    def _has_dementia_specialism(self, element: Dict[str, Any]) -> bool:
        """Check if specializes in dementia care."""
        specialisms = element.get('specialism_names', [])
        dementia_terms = ['dementia', 'alzheimer', 'memory']
        return any(term in ' '.join(specialisms).lower() for term in dementia_terms)
    
    def _has_nursing_specialism(self, element: Dict[str, Any]) -> bool:
        """Check if provides nursing care."""
        activities = element.get('regulated_activity_names', [])
        nursing_terms = ['nursing', 'clinical', 'medical']
        return any(term in ' '.join(activities).lower() for term in nursing_terms)
    
    def _has_residential_specialism(self, element: Dict[str, Any]) -> bool:
        """Check if provides residential care."""
        activities = element.get('regulated_activity_names', [])
        residential_terms = ['residential', 'personal care', 'accommodation']
        return any(term in ' '.join(activities).lower() for term in residential_terms)
    
    def _provides_end_of_life_care(self, element: Dict[str, Any]) -> bool:
        """Check if provides end of life care."""
        specialisms = element.get('specialism_names', [])
        activities = element.get('regulated_activity_names', [])
        eol_terms = ['end of life', 'palliative', 'hospice', 'terminal']
        all_text = ' '.join(specialisms + activities).lower()
        return any(term in all_text for term in eol_terms)
    
    def _accepts_mental_health(self, element: Dict[str, Any]) -> bool:
        """Check if accepts mental health service users."""
        specialisms = element.get('specialism_names', [])
        mental_health_terms = ['mental health', 'psychiatric', 'psychological']
        return any(term in ' '.join(specialisms).lower() for term in mental_health_terms)
    
    def _accepts_learning_disabilities(self, element: Dict[str, Any]) -> bool:
        """Check if accepts learning disabilities service users."""
        specialisms = element.get('specialism_names', [])
        ld_terms = ['learning disabil', 'intellectual disabil', 'developmental']
        return any(term in ' '.join(specialisms).lower() for term in ld_terms)
    
    def _calculate_risk_score(self, element: Dict[str, Any]) -> Optional[float]:
        """Calculate risk score based on various factors."""
        try:
            risk_factors = 0
            total_factors = 0
            
            # Rating-based risk
            overall_rating = element.get('overall_rating')
            if overall_rating:
                total_factors += 1
                if overall_rating == 'Inadequate':
                    risk_factors += 1
                elif overall_rating == 'Requires improvement':
                    risk_factors += 0.5
            
            # Has any inadequate rating
            if element.get('has_inadequate_rating'):
                risk_factors += 1
            total_factors += 1
            
            # Days since inspection (higher = more risk)
            days_since_inspection = element.get('days_since_last_inspection')
            if days_since_inspection:
                total_factors += 1
                if days_since_inspection > 730:  # 2 years
                    risk_factors += 1
                elif days_since_inspection > 365:  # 1 year
                    risk_factors += 0.5
            
            # Data quality issues
            data_issues = len(element.get('data_quality_issues', []))
            if data_issues > 0:
                risk_factors += min(data_issues / 10, 1)  # Cap at 1
            total_factors += 1
            
            return (risk_factors / total_factors) * 100 if total_factors > 0 else None
            
        except Exception:
            return None
    
    def _assess_compliance_trend(self, element: Dict[str, Any]) -> Optional[str]:
        """Assess compliance trend based on available data."""
        overall_rating = element.get('overall_rating')
        has_inadequate = element.get('has_inadequate_rating', False)
        
        if has_inadequate:
            return 'declining'
        elif overall_rating == 'Outstanding':
            return 'excellent'
        elif overall_rating == 'Good':
            return 'stable'
        elif overall_rating == 'Requires improvement':
            return 'concerning'
        else:
            return 'unknown'
    
    def _categorize_inspection_frequency(self, element: Dict[str, Any]) -> Optional[str]:
        """Categorize expected inspection frequency based on risk."""
        days_since_inspection = element.get('days_since_last_inspection')
        overall_rating = element.get('overall_rating')
        
        if not days_since_inspection or not overall_rating:
            return 'unknown'
        
        # CQC inspection frequency guidelines
        if overall_rating in ['Inadequate', 'Requires improvement']:
            expected_frequency = 365  # Annual
        else:
            expected_frequency = 730  # Biennial
        
        if days_since_inspection > expected_frequency * 1.5:
            return 'overdue'
        elif days_since_inspection > expected_frequency:
            return 'due'
        else:
            return 'current'


class EnhancedDerivedMetrics(beam.DoFn):
    """Calculate advanced derived metrics for ML features."""
    
    def __init__(self):
        self.metrics_calculated = Metrics.counter('metrics', 'derived_calculated')
        self.calculation_errors = Metrics.counter('metrics', 'calculation_errors')
    
    def process(self, element: Dict[str, Any]):
        """Calculate advanced derived metrics."""
        try:
            if element.get('error'):
                yield element
                return
            
            features = element.copy()
            
            # Regional statistics (would typically come from a side input)
            features['regional_care_home_density'] = self._estimate_regional_density(element)
            
            # Temporal features
            features['registration_year'] = self._extract_year(element.get('registration_date'))
            features['last_inspection_year'] = self._extract_year(element.get('last_inspection_date'))
            features['inspection_to_registration_ratio'] = self._calculate_inspection_ratio(element)
            
            # Service complexity
            features['service_complexity_score'] = self._calculate_service_complexity(element)
            
            # Geographic risk factors
            features['urban_rural_indicator'] = self._classify_urban_rural(element)
            
            # Capacity utilization indicators
            features['theoretical_staff_ratio'] = self._estimate_staff_ratio(element)
            
            # Rating stability indicators
            features['rating_consistency_score'] = self._calculate_rating_consistency(element)
            
            self.metrics_calculated.inc()
            yield features
            
        except Exception as e:
            self.calculation_errors.inc()
            logger.error(f"Derived metrics calculation failed: {e}")
            yield {
                **element,
                'error': True,
                'error_message': f"Derived metrics calculation failed: {str(e)}",
                'error_stage': 'derived_metrics'
            }
    
    def _estimate_regional_density(self, element: Dict[str, Any]) -> Optional[str]:
        """Estimate care home density in region (simplified)."""
        region = element.get('region')
        if not region:
            return None
        
        # Simplified density estimation based on known high-density regions
        high_density_regions = ['London', 'South East', 'North West']
        if region in high_density_regions:
            return 'high'
        else:
            return 'medium'  # Would use actual data in production
    
    def _extract_year(self, date_str: Optional[str]) -> Optional[int]:
        """Extract year from date string."""
        if not date_str:
            return None
        try:
            return datetime.fromisoformat(date_str.replace('Z', '+00:00')).year
        except Exception:
            return None
    
    def _calculate_inspection_ratio(self, element: Dict[str, Any]) -> Optional[float]:
        """Calculate ratio of time since inspection to time since registration."""
        days_since_inspection = element.get('days_since_last_inspection')
        days_since_registration = element.get('days_since_registration')
        
        if not days_since_inspection or not days_since_registration or days_since_registration <= 0:
            return None
        
        return min(days_since_inspection / days_since_registration, 2.0)  # Cap at 2.0
    
    def _calculate_service_complexity(self, element: Dict[str, Any]) -> Optional[float]:
        """Calculate service complexity score."""
        try:
            complexity = 0
            
            # Base complexity from number of activities and specialisms
            num_activities = element.get('num_regulated_activities', 0)
            num_specialisms = element.get('num_specialisms', 0)
            complexity += (num_activities * 0.5) + (num_specialisms * 0.3)
            
            # Bonus for complex specialisms
            if element.get('specializes_in_dementia'):
                complexity += 1
            if element.get('accepts_mental_health'):
                complexity += 1
            if element.get('provides_end_of_life_care'):
                complexity += 1.5
            
            # Capacity factor
            bed_capacity = element.get('total_bed_capacity', 0)
            if bed_capacity > 50:
                complexity += 1
            elif bed_capacity > 100:
                complexity += 2
            
            return min(complexity, 10.0)  # Cap at 10
            
        except Exception:
            return None
    
    def _classify_urban_rural(self, element: Dict[str, Any]) -> Optional[str]:
        """Classify location as urban or rural (simplified)."""
        # This would typically use postcodes and ONS data
        postal_code = element.get('postal_code', '')
        if not postal_code:
            return None
        
        # Simplified classification based on postal code patterns
        # London postcodes
        if postal_code.startswith(('E', 'N', 'S', 'W')):
            return 'urban'
        # Major city indicators
        elif postal_code.startswith(('M', 'B', 'L')):
            return 'urban'
        else:
            return 'rural'  # Simplified assumption
    
    def _estimate_staff_ratio(self, element: Dict[str, Any]) -> Optional[float]:
        """Estimate theoretical staff-to-bed ratio."""
        bed_capacity = element.get('total_bed_capacity')
        if not bed_capacity or bed_capacity <= 0:
            return None
        
        # Industry standard ratios vary by care type
        if element.get('care_home_type') == 'nursing_home':
            return 0.8  # Higher staff ratio for nursing homes
        elif element.get('specializes_in_dementia'):
            return 0.7  # Higher ratio for dementia care
        else:
            return 0.5  # Standard residential care ratio
    
    def _calculate_rating_consistency(self, element: Dict[str, Any]) -> Optional[float]:
        """Calculate consistency across different rating dimensions."""
        rating_fields = ['safe_rating', 'effective_rating', 'caring_rating', 
                        'well_led_rating', 'responsive_rating']
        
        rating_scores = []
        for field in rating_fields:
            rating = element.get(field)
            if rating:
                score = {'Outstanding': 4, 'Good': 3, 'Requires improvement': 2, 'Inadequate': 1}.get(rating)
                if score:
                    rating_scores.append(score)
        
        if len(rating_scores) < 2:
            return None
        
        # Calculate coefficient of variation (lower = more consistent)
        mean_score = sum(rating_scores) / len(rating_scores)
        variance = sum((x - mean_score) ** 2 for x in rating_scores) / len(rating_scores)
        std_dev = variance ** 0.5
        
        return 1 - (std_dev / mean_score) if mean_score > 0 else 0


class ComprehensiveETLPipeline:
    """Comprehensive ETL Pipeline for CQC data with enhanced features."""
    
    def __init__(self, project_id: str, dataset_id: str, temp_location: str, region: str = 'europe-west2'):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.temp_location = temp_location
        self.region = region
    
    def get_pipeline_options(self, 
                           runner: str = 'DataflowRunner',
                           job_name: str = None,
                           streaming: bool = False,
                           **kwargs) -> PipelineOptions:
        """Configure comprehensive pipeline options."""
        options = PipelineOptions()
        
        # Google Cloud options
        google_cloud_options = options.view_as(GoogleCloudOptions)
        google_cloud_options.project = self.project_id
        google_cloud_options.region = self.region
        google_cloud_options.temp_location = self.temp_location
        google_cloud_options.staging_location = f"{self.temp_location}/staging"
        
        # Standard options
        standard_options = options.view_as(StandardOptions)
        standard_options.runner = runner
        standard_options.streaming = streaming
        
        # Worker options for better performance
        worker_options = options.view_as(WorkerOptions)
        worker_options.num_workers = kwargs.get('num_workers', 2)
        worker_options.max_num_workers = kwargs.get('max_num_workers', 10)
        worker_options.machine_type = kwargs.get('machine_type', 'n1-standard-2')
        worker_options.disk_size_gb = kwargs.get('disk_size_gb', 50)
        
        # Setup options
        setup_options = options.view_as(SetupOptions)
        setup_options.setup_file = './setup.py'
        
        # Job name
        if job_name:
            google_cloud_options.job_name = job_name
        
        return options
    
    def get_locations_complete_schema(self) -> List[Dict]:
        """Enhanced schema for complete locations table."""
        return [
            # Basic identification
            {'name': 'location_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'provider_id', 'type': 'STRING', 'mode': 'REQUIRED'},
            {'name': 'location_name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'brand_name', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'provider_name', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Location and registration
            {'name': 'postal_code', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'region', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'local_authority', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'registration_status', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'registration_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            {'name': 'deregistration_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            
            # Service classification
            {'name': 'type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'main_service', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'is_active', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            
            # Current ratings
            {'name': 'overall_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'safe_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'effective_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'caring_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'well_led_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'responsive_rating', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Inspection information
            {'name': 'last_inspection_date', 'type': 'DATE', 'mode': 'NULLABLE'},
            {'name': 'days_since_last_inspection', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'days_since_registration', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            
            # Services and activities
            {'name': 'regulated_activity_names', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'specialism_names', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'gac_service_type_names', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'num_regulated_activities', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'num_specialisms', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'num_gac_service_types', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            
            # Capacity and user information
            {'name': 'total_bed_capacity', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            
            # Derived rating metrics
            {'name': 'overall_rating_score', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'average_rating_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'has_inadequate_rating', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'rating_consistency_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            
            # Care home classification
            {'name': 'is_care_home', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'care_home_type', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'bed_capacity_category', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Specialisms flags
            {'name': 'specializes_in_dementia', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'specializes_in_nursing', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'specializes_in_residential', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'provides_end_of_life_care', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'accepts_mental_health', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            {'name': 'accepts_learning_disabilities', 'type': 'BOOLEAN', 'mode': 'NULLABLE'},
            
            # Risk and compliance
            {'name': 'risk_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'compliance_trend', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'inspection_frequency_category', 'type': 'STRING', 'mode': 'NULLABLE'},
            
            # Advanced derived metrics
            {'name': 'regional_care_home_density', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'registration_year', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'last_inspection_year', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'inspection_to_registration_ratio', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'service_complexity_score', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            {'name': 'urban_rural_indicator', 'type': 'STRING', 'mode': 'NULLABLE'},
            {'name': 'theoretical_staff_ratio', 'type': 'FLOAT', 'mode': 'NULLABLE'},
            
            # Data quality
            {'name': 'data_quality_issues', 'type': 'STRING', 'mode': 'REPEATED'},
            {'name': 'data_quality_score', 'type': 'INTEGER', 'mode': 'NULLABLE'},
            {'name': 'validation_timestamp', 'type': 'TIMESTAMP', 'mode': 'NULLABLE'},
            
            # Processing metadata
            {'name': 'processed_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'processing_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
            {'name': 'pipeline_version', 'type': 'STRING', 'mode': 'NULLABLE'}
        ]
    
    def get_care_homes_schema(self) -> List[Dict]:
        """Schema specifically for care homes table."""
        return [field for field in self.get_locations_complete_schema() 
                if field['name'] not in ['gac_service_type_names', 'num_gac_service_types']]
    
    def run_batch_pipeline(self, input_pattern: str, job_name: str = None):
        """Run comprehensive batch ETL pipeline."""
        if not job_name:
            job_name = f"cqc-etl-complete-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        options = self.get_pipeline_options(job_name=job_name, streaming=False)
        
        with beam.Pipeline(options=options) as pipeline:
            # Read and process data
            raw_data = (
                pipeline
                | 'Read Raw Data' >> ReadFromText(input_pattern, coder=beam.coders.StrUtf8Coder())
            )
            
            # Main processing pipeline
            processed_data = (
                raw_data
                | 'Parse JSON' >> beam.ParDo(ParseJsonFn())
                | 'Data Quality Validation' >> beam.ParDo(DataQualityValidation())
                | 'Extract Location Features' >> beam.ParDo(ExtractLocationFeatures())
                | 'Calculate Derived Features' >> beam.ParDo(CalculateDerivedFeatures())
                | 'Extract Care Home Features' >> beam.ParDo(CareHomeFeatureExtractor())
                | 'Calculate Enhanced Metrics' >> beam.ParDo(EnhancedDerivedMetrics())
                | 'Add Pipeline Metadata' >> beam.Map(
                    lambda x: {
                        **x, 
                        'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                        'pipeline_version': 'v2.0.0'
                    }
                )
            )
            
            # Split into valid and error records
            valid_records, error_records = (
                processed_data
                | 'Split Valid/Error' >> beam.Partition(
                    lambda element, _: 0 if element.get('error') else 1, 2
                )
            )
            
            # Write valid records to locations_complete table
            valid_records | 'Write to Locations Complete' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.locations_complete",
                schema={'fields': self.get_locations_complete_schema()},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                additional_bq_parameters={
                    'timePartitioning': {'type': 'DAY', 'field': 'processing_timestamp'},
                    'clustering': {'fields': ['region', 'overall_rating', 'is_care_home']}
                }
            )
            
            # Filter and write care homes to separate table
            care_homes = (
                valid_records
                | 'Filter Care Homes' >> beam.Filter(lambda x: x.get('is_care_home', False))
            )
            
            care_homes | 'Write to Care Homes' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.care_homes",
                schema={'fields': self.get_care_homes_schema()},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                additional_bq_parameters={
                    'timePartitioning': {'type': 'DAY', 'field': 'processing_timestamp'},
                    'clustering': {'fields': ['region', 'overall_rating', 'care_home_type']}
                }
            )
            
            # Write error records to dead letter queue
            error_records | 'Write Errors to DLQ' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.processing_errors",
                schema={'fields': [
                    {'name': 'error_message', 'type': 'STRING', 'mode': 'NULLABLE'},
                    {'name': 'error_stage', 'type': 'STRING', 'mode': 'NULLABLE'},
                    {'name': 'raw_data', 'type': 'STRING', 'mode': 'NULLABLE'},
                    {'name': 'error_timestamp', 'type': 'TIMESTAMP', 'mode': 'REQUIRED'},
                    {'name': 'pipeline_version', 'type': 'STRING', 'mode': 'NULLABLE'}
                ]},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND
            )
            
            # Log processing statistics
            valid_count = (
                valid_records
                | 'Count Valid Records' >> beam.combiners.Count.Globally()
                | 'Log Valid Count' >> beam.Map(
                    lambda count: logger.info(f"Successfully processed {count} valid records")
                )
            )
            
            error_count = (
                error_records
                | 'Count Error Records' >> beam.combiners.Count.Globally()
                | 'Log Error Count' >> beam.Map(
                    lambda count: logger.info(f"Found {count} error records")
                )
            )
            
            care_home_count = (
                care_homes
                | 'Count Care Homes' >> beam.combiners.Count.Globally()
                | 'Log Care Home Count' >> beam.Map(
                    lambda count: logger.info(f"Processed {count} care home records")
                )
            )
    
    def run_streaming_pipeline(self, pubsub_subscription: str, job_name: str = None):
        """Run streaming ETL pipeline from Pub/Sub."""
        if not job_name:
            job_name = f"cqc-etl-streaming-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        options = self.get_pipeline_options(job_name=job_name, streaming=True)
        
        with beam.Pipeline(options=options) as pipeline:
            # Read from Pub/Sub
            messages = (
                pipeline
                | 'Read from Pub/Sub' >> ReadFromPubSub(subscription=pubsub_subscription)
                | 'Decode Messages' >> beam.Map(lambda x: x.decode('utf-8'))
                | 'Window Messages' >> beam.WindowInto(
                    FixedWindows(300),  # 5-minute windows
                    trigger=AfterProcessingTime(60),  # Trigger every minute
                    accumulation_mode=AccumulationMode.DISCARDING
                )
            )
            
            # Process messages through same pipeline
            processed_data = (
                messages
                | 'Parse JSON Stream' >> beam.ParDo(ParseJsonFn())
                | 'Validate Stream Data' >> beam.ParDo(DataQualityValidation())
                | 'Extract Stream Features' >> beam.ParDo(ExtractLocationFeatures())
                | 'Calculate Stream Derived' >> beam.ParDo(CalculateDerivedFeatures())
                | 'Extract Stream Care Home' >> beam.ParDo(CareHomeFeatureExtractor())
                | 'Calculate Stream Metrics' >> beam.ParDo(EnhancedDerivedMetrics())
                | 'Add Stream Metadata' >> beam.Map(
                    lambda x: {
                        **x,
                        'processing_timestamp': datetime.now(timezone.utc).isoformat(),
                        'pipeline_version': 'v2.0.0-streaming'
                    }
                )
            )
            
            # Write to BigQuery (streaming inserts)
            processed_data | 'Stream to BigQuery' >> WriteToBigQuery(
                table=f"{self.project_id}:{self.dataset_id}.locations_complete",
                schema={'fields': self.get_locations_complete_schema()},
                create_disposition=BigQueryDisposition.CREATE_IF_NEEDED,
                write_disposition=BigQueryDisposition.WRITE_APPEND,
                method='STREAMING_INSERTS'
            )


def main():
    """Main entry point for comprehensive ETL pipeline."""
    parser = argparse.ArgumentParser(description='CQC Comprehensive ETL Pipeline')
    parser.add_argument('--project-id', required=True, help='GCP Project ID')
    parser.add_argument('--dataset-id', required=True, help='BigQuery Dataset ID')
    parser.add_argument('--temp-location', required=True, help='GCS temp location')
    parser.add_argument('--region', default='europe-west2', help='GCP region')
    parser.add_argument('--runner', default='DataflowRunner', help='Pipeline runner')
    
    # Input source options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--input-pattern', help='Input file pattern for batch processing')
    input_group.add_argument('--pubsub-subscription', help='Pub/Sub subscription for streaming')
    
    parser.add_argument('--job-name', help='Custom job name')
    parser.add_argument('--streaming', action='store_true', help='Run in streaming mode')
    
    # Worker configuration
    parser.add_argument('--num-workers', type=int, default=2, help='Number of workers')
    parser.add_argument('--max-num-workers', type=int, default=10, help='Max number of workers')
    parser.add_argument('--machine-type', default='n1-standard-2', help='Worker machine type')
    parser.add_argument('--disk-size-gb', type=int, default=50, help='Worker disk size in GB')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ComprehensiveETLPipeline(
        project_id=args.project_id,
        dataset_id=args.dataset_id,
        temp_location=args.temp_location,
        region=args.region
    )
    
    # Run appropriate pipeline
    if args.streaming or args.pubsub_subscription:
        if not args.pubsub_subscription:
            raise ValueError("Pub/Sub subscription required for streaming mode")
        logger.info(f"Starting streaming pipeline from {args.pubsub_subscription}")
        pipeline.run_streaming_pipeline(
            pubsub_subscription=args.pubsub_subscription,
            job_name=args.job_name
        )
    else:
        logger.info(f"Starting batch pipeline with input pattern: {args.input_pattern}")
        pipeline.run_batch_pipeline(
            input_pattern=args.input_pattern,
            job_name=args.job_name
        )
    
    logger.info("Pipeline execution completed")


if __name__ == '__main__':
    main()