"""
Feature Alignment & Transformation Service

This service transforms dashboard features to match CQC training feature space,
ensuring compatibility between dashboard operational data and CQC training features.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from google.cloud import bigquery
import json
import os

logger = logging.getLogger(__name__)


class FeatureAlignmentService:
    """Align dashboard features with CQC training features"""
    
    def __init__(self, project_id: str = None):
        """
        Initialize the Feature Alignment Service
        
        Args:
            project_id: Google Cloud Project ID for BigQuery operations
        """
        self.project_id = project_id or os.environ.get('GCP_PROJECT', 'cqc-rating-predictor')
        self.client = bigquery.Client(project=self.project_id)
        
        # Load CQC training feature statistics for normalization
        self.cqc_feature_ranges = self._load_training_feature_stats()
        
        # Regional risk lookup cache
        self._regional_risk_cache = {}
        
        logger.info("FeatureAlignmentService initialized")
    
    def transform_dashboard_to_cqc_features(self, dashboard_features: Dict[str, Any]) -> Dict[str, float]:
        """
        Transform dashboard metrics to match CQC training feature space
        
        Args:
            dashboard_features: Raw dashboard features extracted from operational data
            
        Returns:
            Dict of aligned features matching CQC training feature schema
        """
        try:
            logger.info("Starting dashboard to CQC feature transformation")
            
            aligned_features = {}
            
            # Direct mappings - features available in both systems
            aligned_features['bed_capacity'] = float(dashboard_features.get('bed_capacity', 30))
            aligned_features['occupancy_rate'] = float(dashboard_features.get('occupancy_rate', 0.85))
            aligned_features['facility_size_numeric'] = float(dashboard_features.get('facility_size_numeric', 2))
            
            # Derived mappings with normalization
            aligned_features['service_complexity_score'] = self._calculate_service_complexity(
                dashboard_features.get('avg_care_complexity', 2.0),
                dashboard_features.get('activity_variety_score', 0.7)
            )
            
            # Risk indicator transformations
            aligned_features['inspection_overdue_risk'] = self._transform_to_inspection_risk(
                dashboard_features.get('incident_frequency_risk', 0.0),
                dashboard_features.get('care_plan_overdue_risk', 0.0)
            )
            
            # Individual risk components
            aligned_features['incident_frequency_risk'] = float(
                dashboard_features.get('incident_frequency_risk', 0.0)
            )
            aligned_features['medication_risk'] = float(
                dashboard_features.get('medication_risk', 0.0)
            )
            aligned_features['safeguarding_risk'] = float(
                dashboard_features.get('safeguarding_risk', 0.0)
            )
            aligned_features['falls_risk'] = float(
                dashboard_features.get('falls_risk', 0.0)
            )
            
            # Temporal features
            aligned_features['days_since_last_incident'] = float(
                dashboard_features.get('days_since_last_incident', 365)
            )
            aligned_features['operational_stability'] = self._calculate_operational_stability(
                dashboard_features
            )
            
            # Provider-level approximations (single location assumption for dashboard)
            aligned_features['provider_location_count'] = 1.0
            aligned_features['provider_avg_rating'] = self._estimate_provider_rating(dashboard_features)
            aligned_features['provider_rating_consistency'] = 0.3  # Default stability for single location
            
            # Regional risk (lookup by postcode)
            aligned_features['regional_risk_rate'] = self._lookup_regional_risk(
                dashboard_features.get('postcode', 'SW1A 1AA')
            )
            aligned_features['regional_avg_beds'] = float(
                dashboard_features.get('regional_avg_beds', 35.0)
            )
            
            # Quality indicators derived from dashboard metrics
            aligned_features['care_quality_indicator'] = self._calculate_care_quality_indicator(
                dashboard_features
            )
            
            # Interaction features - combinations that improve prediction accuracy
            aligned_features['complexity_scale_interaction'] = (
                aligned_features['service_complexity_score'] * 
                aligned_features['provider_location_count']
            )
            
            aligned_features['inspection_regional_risk'] = (
                aligned_features['inspection_overdue_risk'] * 
                aligned_features['regional_risk_rate']
            )
            
            aligned_features['capacity_complexity_interaction'] = (
                np.log1p(aligned_features['bed_capacity']) *
                aligned_features['service_complexity_score']
            )
            
            aligned_features['risk_quality_balance'] = (
                aligned_features['care_quality_indicator'] - 
                (aligned_features['incident_frequency_risk'] + aligned_features['medication_risk']) / 2
            )
            
            # Normalize features to match CQC training distribution
            aligned_features = self._normalize_features(aligned_features)
            
            logger.info(f"Feature transformation completed. Generated {len(aligned_features)} aligned features")
            return aligned_features
            
        except Exception as e:
            logger.error(f"Error transforming dashboard features: {str(e)}")
            raise
    
    def _calculate_service_complexity(self, avg_care_complexity: float, activity_variety: float) -> float:
        """
        Calculate service complexity score based on care complexity and activity variety
        
        Args:
            avg_care_complexity: Average care complexity (1-3 scale)
            activity_variety: Variety of activities offered (0-1 scale)
            
        Returns:
            Service complexity score normalized to CQC scale
        """
        try:
            # Normalize care complexity to 0-1 scale
            care_complexity_norm = (avg_care_complexity - 1.0) / 2.0
            
            # Combine care complexity and activity variety
            # Higher care complexity and more varied activities = higher complexity
            service_complexity = (
                care_complexity_norm * 0.7 +  # Care complexity weighted higher
                activity_variety * 0.3
            )
            
            # Scale to match CQC service complexity range (typically 1-8)
            scaled_complexity = 1.0 + (service_complexity * 7.0)
            
            return max(1.0, min(8.0, scaled_complexity))
            
        except Exception as e:
            logger.error(f"Error calculating service complexity: {str(e)}")
            return 4.0  # Default moderate complexity
    
    def _transform_to_inspection_risk(self, incident_risk: float, care_plan_risk: float) -> float:
        """
        Transform incident and care plan risks to inspection overdue risk equivalent
        
        Args:
            incident_risk: Incident frequency risk (0-1 scale)
            care_plan_risk: Care plan overdue risk (0-1 scale)
            
        Returns:
            Inspection overdue risk score (0-1 scale)
        """
        try:
            # Combine multiple risk factors
            # High incidents or overdue care plans suggest higher inspection risk
            combined_risk = (
                incident_risk * 0.6 +      # Incidents weighted higher
                care_plan_risk * 0.4
            )
            
            # Apply non-linear transformation to match CQC inspection risk patterns
            # Inspection risk tends to be binary (overdue or not), but we model it as continuous
            inspection_risk = np.tanh(combined_risk * 2.0)  # Smooth step function
            
            return max(0.0, min(1.0, inspection_risk))
            
        except Exception as e:
            logger.error(f"Error transforming to inspection risk: {str(e)}")
            return 0.3  # Default moderate risk
    
    def _estimate_provider_rating(self, dashboard_features: Dict[str, Any]) -> float:
        """
        Estimate provider rating from dashboard metrics
        
        Args:
            dashboard_features: Raw dashboard features
            
        Returns:
            Estimated provider rating (1-4 CQC scale)
        """
        try:
            # Risk factors (negative impact)
            risk_score = (
                dashboard_features.get('incident_frequency_risk', 0.0) * 0.3 +
                dashboard_features.get('medication_risk', 0.0) * 0.3 +
                dashboard_features.get('safeguarding_risk', 0.0) * 0.4
            )
            
            # Quality factors (positive impact)
            quality_score = (
                dashboard_features.get('care_plan_compliance', 0.8) * 0.4 +
                dashboard_features.get('resident_engagement', 0.7) * 0.3 +
                dashboard_features.get('staff_compliance_score', 0.9) * 0.3
            )
            
            # Staff performance factors
            staff_score = (
                dashboard_features.get('staff_training_current', 0.85) * 0.5 +
                (1.0 - dashboard_features.get('staff_incident_response', 0.5)) * 0.5  # Lower response time is better
            )
            
            # Combine scores with weights
            overall_score = (
                quality_score * 0.5 +      # Quality weighted highest
                staff_score * 0.3 +        # Staff performance important
                (1.0 - risk_score) * 0.2   # Risk factors (inverted)
            )
            
            # Convert to CQC scale (1-4) with 3.0 as default "Good"
            # Scale: 0.9+ = Outstanding (4), 0.7-0.9 = Good (3), 0.5-0.7 = Requires Improvement (2), <0.5 = Inadequate (1)
            if overall_score >= 0.9:
                estimated_rating = 3.5 + (overall_score - 0.9) * 5.0  # 3.5-4.0
            elif overall_score >= 0.7:
                estimated_rating = 2.5 + (overall_score - 0.7) * 2.5   # 2.5-3.0
            elif overall_score >= 0.5:
                estimated_rating = 1.5 + (overall_score - 0.5) * 5.0   # 1.5-2.5
            else:
                estimated_rating = 1.0 + overall_score                  # 1.0-1.5
            
            return max(1.0, min(4.0, estimated_rating))
            
        except Exception as e:
            logger.error(f"Error estimating provider rating: {str(e)}")
            return 3.0  # Default "Good" rating
    
    def _lookup_regional_risk(self, postcode: str) -> float:
        """
        Lookup regional risk rate based on postcode
        
        Args:
            postcode: UK postcode
            
        Returns:
            Regional risk rate (0-1 scale)
        """
        try:
            # Extract postcode area (first 1-2 letters)
            postcode_area = ''.join(c for c in postcode.upper()[:2] if c.isalpha())
            
            if postcode_area in self._regional_risk_cache:
                return self._regional_risk_cache[postcode_area]
            
            # Query BigQuery for regional statistics if available
            # Otherwise use default mapping
            regional_risk_map = {
                'E': 0.25,   # East of England
                'EC': 0.35,  # East Central London
                'EN': 0.28,  # Enfield
                'IG': 0.32,  # Ilford
                'N': 0.30,   # North London
                'NW': 0.33,  # North West London
                'SE': 0.38,  # South East London
                'SW': 0.22,  # South West London
                'W': 0.27,   # West London
                'WC': 0.24,  # West Central London
                'M': 0.35,   # Manchester
                'B': 0.32,   # Birmingham
                'L': 0.28,   # Liverpool
                'S': 0.30,   # Sheffield
                'LS': 0.26,  # Leeds
                'NE': 0.34,  # Newcastle
                'default': 0.30
            }
            
            risk_rate = regional_risk_map.get(postcode_area, regional_risk_map['default'])
            self._regional_risk_cache[postcode_area] = risk_rate
            
            return risk_rate
            
        except Exception as e:
            logger.error(f"Error looking up regional risk for postcode {postcode}: {str(e)}")
            return 0.30  # Default moderate regional risk
    
    def _calculate_operational_stability(self, dashboard_features: Dict[str, Any]) -> float:
        """
        Calculate operational stability score from dashboard metrics
        
        Args:
            dashboard_features: Raw dashboard features
            
        Returns:
            Operational stability score (0-1 scale, higher = more stable)
        """
        try:
            # Factors that indicate stability
            incident_stability = 1.0 - dashboard_features.get('incident_frequency_risk', 0.0)
            care_plan_stability = dashboard_features.get('care_plan_compliance', 0.8)
            staff_stability = dashboard_features.get('staff_compliance_score', 0.9)
            occupancy_stability = min(1.0, dashboard_features.get('occupancy_rate', 0.85))
            
            # Days since last incident (normalized)
            days_since_incident = dashboard_features.get('days_since_last_incident', 30)
            incident_recency_stability = min(1.0, days_since_incident / 90.0)  # 90 days = full stability
            
            # Combine stability factors
            overall_stability = (
                incident_stability * 0.3 +
                care_plan_stability * 0.25 +
                staff_stability * 0.25 +
                occupancy_stability * 0.1 +
                incident_recency_stability * 0.1
            )
            
            return max(0.0, min(1.0, overall_stability))
            
        except Exception as e:
            logger.error(f"Error calculating operational stability: {str(e)}")
            return 0.7  # Default moderate stability
    
    def _calculate_care_quality_indicator(self, dashboard_features: Dict[str, Any]) -> float:
        """
        Calculate overall care quality indicator from dashboard metrics
        
        Args:
            dashboard_features: Raw dashboard features
            
        Returns:
            Care quality indicator score (0-1 scale)
        """
        try:
            # Care plan metrics
            care_plan_score = (
                dashboard_features.get('care_plan_compliance', 0.8) * 0.6 +
                dashboard_features.get('care_goal_achievement', 0.7) * 0.4
            )
            
            # Resident engagement and wellbeing
            engagement_score = (
                dashboard_features.get('resident_engagement', 0.7) * 0.7 +
                (1.0 - dashboard_features.get('social_isolation_risk', 0.3)) * 0.3
            )
            
            # Safety and incident management (inverted - lower incidents = better quality)
            safety_score = 1.0 - (
                dashboard_features.get('incident_frequency_risk', 0.0) * 0.4 +
                dashboard_features.get('falls_risk', 0.0) * 0.3 +
                dashboard_features.get('medication_risk', 0.0) * 0.3
            )
            
            # Staff performance contribution
            staff_score = dashboard_features.get('staff_compliance_score', 0.9)
            
            # Combine all quality indicators
            overall_quality = (
                care_plan_score * 0.35 +
                engagement_score * 0.25 +
                safety_score * 0.25 +
                staff_score * 0.15
            )
            
            return max(0.0, min(1.0, overall_quality))
            
        except Exception as e:
            logger.error(f"Error calculating care quality indicator: {str(e)}")
            return 0.75  # Default good quality score
    
    def _normalize_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Normalize features to match CQC training feature distributions
        
        Args:
            features: Raw aligned features
            
        Returns:
            Normalized features matching CQC training ranges
        """
        try:
            normalized = features.copy()
            
            # Apply known normalization ranges based on CQC training data
            normalization_rules = {
                'bed_capacity': {'min': 1, 'max': 200, 'clip': True},
                'occupancy_rate': {'min': 0.0, 'max': 1.0, 'clip': True},
                'service_complexity_score': {'min': 1.0, 'max': 8.0, 'clip': True},
                'inspection_overdue_risk': {'min': 0.0, 'max': 1.0, 'clip': True},
                'incident_frequency_risk': {'min': 0.0, 'max': 1.0, 'clip': True},
                'medication_risk': {'min': 0.0, 'max': 1.0, 'clip': True},
                'safeguarding_risk': {'min': 0.0, 'max': 1.0, 'clip': True},
                'falls_risk': {'min': 0.0, 'max': 1.0, 'clip': True},
                'days_since_last_incident': {'min': 0, 'max': 1000, 'clip': True},
                'provider_avg_rating': {'min': 1.0, 'max': 4.0, 'clip': True},
                'regional_risk_rate': {'min': 0.0, 'max': 1.0, 'clip': True},
                'care_quality_indicator': {'min': 0.0, 'max': 1.0, 'clip': True},
                'operational_stability': {'min': 0.0, 'max': 1.0, 'clip': True}
            }
            
            for feature_name, value in features.items():
                if feature_name in normalization_rules:
                    rules = normalization_rules[feature_name]
                    if rules.get('clip', False):
                        normalized[feature_name] = max(rules['min'], min(rules['max'], value))
                    else:
                        # Z-score normalization if specific rules not defined
                        normalized[feature_name] = value
                        
            return normalized
            
        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return features  # Return original if normalization fails
    
    def _load_training_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Load CQC training feature statistics for normalization
        
        Returns:
            Dictionary of feature statistics (mean, std, min, max)
        """
        try:
            # Default feature statistics based on typical CQC data
            # In production, these would be loaded from BigQuery or stored model metadata
            default_stats = {
                'bed_capacity': {'mean': 35.0, 'std': 25.0, 'min': 1, 'max': 200},
                'occupancy_rate': {'mean': 0.85, 'std': 0.15, 'min': 0.0, 'max': 1.0},
                'service_complexity_score': {'mean': 4.5, 'std': 2.0, 'min': 1.0, 'max': 8.0},
                'provider_avg_rating': {'mean': 2.8, 'std': 0.8, 'min': 1.0, 'max': 4.0},
                'regional_risk_rate': {'mean': 0.3, 'std': 0.1, 'min': 0.1, 'max': 0.6}
            }
            
            logger.info("Loaded default CQC feature statistics")
            return default_stats
            
        except Exception as e:
            logger.error(f"Error loading training feature stats: {str(e)}")
            return {}
    
    def get_feature_importance_mapping(self) -> Dict[str, str]:
        """
        Get mapping of aligned features to human-readable descriptions
        
        Returns:
            Dictionary mapping feature names to descriptions
        """
        return {
            'bed_capacity': 'Care Home Size (Number of Beds)',
            'occupancy_rate': 'Bed Occupancy Rate',
            'service_complexity_score': 'Care Service Complexity',
            'inspection_overdue_risk': 'Regulatory Compliance Risk',
            'incident_frequency_risk': 'Incident Frequency Risk',
            'medication_risk': 'Medication Error Risk',
            'safeguarding_risk': 'Safeguarding Concerns Risk',
            'falls_risk': 'Falls Prevention Risk',
            'care_quality_indicator': 'Overall Care Quality Score',
            'operational_stability': 'Operational Stability',
            'provider_avg_rating': 'Provider Track Record',
            'regional_risk_rate': 'Regional Risk Context',
            'complexity_scale_interaction': 'Scale-Complexity Interaction',
            'inspection_regional_risk': 'Regional Inspection Risk',
            'capacity_complexity_interaction': 'Size-Complexity Balance',
            'risk_quality_balance': 'Risk-Quality Balance'
        }


def main():
    """Test the Feature Alignment Service"""
    service = FeatureAlignmentService()
    
    # Test with sample dashboard features
    sample_dashboard_features = {
        'bed_capacity': 45,
        'occupancy_rate': 0.88,
        'avg_care_complexity': 2.3,
        'incident_frequency_risk': 0.15,
        'medication_risk': 0.08,
        'safeguarding_risk': 0.0,
        'falls_risk': 0.12,
        'care_plan_compliance': 0.92,
        'resident_engagement': 0.78,
        'staff_compliance_score': 0.94,
        'activity_variety_score': 0.85,
        'days_since_last_incident': 45,
        'postcode': 'SW1A 1AA'
    }
    
    aligned_features = service.transform_dashboard_to_cqc_features(sample_dashboard_features)
    
    print("Dashboard to CQC Feature Alignment Test:")
    print("=" * 50)
    for feature, value in aligned_features.items():
        print(f"{feature}: {value:.3f}")


if __name__ == "__main__":
    main()