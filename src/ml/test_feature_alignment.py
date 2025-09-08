"""
Unit tests for Feature Alignment & Transformation Service

Tests the transformation of dashboard features to CQC training feature space.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch

# Add the project root to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.ml.feature_alignment import FeatureAlignmentService


class TestFeatureAlignmentService(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.service = FeatureAlignmentService(project_id='test-project')
        
        self.sample_dashboard_features = {
            'bed_capacity': 45,
            'occupancy_rate': 0.88,
            'facility_size_numeric': 3,
            'avg_care_complexity': 2.3,
            'activity_variety_score': 0.85,
            'incident_frequency_risk': 0.15,
            'medication_risk': 0.08,
            'safeguarding_risk': 0.0,
            'falls_risk': 0.12,
            'care_plan_overdue_risk': 0.05,
            'care_plan_compliance': 0.92,
            'resident_engagement': 0.78,
            'staff_compliance_score': 0.94,
            'staff_training_current': 0.88,
            'staff_incident_response': 0.3,
            'care_goal_achievement': 0.82,
            'social_isolation_risk': 0.15,
            'days_since_last_incident': 45,
            'postcode': 'SW1A 1AA',
            'regional_avg_beds': 38.0
        }
    
    def test_transform_dashboard_to_cqc_features(self):
        """Test complete dashboard to CQC feature transformation"""
        aligned_features = self.service.transform_dashboard_to_cqc_features(
            self.sample_dashboard_features
        )
        
        # Check that all expected features are present
        expected_features = [
            'bed_capacity', 'occupancy_rate', 'facility_size_numeric',
            'service_complexity_score', 'inspection_overdue_risk',
            'incident_frequency_risk', 'medication_risk', 'safeguarding_risk',
            'falls_risk', 'days_since_last_incident', 'operational_stability',
            'provider_location_count', 'provider_avg_rating', 
            'provider_rating_consistency', 'regional_risk_rate',
            'regional_avg_beds', 'care_quality_indicator',
            'complexity_scale_interaction', 'inspection_regional_risk',
            'capacity_complexity_interaction', 'risk_quality_balance'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, aligned_features)
            self.assertIsInstance(aligned_features[feature], (int, float))
        
        # Check value ranges
        self.assertGreaterEqual(aligned_features['bed_capacity'], 0)
        self.assertGreaterEqual(aligned_features['occupancy_rate'], 0.0)
        self.assertLessEqual(aligned_features['occupancy_rate'], 1.0)
        self.assertGreaterEqual(aligned_features['provider_avg_rating'], 1.0)
        self.assertLessEqual(aligned_features['provider_avg_rating'], 4.0)
    
    def test_calculate_service_complexity(self):
        """Test service complexity calculation"""
        # Test normal case
        complexity = self.service._calculate_service_complexity(2.5, 0.8)
        self.assertGreaterEqual(complexity, 1.0)
        self.assertLessEqual(complexity, 8.0)
        
        # Test edge cases
        complexity_min = self.service._calculate_service_complexity(1.0, 0.0)
        self.assertGreaterEqual(complexity_min, 1.0)
        
        complexity_max = self.service._calculate_service_complexity(3.0, 1.0)
        self.assertLessEqual(complexity_max, 8.0)
    
    def test_transform_to_inspection_risk(self):
        """Test inspection risk transformation"""
        # Test normal case
        risk = self.service._transform_to_inspection_risk(0.2, 0.1)
        self.assertGreaterEqual(risk, 0.0)
        self.assertLessEqual(risk, 1.0)
        
        # Test edge cases
        risk_zero = self.service._transform_to_inspection_risk(0.0, 0.0)
        self.assertEqual(risk_zero, 0.0)
        
        risk_high = self.service._transform_to_inspection_risk(1.0, 1.0)
        self.assertLessEqual(risk_high, 1.0)
    
    def test_estimate_provider_rating(self):
        """Test provider rating estimation"""
        rating = self.service._estimate_provider_rating(self.sample_dashboard_features)
        
        # Should be in CQC rating range
        self.assertGreaterEqual(rating, 1.0)
        self.assertLessEqual(rating, 4.0)
        
        # Test with high quality features
        high_quality_features = self.sample_dashboard_features.copy()
        high_quality_features.update({
            'incident_frequency_risk': 0.01,
            'medication_risk': 0.0,
            'safeguarding_risk': 0.0,
            'care_plan_compliance': 0.98,
            'resident_engagement': 0.95,
            'staff_compliance_score': 0.98
        })
        
        high_rating = self.service._estimate_provider_rating(high_quality_features)
        self.assertGreater(high_rating, rating)  # Should be higher than baseline
    
    def test_lookup_regional_risk(self):
        """Test regional risk lookup"""
        # Test various postcodes
        test_postcodes = ['SW1A 1AA', 'M1 1AA', 'B1 1AA', 'E1 6AN', 'N1 7GU']
        
        for postcode in test_postcodes:
            risk = self.service._lookup_regional_risk(postcode)
            self.assertGreaterEqual(risk, 0.0)
            self.assertLessEqual(risk, 1.0)
        
        # Test caching
        risk1 = self.service._lookup_regional_risk('SW1A 1AA')
        risk2 = self.service._lookup_regional_risk('SW1A 1AA')
        self.assertEqual(risk1, risk2)
    
    def test_calculate_operational_stability(self):
        """Test operational stability calculation"""
        stability = self.service._calculate_operational_stability(
            self.sample_dashboard_features
        )
        
        self.assertGreaterEqual(stability, 0.0)
        self.assertLessEqual(stability, 1.0)
        
        # Test with unstable features
        unstable_features = self.sample_dashboard_features.copy()
        unstable_features.update({
            'incident_frequency_risk': 0.8,
            'care_plan_compliance': 0.5,
            'staff_compliance_score': 0.6,
            'days_since_last_incident': 5
        })
        
        unstable_stability = self.service._calculate_operational_stability(
            unstable_features
        )
        self.assertLess(unstable_stability, stability)
    
    def test_calculate_care_quality_indicator(self):
        """Test care quality indicator calculation"""
        quality = self.service._calculate_care_quality_indicator(
            self.sample_dashboard_features
        )
        
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        
        # Test with poor quality features
        poor_quality_features = self.sample_dashboard_features.copy()
        poor_quality_features.update({
            'care_plan_compliance': 0.4,
            'resident_engagement': 0.3,
            'incident_frequency_risk': 0.7,
            'falls_risk': 0.6,
            'medication_risk': 0.5
        })
        
        poor_quality = self.service._calculate_care_quality_indicator(
            poor_quality_features
        )
        self.assertLess(poor_quality, quality)
    
    def test_normalize_features(self):
        """Test feature normalization"""
        test_features = {
            'bed_capacity': 250,  # Above normal range
            'occupancy_rate': 1.2,  # Above valid range
            'service_complexity_score': -1.0,  # Below valid range
            'provider_avg_rating': 5.0  # Above CQC scale
        }
        
        normalized = self.service._normalize_features(test_features)
        
        # Check that values are clipped to valid ranges
        self.assertLessEqual(normalized['bed_capacity'], 200)
        self.assertLessEqual(normalized['occupancy_rate'], 1.0)
        self.assertGreaterEqual(normalized['service_complexity_score'], 1.0)
        self.assertLessEqual(normalized['provider_avg_rating'], 4.0)
    
    def test_get_feature_importance_mapping(self):
        """Test feature importance mapping"""
        mapping = self.service.get_feature_importance_mapping()
        
        # Check that mapping contains expected features
        expected_features = [
            'bed_capacity', 'service_complexity_score', 'care_quality_indicator',
            'operational_stability', 'inspection_overdue_risk'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, mapping)
            self.assertIsInstance(mapping[feature], str)
            self.assertGreater(len(mapping[feature]), 0)
    
    def test_interaction_features(self):
        """Test interaction feature calculations"""
        aligned_features = self.service.transform_dashboard_to_cqc_features(
            self.sample_dashboard_features
        )
        
        # Test complexity_scale_interaction
        expected_complexity_scale = (
            aligned_features['service_complexity_score'] * 
            aligned_features['provider_location_count']
        )
        self.assertAlmostEqual(
            aligned_features['complexity_scale_interaction'],
            expected_complexity_scale,
            places=3
        )
        
        # Test inspection_regional_risk
        expected_inspection_regional = (
            aligned_features['inspection_overdue_risk'] * 
            aligned_features['regional_risk_rate']
        )
        self.assertAlmostEqual(
            aligned_features['inspection_regional_risk'],
            expected_inspection_regional,
            places=3
        )
    
    def test_missing_features_handling(self):
        """Test handling of missing dashboard features"""
        minimal_features = {
            'bed_capacity': 30,
            'occupancy_rate': 0.8
        }
        
        # Should not raise exception
        aligned_features = self.service.transform_dashboard_to_cqc_features(
            minimal_features
        )
        
        # Should have default values for missing features
        self.assertIn('service_complexity_score', aligned_features)
        self.assertIn('provider_avg_rating', aligned_features)
        self.assertIn('regional_risk_rate', aligned_features)
    
    @patch('src.ml.feature_alignment.logger')
    def test_error_handling(self, mock_logger):
        """Test error handling in feature transformation"""
        # Test with invalid feature types
        invalid_features = {
            'bed_capacity': 'invalid',
            'occupancy_rate': None
        }
        
        # Should handle gracefully and log errors
        try:
            aligned_features = self.service.transform_dashboard_to_cqc_features(
                invalid_features
            )
            # Should still return some features with defaults
            self.assertIsInstance(aligned_features, dict)
        except Exception as e:
            # If exception is raised, ensure it's logged
            mock_logger.error.assert_called()


if __name__ == '__main__':
    # Set up basic logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the tests
    unittest.main(verbosity=2)