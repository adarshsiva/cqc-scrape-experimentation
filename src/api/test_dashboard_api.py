#!/usr/bin/env python3
"""
Test suite for Dashboard Prediction API

This file provides comprehensive tests for the real-time CQC prediction service,
including unit tests for individual components and integration tests for the full API.
"""

import unittest
import json
import os
from unittest.mock import patch, MagicMock
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the service modules
from api.dashboard_prediction_service import (
    app, ModelPredictionService, validate_api_token, 
    get_client_id, _calculate_data_coverage
)

class TestModelPredictionService(unittest.TestCase):
    """Test cases for ModelPredictionService."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.service = ModelPredictionService()
        self.test_features = {
            'bed_capacity': 45.0,
            'occupancy_rate': 0.85,
            'incident_frequency_risk': 0.2,
            'medication_risk': 0.1,
            'safeguarding_risk': 0.0,
            'care_quality_indicator': 0.8,
            'operational_stability': 0.7
        }
    
    def test_predict_cqc_rating(self):
        """Test CQC rating prediction."""
        prediction = self.service.predict_cqc_rating(self.test_features)
        
        # Check required fields
        self.assertIn('rating', prediction)
        self.assertIn('rating_text', prediction)
        self.assertIn('confidence', prediction)
        self.assertIn('risk_level', prediction)
        
        # Check value ranges
        self.assertGreaterEqual(prediction['rating'], 1)
        self.assertLessEqual(prediction['rating'], 4)
        self.assertGreaterEqual(prediction['confidence'], 0.0)
        self.assertLessEqual(prediction['confidence'], 1.0)
        
        # Check rating text mapping
        expected_text = self.service.RATING_LABELS[prediction['rating']]
        self.assertEqual(prediction['rating_text'], expected_text)
    
    def test_explain_prediction(self):
        """Test feature importance explanations."""
        explanation = self.service.explain_prediction(self.test_features)
        
        # Check structure
        self.assertIn('positive', explanation)
        self.assertIn('negative', explanation)
        self.assertIsInstance(explanation['positive'], list)
        self.assertIsInstance(explanation['negative'], list)
        
        # Check factor structure
        if explanation['positive']:
            factor = explanation['positive'][0]
            self.assertIn('factor', factor)
            self.assertIn('impact', factor)
            self.assertIn('value', factor)
            self.assertIn('interpretation', factor)
    
    def test_generate_recommendations(self):
        """Test recommendation generation."""
        prediction_result = {
            'rating': 2,
            'rating_text': 'Requires improvement',
            'confidence': 0.7
        }
        
        recommendations = self.service.generate_recommendations(
            prediction_result, self.test_features
        )
        
        # Check structure
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        # Check recommendation structure
        rec = recommendations[0]
        self.assertIn('category', rec)
        self.assertIn('priority', rec)
        self.assertIn('action', rec)
        self.assertIn('timeline', rec)
        
        # High priority recommendations for low ratings
        high_priority_found = any(
            rec['priority'] == 'High' for rec in recommendations
        )
        self.assertTrue(high_priority_found, "Expected high priority recommendations for low rating")
    
    def test_mock_prediction_logic(self):
        """Test mock prediction logic with different risk scenarios."""
        # High risk scenario
        high_risk_features = self.test_features.copy()
        high_risk_features.update({
            'incident_frequency_risk': 0.8,
            'medication_risk': 0.9,
            'safeguarding_risk': 1.0,
            'care_quality_indicator': 0.3
        })
        
        prediction = self.service._mock_prediction(None, high_risk_features)
        self.assertLessEqual(prediction['rating'], 2, "High risk should result in low rating")
        
        # Low risk scenario
        low_risk_features = self.test_features.copy()
        low_risk_features.update({
            'incident_frequency_risk': 0.05,
            'medication_risk': 0.02,
            'safeguarding_risk': 0.0,
            'care_quality_indicator': 0.95
        })
        
        prediction = self.service._mock_prediction(None, low_risk_features)
        self.assertGreaterEqual(prediction['rating'], 3, "Low risk should result in good rating")


class TestAPIEndpoints(unittest.TestCase):
    """Test cases for API endpoints."""
    
    def setUp(self):
        """Set up test fixtures."""
        app.config['TESTING'] = True
        self.client = app.test_client()
        self.test_care_home_id = 'test_care_home_123'
    
    def test_health_endpoint(self):
        """Test health check endpoint."""
        response = self.client.get('/health')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        self.assertEqual(data['service'], 'CQC Dashboard Prediction API')
        self.assertIn('timestamp', data)
    
    @patch.dict(os.environ, {'ENABLE_AUTH': 'false'})
    @patch('api.dashboard_prediction_service.DashboardFeatureExtractor')
    @patch('api.dashboard_prediction_service.feature_alignment_service')
    @patch('api.dashboard_prediction_service.model_prediction_service')
    def test_prediction_endpoint(self, mock_model_service, mock_alignment_service, mock_extractor_class):
        """Test prediction endpoint with mocked dependencies."""
        # Mock feature extraction
        mock_extractor = MagicMock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_care_home_features.return_value = {
            'bed_capacity': 45.0,
            'occupancy_rate': 0.85,
            'incident_frequency_risk': 0.2
        }
        
        # Mock feature alignment
        mock_alignment_service.transform_dashboard_to_cqc_features.return_value = {
            'bed_capacity': 45.0,
            'occupancy_rate': 0.85,
            'care_quality_indicator': 0.8
        }
        
        # Mock prediction service
        mock_model_service.predict_cqc_rating.return_value = {
            'rating': 3,
            'rating_text': 'Good',
            'confidence': 0.85,
            'risk_level': 'Low'
        }
        mock_model_service.explain_prediction.return_value = {
            'positive': [{'factor': 'Test positive', 'impact': 0.8, 'value': 0.9, 'interpretation': 'Good'}],
            'negative': []
        }
        mock_model_service.generate_recommendations.return_value = [
            {'category': 'Test', 'priority': 'Low', 'action': 'Test action', 'timeline': 'Test timeline'}
        ]
        
        # Make request
        response = self.client.get(f'/api/cqc-prediction/dashboard/{self.test_care_home_id}')
        
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertEqual(data['care_home_id'], self.test_care_home_id)
        self.assertIn('prediction', data)
        self.assertIn('contributing_factors', data)
        self.assertIn('recommendations', data)
        self.assertIn('data_freshness', data)
        
        # Verify prediction structure
        prediction = data['prediction']
        self.assertEqual(prediction['predicted_rating'], 3)
        self.assertEqual(prediction['predicted_rating_text'], 'Good')
        self.assertEqual(prediction['confidence_score'], 0.85)


class TestUtilityFunctions(unittest.TestCase):
    """Test cases for utility functions."""
    
    def test_calculate_data_coverage(self):
        """Test data coverage calculation."""
        # Full coverage scenario
        full_features = {f'feature_{i}': float(i) for i in range(20)}
        coverage = _calculate_data_coverage(full_features)
        self.assertEqual(coverage, 1.0)
        
        # Partial coverage scenario
        partial_features = {f'feature_{i}': float(i) for i in range(10)}
        coverage = _calculate_data_coverage(partial_features)
        self.assertEqual(coverage, 0.5)
        
        # Empty features
        empty_features = {}
        coverage = _calculate_data_coverage(empty_features)
        self.assertEqual(coverage, 0.0)
        
        # Features with None/zero values
        mixed_features = {
            'feature_1': 1.0,
            'feature_2': 0.0,  # Should not count
            'feature_3': None,  # Should not count
            'feature_4': 2.0
        }
        coverage = _calculate_data_coverage(mixed_features)
        self.assertEqual(coverage, 0.1)  # 2 out of 20 expected
    
    @patch('api.dashboard_prediction_service.secret_client')
    def test_validate_api_token_success(self, mock_secret_client):
        """Test successful API token validation."""
        # Mock secret manager response
        mock_response = MagicMock()
        mock_response.payload.data.decode.return_value = 'valid_token_123'
        mock_secret_client.access_secret_version.return_value = mock_response
        
        result = validate_api_token('valid_token_123')
        self.assertTrue(result)
    
    @patch('api.dashboard_prediction_service.secret_client')
    def test_validate_api_token_failure(self, mock_secret_client):
        """Test failed API token validation."""
        # Mock secret manager response
        mock_response = MagicMock()
        mock_response.payload.data.decode.return_value = 'valid_token_123'
        mock_secret_client.access_secret_version.return_value = mock_response
        
        result = validate_api_token('invalid_token_456')
        self.assertFalse(result)
    
    @patch('api.dashboard_prediction_service.secret_client')
    def test_validate_api_token_exception(self, mock_secret_client):
        """Test API token validation with exception."""
        mock_secret_client.access_secret_version.side_effect = Exception('Secret not found')
        
        result = validate_api_token('any_token')
        self.assertFalse(result)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration test scenarios."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        app.config['TESTING'] = True
        self.client = app.test_client()
    
    @patch.dict(os.environ, {'ENABLE_AUTH': 'false'})
    def test_error_handling(self):
        """Test API error handling."""
        # Test with invalid care home ID
        response = self.client.get('/api/cqc-prediction/dashboard/invalid_id')
        
        # Should return 500 with error structure
        self.assertEqual(response.status_code, 500)
        
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('error_type', data)
        self.assertIn('message', data)
        self.assertEqual(data['error_type'], 'PREDICTION_ERROR')
    
    def test_authentication_required(self):
        """Test that authentication is required when enabled."""
        with patch.dict(os.environ, {'ENABLE_AUTH': 'true'}):
            response = self.client.get('/api/cqc-prediction/dashboard/test_id')
            
            self.assertEqual(response.status_code, 401)
            
            data = json.loads(response.data)
            self.assertEqual(data['error_type'], 'AUTHENTICATION_ERROR')


def run_tests():
    """Run all tests and return results."""
    # Create test suite
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestModelPredictionService,
        TestAPIEndpoints,
        TestUtilityFunctions,
        TestIntegrationScenarios
    ]
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    print("🧪 Running Dashboard Prediction API Tests")
    print("=" * 50)
    
    success = run_tests()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ All tests passed!")
        exit(0)
    else:
        print("❌ Some tests failed!")
        exit(1)