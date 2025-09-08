#!/usr/bin/env python3
"""
Test script for Unified CQC Model Trainer.
Validates that the trainer can be imported and initialized correctly.
"""

import os
import sys
import unittest
from unittest.mock import Mock, patch
import pandas as pd
import numpy as np

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from src.ml.unified_model_trainer import UnifiedCQCModelTrainer


class TestUnifiedCQCModelTrainer(unittest.TestCase):
    """Test cases for UnifiedCQCModelTrainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.project_id = 'test-project-123'
        self.region = 'europe-west2'
        
        # Mock GCP clients to avoid authentication issues in testing
        with patch('google.cloud.bigquery.Client'), \
             patch('google.cloud.storage.Client'), \
             patch('google.cloud.aiplatform.init'):
            self.trainer = UnifiedCQCModelTrainer(self.project_id, self.region)
    
    def test_initialization(self):
        """Test trainer initialization."""
        self.assertEqual(self.trainer.project_id, self.project_id)
        self.assertEqual(self.trainer.region, self.region)
        self.assertEqual(self.trainer.dataset_id, 'cqc_dataset')
        self.assertIsNotNone(self.trainer.UNIFIED_FEATURE_COLUMNS)
        self.assertEqual(len(self.trainer.RATING_LABELS), 5)
    
    def test_rating_mapping(self):
        """Test CQC rating mapping."""
        expected_mappings = {
            'Outstanding': 4,
            'Good': 3,
            'Requires improvement': 2,
            'Inadequate': 1,
            'No published rating': 0,
            None: 0
        }
        
        for rating, expected_value in expected_mappings.items():
            self.assertEqual(self.trainer.RATING_MAP[rating], expected_value)
    
    def test_unified_feature_columns(self):
        """Test unified feature column definitions."""
        features = self.trainer.UNIFIED_FEATURE_COLUMNS
        
        # Check that key feature types are included
        operational_features = [f for f in features if any(term in f for term in ['bed', 'capacity', 'size'])]
        risk_features = [f for f in features if 'risk' in f]
        quality_features = [f for f in features if any(term in f for term in ['rating', 'score', 'quality'])]
        temporal_features = [f for f in features if any(term in f for term in ['days', 'age'])]
        
        self.assertGreater(len(operational_features), 0, "Should have operational features")
        self.assertGreater(len(risk_features), 0, "Should have risk features")
        self.assertGreater(len(quality_features), 0, "Should have quality features")
        self.assertGreater(len(temporal_features), 0, "Should have temporal features")
    
    def test_prepare_unified_features_with_sample_data(self):
        """Test feature preparation with sample data."""
        # Create sample training data
        sample_data = pd.DataFrame({
            'locationId': ['LOC001', 'LOC002', 'LOC003'],
            'numberOfBeds': [50, 30, 80],
            'overall_rating': ['Good', 'Outstanding', 'Requires improvement'],
            'overall_rating_numeric': [3, 4, 2],
            'days_since_inspection': [365, 200, 800],
            'registration_age_years': [5.0, 10.0, 2.0],
            'service_complexity': [3, 5, 2],
            'provider_scale': [1, 3, 1],
            'provider_reputation': [3.2, 4.0, 2.5],
            'regional_risk_rate': [0.15, 0.10, 0.25]
        })
        
        # Test feature preparation
        X, y = self.trainer._prepare_unified_features(sample_data)
        
        # Validate results
        self.assertIsInstance(X, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(X), 3)
        self.assertEqual(len(y), 3)
        self.assertTrue(all(col in X.columns for col in ['bed_capacity', 'facility_size_numeric']))
    
    def test_validate_feature_alignment(self):
        """Test feature alignment validation."""
        # Create sample features
        sample_features = pd.DataFrame({
            'bed_capacity': [30, 50, 80],
            'inspection_overdue_risk': [0.1, 0.5, 0.9],
            'service_complexity_score': [2.5, 3.0, 4.2],
            'provider_avg_rating': [3.0, 3.5, 2.8],
            'regional_risk_rate': [0.2, 0.15, 0.3]
        })
        
        validation_results = self.trainer._validate_feature_alignment(sample_features)
        
        # Validate results structure
        required_keys = ['valid_features', 'problematic_features', 'missing_features', 
                        'feature_stats', 'compatibility_score']
        for key in required_keys:
            self.assertIn(key, validation_results)
        
        self.assertIsInstance(validation_results['compatibility_score'], float)
        self.assertGreaterEqual(validation_results['compatibility_score'], 0.0)
        self.assertLessEqual(validation_results['compatibility_score'], 1.0)
    
    def test_feature_types_mapping(self):
        """Test feature type mapping for dashboard compatibility."""
        feature_types = self.trainer._get_feature_types()
        
        # Test that feature types are assigned correctly
        self.assertIsInstance(feature_types, dict)
        
        # Check some expected mappings
        if 'bed_capacity' in feature_types:
            self.assertEqual(feature_types['bed_capacity'], 'operational')
        
        if 'inspection_overdue_risk' in feature_types:
            self.assertEqual(feature_types['inspection_overdue_risk'], 'risk_indicator')
        
        if 'provider_avg_rating' in feature_types:
            self.assertEqual(feature_types['provider_avg_rating'], 'provider_context')


def run_basic_validation():
    """Run basic validation without full test suite."""
    print("="*60)
    print("UNIFIED CQC MODEL TRAINER - BASIC VALIDATION")
    print("="*60)
    
    try:
        # Test import
        print("✓ Successfully imported UnifiedCQCModelTrainer")
        
        # Test initialization (mocked)
        with patch('google.cloud.bigquery.Client'), \
             patch('google.cloud.storage.Client'), \
             patch('google.cloud.aiplatform.init'):
            trainer = UnifiedCQCModelTrainer('test-project', 'europe-west2')
            print("✓ Successfully initialized trainer")
        
        # Test feature columns
        print(f"✓ Defined {len(trainer.UNIFIED_FEATURE_COLUMNS)} unified features")
        
        # Test rating mapping
        print(f"✓ Rating mapping includes {len(trainer.RATING_MAP)} rating types")
        
        # Test feature type mapping
        feature_types = trainer._get_feature_types()
        print(f"✓ Feature type mapping includes {len(feature_types)} feature types")
        
        # Sample data test
        sample_data = pd.DataFrame({
            'numberOfBeds': [30, 50, 80],
            'overall_rating_numeric': [2, 3, 4],
            'days_since_inspection': [300, 400, 500]
        })
        
        X, y = trainer._prepare_unified_features(sample_data)
        print(f"✓ Feature preparation works: {len(X)} samples, {len(X.columns)} features")
        
        validation = trainer._validate_feature_alignment(X)
        print(f"✓ Feature validation works: {validation['compatibility_score']:.2%} compatibility")
        
        print("="*60)
        print("✅ ALL BASIC VALIDATIONS PASSED")
        print("✅ Unified trainer is ready for deployment!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--unittest':
        # Run full unit tests
        unittest.main(argv=[''], exit=False, verbosity=2)
    else:
        # Run basic validation
        success = run_basic_validation()
        sys.exit(0 if success else 1)