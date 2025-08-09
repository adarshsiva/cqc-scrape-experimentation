#!/usr/bin/env python3
"""
Test script to verify the ML training pipeline fixes
This script tests the key components without running the full training pipeline
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from src.ml.train_model_cloud import CQCModelTrainer

def test_classification_report_handling():
    """Test the classification report handling with edge cases"""
    
    # Test case 1: Normal case with both classes
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 1]
    report = classification_report(y_true, y_pred, output_dict=True)
    
    print("Test 1 - Normal case:")
    print(f"Available classes: {list(report.keys())}")
    
    # Test safe access
    if '1' in report and isinstance(report['1'], dict):
        print(f"  At-Risk Precision: {report['1']['precision']:.4f}")
        print(f"  At-Risk Recall: {report['1']['recall']:.4f}")
    else:
        print("  At-Risk metrics N/A")
    
    # Test case 2: Edge case - all predictions are class 0
    y_true = [0, 1, 0, 1, 1, 0]
    y_pred = [0, 0, 0, 0, 0, 0]  # All predicted as class 0
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    print("\nTest 2 - All predictions class 0:")
    print(f"Available classes: {list(report.keys())}")
    
    # Test safe access
    if '1' in report and isinstance(report['1'], dict):
        print(f"  At-Risk Precision: {report['1']['precision']:.4f}")
        print(f"  At-Risk Recall: {report['1']['recall']:.4f}")
    else:
        print("  At-Risk metrics N/A (class not found)")

def test_model_validation():
    """Test the model package validation"""
    trainer = CQCModelTrainer()
    
    # Test case 1: Valid model package
    valid_package = {
        'models': {'test_model': 'dummy'},
        'feature_columns': ['feature1', 'feature2'],
        'scaler': 'dummy_scaler',
        'ensemble_predict': lambda x: x,
        'training_timestamp': '2024-01-01'
    }
    
    try:
        trainer.validate_model_package(valid_package)
        print("\nTest 3 - Valid package: PASSED")
    except Exception as e:
        print(f"\nTest 3 - Valid package: FAILED - {e}")
    
    # Test case 2: Invalid model package (missing models)
    invalid_package = {
        'models': {},  # Empty models
        'feature_columns': ['feature1', 'feature2'],
        'scaler': 'dummy_scaler',
        'ensemble_predict': lambda x: x,
        'training_timestamp': '2024-01-01'
    }
    
    try:
        trainer.validate_model_package(invalid_package)
        print("Test 4 - Invalid package (empty models): FAILED - Should have raised error")
    except ValueError as e:
        print(f"Test 4 - Invalid package (empty models): PASSED - {e}")

if __name__ == "__main__":
    print("Testing ML training pipeline fixes...\n")
    test_classification_report_handling()
    test_model_validation()
    print("\nAll tests completed!")