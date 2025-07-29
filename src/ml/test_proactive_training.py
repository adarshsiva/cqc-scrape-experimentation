#!/usr/bin/env python3
"""
Test script for the proactive model training pipeline.

This script creates synthetic data to test the ProactiveModelTrainer
without requiring access to BigQuery or real data.
"""

import pandas as pd
import numpy as np
from train_proactive_model import ProactiveModelTrainer
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def generate_synthetic_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate synthetic data for testing the training pipeline.
    
    Args:
        n_samples: Number of samples to generate
        
    Returns:
        DataFrame with synthetic CQC data
    """
    logger.info(f"Generating {n_samples} synthetic samples...")
    
    np.random.seed(42)
    
    # Generate features
    data = {
        'providerId': [f'P{i:04d}' for i in range(n_samples)],
        'locationId': [f'L{i:04d}' for i in range(n_samples)],
        
        # Numerical features
        'number_of_beds': np.random.poisson(50, n_samples),
        'days_since_registration': np.random.exponential(1000, n_samples),
        'days_since_inspection': np.random.exponential(200, n_samples),
        'regulatedActivitiesCount': np.random.poisson(5, n_samples),
        'specialismsCount': np.random.poisson(3, n_samples),
        'serviceTypesCount': np.random.poisson(2, n_samples),
        'serviceUserGroupsCount': np.random.poisson(4, n_samples),
        'inspection_frequency': np.random.uniform(0.5, 2.0, n_samples),
        'historical_rating_changes': np.random.normal(0, 1, n_samples),
        
        # Binary features
        'overdue_inspection': np.random.binomial(1, 0.2, n_samples),
        'inherited_rating': np.random.binomial(1, 0.1, n_samples),
        
        # Risk scores (correlated with target)
        'overall_rating_score': np.random.uniform(1, 5, n_samples),
        'domain_risk_score': np.random.uniform(0, 10, n_samples),
        
        # Categorical features
        'region': np.random.choice(['North', 'South', 'East', 'West', 'Central'], n_samples),
        'provider_type': np.random.choice(['NHS', 'Private', 'Voluntary'], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate domain risk indicators (correlated with target)
    risk_factor = df['domain_risk_score'] / 10 + np.random.normal(0, 0.1, n_samples)
    df['safe_at_risk'] = (risk_factor + np.random.normal(0, 0.2, n_samples) > 0.6).astype(int)
    df['effective_at_risk'] = (risk_factor + np.random.normal(0, 0.2, n_samples) > 0.7).astype(int)
    df['caring_at_risk'] = (risk_factor + np.random.normal(0, 0.2, n_samples) > 0.8).astype(int)
    df['responsive_at_risk'] = (risk_factor + np.random.normal(0, 0.2, n_samples) > 0.7).astype(int)
    df['well_led_at_risk'] = (risk_factor + np.random.normal(0, 0.2, n_samples) > 0.6).astype(int)
    
    # Generate target variable (at_risk_label)
    # Higher risk score and more domain risks increase probability of being at risk
    at_risk_probability = (
        0.1 +  # Base probability
        0.1 * (df['domain_risk_score'] / 10) +
        0.1 * df['overdue_inspection'] +
        0.05 * (df['safe_at_risk'] + df['effective_at_risk'] + 
                df['caring_at_risk'] + df['responsive_at_risk'] + 
                df['well_led_at_risk']) +
        0.05 * (df['days_since_inspection'] > 365).astype(int)
    )
    
    df['at_risk_label'] = (np.random.random(n_samples) < at_risk_probability).astype(int)
    
    logger.info(f"Generated data with {df['at_risk_label'].sum()} at-risk providers "
                f"({df['at_risk_label'].mean():.1%})")
    
    return df


class MockProactiveModelTrainer(ProactiveModelTrainer):
    """Mock trainer that doesn't require GCP access."""
    
    def __init__(self):
        """Initialize without GCP clients."""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        
    def load_training_data(self, synthetic_data: pd.DataFrame = None) -> pd.DataFrame:
        """Load synthetic data instead of from BigQuery."""
        if synthetic_data is None:
            return generate_synthetic_data(2000)
        return synthetic_data
    
    def save_models(self, save_locally: bool = True) -> None:
        """Save models locally instead of to GCS."""
        if save_locally:
            import pickle
            import os
            
            # Create local directory for models
            os.makedirs('/tmp/proactive_models', exist_ok=True)
            
            # Save model package
            model_package = {
                'models': self.models,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'metrics': self.metrics,
                'training_timestamp': datetime.now().isoformat()
            }
            
            with open('/tmp/proactive_models/model_package.pkl', 'wb') as f:
                pickle.dump(model_package, f)
                
            logger.info("Models saved to /tmp/proactive_models/")


def test_training_pipeline():
    """Test the proactive model training pipeline."""
    logger.info("Starting test of proactive model training pipeline...")
    
    # Initialize mock trainer
    trainer = MockProactiveModelTrainer()
    
    # Generate and load synthetic data
    synthetic_data = generate_synthetic_data(3000)
    df = trainer.load_training_data(synthetic_data)
    
    # Prepare features
    X, y, feature_cols = trainer.prepare_features(df)
    logger.info(f"Feature shape: {X.shape}")
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train models
    results = trainer.train_models(X_train, y_train, X_val, y_val, use_cross_validation=False)
    
    # Print results
    print("\n" + "="*60)
    print("MODEL PERFORMANCE RESULTS")
    print("="*60)
    
    for model_name, metrics in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  AUC Score: {metrics['auc_score']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  At-Risk Precision: {metrics['at_risk_precision']:.4f}")
        print(f"  At-Risk Recall: {metrics['at_risk_recall']:.4f}")
        print(f"  At-Risk F1-Score: {metrics['at_risk_f1']:.4f}")
    
    # Test ensemble predictor
    ensemble_predict = trainer.create_ensemble_predictor()
    ensemble_proba = ensemble_predict(X_val)
    
    print(f"\nEnsemble predictions shape: {ensemble_proba.shape}")
    print(f"Ensemble prediction range: [{ensemble_proba.min():.3f}, {ensemble_proba.max():.3f}]")
    
    # Save models locally
    trainer.save_models(save_locally=True)
    
    # Generate report
    report = trainer.generate_report()
    print("\n" + report)
    
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    test_training_pipeline()