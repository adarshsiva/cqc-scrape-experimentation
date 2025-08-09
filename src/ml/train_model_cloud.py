#!/usr/bin/env python3
"""
Cloud-based model training for CQC rating prediction
Simplified version that works with available data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import xgboost as xgb
import lightgbm as lgb
from google.cloud import bigquery
from google.cloud import storage
import pickle
import json
import logging
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CQCModelTrainer:
    def __init__(self):
        self.project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        self.client = bigquery.Client(project=self.project_id)
        self.storage_client = storage.Client(project=self.project_id)
        self.models = {}
        self.scaler = StandardScaler()
        
    def load_training_data(self):
        """Load features from BigQuery ML features view"""
        logger.info("Loading training data from BigQuery...")
        
        query = """
        SELECT * 
        FROM `machine-learning-exp-467008.cqc_data.ml_features_proactive`
        WHERE days_since_inspection IS NOT NULL
        """
        
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} samples")
        
        # Check target distribution
        target_dist = df['at_risk_label'].value_counts()
        logger.info(f"Target distribution:\n{target_dist}")
        
        return df
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Numerical features
        numerical_features = [
            'number_of_beds', 'days_since_registration', 'days_since_inspection',
            'regulatedActivitiesCount', 'specialismsCount', 'serviceTypesCount',
            'overall_rating_score', 'domain_risk_score'
        ]
        
        # Binary features
        binary_features = [
            'overdue_inspection', 'safe_at_risk', 'effective_at_risk',
            'caring_at_risk', 'responsive_at_risk', 'well_led_at_risk'
        ]
        
        # Categorical features
        categorical_features = ['region']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix_sep='_')
        
        # Get all feature columns
        feature_cols = numerical_features + binary_features
        feature_cols += [col for col in df_encoded.columns if col.startswith('region_')]
        
        # Ensure columns exist
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        
        X = df_encoded[feature_cols]
        y = df_encoded['at_risk_label']
        
        # Fill missing values
        X = X.fillna(0)
        
        # Scale numerical features
        X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        self.feature_columns = feature_cols
        
        return X, y
        
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models"""
        logger.info("Training models...")
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            use_label_encoder=False,
            eval_metric='logloss'
        )
        self.models['xgboost'].fit(X_train, y_train)
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            verbosity=-1
        )
        self.models['lightgbm'].fit(X_train, y_train)
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.models['random_forest'].fit(X_train, y_train)
        
        # Evaluate models
        results = {}
        for name, model in self.models.items():
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            results[name] = {
                'auc_score': roc_auc_score(y_val, y_proba),
                'report': classification_report(y_val, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
            logger.info(f"{name} AUC: {results[name]['auc_score']:.4f}")
            
        return results
        
    def create_ensemble_predictor(self):
        """Create ensemble model"""
        def ensemble_predict(X):
            if not self.models:
                raise ValueError("No models available for ensemble prediction")
                
            predictions = []
            for model_name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed during prediction: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("All models failed during prediction")
            
            # Average predictions
            ensemble_proba = np.mean(predictions, axis=0)
            return ensemble_proba
            
        return ensemble_predict
        
    def validate_model_package(self, model_package):
        """Validate the model package before saving"""
        required_keys = ['models', 'feature_columns', 'scaler', 'ensemble_predict', 'training_timestamp']
        
        for key in required_keys:
            if key not in model_package:
                raise ValueError(f"Missing required key in model package: {key}")
        
        if not model_package['models']:
            raise ValueError("No models in model package")
            
        if not model_package['feature_columns']:
            raise ValueError("No feature columns in model package")
        
        logger.info(f"Model package validation passed: {len(model_package['models'])} models, {len(model_package['feature_columns'])} features")
        
    def save_models(self):
        """Save models to GCS"""
        logger.info("Saving models to GCS...")
        
        # Validate that we have models to save
        if not self.models:
            logger.error("No models trained - cannot save model package")
            raise ValueError("No models available to save")
            
        # Validate required attributes exist
        if not hasattr(self, 'feature_columns'):
            logger.error("feature_columns not available - training may have failed")
            raise ValueError("Feature columns not defined - training incomplete")
        
        try:
            model_package = {
                'models': self.models,
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'ensemble_predict': self.create_ensemble_predictor(),
                'training_timestamp': datetime.now().isoformat()
            }
            
            # Validate model package before saving
            self.validate_model_package(model_package)
            
            # Save to GCS
            bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-ml-artifacts")
            blob = bucket.blob("models/proactive/model_package.pkl")
            
            with blob.open('wb') as f:
                pickle.dump(model_package, f)
                
            logger.info(f"Models saved successfully to GCS: {blob.name}")
            logger.info(f"Saved {len(self.models)} models with {len(self.feature_columns)} features")
            
        except Exception as e:
            logger.error(f"Failed to save models to GCS: {str(e)}")
            raise
        
    def run_training_pipeline(self):
        """Main training pipeline"""
        # Load data
        df = self.load_training_data()
        
        if len(df) == 0:
            logger.error("No data available for training")
            return
            
        # Prepare features
        X, y = self.prepare_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set: {len(X_train)}, Validation set: {len(X_val)}")
        
        # Train models
        try:
            results = self.train_models(X_train, y_train, X_val, y_val)
            
            # Print results
            print("\n=== MODEL TRAINING RESULTS ===")
            for model_name, metrics in results.items():
                print(f"\n{model_name.upper()}:")
                print(f"  AUC Score: {metrics['auc_score']:.4f}")
                print(f"  Accuracy: {metrics['report']['accuracy']:.4f}")
                
                # Safely access at-risk class metrics (class '1')
                if '1' in metrics['report'] and isinstance(metrics['report']['1'], dict):
                    print(f"  At-Risk Precision: {metrics['report']['1']['precision']:.4f}")
                    print(f"  At-Risk Recall: {metrics['report']['1']['recall']:.4f}")
                else:
                    print("  At-Risk Precision: N/A (class not found in predictions)")
                    print("  At-Risk Recall: N/A (class not found in predictions)")
                    logger.warning(f"Class '1' not found in classification report for {model_name}")
                    logger.warning(f"Available classes: {list(metrics['report'].keys())}")
            
            # Save models only if training succeeded
            self.save_models()
            
            logger.info("Training completed successfully!")
            
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            # Try to save any models that were successfully trained
            if self.models:
                logger.info("Attempting to save partially trained models...")
                try:
                    self.save_models()
                    logger.info("Partially trained models saved successfully")
                except Exception as save_e:
                    logger.error(f"Failed to save partially trained models: {str(save_e)}")
            raise

def main():
    trainer = CQCModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()