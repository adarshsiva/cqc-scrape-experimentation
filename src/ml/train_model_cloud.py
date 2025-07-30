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
            predictions = []
            for model in self.models.values():
                pred_proba = model.predict_proba(X)[:, 1]
                predictions.append(pred_proba)
            
            # Average predictions
            ensemble_proba = np.mean(predictions, axis=0)
            return ensemble_proba
            
        return ensemble_predict
        
    def save_models(self):
        """Save models to GCS"""
        logger.info("Saving models to GCS...")
        
        model_package = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'ensemble_predict': self.create_ensemble_predictor(),
            'training_timestamp': datetime.now().isoformat()
        }
        
        # Save to GCS
        bucket = self.storage_client.bucket("machine-learning-exp-467008-cqc-ml-artifacts")
        blob = bucket.blob("models/proactive/model_package.pkl")
        
        with blob.open('wb') as f:
            pickle.dump(model_package, f)
            
        logger.info("Models saved successfully")
        
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
        results = self.train_models(X_train, y_train, X_val, y_val)
        
        # Print results
        print("\n=== MODEL TRAINING RESULTS ===")
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            print(f"  AUC Score: {metrics['auc_score']:.4f}")
            print(f"  Accuracy: {metrics['report']['accuracy']:.4f}")
            print(f"  At-Risk Precision: {metrics['report']['1']['precision']:.4f}")
            print(f"  At-Risk Recall: {metrics['report']['1']['recall']:.4f}")
        
        # Save models
        self.save_models()
        
        logger.info("Training completed successfully!")

def main():
    trainer = CQCModelTrainer()
    trainer.run_training_pipeline()

if __name__ == "__main__":
    main()