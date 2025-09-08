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
        """Load features from BigQuery ML features table"""
        logger.info("Loading training data from BigQuery...")
        
        # Use environment variable for table or default to real features
        training_table = os.environ.get('TRAINING_TABLE', 'cqc_data.ml_training_features_real')
        
        query = f"""
        SELECT * 
        FROM `machine-learning-exp-467008.{training_table}`
        WHERE days_since_inspection IS NOT NULL
        AND overall_rating_numeric IS NOT NULL
        """
        
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} samples")
        
        # Check target distribution for multi-class
        if 'overall_rating_numeric' in df.columns:
            target_dist = df['overall_rating_numeric'].value_counts().sort_index()
            logger.info(f"Rating distribution:\n{target_dist}")
        elif 'at_risk_label' in df.columns:
            target_dist = df['at_risk_label'].value_counts()
            logger.info(f"At-risk distribution:\n{target_dist}")
        
        return df
        
    def prepare_features(self, df):
        """Prepare features for training"""
        # Check if this is the new real data format or legacy format
        if 'overall_rating_numeric' in df.columns:
            # New real data format
            numerical_features = [
                'bed_capacity', 'days_since_registration', 'days_since_inspection',
                'regulated_activities_count', 'service_types_count', 'specialisms_count',
                'service_complexity_score', 'provider_location_count', 'provider_avg_rating',
                'provider_rating_consistency', 'regional_risk_rate', 'regional_avg_beds',
                'complexity_scale_interaction', 'inspection_regional_risk'
            ]
            
            # Binary features
            binary_features = [
                'inspection_overdue_risk', 'registration_issues', 'high_risk_current',
                'high_risk_predicted_features'
            ]
            
            target_col = 'overall_rating_numeric'
        else:
            # Legacy format
            numerical_features = [
                'number_of_beds', 'days_since_registration', 'days_since_inspection',
                'regulatedActivitiesCount', 'specialismsCount', 'serviceTypesCount',
                'overall_rating_score', 'domain_risk_score'
            ]
            
            binary_features = [
                'overdue_inspection', 'safe_at_risk', 'effective_at_risk',
                'caring_at_risk', 'responsive_at_risk', 'well_led_at_risk'
            ]
            
            target_col = 'at_risk_label'
        
        # Categorical features
        categorical_features = ['region']
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix_sep='_')
        
        # Get all feature columns that exist in the data
        feature_cols = []
        for col in numerical_features + binary_features:
            if col in df_encoded.columns:
                feature_cols.append(col)
        
        # Add region columns
        feature_cols += [col for col in df_encoded.columns if col.startswith('region_')]
        
        # Ensure we have some features
        if not feature_cols:
            raise ValueError("No valid feature columns found in the data")
        
        X = df_encoded[feature_cols].copy()
        y = df_encoded[target_col]
        
        # Fill missing values
        X = X.fillna(0)
        
        # Scale numerical features that exist
        existing_numerical = [col for col in numerical_features if col in X.columns]
        if existing_numerical:
            X[existing_numerical] = self.scaler.fit_transform(X[existing_numerical])
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        logger.info(f"Feature columns: {feature_cols}")
        self.feature_columns = feature_cols
        
        return X, y
        
    def train_models(self, X_train, y_train, X_val, y_val):
        """Train multiple models"""
        logger.info("Training models...")
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(y_train))
        is_multiclass = n_classes > 2
        
        logger.info(f"Training for {n_classes} classes (multiclass: {is_multiclass})")
        
        # XGBoost
        if is_multiclass:
            self.models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                objective='multi:softprob',
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        else:
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
            y_proba = model.predict_proba(X_val)
            
            # Calculate metrics
            results[name] = {
                'report': classification_report(y_val, y_pred, output_dict=True),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
            # AUC score calculation
            if is_multiclass:
                # For multi-class, use macro average or one-vs-rest
                try:
                    from sklearn.metrics import roc_auc_score
                    auc_score = roc_auc_score(y_val, y_proba, multi_class='ovr', average='macro')
                    results[name]['auc_score'] = auc_score
                except ValueError:
                    # Fallback to accuracy if AUC fails
                    results[name]['auc_score'] = results[name]['report']['accuracy']
            else:
                # Binary classification
                results[name]['auc_score'] = roc_auc_score(y_val, y_proba[:, 1])
            
            logger.info(f"{name} Score: {results[name]['auc_score']:.4f}")
            
        return results
        
    def create_ensemble_predictor(self):
        """Create ensemble model"""
        def ensemble_predict(X):
            if not self.models:
                raise ValueError("No models available for ensemble prediction")
                
            predictions = []
            for model_name, model in self.models.items():
                try:
                    pred_proba = model.predict_proba(X)
                    predictions.append(pred_proba)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed during prediction: {str(e)}")
                    continue
            
            if not predictions:
                raise ValueError("All models failed during prediction")
            
            # Average predictions across all models
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
                print(f"  Score: {metrics['auc_score']:.4f}")
                print(f"  Accuracy: {metrics['report']['accuracy']:.4f}")
                
                # Print class-wise metrics for available classes
                available_classes = [k for k in metrics['report'].keys() 
                                   if isinstance(metrics['report'][k], dict) and 'precision' in metrics['report'][k]]
                
                if available_classes:
                    for class_label in available_classes:
                        class_name = {
                            '1': 'Inadequate',
                            '2': 'Requires improvement', 
                            '3': 'Good',
                            '4': 'Outstanding'
                        }.get(str(class_label), f'Class {class_label}')
                        
                        precision = metrics['report'][class_label]['precision']
                        recall = metrics['report'][class_label]['recall']
                        print(f"  {class_name} - Precision: {precision:.4f}, Recall: {recall:.4f}")
                else:
                    print("  Individual class metrics: N/A")
                    logger.warning(f"No class-specific metrics found for {model_name}")
                    logger.warning(f"Available report keys: {list(metrics['report'].keys())}")
            
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