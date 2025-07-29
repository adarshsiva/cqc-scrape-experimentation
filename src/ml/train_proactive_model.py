#!/usr/bin/env python3
"""
Proactive Model Trainer for CQC Rating Prediction

This module trains multiple models (XGBoost, LightGBM, Random Forest) to identify
healthcare providers at risk of rating downgrades. It creates an ensemble predictor
that combines predictions from all models for improved accuracy.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    precision_recall_curve,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from google.cloud import bigquery
from google.cloud import storage
import pickle
import json
from datetime import datetime
import logging
from typing import Dict, List, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProactiveModelTrainer:
    """Trains models to proactively identify at-risk healthcare providers."""
    
    def __init__(self, project_id: str = None, dataset_id: str = "cqc_data"):
        """Initialize the trainer with GCP clients.
        
        Args:
            project_id: GCP project ID (defaults to environment)
            dataset_id: BigQuery dataset ID
        """
        self.client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        self.dataset_id = dataset_id
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.metrics = {}
        
    def load_training_data(self, 
                          table_name: str = "ml_features_proactive",
                          sample_size: int = None) -> pd.DataFrame:
        """Load features from BigQuery.
        
        Args:
            table_name: Name of the BigQuery table containing features
            sample_size: Optional sample size for testing
            
        Returns:
            DataFrame with training data
        """
        logger.info(f"Loading training data from {self.dataset_id}.{table_name}")
        
        query = f"""
        SELECT 
            -- Provider identifiers
            providerId,
            locationId,
            
            -- Core features
            number_of_beds,
            days_since_registration,
            days_since_inspection,
            regulatedActivitiesCount,
            specialismsCount,
            serviceTypesCount,
            serviceUserGroupsCount,
            
            -- Inspection features
            overdue_inspection,
            overall_rating_score,
            safe_at_risk,
            effective_at_risk,
            caring_at_risk,
            responsive_at_risk,
            well_led_at_risk,
            domain_risk_score,
            
            -- Provider characteristics
            region,
            provider_type,
            inherited_rating,
            
            -- Historical features
            inspection_frequency,
            historical_rating_changes,
            
            -- Target variable
            at_risk_label
        FROM `{self.dataset_id}.{table_name}`
        WHERE days_since_inspection < 1000  -- Recent data only
        AND at_risk_label IS NOT NULL
        """
        
        if sample_size:
            query += f"\nLIMIT {sample_size}"
        
        df = self.client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} samples from BigQuery")
        
        # Basic data validation
        self._validate_data(df)
        
        return df
        
    def _validate_data(self, df: pd.DataFrame) -> None:
        """Validate the loaded data.
        
        Args:
            df: DataFrame to validate
        """
        logger.info("Validating data...")
        
        # Check for required columns
        required_cols = ['at_risk_label', 'number_of_beds', 'days_since_inspection']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check target distribution
        target_dist = df['at_risk_label'].value_counts(normalize=True)
        logger.info(f"Target distribution:\n{target_dist}")
        
        # Warn if highly imbalanced
        if target_dist.min() < 0.1:
            logger.warning("Highly imbalanced dataset detected. Consider using SMOTE or class weights.")
        
    def prepare_features(self, 
                        df: pd.DataFrame,
                        scale_features: bool = True) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare features and labels for training.
        
        Args:
            df: Raw DataFrame
            scale_features: Whether to scale numerical features
            
        Returns:
            Tuple of (features, labels, feature_column_names)
        """
        logger.info("Preparing features...")
        
        # Define feature columns
        numerical_features = [
            'number_of_beds', 'days_since_registration', 'days_since_inspection',
            'regulatedActivitiesCount', 'specialismsCount', 'serviceTypesCount',
            'serviceUserGroupsCount', 'overall_rating_score', 'domain_risk_score',
            'inspection_frequency', 'historical_rating_changes'
        ]
        
        categorical_features = ['region', 'provider_type']
        
        binary_features = [
            'overdue_inspection', 'safe_at_risk', 'effective_at_risk',
            'caring_at_risk', 'responsive_at_risk', 'well_led_at_risk',
            'inherited_rating'
        ]
        
        # Handle missing values
        df[numerical_features] = df[numerical_features].fillna(0)
        df[categorical_features] = df[categorical_features].fillna('unknown')
        df[binary_features] = df[binary_features].fillna(0)
        
        # One-hot encode categorical features
        df_encoded = pd.get_dummies(df, columns=categorical_features, prefix_sep='_')
        
        # Get all feature columns
        feature_cols = numerical_features + binary_features
        feature_cols += [col for col in df_encoded.columns if any(cat in col for cat in categorical_features)]
        
        # Ensure all columns exist
        feature_cols = [col for col in feature_cols if col in df_encoded.columns]
        
        X = df_encoded[feature_cols].copy()
        
        # Scale numerical features if requested
        if scale_features:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        
        y = df_encoded['at_risk_label'].astype(int)
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        logger.info(f"Prepared {X.shape[0]} samples with {X.shape[1]} features")
        
        return X, y, feature_cols
        
    def train_models(self, 
                    X_train: pd.DataFrame, 
                    y_train: pd.Series, 
                    X_val: pd.DataFrame, 
                    y_val: pd.Series,
                    use_cross_validation: bool = True) -> Dict[str, Dict[str, Any]]:
        """Train multiple models and evaluate performance.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_cross_validation: Whether to use CV for hyperparameter tuning
            
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training models...")
        
        # Initialize models with optimized hyperparameters
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42,
            verbosity=-1
        )
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Train and evaluate each model
        results = {}
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            
            # Train model
            if name == 'xgboost':
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=50,
                    verbose=False
                )
            else:
                model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            
            # Calculate metrics
            results[name] = self._calculate_metrics(y_val, y_pred, y_proba)
            
            # Cross-validation if requested
            if use_cross_validation:
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                    scoring='roc_auc'
                )
                results[name]['cv_auc_mean'] = cv_scores.mean()
                results[name]['cv_auc_std'] = cv_scores.std()
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(20)
                results[name]['top_features'] = importance_df.to_dict('records')
            
            logger.info(f"{name} - AUC: {results[name]['auc_score']:.4f}")
        
        self.metrics = results
        return results
        
    def _calculate_metrics(self, 
                          y_true: pd.Series, 
                          y_pred: np.ndarray, 
                          y_proba: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive metrics for model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            
        Returns:
            Dictionary of metrics
        """
        # Basic metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        auc_score = roc_auc_score(y_true, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Find optimal threshold for F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        return {
            'auc_score': auc_score,
            'accuracy': report['accuracy'],
            'at_risk_precision': report['1']['precision'],
            'at_risk_recall': report['1']['recall'],
            'at_risk_f1': report['1']['f1-score'],
            'not_at_risk_precision': report['0']['precision'],
            'not_at_risk_recall': report['0']['recall'],
            'confusion_matrix': cm.tolist(),
            'optimal_threshold': optimal_threshold,
            'classification_report': report
        }
        
    def create_ensemble_predictor(self) -> callable:
        """Create ensemble model that combines all trained models.
        
        Returns:
            Function that makes ensemble predictions
        """
        logger.info("Creating ensemble predictor...")
        
        def ensemble_predict(X: pd.DataFrame, return_proba: bool = True) -> np.ndarray:
            """Make ensemble predictions.
            
            Args:
                X: Features to predict
                return_proba: Whether to return probabilities
                
            Returns:
                Predictions or prediction probabilities
            """
            predictions = []
            
            for name, model in self.models.items():
                if return_proba:
                    pred_proba = model.predict_proba(X)[:, 1]
                    predictions.append(pred_proba)
                else:
                    pred = model.predict(X)
                    predictions.append(pred)
            
            # Average predictions (could use weighted average based on performance)
            if return_proba:
                ensemble_proba = np.mean(predictions, axis=0)
                return ensemble_proba
            else:
                # Majority vote for binary predictions
                ensemble_pred = np.round(np.mean(predictions, axis=0))
                return ensemble_pred.astype(int)
            
        return ensemble_predict
        
    def save_models(self, 
                   bucket_name: str = "machine-learning-exp-467008-cqc-ml-artifacts",
                   model_prefix: str = "models/proactive") -> None:
        """Save models and metadata to Google Cloud Storage.
        
        Args:
            bucket_name: GCS bucket name
            model_prefix: Prefix for model paths in bucket
        """
        logger.info(f"Saving models to gs://{bucket_name}/{model_prefix}")
        
        # Create model package
        model_package = {
            'models': self.models,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'ensemble_predict': self.create_ensemble_predictor(),
            'metrics': self.metrics,
            'training_timestamp': datetime.now().isoformat(),
            'model_version': '1.0.0'
        }
        
        # Save main model package
        bucket = self.storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{model_prefix}/model_package.pkl")
        
        with blob.open('wb') as f:
            pickle.dump(model_package, f)
            
        # Save individual models for flexibility
        for name, model in self.models.items():
            individual_blob = bucket.blob(f"{model_prefix}/{name}_model.pkl")
            with individual_blob.open('wb') as f:
                pickle.dump({
                    'model': model,
                    'feature_columns': self.feature_columns,
                    'scaler': self.scaler,
                    'metrics': self.metrics.get(name, {})
                }, f)
        
        # Save metrics report as JSON
        metrics_blob = bucket.blob(f"{model_prefix}/metrics_report.json")
        with metrics_blob.open('w') as f:
            json.dump(self.metrics, f, indent=2)
            
        # Save feature importance summary
        importance_data = {}
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importance_data[name] = pd.DataFrame({
                    'feature': self.feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False).head(30).to_dict('records')
        
        importance_blob = bucket.blob(f"{model_prefix}/feature_importance.json")
        with importance_blob.open('w') as f:
            json.dump(importance_data, f, indent=2)
            
        logger.info("Models saved successfully")
        
    def generate_report(self) -> str:
        """Generate a comprehensive training report.
        
        Returns:
            Formatted report string
        """
        report_lines = [
            "=" * 60,
            "PROACTIVE MODEL TRAINING REPORT",
            "=" * 60,
            f"Training completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "MODEL PERFORMANCE SUMMARY",
            "-" * 40
        ]
        
        # Sort models by AUC score
        sorted_models = sorted(
            self.metrics.items(), 
            key=lambda x: x[1]['auc_score'], 
            reverse=True
        )
        
        for model_name, metrics in sorted_models:
            report_lines.extend([
                f"\n{model_name.upper()}:",
                f"  AUC Score: {metrics['auc_score']:.4f}",
                f"  Accuracy: {metrics['accuracy']:.4f}",
                f"  At-Risk Class Performance:",
                f"    - Precision: {metrics['at_risk_precision']:.4f}",
                f"    - Recall: {metrics['at_risk_recall']:.4f}",
                f"    - F1-Score: {metrics['at_risk_f1']:.4f}",
                f"  Optimal Threshold: {metrics['optimal_threshold']:.3f}"
            ])
            
            if 'cv_auc_mean' in metrics:
                report_lines.append(
                    f"  Cross-Validation AUC: {metrics['cv_auc_mean']:.4f} "
                    f"(± {metrics['cv_auc_std']:.4f})"
                )
        
        report_lines.extend([
            "",
            "ENSEMBLE MODEL",
            "-" * 40,
            "The ensemble model combines predictions from all individual models",
            "using average probability aggregation.",
            "",
            "=" * 60
        ])
        
        return "\n".join(report_lines)


def main():
    """Main training pipeline."""
    # Initialize trainer
    trainer = ProactiveModelTrainer()
    
    try:
        # Load data
        logger.info("Starting proactive model training pipeline...")
        df = trainer.load_training_data()
        
        # Prepare features
        X, y, feature_cols = trainer.prepare_features(df)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        logger.info(f"Training set size: {len(X_train)}")
        logger.info(f"Validation set size: {len(X_val)}")
        
        # Train models
        results = trainer.train_models(X_train, y_train, X_val, y_val)
        
        # Generate and print report
        report = trainer.generate_report()
        print(report)
        
        # Save models
        trainer.save_models()
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()