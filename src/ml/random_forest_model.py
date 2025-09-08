#!/usr/bin/env python3
"""
Random Forest model for CQC rating prediction with emphasis on feature importance.
Production-ready training script designed for Vertex AI.
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Google Cloud imports
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.exceptions import NotFound

# ML imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
from sklearn.inspection import permutation_importance
import optuna

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RandomForestCQCPredictor:
    """Random Forest model for CQC rating prediction with enhanced feature analysis."""
    
    RATING_MAP = {
        'Outstanding': 4,
        'Good': 3,
        'Requires improvement': 2,
        'Inadequate': 1,
        'No published rating': 0,
        None: 0
    }
    
    RATING_LABELS = ['No Rating', 'Inadequate', 'Requires Improvement', 'Good', 'Outstanding']
    
    def __init__(self, project_id: str, region: str = 'us-central1'):
        """Initialize Random Forest predictor."""
        self.project_id = project_id
        self.region = region
        self.dataset_id = 'cqc_dataset'
        
        # Initialize clients
        self.bigquery_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Model artifacts
        self.model = None
        self.feature_names = None
        self.label_encoders = {}
        
        # Experiment tracking
        self.experiment_name = "cqc-rating-random-forest"
        self.run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Initialized RandomForestCQCPredictor for project {project_id}")
    
    def load_data_from_bigquery(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load engineered features from BigQuery."""
        query = f"""
        SELECT *
        FROM `{self.project_id}.{self.dataset_id}.ml_features`
        WHERE target IS NOT NULL
        AND target >= 0
        ORDER BY created_timestamp DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info("Loading data from BigQuery...")
        df = self.bigquery_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} samples with {len(df.columns)} columns")
        
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """Prepare data for training."""
        logger.info("Preparing data for training...")
        
        # Remove non-feature columns
        exclude_cols = [
            'locationId', 'target', 'feature_version', 'feature_date', 
            'created_timestamp', 'overall_rating'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        X = df[feature_cols].copy()
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)  # For any remaining non-numeric columns
        
        # Prepare target
        y = df['target'].copy()
        
        logger.info(f"Prepared {len(feature_cols)} features for {len(X)} samples")
        logger.info(f"Target distribution:\n{y.value_counts().sort_index()}")
        
        return X, y, feature_cols
    
    def create_temporal_split(self, df: pd.DataFrame, 
                            test_size: float = 0.2, 
                            val_size: float = 0.1) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Create temporal train/validation/test split."""
        logger.info("Creating temporal data split...")
        
        # Sort by creation timestamp
        df_sorted = df.sort_values('created_timestamp')
        
        # Calculate split indices
        n_samples = len(df_sorted)
        train_end = int(n_samples * (1 - test_size - val_size))
        val_end = int(n_samples * (1 - test_size))
        
        train_df = df_sorted.iloc[:train_end]
        val_df = df_sorted.iloc[train_end:val_end]
        test_df = df_sorted.iloc[val_end:]
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def objective(self, trial: optuna.Trial, X_train: pd.DataFrame, y_train: pd.Series,
                  X_val: pd.DataFrame, y_val: pd.Series) -> float:
        """Optuna objective function for hyperparameter tuning."""
        
        # Suggest hyperparameters
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1),
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        }
        
        # If bootstrap is False, need to adjust max_samples
        if not params['bootstrap']:
            params['max_samples'] = None
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predict and calculate loss
        y_pred_proba = model.predict_proba(X_val)
        loss = log_loss(y_val, y_pred_proba)
        
        return loss
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='minimize',
            study_name=f"random-forest-{self.run_name}",
            storage=f"sqlite:///optuna_study_rf_{self.run_name}.db"
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=3600  # 1 hour timeout
        )
        
        best_params = study.best_params
        best_params.update({
            'random_state': 42,
            'n_jobs': -1,
            'class_weight': 'balanced'
        })
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return best_params
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: Optional[Dict[str, Any]] = None) -> RandomForestClassifier:
        """Train Random Forest model."""
        logger.info("Training Random Forest model...")
        
        if params is None:
            params = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'bootstrap': True,
                'random_state': 42,
                'n_jobs': -1,
                'class_weight': 'balanced'
            }
        
        # Create Random Forest classifier
        model = RandomForestClassifier(**params)
        
        # Train model
        model.fit(X_train, y_train)
        
        self.model = model
        self.feature_names = X_train.columns.tolist()
        
        logger.info("Model training completed")
        return model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        logger.info("Evaluating model performance...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision_macro': precision_score(y_test, y_pred, average='macro'),
            'recall_macro': recall_score(y_test, y_pred, average='macro'),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'log_loss': log_loss(y_test, y_pred_proba)
        }
        
        # Multi-class ROC AUC
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except ValueError:
            metrics['roc_auc_ovr'] = 0.0
        
        # Per-class metrics
        class_report = classification_report(y_test, y_pred, output_dict=True)
        for i, label in enumerate(self.RATING_LABELS):
            if str(i) in class_report:
                metrics[f'{label}_precision'] = class_report[str(i)]['precision']
                metrics[f'{label}_recall'] = class_report[str(i)]['recall']
                metrics[f'{label}_f1'] = class_report[str(i)]['f1-score']
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        logger.info("Model evaluation metrics:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return metrics, cm
    
    def analyze_feature_importance(self, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
        """Comprehensive feature importance analysis."""
        logger.info("Analyzing feature importance...")
        
        # 1. Built-in feature importance (Gini impurity based)
        gini_importance = self.model.feature_importances_
        
        # 2. Permutation importance
        perm_importance = permutation_importance(
            self.model, X_test, y_test, 
            n_repeats=10, random_state=42, n_jobs=-1
        )
        perm_importance_mean = perm_importance.importances_mean
        perm_importance_std = perm_importance.importances_std
        
        # 3. SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test.sample(min(1000, len(X_test))))
        
        # For multi-class, calculate mean absolute SHAP values
        if isinstance(shap_values, list):
            shap_importance = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
        else:
            shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Create comprehensive feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'gini_importance': gini_importance,
            'permutation_importance': perm_importance_mean,
            'permutation_importance_std': perm_importance_std,
            'shap_importance': shap_importance
        })
        
        # Calculate ranking for each method
        feature_importance['gini_rank'] = feature_importance['gini_importance'].rank(ascending=False)
        feature_importance['perm_rank'] = feature_importance['permutation_importance'].rank(ascending=False)
        feature_importance['shap_rank'] = feature_importance['shap_importance'].rank(ascending=False)
        
        # Average rank
        feature_importance['avg_rank'] = (
            feature_importance['gini_rank'] + 
            feature_importance['perm_rank'] + 
            feature_importance['shap_rank']
        ) / 3
        
        # Sort by average rank
        feature_importance = feature_importance.sort_values('avg_rank')
        
        logger.info(f"Top 10 features by average ranking:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: Gini={row['gini_importance']:.4f}, "
                       f"Perm={row['permutation_importance']:.4f}, "
                       f"SHAP={row['shap_importance']:.4f}")
        
        return feature_importance
    
    def create_feature_importance_visualizations(self, feature_importance: pd.DataFrame, 
                                               save_path: str = '/tmp/plots') -> None:
        """Create detailed feature importance visualizations."""
        logger.info("Creating feature importance visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Comparison of importance methods
        plt.figure(figsize=(15, 10))
        top_features = feature_importance.head(20)
        
        x = np.arange(len(top_features))
        width = 0.25
        
        plt.bar(x - width, top_features['gini_importance'], width, label='Gini Importance', alpha=0.8)
        plt.bar(x, top_features['permutation_importance'], width, label='Permutation Importance', alpha=0.8)
        plt.bar(x + width, top_features['shap_importance'], width, label='SHAP Importance', alpha=0.8)
        
        plt.xlabel('Features')
        plt.ylabel('Importance Score')
        plt.title('Feature Importance Comparison - Top 20 Features')
        plt.xticks(x, top_features['feature'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Permutation importance with error bars
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        
        plt.barh(range(len(top_features)), top_features['permutation_importance'],
                xerr=top_features['permutation_importance_std'], capsize=5)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Permutation Importance')
        plt.title('Permutation Feature Importance with Standard Deviation')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{save_path}/permutation_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance ranking correlation
        plt.figure(figsize=(10, 8))
        correlation_data = feature_importance[['gini_importance', 'permutation_importance', 'shap_importance']]
        correlation_matrix = correlation_data.corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={'shrink': 0.8})
        plt.title('Feature Importance Method Correlation')
        plt.tight_layout()
        plt.savefig(f'{save_path}/importance_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature categories analysis
        self.create_feature_category_analysis(feature_importance, save_path)
        
        logger.info(f"Feature importance visualizations saved to {save_path}")
    
    def create_feature_category_analysis(self, feature_importance: pd.DataFrame, save_path: str):
        """Analyze feature importance by categories."""
        
        # Define feature categories based on naming patterns
        def categorize_feature(feature_name):
            if any(x in feature_name for x in ['region', 'la_', 'urban']):
                return 'Geographic'
            elif any(x in feature_name for x in ['provider', 'location_count']):
                return 'Provider'
            elif any(x in feature_name for x in ['bed', 'capacity', 'size']):
                return 'Capacity'
            elif any(x in feature_name for x in ['rating', 'domain']):
                return 'Rating History'
            elif any(x in feature_name for x in ['year', 'days', 'inspection', 'time']):
                return 'Temporal'
            elif any(x in feature_name for x in ['specialism', 'nursing', 'care', 'service']):
                return 'Services'
            elif any(x in feature_name for x in ['_encoded']):
                return 'Encoded Categorical'
            else:
                return 'Other'
        
        feature_importance['category'] = feature_importance['feature'].apply(categorize_feature)
        
        # Aggregate importance by category
        category_importance = feature_importance.groupby('category').agg({
            'gini_importance': 'mean',
            'permutation_importance': 'mean',
            'shap_importance': 'mean',
            'feature': 'count'
        }).rename(columns={'feature': 'feature_count'})
        
        # Create category visualization
        plt.figure(figsize=(12, 8))
        categories = category_importance.index
        x = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x - width, category_importance['gini_importance'], width, 
               label='Gini Importance', alpha=0.8)
        plt.bar(x, category_importance['permutation_importance'], width, 
               label='Permutation Importance', alpha=0.8)
        plt.bar(x + width, category_importance['shap_importance'], width, 
               label='SHAP Importance', alpha=0.8)
        
        plt.xlabel('Feature Categories')
        plt.ylabel('Average Importance Score')
        plt.title('Average Feature Importance by Category')
        plt.xticks(x, categories, rotation=45, ha='right')
        plt.legend()
        
        # Add feature count annotations
        for i, count in enumerate(category_importance['feature_count']):
            plt.text(i, max(category_importance.iloc[i, :3]) + 0.002, 
                    f'n={count}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/category_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_visualizations(self, metrics: Dict[str, float], cm: np.ndarray,
                            feature_importance: pd.DataFrame, 
                            save_path: str = '/tmp/plots') -> None:
        """Create and save visualization plots."""
        logger.info("Creating visualizations...")
        
        os.makedirs(save_path, exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.RATING_LABELS,
                   yticklabels=self.RATING_LABELS)
        plt.title('Confusion Matrix - Random Forest Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Model Performance Metrics
        plt.figure(figsize=(10, 6))
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange', 'purple'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics - Random Forest')
        plt.xticks(rotation=45)
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Feature importance visualizations
        self.create_feature_importance_visualizations(feature_importance, save_path)
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def save_model_artifacts(self, bucket_name: str, model_version: str = 'v1') -> str:
        """Save model artifacts to GCS."""
        logger.info(f"Saving model artifacts to gs://{bucket_name}/models/random_forest/{model_version}/")
        
        bucket = self.storage_client.bucket(bucket_name)
        base_path = f"models/random_forest/{model_version}"
        
        # Save model
        model_path = f'/tmp/random_forest_model.pkl'
        joblib.dump(self.model, model_path)
        
        blob = bucket.blob(f"{base_path}/model.pkl")
        blob.upload_from_filename(model_path)
        
        # Save feature names
        feature_path = f'/tmp/feature_names.json'
        with open(feature_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        blob = bucket.blob(f"{base_path}/feature_names.json")
        blob.upload_from_filename(feature_path)
        
        # Save model metadata
        metadata = {
            'model_type': 'random_forest',
            'version': model_version,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'project_id': self.project_id,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth
        }
        
        metadata_path = f'/tmp/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        blob = bucket.blob(f"{base_path}/metadata.json")
        blob.upload_from_filename(metadata_path)
        
        model_uri = f"gs://{bucket_name}/{base_path}/"
        logger.info(f"Model artifacts saved to {model_uri}")
        
        return model_uri
    
    def compare_with_xgboost(self, xgboost_results_path: str = None) -> Dict[str, Any]:
        """Compare Random Forest performance with XGBoost if available."""
        logger.info("Comparing with XGBoost model...")
        
        comparison = {
            'random_forest': {
                'model_type': 'Random Forest',
                'interpretability': 'High',
                'training_speed': 'Fast',
                'feature_importance_methods': ['Gini', 'Permutation', 'SHAP']
            }
        }
        
        if xgboost_results_path and os.path.exists(xgboost_results_path):
            try:
                with open(xgboost_results_path, 'r') as f:
                    xgb_results = json.load(f)
                    
                comparison['xgboost'] = {
                    'model_type': 'XGBoost',
                    'interpretability': 'Medium',
                    'training_speed': 'Medium',
                    'feature_importance_methods': ['Gain', 'SHAP']
                }
                
                # Performance comparison
                comparison['performance_comparison'] = {
                    'metric_comparison': 'Random Forest vs XGBoost performance analysis',
                    'rf_advantages': [
                        'Better interpretability',
                        'Less prone to overfitting',
                        'Handles missing values naturally',
                        'Multiple importance methods'
                    ],
                    'xgb_advantages': [
                        'Often higher performance',
                        'Better handling of imbalanced data',
                        'Advanced regularization',
                        'Gradient-based optimization'
                    ]
                }
                
            except Exception as e:
                logger.warning(f"Could not load XGBoost results: {e}")
        
        return comparison
    
    def run_training_pipeline(self, tune_hyperparameters: bool = True, 
                            deploy_model: bool = False) -> Dict[str, Any]:
        """Run the complete Random Forest training pipeline."""
        logger.info("Starting Random Forest training pipeline...")
        
        try:
            # 1. Load and prepare data
            df = self.load_data_from_bigquery()
            train_df, val_df, test_df = self.create_temporal_split(df)
            
            X_train, y_train, feature_names = self.prepare_data(train_df)
            X_val, y_val, _ = self.prepare_data(val_df)
            X_test, y_test, _ = self.prepare_data(test_df)
            
            # 2. Hyperparameter tuning (optional)
            if tune_hyperparameters:
                best_params = self.tune_hyperparameters(X_train, y_train, X_val, y_val)
            else:
                best_params = None
            
            # 3. Train model
            model = self.train_model(X_train, y_train, X_val, y_val, best_params)
            
            # 4. Evaluate model
            metrics, cm = self.evaluate_model(X_test, y_test)
            
            # 5. Comprehensive feature importance analysis
            feature_importance = self.analyze_feature_importance(X_test, y_test)
            
            # 6. Create visualizations
            self.create_visualizations(metrics, cm, feature_importance)
            
            # 7. Save model artifacts
            bucket_name = f"{self.project_id}-cqc-models"
            model_uri = self.save_model_artifacts(bucket_name)
            
            # 8. Model comparison analysis
            comparison = self.compare_with_xgboost()
            
            # 9. Prepare results
            results = {
                'metrics': metrics,
                'model_uri': model_uri,
                'feature_importance': feature_importance.to_dict('records'),
                'training_samples': len(X_train),
                'validation_samples': len(X_val),
                'test_samples': len(X_test),
                'feature_count': len(feature_names),
                'model_params': best_params if tune_hyperparameters else model.get_params(),
                'model_comparison': comparison,
                'feature_analysis': {
                    'top_10_features': feature_importance.head(10)[['feature', 'avg_rank']].to_dict('records'),
                    'feature_categories': feature_importance['category'].value_counts().to_dict()
                }
            }
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Final test accuracy: {metrics['accuracy']:.4f}")
            logger.info(f"Top 3 most important features:")
            for i, row in feature_importance.head(3).iterrows():
                logger.info(f"  {i+1}. {row['feature']} (avg rank: {row['avg_rank']:.1f})")
            
            return results
            
        except Exception as e:
            logger.error(f"Training pipeline failed: {e}")
            raise


def main():
    """Main function for Vertex AI custom training."""
    # Get environment variables
    project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
    region = os.environ.get('GCP_REGION', 'us-central1')
    
    # Training configuration
    tune_hyperparameters = os.environ.get('TUNE_HYPERPARAMETERS', 'true').lower() == 'true'
    deploy_model = os.environ.get('DEPLOY_MODEL', 'false').lower() == 'true'
    
    # Initialize and run training
    trainer = RandomForestCQCPredictor(project_id, region)
    results = trainer.run_training_pipeline(
        tune_hyperparameters=tune_hyperparameters,
        deploy_model=deploy_model
    )
    
    # Print results
    print("\n" + "="*80)
    print("RANDOM FOREST TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['metrics']['f1_macro']:.4f}")
    print(f"ROC AUC (OvR): {results['metrics']['roc_auc_ovr']:.4f}")
    print(f"Model URI: {results['model_uri']}")
    print("\nTop 5 Most Important Features:")
    for i, feature in enumerate(results['feature_analysis']['top_10_features'][:5], 1):
        print(f"  {i}. {feature['feature']} (rank: {feature['avg_rank']:.1f})")
    print("="*80)


if __name__ == "__main__":
    main()