#!/usr/bin/env python3
"""
XGBoost baseline model for CQC rating prediction.
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
import xgboost as xgb
from sklearn.model_selection import train_test_split, TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import LabelEncoder
import optuna
from optuna.integration import XGBoostPruningCallback

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

class XGBoostCQCPredictor:
    """XGBoost model for CQC rating prediction."""
    
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
        """Initialize XGBoost predictor."""
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
        self.experiment_name = "cqc-rating-xgboost"
        self.run_name = f"run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Initialized XGBoostCQCPredictor for project {project_id}")
    
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
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'random_state': 42,
            
            # Tuned parameters
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'gamma': trial.suggest_float('gamma', 0.0, 2.0)
        }
        
        # Train model
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval = xgb.DMatrix(X_val, label=y_val)
        
        # Pruning callback
        pruning_callback = XGBoostPruningCallback(trial, 'validation_0-mlogloss')
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_estimators'],
            evals=[(dval, 'validation_0')],
            early_stopping_rounds=20,
            callbacks=[pruning_callback],
            verbose_eval=False
        )
        
        # Predict and calculate loss
        y_pred_proba = model.predict(dval)
        loss = log_loss(y_val, y_pred_proba)
        
        return loss
    
    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series,
                           X_val: pd.DataFrame, y_val: pd.Series,
                           n_trials: int = 50) -> Dict[str, Any]:
        """Tune hyperparameters using Optuna."""
        logger.info(f"Starting hyperparameter tuning with {n_trials} trials...")
        
        study = optuna.create_study(
            direction='minimize',
            study_name=f"xgboost-{self.run_name}",
            storage=f"sqlite:///optuna_study_{self.run_name}.db"
        )
        
        study.optimize(
            lambda trial: self.objective(trial, X_train, y_train, X_val, y_val),
            n_trials=n_trials,
            timeout=3600  # 1 hour timeout
        )
        
        best_params = study.best_params
        best_params.update({
            'objective': 'multi:softprob',
            'num_class': 5,
            'eval_metric': 'mlogloss',
            'verbosity': 0,
            'random_state': 42
        })
        
        logger.info(f"Best parameters: {best_params}")
        logger.info(f"Best score: {study.best_value:.4f}")
        
        return best_params
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   X_val: pd.DataFrame, y_val: pd.Series,
                   params: Optional[Dict[str, Any]] = None) -> xgb.XGBClassifier:
        """Train XGBoost model."""
        logger.info("Training XGBoost model...")
        
        if params is None:
            params = {
                'objective': 'multi:softprob',
                'num_class': 5,
                'max_depth': 6,
                'learning_rate': 0.1,
                'n_estimators': 200,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'eval_metric': 'mlogloss',
                'random_state': 42,
                'verbosity': 0
            }
        
        # Create XGBoost classifier
        model = xgb.XGBClassifier(**params)
        
        # Train with validation
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        
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
    
    def analyze_feature_importance(self, X_test: pd.DataFrame) -> pd.DataFrame:
        """Analyze feature importance using multiple methods."""
        logger.info("Analyzing feature importance...")
        
        # XGBoost built-in feature importance
        importance_gain = self.model.feature_importances_
        
        # SHAP values
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(X_test.sample(min(1000, len(X_test))))
        
        # For multi-class, take mean absolute SHAP values across classes
        if len(shap_values.shape) == 3:
            shap_importance = np.mean(np.abs(shap_values), axis=(0, 2))
        else:
            shap_importance = np.mean(np.abs(shap_values), axis=0)
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'xgb_importance': importance_gain,
            'shap_importance': shap_importance
        })
        
        feature_importance = feature_importance.sort_values('xgb_importance', ascending=False)
        
        logger.info(f"Top 10 features by XGB importance:")
        for _, row in feature_importance.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['xgb_importance']:.4f}")
        
        return feature_importance
    
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
        plt.title('Confusion Matrix - XGBoost Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'{save_path}/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature Importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance.head(20)
        plt.barh(range(len(top_features)), top_features['xgb_importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('XGBoost Feature Importance')
        plt.title('Top 20 Feature Importance - XGBoost Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'{save_path}/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Model Performance Metrics
        plt.figure(figsize=(10, 6))
        metric_names = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc_ovr']
        metric_values = [metrics.get(m, 0) for m in metric_names]
        
        plt.bar(metric_names, metric_values, color=['blue', 'green', 'red', 'orange', 'purple'])
        plt.ylim(0, 1)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics - XGBoost')
        plt.xticks(rotation=45)
        for i, v in enumerate(metric_values):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig(f'{save_path}/performance_metrics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {save_path}")
    
    def save_model_artifacts(self, bucket_name: str, model_version: str = 'v1') -> str:
        """Save model artifacts to GCS."""
        logger.info(f"Saving model artifacts to gs://{bucket_name}/models/xgboost/{model_version}/")
        
        bucket = self.storage_client.bucket(bucket_name)
        base_path = f"models/xgboost/{model_version}"
        
        # Save model
        model_path = f'/tmp/xgboost_model.pkl'
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
            'model_type': 'xgboost',
            'version': model_version,
            'training_date': datetime.now().isoformat(),
            'feature_count': len(self.feature_names),
            'project_id': self.project_id
        }
        
        metadata_path = f'/tmp/model_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        blob = bucket.blob(f"{base_path}/metadata.json")
        blob.upload_from_filename(metadata_path)
        
        model_uri = f"gs://{bucket_name}/{base_path}/"
        logger.info(f"Model artifacts saved to {model_uri}")
        
        return model_uri
    
    def register_model_vertex_ai(self, model_uri: str, model_display_name: str) -> aiplatform.Model:
        """Register model with Vertex AI Model Registry."""
        logger.info(f"Registering model with Vertex AI: {model_display_name}")
        
        # Create serving container spec
        serving_container_spec = {
            "image_uri": "gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest",
            "predict_route": "/predict",
            "health_route": "/health"
        }
        
        # Upload model
        model = aiplatform.Model.upload(
            display_name=model_display_name,
            artifact_uri=model_uri,
            serving_container_image_uri=serving_container_spec["image_uri"],
            description=f"XGBoost model for CQC rating prediction - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            labels={
                "model_type": "xgboost",
                "use_case": "cqc_rating_prediction",
                "framework": "scikit-learn"
            }
        )
        
        logger.info(f"Model registered with resource name: {model.resource_name}")
        return model
    
    def deploy_to_endpoint(self, model: aiplatform.Model, 
                          endpoint_display_name: str = "cqc-rating-xgboost-endpoint") -> aiplatform.Endpoint:
        """Deploy model to Vertex AI endpoint."""
        logger.info(f"Deploying model to endpoint: {endpoint_display_name}")
        
        # Check if endpoint exists
        try:
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_display_name}"'
            )
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Using existing endpoint: {endpoint.display_name}")
            else:
                # Create new endpoint
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_display_name,
                    description="Endpoint for CQC rating prediction using XGBoost"
                )
                logger.info(f"Created new endpoint: {endpoint.display_name}")
        except Exception as e:
            logger.error(f"Error managing endpoint: {e}")
            # Create new endpoint
            endpoint = aiplatform.Endpoint.create(
                display_name=endpoint_display_name,
                description="Endpoint for CQC rating prediction using XGBoost"
            )
        
        # Deploy model to endpoint
        deployed_model = model.deploy(
            endpoint=endpoint,
            deployed_model_display_name=f"xgboost-{self.run_name}",
            machine_type="n1-standard-4",
            min_replica_count=1,
            max_replica_count=3,
            traffic_percentage=100,
            sync=True
        )
        
        logger.info(f"Model deployed to endpoint: {endpoint.resource_name}")
        return endpoint
    
    def run_training_pipeline(self, tune_hyperparameters: bool = True, 
                            deploy_model: bool = True) -> Dict[str, Any]:
        """Run the complete training pipeline."""
        logger.info("Starting XGBoost training pipeline...")
        
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
            
            # 5. Feature importance analysis
            feature_importance = self.analyze_feature_importance(X_test)
            
            # 6. Create visualizations
            self.create_visualizations(metrics, cm, feature_importance)
            
            # 7. Save model artifacts
            bucket_name = f"{self.project_id}-cqc-models"
            model_uri = self.save_model_artifacts(bucket_name)
            
            # 8. Register and deploy model (optional)
            if deploy_model:
                model_display_name = f"cqc-rating-xgboost-{self.run_name}"
                vertex_model = self.register_model_vertex_ai(model_uri, model_display_name)
                endpoint = self.deploy_to_endpoint(vertex_model)
            else:
                vertex_model = None
                endpoint = None
            
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
                'vertex_model_name': vertex_model.resource_name if vertex_model else None,
                'endpoint_name': endpoint.resource_name if endpoint else None
            }
            
            logger.info("Training pipeline completed successfully!")
            logger.info(f"Final test accuracy: {metrics['accuracy']:.4f}")
            
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
    deploy_model = os.environ.get('DEPLOY_MODEL', 'true').lower() == 'true'
    
    # Initialize and run training
    trainer = XGBoostCQCPredictor(project_id, region)
    results = trainer.run_training_pipeline(
        tune_hyperparameters=tune_hyperparameters,
        deploy_model=deploy_model
    )
    
    # Print results
    print("\n" + "="*80)
    print("TRAINING RESULTS SUMMARY")
    print("="*80)
    print(f"Test Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"F1 Score (Macro): {results['metrics']['f1_macro']:.4f}")
    print(f"ROC AUC (OvR): {results['metrics']['roc_auc_ovr']:.4f}")
    print(f"Model URI: {results['model_uri']}")
    if results['endpoint_name']:
        print(f"Endpoint: {results['endpoint_name']}")
    print("="*80)


if __name__ == "__main__":
    main()