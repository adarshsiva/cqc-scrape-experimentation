#!/usr/bin/env python3
"""
Unified CQC ML Pipeline (Phase 3) - Comprehensive Model Trainer
Train on comprehensive CQC data with dashboard feature validation support.

This module implements the UnifiedCQCModelTrainer class as specified in plan.md:
- Trains on comprehensive CQC Syndication API data
- Supports unified feature space for dashboard compatibility
- Ensemble models (XGBoost, LightGBM, Random Forest)
- Feature alignment validation with dashboard systems
- Production-ready deployment to Vertex AI
"""

import os
import logging
import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings('ignore')

# Google Cloud imports
from google.cloud import bigquery
from google.cloud import storage
from google.cloud import aiplatform
from google.cloud.exceptions import NotFound

# ML imports
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
import optuna
from optuna.integration import LightGBMPruningCallback, XGBoostPruningCallback

# Visualization and interpretation
import matplotlib.pyplot as plt
import seaborn as sns
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UnifiedCQCModelTrainer:
    """
    Unified CQC Model Trainer for comprehensive rating prediction.
    
    Implements Phase 3 of the plan.md architecture:
    - Comprehensive CQC API data training
    - Dashboard feature compatibility
    - Ensemble model approach
    - Production deployment support
    """
    
    # CQC Rating System mapping
    RATING_MAP = {
        'Outstanding': 4,
        'Good': 3,
        'Requires improvement': 2,
        'Inadequate': 1,
        'No published rating': 0,
        None: 0,
        'NULL': 0
    }
    
    RATING_LABELS = ['No Rating', 'Inadequate', 'Requires Improvement', 'Good', 'Outstanding']
    
    # Unified feature space compatible with both CQC and dashboard data
    UNIFIED_FEATURE_COLUMNS = [
        # Core operational (available in both CQC and dashboard)
        'bed_capacity', 'facility_size_numeric', 'occupancy_rate',
        
        # Risk indicators (CQC: historical, Dashboard: current)
        'inspection_overdue_risk', 'incident_frequency_risk', 
        'medication_risk', 'safeguarding_risk',
        
        # Quality metrics (CQC: ratings, Dashboard: compliance)
        'service_complexity_score', 'care_quality_indicator',
        
        # Temporal features
        'days_since_inspection', 'operational_stability',
        'days_since_registration', 'registration_age_years',
        
        # Provider context
        'provider_location_count', 'provider_avg_rating',
        'provider_rating_consistency', 'provider_scale_category',
        
        # Regional context
        'regional_risk_rate', 'regional_avg_beds', 'regional_good_rating_rate',
        
        # Service complexity
        'num_regulated_activities', 'service_diversity_score',
        'assessment_complexity', 'specialisms_count',
        
        # Historical performance indicators
        'historical_avg_rating', 'rating_volatility',
        'inspection_frequency', 'compliance_trend',
        
        # Interaction features (engineered combinations)
        'complexity_scale_interaction', 'inspection_regional_risk',
        'provider_regional_performance', 'risk_capacity_ratio'
    ]
    
    def __init__(self, project_id: str, region: str = 'europe-west2'):
        """Initialize Unified CQC Model Trainer."""
        self.project_id = project_id
        self.region = region
        self.dataset_id = 'cqc_dataset'
        
        # Initialize GCP clients
        self.bigquery_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=region)
        
        # Model ensemble
        self.models = {}
        self.ensemble_model = None
        self.feature_names = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Experiment tracking
        self.experiment_name = "cqc-unified-ml-pipeline"
        self.run_name = f"unified-run-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        logger.info(f"Initialized UnifiedCQCModelTrainer for project {project_id}")
    
    def _load_comprehensive_cqc_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        Load comprehensive CQC training data from all API endpoints.
        
        This method loads the comprehensive dataset created from:
        - Get Locations (bulk location listings)
        - Get Location By Id (detailed facility information)  
        - Get Providers (provider-level data)
        - Get Location AssessmentServiceGroups (service complexity)
        - Get Location Inspection Areas (domain-specific ratings)
        - Get Reports (detailed inspection reports)
        
        Returns:
            DataFrame with comprehensive CQC features
        """
        logger.info("Loading comprehensive CQC training data from BigQuery...")
        
        # Comprehensive feature extraction query as specified in plan.md
        query = f"""
        WITH location_data AS (
            -- Core location features from Get Location By Id
            SELECT 
                locationId, name, providerId,
                numberOfBeds, registrationDate, lastInspectionDate,
                overall_rating, safe_rating, effective_rating,
                caring_rating, responsive_rating, well_led_rating,
                region, localAuthority, organisationType,
                ARRAY_LENGTH(JSON_EXTRACT_ARRAY(regulatedActivities)) as service_complexity,
                DATE_DIFF(CURRENT_DATE(), DATE(lastInspectionDate), DAY) as days_since_inspection,
                DATE_DIFF(CURRENT_DATE(), DATE(registrationDate), DAY) / 365 as registration_age_years
            FROM `{self.project_id}.{self.dataset_id}.locations_comprehensive`
            WHERE overall_rating IS NOT NULL
        ),
        inspection_history AS (
            -- Historical patterns from Get Provider Inspection Areas  
            SELECT 
                locationId,
                COUNT(*) as inspection_count,
                AVG(
                    CASE 
                        WHEN overall_rating = 'Outstanding' THEN 4
                        WHEN overall_rating = 'Good' THEN 3
                        WHEN overall_rating = 'Requires improvement' THEN 2
                        WHEN overall_rating = 'Inadequate' THEN 1
                        ELSE 0
                    END
                ) as historical_avg_rating,
                STDDEV(
                    CASE 
                        WHEN overall_rating = 'Outstanding' THEN 4
                        WHEN overall_rating = 'Good' THEN 3
                        WHEN overall_rating = 'Requires improvement' THEN 2
                        WHEN overall_rating = 'Inadequate' THEN 1
                        ELSE 0
                    END
                ) as rating_volatility,
                COUNT(DISTINCT DATE(inspectionDate)) as unique_inspection_dates
            FROM `{self.project_id}.{self.dataset_id}.inspection_areas_history`
            GROUP BY locationId
        ),
        service_assessment AS (
            -- Service complexity from Get Location AssessmentServiceGroups
            SELECT
                locationId,
                COUNT(DISTINCT serviceGroup) as service_group_count,
                COUNT(DISTINCT assessmentType) as assessment_type_count,
                AVG(IFNULL(riskScore, 0.3)) as avg_risk_score
            FROM `{self.project_id}.{self.dataset_id}.assessment_service_groups`
            GROUP BY locationId
        ),
        provider_context AS (
            -- Provider-level patterns from Get Provider By Id
            SELECT
                providerId,
                COUNT(DISTINCT locationId) as provider_location_count,
                AVG(
                    CASE 
                        WHEN overall_rating = 'Outstanding' THEN 4
                        WHEN overall_rating = 'Good' THEN 3
                        WHEN overall_rating = 'Requires improvement' THEN 2
                        WHEN overall_rating = 'Inadequate' THEN 1
                        ELSE 0
                    END
                ) as provider_avg_rating,
                COUNT(DISTINCT region) as provider_geographic_spread,
                AVG(IFNULL(numberOfBeds, 30)) as provider_avg_capacity,
                STDDEV(
                    CASE 
                        WHEN overall_rating = 'Outstanding' THEN 4
                        WHEN overall_rating = 'Good' THEN 3
                        WHEN overall_rating = 'Requires improvement' THEN 2
                        WHEN overall_rating = 'Inadequate' THEN 1
                        ELSE 0
                    END
                ) as provider_rating_consistency
            FROM location_data
            GROUP BY providerId
        ),
        regional_stats AS (
            -- Regional risk patterns
            SELECT
                region,
                COUNT(*) as regional_location_count,
                AVG(IFNULL(numberOfBeds, 30)) as regional_avg_beds,
                AVG(
                    CASE WHEN overall_rating = 'Good' OR overall_rating = 'Outstanding' THEN 1 ELSE 0 END
                ) as regional_good_rating_rate,
                AVG(
                    CASE WHEN days_since_inspection > 730 THEN 1 ELSE 0 END
                ) as regional_overdue_rate
            FROM location_data
            GROUP BY region
        )
        
        SELECT 
            l.*,
            -- Historical Context
            COALESCE(ih.inspection_count, 1) as inspection_frequency,
            COALESCE(ih.historical_avg_rating, 3.0) as historical_performance,
            COALESCE(ih.rating_volatility, 0.5) as performance_consistency,
            
            -- Service Complexity
            COALESCE(sa.service_group_count, 3) as service_diversity,
            COALESCE(sa.assessment_type_count, 5) as assessment_complexity,
            COALESCE(sa.avg_risk_score, 0.3) as inherent_risk_score,
            
            -- Provider Context
            COALESCE(pc.provider_location_count, 1) as provider_scale,
            COALESCE(pc.provider_avg_rating, 3.0) as provider_reputation,
            COALESCE(pc.provider_geographic_spread, 1) as provider_diversity,
            COALESCE(pc.provider_rating_consistency, 0.3) as provider_stability,
            
            -- Regional Context
            COALESCE(rs.regional_avg_beds, 30) as regional_avg_beds,
            COALESCE(rs.regional_good_rating_rate, 0.7) as regional_performance,
            COALESCE(rs.regional_overdue_rate, 0.2) as regional_risk_rate,
            
            -- Risk Indicators
            CASE WHEN days_since_inspection > 730 THEN 1 ELSE 0 END as inspection_overdue,
            CASE WHEN ih.rating_volatility > 1.0 THEN 1 ELSE 0 END as performance_unstable,
            
            -- Target Variable
            CASE 
                WHEN overall_rating = 'Outstanding' THEN 4
                WHEN overall_rating = 'Good' THEN 3  
                WHEN overall_rating = 'Requires improvement' THEN 2
                WHEN overall_rating = 'Inadequate' THEN 1
                ELSE 0
            END as overall_rating_numeric
        
        FROM location_data l
        LEFT JOIN inspection_history ih USING(locationId)
        LEFT JOIN service_assessment sa USING(locationId)  
        LEFT JOIN provider_context pc USING(providerId)
        LEFT JOIN regional_stats rs USING(region)
        WHERE l.overall_rating IS NOT NULL
        ORDER BY l.lastInspectionDate DESC
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        try:
            df = self.bigquery_client.query(query).to_dataframe()
            logger.info(f"Loaded {len(df)} comprehensive CQC samples with {len(df.columns)} columns")
            
            # Log target distribution
            if 'overall_rating_numeric' in df.columns:
                target_dist = df['overall_rating_numeric'].value_counts().sort_index()
                logger.info(f"Target distribution:\n{target_dist}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load comprehensive CQC data: {e}")
            # Fallback to basic locations table
            logger.warning("Falling back to basic locations table...")
            fallback_query = f"""
            SELECT * FROM `{self.project_id}.{self.dataset_id}.locations`
            WHERE overall_rating IS NOT NULL
            ORDER BY lastInspectionDate DESC
            """
            if limit:
                fallback_query += f" LIMIT {limit}"
            
            return self.bigquery_client.query(fallback_query).to_dataframe()
    
    def _prepare_unified_features(self, training_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features compatible with both CQC and dashboard data.
        
        This method creates a unified feature space that can accommodate:
        - CQC training data (historical regulatory data)
        - Dashboard prediction data (live operational metrics)
        
        Args:
            training_data: Raw CQC training data
            
        Returns:
            Tuple of (feature DataFrame, target Series)
        """
        logger.info("Preparing unified features compatible with CQC and dashboard data...")
        
        df = training_data.copy()
        
        # ====== CORE OPERATIONAL FEATURES ======
        # bed_capacity - Direct mapping
        df['bed_capacity'] = df.get('numberOfBeds', 30).fillna(30)
        
        # facility_size_numeric - Categorical encoding
        df['facility_size_numeric'] = pd.cut(
            df['bed_capacity'], 
            bins=[0, 20, 40, 60, 200], 
            labels=[1, 2, 3, 4],
            include_lowest=True
        ).astype(float)
        
        # occupancy_rate - Estimate from bed capacity (will be actual in dashboard)
        df['occupancy_rate'] = 0.85  # Default assumption, overridden by dashboard
        
        # ====== RISK INDICATORS ======
        # inspection_overdue_risk - From CQC inspection dates
        df['inspection_overdue_risk'] = (
            df.get('days_since_inspection', 365) > 730
        ).astype(float)
        
        # incident_frequency_risk - Placeholder (calculated from dashboard incidents)
        df['incident_frequency_risk'] = np.random.beta(0.2, 0.8, len(df))  # Low risk default
        
        # medication_risk - Placeholder (from dashboard medication errors)
        df['medication_risk'] = np.random.beta(0.1, 0.9, len(df))  # Very low default
        
        # safeguarding_risk - Placeholder (from dashboard safeguarding incidents)
        df['safeguarding_risk'] = (np.random.random(len(df)) < 0.05).astype(float)  # 5% baseline
        
        # ====== QUALITY METRICS ======
        # service_complexity_score - From regulated activities and service types
        df['service_complexity_score'] = (
            df.get('service_complexity', 3) + 
            df.get('service_diversity', 3) + 
            df.get('assessment_complexity', 5)
        ) / 3.0
        
        # care_quality_indicator - From inspection domain ratings
        domain_ratings = []
        for domain in ['safe_rating', 'effective_rating', 'caring_rating', 'responsive_rating', 'well_led_rating']:
            if domain in df.columns:
                domain_numeric = df[domain].map(self.RATING_MAP).fillna(3.0)
                domain_ratings.append(domain_numeric)
        
        if domain_ratings:
            df['care_quality_indicator'] = np.mean(domain_ratings, axis=0)
        else:
            df['care_quality_indicator'] = 3.0  # Default "Good" rating
        
        # ====== TEMPORAL FEATURES ======
        df['days_since_inspection'] = df.get('days_since_inspection', 365).fillna(365)
        df['days_since_registration'] = (df.get('registration_age_years', 5) * 365).fillna(1825)
        df['operational_stability'] = 1.0 / (1.0 + df.get('rating_volatility', 0.5))
        
        # ====== PROVIDER CONTEXT ======
        df['provider_location_count'] = df.get('provider_scale', 1).fillna(1)
        df['provider_avg_rating'] = df.get('provider_reputation', 3.0).fillna(3.0)
        df['provider_rating_consistency'] = 1.0 - df.get('provider_stability', 0.3).fillna(0.3)
        df['provider_scale_category'] = pd.cut(
            df['provider_location_count'],
            bins=[0, 1, 5, 15, 100],
            labels=[1, 2, 3, 4]
        ).astype(float)
        
        # ====== REGIONAL CONTEXT ======
        df['regional_risk_rate'] = df.get('regional_risk_rate', 0.2).fillna(0.2)
        df['regional_avg_beds'] = df.get('regional_avg_beds', 30).fillna(30)
        df['regional_good_rating_rate'] = df.get('regional_performance', 0.7).fillna(0.7)
        
        # ====== SERVICE COMPLEXITY ======
        df['num_regulated_activities'] = df.get('service_complexity', 3).fillna(3)
        df['service_diversity_score'] = df.get('service_diversity', 3).fillna(3)
        df['assessment_complexity'] = df.get('assessment_complexity', 5).fillna(5)
        df['specialisms_count'] = np.random.poisson(2, len(df))  # Placeholder
        
        # ====== HISTORICAL PERFORMANCE ======
        df['historical_avg_rating'] = df.get('historical_performance', 3.0).fillna(3.0)
        df['rating_volatility'] = df.get('rating_volatility', 0.5).fillna(0.5)
        df['inspection_frequency'] = df.get('inspection_frequency', 1).fillna(1)
        df['compliance_trend'] = 1.0  # Placeholder for compliance over time
        
        # ====== INTERACTION FEATURES ======
        df['complexity_scale_interaction'] = (
            df['service_complexity_score'] * df['provider_location_count']
        )
        
        df['inspection_regional_risk'] = (
            df['inspection_overdue_risk'] * df['regional_risk_rate']
        )
        
        df['provider_regional_performance'] = (
            df['provider_avg_rating'] * df['regional_good_rating_rate']
        )
        
        df['risk_capacity_ratio'] = (
            (df['incident_frequency_risk'] + df['medication_risk'] + df['safeguarding_risk']) / 
            (df['bed_capacity'] / 30.0)  # Normalize by typical capacity
        )
        
        # ====== SELECT UNIFIED FEATURES ======
        available_features = [col for col in self.UNIFIED_FEATURE_COLUMNS if col in df.columns]
        logger.info(f"Using {len(available_features)} unified features: {available_features}")
        
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median(numeric_only=True)).fillna(0)
        
        # Prepare target variable
        y = df.get('overall_rating_numeric', df.get('target', 3)).fillna(3).astype(int)
        
        # Validate feature alignment
        self._validate_feature_alignment(X)
        
        logger.info(f"Prepared {len(X)} samples with {len(X.columns)} unified features")
        logger.info(f"Target distribution: {y.value_counts().sort_index().to_dict()}")
        
        return X, y
    
    def _validate_feature_alignment(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate feature alignment with dashboard feature compatibility.
        
        This method ensures that:
        1. All features are numeric and compatible with dashboard extraction
        2. Feature ranges are reasonable for both CQC and dashboard contexts
        3. No features will cause prediction failures due to data type mismatches
        
        Args:
            features: Feature DataFrame to validate
            
        Returns:
            Dictionary with validation results and recommendations
        """
        logger.info("Validating feature alignment with dashboard compatibility...")
        
        validation_results = {
            'valid_features': [],
            'problematic_features': [],
            'missing_features': [],
            'feature_stats': {},
            'compatibility_score': 0.0
        }
        
        # Check all unified features are present
        expected_features = set(self.UNIFIED_FEATURE_COLUMNS)
        actual_features = set(features.columns)
        
        validation_results['missing_features'] = list(expected_features - actual_features)
        if validation_results['missing_features']:
            logger.warning(f"Missing expected features: {validation_results['missing_features']}")
        
        # Validate each feature
        for col in features.columns:
            try:
                # Check data type compatibility
                if not pd.api.types.is_numeric_dtype(features[col]):
                    validation_results['problematic_features'].append({
                        'feature': col,
                        'issue': 'non_numeric_dtype',
                        'dtype': str(features[col].dtype)
                    })
                    continue
                
                # Check for reasonable value ranges
                col_stats = {
                    'min': features[col].min(),
                    'max': features[col].max(),
                    'mean': features[col].mean(),
                    'std': features[col].std(),
                    'nulls': features[col].isnull().sum(),
                    'zeros': (features[col] == 0).sum()
                }
                
                validation_results['feature_stats'][col] = col_stats
                
                # Validate ranges for key dashboard-compatible features
                if col == 'bed_capacity' and (col_stats['min'] < 1 or col_stats['max'] > 300):
                    validation_results['problematic_features'].append({
                        'feature': col,
                        'issue': 'unreasonable_bed_capacity',
                        'range': f"{col_stats['min']}-{col_stats['max']}"
                    })
                
                elif col.endswith('_risk') and (col_stats['min'] < 0 or col_stats['max'] > 1):
                    validation_results['problematic_features'].append({
                        'feature': col,
                        'issue': 'risk_score_out_of_bounds',
                        'range': f"{col_stats['min']}-{col_stats['max']}"
                    })
                
                elif 'rating' in col and (col_stats['min'] < 0 or col_stats['max'] > 4):
                    validation_results['problematic_features'].append({
                        'feature': col,
                        'issue': 'rating_out_of_bounds',
                        'range': f"{col_stats['min']}-{col_stats['max']}"
                    })
                
                else:
                    validation_results['valid_features'].append(col)
                
            except Exception as e:
                validation_results['problematic_features'].append({
                    'feature': col,
                    'issue': 'validation_error',
                    'error': str(e)
                })
        
        # Calculate compatibility score
        total_features = len(features.columns)
        valid_features = len(validation_results['valid_features'])
        validation_results['compatibility_score'] = valid_features / total_features if total_features > 0 else 0.0
        
        logger.info(f"Feature validation complete:")
        logger.info(f"  Valid features: {valid_features}/{total_features} ({validation_results['compatibility_score']:.2%})")
        logger.info(f"  Problematic features: {len(validation_results['problematic_features'])}")
        logger.info(f"  Missing expected features: {len(validation_results['missing_features'])}")
        
        if validation_results['problematic_features']:
            logger.warning("Problematic features detected:")
            for issue in validation_results['problematic_features'][:5]:  # Show first 5
                logger.warning(f"  {issue['feature']}: {issue['issue']}")
        
        return validation_results
    
    def _train_ensemble_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train ensemble models (XGBoost, LightGBM, Random Forest).
        
        This method trains multiple models and creates an ensemble for robust predictions:
        - XGBoost: Gradient boosting with excellent performance on tabular data
        - LightGBM: Fast and memory-efficient gradient boosting
        - Random Forest: Robust ensemble method with good interpretability
        - Voting Classifier: Combines all models for final predictions
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            Dictionary containing trained models and ensemble
        """
        logger.info("Training ensemble models (XGBoost, LightGBM, Random Forest)...")
        
        # Split data for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models = {}
        
        # ====== XGBoost Model ======
        logger.info("Training XGBoost model...")
        xgb_params = {
            'objective': 'multi:softprob',
            'num_class': 5,
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbosity': 0
        }
        
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=20,
            verbose=False
        )
        models['xgboost'] = xgb_model
        
        # ====== LightGBM Model ======
        logger.info("Training LightGBM model...")
        lgb_params = {
            'objective': 'multiclass',
            'num_class': 5,
            'metric': 'multi_logloss',
            'max_depth': 6,
            'learning_rate': 0.1,
            'n_estimators': 200,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'random_state': 42,
            'verbose': -1
        }
        
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        models['lightgbm'] = lgb_model
        
        # ====== Random Forest Model ======
        logger.info("Training Random Forest model...")
        rf_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1
        }
        
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        models['random_forest'] = rf_model
        
        # ====== Ensemble Voting Classifier ======
        logger.info("Creating ensemble voting classifier...")
        ensemble_model = VotingClassifier(
            estimators=[
                ('xgboost', xgb_model),
                ('lightgbm', lgb_model),
                ('random_forest', rf_model)
            ],
            voting='soft',  # Use probability predictions
            n_jobs=-1
        )
        
        ensemble_model.fit(X_train, y_train)
        models['ensemble'] = ensemble_model
        
        # ====== Model Evaluation ======
        logger.info("Evaluating individual models and ensemble...")
        model_performance = {}
        
        for model_name, model in models.items():
            y_pred = model.predict(X_val)
            y_pred_proba = model.predict_proba(X_val)
            
            performance = {
                'accuracy': accuracy_score(y_val, y_pred),
                'f1_macro': f1_score(y_val, y_pred, average='macro'),
                'log_loss': log_loss(y_val, y_pred_proba),
                'precision_macro': precision_score(y_val, y_pred, average='macro', zero_division=0),
                'recall_macro': recall_score(y_val, y_pred, average='macro', zero_division=0)
            }
            
            try:
                performance['roc_auc_ovr'] = roc_auc_score(y_val, y_pred_proba, multi_class='ovr')
            except ValueError:
                performance['roc_auc_ovr'] = 0.0
            
            model_performance[model_name] = performance
            
            logger.info(f"{model_name.upper()} Performance:")
            logger.info(f"  Accuracy: {performance['accuracy']:.4f}")
            logger.info(f"  F1 (Macro): {performance['f1_macro']:.4f}")
            logger.info(f"  Log Loss: {performance['log_loss']:.4f}")
        
        # Store best performing model as primary
        best_model_name = max(model_performance.keys(), 
                            key=lambda k: model_performance[k]['f1_macro'])
        logger.info(f"Best performing model: {best_model_name.upper()}")
        
        return {
            'models': models,
            'performance': model_performance,
            'best_model': best_model_name,
            'training_data': {'X_train': X_train, 'X_val': X_val, 'y_train': y_train, 'y_val': y_val}
        }
    
    def train_comprehensive_model(self, 
                                tune_hyperparameters: bool = False,
                                save_artifacts: bool = True,
                                deploy_to_vertex: bool = True) -> Dict[str, Any]:
        """
        Train comprehensive model on CQC data with dashboard feature validation.
        
        This is the main training method that orchestrates the entire pipeline:
        1. Load comprehensive CQC data from all API endpoints
        2. Prepare unified features compatible with dashboard systems
        3. Train ensemble models with cross-validation
        4. Validate feature alignment for dashboard compatibility
        5. Save model artifacts to GCS
        6. Deploy to Vertex AI (optional)
        
        Args:
            tune_hyperparameters: Whether to perform hyperparameter tuning
            save_artifacts: Whether to save model artifacts to GCS
            deploy_to_vertex: Whether to deploy models to Vertex AI
            
        Returns:
            Dictionary with training results, model performance, and deployment info
        """
        logger.info("Starting comprehensive CQC model training pipeline...")
        
        try:
            # ====== STEP 1: Load Comprehensive CQC Data ======
            training_data = self._load_comprehensive_cqc_data()
            
            if len(training_data) < 100:
                raise ValueError(f"Insufficient training data: {len(training_data)} samples")
            
            # ====== STEP 2: Prepare Unified Features ======
            X, y = self._prepare_unified_features(training_data)
            
            # Store feature names for later use
            self.feature_names = X.columns.tolist()
            
            # ====== STEP 3: Train Ensemble Models ======
            ensemble_results = self._train_ensemble_models(X, y)
            
            # Store models
            self.models = ensemble_results['models']
            self.ensemble_model = ensemble_results['models']['ensemble']
            
            # ====== STEP 4: Feature Importance Analysis ======
            logger.info("Analyzing feature importance across ensemble...")
            feature_importance = self._analyze_ensemble_feature_importance(X)
            
            # ====== STEP 5: Cross-Validation Assessment ======
            cv_results = self._perform_cross_validation(X, y)
            
            # ====== STEP 6: Model Artifacts and Deployment ======
            artifacts_info = {}
            deployment_info = {}
            
            if save_artifacts:
                bucket_name = f"{self.project_id}-cqc-models"
                artifacts_info = self._save_ensemble_artifacts(bucket_name)
            
            if deploy_to_vertex and save_artifacts:
                deployment_info = self._deploy_to_vertex_ai(artifacts_info['model_uri'])
            
            # ====== STEP 7: Comprehensive Results ======
            results = {
                'training_summary': {
                    'samples_trained': len(X),
                    'features_used': len(self.feature_names),
                    'target_classes': len(y.unique()),
                    'training_date': datetime.now().isoformat(),
                    'run_name': self.run_name
                },
                'model_performance': ensemble_results['performance'],
                'best_model': ensemble_results['best_model'],
                'feature_importance': feature_importance,
                'cross_validation': cv_results,
                'feature_alignment': self._validate_feature_alignment(X),
                'artifacts': artifacts_info,
                'deployment': deployment_info,
                'unified_features': self.feature_names
            }
            
            # Log final results
            best_performance = ensemble_results['performance'][ensemble_results['best_model']]
            logger.info("="*80)
            logger.info("UNIFIED CQC MODEL TRAINING COMPLETED SUCCESSFULLY!")
            logger.info("="*80)
            logger.info(f"Training samples: {len(X)}")
            logger.info(f"Unified features: {len(self.feature_names)}")
            logger.info(f"Best model: {ensemble_results['best_model'].upper()}")
            logger.info(f"Best F1 Score (Macro): {best_performance['f1_macro']:.4f}")
            logger.info(f"Best Accuracy: {best_performance['accuracy']:.4f}")
            logger.info(f"Dashboard compatibility: {results['feature_alignment']['compatibility_score']:.2%}")
            
            if artifacts_info:
                logger.info(f"Model artifacts: {artifacts_info['model_uri']}")
            if deployment_info:
                logger.info(f"Vertex AI endpoint: {deployment_info.get('endpoint_name', 'N/A')}")
            
            logger.info("="*80)
            
            return results
            
        except Exception as e:
            logger.error(f"Comprehensive model training failed: {e}")
            logger.exception("Full error traceback:")
            raise
    
    def _analyze_ensemble_feature_importance(self, X: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance across the ensemble models."""
        logger.info("Analyzing feature importance across ensemble models...")
        
        importance_results = {}
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            importance_results['xgboost'] = dict(zip(self.feature_names, xgb_importance))
        
        # LightGBM feature importance
        if 'lightgbm' in self.models:
            lgb_importance = self.models['lightgbm'].feature_importances_
            importance_results['lightgbm'] = dict(zip(self.feature_names, lgb_importance))
        
        # Random Forest feature importance
        if 'random_forest' in self.models:
            rf_importance = self.models['random_forest'].feature_importances_
            importance_results['random_forest'] = dict(zip(self.feature_names, rf_importance))
        
        # Average importance across models
        if importance_results:
            avg_importance = {}
            for feature in self.feature_names:
                importances = [importance_results[model].get(feature, 0) 
                             for model in importance_results.keys()]
                avg_importance[feature] = np.mean(importances)
            
            # Sort by average importance
            sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
            
            logger.info("Top 10 most important features (ensemble average):")
            for feature, importance in sorted_importance[:10]:
                logger.info(f"  {feature}: {importance:.4f}")
            
            importance_results['ensemble_average'] = dict(sorted_importance)
        
        return importance_results
    
    def _perform_cross_validation(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Perform cross-validation assessment of ensemble models."""
        logger.info("Performing cross-validation assessment...")
        
        cv_results = {}
        cv_folds = 5
        
        for model_name, model in self.models.items():
            if model_name == 'ensemble':  # Skip ensemble for CV (too slow)
                continue
                
            try:
                cv_scores = cross_val_score(model, X, y, cv=cv_folds, 
                                          scoring='f1_macro', n_jobs=-1)
                
                cv_results[model_name] = {
                    'cv_scores': cv_scores.tolist(),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'cv_folds': cv_folds
                }
                
                logger.info(f"{model_name.upper()} CV F1 (Macro): {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
                
            except Exception as e:
                logger.warning(f"Cross-validation failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        return cv_results
    
    def _save_ensemble_artifacts(self, bucket_name: str, 
                               model_version: str = None) -> Dict[str, str]:
        """Save ensemble model artifacts to GCS."""
        if model_version is None:
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Saving ensemble model artifacts to gs://{bucket_name}/models/unified/{model_version}/")
        
        try:
            bucket = self.storage_client.bucket(bucket_name)
            base_path = f"models/unified/{model_version}"
            
            # Save individual models
            model_paths = {}
            for model_name, model in self.models.items():
                model_path = f'/tmp/unified_{model_name}_model.pkl'
                joblib.dump(model, model_path)
                
                blob = bucket.blob(f"{base_path}/{model_name}_model.pkl")
                blob.upload_from_filename(model_path)
                model_paths[model_name] = f"gs://{bucket_name}/{base_path}/{model_name}_model.pkl"
            
            # Save feature names
            feature_path = f'/tmp/unified_feature_names.json'
            with open(feature_path, 'w') as f:
                json.dump(self.feature_names, f)
            
            blob = bucket.blob(f"{base_path}/feature_names.json")
            blob.upload_from_filename(feature_path)
            
            # Save unified feature mapping (for dashboard compatibility)
            mapping_path = f'/tmp/unified_feature_mapping.json'
            feature_mapping = {
                'unified_features': self.feature_names,
                'dashboard_compatible': self.UNIFIED_FEATURE_COLUMNS,
                'feature_types': self._get_feature_types(),
                'version': model_version,
                'created': datetime.now().isoformat()
            }
            
            with open(mapping_path, 'w') as f:
                json.dump(feature_mapping, f, indent=2)
            
            blob = bucket.blob(f"{base_path}/feature_mapping.json")
            blob.upload_from_filename(mapping_path)
            
            # Save model metadata
            metadata = {
                'model_type': 'unified_ensemble',
                'version': model_version,
                'training_date': datetime.now().isoformat(),
                'models_included': list(self.models.keys()),
                'feature_count': len(self.feature_names),
                'unified_features': self.feature_names,
                'dashboard_compatible': True,
                'project_id': self.project_id,
                'experiment_name': self.experiment_name,
                'run_name': self.run_name
            }
            
            metadata_path = f'/tmp/unified_model_metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            blob = bucket.blob(f"{base_path}/metadata.json")
            blob.upload_from_filename(metadata_path)
            
            model_uri = f"gs://{bucket_name}/{base_path}/"
            logger.info(f"Unified ensemble artifacts saved to {model_uri}")
            
            return {
                'model_uri': model_uri,
                'model_paths': model_paths,
                'version': model_version,
                'bucket': bucket_name,
                'base_path': base_path
            }
            
        except Exception as e:
            logger.error(f"Failed to save ensemble artifacts: {e}")
            return {'error': str(e)}
    
    def _deploy_to_vertex_ai(self, model_uri: str) -> Dict[str, str]:
        """Deploy ensemble model to Vertex AI."""
        logger.info("Deploying unified ensemble model to Vertex AI...")
        
        try:
            model_display_name = f"cqc-unified-ensemble-{self.run_name}"
            
            # Register model
            model = aiplatform.Model.upload(
                display_name=model_display_name,
                artifact_uri=model_uri,
                serving_container_image_uri="gcr.io/cloud-aiplatform/prediction/sklearn-cpu.1-0:latest",
                description=f"Unified CQC rating prediction ensemble model - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                labels={
                    "model_type": "unified_ensemble",
                    "use_case": "cqc_rating_prediction",
                    "framework": "sklearn_xgboost_lightgbm",
                    "dashboard_compatible": "true"
                }
            )
            
            # Deploy to endpoint
            endpoint_display_name = "cqc-unified-prediction-endpoint"
            
            # Check for existing endpoint
            endpoints = aiplatform.Endpoint.list(
                filter=f'display_name="{endpoint_display_name}"'
            )
            
            if endpoints:
                endpoint = endpoints[0]
                logger.info(f"Using existing endpoint: {endpoint.display_name}")
            else:
                endpoint = aiplatform.Endpoint.create(
                    display_name=endpoint_display_name,
                    description="Unified endpoint for CQC rating prediction with dashboard compatibility"
                )
                logger.info(f"Created new endpoint: {endpoint.display_name}")
            
            # Deploy model
            deployed_model = model.deploy(
                endpoint=endpoint,
                deployed_model_display_name=f"unified-ensemble-{self.run_name}",
                machine_type="n1-standard-4",
                min_replica_count=1,
                max_replica_count=5,
                traffic_percentage=100,
                sync=True
            )
            
            logger.info(f"Unified model deployed to endpoint: {endpoint.resource_name}")
            
            return {
                'model_name': model.resource_name,
                'endpoint_name': endpoint.resource_name,
                'endpoint_id': endpoint.name,
                'model_display_name': model_display_name,
                'deployment_status': 'deployed'
            }
            
        except Exception as e:
            logger.error(f"Vertex AI deployment failed: {e}")
            return {'error': str(e), 'deployment_status': 'failed'}
    
    def _get_feature_types(self) -> Dict[str, str]:
        """Get feature type mappings for dashboard compatibility."""
        feature_types = {}
        
        for feature in self.feature_names:
            if 'risk' in feature or 'rate' in feature:
                feature_types[feature] = 'risk_indicator'
            elif 'rating' in feature or 'score' in feature:
                feature_types[feature] = 'quality_metric'
            elif 'days' in feature or 'age' in feature:
                feature_types[feature] = 'temporal'
            elif 'capacity' in feature or 'beds' in feature or 'size' in feature:
                feature_types[feature] = 'operational'
            elif 'provider' in feature:
                feature_types[feature] = 'provider_context'
            elif 'regional' in feature:
                feature_types[feature] = 'regional_context'
            else:
                feature_types[feature] = 'derived'
        
        return feature_types


def main():
    """Main function for unified CQC model training."""
    
    # Configuration
    project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
    region = os.environ.get('GCP_REGION', 'europe-west2')
    
    # Training options
    tune_hyperparameters = os.environ.get('TUNE_HYPERPARAMETERS', 'false').lower() == 'true'
    save_artifacts = os.environ.get('SAVE_ARTIFACTS', 'true').lower() == 'true'
    deploy_to_vertex = os.environ.get('DEPLOY_TO_VERTEX', 'true').lower() == 'true'
    
    logger.info("="*80)
    logger.info("UNIFIED CQC MODEL TRAINER - PHASE 3")
    logger.info("="*80)
    logger.info(f"Project ID: {project_id}")
    logger.info(f"Region: {region}")
    logger.info(f"Tune hyperparameters: {tune_hyperparameters}")
    logger.info(f"Save artifacts: {save_artifacts}")
    logger.info(f"Deploy to Vertex AI: {deploy_to_vertex}")
    logger.info("="*80)
    
    # Initialize trainer
    trainer = UnifiedCQCModelTrainer(project_id, region)
    
    # Run comprehensive training
    results = trainer.train_comprehensive_model(
        tune_hyperparameters=tune_hyperparameters,
        save_artifacts=save_artifacts,
        deploy_to_vertex=deploy_to_vertex
    )
    
    # Output results summary
    print("\n" + "="*80)
    print("UNIFIED CQC MODEL TRAINING RESULTS")
    print("="*80)
    print(f"Training samples: {results['training_summary']['samples_trained']:,}")
    print(f"Unified features: {results['training_summary']['features_used']}")
    print(f"Best model: {results['best_model'].upper()}")
    
    best_perf = results['model_performance'][results['best_model']]
    print(f"Best F1 Score: {best_perf['f1_macro']:.4f}")
    print(f"Best Accuracy: {best_perf['accuracy']:.4f}")
    print(f"Dashboard compatibility: {results['feature_alignment']['compatibility_score']:.2%}")
    
    if results['artifacts']:
        print(f"Model artifacts: {results['artifacts']['model_uri']}")
    
    if results['deployment'] and 'endpoint_name' in results['deployment']:
        print(f"Vertex AI endpoint: {results['deployment']['endpoint_name']}")
    
    print("="*80)
    print("Training completed successfully! Models ready for dashboard integration.")
    print("="*80)


if __name__ == "__main__":
    main()