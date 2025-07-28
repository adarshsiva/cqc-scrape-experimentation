"""
Feature engineering functions for CQC ML pipeline.

This module contains functions for feature engineering including:
- Categorical encoding for service types, regions, etc
- Numerical feature scaling
- Text feature extraction from inspection areas
- Feature selection logic
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from typing import Dict, List, Tuple, Optional
import re
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CQCFeatureEngineer:
    """Feature engineering class for CQC data."""
    
    def __init__(self):
        """Initialize feature engineering components."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.one_hot_encoders = {}
        self.tfidf_vectorizers = {}
        self.feature_selector = None
        self.feature_names = []
        
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract temporal features from date columns.
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with additional temporal features
        """
        logger.info("Extracting temporal features")
        
        # Convert date columns to datetime
        date_columns = ['lastInspectionDate', 'registrationDate']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        
        # Calculate time since last inspection
        if 'lastInspectionDate' in df.columns:
            df['daysSinceLastInspection'] = (
                datetime.now() - df['lastInspectionDate']
            ).dt.days
            df['monthsSinceLastInspection'] = df['daysSinceLastInspection'] / 30
            df['yearsSinceLastInspection'] = df['daysSinceLastInspection'] / 365
            
        # Calculate registration age
        if 'registrationDate' in df.columns:
            df['registrationAge'] = (
                datetime.now() - df['registrationDate']
            ).dt.days / 365
            
        # Extract inspection patterns
        if 'lastInspectionDate' in df.columns:
            df['inspectionMonth'] = df['lastInspectionDate'].dt.month
            df['inspectionQuarter'] = df['lastInspectionDate'].dt.quarter
            df['inspectionDayOfWeek'] = df['lastInspectionDate'].dt.dayofweek
            
        return df
    
    def encode_categorical_features(self, 
                                  df: pd.DataFrame, 
                                  categorical_columns: List[str],
                                  encoding_type: str = 'onehot',
                                  is_training: bool = True) -> pd.DataFrame:
        """Encode categorical features.
        
        Args:
            df: DataFrame with categorical columns
            categorical_columns: List of column names to encode
            encoding_type: 'onehot' or 'label'
            is_training: Whether this is training data (fit encoders)
            
        Returns:
            DataFrame with encoded features
        """
        logger.info(f"Encoding categorical features using {encoding_type}")
        
        for col in categorical_columns:
            if col not in df.columns:
                logger.warning(f"Column {col} not found in DataFrame")
                continue
                
            if encoding_type == 'label':
                if is_training:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].fillna('unknown')
                    )
                else:
                    # Handle unseen categories
                    df[f'{col}_encoded'] = df[col].apply(
                        lambda x: self.label_encoders[col].transform([x])[0] 
                        if x in self.label_encoders[col].classes_ 
                        else -1
                    )
            
            elif encoding_type == 'onehot':
                if is_training:
                    self.one_hot_encoders[col] = OneHotEncoder(
                        sparse_output=False, 
                        handle_unknown='ignore'
                    )
                    encoded = self.one_hot_encoders[col].fit_transform(
                        df[[col]].fillna('unknown')
                    )
                else:
                    encoded = self.one_hot_encoders[col].transform(
                        df[[col]].fillna('unknown')
                    )
                
                # Create column names
                feature_names = [
                    f'{col}_{cat}' 
                    for cat in self.one_hot_encoders[col].categories_[0]
                ]
                
                # Add encoded columns to dataframe
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=feature_names, 
                    index=df.index
                )
                df = pd.concat([df, encoded_df], axis=1)
                
        return df
    
    def scale_numerical_features(self, 
                               df: pd.DataFrame, 
                               numerical_columns: List[str],
                               is_training: bool = True) -> pd.DataFrame:
        """Scale numerical features.
        
        Args:
            df: DataFrame with numerical columns
            numerical_columns: List of column names to scale
            is_training: Whether this is training data (fit scaler)
            
        Returns:
            DataFrame with scaled features
        """
        logger.info("Scaling numerical features")
        
        # Filter columns that exist
        existing_columns = [col for col in numerical_columns if col in df.columns]
        
        if not existing_columns:
            logger.warning("No numerical columns found to scale")
            return df
            
        if is_training:
            df[existing_columns] = self.scaler.fit_transform(df[existing_columns])
        else:
            df[existing_columns] = self.scaler.transform(df[existing_columns])
            
        return df
    
    def extract_text_features(self, 
                            df: pd.DataFrame, 
                            text_columns: List[str],
                            max_features: int = 100,
                            is_training: bool = True) -> pd.DataFrame:
        """Extract features from text columns using TF-IDF.
        
        Args:
            df: DataFrame with text columns
            text_columns: List of column names with text
            max_features: Maximum number of features to extract
            is_training: Whether this is training data (fit vectorizer)
            
        Returns:
            DataFrame with text features
        """
        logger.info("Extracting text features")
        
        for col in text_columns:
            if col not in df.columns:
                logger.warning(f"Text column {col} not found")
                continue
                
            # Clean text
            df[col] = df[col].fillna('').apply(self._clean_text)
            
            if is_training:
                self.tfidf_vectorizers[col] = TfidfVectorizer(
                    max_features=max_features,
                    stop_words='english',
                    ngram_range=(1, 2)
                )
                tfidf_features = self.tfidf_vectorizers[col].fit_transform(df[col])
            else:
                tfidf_features = self.tfidf_vectorizers[col].transform(df[col])
            
            # Create feature names
            feature_names = [
                f'{col}_tfidf_{i}' 
                for i in range(tfidf_features.shape[1])
            ]
            
            # Convert to DataFrame
            tfidf_df = pd.DataFrame(
                tfidf_features.toarray(),
                columns=feature_names,
                index=df.index
            )
            
            df = pd.concat([df, tfidf_df], axis=1)
            
        return df
    
    def extract_aggregated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract aggregated features based on groupings.
        
        Args:
            df: DataFrame with raw features
            
        Returns:
            DataFrame with additional aggregated features
        """
        logger.info("Extracting aggregated features")
        
        # Count of regulated activities
        if 'regulatedActivities' in df.columns:
            df['numRegulatedActivities'] = df['regulatedActivities'].apply(
                lambda x: len(x.split(',')) if pd.notna(x) else 0
            )
        
        # Count of inspection areas
        if 'inspectionAreas' in df.columns:
            df['numInspectionAreas'] = df['inspectionAreas'].apply(
                lambda x: len(x.split(',')) if pd.notna(x) else 0
            )
        
        # Regional statistics (would require historical data)
        if 'region' in df.columns:
            # This would be calculated from historical data in practice
            regional_stats = df.groupby('region').agg({
                'rating': lambda x: (x == 'Good').mean() if 'rating' in df.columns else 0
            }).rename(columns={'rating': 'regionalGoodRatingRate'})
            
            df = df.merge(regional_stats, on='region', how='left')
        
        # Service type combinations
        if 'serviceType' in df.columns:
            # Create binary flags for common service types
            common_types = ['Residential', 'Community', 'Hospital', 'Primary']
            for service in common_types:
                df[f'is{service}Service'] = df['serviceType'].str.contains(
                    service, case=False, na=False
                ).astype(int)
        
        return df
    
    def select_features(self, 
                       X: pd.DataFrame, 
                       y: pd.Series,
                       method: str = 'chi2',
                       k: int = 50) -> Tuple[pd.DataFrame, List[str]]:
        """Select top k features based on statistical tests.
        
        Args:
            X: Feature DataFrame
            y: Target variable
            method: Feature selection method ('chi2' or 'f_classif')
            k: Number of features to select
            
        Returns:
            Tuple of (selected features DataFrame, selected feature names)
        """
        logger.info(f"Selecting top {k} features using {method}")
        
        # Ensure all features are non-negative for chi2
        if method == 'chi2':
            X_positive = X.abs()
        else:
            X_positive = X
            
        # Initialize selector
        score_func = chi2 if method == 'chi2' else f_classif
        self.feature_selector = SelectKBest(score_func=score_func, k=k)
        
        # Fit and transform
        X_selected = self.feature_selector.fit_transform(X_positive, y)
        
        # Get selected feature names
        selected_features = X.columns[self.feature_selector.get_support()].tolist()
        
        return pd.DataFrame(X_selected, columns=selected_features, index=X.index), selected_features
    
    def create_feature_pipeline(self, 
                              df: pd.DataFrame,
                              target_column: str = 'rating',
                              is_training: bool = True) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """Complete feature engineering pipeline.
        
        Args:
            df: Raw DataFrame
            target_column: Name of target column
            is_training: Whether this is training data
            
        Returns:
            Tuple of (feature DataFrame, target Series if training)
        """
        logger.info("Running complete feature engineering pipeline")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Extract target if training
        y = None
        if is_training and target_column in df.columns:
            y = df[target_column]
            # Encode target variable
            if y.dtype == 'object':
                self.label_encoders['target'] = LabelEncoder()
                y = pd.Series(
                    self.label_encoders['target'].fit_transform(y),
                    index=y.index
                )
        
        # Define feature types
        categorical_features = [
            'type', 'serviceType', 'region', 'localAuthority', 
            'constituency', 'locationStatus'
        ]
        
        numerical_features = [
            'daysSinceLastInspection', 'monthsSinceLastInspection',
            'yearsSinceLastInspection', 'registrationAge',
            'numRegulatedActivities', 'numInspectionAreas'
        ]
        
        text_features = ['inspectionAreas', 'regulatedActivities']
        
        # Apply feature engineering steps
        df = self.extract_temporal_features(df)
        df = self.extract_aggregated_features(df)
        df = self.encode_categorical_features(
            df, categorical_features, 'onehot', is_training
        )
        df = self.extract_text_features(
            df, text_features, max_features=50, is_training=is_training
        )
        df = self.scale_numerical_features(
            df, numerical_features, is_training
        )
        
        # Select feature columns (exclude original raw columns)
        feature_columns = [
            col for col in df.columns 
            if col not in categorical_features + text_features + [target_column]
            and not col.startswith('last') 
            and not col.startswith('registration')
        ]
        
        X = df[feature_columns]
        
        # Handle missing values
        X = X.fillna(0)
        
        return X, y
    
    def _clean_text(self, text: str) -> str:
        """Clean text for feature extraction.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if pd.isna(text):
            return ''
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def get_feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_ attribute
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance scores
        """
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            return importance_df
        else:
            logger.warning("Model does not have feature_importances_ attribute")
            return pd.DataFrame()