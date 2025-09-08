#!/usr/bin/env python3
"""
Feature engineering module for CQC rating prediction.
Generates ML-ready features from processed CQC data.
Designed to run on Vertex AI Pipelines.
"""

import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from google.cloud import bigquery
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CQCFeatureEngineer:
    """Feature engineering for CQC rating prediction."""
    
    # Rating mapping
    RATING_MAP = {
        'Outstanding': 4,
        'Good': 3,
        'Requires improvement': 2,
        'Inadequate': 1,
        'No published rating': 0,
        None: 0
    }
    
    # High-risk specialisms
    HIGH_RISK_SPECIALISMS = {
        'Dementia',
        'Mental health conditions',
        'Substance misuse problems',
        'Acquired brain injury',
        'People detained under the Mental Health Act'
    }
    
    def __init__(self, project_id: str):
        self.project_id = project_id
        self.bigquery_client = bigquery.Client(project=project_id)
        self.storage_client = storage.Client(project=project_id)
        self.dataset_id = 'cqc_dataset'
        
        # Initialize encoders
        self.label_encoders = {}
        self.scaler = StandardScaler()
        
        logger.info(f"Initialized CQCFeatureEngineer for project {project_id}")
    
    def load_care_homes_data(self, limit: Optional[int] = None) -> pd.DataFrame:
        """Load care homes data from BigQuery."""
        query = f"""
        SELECT 
            locationId,
            locationName,
            providerId,
            providerName,
            type,
            care_home_type,
            registrationStatus,
            registrationDate,
            numberOfBeds,
            postalCode,
            region,
            localAuthority,
            overall_rating,
            safe_rating,
            effective_rating,
            caring_rating,
            responsive_rating,
            wellLed_rating,
            last_report_date,
            cares_for_adults_over_65,
            cares_for_adults_under_65,
            dementia_care,
            mental_health_care,
            physical_disabilities_care,
            learning_disabilities_care,
            has_nursing,
            days_since_last_inspection,
            rating_trend
        FROM `{self.project_id}.{self.dataset_id}.care_homes`
        WHERE registrationStatus = 'Registered'
        AND overall_rating IS NOT NULL
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        logger.info("Loading care homes data from BigQuery...")
        df = self.bigquery_client.query(query).to_dataframe()
        logger.info(f"Loaded {len(df)} care homes")
        
        return df
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features."""
        logger.info("Creating temporal features...")
        
        # Convert dates
        df['registrationDate'] = pd.to_datetime(df['registrationDate'])
        df['last_report_date'] = pd.to_datetime(df['last_report_date'])
        
        # Years in operation
        current_date = pd.Timestamp.now()
        df['years_registered'] = (current_date - df['registrationDate']).dt.days / 365.25
        
        # Time since last inspection
        df['days_since_inspection'] = (current_date - df['last_report_date']).dt.days
        
        # Inspection frequency indicator
        df['inspection_overdue'] = (df['days_since_inspection'] > 365).astype(int)
        df['inspection_very_overdue'] = (df['days_since_inspection'] > 730).astype(int)
        
        # Seasonal factors
        df['inspection_month'] = df['last_report_date'].dt.month
        df['inspection_quarter'] = df['last_report_date'].dt.quarter
        df['winter_inspection'] = df['inspection_month'].isin([12, 1, 2]).astype(int)
        
        # Registration cohort
        df['registration_year'] = df['registrationDate'].dt.year
        df['is_new_home'] = (df['years_registered'] < 2).astype(int)
        df['is_established_home'] = (df['years_registered'] > 10).astype(int)
        
        return df
    
    def create_geographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create geographic and regional features."""
        logger.info("Creating geographic features...")
        
        # Regional statistics
        regional_stats = df.groupby('region').agg({
            'overall_rating': lambda x: x.map(self.RATING_MAP).mean(),
            'locationId': 'count',
            'numberOfBeds': 'mean'
        }).rename(columns={
            'overall_rating': 'region_avg_rating',
            'locationId': 'region_home_count',
            'numberOfBeds': 'region_avg_beds'
        })
        
        df = df.merge(regional_stats, on='region', how='left')
        
        # Local authority statistics
        la_stats = df.groupby('localAuthority').agg({
            'overall_rating': lambda x: x.map(self.RATING_MAP).mean(),
            'locationId': 'count'
        }).rename(columns={
            'overall_rating': 'la_avg_rating',
            'locationId': 'la_home_count'
        })
        
        df = df.merge(la_stats, on='localAuthority', how='left')
        
        # Relative performance
        df['rating_numeric'] = df['overall_rating'].map(self.RATING_MAP)
        df['above_region_avg'] = (df['rating_numeric'] > df['region_avg_rating']).astype(int)
        df['above_la_avg'] = (df['rating_numeric'] > df['la_avg_rating']).astype(int)
        
        # Urban/rural indicator (simplified based on region)
        urban_regions = ['London', 'West Midlands', 'Greater Manchester']
        df['is_urban'] = df['region'].isin(urban_regions).astype(int)
        
        return df
    
    def create_provider_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create provider-level features."""
        logger.info("Creating provider features...")
        
        # Provider statistics
        provider_stats = df.groupby('providerId').agg({
            'locationId': 'count',
            'overall_rating': lambda x: x.map(self.RATING_MAP).mean(),
            'numberOfBeds': 'sum',
            'has_nursing': 'mean'
        }).rename(columns={
            'locationId': 'provider_location_count',
            'overall_rating': 'provider_avg_rating',
            'numberOfBeds': 'provider_total_beds',
            'has_nursing': 'provider_nursing_ratio'
        })
        
        # Provider rating consistency
        provider_std = df.groupby('providerId')['rating_numeric'].std().rename('provider_rating_std')
        provider_stats = provider_stats.join(provider_std)
        
        df = df.merge(provider_stats, on='providerId', how='left')
        
        # Provider size categories
        df['is_large_provider'] = (df['provider_location_count'] > 10).astype(int)
        df['is_single_location'] = (df['provider_location_count'] == 1).astype(int)
        
        # Provider performance relative to locations
        df['better_than_provider_avg'] = (df['rating_numeric'] > df['provider_avg_rating']).astype(int)
        
        return df
    
    def create_capacity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to home capacity and size."""
        logger.info("Creating capacity features...")
        
        # Bed capacity categories
        df['bed_category'] = pd.cut(
            df['numberOfBeds'].fillna(0),
            bins=[0, 10, 25, 50, 100, float('inf')],
            labels=['very_small', 'small', 'medium', 'large', 'very_large']
        )
        
        # Capacity utilization proxy (would need actual occupancy data)
        df['beds_log'] = np.log1p(df['numberOfBeds'].fillna(0))
        
        # Size relative to region
        df['beds_vs_region_avg'] = df['numberOfBeds'] / df['region_avg_beds']
        df['is_above_avg_size'] = (df['beds_vs_region_avg'] > 1).astype(int)
        
        return df
    
    def create_service_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features related to services and specialisms."""
        logger.info("Creating service features...")
        
        # Count specialisms
        specialism_cols = [
            'cares_for_adults_over_65', 'cares_for_adults_under_65',
            'dementia_care', 'mental_health_care', 'physical_disabilities_care',
            'learning_disabilities_care'
        ]
        
        df['num_specialisms'] = df[specialism_cols].sum(axis=1)
        df['is_specialized'] = (df['num_specialisms'] == 1).astype(int)
        df['is_multi_specialty'] = (df['num_specialisms'] > 2).astype(int)
        
        # High-risk specialisms
        df['has_high_risk_specialism'] = (
            df['dementia_care'] | 
            df['mental_health_care'] | 
            df['learning_disabilities_care']
        ).astype(int)
        
        # Service complexity
        df['service_complexity'] = (
            df['has_nursing'].astype(int) + 
            df['has_high_risk_specialism'] + 
            df['is_multi_specialty']
        )
        
        # Age group focus
        df['elderly_focused'] = (
            df['cares_for_adults_over_65'] & 
            ~df['cares_for_adults_under_65']
        ).astype(int)
        
        df['working_age_focused'] = (
            ~df['cares_for_adults_over_65'] & 
            df['cares_for_adults_under_65']
        ).astype(int)
        
        return df
    
    def create_rating_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from rating domains."""
        logger.info("Creating rating features...")
        
        # Convert ratings to numeric
        rating_cols = ['safe_rating', 'effective_rating', 'caring_rating', 
                      'responsive_rating', 'wellLed_rating']
        
        for col in rating_cols:
            df[f'{col}_numeric'] = df[col].map(self.RATING_MAP)
        
        # Rating statistics
        df['avg_domain_rating'] = df[[f'{col}_numeric' for col in rating_cols]].mean(axis=1)
        df['min_domain_rating'] = df[[f'{col}_numeric' for col in rating_cols]].min(axis=1)
        df['max_domain_rating'] = df[[f'{col}_numeric' for col in rating_cols]].max(axis=1)
        df['rating_range'] = df['max_domain_rating'] - df['min_domain_rating']
        
        # Rating consistency
        df['rating_std'] = df[[f'{col}_numeric' for col in rating_cols]].std(axis=1)
        df['ratings_consistent'] = (df['rating_std'] < 0.5).astype(int)
        
        # Specific domain indicators
        df['safe_inadequate'] = (df['safe_rating'] == 'Inadequate').astype(int)
        df['wellled_inadequate'] = (df['wellLed_rating'] == 'Inadequate').astype(int)
        df['any_inadequate'] = (df['min_domain_rating'] == 1).astype(int)
        df['all_good_or_better'] = (df['min_domain_rating'] >= 3).astype(int)
        
        # Rating trend features
        df['improving_trend'] = (df['rating_trend'] == 'improving').astype(int)
        df['declining_trend'] = (df['rating_trend'] == 'declining').astype(int)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different feature groups."""
        logger.info("Creating interaction features...")
        
        # Size and complexity interaction
        df['large_complex'] = (df['is_above_avg_size'] & df['has_high_risk_specialism']).astype(int)
        
        # New home with specialisms
        df['new_specialized'] = (df['is_new_home'] & df['is_specialized']).astype(int)
        
        # Provider and performance interaction
        df['large_provider_poor_performance'] = (
            df['is_large_provider'] & (df['rating_numeric'] < 3)
        ).astype(int)
        
        # Geographic and service interaction
        df['urban_nursing'] = (df['is_urban'] & df['has_nursing']).astype(int)
        
        # Inspection and rating interaction
        df['overdue_poor_rating'] = (
            df['inspection_overdue'] & (df['rating_numeric'] < 3)
        ).astype(int)
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables."""
        logger.info("Encoding categorical features...")
        
        categorical_cols = ['region', 'care_home_type', 'bed_category']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].fillna('Unknown'))
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final features for ML model."""
        logger.info("Selecting final features...")
        
        # Define feature columns
        feature_cols = [
            # Temporal
            'years_registered', 'days_since_inspection', 'inspection_overdue',
            'winter_inspection', 'is_new_home', 'is_established_home',
            
            # Geographic
            'region_avg_rating', 'region_home_count', 'la_avg_rating',
            'above_region_avg', 'above_la_avg', 'is_urban',
            
            # Provider
            'provider_location_count', 'provider_avg_rating', 'provider_rating_std',
            'is_large_provider', 'is_single_location', 'better_than_provider_avg',
            
            # Capacity
            'numberOfBeds', 'beds_log', 'beds_vs_region_avg', 'is_above_avg_size',
            
            # Service
            'has_nursing', 'num_specialisms', 'is_specialized', 'is_multi_specialty',
            'has_high_risk_specialism', 'service_complexity', 'elderly_focused',
            
            # Ratings
            'avg_domain_rating', 'min_domain_rating', 'rating_range', 'rating_std',
            'safe_inadequate', 'wellled_inadequate', 'any_inadequate',
            'improving_trend', 'declining_trend',
            
            # Interactions
            'large_complex', 'new_specialized', 'large_provider_poor_performance',
            'urban_nursing', 'overdue_poor_rating',
            
            # Encoded categoricals
            'region_encoded', 'care_home_type_encoded', 'bed_category_encoded'
        ]
        
        # Keep only available features
        available_features = [col for col in feature_cols if col in df.columns]
        
        # Add target
        df['target'] = df['overall_rating'].map(self.RATING_MAP)
        
        # Select features and target
        final_df = df[['locationId'] + available_features + ['target']].copy()
        
        logger.info(f"Selected {len(available_features)} features")
        
        return final_df
    
    def scale_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features."""
        logger.info("Scaling features...")
        
        # Identify numerical columns (excluding IDs and target)
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numerical_cols = [col for col in numerical_cols 
                         if col not in ['locationId', 'target']]
        
        if fit:
            df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
        else:
            df[numerical_cols] = self.scaler.transform(df[numerical_cols])
        
        return df
    
    def save_features_to_bigquery(self, df: pd.DataFrame, table_suffix: str = ''):
        """Save engineered features to BigQuery."""
        table_id = f"{self.project_id}.{self.dataset_id}.ml_features{table_suffix}"
        
        logger.info(f"Saving features to {table_id}...")
        
        # Add metadata
        df['feature_version'] = 'v1.0'
        df['feature_date'] = datetime.now().date()
        df['created_timestamp'] = datetime.now()
        
        # Save to BigQuery
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
            schema_update_options=[
                bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
            ]
        )
        
        job = self.bigquery_client.load_table_from_dataframe(
            df, table_id, job_config=job_config
        )
        job.result()
        
        logger.info(f"Saved {len(df)} records to {table_id}")
    
    def save_artifacts(self, bucket_name: str):
        """Save feature engineering artifacts to GCS."""
        logger.info(f"Saving artifacts to gs://{bucket_name}/feature_artifacts/...")
        
        bucket = self.storage_client.bucket(bucket_name)
        
        # Save encoders
        for name, encoder in self.label_encoders.items():
            blob = bucket.blob(f"feature_artifacts/encoders/{name}_encoder.pkl")
            blob.upload_from_string(joblib.dumps(encoder))
        
        # Save scaler
        blob = bucket.blob("feature_artifacts/scaler/scaler.pkl")
        blob.upload_from_string(joblib.dumps(self.scaler))
        
        logger.info("Artifacts saved successfully")
    
    def run_feature_pipeline(self, save_to_bigquery: bool = True, save_artifacts: bool = True):
        """Run the complete feature engineering pipeline."""
        logger.info("Starting feature engineering pipeline...")
        
        # Load data
        df = self.load_care_homes_data()
        
        # Create features
        df = self.create_temporal_features(df)
        df = self.create_geographic_features(df)
        df = self.create_provider_features(df)
        df = self.create_capacity_features(df)
        df = self.create_service_features(df)
        df = self.create_rating_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df, fit=True)
        
        # Select and scale features
        df = self.select_features(df)
        df = self.scale_features(df, fit=True)
        
        # Save results
        if save_to_bigquery:
            self.save_features_to_bigquery(df)
        
        if save_artifacts:
            self.save_artifacts(f"{self.project_id}-cqc-processed")
        
        logger.info(f"Feature engineering completed. Generated {len(df.columns)-2} features for {len(df)} samples")
        
        return df

def main():
    """Main function for Vertex AI Pipeline."""
    project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
    
    engineer = CQCFeatureEngineer(project_id)
    features_df = engineer.run_feature_pipeline()
    
    # Print feature importance preview
    logger.info("\nFeature Engineering Summary:")
    logger.info(f"Total samples: {len(features_df)}")
    logger.info(f"Total features: {len(features_df.columns) - 2}")
    logger.info(f"Target distribution:\n{features_df['target'].value_counts()}")

if __name__ == "__main__":
    main()