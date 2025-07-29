import argparse
import json
import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_data_from_gcs(input_path: str) -> pd.DataFrame:
    """Download training data from Google Cloud Storage."""
    logger.info(f"Downloading data from {input_path}")
    
    # Parse GCS path
    if input_path.startswith("gs://"):
        path_parts = input_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_path = path_parts[1] if len(path_parts) > 1 else ""
    else:
        raise ValueError("Input path must start with gs://")
    
    # Download file to temporary location
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Find CSV files
    blobs = list(bucket.list_blobs(prefix=blob_path))
    csv_blobs = [b for b in blobs if b.name.endswith('.csv')]
    
    if not csv_blobs:
        raise ValueError(f"No CSV files found in {input_path}")
    
    # Download and read the first CSV file
    blob = csv_blobs[0]
    temp_file = f"/tmp/{os.path.basename(blob.name)}"
    blob.download_to_filename(temp_file)
    
    df = pd.read_csv(temp_file)
    logger.info(f"Loaded data with shape: {df.shape}")
    
    # Clean up
    os.remove(temp_file)
    
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder, StandardScaler]:
    """Preprocess the data for training."""
    logger.info("Preprocessing data...")
    
    # Remove rows without ratings
    df = df[df['rating_label'].notna()]
    
    # Select features for training
    feature_columns = [
        'number_of_beds', 'number_of_locations', 'inspection_history_length',
        'days_since_last_inspection', 'service_types_count', 'specialisms_count',
        'regulated_activities_count', 'service_user_groups_count',
        'has_previous_rating', 'ownership_changed_recently',
        'nominated_individual_exists'
    ]
    
    # Handle missing values
    df['days_since_last_inspection'] = df['days_since_last_inspection'].fillna(365)
    
    # Convert boolean columns to int
    bool_columns = ['has_previous_rating', 'ownership_changed_recently', 
                    'nominated_individual_exists']
    for col in bool_columns:
        df[col] = df[col].astype(int)
    
    # Encode categorical features
    le_ownership = LabelEncoder()
    df['ownership_type_encoded'] = le_ownership.fit_transform(df['ownership_type'])
    feature_columns.append('ownership_type_encoded')
    
    le_region = LabelEncoder()
    df['region_encoded'] = le_region.fit_transform(df['region'])
    feature_columns.append('region_encoded')
    
    # Encode target variable
    le_rating = LabelEncoder()
    df['rating_encoded'] = le_rating.fit_transform(df['rating_label'])
    
    # Scale numerical features
    scaler = StandardScaler()
    X = df[feature_columns]
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_columns)
    
    logger.info(f"Preprocessed data shape: {X_scaled_df.shape}")
    
    return X_scaled_df, df['rating_encoded'], le_rating, scaler


def train_model(X: pd.DataFrame, y: pd.Series, model_type: str) -> object:
    """Train the specified model."""
    logger.info(f"Training {model_type} model...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train model based on type
    if model_type == "xgboost":
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multi:softprob',
            random_state=42
        )
    elif model_type == "lightgbm":
        model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            objective='multiclass',
            random_state=42
        )
    elif model_type == "random_forest":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    
    # Print classification report
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))
    
    return model, accuracy


def save_model_to_gcs(model: object, scaler: StandardScaler, 
                     label_encoder: LabelEncoder, output_path: str,
                     model_metadata: Dict) -> None:
    """Save model and preprocessing objects to GCS."""
    logger.info(f"Saving model to {output_path}")
    
    # Parse GCS path
    if output_path.startswith("gs://"):
        path_parts = output_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        blob_prefix = path_parts[1] if len(path_parts) > 1 else ""
    else:
        raise ValueError("Output path must start with gs://")
    
    # Create model package
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'metadata': model_metadata,
        'feature_columns': [
            'number_of_beds', 'number_of_locations', 'inspection_history_length',
            'days_since_last_inspection', 'service_types_count', 'specialisms_count',
            'regulated_activities_count', 'service_user_groups_count',
            'has_previous_rating', 'ownership_changed_recently',
            'nominated_individual_exists', 'ownership_type_encoded', 'region_encoded'
        ]
    }
    
    # Save to temporary file
    temp_file = "/tmp/model.pkl"
    with open(temp_file, 'wb') as f:
        pickle.dump(model_package, f)
    
    # Upload to GCS
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(os.path.join(blob_prefix, "model.pkl"))
    blob.upload_from_filename(temp_file)
    
    # Save metadata as JSON
    metadata_file = "/tmp/model_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    metadata_blob = bucket.blob(os.path.join(blob_prefix, "model_metadata.json"))
    metadata_blob.upload_from_filename(metadata_file)
    
    # Clean up
    os.remove(temp_file)
    os.remove(metadata_file)
    
    logger.info("Model saved successfully")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-id', type=str, required=True)
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--output-path', type=str, required=True)
    parser.add_argument('--model-type', type=str, default='xgboost',
                       choices=['xgboost', 'lightgbm', 'random_forest'])
    
    args = parser.parse_args()
    
    logger.info(f"Starting training job with model type: {args.model_type}")
    
    # Download and preprocess data
    df = download_data_from_gcs(args.input_path)
    X, y, label_encoder, scaler = preprocess_data(df)
    
    # Train model
    model, accuracy = train_model(X, y, args.model_type)
    
    # Prepare metadata
    metadata = {
        'model_type': args.model_type,
        'accuracy': float(accuracy),
        'num_samples': len(df),
        'num_features': X.shape[1],
        'label_classes': label_encoder.classes_.tolist(),
        'training_date': pd.Timestamp.now().isoformat()
    }
    
    # Save model
    save_model_to_gcs(model, scaler, label_encoder, args.output_path, metadata)
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()