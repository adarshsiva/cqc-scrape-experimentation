#!/usr/bin/env python3
"""
Simple local trainer for quick testing of the ML pipeline
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from google.cloud import storage
import os

def main():
    print("Starting simple ML training...")
    
    # Download training data
    client = storage.Client()
    bucket = client.bucket("machine-learning-exp-467008-cqc-ml-artifacts")
    blob = bucket.blob("training_data/features.csv")
    
    # Download to temp file
    temp_file = "/tmp/features.csv"
    blob.download_to_filename(temp_file)
    
    # Load data
    df = pd.read_csv(temp_file)
    print(f"Loaded {len(df)} samples")
    
    # Filter out samples without ratings
    df = df[df['rating_label'].notna()]
    print(f"Filtered to {len(df)} samples with ratings")
    
    # Prepare features
    feature_cols = [
        'number_of_beds', 'number_of_locations', 'inspection_history_length',
        'days_since_last_inspection', 'service_types_count', 'specialisms_count',
        'regulated_activities_count', 'service_user_groups_count'
    ]
    
    # Handle missing values
    df['days_since_last_inspection'] = df['days_since_last_inspection'].fillna(365)
    
    X = df[feature_cols]
    y = df['rating_label']
    
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train simple model
    model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel accuracy: {accuracy:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Save model locally
    model_data = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'feature_columns': feature_cols,
        'accuracy': accuracy
    }
    
    with open('/tmp/simple_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Upload to GCS
    output_blob = bucket.blob("models/simple/model.pkl")
    output_blob.upload_from_filename('/tmp/simple_model.pkl')
    
    print("\nModel saved to gs://machine-learning-exp-467008-cqc-ml-artifacts/models/simple/model.pkl")
    
    # Clean up
    os.remove(temp_file)
    os.remove('/tmp/simple_model.pkl')

if __name__ == "__main__":
    main()