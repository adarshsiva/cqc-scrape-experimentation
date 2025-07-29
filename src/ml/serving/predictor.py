import os
import pickle
import numpy as np
from google.cloud import storage
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CQCPredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = None
        
    def load(self, model_path):
        """Load model from GCS path"""
        logger.info(f"Loading model from {model_path}")
        
        # Parse GCS path
        if model_path.startswith("gs://"):
            path_parts = model_path[5:].split("/", 1)
            bucket_name = path_parts[0]
            blob_path = path_parts[1]
        else:
            raise ValueError("Model path must start with gs://")
        
        # Download model
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)
        
        temp_file = "/tmp/model.pkl"
        blob.download_to_filename(temp_file)
        
        # Load model package
        with open(temp_file, 'rb') as f:
            model_package = pickle.load(f)
        
        self.model = model_package['model']
        self.scaler = model_package['scaler']
        self.label_encoder = model_package['label_encoder']
        self.feature_columns = model_package['feature_columns']
        
        logger.info("Model loaded successfully")
        
    def predict(self, instances):
        """Make predictions on instances"""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        predictions = []
        
        for instance in instances:
            # Extract features in correct order
            features = []
            for col in self.feature_columns:
                if col in instance:
                    features.append(instance[col])
                else:
                    # Handle encoded features
                    if col == 'ownership_type_encoded':
                        # Simple encoding for demo
                        ownership_map = {'Individual': 0, 'Organisation': 1, 'Social Care Org': 2}
                        features.append(ownership_map.get(instance.get('ownership_type', ''), 0))
                    elif col == 'region_encoded':
                        # Simple hash encoding
                        features.append(hash(instance.get('region', '')) % 10)
                    else:
                        features.append(0)  # Default value
            
            # Scale features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Get prediction
            pred_class = self.model.predict(features_scaled)[0]
            pred_proba = self.model.predict_proba(features_scaled)[0]
            
            # Format response
            prediction = {
                'class': int(pred_class),
                'label': self.label_encoder.inverse_transform([pred_class])[0],
                'scores': pred_proba.tolist()
            }
            predictions.append(prediction)
        
        return predictions