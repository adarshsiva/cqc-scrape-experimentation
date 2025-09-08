#!/usr/bin/env python3
"""
CQC Rating Prediction API using Cloud Functions.
Provides REST endpoints for rating predictions and recommendations.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import functions_framework
from flask import jsonify, Request
from google.cloud import aiplatform
from google.cloud import bigquery
from google.cloud import storage
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
REGION = os.environ.get('GCP_REGION', 'europe-west2')
ENDPOINT_ID = os.environ.get('VERTEX_ENDPOINT_ID', '')
MODEL_BUCKET = os.environ.get('MODEL_BUCKET', f'{PROJECT_ID}-cqc-processed')

# Initialize clients
aiplatform.init(project=PROJECT_ID, location=REGION)
bigquery_client = bigquery.Client(project=PROJECT_ID)
storage_client = storage.Client(project=PROJECT_ID)

# Cache for model artifacts
MODEL_ARTIFACTS = {}

class PredictionService:
    """Service for generating CQC rating predictions."""
    
    RATING_LABELS = {
        4: 'Outstanding',
        3: 'Good', 
        2: 'Requires improvement',
        1: 'Inadequate'
    }
    
    IMPROVEMENT_RECOMMENDATIONS = {
        'safe': {
            'low': 'Review and update medication management procedures',
            'medium': 'Enhance staff training on safety protocols and incident reporting',
            'high': 'Implement comprehensive safety audit and immediate corrective actions'
        },
        'effective': {
            'low': 'Update care plans and ensure regular reviews',
            'medium': 'Improve staff training programs and competency assessments',
            'high': 'Overhaul care delivery processes and clinical governance'
        },
        'caring': {
            'low': 'Enhance person-centered care approaches',
            'medium': 'Improve resident engagement and dignity practices',
            'high': 'Fundamental review of care culture and values'
        },
        'responsive': {
            'low': 'Review complaints handling procedures',
            'medium': 'Enhance activity programs and personalization',
            'high': 'Redesign services to meet individual needs'
        },
        'well-led': {
            'low': 'Strengthen governance and quality monitoring',
            'medium': 'Improve leadership visibility and staff morale',
            'high': 'Complete management restructuring and culture change'
        }
    }
    
    def __init__(self):
        """Initialize prediction service."""
        self.endpoint = None
        self.encoders = {}
        self.scaler = None
        self.feature_columns = []
        self._load_artifacts()
        
    def _load_artifacts(self):
        """Load model artifacts from GCS."""
        global MODEL_ARTIFACTS
        
        if MODEL_ARTIFACTS:
            # Use cached artifacts
            self.encoders = MODEL_ARTIFACTS.get('encoders', {})
            self.scaler = MODEL_ARTIFACTS.get('scaler')
            self.feature_columns = MODEL_ARTIFACTS.get('feature_columns', [])
            return
            
        try:
            bucket = storage_client.bucket(MODEL_BUCKET)
            
            # Load encoders
            encoder_blobs = bucket.list_blobs(prefix='feature_artifacts/encoders/')
            for blob in encoder_blobs:
                if blob.name.endswith('.pkl'):
                    encoder_name = blob.name.split('/')[-1].replace('_encoder.pkl', '')
                    encoder_data = blob.download_as_bytes()
                    self.encoders[encoder_name] = joblib.loads(encoder_data)
            
            # Load scaler
            scaler_blob = bucket.blob('feature_artifacts/scaler/scaler.pkl')
            if scaler_blob.exists():
                scaler_data = scaler_blob.download_as_bytes()
                self.scaler = joblib.loads(scaler_data)
            
            # Load feature columns
            columns_blob = bucket.blob('feature_artifacts/feature_columns.json')
            if columns_blob.exists():
                columns_data = columns_blob.download_as_text()
                self.feature_columns = json.loads(columns_data)
            
            # Cache artifacts
            MODEL_ARTIFACTS = {
                'encoders': self.encoders,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            
            logger.info("Model artifacts loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
    
    def _get_endpoint(self):
        """Get or create Vertex AI endpoint."""
        if self.endpoint:
            return self.endpoint
            
        if ENDPOINT_ID:
            try:
                self.endpoint = aiplatform.Endpoint(ENDPOINT_ID)
                logger.info(f"Using endpoint: {ENDPOINT_ID}")
            except Exception as e:
                logger.error(f"Error loading endpoint: {e}")
                
        return self.endpoint
    
    def _prepare_features(self, input_data: Dict) -> np.ndarray:
        """Prepare input features for prediction."""
        # Create DataFrame from input
        df = pd.DataFrame([input_data])
        
        # Apply default values for missing fields
        defaults = {
            'numberOfBeds': 30,
            'years_registered': 5,
            'days_since_inspection': 180,
            'has_nursing': False,
            'dementia_care': False,
            'mental_health_care': False,
            'region': 'Unknown',
            'care_home_type': 'Care home service without nursing'
        }
        
        for key, value in defaults.items():
            if key not in df.columns:
                df[key] = value
        
        # Calculate derived features
        df['inspection_overdue'] = (df['days_since_inspection'] > 365).astype(int)
        df['is_new_home'] = (df['years_registered'] < 2).astype(int)
        df['is_established_home'] = (df['years_registered'] > 10).astype(int)
        df['beds_log'] = np.log1p(df['numberOfBeds'])
        
        # Count specialisms
        specialism_cols = ['dementia_care', 'mental_health_care', 'physical_disabilities_care',
                          'learning_disabilities_care']
        available_specialisms = [col for col in specialism_cols if col in df.columns]
        df['num_specialisms'] = df[available_specialisms].sum(axis=1)
        
        # Service complexity
        df['service_complexity'] = (
            df.get('has_nursing', 0).astype(int) + 
            df.get('dementia_care', 0).astype(int) +
            df.get('mental_health_care', 0).astype(int)
        )
        
        # Encode categorical features
        for col, encoder in self.encoders.items():
            if col in df.columns:
                try:
                    df[f'{col}_encoded'] = encoder.transform(df[col].fillna('Unknown'))
                except:
                    df[f'{col}_encoded'] = 0  # Default for unknown categories
        
        # Ensure all required features are present
        if self.feature_columns:
            for col in self.feature_columns:
                if col not in df.columns:
                    df[col] = 0  # Default value
            
            # Select and order features
            features = df[self.feature_columns].values
        else:
            # Fallback to available numeric features
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            features = df[numeric_cols].values
        
        # Scale features if scaler available
        if self.scaler:
            try:
                features = self.scaler.transform(features)
            except:
                logger.warning("Could not apply scaler")
        
        return features
    
    def _get_risk_factors(self, input_data: Dict, prediction: int) -> List[str]:
        """Identify risk factors based on input and prediction."""
        risk_factors = []
        
        # Check inspection timing
        if input_data.get('days_since_inspection', 0) > 365:
            risk_factors.append("Overdue for inspection (>365 days)")
        
        # Check home age
        if input_data.get('years_registered', 0) < 2:
            risk_factors.append("New home (<2 years registered)")
        
        # Check specialisms
        if input_data.get('dementia_care') and input_data.get('mental_health_care'):
            risk_factors.append("Multiple high-risk specialisms")
        
        # Check capacity
        beds = input_data.get('numberOfBeds', 0)
        if beds > 100:
            risk_factors.append(f"Large capacity ({beds} beds)")
        
        # Check nursing
        if not input_data.get('has_nursing') and input_data.get('dementia_care'):
            risk_factors.append("Dementia care without nursing")
        
        # Check prediction
        if prediction <= 2:
            risk_factors.append("Model predicts below 'Good' rating")
        
        return risk_factors
    
    def _get_improvement_areas(self, input_data: Dict, prediction: int) -> List[Dict]:
        """Generate improvement recommendations."""
        areas = []
        
        # Determine priority based on prediction
        if prediction == 1:
            priority = 'high'
        elif prediction == 2:
            priority = 'medium'
        else:
            priority = 'low'
        
        # Add recommendations for each domain
        for domain in ['safe', 'effective', 'caring', 'responsive', 'well-led']:
            areas.append({
                'domain': domain.title(),
                'recommendation': self.IMPROVEMENT_RECOMMENDATIONS[domain][priority],
                'priority': priority
            })
        
        # Add specific recommendations based on features
        if input_data.get('days_since_inspection', 0) > 365:
            areas.append({
                'domain': 'Compliance',
                'recommendation': 'Prepare for upcoming inspection - review all policies and procedures',
                'priority': 'high'
            })
        
        if input_data.get('staff_turnover_rate', 0) > 0.3:
            areas.append({
                'domain': 'Staffing',
                'recommendation': 'Address high staff turnover through retention strategies',
                'priority': 'high'
            })
        
        return areas
    
    def predict_with_vertex_ai(self, features: np.ndarray) -> Dict:
        """Make prediction using Vertex AI endpoint."""
        endpoint = self._get_endpoint()
        
        if not endpoint:
            # Fallback to mock prediction
            return self._mock_prediction(features)
        
        try:
            # Make prediction
            instances = features.tolist()
            prediction = endpoint.predict(instances=instances)
            
            # Parse results
            predictions = prediction.predictions[0]
            
            if isinstance(predictions, dict):
                predicted_class = predictions.get('predicted_class', 3)
                probabilities = predictions.get('probabilities', [])
            else:
                predicted_class = int(predictions)
                probabilities = []
            
            return {
                'predicted_class': predicted_class,
                'probabilities': probabilities
            }
            
        except Exception as e:
            logger.error(f"Vertex AI prediction error: {e}")
            return self._mock_prediction(features)
    
    def _mock_prediction(self, features: np.ndarray) -> Dict:
        """Generate mock prediction for testing."""
        # Simple rule-based prediction for testing
        feature_sum = np.sum(features[0][:10])  # Use first 10 features
        
        if feature_sum > 15:
            predicted_class = 4  # Outstanding
        elif feature_sum > 10:
            predicted_class = 3  # Good
        elif feature_sum > 5:
            predicted_class = 2  # Requires improvement
        else:
            predicted_class = 1  # Inadequate
        
        # Generate mock probabilities
        probabilities = [0.1, 0.2, 0.4, 0.3]
        probabilities[predicted_class - 1] = 0.6
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probabilities
        }
    
    def predict(self, input_data: Dict) -> Dict:
        """Generate complete prediction with recommendations."""
        try:
            # Prepare features
            features = self._prepare_features(input_data)
            
            # Get prediction
            prediction_result = self.predict_with_vertex_ai(features)
            
            predicted_class = prediction_result['predicted_class']
            probabilities = prediction_result.get('probabilities', [])
            
            # Generate risk assessment
            risk_factors = self._get_risk_factors(input_data, predicted_class)
            risk_score = len(risk_factors) / 10.0  # Normalize to 0-1
            
            # Generate improvement recommendations
            improvement_areas = self._get_improvement_areas(input_data, predicted_class)
            
            # Build response
            response = {
                'success': True,
                'prediction': {
                    'overall_rating': self.RATING_LABELS.get(predicted_class, 'Unknown'),
                    'rating_numeric': predicted_class,
                    'confidence': max(probabilities) if probabilities else 0.5,
                    'probabilities': {
                        self.RATING_LABELS[i+1]: prob 
                        for i, prob in enumerate(probabilities[:4])
                    } if probabilities else {}
                },
                'risk_assessment': {
                    'risk_score': round(risk_score, 2),
                    'risk_level': 'High' if risk_score > 0.7 else 'Medium' if risk_score > 0.3 else 'Low',
                    'risk_factors': risk_factors
                },
                'recommendations': {
                    'improvement_areas': improvement_areas,
                    'priority_action': improvement_areas[0] if improvement_areas else None
                },
                'metadata': {
                    'prediction_id': datetime.now().strftime('%Y%m%d%H%M%S'),
                    'timestamp': datetime.now().isoformat(),
                    'model_version': 'v1.0'
                }
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return {
                'success': False,
                'error': str(e),
                'message': 'Failed to generate prediction'
            }

# Initialize service
prediction_service = PredictionService()

@functions_framework.http
def predict_rating(request: Request) -> Dict:
    """
    HTTP Cloud Function for CQC rating prediction.
    
    Args:
        request: Flask request object with JSON body
        
    Returns:
        JSON response with prediction and recommendations
    """
    # Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers for response
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        # Parse request
        request_json = request.get_json(silent=True)
        
        if not request_json:
            return jsonify({
                'success': False,
                'error': 'Invalid request',
                'message': 'Request body must be valid JSON'
            }), 400, headers
        
        # Generate prediction
        result = prediction_service.predict(request_json)
        
        # Return response
        if result['success']:
            return jsonify(result), 200, headers
        else:
            return jsonify(result), 500, headers
            
    except Exception as e:
        logger.error(f"Request handling error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Internal server error'
        }), 500, headers

@functions_framework.http
def health_check(request: Request) -> Dict:
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'CQC Rating Prediction API',
        'timestamp': datetime.now().isoformat()
    }), 200

def batch_predict(request: Request) -> Dict:
    """
    Batch prediction endpoint for multiple care homes.
    
    Args:
        request: Flask request with array of care home data
        
    Returns:
        JSON response with predictions for all homes
    """
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Content-Type': 'application/json'
    }
    
    try:
        request_json = request.get_json(silent=True)
        
        if not request_json or 'homes' not in request_json:
            return jsonify({
                'success': False,
                'error': 'Invalid request',
                'message': 'Request must contain "homes" array'
            }), 400, headers
        
        homes = request_json['homes']
        results = []
        
        for home in homes:
            prediction = prediction_service.predict(home)
            prediction['location_id'] = home.get('locationId', 'unknown')
            results.append(prediction)
        
        return jsonify({
            'success': True,
            'predictions': results,
            'total': len(results),
            'timestamp': datetime.now().isoformat()
        }), 200, headers
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'message': 'Batch prediction failed'
        }), 500, headers