import json
import logging
import os
from typing import Dict, List, Union, Any
from datetime import datetime

import functions_framework
from google.cloud import aiplatform
from google.cloud import logging as cloud_logging
import numpy as np

# Initialize logging
cloud_client = cloud_logging.Client()
cloud_client.setup_logging()
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get('GCP_PROJECT')
REGION = os.environ.get('GCP_REGION', 'europe-west2')
ENDPOINT_ID = os.environ.get('VERTEX_ENDPOINT_ID')
MODEL_NAME = os.environ.get('MODEL_NAME', 'cqc-rating-predictor')

# Initialize Vertex AI
aiplatform.init(project=PROJECT_ID, location=REGION)

# Feature schema for validation
REQUIRED_FEATURES = {
    'number_of_beds': int,
    'number_of_locations': int,
    'inspection_history_length': int,
    'days_since_last_inspection': int,
    'ownership_type': str,
    'service_types': list,
    'specialisms': list,
    'region': str,
    'local_authority': str,
    'constituency': str,
    'regulated_activities': list,
    'service_user_groups': list,
    'has_previous_rating': bool,
    'previous_rating': str,
    'ownership_changed_recently': bool,
    'nominated_individual_exists': bool
}

# Rating mappings
RATING_LABELS = {
    0: 'Inadequate',
    1: 'Requires improvement',
    2: 'Good',
    3: 'Outstanding'
}

RATING_TO_INDEX = {v: k for k, v in RATING_LABELS.items()}


class PredictionError(Exception):
    """Custom exception for prediction errors"""
    pass


def validate_input(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate input data against expected schema
    
    Args:
        data: Input data dictionary
        
    Returns:
        Validated and cleaned data
        
    Raises:
        ValueError: If validation fails
    """
    validated = {}
    errors = []
    
    for feature, expected_type in REQUIRED_FEATURES.items():
        if feature not in data:
            errors.append(f"Missing required feature: {feature}")
            continue
            
        value = data[feature]
        
        # Type validation and conversion
        try:
            if expected_type == int:
                validated[feature] = int(value)
            elif expected_type == str:
                validated[feature] = str(value)
            elif expected_type == bool:
                validated[feature] = bool(value)
            elif expected_type == list:
                if not isinstance(value, list):
                    errors.append(f"{feature} must be a list")
                else:
                    validated[feature] = value
            else:
                validated[feature] = value
        except (ValueError, TypeError) as e:
            errors.append(f"Invalid type for {feature}: expected {expected_type.__name__}, got {type(value).__name__}")
    
    if errors:
        raise ValueError(f"Validation errors: {'; '.join(errors)}")
    
    return validated


def transform_features(data: Dict[str, Any]) -> List[float]:
    """
    Transform validated input data into feature vector for model
    
    Args:
        data: Validated input data
        
    Returns:
        Feature vector as list of floats
    """
    features = []
    
    # Numerical features (direct)
    features.append(float(data['number_of_beds']))
    features.append(float(data['number_of_locations']))
    features.append(float(data['inspection_history_length']))
    features.append(float(data['days_since_last_inspection']))
    
    # One-hot encode ownership type
    ownership_types = ['Individual', 'Organisation', 'Partnership']
    for ot in ownership_types:
        features.append(1.0 if data['ownership_type'] == ot else 0.0)
    
    # Count features for lists
    features.append(float(len(data['service_types'])))
    features.append(float(len(data['specialisms'])))
    features.append(float(len(data['regulated_activities'])))
    features.append(float(len(data['service_user_groups'])))
    
    # Boolean features
    features.append(1.0 if data['has_previous_rating'] else 0.0)
    features.append(1.0 if data['ownership_changed_recently'] else 0.0)
    features.append(1.0 if data['nominated_individual_exists'] else 0.0)
    
    # Previous rating encoding (if exists)
    if data['has_previous_rating'] and data['previous_rating']:
        rating_idx = RATING_TO_INDEX.get(data['previous_rating'], -1)
        features.append(float(rating_idx))
    else:
        features.append(-1.0)  # No previous rating
    
    # Regional encoding (simplified - in production, use proper encoding)
    # For now, using hash function for demonstration
    region_hash = hash(data['region']) % 10
    features.append(float(region_hash))
    
    local_auth_hash = hash(data['local_authority']) % 20
    features.append(float(local_auth_hash))
    
    constituency_hash = hash(data['constituency']) % 50
    features.append(float(constituency_hash))
    
    return features


def get_prediction(features: List[List[float]], endpoint_id: str) -> Dict[str, Any]:
    """
    Get prediction from Vertex AI endpoint
    
    Args:
        features: Feature vectors for prediction
        endpoint_id: Vertex AI endpoint ID
        
    Returns:
        Prediction results
    """
    try:
        endpoint = aiplatform.Endpoint(endpoint_id)
        
        # Make prediction
        prediction = endpoint.predict(instances=features)
        
        return {
            'predictions': prediction.predictions,
            'deployed_model_id': prediction.deployed_model_id,
            'model_version_id': prediction.model_version_id if hasattr(prediction, 'model_version_id') else None
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise PredictionError(f"Failed to get prediction: {str(e)}")


def format_prediction_response(predictions: List[Any], input_data: List[Dict]) -> List[Dict]:
    """
    Format prediction results for response
    
    Args:
        predictions: Raw predictions from model
        input_data: Original input data for reference
        
    Returns:
        Formatted prediction results
    """
    results = []
    
    for i, pred in enumerate(predictions):
        # Handle different prediction formats
        if isinstance(pred, dict) and 'scores' in pred:
            # Multi-class probability output
            scores = pred['scores']
            predicted_class = np.argmax(scores)
            confidence_scores = {
                RATING_LABELS[j]: float(score) 
                for j, score in enumerate(scores)
            }
        else:
            # Single class prediction
            predicted_class = int(pred)
            confidence_scores = {RATING_LABELS[predicted_class]: 1.0}
        
        result = {
            'location_id': input_data[i].get('location_id', f'location_{i}'),
            'predicted_rating': RATING_LABELS.get(predicted_class, 'Unknown'),
            'confidence_scores': confidence_scores,
            'prediction_timestamp': datetime.utcnow().isoformat(),
            'model_version': MODEL_NAME
        }
        
        # Add confidence level
        max_confidence = max(confidence_scores.values())
        if max_confidence >= 0.8:
            result['confidence_level'] = 'High'
        elif max_confidence >= 0.6:
            result['confidence_level'] = 'Medium'
        else:
            result['confidence_level'] = 'Low'
        
        results.append(result)
    
    return results


@functions_framework.http
def predict(request):
    """
    HTTP Cloud Function for CQC rating prediction
    
    Args:
        request: Flask Request object
        
    Returns:
        Flask Response object
    """
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'POST, OPTIONS',
        'Access-Control-Allow-Headers': 'Content-Type',
    }
    
    # Handle preflight request
    if request.method == 'OPTIONS':
        return ('', 204, headers)
    
    # Validate request method
    if request.method != 'POST':
        return json.dumps({'error': 'Only POST method is allowed'}), 405, headers
    
    # Validate endpoint configuration
    if not ENDPOINT_ID:
        logger.error("VERTEX_ENDPOINT_ID environment variable not set")
        return json.dumps({'error': 'Service not configured properly'}), 500, headers
    
    try:
        # Parse request data
        request_json = request.get_json()
        if not request_json:
            return json.dumps({'error': 'No data provided'}), 400, headers
        
        # Determine if batch or single prediction
        is_batch = isinstance(request_json, dict) and 'instances' in request_json
        
        if is_batch:
            # Batch prediction
            instances = request_json['instances']
            if not isinstance(instances, list):
                return json.dumps({'error': 'instances must be a list'}), 400, headers
            
            if len(instances) > 100:  # Limit batch size
                return json.dumps({'error': 'Batch size limited to 100 instances'}), 400, headers
            
            # Validate and transform all instances
            validated_instances = []
            feature_vectors = []
            
            for i, instance in enumerate(instances):
                try:
                    validated = validate_input(instance)
                    validated_instances.append(validated)
                    features = transform_features(validated)
                    feature_vectors.append(features)
                except ValueError as e:
                    return json.dumps({
                        'error': f'Validation error in instance {i}: {str(e)}'
                    }), 400, headers
            
            # Get predictions
            prediction_result = get_prediction(feature_vectors, ENDPOINT_ID)
            
            # Format response
            predictions = format_prediction_response(
                prediction_result['predictions'], 
                instances
            )
            
            response = {
                'predictions': predictions,
                'metadata': {
                    'batch_size': len(instances),
                    'model_info': {
                        'deployed_model_id': prediction_result['deployed_model_id'],
                        'model_version': MODEL_NAME
                    },
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
        else:
            # Single prediction
            try:
                validated = validate_input(request_json)
                features = transform_features(validated)
            except ValueError as e:
                return json.dumps({'error': f'Validation error: {str(e)}'}), 400, headers
            
            # Get prediction
            prediction_result = get_prediction([features], ENDPOINT_ID)
            
            # Format response
            predictions = format_prediction_response(
                prediction_result['predictions'], 
                [request_json]
            )
            
            response = predictions[0]  # Return single prediction
        
        # Log successful prediction
        logger.info(f"Successful prediction request: {len(instances) if is_batch else 1} instance(s)")
        
        return json.dumps(response), 200, headers
        
    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        return json.dumps({'error': str(e)}), 500, headers
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return json.dumps({'error': 'Internal server error'}), 500, headers


# Health check endpoint
@functions_framework.http
def health(request):
    """
    Health check endpoint
    
    Args:
        request: Flask Request object
        
    Returns:
        Flask Response object
    """
    return json.dumps({
        'status': 'healthy',
        'service': 'cqc-prediction-api',
        'timestamp': datetime.utcnow().isoformat()
    }), 200