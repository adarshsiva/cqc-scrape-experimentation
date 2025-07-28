import os
import json
import logging
from flask import Flask, request, jsonify
from datetime import datetime
from main import validate_input, transform_features, get_prediction, format_prediction_response, PredictionError

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get('GCP_PROJECT')
REGION = os.environ.get('GCP_REGION', 'europe-west2')
ENDPOINT_ID = os.environ.get('VERTEX_ENDPOINT_ID')
MODEL_NAME = os.environ.get('MODEL_NAME', 'cqc-rating-predictor')

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'cqc-prediction-api',
        'timestamp': datetime.utcnow().isoformat()
    }), 200

@app.route('/', methods=['POST', 'OPTIONS'])
def predict():
    """Main prediction endpoint"""
    # Handle CORS preflight
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response, 200
    
    # Validate endpoint configuration
    if not ENDPOINT_ID:
        logger.error("VERTEX_ENDPOINT_ID environment variable not set")
        return jsonify({'error': 'Service not configured properly'}), 500
    
    try:
        # Parse request data
        request_json = request.get_json()
        if not request_json:
            return jsonify({'error': 'No data provided'}), 400
        
        # Determine if batch or single prediction
        is_batch = isinstance(request_json, dict) and 'instances' in request_json
        
        if is_batch:
            # Batch prediction
            instances = request_json['instances']
            if not isinstance(instances, list):
                return jsonify({'error': 'instances must be a list'}), 400
            
            if len(instances) > 100:  # Limit batch size
                return jsonify({'error': 'Batch size limited to 100 instances'}), 400
            
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
                    return jsonify({
                        'error': f'Validation error in instance {i}: {str(e)}'
                    }), 400
            
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
                return jsonify({'error': f'Validation error: {str(e)}'}), 400
            
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
        
        return jsonify(response), 200
        
    except PredictionError as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)