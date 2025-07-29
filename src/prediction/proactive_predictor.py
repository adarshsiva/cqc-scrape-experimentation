#!/usr/bin/env python3
"""
Proactive Risk Assessment API for CQC Ratings
Identifies locations at risk of rating downgrades
"""

import os
import json
import logging
import pickle
from datetime import datetime
from typing import Dict, List, Any, Tuple

from flask import Flask, request, jsonify
from google.cloud import storage
import pandas as pd
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
PROJECT_ID = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
BUCKET_NAME = os.environ.get('MODEL_BUCKET', 'machine-learning-exp-467008-cqc-ml-artifacts')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/proactive/model_package.pkl')

# Global model package
MODEL_PACKAGE = None


def load_model() -> Dict[str, Any]:
    """Load the trained model package from Google Cloud Storage"""
    try:
        logger.info(f"Loading model from gs://{BUCKET_NAME}/{MODEL_PATH}")
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(MODEL_PATH)
        
        # Download to memory
        model_bytes = blob.download_as_bytes()
        model_package = pickle.loads(model_bytes)
        
        logger.info("Model loaded successfully")
        return model_package
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise


def analyze_risk_factors(features: pd.DataFrame, model: Any) -> List[Dict[str, float]]:
    """Analyze and rank risk factors based on feature importance"""
    try:
        # Get feature importance from the model
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_names = features.columns.tolist()
            
            # Create importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Get feature values for the instance
            feature_values = features.iloc[0].to_dict()
            
            # Calculate risk contribution for each feature
            risk_factors = []
            for feature, importance in importance_dict.items():
                if importance > 0 and feature in feature_values:
                    value = feature_values[feature]
                    
                    # Determine if this feature contributes to risk
                    # (This is a simplified approach - you might want more sophisticated logic)
                    risk_contribution = importance
                    
                    # For certain features, adjust risk based on value
                    if 'complaints' in feature.lower() and value > 0:
                        risk_contribution *= (1 + value * 0.1)
                    elif 'staff_' in feature.lower() and value < 0.8:
                        risk_contribution *= (2 - value)
                    elif 'inspection_days_since' in feature.lower() and value > 365:
                        risk_contribution *= (value / 365)
                    
                    risk_factors.append({
                        'feature': feature,
                        'value': float(value),
                        'risk_contribution': float(risk_contribution),
                        'importance': float(importance)
                    })
            
            # Sort by risk contribution
            risk_factors.sort(key=lambda x: x['risk_contribution'], reverse=True)
            return risk_factors[:10]  # Return top 10 risk factors
            
    except Exception as e:
        logger.error(f"Error analyzing risk factors: {str(e)}")
        return []


def generate_recommendations(risk_factors: List[Dict[str, float]]) -> List[str]:
    """Generate actionable recommendations based on risk factors"""
    recommendations = []
    
    for factor in risk_factors[:5]:  # Top 5 risk factors
        feature = factor['feature']
        value = factor['value']
        
        # Generate specific recommendations based on feature type
        if 'complaints' in feature.lower():
            if value > 0:
                recommendations.append(
                    f"Address complaint trends: {int(value)} recent complaints detected. "
                    "Implement complaint resolution process and root cause analysis."
                )
        
        elif 'staff_vacancy' in feature.lower():
            if value > 0.2:
                recommendations.append(
                    f"Reduce staff vacancy rate: Currently at {value*100:.1f}%. "
                    "Focus on recruitment and retention strategies."
                )
        
        elif 'staff_turnover' in feature.lower():
            if value > 0.3:
                recommendations.append(
                    f"Address staff turnover: Rate at {value*100:.1f}%. "
                    "Investigate causes and improve working conditions."
                )
        
        elif 'inspection_days_since' in feature.lower():
            if value > 365:
                recommendations.append(
                    f"Prepare for inspection: {int(value)} days since last inspection. "
                    "Conduct internal audits and address any compliance gaps."
                )
        
        elif 'safe_key_questions' in feature.lower() and 'yes_ratio' in feature.lower():
            if value < 0.8:
                recommendations.append(
                    f"Improve safety metrics: Safety compliance at {value*100:.1f}%. "
                    "Review and enhance safety protocols and training."
                )
        
        elif 'effective_key_questions' in feature.lower() and 'yes_ratio' in feature.lower():
            if value < 0.8:
                recommendations.append(
                    f"Enhance effectiveness measures: Currently at {value*100:.1f}%. "
                    "Review care plans and outcome monitoring processes."
                )
        
        elif 'caring_key_questions' in feature.lower() and 'yes_ratio' in feature.lower():
            if value < 0.8:
                recommendations.append(
                    f"Improve caring standards: Score at {value*100:.1f}%. "
                    "Focus on person-centered care and dignity training."
                )
        
        elif 'responsive_key_questions' in feature.lower() and 'yes_ratio' in feature.lower():
            if value < 0.8:
                recommendations.append(
                    f"Enhance responsiveness: Currently at {value*100:.1f}%. "
                    "Review how services meet individual needs and preferences."
                )
        
        elif 'well_led_key_questions' in feature.lower() and 'yes_ratio' in feature.lower():
            if value < 0.8:
                recommendations.append(
                    f"Strengthen leadership: Leadership score at {value*100:.1f}%. "
                    "Invest in management training and governance structures."
                )
    
    # Add general recommendations if list is short
    if len(recommendations) < 3:
        recommendations.extend([
            "Conduct comprehensive internal audit against CQC standards",
            "Implement continuous improvement program with regular monitoring",
            "Ensure all staff complete mandatory training and competency assessments"
        ])
    
    return recommendations[:5]  # Return top 5 recommendations


def assess_single_location(location_data: Dict[str, Any]) -> Dict[str, Any]:
    """Assess risk for a single location"""
    try:
        # Create DataFrame from location data
        features = pd.DataFrame([location_data])
        
        # Get the ensemble prediction function from model package
        if 'ensemble_predict' in MODEL_PACKAGE:
            ensemble_predict = MODEL_PACKAGE['ensemble_predict']
        else:
            # Fallback to using individual models
            def ensemble_predict(X):
                predictions = []
                for name, model in MODEL_PACKAGE['models'].items():
                    pred = model.predict_proba(X)[:, 1]  # Probability of being at-risk
                    predictions.append(pred)
                return np.mean(predictions, axis=0)
        
        # Predict risk score (probability of rating downgrade)
        risk_score = float(ensemble_predict(features)[0] * 100)
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = "HIGH"
        elif risk_score >= 40:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Get primary model for feature importance (prefer XGBoost)
        primary_model = MODEL_PACKAGE['models'].get('xgboost', 
                        MODEL_PACKAGE['models'].get('lightgbm',
                        list(MODEL_PACKAGE['models'].values())[0]))
        
        # Analyze risk factors
        risk_factors = analyze_risk_factors(features, primary_model)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_factors)
        
        # Format response
        response = {
            'locationId': location_data.get('locationId', 'unknown'),
            'locationName': location_data.get('locationName', 'Unknown Location'),
            'riskScore': risk_score,
            'riskLevel': risk_level,
            'topRiskFactors': [
                {
                    'factor': rf['feature'],
                    'currentValue': rf['value'],
                    'riskContribution': rf['risk_contribution']
                }
                for rf in risk_factors[:5]
            ],
            'recommendations': recommendations,
            'assessmentDate': datetime.now().isoformat(),
            'confidence': calculate_confidence(risk_score, risk_factors)
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Error assessing location: {str(e)}")
        raise


def calculate_confidence(risk_score: float, risk_factors: List[Dict]) -> str:
    """Calculate confidence level of the assessment"""
    # Simple confidence calculation based on risk score extremity and factor clarity
    if risk_score >= 80 or risk_score <= 20:
        confidence = "HIGH"
    elif risk_score >= 60 or risk_score <= 40:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    # Adjust based on risk factor concentration
    if len(risk_factors) > 0:
        top_factor_contribution = risk_factors[0]['risk_contribution']
        total_contribution = sum(rf['risk_contribution'] for rf in risk_factors[:5])
        
        if top_factor_contribution / total_contribution > 0.5:
            # One factor dominates - high confidence
            confidence = "HIGH"
    
    return confidence


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'proactive-risk-assessment',
        'modelLoaded': MODEL_PACKAGE is not None,
        'timestamp': datetime.utcnow().isoformat()
    }), 200


@app.route('/assess-risk', methods=['POST'])
def assess_risk():
    """Assess risk for a single location"""
    try:
        # Get request data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Assess the location
        assessment = assess_single_location(data)
        
        # Log the assessment
        logger.info(f"Risk assessment completed for location {assessment['locationId']}: "
                   f"Risk Level = {assessment['riskLevel']}, Score = {assessment['riskScore']:.1f}")
        
        return jsonify(assessment), 200
        
    except Exception as e:
        logger.error(f"Error in risk assessment: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to assess risk', 'details': str(e)}), 500


@app.route('/batch-assess', methods=['POST'])
def batch_assess():
    """Assess risk for multiple locations"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'locations' not in data:
            return jsonify({'error': 'No locations provided'}), 400
        
        locations = data['locations']
        if not isinstance(locations, list):
            return jsonify({'error': 'Locations must be a list'}), 400
        
        if len(locations) > 100:
            return jsonify({'error': 'Batch size limited to 100 locations'}), 400
        
        # Process each location
        results = []
        errors = []
        
        for i, location in enumerate(locations):
            try:
                assessment = assess_single_location(location)
                results.append(assessment)
            except Exception as e:
                errors.append({
                    'index': i,
                    'locationId': location.get('locationId', 'unknown'),
                    'error': str(e)
                })
        
        # Prepare response
        response = {
            'assessments': results,
            'summary': {
                'total': len(locations),
                'assessed': len(results),
                'failed': len(errors),
                'highRisk': sum(1 for r in results if r['riskLevel'] == 'HIGH'),
                'mediumRisk': sum(1 for r in results if r['riskLevel'] == 'MEDIUM'),
                'lowRisk': sum(1 for r in results if r['riskLevel'] == 'LOW')
            },
            'errors': errors if errors else None,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        logger.info(f"Batch assessment completed: {len(results)} successful, {len(errors)} failed")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error in batch assessment: {str(e)}", exc_info=True)
        return jsonify({'error': 'Failed to process batch', 'details': str(e)}), 500


@app.route('/risk-thresholds', methods=['GET'])
def get_risk_thresholds():
    """Get current risk threshold definitions"""
    return jsonify({
        'thresholds': {
            'HIGH': {'min': 70, 'max': 100, 'description': 'Immediate attention required'},
            'MEDIUM': {'min': 40, 'max': 69, 'description': 'Monitor closely and implement improvements'},
            'LOW': {'min': 0, 'max': 39, 'description': 'Continue current practices'}
        },
        'confidenceLevels': {
            'HIGH': 'Strong indicators present',
            'MEDIUM': 'Moderate certainty',
            'LOW': 'Limited data or mixed signals'
        }
    }), 200


# Initialize model on startup
try:
    MODEL_PACKAGE = load_model()
except Exception as e:
    logger.error(f"Failed to load model on startup: {str(e)}")
    # Continue running but model will need to be loaded on first request


if __name__ == '__main__':
    # Run the Flask app
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)