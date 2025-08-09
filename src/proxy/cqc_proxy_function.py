"""
Cloud Function to act as a proxy for CQC API requests.
This function runs on Google Cloud Functions which may have different IP ranges
that aren't blocked by the CQC API.
"""

import functions_framework
import requests
import json
from flask import jsonify
import os
from google.cloud import secretmanager

def get_api_key():
    """Get API key from Secret Manager."""
    try:
        client = secretmanager.SecretManagerServiceClient()
        project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        name = f"projects/{project_id}/secrets/cqc-subscription-key/versions/latest"
        response = client.access_secret_version(request={"name": name})
        return response.payload.data.decode("UTF-8")
    except Exception:
        return os.environ.get('CQC_API_KEY', '')

@functions_framework.http
def cqc_proxy(request):
    """
    HTTP Cloud Function to proxy CQC API requests.
    
    Query parameters:
    - endpoint: The API endpoint to call (e.g., 'locations', 'providers')
    - page: Page number (optional)
    - perPage: Items per page (optional)
    - locationId: For specific location details (optional)
    """
    
    # Handle CORS
    if request.method == 'OPTIONS':
        headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600'
        }
        return ('', 204, headers)
    
    # Set CORS headers
    headers = {
        'Access-Control-Allow-Origin': '*'
    }
    
    try:
        # Get parameters from request
        endpoint = request.args.get('endpoint', 'locations')
        page = request.args.get('page', '1')
        per_page = request.args.get('perPage', '100')
        location_id = request.args.get('locationId')
        
        # Build URL
        base_url = "https://api.service.cqc.org.uk/public/v1"
        
        if location_id:
            url = f"{base_url}/locations/{location_id}"
            params = {}
        else:
            url = f"{base_url}/{endpoint}"
            params = {
                'page': page,
                'perPage': per_page
            }
        
        # Get API key
        api_key = get_api_key()
        if not api_key:
            return jsonify({
                'error': 'API key not configured',
                'message': 'Please set CQC_API_KEY in environment or Secret Manager'
            }), 500, headers
        
        # Make request to CQC API
        response = requests.get(
            url,
            params=params,
            headers={
                'Ocp-Apim-Subscription-Key': api_key,
                'User-Agent': 'CQC-Proxy-Function/1.0',
                'Accept': 'application/json'
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return jsonify(response.json()), 200, headers
        else:
            return jsonify({
                'error': f'CQC API returned {response.status_code}',
                'message': response.text
            }), response.status_code, headers
            
    except requests.exceptions.Timeout:
        return jsonify({
            'error': 'Request timeout',
            'message': 'CQC API request timed out'
        }), 504, headers
        
    except Exception as e:
        return jsonify({
            'error': 'Internal error',
            'message': str(e)
        }), 500, headers