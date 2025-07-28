import os
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional
import requests
from flask import Flask, jsonify, request
from google.cloud import storage
from google.cloud import secretmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class CQCAPIClient:
    """Client for interacting with CQC API."""
    
    BASE_URL = "https://api.cqc.org.uk/public/v1"
    
    def __init__(self, subscription_key: str, partner_code: str):
        self.subscription_key = subscription_key
        self.partner_code = partner_code
        self.session = requests.Session()
        # CQC API uses Azure API Management which expects this header format
        self.session.headers.update({
            'Ocp-Apim-Subscription-Key': self.subscription_key
        })
    
    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request with retry logic."""
        if params is None:
            params = {}
        if self.partner_code:  # Only add partner code if provided
            params['partnerCode'] = self.partner_code
        
        url = f"{self.BASE_URL}{endpoint}"
        
        for attempt in range(3):
            try:
                response = self.session.get(url, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt + 1}/3): {e}")
                if attempt == 2:
                    raise
    
    def fetch_providers(self, page: int = 1, per_page: int = 1000) -> Dict:
        """Fetch providers from CQC API."""
        params = {
            'page': page,
            'perPage': per_page
        }
        return self._make_request('/providers', params)
    
    def fetch_locations(self, page: int = 1, per_page: int = 1000) -> Dict:
        """Fetch locations from CQC API."""
        params = {
            'page': page,
            'perPage': per_page
        }
        return self._make_request('/locations', params)
    
    def fetch_all_data(self, endpoint: str) -> List[Dict]:
        """Fetch all pages of data from an endpoint."""
        all_data = []
        page = 1
        
        while True:
            logger.info(f"Fetching {endpoint} page {page}")
            
            if endpoint == 'providers':
                response = self.fetch_providers(page=page)
            elif endpoint == 'locations':
                response = self.fetch_locations(page=page)
            else:
                raise ValueError(f"Unknown endpoint: {endpoint}")
            
            data = response.get(endpoint, [])
            all_data.extend(data)
            
            # Check if there are more pages
            total_pages = response.get('totalPages', 1)
            if page >= total_pages:
                break
            
            page += 1
        
        logger.info(f"Fetched {len(all_data)} {endpoint}")
        return all_data


def get_secret(secret_id: str, project_id: str) -> str:
    """Retrieve secret from Secret Manager."""
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/{secret_id}/versions/latest"
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")


def upload_to_gcs(bucket_name: str, data: List[Dict], endpoint: str) -> str:
    """Upload data to Google Cloud Storage."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
    blob_name = f"raw/{endpoint}/{timestamp}_{endpoint}.json"
    
    blob = bucket.blob(blob_name)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type='application/json'
    )
    
    logger.info(f"Uploaded {len(data)} {endpoint} to gs://{bucket_name}/{blob_name}")
    return blob_name


@app.route('/', methods=['POST'])
def ingest_cqc_data():
    """Cloud Function entry point for CQC data ingestion."""
    try:
        # Get configuration from environment
        project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        bucket_name = os.environ.get('GCS_BUCKET', 'machine-learning-exp-467008-cqc-raw-data')
        
        if not project_id or not bucket_name:
            raise ValueError("Missing required environment variables")
        
        # Get secrets
        subscription_key = get_secret('cqc-subscription-key', project_id)
        try:
            partner_code = get_secret('cqc-partner-code', project_id)
        except Exception:
            partner_code = ""  # Partner code is optional
            logger.info("No partner code found - using standard rate limits")
        
        # Initialize API client
        client = CQCAPIClient(subscription_key, partner_code)
        
        # Fetch data from both endpoints
        results = {}
        
        for endpoint in ['providers', 'locations']:
            logger.info(f"Starting ingestion for {endpoint}")
            data = client.fetch_all_data(endpoint)
            blob_name = upload_to_gcs(bucket_name, data, endpoint)
            results[endpoint] = {
                'count': len(data),
                'blob_name': blob_name
            }
        
        return jsonify({
            'status': 'success',
            'timestamp': datetime.utcnow().isoformat(),
            'results': results
        }), 200
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        return jsonify({
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.utcnow().isoformat()
        }), 500


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy'}), 200


@app.route('/test-api', methods=['GET'])
def test_api():
    """Test CQC API connection."""
    try:
        project_id = os.environ.get('GCP_PROJECT', 'machine-learning-exp-467008')
        subscription_key = get_secret('cqc-subscription-key', project_id)
        
        tests = []
        
        # Test without any auth first
        url = "https://api.cqc.org.uk/public/v1/providers"
        response_no_auth = requests.get(url, params={'page': 1, 'perPage': 1}, timeout=10)
        tests.append({
            'test': 'No authentication',
            'status': response_no_auth.status_code,
            'response': response_no_auth.text[:200]
        })
        
        # Test with Ocp-Apim-Subscription-Key (most likely for Azure APIM)
        headers = {'Ocp-Apim-Subscription-Key': subscription_key}
        response_with_key = requests.get(url, headers=headers, params={'page': 1, 'perPage': 1}, timeout=10)
        tests.append({
            'test': 'With Ocp-Apim-Subscription-Key',
            'status': response_with_key.status_code,
            'response': response_with_key.text[:200],
            'headers_sent': dict(response_with_key.request.headers)
        })
        
        # Test changes endpoint
        changes_url = "https://api.cqc.org.uk/public/v1/changes/provider"
        response_changes = requests.get(changes_url, headers=headers, timeout=10)
        tests.append({
            'test': 'Changes endpoint',
            'status': response_changes.status_code,
            'response': response_changes.text[:200]
        })
        
        return jsonify({
            'subscription_key': subscription_key[:8] + '...',
            'tests': tests
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))