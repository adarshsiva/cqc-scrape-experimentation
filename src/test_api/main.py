import functions_framework
import requests
from google.cloud import secretmanager
import json

@functions_framework.http
def test_cqc_api(request):
    """Cloud Function to test CQC API connection."""
    
    # Get API key from Secret Manager
    project_id = "machine-learning-exp-467008"
    client = secretmanager.SecretManagerServiceClient()
    name = f"projects/{project_id}/secrets/cqc-subscription-key/versions/latest"
    
    try:
        response = client.access_secret_version(request={"name": name})
        api_key = response.payload.data.decode("UTF-8")
    except Exception as e:
        return json.dumps({"error": f"Failed to get API key: {str(e)}"})
    
    # Test the API
    url = "https://api.service.cqc.org.uk/public/v1/locations"
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "CQC-Fetcher/1.0"
    }
    params = {"page": 1, "perPage": 1}
    
    try:
        api_response = requests.get(url, headers=headers, params=params, timeout=30)
        
        result = {
            "status_code": api_response.status_code,
            "headers": dict(api_response.headers),
            "content_preview": api_response.text[:1000],
            "api_key_prefix": api_key[:10] + "...",
        }
        
        # Try to parse JSON
        try:
            json_data = api_response.json()
            result["json_parsed"] = True
            result["json_keys"] = list(json_data.keys()) if isinstance(json_data, dict) else "Not a dict"
            result["locations_count"] = len(json_data.get("locations", [])) if isinstance(json_data, dict) else 0
        except:
            result["json_parsed"] = False
            
        return json.dumps(result, indent=2), 200, {'Content-Type': 'application/json'}
        
    except Exception as e:
        return json.dumps({"error": f"Request failed: {str(e)}"}), 500, {'Content-Type': 'application/json'}