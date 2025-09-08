#!/usr/bin/env python3
"""Direct API test to diagnose connection issues."""

import requests
import os
import json
from google.cloud import secretmanager

def get_api_key():
    """Get API key from Secret Manager."""
    try:
        project_id = "machine-learning-exp-467008"
        client = secretmanager.SecretManagerServiceClient()
        name = f"projects/{project_id}/secrets/cqc-subscription-key/versions/latest"
        print(f"Attempting to retrieve API key from: {name}")
        response = client.access_secret_version(request={"name": name})
        key = response.payload.data.decode("UTF-8")
        print(f"✅ API key retrieved (length: {len(key)}): {key[:10]}...{key[-4:]}")
        return key
    except Exception as e:
        print(f"❌ Error getting API key: {e}")
        print(f"Project ID: {project_id}")
        return None

def test_simple_request():
    """Test with minimal configuration like Cloud Function."""
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key available")
        return
    
    # Test URL
    url = "https://api.service.cqc.org.uk/public/v1/locations"
    
    # Minimal headers like a working Cloud Function
    headers = {
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "Google-Cloud-Functions/2.0 Python/3.11"
    }
    
    params = {
        "page": 1,
        "perPage": 1
    }
    
    print(f"\n🔗 Testing API at: {url}")
    print(f"📋 Headers: {dict((k, v[:10] + '...' + v[-4:] if k == 'Ocp-Apim-Subscription-Key' else v) for k, v in headers.items())}")
    print(f"📋 Params: {params}")
    
    try:
        print(f"\n📡 Making request...")
        response = requests.get(url, headers=headers, params=params, timeout=30)
        
        print(f"\n📊 Response Status: {response.status_code}")
        print(f"📊 Response Headers: {dict(response.headers)}")
        print(f"📊 Content-Type: {response.headers.get('content-type', 'Not specified')}")
        print(f"📊 Content-Length: {len(response.text)}")
        print(f"📊 Response Encoding: {response.encoding}")
        
        # Print response content preview
        print(f"\n📄 Response Content Preview (first 500 chars):")
        print("-" * 50)
        print(response.text[:500])
        if len(response.text) > 500:
            print("... (truncated)")
        print("-" * 50)
        
        # Try to parse as JSON
        try:
            json_data = response.json()
            print(f"\n✅ Successfully parsed JSON!")
            print(f"📋 JSON Structure: {list(json_data.keys())}")
            if 'locations' in json_data:
                print(f"📋 Locations count: {len(json_data['locations'])}")
            if 'totalCount' in json_data:
                print(f"📋 Total locations available: {json_data['totalCount']}")
            print(f"📋 Sample JSON (formatted):")
            print(json.dumps(json_data, indent=2)[:1000] + "..." if len(str(json_data)) > 1000 else json.dumps(json_data, indent=2))
        except json.JSONDecodeError as e:
            print(f"\n❌ JSON parsing failed: {e}")
            print(f"Error details: {str(e)}")
            # Check if response is HTML (error page)
            if response.text.strip().startswith('<'):
                print("⚠️  Response appears to be HTML (possibly an error page)")
            elif not response.text.strip():
                print("⚠️  Response is empty")
            else:
                print("⚠️  Response is not valid JSON")
        except Exception as e:
            print(f"\n❌ Unexpected error parsing response: {e}")
            
    except requests.exceptions.Timeout:
        print(f"\n❌ Request timed out after 30 seconds")
    except requests.exceptions.ConnectionError as e:
        print(f"\n❌ Connection error: {e}")
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Request failed: {e}")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

def test_with_session():
    """Test with session configuration like the enhanced fetcher."""
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    api_key = get_api_key()
    if not api_key:
        print("❌ No API key available")
        return
    
    print(f"\n🔄 Testing with session configuration...")
    
    # Create session like the enhanced fetcher
    session = requests.Session()
    
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[403, 429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    session.headers.update({
        "Ocp-Apim-Subscription-Key": api_key,
        "Accept": "application/json",
        "User-Agent": "Google-Cloud-Functions/2.0 Python/3.11"
    })
    
    url = "https://api.service.cqc.org.uk/public/v1/locations"
    params = {"page": 1, "perPage": 1}
    
    try:
        print(f"📡 Making session request to: {url}")
        response = session.get(url, params=params, timeout=30)
        print(f"📊 Status: {response.status_code}")
        print(f"📄 Content preview: {response.text[:200]}...")
        
        if response.status_code == 200:
            try:
                data = response.json()
                print(f"✅ Session request successful! Keys: {list(data.keys())}")
            except:
                print(f"❌ Session request got 200 but JSON parsing failed")
        else:
            print(f"❌ Session request failed with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Session request failed: {e}")

def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 CQC API DIAGNOSTIC TEST")
    print("=" * 60)
    
    # Test 1: Simple request
    print("\n🔬 TEST 1: Simple Request")
    test_simple_request()
    
    # Test 2: Session-based request
    print("\n🔬 TEST 2: Session Request")
    test_with_session()
    
    print("\n" + "=" * 60)
    print("🏁 DIAGNOSTIC TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()