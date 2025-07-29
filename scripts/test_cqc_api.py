#!/usr/bin/env python3
"""
Test CQC API access and troubleshoot connection issues
"""
import requests
import sys
import subprocess
import json

def get_api_key():
    """Get API key from Secret Manager"""
    try:
        result = subprocess.run(
            ["gcloud", "secrets", "versions", "access", "latest", "--secret=cqc-subscription-key"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Failed to get API key from Secret Manager: {e}")
        sys.exit(1)

def test_api_access(api_key):
    """Test various API endpoints and headers"""
    base_url = "https://api.cqc.org.uk/public/v1"
    
    # Different header combinations to try
    header_combinations = [
        {"Ocp-Apim-Subscription-Key": api_key},
        {"subscription-key": api_key},
        {"x-api-key": api_key},
        {"apikey": api_key}
    ]
    
    endpoints = [
        "/providers?page=1&perPage=1",
        "/locations?page=1&perPage=1",
        "/changes?startTimestamp=2024-01-01&endTimestamp=2024-01-02"
    ]
    
    print("Testing CQC API Access")
    print(f"API Key (first 10 chars): {api_key[:10]}...")
    print("=" * 50)
    
    for endpoint in endpoints:
        print(f"\nTesting endpoint: {endpoint}")
        print("-" * 30)
        
        for headers in header_combinations:
            header_name = list(headers.keys())[0]
            try:
                response = requests.get(
                    f"{base_url}{endpoint}",
                    headers=headers,
                    timeout=10
                )
                
                print(f"Header '{header_name}': Status {response.status_code}")
                
                if response.status_code == 200:
                    print("✅ SUCCESS! This header works.")
                    data = response.json()
                    print(f"Response preview: {json.dumps(data, indent=2)[:200]}...")
                    return True
                elif response.status_code == 401:
                    print("❌ Authentication failed - invalid API key")
                elif response.status_code == 403:
                    print("❌ Forbidden - API key might not be activated or lacks permissions")
                elif response.status_code == 429:
                    print("⚠️  Rate limit exceeded")
                else:
                    print(f"❌ Unexpected status: {response.text[:100]}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
    
    print("\n" + "=" * 50)
    print("TROUBLESHOOTING STEPS:")
    print("1. Check if you received an activation email from CQC")
    print("2. Ensure the API key is activated (click link in email)")
    print("3. Verify the key in the CQC API portal")
    print("4. Contact CQC support if the issue persists")
    print("5. The API key might need manual approval from CQC")
    
    return False

def main():
    api_key = get_api_key()
    
    if test_api_access(api_key):
        print("\n✅ API access is working! You can now fetch CQC data.")
    else:
        print("\n❌ API access is not working. Please follow the troubleshooting steps above.")
        sys.exit(1)

if __name__ == "__main__":
    main()