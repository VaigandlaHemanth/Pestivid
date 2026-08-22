#!/usr/bin/env python3
"""
Test script to verify Flask endpoint with improved recommendations
"""

import requests
import json
from pathlib import Path

def test_predict_endpoint():
    """Test the /predict endpoint with a sample image"""
    
    # Check if test image exists
    test_image_path = "20230712_114552.jpg"
    if not Path(test_image_path).exists():
        print(f"❌ Test image not found: {test_image_path}")
        return
    
    # Flask server URL
    url = "http://localhost:5000/predict"
    
    try:
        # Prepare the file for upload
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            
            print(f"🔄 Sending request to {url}...")
            response = requests.post(url, files=files, timeout=60)
            
        if response.status_code == 200:
            result = response.json()
            print("✅ Request successful!")
            print(f"📊 Response: {json.dumps(result, indent=2)}")
            
            # Check if we have the expected fields
            if 'disease' in result:
                print(f"🦠 Disease: {result['disease']}")
            if 'confidence' in result:
                print(f"📈 Confidence: {result['confidence']:.2%}")
            if 'recommendation' in result:
                print(f"💊 Recommendation: {result['recommendation']}")
                print(f"📏 Recommendation length: {len(result['recommendation'])} characters")
            
        else:
            print(f"❌ Request failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to Flask server. Make sure it's running on localhost:5000")
    except requests.exceptions.Timeout:
        print("❌ Request timed out. The server might be processing...")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_server_status():
    """Check if the Flask server is running"""
    try:
        response = requests.get("http://localhost:5000", timeout=5)
        print("✅ Flask server is running")
        return True
    except:
        print("❌ Flask server is not responding")
        return False

if __name__ == "__main__":
    print("--- Testing Flask Endpoint ---")
    
    if check_server_status():
        test_predict_endpoint()
    else:
        print("Please start the Flask server first by running: python flask_server.py")
