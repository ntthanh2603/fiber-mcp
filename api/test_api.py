"""
API Testing Script
Test all endpoints of the Phishing Detection API
"""

import requests
import json
from typing import Dict, Any

# API Base URL
BASE_URL = "http://localhost:8000"

# Test URLs
LEGITIMATE_URL = "https://www.google.com"
PHISHING_URL = "http://secure-login-verify.suspicious-domain.com/account/update?session=abc123"


def print_response(title: str, response: requests.Response):
    """Pretty print API response"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"Status Code: {response.status_code}")
    print(f"\nResponse:")
    print(json.dumps(response.json(), indent=2))


def test_health():
    """Test health check endpoint"""
    response = requests.get(f"{BASE_URL}/health")
    print_response("Health Check", response)
    return response.status_code == 200


def test_list_models():
    """Test list models endpoint"""
    response = requests.get(f"{BASE_URL}/models")
    print_response("List Models", response)
    return response.status_code == 200


def test_single_prediction():
    """Test single URL prediction"""
    # Test with phishing URL
    payload = {"url": PHISHING_URL}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_response("Single Prediction (Phishing URL)", response)

    # Test with legitimate URL
    payload = {"url": LEGITIMATE_URL}
    response = requests.post(f"{BASE_URL}/predict", json=payload)
    print_response("Single Prediction (Legitimate URL)", response)

    return response.status_code == 200


def test_model_specific_predictions():
    """Test predictions with specific models"""
    endpoints = [
        "/predict/xgboost",
        "/predict/random-forest",
        "/predict/logistic-regression"
    ]

    payload = {"url": PHISHING_URL}

    for endpoint in endpoints:
        response = requests.post(f"{BASE_URL}{endpoint}", json=payload)
        print_response(f"Prediction - {endpoint.split('/')[-1].upper()}", response)


def test_ensemble_prediction():
    """Test ensemble prediction"""
    payload = {"url": PHISHING_URL}
    response = requests.post(f"{BASE_URL}/predict/ensemble", json=payload)
    print_response("Ensemble Prediction", response)
    return response.status_code == 200


def test_batch_prediction():
    """Test batch prediction"""
    payload = {
        "urls": [
            "https://www.google.com",
            "https://www.github.com",
            PHISHING_URL,
            "http://suspicious-paypal-verify.com/login"
        ]
    }
    response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
    print_response("Batch Prediction", response)
    return response.status_code == 200


def test_detailed_analysis():
    """Test detailed URL analysis"""
    payload = {"url": PHISHING_URL}
    response = requests.post(f"{BASE_URL}/analyze", json=payload)
    print_response("Detailed Analysis", response)
    return response.status_code == 200


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*80)
    print("  PHISHING DETECTION API - COMPREHENSIVE TESTING")
    print("="*80)

    tests = [
        ("Health Check", test_health),
        ("List Models", test_list_models),
        ("Single Prediction", test_single_prediction),
        ("Model-Specific Predictions", test_model_specific_predictions),
        ("Ensemble Prediction", test_ensemble_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Detailed Analysis", test_detailed_analysis)
    ]

    results = []

    for test_name, test_func in tests:
        try:
            print(f"\n\n🧪 Running: {test_name}...")
            result = test_func()
            results.append((test_name, "✅ PASSED" if result else "⚠️ COMPLETED"))
        except requests.exceptions.ConnectionError:
            print(f"\n❌ ERROR: Cannot connect to API at {BASE_URL}")
            print("   Make sure the API is running: uvicorn main:app --reload")
            return
        except Exception as e:
            print(f"\n❌ ERROR: {test_name} failed with: {str(e)}")
            results.append((test_name, f"❌ FAILED: {str(e)}"))

    # Print summary
    print("\n" + "="*80)
    print("  TEST SUMMARY")
    print("="*80)
    for test_name, status in results:
        print(f"{status:15} {test_name}")
    print("="*80)


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════════════╗
    ║        Phishing Detection API - Test Suite                    ║
    ║                                                                ║
    ║  Make sure the API is running before running tests:           ║
    ║  $ cd api                                                      ║
    ║  $ uvicorn main:app --reload                                   ║
    ╚════════════════════════════════════════════════════════════════╝
    """)

    try:
        # Quick connectivity check
        response = requests.get(f"{BASE_URL}/", timeout=2)
        print(f"✅ API is reachable at {BASE_URL}\n")
        run_all_tests()
    except requests.exceptions.ConnectionError:
        print(f"❌ Cannot connect to API at {BASE_URL}")
        print("\n📝 To start the API:")
        print("   1. cd api")
        print("   2. pip install -r requirements.txt")
        print("   3. uvicorn main:app --reload")
        print("\n   Then run this test script again.")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
