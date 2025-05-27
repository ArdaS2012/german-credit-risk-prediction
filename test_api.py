#!/usr/bin/env python3
"""
Simple test script to verify the API functionality.
"""

import requests
import json
import time
import sys

def test_api(base_url="http://localhost:8000"):
    """Test the API endpoints."""
    
    print(f"Testing API at {base_url}")
    print("=" * 50)
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            health_data = response.json()
            print(f"   Model loaded: {health_data['model_loaded']}")
            print(f"   Preprocessor loaded: {health_data['preprocessor_loaded']}")
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print("✅ Root endpoint passed")
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Root endpoint failed: {e}")
    
    # Test model info endpoint
    try:
        response = requests.get(f"{base_url}/model/info", timeout=5)
        if response.status_code == 200:
            print("✅ Model info endpoint passed")
            model_info = response.json()
            print(f"   Model type: {model_info['model_type']}")
        else:
            print(f"❌ Model info endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Model info endpoint failed: {e}")
    
    # Test prediction endpoint with sample data
    sample_application = {
        "checking_account_status": "A11",
        "duration_months": 12,
        "credit_history": "A32",
        "purpose": "A43",
        "credit_amount": 5000.0,
        "savings_account": "A61",
        "employment_since": "A73",
        "installment_rate": 2,
        "personal_status_sex": "A93",
        "other_debtors": "A101",
        "residence_since": 2,
        "property": "A121",
        "age": 35,
        "other_installment_plans": "A143",
        "housing": "A152",
        "existing_credits": 1,
        "job": "A173",
        "dependents": 1,
        "telephone": "A192",
        "foreign_worker": "A201"
    }
    
    try:
        response = requests.post(
            f"{base_url}/predict", 
            json=sample_application,
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        if response.status_code == 200:
            print("✅ Prediction endpoint passed")
            prediction = response.json()
            print(f"   Creditworthy: {prediction['creditworthy']}")
            print(f"   Probability: {prediction['probability']:.4f}")
            print(f"   Risk score: {prediction['risk_score']:.4f}")
        else:
            print(f"❌ Prediction endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Prediction endpoint failed: {e}")
    
    # Test metrics endpoint
    try:
        response = requests.get(f"{base_url}/metrics", timeout=5)
        if response.status_code == 200:
            print("✅ Metrics endpoint passed")
            metrics_text = response.text
            if "credit_predictions_total" in metrics_text:
                print("   Prometheus metrics are being collected")
        else:
            print(f"❌ Metrics endpoint failed: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Metrics endpoint failed: {e}")
    
    print("=" * 50)
    print("API testing completed!")
    return True


if __name__ == "__main__":
    # Check if custom URL provided
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    test_api(base_url) 