#!/usr/bin/env python3
"""
German Credit Risk Prediction Client
====================================

This script demonstrates how to use the deployed credit risk prediction API.
You can use this as a starting point for integrating the model into your applications.

Usage:
    python client_example.py
"""

import requests
import json
from typing import Dict, List, Any

class CreditRiskClient:
    """Client for the German Credit Risk Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict[str, Any]:
        """Check if the API is healthy and ready"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e), "status": "unhealthy"}
    
    def predict_single(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a single credit risk prediction"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict",
                json=customer_data
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def predict_batch(self, customers_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Make batch credit risk predictions"""
        try:
            response = self.session.post(
                f"{self.base_url}/predict/batch",
                json={"predictions": customers_list}
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}
    
    def get_metrics(self) -> str:
        """Get Prometheus metrics from the API"""
        try:
            response = self.session.get(f"{self.base_url}/metrics")
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            return f"Error getting metrics: {e}"

def create_sample_customer() -> Dict[str, Any]:
    """Create a sample customer for testing"""
    return {
        "checking_account_status": "A11",    # < 0 DM
        "duration_months": 6,                # 6 months
        "credit_history": "A34",             # critical/other existing credit
        "purpose": "A43",                    # radio/television
        "credit_amount": 1169,               # 1169 DM
        "savings_account": "A65",            # unknown/no savings account
        "employment_since": "A75",           # unemployed
        "installment_rate": 4,               # 4% of disposable income
        "personal_status_sex": "A93",        # male : single
        "other_debtors": "A101",             # none
        "residence_since": 4,                # 4 years
        "property": "A121",                  # real estate
        "age": 67,                           # 67 years old
        "other_installment_plans": "A143",   # none
        "housing": "A152",                   # own
        "existing_credits": 2,               # 2 existing credits
        "job": "A173",                       # skilled employee/official
        "dependents": 1,                     # 1 dependent
        "telephone": "A192",                 # none
        "foreign_worker": "A201"             # yes
    }

def create_high_risk_customer() -> Dict[str, Any]:
    """Create a high-risk customer example"""
    return {
        "checking_account_status": "A14",    # no checking account
        "duration_months": 48,               # 48 months (long term)
        "credit_history": "A30",             # no credits taken/all credits paid back duly
        "purpose": "A40",                    # car (new)
        "credit_amount": 7500,               # high amount
        "savings_account": "A61",            # < 100 DM
        "employment_since": "A71",           # unemployed
        "installment_rate": 4,               # 4% of disposable income
        "personal_status_sex": "A91",        # male : divorced/separated
        "other_debtors": "A101",             # none
        "residence_since": 1,                # 1 year (recent)
        "property": "A124",                  # no known property
        "age": 22,                           # young age
        "other_installment_plans": "A141",   # bank
        "housing": "A153",                   # for free
        "existing_credits": 1,               # 1 existing credit
        "job": "A172",                       # unemployed/unskilled - non-resident
        "dependents": 2,                     # 2 dependents
        "telephone": "A191",                 # none
        "foreign_worker": "A201"             # yes
    }

def main():
    """Main function demonstrating the API usage"""
    print("🏦 German Credit Risk Prediction Client")
    print("=" * 50)
    
    # Initialize client
    client = CreditRiskClient()
    
    # Check API health
    print("\n1. Checking API Health...")
    health = client.health_check()
    if "error" in health:
        print(f"❌ API is not available: {health['error']}")
        print("\n💡 To start the API:")
        print("   docker run -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main")
        return
    
    print(f"✅ API Status: {health.get('status', 'unknown')}")
    print(f"   Model Loaded: {health.get('model_loaded', False)}")
    print(f"   Preprocessor Loaded: {health.get('preprocessor_loaded', False)}")
    
    # Single prediction example
    print("\n2. Single Prediction Example...")
    sample_customer = create_sample_customer()
    
    print("Customer Profile:")
    print(f"   Age: {sample_customer['age']} years")
    print(f"   Credit Amount: {sample_customer['credit_amount']} DM")
    print(f"   Duration: {sample_customer['duration_months']} months")
    print(f"   Purpose: {sample_customer['purpose']}")
    
    result = client.predict_single(sample_customer)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
    else:
        creditworthy = result['creditworthy']
        probability = result['probability']
        risk_score = result['risk_score']
        
        print(f"\n📊 Prediction Results:")
        print(f"   Prediction: {'✅ Good Credit' if creditworthy else '❌ Bad Credit'}")
        print(f"   Probability: {probability:.3f}")
        print(f"   Risk Score: {risk_score:.3f}")
    
    # High-risk customer example
    print("\n3. High-Risk Customer Example...")
    high_risk_customer = create_high_risk_customer()
    
    print("High-Risk Customer Profile:")
    print(f"   Age: {high_risk_customer['age']} years")
    print(f"   Credit Amount: {high_risk_customer['credit_amount']} DM")
    print(f"   Duration: {high_risk_customer['duration_months']} months")
    print(f"   Employment: {high_risk_customer['employment_since']}")
    
    result = client.predict_single(high_risk_customer)
    
    if "error" in result:
        print(f"❌ Prediction failed: {result['error']}")
    else:
        creditworthy = result['creditworthy']
        probability = result['probability']
        risk_score = result['risk_score']
        
        print(f"\n📊 Prediction Results:")
        print(f"   Prediction: {'✅ Good Credit' if creditworthy else '❌ Bad Credit'}")
        print(f"   Probability: {probability:.3f}")
        print(f"   Risk Score: {risk_score:.3f}")
    
    # Batch prediction example
    print("\n4. Batch Prediction Example...")
    customers = [sample_customer, high_risk_customer]
    
    batch_result = client.predict_batch(customers)
    
    if "error" in batch_result:
        print(f"❌ Batch prediction failed: {batch_result['error']}")
    else:
        predictions = batch_result
        print(f"✅ Processed {len(predictions)} customers:")
        
        for i, pred in enumerate(predictions):
            customer_type = "Sample" if i == 0 else "High-Risk"
            creditworthy = pred['creditworthy']
            probability = pred['probability']
            
            print(f"   {customer_type}: {'Good' if creditworthy else 'Bad'} Credit ({probability:.3f})")
    
    print("\n5. API Documentation")
    print(f"   📖 Interactive docs: http://localhost:8000/docs")
    print(f"   📊 Metrics: http://localhost:8000/metrics")
    print(f"   ❤️  Health: http://localhost:8000/health")
    
    print("\n🎉 Demo completed! Your credit risk prediction API is working perfectly.")

if __name__ == "__main__":
    main() 