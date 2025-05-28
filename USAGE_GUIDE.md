# German Credit Risk Prediction - Usage Guide

Now that your MLOps pipeline is deployed, here are all the ways you can use your model:

## 🚀 Option 1: Use the Deployed Docker Container

### Pull and Run from GitHub Container Registry
```bash
# Pull the latest image
docker pull ghcr.io/ardas2012/german-credit-risk-prediction:main

# Run the API server
docker run -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main

# Access the API at http://localhost:8000
```

### Interactive API Documentation
Visit http://localhost:8000/docs for the Swagger UI where you can:
- Test predictions interactively
- See all available endpoints
- View request/response schemas

## 🔮 Option 2: Make Predictions via API

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account": "A11",
    "duration": 6,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1169,
    "savings_account": "A65",
    "employment": "A75",
    "installment_rate": 4,
    "personal_status": "A93",
    "other_parties": "A101",
    "residence_since": 4,
    "property_magnitude": "A121",
    "age": 67,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 2,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201"
  }'
```

### Batch Predictions
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {
        "checking_account": "A11",
        "duration": 6,
        "credit_history": "A34",
        "purpose": "A43",
        "credit_amount": 1169,
        "savings_account": "A65",
        "employment": "A75",
        "installment_rate": 4,
        "personal_status": "A93",
        "other_parties": "A101",
        "residence_since": 4,
        "property_magnitude": "A121",
        "age": 67,
        "other_payment_plans": "A143",
        "housing": "A152",
        "existing_credits": 2,
        "job": "A173",
        "num_dependents": 1,
        "own_telephone": "A192",
        "foreign_worker": "A201"
      }
    ]
  }'
```

## 🐍 Option 3: Use Python Client

### Create a Python Client Script
```python
import requests
import json

class CreditRiskClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def predict_single(self, customer_data):
        """Make a single prediction"""
        response = requests.post(
            f"{self.base_url}/predict",
            json=customer_data
        )
        return response.json()
    
    def predict_batch(self, customers_list):
        """Make batch predictions"""
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json={"predictions": customers_list}
        )
        return response.json()
    
    def health_check(self):
        """Check API health"""
        response = requests.get(f"{self.base_url}/health")
        return response.json()

# Example usage
client = CreditRiskClient()

# Sample customer data
customer = {
    "checking_account": "A11",
    "duration": 6,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1169,
    "savings_account": "A65",
    "employment": "A75",
    "installment_rate": 4,
    "personal_status": "A93",
    "other_parties": "A101",
    "residence_since": 4,
    "property_magnitude": "A121",
    "age": 67,
    "other_payment_plans": "A143",
    "housing": "A152",
    "existing_credits": 2,
    "job": "A173",
    "num_dependents": 1,
    "own_telephone": "A192",
    "foreign_worker": "A201"
}

# Make prediction
result = client.predict_single(customer)
print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.3f}")
print(f"Risk Level: {result['risk_level']}")
```

## 🔬 Option 4: Local Development and Experimentation

### Clone and Run Locally
```bash
# Clone the repository
git clone https://github.com/ArdaS2012/german-credit-risk-prediction.git
cd german-credit-risk-prediction

# Set up virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download data and train models
mkdir -p data
wget -O data/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

# Run preprocessing and training
cd src
python data_preprocessing.py
python model_training.py

# Start the API server
python api.py
```

### Jupyter Notebook Exploration
```bash
# Start Jupyter
jupyter notebook

# Open the EDA notebook
# Navigate to notebooks/german_credit_eda.ipynb
```

## 📊 Option 5: Monitor and Analyze

### Check API Metrics
```bash
# Prometheus metrics
curl http://localhost:8000/metrics

# Health status
curl http://localhost:8000/health
```

### MLflow Experiment Tracking
```bash
# Start MLflow UI (if running locally)
cd src
mlflow ui

# View at http://localhost:5000
```

## 🏭 Option 6: Production Deployment Options

### Deploy to Cloud Platforms

#### AWS ECS/Fargate
```bash
# Use the Docker image
docker pull ghcr.io/ardas2012/german-credit-risk-prediction:main

# Deploy to ECS using the image
```

#### Google Cloud Run
```bash
# Deploy directly from container registry
gcloud run deploy credit-risk-api \
  --image ghcr.io/ardas2012/german-credit-risk-prediction:main \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### Azure Container Instances
```bash
az container create \
  --resource-group myResourceGroup \
  --name credit-risk-api \
  --image ghcr.io/ardas2012/german-credit-risk-prediction:main \
  --ports 8000
```

### Kubernetes Deployment
```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: credit-risk-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: credit-risk-api
  template:
    metadata:
      labels:
        app: credit-risk-api
    spec:
      containers:
      - name: api
        image: ghcr.io/ardas2012/german-credit-risk-prediction:main
        ports:
        - containerPort: 8000
---
apiVersion: v1
kind: Service
metadata:
  name: credit-risk-service
spec:
  selector:
    app: credit-risk-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🔧 Option 7: Customize and Extend

### Add New Features
1. **New Models**: Add more algorithms in `src/model_training.py`
2. **Feature Engineering**: Extend preprocessing in `src/data_preprocessing.py`
3. **API Endpoints**: Add new routes in `src/api.py`
4. **Monitoring**: Enhance metrics and logging

### Retrain with New Data
```bash
# Add new data to data/ directory
# Run the training pipeline
cd src
python data_preprocessing.py
python model_training.py

# Rebuild Docker image
docker build -t credit-risk-api:updated .
```

## 📱 Option 8: Integration Examples

### Web Application Integration
```javascript
// Frontend JavaScript example
async function predictCreditRisk(customerData) {
    const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(customerData)
    });
    
    const result = await response.json();
    return result;
}
```

### Database Integration
```python
# Example: Save predictions to database
import sqlite3
import requests

def save_prediction_to_db(customer_data, prediction_result):
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions 
        (customer_id, prediction, probability, risk_level, timestamp)
        VALUES (?, ?, ?, ?, datetime('now'))
    ''', (
        customer_data.get('customer_id'),
        prediction_result['prediction'],
        prediction_result['probability'],
        prediction_result['risk_level']
    ))
    
    conn.commit()
    conn.close()
```

## 🎯 Quick Start Recommendations

1. **For Testing**: Use Option 1 (Docker) + Option 2 (API calls)
2. **For Development**: Use Option 4 (Local setup)
3. **For Production**: Use Option 6 (Cloud deployment)
4. **For Integration**: Use Option 3 (Python client) or Option 8 (Web/DB integration)

## 📚 Next Steps

1. **Explore the API**: Start with http://localhost:8000/docs
2. **Test Predictions**: Use the sample data provided
3. **Monitor Performance**: Check metrics and logs
4. **Scale Up**: Deploy to cloud platforms
5. **Customize**: Add your own features and improvements

Your German Credit Risk Prediction system is now ready for real-world use! 🚀 