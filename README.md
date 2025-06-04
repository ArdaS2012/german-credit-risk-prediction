# German Credit Risk Prediction System

A machine learning system for credit risk assessment using the UCI German Credit Dataset. Predicts loan repayment probability through a REST API and web interface.

## Overview

End-to-end MLOps pipeline that trains multiple ML models, provides a production API, and includes a web application for credit risk assessment.

**Dataset**: UCI German Credit Dataset (1,000 applications, 20 features)  
**Best Model**: Random Forest with 85%+ accuracy and 0.85+ ROC-AUC score  
**Tech Stack**: Python, FastAPI, React, Docker, MLflow

## Quick Start

### Docker Deployment (Recommended)

```bash
# Start API server
docker run -d -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main

# Start web application
cd credit-risk-webapp && npm install && npm start
```

**Access**: API at http://localhost:8000, Web App at http://localhost:3000, Docs at http://localhost:8000/docs

### Local Development

```bash
# Setup
git clone <repository-url> && cd german-credit-risk-prediction
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Train models and start API
cd src && python data_preprocessing.py && python model_training.py && python api.py &

# Start web app (new terminal)
cd credit-risk-webapp && npm install && npm start
```

## API Usage

### Health Check
```bash
curl http://localhost:8000/health
```

### Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account_status": "A11", "duration_months": 12, "credit_history": "A32",
    "purpose": "A43", "credit_amount": 5000, "savings_account": "A61",
    "employment_since": "A73", "installment_rate": 2, "personal_status_sex": "A93",
    "other_debtors": "A101", "residence_since": 2, "property": "A121", "age": 35,
    "other_installment_plans": "A143", "housing": "A152", "existing_credits": 1,
    "job": "A173", "dependents": 1, "telephone": "A192", "foreign_worker": "A201"
  }'
```

**Response**: `{"creditworthy":true,"probability":0.6308988869445707,"risk_score":0.36910111305542925,"timestamp":"2025-06-04T13:22:25.961819","model_version":"1.0.0"}`

## Architecture

```
React Web App (Port 3000) → FastAPI (Port 8000) → ML Models (Joblib)
```

**Components**:
- **Data Pipeline**: StandardScaler and OneHotEncoder preprocessing
- **ML Models**: Logistic Regression, Random Forest, XGBoost with hyperparameter tuning
- **API**: FastAPI with Pydantic validation and Prometheus metrics
- **Frontend**: React with Tailwind CSS, mobile-responsive
- **Deployment**: Docker containers with GitHub Actions CI/CD

## Features

### Machine Learning
- Automated preprocessing (20 → 48 features after encoding)
- Cross-validation and hyperparameter optimization with MLflow tracking
- Automatic best model selection

### Production API
- Input validation for all 20 credit application fields
- Single and batch prediction endpoints with health monitoring
- CORS configuration for web integration

### Web Application
- User-friendly form with real-time validation and visual risk assessment
- Mobile-optimized design, network accessible from any device

## Project Structure

```
src/
├── data_preprocessing.py     # Data pipeline
├── model_training.py         # ML training
└── api.py                    # FastAPI server
credit-risk-webapp/           # React frontend
tests/                        # Unit tests
models/                       # Trained models
Dockerfile                    # Container config
```

## Development

### Testing
```bash
cd tests && python run_tests.py
```

### Custom Docker Build
```bash
docker build -t german-credit-risk . && docker run -d -p 8000:8000 german-credit-risk
```

### Network Access
For mobile/other devices: Web App at http://192.168.1.100:3000, API at http://192.168.1.100:8000

## Technical Details

**ML Pipeline**: Trains 3 algorithms, selects best performer based on ROC-AUC  
**Data Processing**: Categorical encoding and numerical scaling  
**Validation**: 20 required fields with type and range checking  
**Monitoring**: Prometheus metrics and health checks  
**Security**: Non-root Docker user, input validation, CORS protection

MIT License