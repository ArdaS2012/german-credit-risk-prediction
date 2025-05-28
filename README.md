# German Credit Risk Prediction

A machine learning system for predicting credit risk using the UCI German Credit Dataset. This project implements a complete MLOps pipeline with data preprocessing, model training, API deployment, and monitoring.

## 🚀 **Quick Deployment**

**Want to get started immediately?** 

👉 **[📋 DEPLOYMENT GUIDE](DEPLOYMENT_GUIDE.md)** - Get the complete system running in 2 commands!

- 🐳 **Docker from GitHub**: `docker run -d -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main`
- 🌐 **Web Interface**: Clone repo → `npm install && npm start`
- ⚡ **Ready in 2 minutes**: Full web app + API + documentation

---

## 🎯 Project Overview

This project develops a binary classification system to predict whether a customer is likely to repay a loan based on structured customer data. The system uses multiple machine learning algorithms and provides a REST API for real-time predictions.

### Key Features

- **Data Processing**: Robust preprocessing pipeline with categorical encoding and numerical scaling
- **Multiple Models**: Logistic Regression, Random Forest, and XGBoost with hyperparameter tuning
- **Experiment Tracking**: MLflow integration for comprehensive experiment management
- **REST API**: FastAPI-based service with input validation and monitoring
- **Containerization**: Docker support for easy deployment
- **CI/CD Pipeline**: GitHub Actions workflow for automated testing and deployment
- **Monitoring**: Prometheus metrics for production monitoring

## 📊 Dataset

- **Source**: [UCI German Credit Dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- **Size**: 1,000 samples with 20 features
- **Target**: Binary classification (creditworthy vs. not creditworthy)
- **Features**: Mix of numerical (7) and categorical (13) attributes

## 🏗️ Architecture

```
├── data/                   # Dataset storage
├── src/                    # Source code
│   ├── data_preprocessing.py   # Data preprocessing pipeline
│   ├── model_training.py       # Model training with MLflow
│   └── api.py                  # FastAPI application
├── notebooks/              # Jupyter notebooks for EDA
├── tests/                  # Unit tests
├── models/                 # Trained model artifacts
├── .github/workflows/      # CI/CD pipeline
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Docker (optional)
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd german-credit-risk
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download and preprocess data**
   ```bash
   cd src
   python data_preprocessing.py
   ```

4. **Train models**
   ```bash
   python model_training.py
   ```

5. **Start the API server**
   ```bash
   python api.py
   ```

The API will be available at `http://localhost:8000`

## 📖 Usage

### Data Preprocessing

```python
from src.data_preprocessing import GermanCreditPreprocessor

# Initialize preprocessor
preprocessor = GermanCreditPreprocessor()

# Load and preprocess data
df = preprocessor.load_data()
X, y, feature_names = preprocessor.fit_transform(df)

# Split data
X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
```

### Model Training

```python
from src.model_training import CreditRiskModelTrainer

# Initialize trainer
trainer = CreditRiskModelTrainer()

# Train all models
models = trainer.train_all_models()

# View results
report = trainer.generate_model_report()
```

### API Usage

#### Single Prediction

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "checking_account_status": "A11",
       "duration_months": 12,
       "credit_history": "A30",
       "purpose": "A40",
       "credit_amount": 5000,
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
     }'
```