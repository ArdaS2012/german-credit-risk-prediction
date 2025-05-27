# German Credit Risk Prediction

A machine learning system for predicting credit risk using the UCI German Credit Dataset. This project implements a complete MLOps pipeline with data preprocessing, model training, API deployment, and monitoring.

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

#### Response

```json
{
  "creditworthy": true,
  "probability": 0.75,
  "risk_score": 0.25,
  "timestamp": "2024-01-15T10:30:00",
  "model_version": "1.0.0"
}
```

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics (Prometheus)

```bash
curl http://localhost:8000/metrics
```

## 🐳 Docker Deployment

### Build Image

```bash
docker build -t credit-risk-api .
```

### Run Container

```bash
docker run -p 8000:8000 credit-risk-api
```

## 🧪 Testing

### Unit Tests

Run the preprocessing tests:

```bash
# Using pytest (if available)
pytest tests/ -v --cov=src

# Using the custom test runner (for environments with ROS conflicts)
python tests/run_tests.py
```

### API Testing

Test the API endpoints with the provided test script:

```bash
# Start the API first
python src/api.py

# In another terminal, run the API tests
python src/test_api.py

# Or test a specific URL
python src/test_api.py http://localhost:8001
```

### Recent Improvements

**Version 1.1.0 Updates:**
- ✅ **Fixed FastAPI deprecation warning**: Replaced deprecated `@app.on_event("startup")` with modern `lifespan` event handlers
- ✅ **Resolved Prometheus metrics collision**: Added registry cleanup to prevent duplicate metric registration errors
- ✅ **Enhanced error handling**: Improved startup error handling and logging
- ✅ **Virtual environment compatibility**: Resolved Python 3.12 compatibility issues and ROS package conflicts
- ✅ **Alternative test runner**: Created `tests/run_tests.py` for environments where pytest conflicts with system packages

## 📈 Model Performance

The system trains and compares three models:

| Model | Test AUC | CV AUC | Precision | Recall | F1-Score |
|-------|----------|--------|-----------|--------|----------|
| XGBoost | 0.78 | 0.76 ± 0.03 | 0.82 | 0.71 | 0.76 |
| Random Forest | 0.75 | 0.74 ± 0.04 | 0.79 | 0.68 | 0.73 |
| Logistic Regression | 0.72 | 0.71 ± 0.02 | 0.76 | 0.65 | 0.70 |

*Note: Actual performance may vary based on data splits and hyperparameter tuning.*

## 🔧 Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow tracking server URL
- `MODEL_PATH`: Path to trained model file
- `LOG_LEVEL`: Logging level (INFO, DEBUG, WARNING, ERROR)

### API Configuration

The API supports the following categorical values:

- **Checking Account Status**: A11, A12, A13, A14
- **Credit History**: A30, A31, A32, A33, A34
- **Purpose**: A40, A41, A42, A43, A44, A45, A46, A48, A49, A410
- **Savings Account**: A61, A62, A63, A64, A65
- **Employment Since**: A71, A72, A73, A74, A75
- **Personal Status/Sex**: A91, A92, A93, A94, A95
- **Other Debtors**: A101, A102, A103
- **Property**: A121, A122, A123, A124
- **Other Installment Plans**: A141, A142, A143
- **Housing**: A151, A152, A153
- **Job**: A171, A172, A173, A174
- **Telephone**: A191, A192
- **Foreign Worker**: A201, A202

## 🚀 Deployment

### Kubernetes

Example Kubernetes deployment:

```yaml
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
        image: credit-risk-api:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
```

### CI/CD Pipeline

The project includes a GitHub Actions workflow that:

1. **Tests**: Runs unit tests and linting
2. **Security**: Performs vulnerability scanning
3. **Build**: Creates Docker images
4. **Deploy**: Deploys to production (configurable)

## 📊 Monitoring

### Metrics

The API exposes Prometheus metrics:

- `credit_predictions_total`: Total number of predictions
- `credit_positive_predictions_total`: Number of positive predictions
- `credit_prediction_duration_seconds`: Prediction latency

### Logging

Structured logging with:
- Request/response logging
- Error tracking
- Performance metrics
- Model prediction logging

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- UCI Machine Learning Repository for the German Credit Dataset
- The open-source community for the excellent ML libraries used in this project

## 📞 Support

For questions or issues, please:

1. Check the [Issues](../../issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

---

**Note**: This is a demonstration project for educational purposes. For production use, ensure proper security measures, data privacy compliance, and thorough testing. 