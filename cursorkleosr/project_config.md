# project_config.md

## Goal
Develop a machine learning system to predict whether a customer is likely to repay a loan, based on structured customer data from the UCI German Credit Dataset. This is a binary classification problem focused on credit risk assessment.

## Dataset
- Source: [German Credit Data - UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))
- Format: CSV
- Features: 20 attributes including age, job status, credit amount, duration, and savings
- Target: 'Creditability' (1 = creditworthy, 0 = not creditworthy)

## Tech Stack
- Language: Python
- ML Libraries: scikit-learn, XGBoost
- Experiment Tracking: MLflow
- API Framework: FastAPI
- Containerization: Docker
- Orchestration: Kubernetes (for scaling)
- CI/CD: GitHub Actions
- Monitoring: Prometheus

## Critical Patterns & Conventions
- Use pandas and numpy for data manipulation
- Load and preprocess the German Credit Data from CSV format
- Handle categorical features with encoding, treat missing values appropriately
- Include preprocessing pipelines for scaling and feature engineering
- Apply stratified k-fold cross-validation
- Use GridSearchCV for hyperparameter tuning
- Track experiments with MLflow: parameters, metrics, artifacts
- Wrap the trained model into a FastAPI service
- Include Dockerfile and CI/CD config (GitHub Actions)
- Design for scalable deployment (horizontal scaling with Kubernetes)
- Ensure inference logging and monitoring for drift detection

## Limitations
- No use of proprietary datasets
- No GUI frontend – API only
- All components must be testable and reproducible
- Avoid use of deprecated ML libraries or hard-coded values
