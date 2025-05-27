# CI/CD Testing Guide

## 🚀 Current CI/CD Implementation Status

### ✅ **What's Already Implemented:**

1. **GitHub Actions Workflow** (`.github/workflows/ci-cd.yml`)
   - ✅ Multi-Python version testing (3.8, 3.9)
   - ✅ Dependency caching
   - ✅ Code linting (flake8)
   - ✅ Code formatting (black)
   - ✅ Unit testing with coverage
   - ✅ Docker image building
   - ✅ Container registry push (GitHub Container Registry)
   - ✅ Security scanning (Trivy)
   - ✅ Environment-based deployment

2. **Docker Configuration**
   - ✅ Multi-stage Dockerfile
   - ✅ Security best practices (non-root user)
   - ✅ Health checks
   - ✅ Proper dependency management

3. **Testing Infrastructure**
   - ✅ Unit tests for preprocessing
   - ✅ API test script
   - ✅ Coverage reporting

---

## ❌ **What's Missing/Needs Fixing:**

### 1. **Python Version Compatibility**
```yaml
# Current (outdated):
python-version: [3.8, 3.9]

# Should be:
python-version: [3.9, 3.10, 3.11, 3.12]
```

### 2. **Test Dependencies Missing**
The workflow tries to use `pytest` but it's not in `requirements.txt`

### 3. **Model Training in CI**
Currently commented out - needs proper implementation

### 4. **Deployment Configuration**
The deploy step is just a placeholder

### 5. **Environment Variables**
Missing MLflow and other service configurations

### 6. **Integration Tests**
No API integration tests in CI

---

## 🧪 **How to Test the CI/CD Pipeline**

### **Method 1: Local Testing (Recommended First)**

#### 1. Test the Docker Build Locally
```bash
# Build the Docker image
docker build -t credit-risk-api:test .

# Run the container
docker run -p 8000:8000 credit-risk-api:test

# Test the API
curl http://localhost:8000/health
```

#### 2. Test Individual CI Steps Locally
```bash
# Install CI dependencies
pip install pytest pytest-cov flake8 black

# Run linting
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics

# Run formatting check
black --check src/

# Run tests
python -m pytest tests/ -v --cov=src
```

### **Method 2: GitHub Actions Testing**

#### Prerequisites:
1. **GitHub Repository**: Push your code to GitHub
2. **GitHub Container Registry**: Enable in repository settings
3. **Secrets Configuration**: Set up required secrets

#### Steps to Test:

1. **Push to GitHub**
```bash
git add .
git commit -m "Test CI/CD pipeline"
git push origin main
```

2. **Monitor Workflow**
   - Go to GitHub → Actions tab
   - Watch the workflow execution
   - Check logs for each step

3. **Test Different Scenarios**
```bash
# Test pull request workflow
git checkout -b feature/test-ci
git push origin feature/test-ci
# Create PR on GitHub

# Test different branches
git checkout develop
git push origin develop
```

---

## 🔧 **Fixes Needed for Full CI/CD**

### 1. **Update Python Versions**
```yaml
# .github/workflows/ci-cd.yml
strategy:
  matrix:
    python-version: [3.9, 3.10, 3.11, 3.12]
```

### 2. **Fix Test Dependencies**
```bash
# Add to requirements.txt
pytest>=7.0.0
pytest-cov>=4.0.0
flake8>=6.0.0
black>=23.0.0
```

### 3. **Fix Test Command**
```yaml
# Current issue: tries to use pytest but runs custom test
- name: Run preprocessing tests
  run: |
    cd src
    python ../tests/run_tests.py  # Use our custom runner
```

### 4. **Add API Integration Tests to CI**
```yaml
- name: Run API integration tests
  run: |
    cd src
    python api.py &
    sleep 10
    python ../test_api.py
    pkill -f "python.*api.py"
```

### 5. **Fix Model Training Step**
```yaml
- name: Train model for deployment
  run: |
    cd src
    python data_preprocessing.py
    python model_training.py  # Uncomment this
```

### 6. **Add Environment Configuration**
```yaml
env:
  MLFLOW_TRACKING_URI: sqlite:///mlflow.db
  LOG_LEVEL: INFO
```

---

## 🚀 **Complete Fixed CI/CD Workflow**

Let me create the fixed version:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}
  MLFLOW_TRACKING_URI: sqlite:///mlflow.db

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.9, 3.10, 3.11, 3.12]

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}

    - name: Cache pip dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt

    - name: Download test data
      run: |
        mkdir -p data
        wget -O data/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data

    - name: Run preprocessing tests
      run: |
        python tests/run_tests.py

    - name: Train model for testing
      run: |
        cd src
        python data_preprocessing.py
        python model_training.py

    - name: Run API integration tests
      run: |
        cd src
        timeout 30s python api.py &
        sleep 10
        python ../test_api.py
        pkill -f "python.*api.py" || true

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.event_name == 'push'

    steps:
    - name: Checkout repository
      uses: actions/checkout@v4

    - name: Log in to Container Registry
      uses: docker/login-action@v3
      with:
        registry: ${{ env.REGISTRY }}
        username: ${{ github.actor }}
        password: ${{ secrets.GITHUB_TOKEN }}

    - name: Extract metadata
      id: meta
      uses: docker/metadata-action@v5
      with:
        images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}

    - name: Prepare for build
      run: |
        mkdir -p data models
        wget -O data/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
        cd src && python data_preprocessing.py && python model_training.py

    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: ${{ steps.meta.outputs.tags }}
        labels: ${{ steps.meta.outputs.labels }}

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    environment: production

    steps:
    - name: Deploy to production
      run: |
        echo "Deploying ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}:main"
        # Add actual deployment commands here
```

---

## 🎯 **Testing Checklist**

### **Local Testing:**
- [ ] Docker build works
- [ ] Container runs successfully
- [ ] API responds to health checks
- [ ] All unit tests pass
- [ ] Integration tests work

### **GitHub Actions Testing:**
- [ ] Workflow triggers on push/PR
- [ ] All Python versions pass tests
- [ ] Docker image builds successfully
- [ ] Image pushes to registry
- [ ] Security scan completes
- [ ] Deployment step executes (if on main)

### **Production Readiness:**
- [ ] Environment variables configured
- [ ] Secrets properly set
- [ ] Monitoring configured
- [ ] Rollback strategy defined
- [ ] Health checks working

---

## 🚨 **Current Limitations**

1. **No Real Deployment Target**: The deploy step is just a placeholder
2. **Missing Monitoring**: No integration with monitoring systems
3. **No Database**: MLflow uses local SQLite (not production-ready)
4. **No Secrets Management**: API keys, DB credentials not handled
5. **No Multi-Environment**: Only production environment defined

---

## 🎯 **Quick Test Commands**

```bash
# Test everything locally first
./test_ci_locally.sh

# Then push to GitHub to test CI/CD
git add .
git commit -m "Test CI/CD pipeline"
git push origin main

# Monitor at: https://github.com/YOUR_USERNAME/YOUR_REPO/actions
``` 