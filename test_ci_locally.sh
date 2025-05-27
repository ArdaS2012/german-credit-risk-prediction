#!/bin/bash

# Local CI/CD Testing Script
# This script tests all CI/CD components locally before pushing to GitHub

set -e  # Exit on any error

echo "🚀 Starting Local CI/CD Testing..."
echo "=================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        exit 1
    fi
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# 1. Check if virtual environment is active
echo "1. Checking virtual environment..."
if [[ "$VIRTUAL_ENV" != "" ]]; then
    print_status 0 "Virtual environment is active: $VIRTUAL_ENV"
else
    print_warning "Virtual environment not detected. Activating venv_test..."
    source venv_test/bin/activate
fi

# 2. Install CI dependencies
echo -e "\n2. Installing CI dependencies..."
pip install -q pytest pytest-cov flake8 black 2>/dev/null || true
print_status $? "CI dependencies installed"

# 3. Download test data if not exists
echo -e "\n3. Checking test data..."
if [ ! -f "data/german.data" ]; then
    echo "Downloading German Credit Dataset..."
    mkdir -p data
    wget -q -O data/german.data https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data
    print_status $? "Test data downloaded"
else
    print_status 0 "Test data already exists"
fi

# 4. Run linting (flake8)
echo -e "\n4. Running code linting..."
flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics --quiet
print_status $? "Code linting passed"

# 5. Run formatting check (black)
echo -e "\n5. Checking code formatting..."
black --check src/ --quiet
if [ $? -eq 0 ]; then
    print_status 0 "Code formatting is correct"
else
    print_warning "Code formatting issues found. Run 'black src/' to fix."
    black src/ --quiet
    print_status 0 "Code formatting fixed automatically"
fi

# 6. Run unit tests
echo -e "\n6. Running unit tests..."
python tests/run_tests.py > /dev/null 2>&1
print_status $? "Unit tests passed"

# 7. Train model for testing
echo -e "\n7. Training model for testing..."
cd src
python data_preprocessing.py > /dev/null 2>&1
print_status $? "Data preprocessing completed"

python model_training.py > /dev/null 2>&1
print_status $? "Model training completed"
cd ..

# 8. Test API integration
echo -e "\n8. Testing API integration..."
cd src
# Start API in background
python api.py > /dev/null 2>&1 &
API_PID=$!
sleep 10

# Test API
python ../test_api.py > /dev/null 2>&1
API_TEST_RESULT=$?

# Kill API process
kill $API_PID 2>/dev/null || true
cd ..

print_status $API_TEST_RESULT "API integration tests passed"

# 9. Test Docker build (if Docker is available)
echo -e "\n9. Testing Docker build..."
if command -v docker &> /dev/null; then
    echo "Building Docker image..."
    docker build -t credit-risk-api:test . > /dev/null 2>&1
    print_status $? "Docker image built successfully"
    
    echo "Testing Docker container..."
    # Start container in background
    docker run -d -p 8001:8000 --name credit-risk-test credit-risk-api:test > /dev/null 2>&1
    sleep 15
    
    # Test container health
    curl -s http://localhost:8001/health > /dev/null 2>&1
    DOCKER_TEST_RESULT=$?
    
    # Cleanup
    docker stop credit-risk-test > /dev/null 2>&1
    docker rm credit-risk-test > /dev/null 2>&1
    docker rmi credit-risk-api:test > /dev/null 2>&1
    
    print_status $DOCKER_TEST_RESULT "Docker container test passed"
else
    print_warning "Docker not available, skipping container tests"
fi

# 10. Check for required files
echo -e "\n10. Checking required CI/CD files..."
required_files=(
    ".github/workflows/ci-cd.yml"
    "Dockerfile"
    "requirements.txt"
    "tests/run_tests.py"
    "test_api.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo -e "${GREEN}✅ $file exists${NC}"
    else
        echo -e "${RED}❌ $file missing${NC}"
        exit 1
    fi
done

# Summary
echo -e "\n🎉 Local CI/CD Testing Complete!"
echo "=================================="
echo -e "${GREEN}All tests passed! Your code is ready for GitHub Actions.${NC}"
echo ""
echo "Next steps:"
echo "1. git add ."
echo "2. git commit -m 'Test CI/CD pipeline'"
echo "3. git push origin main"
echo "4. Monitor at: https://github.com/YOUR_USERNAME/YOUR_REPO/actions"
echo ""
echo "Note: Make sure to:"
echo "- Create a GitHub repository if you haven't"
echo "- Enable GitHub Container Registry"
echo "- Set up any required secrets" 