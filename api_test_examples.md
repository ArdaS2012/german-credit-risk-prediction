# API Testing Guide

## 🚀 How to Test the German Credit Risk API

### 1. Start the API Server

```bash
cd /home/arda/Schreibtisch/test_interview_project
source venv_test/bin/activate
cd src
python api.py
```

The API will be available at: `http://localhost:8000`

---

## 📝 Input Format

The API expects a JSON object with **20 required fields**:

### Required Fields:

| Field | Type | Range/Values | Description |
|-------|------|--------------|-------------|
| `checking_account_status` | string | A11, A12, A13, A14 | Status of existing checking account |
| `duration_months` | integer | 1-72 | Duration in months |
| `credit_history` | string | A30, A31, A32, A33, A34 | Credit history |
| `purpose` | string | A40, A41, A42, A43, A44, A45, A46, A48, A49, A410 | Purpose of credit |
| `credit_amount` | float | 250-20000 | Credit amount in DM |
| `savings_account` | string | A61, A62, A63, A64, A65 | Savings account/bonds |
| `employment_since` | string | A71, A72, A73, A74, A75 | Present employment since |
| `installment_rate` | integer | 1-4 | Installment rate in % of disposable income |
| `personal_status_sex` | string | A91, A92, A93, A94, A95 | Personal status and sex |
| `other_debtors` | string | A101, A102, A103 | Other debtors/guarantors |
| `residence_since` | integer | 1-4 | Present residence since |
| `property` | string | A121, A122, A123, A124 | Property |
| `age` | integer | 18-100 | Age in years |
| `other_installment_plans` | string | A141, A142, A143 | Other installment plans |
| `housing` | string | A151, A152, A153 | Housing |
| `existing_credits` | integer | 1-4 | Number of existing credits |
| `job` | string | A171, A172, A173, A174 | Job |
| `dependents` | integer | 1-2 | Number of dependents |
| `telephone` | string | A191, A192 | Telephone |
| `foreign_worker` | string | A201, A202 | Foreign worker |

---

## 🧪 Testing Methods

### Method 1: Using the Test Script (Recommended)

```bash
# Run the automated test
python test_api.py

# Test a different URL/port
python test_api.py http://localhost:8001
```

### Method 2: Using curl

#### Example 1: Good Credit Risk (Low Risk Customer)
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "checking_account_status": "A11",
       "duration_months": 12,
       "credit_history": "A34",
       "purpose": "A43",
       "credit_amount": 3000.0,
       "savings_account": "A63",
       "employment_since": "A74",
       "installment_rate": 2,
       "personal_status_sex": "A93",
       "other_debtors": "A101",
       "residence_since": 3,
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

#### Example 2: Higher Risk Customer
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "checking_account_status": "A14",
       "duration_months": 48,
       "credit_history": "A30",
       "purpose": "A40",
       "credit_amount": 15000.0,
       "savings_account": "A61",
       "employment_since": "A71",
       "installment_rate": 4,
       "personal_status_sex": "A91",
       "other_debtors": "A103",
       "residence_since": 1,
       "property": "A124",
       "age": 22,
       "other_installment_plans": "A141",
       "housing": "A153",
       "existing_credits": 3,
       "job": "A171",
       "dependents": 2,
       "telephone": "A191",
       "foreign_worker": "A202"
     }'
```

#### Example 3: Young Professional
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "checking_account_status": "A12",
       "duration_months": 24,
       "credit_history": "A32",
       "purpose": "A42",
       "credit_amount": 8000.0,
       "savings_account": "A62",
       "employment_since": "A73",
       "installment_rate": 3,
       "personal_status_sex": "A92",
       "other_debtors": "A101",
       "residence_since": 2,
       "property": "A122",
       "age": 28,
       "other_installment_plans": "A143",
       "housing": "A151",
       "existing_credits": 1,
       "job": "A172",
       "dependents": 1,
       "telephone": "A192",
       "foreign_worker": "A201"
     }'
```

### Method 3: Using Python requests

```python
import requests
import json

# API endpoint
url = "http://localhost:8000/predict"

# Sample input data
data = {
    "checking_account_status": "A11",
    "duration_months": 18,
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
    "age": 30,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 1,
    "job": "A173",
    "dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A201"
}

# Make request
response = requests.post(url, json=data)
result = response.json()

print(f"Creditworthy: {result['creditworthy']}")
print(f"Probability: {result['probability']:.4f}")
print(f"Risk Score: {result['risk_score']:.4f}")
```

### Method 4: Using Postman or Similar Tools

1. **URL**: `POST http://localhost:8000/predict`
2. **Headers**: `Content-Type: application/json`
3. **Body**: Use any of the JSON examples above

---

## 📊 Expected Response Format

```json
{
  "creditworthy": true,
  "probability": 0.7234,
  "risk_score": 0.2766,
  "timestamp": "2024-01-15T10:30:00.123456",
  "model_version": "1.0.0"
}
```

### Response Fields:
- `creditworthy`: Boolean - Whether the customer is likely to repay
- `probability`: Float (0-1) - Probability of being creditworthy
- `risk_score`: Float (0-1) - Risk score (1 - probability)
- `timestamp`: String - When the prediction was made
- `model_version`: String - Version of the model used

---

## 🔍 Other Useful Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Information
```bash
curl http://localhost:8000/model/info
```

### Prometheus Metrics
```bash
curl http://localhost:8000/metrics
```

### API Documentation (Interactive)
Open in browser: `http://localhost:8000/docs`

---

## ⚠️ Common Input Errors

### Invalid Categorical Values
```bash
# This will fail - A99 is not a valid checking_account_status
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"checking_account_status": "A99", ...}'
```

### Out of Range Values
```bash
# This will fail - age must be 18-100
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 150, ...}'
```

### Missing Required Fields
```bash
# This will fail - missing required fields
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"checking_account_status": "A11"}'
```

---

## 🎯 Quick Test Commands

```bash
# 1. Start API
python src/api.py &

# 2. Wait for startup
sleep 3

# 3. Quick health check
curl http://localhost:8000/health

# 4. Run full test suite
python test_api.py

# 5. Stop API
pkill -f "python.*api.py"
``` 