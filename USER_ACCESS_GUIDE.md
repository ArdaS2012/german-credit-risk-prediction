# 🏦 Credit Risk Assessment - User Guide

## Quick Start (2 Minutes Setup)

### Step 1: Start the AI Model
```bash
docker run -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main
```

### Step 2: Start the Web Application
```bash
cd credit-risk-webapp
npm install  # First time only
npm start
```

### Step 3: Open Your Browser
Go to: **http://localhost:3000**

## 🎯 How to Use the Credit Risk Assessment Tool

### 1. Fill Out the Form
The web application presents a user-friendly form with all the required information:

**Personal Information:**
- Age (18-100 years)
- Personal status and gender
- Number of dependents (1-2)

**Financial Details:**
- Checking account balance range
- Credit amount requested (250-20,000 DM)
- Loan duration (1-72 months)
- Savings account balance
- Monthly installment rate (1-4% of income)

**Employment & Housing:**
- Employment duration
- Job type and skill level
- Housing situation (own/rent/free)
- Years at current residence

**Credit History:**
- Previous credit performance
- Purpose of the loan
- Existing payment plans
- Number of current credits

### 2. Submit for Assessment
- Click the "Assess Credit Risk" button
- Wait for the AI model to process (usually 1-2 seconds)

### 3. View Your Results
The system provides:
- **Risk Level**: Low Risk ✅ or High Risk ❌
- **Creditworthiness**: Approved or Declined
- **Confidence Score**: How confident the AI is (percentage)
- **Risk Score**: Detailed risk assessment with visual indicator

### 4. Assess Another Application
Click "Assess Another Application" to reset and evaluate a new case.

## 📱 Device Compatibility

✅ **Desktop**: Full-featured experience
✅ **Tablet**: Optimized layout
✅ **Mobile**: Touch-friendly interface
✅ **All Browsers**: Chrome, Firefox, Safari, Edge

## 🔍 Sample Test Cases

### Low Risk Customer Example:
- Age: 35, Amount: 2000 DM, Duration: 12 months
- Good checking account (≥ 200 DM)
- Stable employment (4+ years)
- Owns property

### High Risk Customer Example:
- Age: 22, Amount: 7500 DM, Duration: 48 months
- No checking account
- Unemployed
- No property

## 🚀 For Developers: API Access

### Direct API Endpoint
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{ /* customer data */ }'
```

### Interactive Documentation
Visit: **http://localhost:8000/docs**

### Python Integration
```python
from client_example import CreditRiskClient
client = CreditRiskClient()
result = client.predict_single(customer_data)
```

## 🌐 Production Deployment

### For Public Access:
1. **Deploy API**: Use the Docker image on any cloud platform
2. **Deploy Web App**: Build and host on Netlify/Vercel
3. **Configure**: Update API endpoint in the web app

### Cloud Platforms:
- **Google Cloud Run**: One-click deployment
- **AWS ECS/Fargate**: Container-based hosting
- **Azure Container Instances**: Simple container deployment

## 📊 What Makes This Special

✅ **Real AI Model**: Trained on 1,000 real credit applications
✅ **Production Ready**: Docker containerized, CI/CD pipeline
✅ **User Friendly**: No technical knowledge required
✅ **Fast**: Sub-second predictions
✅ **Accurate**: Multiple ML algorithms (XGBoost, Random Forest, Logistic Regression)
✅ **Monitored**: Health checks and performance metrics
✅ **Secure**: Input validation and error handling

## 🎉 You're Ready!

Your German Credit Risk Prediction system is now accessible to users through a beautiful, professional web interface. Users can:

1. **Visit the web app** (http://localhost:3000)
2. **Fill out the assessment form** with their information
3. **Get instant AI-powered risk assessment** 
4. **Make informed credit decisions** based on the results

The system transforms complex machine learning into a simple, user-friendly tool that anyone can use! 🚀 