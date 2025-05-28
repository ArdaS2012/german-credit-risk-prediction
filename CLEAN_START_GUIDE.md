# 🚀 Clean Start Guide - German Credit Risk Prediction

## ✅ **System Cleaned Successfully**

All Docker containers and APIs have been stopped and removed. Ports 8000 and 3001 are now free.

---

## 🎯 **Manual Start Commands** (Run when ready)

### **Option 1A: Start API (CORS-Enabled - Recommended)**

```bash
# Start the CORS-enabled API container (works with web app)
docker run -d -p 8000:8000 --name credit-risk-api-cors --restart unless-stopped ghcr.io/ardas2012/german-credit-risk-prediction:cors-fixed

# Verify it's running
docker ps
curl http://localhost:8000/health
```

**Access Points:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

### **Option 1B: Start API (Latest from Registry)**

```bash
# Start the latest API container from GitHub registry
docker run -d -p 8000:8000 --name credit-risk-api --restart unless-stopped ghcr.io/ardas2012/german-credit-risk-prediction:main

# Verify it's running
docker ps
curl http://localhost:8000/health
```

**Access Points:**
- API: http://localhost:8000
- Documentation: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

---

### **Option 2: Start Web App Only**

```bash
# Navigate to web app directory
cd credit-risk-webapp

# Install dependencies (if not done before)
npm install

# Start the React app
npm start
```

**Access Point:**
- Web App: http://localhost:3000 (or http://localhost:3001 if 3000 is busy)

---

### **Option 3: Start Both API + Web App**

```bash
# Terminal 1: Start CORS-enabled API (recommended)
docker run -d -p 8000:8000 --name credit-risk-api-cors --restart unless-stopped ghcr.io/ardas2012/german-credit-risk-prediction:cors-fixed

# Terminal 2: Start Web App
cd credit-risk-webapp
npm start
```

**Access Points:**
- API: http://localhost:8000
- Web App: http://localhost:3000 or http://localhost:3001
- API Docs: http://localhost:8000/docs

---

## 🧪 **Testing Commands** (After starting)

### Test API Health
```bash
curl http://localhost:8000/health
```

### Test API Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "checking_account_status": "A11",
    "duration_months": 6,
    "credit_history": "A34",
    "purpose": "A43",
    "credit_amount": 1169,
    "savings_account": "A65",
    "employment_since": "A75",
    "installment_rate": 4,
    "personal_status_sex": "A93",
    "other_debtors": "A101",
    "residence_since": 4,
    "property": "A121",
    "age": 67,
    "other_installment_plans": "A143",
    "housing": "A152",
    "existing_credits": 2,
    "job": "A173",
    "dependents": 1,
    "telephone": "A192",
    "foreign_worker": "A201"
  }'
```

### Test Web App
- Open browser to http://localhost:3000 or http://localhost:3001
- Fill out the credit assessment form
- Submit and verify results

---

## 🛑 **Stop Commands** (When you want to stop)

### Stop API
```bash
# Stop CORS-enabled API
docker stop credit-risk-api-cors
docker rm credit-risk-api-cors

# OR stop latest API (if using that version)
docker stop credit-risk-api
docker rm credit-risk-api
```

### Stop Web App
```bash
# Press Ctrl+C in the terminal running npm start
```

### Stop All Docker Containers
```bash
docker stop $(docker ps -q)
docker rm $(docker ps -aq)
```

---

## 🔧 **Troubleshooting**

### Port Already in Use
```bash
# Check what's using the port
lsof -i :8000  # or :3000 or :3001

# Kill the process
kill <PID>
```

### Docker Issues
```bash
# Check Docker status
docker ps -a

# Clean up all containers
docker system prune -a
```

### Web App Issues
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

---

## 📋 **Current Status**

- ✅ **All Docker containers**: Stopped and removed
- ✅ **Port 8000**: Free
- ✅ **Port 3001**: Free
- ✅ **React app**: Stopped
- ✅ **System**: Ready for clean start

---

## 🎉 **Ready to Start Fresh!**

Choose your preferred option above and run the commands when you're ready. The system is completely clean and ready for a fresh start. 