# 🚀 German Credit Risk Prediction - Deployment Guide

## Quick Start (2 Commands)

Get the complete system running in under 2 minutes:

```bash
# 1. Start the API (with CORS enabled)
docker run -d -p 8000:8000 --name credit-risk-api ghcr.io/ardas2012/german-credit-risk-prediction:main

# 2. Clone and start the web application
git clone https://github.com/ArdaS2012/german-credit-risk-prediction.git
cd german-credit-risk-prediction/credit-risk-webapp
npm install && npm start
```

**That's it!** 
- 🌐 Web App: http://localhost:3000
- 🔗 API: http://localhost:8000
- 📚 API Docs: http://localhost:8000/docs

---

## 📋 Prerequisites

- **Docker** (for API)
- **Node.js 16+** (for web app)
- **Git** (to clone repository)

---

## 🐳 Option 1: Docker from GitHub Registry (Recommended)

### Step 1: Pull and Run API Container

```bash
# Pull the latest image
docker pull ghcr.io/ardas2012/german-credit-risk-prediction:main

# Run the container
docker run -d \
  -p 8000:8000 \
  --name credit-risk-api \
  --restart unless-stopped \
  ghcr.io/ardas2012/german-credit-risk-prediction:main
```

### Step 2: Verify API is Running

```bash
# Check container status
docker ps

# Test API health
curl http://localhost:8000/health
```

Expected response:
```json
{"status": "healthy", "timestamp": "2024-01-XX..."}
```

### Step 3: Start Web Application

```bash
# Clone repository
git clone https://github.com/ArdaS2012/german-credit-risk-prediction.git
cd german-credit-risk-prediction/credit-risk-webapp

# Install dependencies and start
npm install
npm start
```

The web application will open automatically at http://localhost:3000

---

## 🛠️ Option 2: Build from Source

### Step 1: Clone Repository

```bash
git clone https://github.com/ArdaS2012/german-credit-risk-prediction.git
cd german-credit-risk-prediction
```

### Step 2: Build and Run API

```bash
# Build Docker image
docker build -t credit-risk-api .

# Run container
docker run -d -p 8000:8000 --name credit-risk-api credit-risk-api
```

### Step 3: Start Web Application

```bash
cd credit-risk-webapp
npm install && npm start
```

---

## 🌐 Production Deployment

### Docker Compose (Recommended for Production)

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  api:
    image: ghcr.io/ardas2012/german-credit-risk-prediction:main
    ports:
      - "8000:8000"
    restart: unless-stopped
    environment:
      - ENV=production
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  webapp:
    build: ./credit-risk-webapp
    ports:
      - "3000:3000"
    depends_on:
      - api
    restart: unless-stopped
```

Deploy:
```bash
docker-compose up -d
```

### Cloud Deployment Options

#### AWS ECS/Fargate
```bash
# Use the public image
ghcr.io/ardas2012/german-credit-risk-prediction:main
```

#### Google Cloud Run
```bash
gcloud run deploy credit-risk-api \
  --image ghcr.io/ardas2012/german-credit-risk-prediction:main \
  --port 8000 \
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

---

## 🔧 Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | API server port |
| `ENV` | development | Environment (development/production) |
| `LOG_LEVEL` | INFO | Logging level |

### API Configuration

```bash
# Custom port
docker run -d -p 9000:8000 -e PORT=8000 ghcr.io/ardas2012/german-credit-risk-prediction:main

# Production mode
docker run -d -p 8000:8000 -e ENV=production ghcr.io/ardas2012/german-credit-risk-prediction:main
```

---

## 📊 Monitoring & Health Checks

### Health Endpoints

- **Health Check**: `GET /health`
- **Metrics**: `GET /metrics` (Prometheus format)
- **API Documentation**: `GET /docs`

### Monitoring Setup

```bash
# Check logs
docker logs credit-risk-api

# Monitor metrics
curl http://localhost:8000/metrics

# Health check
curl http://localhost:8000/health
```

---

## 🧪 Testing the Deployment

### 1. API Test

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

Expected response:
```json
{
  "prediction": "Good Credit",
  "probability": 0.9,
  "risk_score": 0.1,
  "confidence": "High"
}
```

### 2. Web App Test

1. Open http://localhost:3000
2. Fill out the credit assessment form
3. Click "Assess Credit Risk"
4. Verify results are displayed

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000

# Kill process or use different port
docker run -d -p 8001:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main
```

#### 2. CORS Errors
The latest image includes CORS configuration. If you still see CORS errors:
```bash
# Ensure you're using the latest image
docker pull ghcr.io/ardas2012/german-credit-risk-prediction:main
docker stop credit-risk-api && docker rm credit-risk-api
docker run -d -p 8000:8000 --name credit-risk-api ghcr.io/ardas2012/german-credit-risk-prediction:main
```

#### 3. Container Won't Start
```bash
# Check logs
docker logs credit-risk-api

# Check container status
docker ps -a
```

#### 4. Web App Won't Connect to API
```bash
# Verify API is running
curl http://localhost:8000/health

# Check if ports are correct
docker port credit-risk-api
```

### Getting Help

1. **Check the logs**: `docker logs credit-risk-api`
2. **Verify health**: `curl http://localhost:8000/health`
3. **Check GitHub Issues**: [Repository Issues](https://github.com/ArdaS2012/german-credit-risk-prediction/issues)

---

## 📚 Additional Resources

- **API Documentation**: http://localhost:8000/docs
- **Usage Guide**: [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Web App Demo**: [WEBAPP_DEMO_GUIDE.md](WEBAPP_DEMO_GUIDE.md)
- **User Access Guide**: [USER_ACCESS_GUIDE.md](USER_ACCESS_GUIDE.md)

---

## 🔄 Updates

To get the latest version:

```bash
# Pull latest API image
docker pull ghcr.io/ardas2012/german-credit-risk-prediction:main

# Update web app
cd german-credit-risk-prediction
git pull origin main
cd credit-risk-webapp
npm install
```

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**🎉 You're all set!** The German Credit Risk Prediction system is now running and ready to assess credit applications through both the web interface and API endpoints. 