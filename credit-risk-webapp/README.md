# Credit Risk Assessment Web Application

A responsive React web application that provides a user-friendly interface for assessing credit risk using the deployed German Credit Risk Prediction model.

## 🌟 Features

- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Real-time Validation**: Form validation with helpful error messages
- **Beautiful UI**: Modern design using Tailwind CSS
- **Loading States**: Visual feedback during API calls
- **Error Handling**: Comprehensive error handling with user-friendly messages
- **Results Display**: Clear visualization of credit risk assessment results

## 🚀 Quick Start

### Prerequisites

- Node.js (version 14 or higher)
- npm or yarn
- Your deployed Credit Risk API running on `localhost:8000`

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   git clone https://github.com/ArdaS2012/german-credit-risk-prediction.git
   cd german-credit-risk-prediction/credit-risk-webapp
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start the development server**:
   ```bash
   npm start
   ```

4. **Open your browser** and navigate to:
   ```
   http://localhost:3000
   ```

## 🏦 Using the Web Application

### Step 1: Start Your API Server

Before using the web application, ensure your credit risk API is running:

```bash
# Option 1: Using Docker (Recommended)
docker run -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main

# Option 2: Local development
cd src
python api.py
```

Verify the API is running by visiting: http://localhost:8000/docs

### Step 2: Access the Web Application

1. Open your browser and go to `http://localhost:3000`
2. You'll see the Credit Risk Assessment form

### Step 3: Fill Out the Assessment Form

The form includes all required fields from the German Credit Dataset:

#### Personal Information
- **Age in years**: Enter age between 18-100
- **Personal status and sex**: Select from dropdown options
- **Number of people being liable to provide maintenance**: 1 or 2

#### Financial Information
- **Status of existing checking account**: Current account balance range
- **Credit amount (DM)**: Amount between 250-20,000 DM
- **Duration in months**: Loan duration (1-72 months)
- **Savings account/bonds**: Current savings range
- **Installment rate**: Percentage of disposable income (1-4%)

#### Employment & Housing
- **Present employment since**: Employment duration category
- **Job**: Employment type and skill level
- **Housing**: Housing situation (own, rent, free)
- **Present residence since**: Years at current address (1-4)

#### Credit History & Purpose
- **Credit history**: Previous credit performance
- **Purpose**: Reason for the loan
- **Other installment plans**: Existing payment plans
- **Number of existing credits**: Current credit accounts (1-4)

#### Additional Information
- **Property**: Type of property owned
- **Other debtors/guarantors**: Co-signers or guarantors
- **Telephone**: Phone availability
- **Foreign worker**: Work permit status

### Step 4: Submit and View Results

1. **Fill all required fields** (marked with red asterisk *)
2. **Click "Assess Credit Risk"** button
3. **Wait for processing** (loading spinner will appear)
4. **View your results**:
   - **Risk Level**: Low Risk (Good Credit) or High Risk (Bad Credit)
   - **Creditworthiness**: Approved or Declined
   - **Confidence Score**: Model's confidence percentage
   - **Risk Score**: Detailed risk assessment

### Step 5: Assess Another Application

Click "Assess Another Application" to reset the form and evaluate a new case.

## 📱 Mobile-Friendly Design

The application is fully responsive and works great on:
- **Desktop**: Full two-column layout
- **Tablet**: Adaptive layout with proper spacing
- **Mobile**: Single-column layout optimized for touch

## 🔧 Configuration

### API Endpoint Configuration

By default, the app connects to `http://localhost:8000`. To change this:

1. Open `src/components/CreditRiskForm.js`
2. Find the API call in the `handleSubmit` function
3. Update the URL:
   ```javascript
   const response = await axios.post('YOUR_API_URL/predict', apiData, {
   ```

### Environment Variables (Production)

For production deployment, create a `.env` file:

```env
REACT_APP_API_URL=https://your-api-domain.com
```

Then update the code to use:
```javascript
const apiUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';
```

## 🚀 Production Deployment

### Build for Production

```bash
npm run build
```

This creates a `build` folder with optimized production files.

### Deploy Options

#### 1. Static Hosting (Netlify, Vercel, GitHub Pages)

```bash
# Build the app
npm run build

# Deploy the build folder to your hosting service
```

#### 2. Docker Deployment

Create a `Dockerfile` in the webapp directory:

```dockerfile
FROM node:16-alpine as build
WORKDIR /app
COPY package*.json ./
RUN npm install
COPY . .
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

Build and run:
```bash
docker build -t credit-risk-webapp .
docker run -p 3000:80 credit-risk-webapp
```

#### 3. Cloud Deployment

- **AWS S3 + CloudFront**: Static hosting with CDN
- **Google Cloud Storage**: Static website hosting
- **Azure Static Web Apps**: Integrated CI/CD

## 🔍 Troubleshooting

### Common Issues

1. **API Connection Error**:
   - Ensure the API server is running on `localhost:8000`
   - Check if Docker container is running: `docker ps`
   - Verify API health: `curl http://localhost:8000/health`

2. **CORS Issues**:
   - The API includes CORS headers for localhost:3000
   - For production, update CORS settings in the API

3. **Form Validation Errors**:
   - All fields are required
   - Check numeric field ranges (age, amounts, etc.)
   - Ensure dropdown selections are made

4. **Build Errors**:
   - Run `npm install` to ensure all dependencies are installed
   - Check Node.js version (requires 14+)

### Getting Help

- Check the browser console for error messages
- Verify API is responding: `curl http://localhost:8000/health`
- Test API directly: `curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d @sample_data.json`

## 🎯 Example Usage Flow

1. **Start API**: `docker run -p 8000:8000 ghcr.io/ardas2012/german-credit-risk-prediction:main`
2. **Start Web App**: `npm start`
3. **Open Browser**: Navigate to `http://localhost:3000`
4. **Fill Form**: Complete all required fields
5. **Submit**: Click "Assess Credit Risk"
6. **View Results**: See risk assessment and confidence scores
7. **Reset**: Click "Assess Another Application" for new assessment

## 📊 Sample Test Data

Here's sample data you can use to test the application:

**Low Risk Customer:**
- Age: 35, Credit Amount: 2000 DM, Duration: 12 months
- Checking Account: ≥ 200 DM, Savings: 500 ≤ … < 1000 DM
- Employment: 4 ≤ … < 7 years, Job: skilled employee/official

**High Risk Customer:**
- Age: 22, Credit Amount: 7500 DM, Duration: 48 months
- Checking Account: no account, Savings: < 100 DM
- Employment: unemployed, Job: unemployed/unskilled – non-resident

## 🔐 Security Notes

- This is a demonstration application
- Do not use for actual credit decisions
- Implement proper authentication for production use
- Consider data privacy regulations (GDPR, etc.)

## 📈 Next Steps

- Add user authentication
- Implement result history
- Add data export functionality
- Create admin dashboard
- Add A/B testing for different models

---

**Your Credit Risk Assessment tool is now ready for users! 🎉**

Users can simply visit the web application, fill out the form, and get instant credit risk assessments powered by your machine learning model. 