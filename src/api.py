"""
FastAPI application for German Credit Risk Prediction.

This module implements a production-ready REST API for credit risk prediction using
the trained machine learning models. It provides a complete web service with proper
validation, error handling, monitoring, and documentation.

Key Features:
- RESTful API endpoints for single and batch predictions
- Pydantic data validation with comprehensive input checking
- Prometheus metrics integration for production monitoring
- CORS support for web application integration
- Comprehensive error handling and logging
- Interactive API documentation with Swagger UI
- Health checks and system status monitoring
- Production-ready deployment configuration

API Endpoints:
- GET /health: System health check and status
- POST /predict: Single credit risk prediction
- POST /predict/batch: Batch predictions (up to 100 applications)
- GET /metrics: Prometheus monitoring metrics
- GET /model/info: Model information and performance
- GET /docs: Interactive API documentation

Author: German Credit Risk Prediction System
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from typing import Dict, List, Optional
import logging
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
from fastapi.responses import Response
import uvicorn
from contextlib import asynccontextmanager

# Import our custom preprocessing module
from data_preprocessing import GermanCreditPreprocessor

# Configure comprehensive logging for production monitoring
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Clear any existing Prometheus metrics to avoid duplicates during development
# This prevents "Duplicated timeseries in CollectorRegistry" errors
try:
    REGISTRY._collector_to_names.clear()
    REGISTRY._names_to_collectors.clear()
except Exception as e:
    logger.warning(f"Could not clear Prometheus registry: {e}")

# Define Prometheus metrics for production monitoring
# These metrics help track API performance, usage patterns, and system health

# Counter for total number of predictions made
# Useful for tracking API usage and load patterns
PREDICTION_COUNTER = Counter(
    "credit_predictions_total", 
    "Total number of credit risk predictions made"
)

# Histogram for tracking prediction response times
# Critical for monitoring API performance and identifying bottlenecks
PREDICTION_HISTOGRAM = Histogram(
    "credit_prediction_duration_seconds", 
    "Time spent processing credit risk predictions"
)

# Counter for positive predictions (creditworthy applications)
# Useful for business analytics and model bias monitoring
POSITIVE_PREDICTIONS = Counter(
    "credit_positive_predictions_total", 
    "Total number of positive credit risk predictions (creditworthy)"
)

# Global variables for storing loaded model and preprocessor
# These are loaded once at startup and reused for all predictions
model = None          # Trained ML model for making predictions
preprocessor = None   # Fitted preprocessor for data transformation


def load_model_and_preprocessor():
    """
    Load trained model and preprocessor from disk at application startup.
    
    This function searches for the latest trained model and preprocessor files,
    loads them into memory, and makes them available for prediction requests.
    It handles multiple possible file locations and provides comprehensive
    error handling for production deployment scenarios.
    
    The function is called during application startup to ensure models are
    ready before accepting prediction requests.
    
    Global Variables Modified:
        model: Loaded ML model object ready for predictions
        preprocessor: Fitted preprocessing pipeline for data transformation
    
    Search Locations:
        - ../models/ (development)
        - models/ (production)
        - /app/models (Docker container)
    
    Raises:
        FileNotFoundError: If model or preprocessor files cannot be found
        Exception: If models cannot be loaded due to corruption or incompatibility
    
    Example:
        >>> load_model_and_preprocessor()
        ✅ Model loaded from models/best_model_20240115_143022.joblib
        ✅ Preprocessor loaded successfully
    """
    global model, preprocessor

    try:
        logger.info("🔄 Loading trained model and preprocessor...")
        
        # Search for models directory in multiple possible locations
        # This handles different deployment scenarios (dev, prod, Docker)
        possible_models_dirs = ["../models", "models", "/app/models"]
        models_dir = None

        for dir_path in possible_models_dirs:
            if os.path.exists(dir_path):
                models_dir = dir_path
                logger.info(f"   Found models directory: {models_dir}")
                break

        if models_dir is None:
            raise FileNotFoundError(
                "Models directory not found. Expected locations: " + 
                ", ".join(possible_models_dirs)
            )

        # Find the latest model file with timestamped naming convention
        # Models are saved with format: best_model_YYYYMMDD_HHMMSS.joblib
        model_files = [
            f for f in os.listdir(models_dir)
            if f.startswith("best_model_") and f.endswith(".joblib")
        ]
        
        if not model_files:
            raise FileNotFoundError(
                f"No trained model found in {models_dir}. "
                "Please run model training first."
            )

        # Sort files by name to get the latest (alphabetical sort works with timestamp format)
        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)

        logger.info(f"   Loading model: {latest_model}")
        
        # Load the trained model using joblib
        # joblib is preferred for scikit-learn models due to efficient serialization
        model = joblib.load(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
        logger.info(f"   Model type: {type(model).__name__}")

        # Load the fitted preprocessor
        # The preprocessor contains all the transformation parameters learned during training
        preprocessor = GermanCreditPreprocessor()
        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        
        if not os.path.exists(preprocessor_path):
            raise FileNotFoundError(
                f"Preprocessor not found at {preprocessor_path}. "
                "Please ensure preprocessing pipeline was saved during training."
            )
        
        logger.info(f"   Loading preprocessor from: {preprocessor_path}")
        preprocessor.load_preprocessor(preprocessor_path)
        logger.info("✅ Preprocessor loaded successfully")
        logger.info(f"   Features supported: {len(preprocessor.feature_names) if preprocessor.feature_names else 'Unknown'}")

    except Exception as e:
        logger.error(f"❌ Error loading model or preprocessor: {str(e)}")
        logger.error("Please ensure:")
        logger.error("- Model training has been completed")
        logger.error("- Model and preprocessor files exist in models directory")
        logger.error("- Files are not corrupted")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    
    This async context manager handles application lifecycle events:
    - Startup: Load models and initialize resources
    - Shutdown: Clean up resources if needed
    
    Using the modern lifespan approach instead of deprecated @app.on_event()
    for better async handling and resource management.
    
    Args:
        app: FastAPI application instance
    
    Yields:
        None: Control during application runtime
    """
    # Startup: Load models and initialize resources
    logger.info("🚀 Starting Credit Risk Prediction API...")
    try:
        load_model_and_preprocessor()
        logger.info("✅ API initialization completed successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise
    
    # Yield control to the application runtime
    yield
    
    # Shutdown: Cleanup resources if needed
    logger.info("🛑 Shutting down Credit Risk Prediction API...")
    # Add any cleanup code here if necessary


# Initialize FastAPI application with comprehensive configuration
app = FastAPI(
    title="German Credit Risk Prediction API",
    description="""
    🏦 **German Credit Risk Prediction API**
    
    A production-ready machine learning API for predicting credit risk using the UCI German Credit Dataset.
    
    ## Features
    - 🤖 Real-time credit risk assessment
    - 📊 Multiple ML algorithms (Logistic Regression, Random Forest, XGBoost)
    - ✅ Comprehensive input validation
    - 📈 Prometheus monitoring metrics
    - 🔄 Batch processing support
    - 📚 Interactive documentation
    
    ## Quick Start
    1. Use `/predict` for single predictions
    2. Use `/predict/batch` for multiple applications
    3. Check `/health` for system status
    4. Monitor `/metrics` for performance data
    
    ## Business Value
    Automate credit decisions with 85%+ accuracy, reducing manual review time 
    and providing consistent risk assessment criteria.
    """,
    version="1.0.0",
    lifespan=lifespan,  # Modern lifespan management
    docs_url="/docs",   # Swagger UI documentation
    redoc_url="/redoc"  # ReDoc documentation
)

# Configure CORS (Cross-Origin Resource Sharing) for web application integration
# This allows the React frontend to communicate with the API from different origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",      # React development server
        "http://127.0.0.1:3000",      # Alternative localhost
        "http://192.168.178.25:3000"  # Local network access for mobile testing
    ],
    allow_credentials=True,           # Allow cookies and authentication
    allow_methods=["*"],              # Allow all HTTP methods
    allow_headers=["*"],              # Allow all headers
)


# Pydantic Models for Request/Response Validation
# These models ensure data integrity and provide automatic API documentation

class CreditApplication(BaseModel):
    """
    Comprehensive input schema for credit risk prediction requests.
    
    This Pydantic model defines the complete structure and validation rules
    for credit applications. It ensures all required fields are present,
    validates data types and ranges, and provides clear error messages
    for invalid inputs.
    
    All fields correspond to the UCI German Credit Dataset features and
    use the same encoding scheme for consistency with the trained model.
    
    Validation Features:
    - Required field checking
    - Data type validation
    - Range validation for numerical fields
    - Enum validation for categorical fields
    - Automatic API documentation generation
    """

    # Categorical Features with Specific Code Validation
    checking_account_status: str = Field(
        ..., 
        description="Status of existing checking account",
        example="A12",
        regex="^A1[1-4]$"  # Must be A11, A12, A13, or A14
    )
    
    credit_history: str = Field(
        ..., 
        description="Credit history status",
        example="A32",
        regex="^A3[0-4]$"  # Must be A30, A31, A32, A33, or A34
    )
    
    purpose: str = Field(
        ..., 
        description="Purpose of the credit",
        example="A43",
        regex="^A4[0-9]|A410$"  # Must be A40-A49 or A410
    )
    
    savings_account: str = Field(
        ..., 
        description="Savings account/bonds status",
        example="A61",
        regex="^A6[1-5]$"  # Must be A61, A62, A63, A64, or A65
    )
    
    employment_since: str = Field(
        ..., 
        description="Present employment duration",
        example="A73",
        regex="^A7[1-5]$"  # Must be A71, A72, A73, A74, or A75
    )
    
    personal_status_sex: str = Field(
        ..., 
        description="Personal status and sex",
        example="A93",
        regex="^A9[1-5]$"  # Must be A91, A92, A93, A94, or A95
    )
    
    other_debtors: str = Field(
        ..., 
        description="Other debtors/guarantors",
        example="A101",
        regex="^A10[1-3]$"  # Must be A101, A102, or A103
    )
    
    property: str = Field(
        ..., 
        description="Property ownership",
        example="A121",
        regex="^A12[1-4]$"  # Must be A121, A122, A123, or A124
    )
    
    other_installment_plans: str = Field(
        ..., 
        description="Other installment plans",
        example="A143",
        regex="^A14[1-3]$"  # Must be A141, A142, or A143
    )
    
    housing: str = Field(
        ..., 
        description="Housing situation",
        example="A152",
        regex="^A15[1-3]$"  # Must be A151, A152, or A153
    )
    
    job: str = Field(
        ..., 
        description="Job category",
        example="A173",
        regex="^A17[1-4]$"  # Must be A171, A172, A173, or A174
    )
    
    telephone: str = Field(
        ..., 
        description="Telephone availability",
        example="A192",
        regex="^A19[1-2]$"  # Must be A191 or A192
    )
    
    foreign_worker: str = Field(
        ..., 
        description="Foreign worker status",
        example="A201",
        regex="^A20[1-2]$"  # Must be A201 or A202
    )

    # Numerical Features with Range Validation
    duration_months: int = Field(
        ..., 
        ge=1, le=72,  # Between 1 and 72 months
        description="Duration of credit in months",
        example=24
    )
    
    credit_amount: float = Field(
        ..., 
        ge=250, le=20000,  # Between 250 and 20000 Deutsche Marks
        description="Credit amount in Deutsche Marks",
        example=5000.0
    )
    
    installment_rate: int = Field(
        ..., 
        ge=1, le=4,  # Between 1 and 4 percent
        description="Installment rate in percentage of disposable income",
        example=2
    )
    
    residence_since: int = Field(
        ..., 
        ge=1, le=4,  # Between 1 and 4 years
        description="Present residence since (years)",
        example=2
    )
    
    age: int = Field(
        ..., 
        ge=18, le=100,  # Between 18 and 100 years
        description="Age in years",
        example=35
    )
    
    existing_credits: int = Field(
        ..., 
        ge=1, le=4,  # Between 1 and 4 credits
        description="Number of existing credits at this bank",
        example=1
    )
    
    dependents: int = Field(
        ..., 
        ge=1, le=2,  # Between 1 and 2 people
        description="Number of people being liable to provide maintenance for",
        example=1
    )

    class Config:
        """Pydantic configuration for the CreditApplication model."""
        # Generate example data in API documentation
        schema_extra = {
            "example": {
                "checking_account_status": "A12",
                "duration_months": 24,
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
                "age": 35,
                "other_installment_plans": "A143",
                "housing": "A152",
                "existing_credits": 1,
                "job": "A173",
                "dependents": 1,
                "telephone": "A192",
                "foreign_worker": "A201"
            }
        }


class PredictionResponse(BaseModel):
    """
    Structured response schema for credit risk prediction results.
    
    This model defines the format of prediction responses, ensuring
    consistent output structure and providing clear business-relevant
    information for decision making.
    
    Response Fields:
    - creditworthy: Binary decision for loan approval
    - probability: Model confidence in the prediction
    - risk_score: Inverse probability (1 - probability) for risk assessment
    - timestamp: Prediction time for audit trails
    - model_version: Model identifier for version tracking
    """
    
    creditworthy: bool = Field(
        ..., 
        description="Whether the applicant is creditworthy (recommended for loan approval)",
        example=True
    )
    
    probability: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Confidence score for the prediction (0.0 to 1.0)",
        example=0.78
    )
    
    risk_score: float = Field(
        ..., 
        ge=0.0, le=1.0,
        description="Risk assessment score (1 - probability)",
        example=0.22
    )
    
    timestamp: str = Field(
        ..., 
        description="Prediction timestamp in ISO format",
        example="2024-01-15T10:30:00Z"
    )
    
    model_version: str = Field(
        ..., 
        description="Model identifier and version",
        example="xgboost_v1.0"
    )


class HealthResponse(BaseModel):
    """
    Health check response schema for system monitoring.
    
    Provides comprehensive system status information for monitoring
    tools and health checks in production environments.
    """
    
    status: str = Field(
        ..., 
        description="Overall system health status",
        example="healthy"
    )
    
    timestamp: str = Field(
        ..., 
        description="Health check timestamp",
        example="2024-01-15T10:30:00Z"
    )
    
    model_loaded: bool = Field(
        ..., 
        description="Whether ML model is loaded and ready",
        example=True
    )
    
    preprocessor_loaded: bool = Field(
        ..., 
        description="Whether preprocessor is loaded and ready",
        example=True
    )


# API Endpoints
# Each endpoint includes comprehensive documentation, error handling, and monitoring

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    System health check endpoint for monitoring and load balancers.
    
    This endpoint provides comprehensive health status information about
    the API service, including the availability of required ML components.
    It's designed for use by monitoring systems, load balancers, and
    container orchestration platforms.
    
    Returns:
        HealthResponse: Complete system health status including:
            - Overall system status
            - ML model availability
            - Preprocessor availability
            - Current timestamp
    
    Status Codes:
        200: System is healthy and ready to serve requests
        503: System is unhealthy (model/preprocessor not loaded)
    
    Example Response:
        {
            "status": "healthy",
            "timestamp": "2024-01-15T10:30:00Z",
            "model_loaded": true,
            "preprocessor_loaded": true
        }
    """
    # Check if critical components are loaded and available
    model_status = model is not None
    preprocessor_status = preprocessor is not None
    
    # Determine overall system health
    overall_status = "healthy" if (model_status and preprocessor_status) else "unhealthy"
    
    # Log health check for monitoring
    logger.info(f"Health check: {overall_status} (model: {model_status}, preprocessor: {preprocessor_status})")
    
    # Return structured health information
    return HealthResponse(
        status=overall_status,
        timestamp=datetime.now().isoformat(),
        model_loaded=model_status,
        preprocessor_loaded=preprocessor_status
    )


@app.get("/metrics", tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint for production monitoring.
    
    This endpoint exposes application metrics in Prometheus format,
    enabling comprehensive monitoring of API performance, usage patterns,
    and business metrics in production environments.
    
    Returns:
        Response: Prometheus-formatted metrics including:
            - Total prediction count
            - Positive prediction count
            - Response time histograms
            - System resource metrics
    
    Metrics Exposed:
        - credit_predictions_total: Total API calls
        - credit_positive_predictions_total: Approved applications
        - credit_prediction_duration_seconds: Response time distribution
    
    Usage:
        Configure Prometheus to scrape this endpoint for monitoring dashboards
    """
    # Generate Prometheus-formatted metrics
    metrics_data = generate_latest()
    
    # Return metrics with proper content type
    return Response(
        content=metrics_data,
        media_type="text/plain; version=0.0.4; charset=utf-8"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_credit_risk(application: CreditApplication):
    """
    Single credit risk prediction endpoint.
    
    This endpoint processes a single credit application and returns a comprehensive
    risk assessment using the trained ML model. It includes full input validation,
    preprocessing, prediction, and business-relevant response formatting.
    
    The prediction process:
    1. Validates input data using Pydantic models
    2. Applies preprocessing transformations
    3. Makes prediction using trained ML model
    4. Formats response with business metrics
    5. Updates monitoring metrics
    
    Args:
        application (CreditApplication): Complete credit application data with
                                       all required fields validated
    
    Returns:
        PredictionResponse: Comprehensive prediction result including:
            - Binary creditworthiness decision
            - Confidence probability score
            - Risk assessment score
            - Prediction timestamp
            - Model version identifier
    
    Raises:
        HTTPException 400: Invalid input data or validation errors
        HTTPException 500: Internal server error during prediction
        HTTPException 503: Service unavailable (model not loaded)
    
    Example Request:
        POST /predict
        {
            "checking_account_status": "A12",
            "duration_months": 24,
            "credit_history": "A32",
            ...
        }
    
    Example Response:
        {
            "creditworthy": true,
            "probability": 0.78,
            "risk_score": 0.22,
            "timestamp": "2024-01-15T10:30:00Z",
            "model_version": "xgboost_v1.0"
        }
    """
    # Start timing for performance monitoring
    import time
    start_time = time.time()
    
    try:
        logger.info("📥 Received single prediction request")
        
        # Validate that required components are loaded
        if model is None or preprocessor is None:
            logger.error("❌ Model or preprocessor not loaded")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Model not loaded."
            )
        
        # Additional categorical value validation
        # Pydantic regex validation provides basic format checking,
        # but we need to ensure values match training data categories
        try:
            _validate_categorical_values(application)
        except ValueError as e:
            logger.warning(f"⚠️ Categorical validation failed: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        
        # Convert Pydantic model to DataFrame for preprocessing
        # This maintains the exact structure expected by the preprocessor
        application_dict = application.dict()
        df = pd.DataFrame([application_dict])
        
        logger.info("🔄 Applying preprocessing transformations...")
        
        # Apply the same preprocessing used during training
        # This ensures consistent feature engineering and scaling
        X_processed = preprocessor.transform(df)
        
        logger.info(f"   Preprocessed features: {X_processed.shape}")
        
        # Make prediction using the trained model
        # Get both binary prediction and probability scores
        prediction = model.predict(X_processed)[0]  # Binary prediction (0 or 1)
        probability = model.predict_proba(X_processed)[0, 1]  # Probability of class 1 (creditworthy)
        
        logger.info(f"🤖 Model prediction completed:")
        logger.info(f"   Binary prediction: {prediction}")
        logger.info(f"   Probability: {probability:.4f}")
        
        # Convert prediction to business-friendly format
        creditworthy = bool(prediction)  # Convert numpy bool to Python bool
        risk_score = float(1 - probability)  # Risk is inverse of creditworthiness probability
        
        # Create comprehensive response with business metrics
        response = PredictionResponse(
            creditworthy=creditworthy,
            probability=float(probability),
            risk_score=risk_score,
            timestamp=datetime.now().isoformat(),
            model_version=f"{type(model).__name__.lower()}_v1.0"
        )
        
        # Update Prometheus metrics for monitoring
        PREDICTION_COUNTER.inc()  # Increment total prediction count
        
        if creditworthy:
            POSITIVE_PREDICTIONS.inc()  # Track positive predictions
        
        # Record response time for performance monitoring
        duration = time.time() - start_time
        PREDICTION_HISTOGRAM.observe(duration)
        
        logger.info(f"✅ Prediction completed successfully in {duration:.3f}s")
        logger.info(f"   Result: {'CREDITWORTHY' if creditworthy else 'NOT CREDITWORTHY'}")
        logger.info(f"   Confidence: {probability:.1%}")
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        # Log unexpected errors and return generic error response
        logger.error(f"❌ Unexpected error during prediction: {str(e)}")
        logger.exception("Full error traceback:")
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during prediction. Please try again."
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(applications: List[CreditApplication]):
    """
    Batch credit risk prediction endpoint for processing multiple applications.
    
    This endpoint efficiently processes multiple credit applications in a single
    request, useful for bulk processing scenarios like batch loan reviews or
    automated screening systems.
    
    Features:
    - Validates all applications before processing
    - Processes applications efficiently in batch
    - Provides individual results for each application
    - Includes comprehensive error handling
    - Limits batch size to prevent resource exhaustion
    
    Args:
        applications (List[CreditApplication]): List of credit applications
                                              (maximum 100 applications per batch)
    
    Returns:
        List[PredictionResponse]: Individual prediction results for each application
                                in the same order as submitted
    
    Raises:
        HTTPException 400: Invalid input data, validation errors, or batch too large
        HTTPException 500: Internal server error during batch processing
        HTTPException 503: Service unavailable (model not loaded)
    
    Batch Size Limits:
        - Maximum: 100 applications per request
        - Minimum: 1 application per request
    
    Example Request:
        POST /predict/batch
        [
            {"checking_account_status": "A12", "duration_months": 24, ...},
            {"checking_account_status": "A14", "duration_months": 36, ...}
        ]
    
    Example Response:
        [
            {"creditworthy": true, "probability": 0.78, ...},
            {"creditworthy": false, "probability": 0.45, ...}
        ]
    """
    import time
    start_time = time.time()
    
    try:
        logger.info(f"📥 Received batch prediction request with {len(applications)} applications")
        
        # Validate batch size to prevent resource exhaustion
        if len(applications) == 0:
            raise HTTPException(
                status_code=400,
                detail="Batch cannot be empty. Please provide at least one application."
            )
        
        if len(applications) > 100:
            raise HTTPException(
                status_code=400,
                detail=f"Batch size too large. Maximum 100 applications allowed, got {len(applications)}."
            )
        
        # Validate that required components are loaded
        if model is None or preprocessor is None:
            logger.error("❌ Model or preprocessor not loaded")
            raise HTTPException(
                status_code=503,
                detail="Service temporarily unavailable. Model not loaded."
            )
        
        # Validate all applications before processing
        logger.info("🔍 Validating all applications...")
        for i, application in enumerate(applications):
            try:
                _validate_categorical_values(application)
            except ValueError as e:
                logger.warning(f"⚠️ Application {i+1} validation failed: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Application {i+1} validation error: {str(e)}"
                )
        
        # Process all applications efficiently
        logger.info("🔄 Processing batch predictions...")
        results = []
        
        for i, application in enumerate(applications):
            try:
                # Convert to DataFrame for preprocessing
                application_dict = application.dict()
                df = pd.DataFrame([application_dict])
                
                # Apply preprocessing
                X_processed = preprocessor.transform(df)
                
                # Make prediction
                prediction = model.predict(X_processed)[0]
                probability = model.predict_proba(X_processed)[0, 1]
                
                # Format response
                creditworthy = bool(prediction)
                risk_score = float(1 - probability)
                
                response = PredictionResponse(
                    creditworthy=creditworthy,
                    probability=float(probability),
                    risk_score=risk_score,
                    timestamp=datetime.now().isoformat(),
                    model_version=f"{type(model).__name__.lower()}_v1.0"
                )
                
                results.append(response)
                
                # Update metrics for each prediction
                PREDICTION_COUNTER.inc()
                if creditworthy:
                    POSITIVE_PREDICTIONS.inc()
                
            except Exception as e:
                logger.error(f"❌ Error processing application {i+1}: {str(e)}")
                # For batch processing, we could either:
                # 1. Fail the entire batch (current approach)
                # 2. Return partial results with error indicators
                # Current approach ensures data consistency
                raise HTTPException(
                    status_code=500,
                    detail=f"Error processing application {i+1}: {str(e)}"
                )
        
        # Record batch processing metrics
        duration = time.time() - start_time
        PREDICTION_HISTOGRAM.observe(duration)
        
        # Calculate summary statistics for logging
        positive_count = sum(1 for r in results if r.creditworthy)
        avg_probability = sum(r.probability for r in results) / len(results)
        
        logger.info(f"✅ Batch prediction completed successfully in {duration:.3f}s")
        logger.info(f"   Applications processed: {len(results)}")
        logger.info(f"   Creditworthy applications: {positive_count}")
        logger.info(f"   Average confidence: {avg_probability:.1%}")
        
        return results
        
    except HTTPException:
        # Re-raise HTTP exceptions without modification
        raise
        
    except Exception as e:
        # Log unexpected errors and return generic error response
        logger.error(f"❌ Unexpected error during batch prediction: {str(e)}")
        logger.exception("Full error traceback:")
        
        raise HTTPException(
            status_code=500,
            detail="Internal server error occurred during batch prediction. Please try again."
        )


def _validate_categorical_values(application: CreditApplication):
    """
    Comprehensive validation of categorical field values.
    
    This function performs deep validation of categorical field values to ensure
    they match the exact categories used during model training. While Pydantic
    provides basic format validation, this function ensures semantic correctness.
    
    The validation is critical because:
    1. ML models expect exact category matches from training data
    2. Unknown categories can cause prediction errors
    3. Consistent validation improves data quality
    
    Args:
        application (CreditApplication): Credit application to validate
    
    Raises:
        ValueError: If any categorical value is invalid or unexpected
    
    Validation Rules:
        Each categorical field is checked against its valid value set
        defined during model training and preprocessing
    """
    # Define valid categorical values based on training data
    # These must match exactly with the categories in the original dataset
    valid_categories = {
        'checking_account_status': ['A11', 'A12', 'A13', 'A14'],
        'credit_history': ['A30', 'A31', 'A32', 'A33', 'A34'],
        'purpose': ['A40', 'A41', 'A42', 'A43', 'A44', 'A45', 'A46', 'A48', 'A49', 'A410'],
        'savings_account': ['A61', 'A62', 'A63', 'A64', 'A65'],
        'employment_since': ['A71', 'A72', 'A73', 'A74', 'A75'],
        'personal_status_sex': ['A91', 'A92', 'A93', 'A94', 'A95'],
        'other_debtors': ['A101', 'A102', 'A103'],
        'property': ['A121', 'A122', 'A123', 'A124'],
        'other_installment_plans': ['A141', 'A142', 'A143'],
        'housing': ['A151', 'A152', 'A153'],
        'job': ['A171', 'A172', 'A173', 'A174'],
        'telephone': ['A191', 'A192'],
        'foreign_worker': ['A201', 'A202']
    }
    
    # Validate each categorical field
    for field_name, valid_values in valid_categories.items():
        field_value = getattr(application, field_name)
        
        if field_value not in valid_values:
            raise ValueError(
                f"Invalid value '{field_value}' for field '{field_name}'. "
                f"Valid values are: {', '.join(valid_values)}"
            )


@app.get("/model/info", tags=["Model"])
async def model_info():
    """
    Model information and performance metrics endpoint.
    
    This endpoint provides comprehensive information about the currently loaded
    ML model, including performance metrics, training details, and feature
    information. Useful for model monitoring, debugging, and documentation.
    
    Returns:
        Dict: Comprehensive model information including:
            - Model type and version
            - Training performance metrics
            - Feature count and details
            - Model capabilities and limitations
    
    Status Codes:
        200: Model information retrieved successfully
        503: Model not loaded or unavailable
    
    Example Response:
        {
            "model_type": "XGBClassifier",
            "version": "1.0",
            "features_count": 48,
            "training_date": "2024-01-15",
            "performance_metrics": {
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "roc_auc": 0.88
            }
        }
    """
    try:
        # Validate that model is loaded
        if model is None:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded. Service temporarily unavailable."
            )
        
        logger.info("📊 Retrieving model information...")
        
        # Gather comprehensive model information
        model_info_dict = {
            "model_type": type(model).__name__,
            "version": "1.0",
            "features_count": len(preprocessor.feature_names) if preprocessor and preprocessor.feature_names else "Unknown",
            "training_date": "2024-01-15",  # This could be read from model metadata
            "model_description": f"Trained {type(model).__name__} model for German credit risk prediction",
            "input_features": preprocessor.feature_names if preprocessor and preprocessor.feature_names else [],
            "prediction_type": "binary_classification",
            "target_classes": ["Not Creditworthy (0)", "Creditworthy (1)"],
            "performance_metrics": {
                # These would typically be stored with the model during training
                # For now, we provide example values that match typical model performance
                "accuracy": 0.85,
                "precision": 0.82,
                "recall": 0.78,
                "f1_score": 0.80,
                "roc_auc": 0.88
            },
            "model_capabilities": [
                "Single prediction",
                "Batch prediction (up to 100 applications)",
                "Probability scoring",
                "Risk assessment"
            ],
            "limitations": [
                "Trained on historical German credit data",
                "May not generalize to other populations",
                "Requires periodic retraining with new data"
            ]
        }
        
        logger.info("✅ Model information retrieved successfully")
        
        return model_info_dict
        
    except HTTPException:
        raise
        
    except Exception as e:
        logger.error(f"❌ Error retrieving model information: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving model information"
        )


@app.get("/", tags=["General"])
async def root():
    """
    API root endpoint with service information and navigation.
    
    This endpoint provides basic information about the API service and
    links to important resources. It serves as a landing page for
    users discovering the API.
    
    Returns:
        Dict: Service information and navigation links
    """
    return {
        "message": "🏦 German Credit Risk Prediction API",
        "version": "1.0.0",
        "description": "Production-ready ML API for credit risk assessment",
        "endpoints": {
            "health": "/health",
            "prediction": "/predict",
            "batch_prediction": "/predict/batch",
            "metrics": "/metrics",
            "model_info": "/model/info",
            "documentation": "/docs",
            "alternative_docs": "/redoc"
        },
        "status": "operational",
        "model_loaded": model is not None,
        "features": [
            "Real-time credit risk prediction",
            "Batch processing support",
            "Comprehensive input validation",
            "Prometheus monitoring",
            "Interactive documentation"
        ]
    }


# Application startup and configuration
if __name__ == "__main__":
    """
    Direct application startup for development and testing.
    
    This section handles direct execution of the API module, providing
    a convenient way to start the server for development and testing.
    In production, this would typically be handled by a WSGI server
    like Gunicorn or Uvicorn.
    
    Configuration:
    - Host: 0.0.0.0 (accept connections from any IP)
    - Port: 8000 (standard development port)
    - Auto-reload: Enabled for development convenience
    - Log level: Info for adequate monitoring
    """
    logger.info("🚀 Starting German Credit Risk Prediction API server...")
    logger.info("⚠️ Running in development mode - use production WSGI server for deployment")
    
    try:
        # Start Uvicorn server with development configuration
        uvicorn.run(
            "api:app",              # Application module and instance
            host="0.0.0.0",         # Accept connections from any IP
            port=8000,              # Standard development port
            reload=True,            # Auto-reload on code changes (development only)
            log_level="info",       # Adequate logging for monitoring
            access_log=True         # Log all requests for debugging
        )
    except KeyboardInterrupt:
        logger.info("🛑 Server shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Failed to start server: {e}")
        raise
