"""
FastAPI application for German Credit Risk Prediction.
Serves the trained ML model with proper validation and monitoring.
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

from data_preprocessing import GermanCreditPreprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Clear any existing metrics to avoid duplicates
try:
    REGISTRY._collector_to_names.clear()
    REGISTRY._names_to_collectors.clear()
except:
    pass

# Prometheus metrics
PREDICTION_COUNTER = Counter(
    "credit_predictions_total", "Total number of credit predictions"
)
PREDICTION_HISTOGRAM = Histogram(
    "credit_prediction_duration_seconds", "Time spent on credit predictions"
)
POSITIVE_PREDICTIONS = Counter(
    "credit_positive_predictions_total", "Total number of positive credit predictions"
)

# Global variables for model and preprocessor
model = None
preprocessor = None


def load_model_and_preprocessor():
    """Load the trained model and preprocessor."""
    global model, preprocessor

    try:
        # Find the latest model file - check multiple possible paths
        possible_models_dirs = ["../models", "models", "/app/models"]
        models_dir = None

        for dir_path in possible_models_dirs:
            if os.path.exists(dir_path):
                models_dir = dir_path
                break

        if models_dir is None:
            raise FileNotFoundError(
                "Models directory not found in any expected location"
            )

        model_files = [
            f
            for f in os.listdir(models_dir)
            if f.startswith("best_model_") and f.endswith(".joblib")
        ]
        if not model_files:
            raise FileNotFoundError("No trained model found")

        latest_model = sorted(model_files)[-1]
        model_path = os.path.join(models_dir, latest_model)

        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")

        # Load preprocessor
        preprocessor = GermanCreditPreprocessor()
        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        preprocessor.load_preprocessor(preprocessor_path)
        logger.info("Preprocessor loaded successfully")

    except Exception as e:
        logger.error(f"Error loading model or preprocessor: {str(e)}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    # Startup
    load_model_and_preprocessor()
    yield
    # Shutdown (cleanup if needed)
    pass


# Initialize FastAPI app with lifespan
app = FastAPI(
    title="German Credit Risk Prediction API",
    description="API for predicting credit risk using machine learning",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CreditApplication(BaseModel):
    """Input schema for credit risk prediction."""

    checking_account_status: str = Field(
        ..., description="Status of existing checking account (A11, A12, A13, A14)"
    )
    duration_months: int = Field(..., ge=1, le=72, description="Duration in months")
    credit_history: str = Field(
        ..., description="Credit history (A30, A31, A32, A33, A34)"
    )
    purpose: str = Field(..., description="Purpose of credit (A40-A410)")
    credit_amount: float = Field(
        ..., ge=250, le=20000, description="Credit amount in DM"
    )
    savings_account: str = Field(..., description="Savings account/bonds (A61-A65)")
    employment_since: str = Field(..., description="Present employment since (A71-A75)")
    installment_rate: int = Field(
        ...,
        ge=1,
        le=4,
        description="Installment rate in percentage of disposable income",
    )
    personal_status_sex: str = Field(
        ..., description="Personal status and sex (A91-A95)"
    )
    other_debtors: str = Field(..., description="Other debtors/guarantors (A101-A103)")
    residence_since: int = Field(..., ge=1, le=4, description="Present residence since")
    property: str = Field(..., description="Property (A121-A124)")
    age: int = Field(..., ge=18, le=100, description="Age in years")
    other_installment_plans: str = Field(
        ..., description="Other installment plans (A141-A143)"
    )
    housing: str = Field(..., description="Housing (A151-A153)")
    existing_credits: int = Field(
        ..., ge=1, le=4, description="Number of existing credits at this bank"
    )
    job: str = Field(..., description="Job (A171-A174)")
    dependents: int = Field(
        ...,
        ge=1,
        le=2,
        description="Number of people being liable to provide maintenance for",
    )
    telephone: str = Field(..., description="Telephone (A191-A192)")
    foreign_worker: str = Field(..., description="Foreign worker (A201-A202)")


class PredictionResponse(BaseModel):
    """Response schema for credit risk prediction."""

    creditworthy: bool
    probability: float
    risk_score: float
    timestamp: str
    model_version: str


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    timestamp: str
    model_loaded: bool
    preprocessor_loaded: bool


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        model_loaded=model is not None,
        preprocessor_loaded=preprocessor is not None,
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=PredictionResponse)
async def predict_credit_risk(application: CreditApplication):
    """Predict credit risk for a loan application."""

    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

    try:
        # Start timing
        with PREDICTION_HISTOGRAM.time():
            # Convert input to DataFrame
            input_data = pd.DataFrame([application.dict()])

            # Validate categorical values
            _validate_categorical_values(application)

            # Preprocess the input
            X = preprocessor.transform(input_data)

            # Make prediction
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[
                0, 1
            ]  # Probability of being creditworthy

            # Calculate risk score (inverse of creditworthy probability)
            risk_score = 1 - probability

            # Update metrics
            PREDICTION_COUNTER.inc()
            if prediction == 1:
                POSITIVE_PREDICTIONS.inc()

            # Log prediction
            logger.info(
                f"Prediction made: creditworthy={prediction}, probability={probability:.4f}"
            )

            return PredictionResponse(
                creditworthy=bool(prediction),
                probability=float(probability),
                risk_score=float(risk_score),
                timestamp=datetime.now().isoformat(),
                model_version="1.0.0",
            )

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(applications: List[CreditApplication]):
    """Batch prediction endpoint for multiple applications."""

    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model or preprocessor not loaded")

    if len(applications) > 100:
        raise HTTPException(status_code=400, detail="Batch size cannot exceed 100")

    try:
        results = []

        for app in applications:
            # Validate categorical values
            _validate_categorical_values(app)

            # Convert to DataFrame
            input_data = pd.DataFrame([app.dict()])

            # Preprocess
            X = preprocessor.transform(input_data)

            # Predict
            prediction = model.predict(X)[0]
            probability = model.predict_proba(X)[0, 1]
            risk_score = 1 - probability

            results.append(
                PredictionResponse(
                    creditworthy=bool(prediction),
                    probability=float(probability),
                    risk_score=float(risk_score),
                    timestamp=datetime.now().isoformat(),
                    model_version="1.0.0",
                )
            )

            # Update metrics
            PREDICTION_COUNTER.inc()
            if prediction == 1:
                POSITIVE_PREDICTIONS.inc()

        logger.info(f"Batch prediction completed for {len(applications)} applications")
        return results

    except Exception as e:
        logger.error(f"Error during batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


def _validate_categorical_values(application: CreditApplication):
    """Validate categorical input values."""

    valid_values = {
        "checking_account_status": ["A11", "A12", "A13", "A14"],
        "credit_history": ["A30", "A31", "A32", "A33", "A34"],
        "purpose": [
            "A40",
            "A41",
            "A42",
            "A43",
            "A44",
            "A45",
            "A46",
            "A48",
            "A49",
            "A410",
        ],
        "savings_account": ["A61", "A62", "A63", "A64", "A65"],
        "employment_since": ["A71", "A72", "A73", "A74", "A75"],
        "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
        "other_debtors": ["A101", "A102", "A103"],
        "property": ["A121", "A122", "A123", "A124"],
        "other_installment_plans": ["A141", "A142", "A143"],
        "housing": ["A151", "A152", "A153"],
        "job": ["A171", "A172", "A173", "A174"],
        "telephone": ["A191", "A192"],
        "foreign_worker": ["A201", "A202"],
    }

    for field, valid_list in valid_values.items():
        value = getattr(application, field)
        if value not in valid_list:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid value '{value}' for field '{field}'. Valid values: {valid_list}",
            )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""

    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": type(model).__name__,
        "model_version": "1.0.0",
        "features_count": (
            len(preprocessor.feature_names) if preprocessor else "unknown"
        ),
        "loaded_at": datetime.now().isoformat(),
    }


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "German Credit Risk Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "batch_predict": "/predict/batch",
            "health": "/health",
            "metrics": "/metrics",
            "model_info": "/model/info",
        },
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
