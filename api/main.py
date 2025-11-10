"""
Phishing Detection API
FastAPI application for real-time phishing website detection
"""

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import List, Optional, Dict
import joblib
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

from feature_extractor import URLFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Phishing Detection API",
    description="Real-time phishing website detection using machine learning",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and feature extractor
MODEL_DIR = Path("../models")
feature_extractor = URLFeatureExtractor()

try:
    # Load all models
    model_xgboost = joblib.load(MODEL_DIR / "best_model_xgboost.pkl")
    model_rf = joblib.load(MODEL_DIR / "model_random_forest.pkl")
    model_lr = joblib.load(MODEL_DIR / "model_logistic_regression.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    feature_names = joblib.load(MODEL_DIR / "feature_names.pkl")

    logger.info("✅ All models loaded successfully")
except Exception as e:
    logger.error(f"❌ Error loading models: {e}")
    raise


# Pydantic models for API
class URLRequest(BaseModel):
    """Single URL analysis request"""
    url: str = Field(..., description="URL to analyze for phishing")
    html_content: Optional[str] = Field(None, description="Optional HTML content of the webpage")

    class Config:
        schema_extra = {
            "example": {
                "url": "https://example.com/login",
                "html_content": None
            }
        }


class BatchURLRequest(BaseModel):
    """Batch URL analysis request"""
    urls: List[str] = Field(..., description="List of URLs to analyze", max_items=100)

    class Config:
        schema_extra = {
            "example": {
                "urls": [
                    "https://example.com",
                    "http://suspicious-site.com",
                    "https://secure-bank.com"
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response"""
    url: str
    is_phishing: bool
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence (0-1)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH")
    model_used: str
    timestamp: str
    features_analyzed: Optional[Dict[str, float]] = None


class EnsemblePredictionResponse(BaseModel):
    """Ensemble prediction response"""
    url: str
    is_phishing: bool
    confidence: float
    risk_level: str
    individual_predictions: Dict[str, Dict[str, float]]
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: bool
    timestamp: str
    version: str


class ModelInfo(BaseModel):
    """Model information"""
    name: str
    accuracy: float
    f1_score: float
    feature_count: int


# Helper functions
def get_risk_level(confidence: float) -> str:
    """Determine risk level based on confidence"""
    if confidence >= 0.8:
        return "HIGH"
    elif confidence >= 0.5:
        return "MEDIUM"
    else:
        return "LOW"


def predict_with_model(url: str, html_content: Optional[str], model, model_name: str) -> PredictionResponse:
    """Make prediction with a specific model"""
    try:
        # Extract features
        feature_vector = feature_extractor.get_feature_vector(url, html_content)

        # Scale features
        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))

        # Make prediction
        prediction = model.predict(feature_vector_scaled)[0]
        probability = model.predict_proba(feature_vector_scaled)[0]

        # Get confidence (probability of predicted class)
        confidence = probability[1] if prediction == 1 else probability[0]

        return PredictionResponse(
            url=url,
            is_phishing=bool(prediction),
            confidence=float(confidence),
            risk_level=get_risk_level(confidence) if prediction else "LOW",
            model_used=model_name,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


# API Endpoints
@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Phishing Detection API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "ensemble": "/predict/ensemble",
            "batch": "/predict/batch",
            "analyze": "/analyze"
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=True,
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.get("/models", tags=["General"])
async def list_models():
    """List available models and their information"""
    return {
        "models": [
            ModelInfo(
                name="XGBoost",
                accuracy=0.9865,
                f1_score=0.9865,
                feature_count=48
            ),
            ModelInfo(
                name="Random Forest",
                accuracy=0.9855,
                f1_score=0.9855,
                feature_count=48
            ),
            ModelInfo(
                name="Logistic Regression",
                accuracy=0.9520,
                f1_score=0.9524,
                feature_count=48
            )
        ],
        "default_model": "XGBoost"
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict_phishing(request: URLRequest):
    """
    Predict if a URL is phishing using the best model (XGBoost)

    Returns prediction with confidence score and risk level.
    """
    return predict_with_model(request.url, request.html_content, model_xgboost, "XGBoost")


@app.post("/predict/xgboost", response_model=PredictionResponse, tags=["Prediction"])
async def predict_xgboost(request: URLRequest):
    """Predict using XGBoost model"""
    return predict_with_model(request.url, request.html_content, model_xgboost, "XGBoost")


@app.post("/predict/random-forest", response_model=PredictionResponse, tags=["Prediction"])
async def predict_random_forest(request: URLRequest):
    """Predict using Random Forest model"""
    return predict_with_model(request.url, request.html_content, model_rf, "Random Forest")


@app.post("/predict/logistic-regression", response_model=PredictionResponse, tags=["Prediction"])
async def predict_logistic_regression(request: URLRequest):
    """Predict using Logistic Regression model"""
    return predict_with_model(request.url, request.html_content, model_lr, "Logistic Regression")


@app.post("/predict/ensemble", response_model=EnsemblePredictionResponse, tags=["Prediction"])
async def predict_ensemble(request: URLRequest):
    """
    Predict using ensemble of all three models

    Combines predictions from XGBoost, Random Forest, and Logistic Regression
    for more robust detection.
    """
    try:
        # Extract features once
        feature_vector = feature_extractor.get_feature_vector(request.url, request.html_content)
        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))

        # Get predictions from all models
        models = {
            "XGBoost": model_xgboost,
            "Random Forest": model_rf,
            "Logistic Regression": model_lr
        }

        predictions = {}
        probabilities = []
        votes = []

        for name, model in models.items():
            pred = model.predict(feature_vector_scaled)[0]
            proba = model.predict_proba(feature_vector_scaled)[0]

            predictions[name] = {
                "prediction": int(pred),
                "phishing_probability": float(proba[1]),
                "legitimate_probability": float(proba[0])
            }

            probabilities.append(proba[1])
            votes.append(pred)

        # Weighted average (weights based on F1-scores)
        weights = {
            "XGBoost": 0.9865,
            "Random Forest": 0.9855,
            "Logistic Regression": 0.9524
        }
        total_weight = sum(weights.values())

        weighted_prob = sum(
            predictions[name]["phishing_probability"] * weights[name]
            for name in models.keys()
        ) / total_weight

        # Majority voting for final prediction
        final_prediction = 1 if sum(votes) >= 2 else 0

        return EnsemblePredictionResponse(
            url=request.url,
            is_phishing=bool(final_prediction),
            confidence=weighted_prob if final_prediction else (1 - weighted_prob),
            risk_level=get_risk_level(weighted_prob) if final_prediction else "LOW",
            individual_predictions=predictions,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        logger.error(f"Ensemble prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ensemble prediction failed: {str(e)}"
        )


@app.post("/predict/batch", tags=["Prediction"])
async def predict_batch(request: BatchURLRequest):
    """
    Batch prediction for multiple URLs

    Maximum 100 URLs per request.
    """
    if len(request.urls) > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 100 URLs allowed per batch request"
        )

    results = []
    for url in request.urls:
        try:
            result = predict_with_model(url, None, model_xgboost, "XGBoost")
            results.append(result.dict())
        except Exception as e:
            logger.error(f"Error processing URL {url}: {e}")
            results.append({
                "url": url,
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })

    return {
        "total_urls": len(request.urls),
        "results": results,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/analyze", tags=["Analysis"])
async def analyze_url(request: URLRequest):
    """
    Detailed URL analysis with feature extraction

    Returns extracted features along with prediction.
    """
    try:
        # Extract features
        features = feature_extractor.extract_features(request.url, request.html_content)
        feature_vector = feature_extractor.get_feature_vector(request.url, request.html_content)
        feature_vector_scaled = scaler.transform(feature_vector.reshape(1, -1))

        # Make prediction
        prediction = model_xgboost.predict(feature_vector_scaled)[0]
        probability = model_xgboost.predict_proba(feature_vector_scaled)[0]
        confidence = probability[1] if prediction == 1 else probability[0]

        # Top features
        feature_importance = model_xgboost.feature_importances_
        top_features = sorted(
            zip(feature_names, features.values(), feature_importance),
            key=lambda x: abs(x[2]),
            reverse=True
        )[:10]

        return {
            "url": request.url,
            "prediction": {
                "is_phishing": bool(prediction),
                "confidence": float(confidence),
                "risk_level": get_risk_level(confidence) if prediction else "LOW",
                "phishing_probability": float(probability[1]),
                "legitimate_probability": float(probability[0])
            },
            "features": {
                "total_features": len(features),
                "all_features": features,
                "top_10_important_features": [
                    {
                        "name": name,
                        "value": value,
                        "importance": float(importance)
                    }
                    for name, value, importance in top_features
                ]
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}"
        )


# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not Found",
        "message": "The requested endpoint does not exist",
        "docs": "/docs"
    }


@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return {
        "error": "Internal Server Error",
        "message": "An internal error occurred. Please try again later."
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
