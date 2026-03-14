from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from typing import List
import os

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI Prediction API",
    description="Professional ML Model Serving API",
    version="1.0.0"
)

# Load model globally to avoid reloading on each request
MODEL_PATH = os.getenv("MODEL_PATH", "model.joblib")
try:
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
    else:
        model = None
        logger.warning(f"No model found at {MODEL_PATH}. Prediction endpoints will fail.")
except Exception as e:
    model = None
    logger.error(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    feature1: float
    feature2: float
    category_B: bool = False
    category_C: bool = False
    
class PredictionResponse(BaseModel):
    prediction: int
    probability: List[float]

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Endpoint to generate predictions from the loaded model."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    try:
        # Convert request to DataFrame format expected by model
        input_data = pd.DataFrame([request.model_dump()])
        prediction = int(model.predict(input_data)[0])
        probability = model.predict_proba(input_data)[0].tolist()
        
        return PredictionResponse(prediction=prediction, probability=probability)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
