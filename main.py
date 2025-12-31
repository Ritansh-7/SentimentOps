from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import os

app = FastAPI(title="Sentiment Analysis API", version="1.0")

# Load your trained model and vectorizer
MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    models_loaded = True
except FileNotFoundError:
    models_loaded = False
    model = None
    vectorizer = None

# Pydantic model for request validation
class SentimentRequest(BaseModel):
    text: str

# Root endpoint
@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running!",
        "version": "1.0"
    }

# Health check endpoint
@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }

# Predict endpoint
@app.post("/predict")
def predict(request: SentimentRequest):
    if not models_loaded:
        return {
            "text": request.text,
            "sentiment": "ERROR",
            "error": "Models not loaded"
        }

    if not request.text.strip():
        return {
            "text": request.text,
            "sentiment": "ERROR",
            "error": "Text is empty"
        }

    vec = vectorizer.transform([request.text])
    prediction = model.predict(vec)[0]

    confidence = None
    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(vec)[0])

    return {
        "text": request.text,
        "sentiment": str(prediction),
        "confidence": confidence
    }

def is_model_ready():
    return models_loaded and model is not None and vectorizer is not None

if not is_model_ready():
    return {
        "text": request.text,
        "sentiment": "ERROR",
        "confidence": None,
        "error": "Models not loaded"
    }

from typing import List
from pydantic import BaseModel

class BatchSentimentRequest(BaseModel):
    texts: List[str]

@app.post("/predict-batch")
def predict_batch(request: BatchSentimentRequest):
    if not models_loaded:
        return {"error": "Models not loaded"}

    if not request.texts:
        return {"predictions": []}

    vectors = vectorizer.transform(request.texts)
    predictions = model.predict(vectors)

    return {
        "predictions": [
            {"text": text, "sentiment": str(pred)}
            for text, pred in zip(request.texts, predictions)
        ]
    }
