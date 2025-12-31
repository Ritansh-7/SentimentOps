from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import joblib

app = FastAPI(title="Sentiment Analysis API", version="1.0")

MODEL_PATH = "models/sentiment_model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    models_loaded = True
except FileNotFoundError:
    model = None
    vectorizer = None
    models_loaded = False


class SentimentRequest(BaseModel):
    text: str


class BatchSentimentRequest(BaseModel):
    texts: List[str]


@app.get("/")
def root():
    return {
        "message": "Sentiment Analysis API is running!",
        "version": "1.0"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": models_loaded
    }


@app.post("/predict")
def predict(request: SentimentRequest):
    if not models_loaded:
        return {
            "text": request.text,
            "sentiment": "ERROR",
            "confidence": None,
            "error": "Models not loaded"
        }

    if not request.text.strip():
        return {
            "text": request.text,
            "sentiment": "ERROR",
            "confidence": None,
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
