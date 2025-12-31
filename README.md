# SentimentOps 🚀

SentimentOps is a FastAPI-based Sentiment Analysis API with automated testing and CI/CD using GitHub Actions.  
The project demonstrates how to build, test, and validate an ML-backed REST API in a production-style setup.

---

## 🛠 Tech Stack

- Python 3.9+
- FastAPI
- pytest
- scikit-learn / NLP model
- GitHub Actions (CI/CD)

---

## 📁 Project Structure


---

## ▶️ Run Locally

### Install dependencies
```bash
pip install -r requirements.txt
Start the API
uvicorn main:app --reload
Server runs at:

http://127.0.0.1:8000
Swagger Docs:

http://127.0.0.1:8000/docs
📡 API Endpoints
Root Endpoint
GET /


Response:

{
  "message": "Sentiment Analysis API is running!",
  "version": "1.0"
}

Health Check
GET /health


Response:

{
  "status": "healthy",
  "models_loaded": true
}

Predict Sentiment
POST /predict


Request:

{
  "text": "This is awesome"
}


Response:

{
  "text": "This is awesome",
  "sentiment": "POSITIVE",
  "confidence": 0.92
}

Batch Sentiment Prediction
POST /predict-batch


Request:

{
  "texts": ["Great!", "Bad!"]
}


Response:

{
  "predictions": [
    {
      "text": "Great!",
      "sentiment": "POSITIVE"
    },
    {
      "text": "Bad!",
      "sentiment": "NEGATIVE"
    }
  ]
}

🧪 Run Tests
pytest -v

🔁 CI/CD Pipeline

Triggered on every push to main

Installs dependencies

Runs pytest

Fails build if any test fails

👤 Author

Ritansh Shrivastava

