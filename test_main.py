from fastapi.testclient import TestClient
import main
from unittest.mock import MagicMock

# ---- FORCE MODELS TO BE AVAILABLE ----
main.models_loaded = True

# ---- MOCK VECTORIZER ----
main.vectorizer = MagicMock()
main.vectorizer.transform.return_value = [[0, 1], [1, 0]]

# ---- MOCK MODEL ----
main.model = MagicMock()
main.model.predict.return_value = ["POSITIVE", "NEGATIVE"]
main.model.predict_proba.return_value = [[0.1, 0.9], [0.8, 0.2]]

client = TestClient(main.app)

# ===== TEST ROOT ENDPOINT =====
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Sentiment Analysis API is running!"

# ===== TEST HEALTH ENDPOINT =====
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["models_loaded"] is True

# ===== TEST PREDICT WITH EMPTY TEXT =====
def test_predict_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "ERROR"

# ===== TEST PREDICT WITH VALID TEXT =====
def test_predict_valid_text():
    response = client.post("/predict", json={"text": "This is great"})
    assert response.status_code == 200
    assert "sentiment" in response.json()

# ===== TEST PREDICT RESPONSE FORMAT =====
def test_predict_response_format():
    response = client.post("/predict", json={"text": "I love this!"})
    data = response.json()
    assert "text" in data
    assert "sentiment" in data
    assert "confidence" in data

# ===== TEST BATCH PREDICT =====
def test_predict_batch():
    response = client.post(
        "/predict-batch",
        json={"texts": ["Great!", "Bad!"]}
    )
    assert response.status_code == 200
    assert "predictions" in response.json()
    assert len(response.json()["predictions"]) == 2

# ===== TEST INVALID REQUEST =====
def test_invalid_request():
    response = client.post("/predict", json={})
    assert response.status_code == 422

# ===== TEST DOCS ENDPOINT =====
def test_docs_endpoint():
    response = client.get("/docs")
    assert response.status_code == 200
