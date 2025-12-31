from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# ===== TEST ROOT ENDPOINT =====
def test_root():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "Sentiment Analysis API is running!"

# ===== TEST HEALTH ENDPOINT =====
def test_health():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

# ===== TEST PREDICT WITH EMPTY TEXT =====
def test_predict_empty_text():
    """Test prediction with empty text"""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["sentiment"] == "ERROR"

# ===== TEST PREDICT WITH VALID TEXT =====
def test_predict_valid_text():
    """Test prediction with valid text"""
    payload = {"text": "This is a great product!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "text" in data

# ===== TEST PREDICT RESPONSE FORMAT =====
def test_predict_response_format():
    """Test response has correct format"""
    payload = {"text": "I love this!"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
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


# ===== TEST INVALID REQUEST =====
def test_invalid_request():
    """Test with missing text field"""
    response = client.post("/predict", json={})
    assert response.status_code == 422  # Validation error

# ===== TEST DOCS ENDPOINT =====
def test_docs_endpoint():
    """Test that docs endpoint exists"""
    response = client.get("/docs")
    assert response.status_code == 200