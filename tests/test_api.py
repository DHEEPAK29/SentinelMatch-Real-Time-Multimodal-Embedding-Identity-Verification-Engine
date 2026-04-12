import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_verification_missing_file():
    # Contract validation: ensure API handles empty requests correctly
    response = client.post("/verify")
    assert response.status_code == 422 # FastAPI validation error