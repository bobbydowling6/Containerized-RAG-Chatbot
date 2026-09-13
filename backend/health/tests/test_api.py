from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data or "chromadb" in data

def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
