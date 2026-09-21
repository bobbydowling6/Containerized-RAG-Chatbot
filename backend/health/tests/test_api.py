import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

# --- Basic Health & System Tests ---

def test_root():
    response = client.get("/")
    assert response.status_code == 200


@patch("backend.main.check_chromadb_health")
@patch("backend.main.check_gemini_health")
def test_health_success(mock_gemini_health, mock_chroma_health):
    """Test /health when both ChromaDB and Gemini are healthy."""
    mock_chroma_health.return_value = True
    mock_gemini_health.return_value = True

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["chromadb"] is True
    assert data["gemini"] is True


@patch("backend.main.check_chromadb_health")
@patch("backend.main.check_gemini_health")
def test_health_failure_when_dependency_down(mock_gemini_health, mock_chroma_health):
    """Test /health returns 533 or unhealthy status when dependencies fail."""
    # Case 1: ChromaDB fails
    mock_chroma_health.return_value = False
    mock_gemini_health.return_value = True

    response = client.get("/health")
    assert response.status_code == 503
    
    # FastAPI places HTTPException details under the 'detail' key
    data = response.json().get("detail", response.json())
    assert data["status"] == "unhealthy"
    assert data["chromadb"] is False
    assert data["gemini"] is True

    # Case 2: Gemini fails
    mock_chroma_health.return_value = True
    mock_gemini_health.return_value = False

    response = client.get("/health")
    assert response.status_code == 503
    data = response.json().get("detail", response.json())
    assert data["gemini"] is False


def test_stats():
    response = client.get("/stats")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data


# --- Tests for /ingest Endpoint ---

@patch("backend.main.ingest_document")
def test_ingest_success(mock_ingest):
    """Test ingesting a valid document."""
    mock_ingest.return_value = {"status": "success", "document_id": "doc123", "chunks": 5}
    
    payload = {"content": "This is sample document text to embed.", "metadata": {"source": "manual"}}
    response = client.post("/ingest", json=payload)
    
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    mock_ingest.assert_called_once()


def test_ingest_empty_content():
    """Test ingesting empty or missing payload content."""
    # Missing required 'content' key
    response = client.post("/ingest", json={})
    assert response.status_code == 422  # Validation Error

    # Empty string content
    response = client.post("/ingest", json={"content": "   "})
    assert response.status_code == 400


# --- Tests for /ask Endpoint & Fallbacks ---

@patch("backend.main.query_rag_pipeline")
def test_ask_success(mock_query):
    """Test /ask with a valid question and matching knowledge context."""
    mock_query.return_value = {
        "answer": "FastAPI is a modern, fast web framework for Python.",
        "sources": ["doc123"]
    }

    response = client.post("/ask", json={"question": "What is FastAPI?"})
    assert response.status_code == 200
    data = response.json()
    assert "FastAPI" in data["answer"]
    assert len(data["sources"]) > 0


def test_ask_empty_question():
    """Test /ask with empty strings or missing body."""
    # Missing 'question' key
    response = client.post("/ask", json={})
    assert response.status_code == 422

    # Empty string question
    response = client.post("/ask", json={"question": ""})
    assert response.status_code == 422

    # Whitespace-only question
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 400


@patch("backend.main.query_rag_pipeline")
def test_ask_fallback_response(mock_query):
    """Test /ask when ChromaDB finds no relevant context or Gemini falls back."""
    fallback_message = "I'm sorry, I couldn't find any relevant information in the knowledge base to answer your question."
    
    mock_query.return_value = {
        "answer": fallback_message,
        "sources": []
    }

    response = client.post("/ask", json={"question": "What is the capital of Mars?"})
    assert response.status_code == 200
    data = response.json()
    assert data["answer"] == fallback_message
    assert data["sources"] == []