# Containerized RAG Chatbot

A Retrieval-Augmented Generation (RAG) assistant for terminal-command questions. Documents are ingested into ChromaDB, relevant chunks are retrieved for each question, and Google Gemini generates grounded answers. A Streamlit UI talks to a FastAPI backend.

## Architecture

- **Backend** (`backend/main.py`): FastAPI app with health, stats, ingest, and ask endpoints
- **Frontend** (`frontend/app.py`): Streamlit chat UI that calls the API
- **Vector store**: ChromaDB (persistent local collection)
- **LLM**: Google Gemini (`google-genai`)
- **Source docs**: `backend/health/docs/*.txt`

```
User (Streamlit) → FastAPI → ChromaDB retrieval → Gemini → answer + sources
```

## Project layout

```
backend/
  main.py                 # FastAPI app and RAG endpoints
  requirements.txt        # Backend/CI Python dependencies
  health/
    config.py             # ChromaDB path, collection name, docs directory
    rag.py                # Gemini API key, model, retrieval settings
    docs/                 # Text files ingested into ChromaDB
    tests/test_api.py     # Pytest coverage for /, /health, /stats
frontend/
  app.py                  # Streamlit chat interface
.github/workflows/ci.yml  # GitHub Actions: pytest + ruff
```

## Prerequisites

- Python 3.12
- A Google Gemini API key

## Setup

From the repo root:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

Create a `.env` file in the project root (or `backend/.env` is also loaded relative to `backend/main.py` via the parent directory):

```bash
Gemini_API_Key=your_gemini_api_key
MODEL=gemini-3.6-flash
CHROMA_DB_PATH=chroma_db
COLLECTION_NAME=documents
```

The app reads `Gemini_API_Key` (that exact name) and will not start without it.

## Run locally

Start the API from the **repo root** so `from backend...` imports resolve:

```bash
source venv/bin/activate
uvicorn backend.main:app --reload 
```

In a second terminal, start Streamlit:

```bash
source venv/bin/activate
API_URL=http://localhost:8000 streamlit run frontend/app.py
```

1. Confirm the sidebar shows **API: Connected**.
2. Click **Re-index Documents** to ingest `backend/health/docs/*.txt` into ChromaDB.
3. Ask questions in the chat. Answers include source snippets when retrieval finds matches.

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service and model info |
| `GET` | `/health` | Gemini connectivity and document count |
| `GET` | `/stats` | Collection document count |
| `POST` | `/ingest` | Clear the collection and reload docs |
| `POST` | `/ask` | JSON body `{"question": "..."}` — RAG answer |

Example:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/ingest
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install packages with pip?"}'
```

## Tests

Pytest covers the root, health, and stats endpoints (`backend/health/tests/test_api.py`). Run from the repo root with a Gemini key set (importing the app requires `Gemini_API_Key`):

```bash
source venv/bin/activate
export Gemini_API_Key=your_gemini_api_key
export CHROMA_PATH=./test_chroma
python -m pytest backend/health/tests/test_api.py 
```

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on pushes and pull requests to `main`:

- **test**: Python 3.12, install `backend/requirements.txt`, then pytest
- **lint**: `ruff check backend/ frontend/`

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `Gemini_API_Key` | (required) | Google Gemini API key |
| `MODEL` | `gemini-3.6-flash` | Gemini model name |
| `CHROMA_DB_PATH` / `CHROMA_PATH` | `chroma_db` | Persistent ChromaDB directory |
| `COLLECTION_NAME` | `documents` | Chroma collection name |
| `DOCS_DIRECTORY` | `backend/health/docs` | Folder of `.txt` files to ingest |
| `MAX_RESULTS` | `3` | Chunks retrieved per question |
| `API_URL` | `http://localhost:8000` | Backend URL used by Streamlit |
| `DEBUG` | `false` | Extra config logging |
