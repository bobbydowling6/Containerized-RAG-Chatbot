# Containerized RAG Chatbot

A Retrieval-Augmented Generation (RAG) assistant for terminal-command questions. Documents are ingested into ChromaDB, relevant chunks are retrieved for each question, and Google Gemini generates grounded answers. A Streamlit UI talks to a FastAPI backend.

## System Architecture

The system uses a Retrieval-Augmented Generation (RAG) architecture to answer terminal command questions based on local text documentation:

```
                  ┌─────────────────────────────────────────┐
                  │            Client Browser               │
                  └────────────────────┬────────────────────┘
                                       │ HTTP
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │        Streamlit Chat UI                │
                  │         (frontend/app.py)               │
                  └────────────────────┬────────────────────┘
                                       │ REST API (JSON)
                                       ▼
                  ┌─────────────────────────────────────────┐
                  │            FastAPI Backend              │
                  │          (backend/main.py)              │
                  └───────────────┬───────────┬─────────────┘
                                  │           │
                 Retrieve Chunks  │           │ Send Query +
                 (Similarity)     │           │ Context
                                  ▼           ▼
   ┌────────────────────────────────┐       ┌────────────────────────────────┐
   │            ChromaDB            │       │       Google Gemini LLM        │
   │      (Persistent Store)        │       │   (gemini-3.6-flash / genai)   │
   └────────────────────────────────┘       └────────────────────────────────┘
```

1. **Document Ingestion**: Text documents (`backend/health/docs/*.txt`) are read, chunked, and stored in ChromaDB when the user triggers the `/ingest` endpoint or clicks **Re-index Documents**.
2. **Retrieval**: When a question is submitted to `/ask`, ChromaDB queries the vector database to retrieve the top $K$ relevant source snippets (default: `MAX_RESULTS=3`).
3. **Generation**: The retrieved document chunks and user question are passed via the `google-genai` SDK to Google Gemini, which formats a grounded response alongside cited source snippets.

## Project Layout

```
backend/
├── main.py                     # FastAPI app and RAG endpoints
├── requirements.txt            # Backend/CI Python dependencies
└── health/
    ├── config.py               # ChromaDB path, collection name, docs directory
    ├── rag.py                  # Gemini API key, model, retrieval settings
    ├── docs/                   # Text files ingested into ChromaDB
    └── tests/test_api.py       # Pytest coverage for /, /health, /stats
frontend/
└── app.py                      # Streamlit chat interface
.github/workflows/ci.yml        # GitHub Actions: pytest + ruff
```

## Prerequisites

- Python 3.12
- A Google Gemini API key

## Required Models & Configuration

### Required Models

- **LLM**: **`gemini-3.6-flash`** (default) — Serves as the primary generation model for answering queries based on retrieved context.
  - *Alternative supported models*: `gemini-3.5-flash-lite` or `gemini-3.5-pro` (configurable via environment variables).

### Configuration

Create a `.env` file in the project root (or `backend/.env` relative to `backend/main.py`):

```bash
# Required
Gemini_API_Key=your_gemini_api_key

# Optional / Default settings
MODEL=gemini-3.6-flash
CHROMA_DB_PATH=chroma_db
COLLECTION_NAME=documents
DOCS_DIRECTORY=backend/health/docs
MAX_RESULTS=3
API_URL=http://localhost:8000
DEBUG=false
```

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

> **Note:** The app reads `Gemini_API_Key` (that exact name) and will not start without it.

## Setup

From the repo root:

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r backend/requirements.txt
```

## Run Locally

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

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Service and model info |
| `GET` | `/health` | Gemini connectivity and document count |
| `GET` | `/stats` | Collection document count |
| `POST` | `/ingest` | Clear the collection and reload docs |
| `POST` | `/ask` | JSON body `{"question": "..."}` — RAG answer |

## Example Usage & Expected Output

### 1. Ingest Documents (`/ingest`)

Populate ChromaDB with the source documentation located in `backend/health/docs/`:

```bash
curl -s -X POST http://localhost:8000/ingest
```

**Expected Output:**
```json
{
  "status": "success",
  "message": "Successfully indexed 12 chunks into ChromaDB collection 'documents'.",
  "document_count": 12
}
```

### 2. Check System Health (`/health`)

Verify backend connectivity to ChromaDB and Gemini:

```bash
curl -s http://localhost:8000/health
```

**Expected Output:**
```json
{
  "status": "healthy",
  "gemini_connected": true,
  "model": "gemini-3.6-flash",
  "documents_indexed": 12
}
```

### 3. Ask a Question (`/ask`)

Query the assistant for terminal commands:

```bash
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I install packages with pip?"}'
```

**Expected Output:**
```json
{
  "question": "How do I install packages with pip?",
  "answer": "To install a package using pip, use the command `pip install <package_name>`. If installing from a requirements file, use `pip install -r requirements.txt`.",
  "sources": [
    {
      "file": "pip_guide.txt",
      "content": "To install packages, run `pip install <package-name>`. For batch installation from a configuration file, use `pip install -r requirements.txt`."
    }
  ]
}
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