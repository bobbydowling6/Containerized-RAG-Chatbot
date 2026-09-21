import os
import uuid
from pathlib import Path

import chromadb
import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel, Field, field_validator

from backend.health.config import settings
from backend.health.rag import gemini

# Load environment variables
env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(env_path)

# Initialize FastAPI App
app = FastAPI(title="Terminal Command Instructor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Clients & Initializations ---

api_key = os.getenv("GEMINI_API_KEY") or gemini.GEMINI_API_KEY or "dummy-test-key"
if not api_key:
    raise ValueError("Gemini_API_Key not found in environment variables")

gemini_client = genai.Client(api_key=api_key)
MODEL = gemini.MODEL

db_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
collection = db_client.get_or_create_collection(settings.COLLECTION_NAME)


# --- Helper Functions (Exposed for Unit Testing / Mocks) ---

def check_gemini_health() -> bool:
    """Check connection to Gemini API."""
    try:
        models_list = list(gemini_client.models.list())
        return len(models_list) > 0
    except Exception as e:  # noqa: BLE001
        print(f"Gemini health check failed: {e}")
        return False


def check_chromadb_health() -> bool:
    """Check connection to ChromaDB."""
    try:
        collection.count()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"ChromaDB health check failed: {e}")
        return False


def load_documents_from_directory(directory: str) -> int:
    """Load text files from directory and chunk into ChromaDB."""
    docs_path = Path(directory)
    if not docs_path.exists():
        print(f"Warning: Docs directory not found: {directory}")
        return 0

    document_count = 0
    text_files = list(docs_path.glob("*.txt"))

    for file_path in text_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

            for i, paragraph in enumerate(paragraphs):
                if paragraph:
                    doc_id = str(uuid.uuid4())
                    collection.add(
                        ids=[doc_id],
                        documents=[paragraph],
                        metadatas=[{"source": file_path.name, "chunk": i}],
                    )
                    document_count += 1
        except (OSError, UnicodeDecodeError, ValueError) as e:
            print(f"Error processing {file_path}: {e}")

    return document_count


def ingest_document() -> dict:
    """Clear collection and re-ingest documents from configured directory."""
    existing = collection.get()
    if existing and existing.get("ids"):
        collection.delete(existing["ids"])

    count = load_documents_from_directory(settings.DOCS_DIRECTORY)
    return {
        "status": "success",
        "message": f"Successfully ingested {count} document chunks",
        "documents_added": count,
    }


def query_rag_pipeline(question: str) -> dict:
    """Query ChromaDB, apply confidence threshold, and generate response via Gemini."""
    results = collection.query(
        query_texts=[question], n_results=gemini.MAX_RESULTS
    )

    sources = []
    context = ""
    valid_distances = []

    if results and results.get("documents") and len(results["documents"]) > 0:
        for i, doc in enumerate(results["documents"][0]):
            if not doc:
                continue

            distance = (
                float(results["distances"][0][i])
                if results.get("distances")
                else 999.0
            )
            metadata = (
                results["metadatas"][0][i] if results.get("metadatas") else {}
            )

            if distance <= settings.CONFIDENCE_THRESHOLD:
                valid_distances.append(distance)
                sources.append(
                    {
                        "source": metadata.get("source", "Unknown"),
                        "distance": distance,
                        "text": doc[:100],
                    }
                )
                context += f"\n\n{doc}"

    if not sources or not context.strip():
        return {
            "question": question,
            "answer": "I don't have information about that in the provided documents.",
            "sources": [],
            "confidence": "none",
        }

    best_distance = min(valid_distances)
    if best_distance < 0.5:
        confidence = "high"
    elif best_distance < 0.9:
        confidence = "medium"
    else:
        confidence = "low"

    system_prompt = f"""You are a helpful assistant that answers questions based on provided documents. 
Answer the question based ONLY on the context provided below. 
If the answer is not in the context, say 'I don't have information about that in the provided documents.'

Context from documents:
{context}

Question: {question}"""

    response = gemini_client.models.generate_content(
        model=MODEL, contents=system_prompt
    )

    answer = response.text if response else "No response from AI"

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
    }


# --- Request Schemas with Pydantic Validation ---

class IngestRequest(BaseModel):
    content: str = Field(..., min_length=1)

    @field_validator("content")
    @classmethod
    def reject_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace.")
        return v


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def reject_whitespace(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty or whitespace.")
        return v


# --- API Routes ---

@app.get("/")
def root():
    return {"message": "RAG API running with Gemini", "model": MODEL}


@app.get("/health")
def health_check():
    gemini_healthy = check_gemini_health()
    chroma_healthy = check_chromadb_health()

    is_healthy = gemini_healthy and chroma_healthy
    status_code = 200 if is_healthy else 503

    payload = {
        "status": "healthy" if is_healthy else "unhealthy",
        "gemini": gemini_healthy,
        "chromadb": chroma_healthy,
        "model": MODEL,
    }

    if not is_healthy:
        raise HTTPException(status_code=status_code, detail=payload)

    return payload


@app.get("/stats")
def get_stats():
    try:
        doc_count = collection.count()
        return {"documents": doc_count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ingest")
def ingest_endpoint(payload: IngestRequest = None):
    try:
        return ingest_document()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/ask")
def ask_question(body: AskRequest):
    try:
        return query_rag_pipeline(body.question.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e