import uuid
from pathlib import Path

import chromadb
import dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from pydantic import BaseModel

env_path = Path(__file__).parent.parent / ".env"
dotenv.load_dotenv(env_path)

from backend.health.config import settings
from backend.health.rag import gemini

app = FastAPI(title="Terminal Command Instructor")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = gemini.GEMINI_API_KEY
if not api_key:
    raise ValueError("Gemini_API_Key not found in environment variables")

gemini_client = genai.Client(api_key=api_key)
MODEL = gemini.MODEL

db_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
client = db_client
collection = client.get_or_create_collection(settings.COLLECTION_NAME)

def load_documents_from_directory(directory: str):
    """Load all text files from a directory and add them to ChromaDB"""
    docs_path = Path(directory)
    if not docs_path.exists():
        print(f"Warning: Docs directory not found: {directory}")
        return 0
    
    document_count = 0
    text_files = list(docs_path.glob("*.txt"))
    print(f"Found {len(text_files)} text files in {directory}")
    
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
                        metadatas=[{"source": file_path.name, "chunk": i}]
                    )
                    document_count += 1
            
            print(f"  Added {len(paragraphs)} chunks from {file_path.name}")
        except (OSError, UnicodeDecodeError, ValueError) as e:
            print(f"Error processing {file_path}: {e}")
    
    return document_count

@app.get("/")
def root():
    return {"message": "RAG API running with Gemini", "model": MODEL}

@app.get("/health")
def health_check():
    gemini_ok = False
    doc_count = 0
    try:
        models_list = list(gemini_client.models.list())
        gemini_ok = len(models_list) > 0
        print(f"Gemini API connected, {len(models_list)} models available")
    except Exception as e:
        print(f"Gemini health check failed: {e}")
    
    try:
        doc_count = collection.count()
        print(f"Document count: {doc_count}")
    except Exception as e:
        print(f"ChromaDB count failed: {e}")
        doc_count = 0
    
    return {
        "status": "healthy",
        "gemini": "connected" if gemini_ok else "unavailable",
        "model": MODEL,
        "documents": doc_count
    }

@app.get("/stats")
def get_stats():
    """Get statistics about the document collection"""
    try:
        doc_count = collection.count()
        return {"documents": doc_count}
    except Exception as e:
        print(f"Error fetching stats: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

@app.post("/ingest")
def ingest_documents():
    """Load and index documents from the docs directory"""
    try:
        existing = collection.get()
        if existing and existing.get("ids"):
            collection.delete(existing["ids"])
            print(f"Cleared {len(existing['ids'])} existing documents from collection")
        else:
            print("Collection is empty, no documents to clear")
        
        count = load_documents_from_directory(settings.DOCS_DIRECTORY)
        
        return {
            "message": f"Successfully ingested {count} document chunks",
            "documents_added": count
        }
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        print(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(body: AskRequest):
    """Ask a question and get an answer using RAG with Gemini"""
    try:
        question = body.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question is required")
            
        results = collection.query(
            query_texts=[question],
            n_results=gemini.MAX_RESULTS
        )
        
        sources = []
        context = ""

        if results and results.get("documents") and len(results["documents"]) > 0:
            for i, doc in enumerate(results["documents"][0]):
                if doc:
                    distance = results["distances"][0][i] if results.get("distances") else 0
                    metadata = results["metadatas"][0][i] if results.get("metadatas") else {}
                    
                    sources.append({
                        "source": metadata.get("source", "Unknown"),
                        "distance": float(distance),
                        "text": doc[:100]
                    })
                    context += f"\n\n{doc}"
                    
        system_prompt = f"""You are a helpful assistant that answers questions based on provided documents. 
        Answer the question based ONLY on the context provided below. 
        If the answer is not in the context, say 'I don't have information about that in the provided documents.'

        Context from documents:
        {context}

        Question: {question}"""
            
        response = gemini_client.models.generate_content(
            model=MODEL,
            contents=system_prompt
        )
            
        answer = response.text if response else "No response from AI"
        confidence = "high" if sources else "low"
        
        return {
            "question": question,
            "answer": answer,
            "sources": sources,
            "confidence": confidence
        }
        
    except HTTPException:
        raise
    except (OSError, ValueError, RuntimeError, KeyError, TypeError) as e:
        print(f"Ask error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e