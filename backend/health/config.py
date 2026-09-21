import os


class Settings:
    # ChromaDB configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", os.getenv("CHROMA_PATH", "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    
    # RAG Retrieval & Chunking Configuration
    # Distance in ChromaDB (lower = closer match). Distances above threshold are ignored.
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "1.2"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))       # Token/character target size
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))   # Overlap between chunks
    
    # Application settings
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    DOCS_DIRECTORY: str = os.getenv("DOCS_DIRECTORY", os.path.join(os.path.dirname(__file__), "docs"))

settings = Settings()

# Print configuration on startup (useful for debugging)
if settings.DEBUG:
    print("=== Configuration ===")
    print(f"  ChromaDB: {settings.CHROMA_DB_PATH}")
    print(f"  Collection: {settings.COLLECTION_NAME}")
    print(f"  Confidence Threshold: {settings.CONFIDENCE_THRESHOLD}")
    print(f"  Chunk Size/Overlap: {settings.CHUNK_SIZE} / {settings.CHUNK_OVERLAP}")