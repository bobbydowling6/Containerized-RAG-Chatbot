import os


class Settings:
    # ChromaDB configuration
    CHROMA_DB_PATH: str = os.getenv("CHROMA_DB_PATH", os.getenv("CHROMA_PATH", "chroma_db"))
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "documents")
    
    # Application settings
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"
    DOCS_DIRECTORY: str = os.getenv("DOCS_DIRECTORY", os.path.join(os.path.dirname(__file__), "docs"))

settings = Settings()

# Print configuration on startup (useful for debugging)
if settings.DEBUG:
    print("=== Configuration ===")
    print(f"  ChromaDB: {settings.CHROMA_DB_PATH}")
