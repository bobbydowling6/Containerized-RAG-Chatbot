import os

class Gemini:
# API Configuration
    GEMINI_API_KEY: str = os.getenv("Gemini_API_Key")
    MODEL: str = os.getenv("MODEL", "gemini-3.6-flash")

# RAG configuration
    MAX_RESULTS: int = int(os.environ.get("MAX_RESULTS", "3"))
    CONFIDENCE_THRESHOLD: float = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"

gemini = Gemini()     

if gemini.DEBUG:
    print(f"  GEMINI_API_KEY: {gemini.GEMINI_API_KEY}")
    print(f"  Model: {gemini.MODEL}")
    print(f"  Max Results: {gemini.MAX_RESULTS}")
    print(f"  Threshold: {gemini.CONFIDENCE_THRESHOLD}")
    print(f"  Debug: {gemini.DEBUG}")
    print(f" Gemini key: {'set' if gemini.GEMINI_API_KEY else 'not set'}")