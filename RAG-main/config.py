"""
Configuration management for RAG Pipeline
Centralizes all settings in one place
"""

import os
from pathlib import Path

# Try to load from .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Config:
    """Central configuration for the RAG pipeline."""
    
    # Ollama Settings
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent
    VECTOR_STORE_PATH: str = os.getenv("VECTOR_STORE_PATH", r"C:\Users\UMAIR\Desktop\chroma_db")
    PDF_PATH: str = os.getenv("PDF_PATH", "FL_Chp1.pdf")
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    
    # RAG Settings
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))
    RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "2"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.1"))
    CONTEXT_WINDOW: int = int(os.getenv("CONTEXT_WINDOW", "2048"))
    
    # Prompts
    QA_PROMPT: str = """Answer based on the context below. Be concise and accurate.

Context:
{context}

Question: {question}

Instructions:
- Only use information from the context
- Say "I couldn't find this in the document" if not found
- Be specific and cite relevant parts when possible

Answer:"""

    CONDENSE_PROMPT: str = """Given the conversation and follow-up question, rephrase it to be standalone.

Chat History:
{chat_history}

Follow-up: {question}

Standalone Question:"""

    @classmethod
    def ensure_dirs(cls):
        """Create necessary directories."""
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def print_config(cls):
        """Print current configuration."""
        print("\n" + "="*50)
        print("📋 Current Configuration")
        print("="*50)
        print(f"  LLM Model: {cls.OLLAMA_MODEL}")
        print(f"  Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"  Ollama URL: {cls.OLLAMA_BASE_URL}")
        print(f"  Vector Store: {cls.VECTOR_STORE_PATH}")
        print(f"  Chunk Size: {cls.CHUNK_SIZE}")
        print(f"  Chunk Overlap: {cls.CHUNK_OVERLAP}")
        print(f"  Retrieval K: {cls.RETRIEVAL_K}")
        print("="*50 + "\n")


# Create singleton instance
config = Config()
