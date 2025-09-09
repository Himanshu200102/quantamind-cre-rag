# app/core/config.py
from __future__ import annotations
import os
from pydantic import BaseModel

class Settings(BaseModel):
    # storage
    CHUNKS_DIR: str = os.getenv("CHUNKS_DIR", "./out/chunks")
    CHROMA_DIR: str = os.getenv("CHROMA_DIR", "./out/chroma")
    COLLECTION: str = os.getenv("COLLECTION", "lease_chunks")

    # embeddings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

    # Enhanced RAG settings
    DEFAULT_K: int = int(os.getenv("DEFAULT_K", "12"))
    MAX_CONTEXT_LENGTH: int = int(os.getenv("MAX_CONTEXT_LENGTH", "15000"))
    MMR_LAMBDA: float = float(os.getenv("MMR_LAMBDA", "0.6"))
    
    # Collection suffixes for enhanced retrieval
    MAIN_COLLECTION_SUFFIX: str = "_main"
    ENTITY_COLLECTION_SUFFIX: str = "_entities"
    CLAUSE_COLLECTION_SUFFIX: str = "_clauses"

    # HF LLM for /ask
    HF_API_KEY: str = os.getenv("HF_API_KEY", "")
    HF_SUMMARY_MODEL: str = os.getenv("HF_SUMMARY_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

settings = Settings()