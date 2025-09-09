from __future__ import annotations
from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """
    Request body for /ingest

    - chunks_dir: folder that contains *.jsonl chunk files
    - collection: Chroma collection name
    - embedding_model: sentence-transformers model to use (must match what you used at index time ideally)
    - batch_size: upsert batch size
    """
    chunks_dir: str = Field(..., description="Path to folder with *.jsonl chunk files")
    collection: str = Field("lease_chunks", description="Chroma collection name")
    embedding_model: str = Field(
        "sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-Transformers embedding model identifier",
    )
    batch_size: int = Field(128, ge=1, le=2048, description="Upsert batch size")
