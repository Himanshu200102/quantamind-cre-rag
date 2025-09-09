from __future__ import annotations
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.services.milvus_embeddings import MilvusIngestService, DEFAULT_DB_COLLECTION, DEFAULT_EMBED_MODEL

router = APIRouter(tags=["ingest"])

class EnhancedIngestRequest(BaseModel):
    chunks_dir: str = Field(..., description="Directory with *.jsonl chunk files (main.jsonl, entities.jsonl, clauses.jsonl)")
    collection: Optional[str] = Field(None, description="Base collection prefix (defaults to MILVUS_COLLECTION_PREFIX)")
    embedding_model: Optional[str] = Field(None, description="Sentence-Transformers model name")
    batch_size: int = Field(128, description="Upsert batch size")

@router.post("/ingest-enhanced", response_model=str, summary="Enhanced ingest into Milvus")
def ingest_enhanced(req: EnhancedIngestRequest):
    try:
        svc = MilvusIngestService(
            base_collection=req.collection or DEFAULT_DB_COLLECTION,
            embedding_model=req.embedding_model or DEFAULT_EMBED_MODEL,
        )
        return svc.ingest_dir(chunks_dir=req.chunks_dir, batch_size=req.batch_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Milvus ingest failed: {e}")

@router.get("/health", summary="Ingest health")
def health_check():
    try:
        return {"status": "healthy", "vector_backend": "milvus", "embed_model": DEFAULT_EMBED_MODEL}
    except Exception as e:
        return {"status": "error", "error": str(e)}
