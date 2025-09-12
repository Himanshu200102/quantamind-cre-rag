# app/api/rag.py
from __future__ import annotations
from typing import Any, Dict, List, Optional
from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

# Router for RAG utilities / diagnostics (no /ask here to avoid route collision)
router = APIRouter(prefix="", tags=["rag"])

# Use the Milvus-backed enterprise retrieval
from app.services.llm_enhanced_retrieval import enterprise_retrieve


class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the documents")
    k: int = Field(12, ge=1, le=50, description="How many chunks to retrieve")
    candidate_pool: int = Field(40, ge=1, le=200, description="(unused here; kept for compatibility)")
    where: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Optional metadata filter (not applied in Milvus path)"
    )
    use_mmr: bool = True  # kept for compatibility with earlier clients
    retrieval_method: str = Field(
        "comprehensive",
        description="Retrieval strategy: comprehensive | hybrid | multi_hop"
    )

    @validator("where", pre=True)
    def _ensure_dict(cls, v):
        # OpenAPI UI sometimes sends {"additionalProp1": ""} placeholder; normalize to {}
        if v is None or (isinstance(v, dict) and "additionalProp1" in v):
            return {}
        return v if isinstance(v, dict) else {}


@router.post("/test-query", summary="Test Enhanced Retrieval (no generation)")
def test_query(payload: RetrieveRequest):
    """
    Runs the enterprise retriever and returns a compact diagnostic payload:
    - retrieval method used
    - total chunks found
    - entities extracted from metadata
    - preview of top chunks
    """
    result = enterprise_retrieve(
        question=payload.question,
        k=payload.k,
        method=payload.retrieval_method
    )

    chunks: List[Dict[str, Any]] = result.get("chunks", []) or []
    entities: Dict[str, List[str]] = result.get("entities_found", {}) or {}

    analysis = {
        "method_used": result.get("method_used"),
        "confidence_score": result.get("confidence_score"),
        "total_chunks": result.get("total_chunks"),
        "context_length": result.get("context_length"),
    }

    return {
        "question": payload.question,
        "method_used": result.get("method_used"),
        "total_chunks_found": len(chunks),
        "entities_extracted": entities,
        "document_analysis": analysis,
        "success": True,
        "chunks_preview": [
            {
                "id": c.get("id"),
                "score": round(float(c.get("score", 0.0)), 4),
                "text_preview": (c.get("text") or "")[:200] + ("..." if len(c.get("text") or "") > 200 else ""),
                "section": (c.get("metadata", {}) or {}).get("header", "Unknown"),
                "doc_type": (c.get("metadata", {}) or {}).get("doc_type", "unknown"),
                "clause_type": (c.get("metadata", {}) or {}).get("clause_type", "general"),
                "page_start": (c.get("metadata", {}) or {}).get("page_start"),
                "doc_id": (c.get("metadata", {}) or {}).get("doc_id"),
            }
            for c in chunks[:5]
        ],
    }
