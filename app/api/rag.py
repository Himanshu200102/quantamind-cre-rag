# app/api/rag.py - Generalized with Enhanced Retrieval and local LLaMA
from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator

from app.services.local_llm import LocalLlama

router = APIRouter(prefix="", tags=["rag"])

class RetrieveRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the document")
    k: int = Field(12, ge=1, le=50)
    candidate_pool: int = Field(40, ge=1, le=200)
    where: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Chroma metadata filter")
    use_mmr: bool = True
    retrieval_method: str = Field("comprehensive", description="comprehensive|hybrid|multi_hop")

    @validator("where", pre=True)
    def _ensure_dict(cls, v):
        if v is None or (isinstance(v, dict) and "additionalProp1" in v):
            return {}
        return v if isinstance(v, dict) else {}

class AskRequest(BaseModel):
    question: str = Field(..., description="User question about the document")
    k: int = Field(15, description="Number of chunks to retrieve")
    retrieval_method: str = Field("comprehensive", description="Retrieval strategy")
    max_new_tokens: int = Field(500, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    use_gemini: bool = Field(True, description="(Deprecated) If True, uses local LLaMA now")

class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class AskResponse(BaseModel):
    answer: str
    used_model: str
    chunks: List[RetrievedChunk]
    retrieval_method: str
    total_chunks_found: int
    analysis: Optional[Dict[str, Any]] = None

def comprehensive_retrieve_with_enterprise_system(question: str, k: int, method: str = "comprehensive") -> Dict[str, Any]:
    try:
        from app.services.llm_enhanced_retrieval import enterprise_retrieve
        result = enterprise_retrieve(question=question, k=k, method=method)
        return {
            "chunks": result.get("chunks", []),
            "context": result.get("context", ""),
            "sources": result.get("sources", []),
            "entities": result.get("entities_found", {}),
            "analysis": {
                "method_used": result.get("method_used", method),
                "confidence_score": result.get("confidence_score", 0.5),
                "total_chunks": result.get("total_chunks", 0)
            },
            "method_used": result.get("method_used", method)
        }
    except Exception as e:
        print(f"Enterprise retrieval failed: {e}")
        # Fallback with manual query_embeddings (no embedding_function in Chroma)
        try:
            from app.services.enhanced_embeddings import EnhancedChromaService, DEFAULT_EMBED_MODEL
            from sentence_transformers import SentenceTransformer

            svc = EnhancedChromaService()
            enc = SentenceTransformer(DEFAULT_EMBED_MODEL)
            q_emb = enc.encode([question], normalize_embeddings=True).tolist()

            chunks: List[Dict[str, Any]] = []

            main = svc.main_collection.query(
                query_embeddings=q_emb, n_results=k, include=["documents", "metadatas", "distances"]
            )
            if main and main.get("documents"):
                for i, (doc, md, dist) in enumerate(zip(
                    main["documents"][0], main["metadatas"][0], main["distances"][0]
                )):
                    chunks.append({"id": f"main_{i}", "text": doc, "score": 1.0 - float(dist), "metadata": md or {}})

            clause = svc.clause_collection.query(
                query_embeddings=q_emb, n_results=max(1, k // 2), include=["documents", "metadatas", "distances"]
            )
            if clause and clause.get("documents"):
                for i, (doc, md, dist) in enumerate(zip(
                    clause["documents"][0], clause["metadatas"][0], clause["distances"][0]
                )):
                    chunks.append({"id": f"clause_{i}", "text": doc, "score": 1.0 - float(dist), "metadata": md or {}})

            chunks.sort(key=lambda x: x["score"], reverse=True)
            chunks = chunks[:k]

            return {
                "chunks": chunks,
                "context": "",
                "sources": [],
                "entities": {},
                "analysis": {"fallback_used": True},
                "method_used": "enhanced_fallback"
            }

        except Exception as e2:
            print(f"Enhanced fallback also failed: {e2}")
            return {
                "chunks": [],
                "context": "",
                "sources": [],
                "entities": {},
                "analysis": {"error": f"All retrieval failed: {str(e)} | {str(e2)}"},
                "method_used": "complete_failure"
            }

def generate_llama_answer(question: str, chunks: List[Dict], entities: Dict[str, List[str]], analysis: Dict[str, Any], temperature: float, max_new_tokens: int) -> str:
    llm = LocalLlama()
    sections = []
    for i, ch in enumerate(chunks[:15]):
        md = ch.get("metadata", {}) or {}
        header = md.get("header", f"Section {i+1}")
        doc_type = md.get("doc_type", "")
        clause_type = md.get("clause_type", "")
        page = md.get("page_start", "")
        parts = [header]
        if doc_type: parts.append(f"({doc_type})")
        if clause_type and clause_type != "general": parts.append(f"[{clause_type}]")
        if page: parts.append(f"p.{page}")
        stamp = f"[{' :: '.join(parts)}]"
        sections.append(f"{stamp}\n{ch['text']}")
    context_text = "\n\n---\n\n".join(sections)

    entity_lines = []
    if entities:
        for k, v in entities.items():
            if v:
                entity_lines.append(f"{k.upper()}: {', '.join(v[:10])}")
    entity_block = ("\n\nKEY ENTITIES:\n" + "\n".join(entity_lines)) if entity_lines else ""

    prompt = f"""You are an expert CRE document analyst.

QUESTION: {question}

DOCUMENT CONTENT:
{context_text}{entity_block}

Answer precisely. Cite sections implicitly by using the provided headers."""
    return llm.gen(prompt, temp=temperature, max_tokens=max_new_tokens)

@router.post("/test-query", summary="Test Enhanced Retrieval System")
def test_query(payload: RetrieveRequest):
    result = comprehensive_retrieve_with_enterprise_system(
        question=payload.question, k=payload.k, method=payload.retrieval_method
    )
    chunks = result.get("chunks", [])
    entities = result.get("entities", {})
    analysis = result.get("analysis", {})
    method_used = result.get("method_used", payload.retrieval_method)

    return {
        "question": payload.question,
        "method_used": method_used,
        "total_chunks_found": len(chunks),
        "entities_extracted": entities,
        "document_analysis": analysis,
        "success": True,
        "chunks_preview": [
            {
                "id": c["id"],
                "score": round(c.get("score", 0), 3),
                "text_preview": c["text"][:200] + "...",
                "section": c.get("metadata", {}).get("header", "Unknown"),
                "doc_type": c.get("metadata", {}).get("doc_type", "unknown"),
                "clause_type": c.get("metadata", {}).get("clause_type", "general"),
            } for c in chunks[:5]
        ],
    }

@router.post("/ask", response_model=AskResponse, summary="Ask Questions with Enhanced CRE Analysis")
def ask(req: AskRequest) -> AskResponse:
    result = comprehensive_retrieve_with_enterprise_system(
        question=req.question, k=req.k, method=req.retrieval_method
    )
    chunks = result.get("chunks", [])
    entities = result.get("entities", {})
    analysis = result.get("analysis", {})
    method = result.get("method_used", req.retrieval_method)

    if not chunks:
        return AskResponse(
            answer="No relevant information found in the document to answer your question. The document may not contain the requested information, or the retrieval system encountered an issue.",
            used_model="No model used",
            chunks=[],
            retrieval_method=method,
            total_chunks_found=0,
            analysis=analysis
        )

    if req.use_gemini:
        answer = generate_llama_answer(req.question, chunks, entities, analysis, req.temperature, req.max_new_tokens)
        used_model = "local-llama"
    else:
        answer = f"Found {len(chunks)} relevant sections in the document."
        used_model = "structured_fallback"

    response_chunks = [
        RetrievedChunk(
            id=c.get("id", f"chunk_{i}"),
            text=c.get("text", ""),
            score=float(c.get("score", 0.0)),
            metadata=c.get("metadata", {}),
        ) for i, c in enumerate(chunks)
    ]

    return AskResponse(
        answer=answer,
        used_model=used_model,
        chunks=response_chunks,
        retrieval_method=method,
        total_chunks_found=len(chunks),
        analysis={
            "entities_found": entities,
            "document_analysis": analysis,
            "confidence": analysis.get("confidence_score", 0.7),
            "retrieval_stats": {
                "method_used": method,
                "chunks_analyzed": len(chunks),
                "entities_extracted": sum(len(elist) for elist in entities.values()) if entities else 0,
            }
        }
    )
