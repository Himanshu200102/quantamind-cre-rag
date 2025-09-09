# app/api/ask.py
from __future__ import annotations
import os
from typing import Any, Dict, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["rag"])

# --- import the enterprise retrieval you are already using (Milvus version) ---
from app.services.llm_enhanced_retrieval import enterprise_retrieve
from app.services.local_llm import LocalLlama  # our llama.cpp wrapper


class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    k: int = Field(12, description="How many chunks to retrieve")
    use_enhanced: bool = Field(True, description="Use enterprise retrieval")
    retrieval_method: str = Field("comprehensive", description="comprehensive|hybrid|multi_hop")
    max_new_tokens: int = Field(500, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    use_gemini: bool = Field(False, description="Use Gemini (off for local llama)")


class AskResponse(BaseModel):
    answer: str
    used_model: str
    chunks: List[RetrievedChunk]
    retrieval_method: str
    total_chunks_found: int
    analysis: Dict[str, Any]


# ---- simple answer builder with local llama over a context window ----
def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 18000) -> str:
    pieces = []
    total = 0
    for ch in chunks:
        md = ch.get("metadata", {}) or {}
        tag_bits = []
        if md.get("doc_id"): tag_bits.append(md["doc_id"])
        if md.get("header"): tag_bits.append(md["header"])
        if md.get("clause_type") and md["clause_type"] != "general": tag_bits.append(f"({md['clause_type']})")
        if md.get("page_start"): tag_bits.append(f"p.{md['page_start']}")
        tag = " :: ".join(tag_bits) if tag_bits else "source"
        txt = (ch.get("text") or "").strip()
        block = f"[{tag}]\n{txt}\n"
        if total + len(block) > max_chars:
            break
        pieces.append(block)
        total += len(block)
    return "\n---\n".join(pieces)


def _llama_answer(question: str, context: str, temp: float, max_tokens: int) -> str:
    prompt = (
        "You are a careful CRE lease analyst. Answer using ONLY the provided context. "
        "If the context is insufficient, say so explicitly.\n\n"
        f"QUESTION:\n{question}\n\n"
        "CONTEXT:\n"
        f"{context}\n\n"
        "REQUIREMENTS:\n"
        "- Cite specific snippets by their [doc/page] tags already shown in the context when helpful.\n"
        "- If asked to list companies/roles, output a clean two-column table: Company | Role.\n"
        "- Be concise but complete.\n"
        "ANSWER:\n"
    )
    llm = LocalLlama()
    return llm.gen(prompt, temp=temp, max_tokens=max_tokens)


@router.post("/ask", response_model=AskResponse, summary="Ask with enterprise retrieval + local Llama")
def ask(req: AskRequest) -> AskResponse:
    # 1) Retrieve
    try:
        result = enterprise_retrieve(question=req.question, k=req.k, method=req.retrieval_method)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    chunks = result.get("chunks", []) or []
    if not chunks:
        return AskResponse(
            answer="No relevant information found in the document to answer your question.",
            used_model="No model used",
            chunks=[],
            retrieval_method=result.get("method_used", req.retrieval_method),
            total_chunks_found=0,
            analysis={"error": "empty retrieval", **{k: v for k, v in result.items() if k != 'chunks'}},
        )

    # 2) Build context
    context = result.get("context") or _build_context(chunks)

    # 3) Generate answer (local Llama, unless you explicitly enable Gemini elsewhere)
    try:
        answer = _llama_answer(req.question, context, req.temperature, req.max_new_tokens)
        used_model = "local-llama"
    except Exception as e:
        # minimal structured fallback
        answer = f"Found {len(chunks)} relevant sections.\n" \
                 f"(Generation fallback: {str(e)})"
        used_model = "structured_fallback"

    # 4) Pack chunks
    response_chunks = [
        RetrievedChunk(
            id=c.get("id", f"chunk_{i}"),
            text=c.get("text", "") or "",
            score=float(c.get("score", 0.0) or 0.0),
            metadata=c.get("metadata", {}) or {},
        )
        for i, c in enumerate(chunks)
    ]

    analysis = {
        "method_used": result.get("method_used"),
        "confidence_score": result.get("confidence_score"),
        "total_chunks": len(chunks),
        "entities_found": result.get("entities_found"),
    }

    return AskResponse(
        answer=answer,
        used_model=used_model,
        chunks=response_chunks,
        retrieval_method=result.get("method_used", req.retrieval_method),
        total_chunks_found=len(chunks),
        analysis=analysis,
    )
