# app/api/ask.py
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

router = APIRouter(tags=["rag"])
log = logging.getLogger(__name__)

# --- imports for retrieval & local LLM ---
from app.services.llm_enhanced_retrieval import enterprise_retrieve
from app.services.local_llm import LocalLlama  # llama.cpp wrapper (only used if use_gemini=True)

# --- optional chat memory (safe import; file works even if memory module not present) ---
try:
    from app.services.chat_memory import ChatMemory
    _chat_memory = ChatMemory()  # expects a simple local store or redis-based store
    CHAT_MEMORY_AVAILABLE = True
except Exception as _e:
    log.warning(f"[chat-memory] not available ({_e}); continuing without memory.")
    _chat_memory = None
    CHAT_MEMORY_AVAILABLE = False


# ------------------ Models ------------------

class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]


class AskRequest(BaseModel):
    question: str = Field(..., description="User question")
    k: int = Field(12, description="How many chunks to retrieve")
    use_enhanced: bool = Field(True, description="Use enterprise retrieval")
    retrieval_method: str = Field(
        "hybrid", description="comprehensive|hybrid|multi_hop"
    )
    max_new_tokens: int = Field(500, ge=1, le=1024)
    temperature: float = Field(0.1, ge=0.0, le=1.0)
    use_gemini: bool = Field(False, description="If true, use local LLaMA to generate an answer; if false, return structured fallback")

    # Chat memory (optional)
    conversation_id: Optional[str] = Field(
        None, description="Stable ID of an ongoing conversation to retain context"
    )
    user_id: Optional[str] = Field(
        None, description="User identifier to partition memory across users"
    )

    class Config:
        schema_extra = {
            "example": {
                "question": "Give me all the companies listed",
                "k": 12,
                "use_enhanced": True,
                "retrieval_method": "hybrid",
                "max_new_tokens": 500,
                "temperature": 0.1,
                "use_gemini": False,
                "conversation_id": "demo-thread-001",
                "user_id": "himanshu"
            }
        }


class AskResponse(BaseModel):
    answer: str
    used_model: str
    chunks: List[RetrievedChunk]
    retrieval_method: str
    total_chunks_found: int
    analysis: Dict[str, Any]


# ------------------ Helpers ------------------

def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 18000) -> str:
    """
    Build a clean, source-tagged context block for the LLM.
    """
    pieces = []
    total = 0
    for ch in chunks:
        md = ch.get("metadata", {}) or {}
        tag_bits = []
        if md.get("doc_id"):
            tag_bits.append(md["doc_id"])
        if md.get("header"):
            tag_bits.append(md["header"])
        if md.get("clause_type") and md["clause_type"] != "general":
            tag_bits.append(f"({md['clause_type']})")
        if md.get("page_start"):
            tag_bits.append(f"p.{md['page_start']}")
        tag = " :: ".join(tag_bits) if tag_bits else "source"
        txt = (ch.get("text") or "").strip()
        block = f"[{tag}]\n{txt}\n"
        if total + len(block) > max_chars:
            break
        pieces.append(block)
        total += len(block)
    return "\n---\n".join(pieces)


def _format_chat_history(history: Optional[List[Dict[str, str]]]) -> str:
    """
    Turn memory records into a compact text block the LLM can use.
    Each history item is expected as {"role": "user"/"assistant", "text": "..."}.
    """
    if not history:
        return ""
    lines = []
    for turn in history[-8:]:  # cap to last 8 messages for brevity
        role = turn.get("role", "user")
        text = (turn.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"{role.upper()}: {text}")
    return "\n".join(lines)


def _llama_answer(question: str, context: str, chat_history_block: str, temp: float, max_tokens: int) -> str:
    """
    Build the final prompt and call local LLaMA if requested.
    """
    history_section = f"\nCHAT HISTORY (most recent first):\n{chat_history_block}\n" if chat_history_block else ""
    prompt = (
        "You are a careful CRE lease analyst. Answer using ONLY the provided context. "
        "If the context is insufficient, say so explicitly.\n"
        f"{history_section}\n"
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


# ------------------ Route ------------------

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask with enterprise retrieval + optional local LLaMA",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "question": "Give me all the companies listed",
                        "k": 12,
                        "use_enhanced": True,
                        "retrieval_method": "hybrid",
                        "max_new_tokens": 500,
                        "temperature": 0.1,
                        "use_gemini": False,
                        "conversation_id": "demo-thread-001",
                        "user_id": "himanshu"
                    }
                }
            }
        }
    },
)
def ask(req: AskRequest) -> AskResponse:
    # 0) Chat history (optional)
    history_records: List[Dict[str, str]] = []
    if CHAT_MEMORY_AVAILABLE and req.conversation_id and req.user_id:
        try:
            history_records = _chat_memory.get_history(
                user_id=req.user_id,
                conversation_id=req.conversation_id,
                limit=16
            ) or []
        except Exception as e:
            log.warning(f"[chat-memory] get_history failed: {e}")

    # 1) Retrieve (Milvus hybrid/comprehensive/multi_hop)
    try:
        result = enterprise_retrieve(question=req.question, k=req.k, method=req.retrieval_method)
    except Exception as e:
        # Even if retrieval fails, still append the user turn to memory so the thread is consistent
        if CHAT_MEMORY_AVAILABLE and req.conversation_id and req.user_id:
            try:
                _chat_memory.append_user_turn(req.user_id, req.conversation_id, req.question)
            except Exception as e2:
                log.warning(f"[chat-memory] append_user_turn failed on retrieval error: {e2}")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {e}")

    chunks = result.get("chunks", []) or []
    method_used = result.get("method_used", req.retrieval_method)
    entities = result.get("entities_found", {}) or {}
    context_text = result.get("context") or _build_context(chunks)

    # 2) If nothing found, answer minimally and record turn
    if not chunks:
        answer = "No relevant information found in the document to answer your question."
        used_model = "No model used"

        # --- memory: record both question and assistant reply ---
        if CHAT_MEMORY_AVAILABLE and req.conversation_id and req.user_id:
            try:
                _chat_memory.append_user_turn(req.user_id, req.conversation_id, req.question)
                _chat_memory.append_assistant_turn(req.user_id, req.conversation_id, answer)
            except Exception as e:
                log.warning(f"[chat-memory] append_turns failed (empty retrieval): {e}")

        return AskResponse(
            answer=answer,
            used_model=used_model,
            chunks=[],
            retrieval_method=method_used,
            total_chunks_found=0,
            analysis={"error": "empty retrieval", **{k: v for k, v in result.items() if k != "chunks"}},
        )

    # 3) Build chat-history text for the LLM prompt (if we’re going to use it)
    history_block = _format_chat_history(history_records)

    # 4) Generate the answer or return structured fallback
    try:
        if req.use_gemini:
            # Local LLaMA generation path
            ans = _llama_answer(req.question, context_text, history_block, req.temperature, req.max_new_tokens)
            used_model = "local-llama"
        else:
            # No LLM generation path (fast). Still persists memory so multi-turn works.
            ans = f"Found {len(chunks)} relevant sections. (Generation disabled; set use_gemini=true to generate a full answer.)"
            used_model = "structured_fallback"
    except Exception as e:
        # Minimal structured fallback if LLM errors
        ans = f"Found {len(chunks)} relevant sections.\n(Generation fallback: {str(e)})"
        used_model = "structured_fallback"

    # 5) Pack chunks for response
    response_chunks = [
        RetrievedChunk(
            id=c.get("id", f"chunk_{i}"),
            text=c.get("text", "") or "",
            score=float(c.get("score", 0.0) or 0.0),
            metadata=c.get("metadata", {}) or {},
        )
        for i, c in enumerate(chunks)
    ]

    # 6) Record the full turn in chat memory (question + assistant reply)
    if CHAT_MEMORY_AVAILABLE and req.conversation_id and req.user_id:
        try:
            _chat_memory.append_user_turn(req.user_id, req.conversation_id, req.question)
            _chat_memory.append_assistant_turn(req.user_id, req.conversation_id, ans)
        except Exception as e:
            log.warning(f"[chat-memory] append_turns failed: {e}")

    analysis = {
        "method_used": method_used,
        "confidence_score": result.get("confidence_score"),
        "total_chunks": len(chunks),
        "entities_found": entities,
    }

    return AskResponse(
        answer=ans,
        used_model=used_model,
        chunks=response_chunks,
        retrieval_method=method_used,
        total_chunks_found=len(chunks),
        analysis=analysis,
    )
