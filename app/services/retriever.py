# app/services/retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

try:
    from app.core.config import settings
    CHROMA_DIR = getattr(settings, "CHROMA_DIR", "./out/chroma")
    COLLECTION_NAME = getattr(settings, "COLLECTION", "lease_chunks")
    EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
except Exception:
    CHROMA_DIR = "./out/chroma"
    COLLECTION_NAME = "lease_chunks"
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

DEFAULT_CHROMA_DIR = CHROMA_DIR
DEFAULT_COLLECTION = COLLECTION_NAME
DEFAULT_EMBED_MODEL = EMBEDDING_MODEL

_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
        print("[retriever] loaded", EMBEDDING_MODEL, 
              "dim=", _embedder.get_sentence_embedding_dimension())
    return _embedder

def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

def get_collection(name: str = COLLECTION_NAME):
    client = _get_client()
    return client.get_or_create_collection(name=name)

def _embed_texts(texts: List[str]) -> List[List[float]]:
    model = _get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()

def _score_from_distance(d: float) -> float:
    return 1.0 / (1.0 + float(d))

def _mmr(query_vec: np.ndarray, doc_vecs: np.ndarray, k: int, lambda_mult: float = 0.6) -> List[int]:
    if doc_vecs.size == 0:
        return []
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
        return v / n
    q = _norm(query_vec.reshape(1, -1))[0]
    V = _norm(doc_vecs)
    selected, candidates = [], list(range(V.shape[0]))
    sim_to_query = V @ q
    while len(selected) < min(k, len(candidates)):
        if not selected:
            j0 = int(np.argmax(sim_to_query[candidates]))
            chosen = candidates[j0]
        else:
            sel_mat = V[selected]
            max_sim_selected = (V[candidates] @ sel_mat.T).max(axis=1)
            mmr_scores = lambda_mult * sim_to_query[candidates] - (1.0 - lambda_mult) * max_sim_selected
            j = int(np.argmax(mmr_scores))
            chosen = candidates[j]
        selected.append(chosen)
        candidates.remove(chosen)
    return selected

def _build_context(chunks: List[Dict[str, Any]], max_chars: int = 12000) -> Tuple[str, List[Dict[str, Any]]]:
    pieces, sources, total = [], [], 0
    by_doc: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        doc_id = c["metadata"].get("doc_id", c["id"])
        by_doc.setdefault(doc_id, []).append(c)
    for doc_id, arr in by_doc.items():
        arr.sort(key=lambda x: (x["metadata"].get("page_start", 0), x["metadata"].get("chunk_index", 0)))
        for c in arr:
            header = (c["metadata"].get("header") or "").strip()
            pstart = c["metadata"].get("page_start")
            pend = c["metadata"].get("page_end", pstart)
            stamp_parts = [doc_id]
            if header:
                stamp_parts.append(header)
            if pstart:
                stamp_parts.append(f"p.{pstart}-{pend}" if pend and pend != pstart else f"p.{pstart}")
            stamp = f"[{' :: '.join(stamp_parts)}]"
            snippet = f"{stamp}\n{c['text'].strip()}\n"
            if total + len(snippet) > max_chars:
                break
            pieces.append(snippet)
            total += len(snippet)
            sources.append({
                "id": c["id"], "doc_id": doc_id, "header": header,
                "page_start": pstart, "page_end": pend,
                "score": round(float(c["score"]), 4),
                "text_preview": c["text"][:100] + "..." if len(c["text"]) > 100 else c["text"],
            })
    return "\n---\n".join(pieces), sources

def retrieve_advanced(question: str, k: int = 8, candidate_pool: int = 40,
                      where: Optional[Dict[str, Any]] = None,
                      use_mmr: bool = True, 
                      embedding_model: Optional[str] = None,
                      **kwargs: Any):
    # Use legacy retrieval for backward compatibility
    query_vec = np.array(_embed_texts([question])[0], dtype=np.float32)
    collection = get_collection(COLLECTION_NAME)
    n_results = max(k, candidate_pool)
    query_kwargs: Dict[str, Any] = dict(
        query_embeddings=[query_vec.tolist()],
        n_results=n_results,
        include=["documents", "metadatas", "distances", "embeddings"],
    )
    if where:
        query_kwargs["where"] = where
    res = collection.query(**query_kwargs)
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    ids = res.get("ids", [[]])[0]
    dists = res.get("distances", [[]])[0]
    embs = res.get("embeddings", [[]])[0]
    if not docs:
        return [], "", []
    candidates: List[Dict[str, Any]] = []
    for _id, _doc, _dist, _meta, _emb in zip(ids, docs, dists, metas, embs):
        candidates.append({
            "id": _id, "text": _doc,
            "score": _score_from_distance(_dist),
            "metadata": _meta or {},
            "embedding": _emb,
        })
    candidates.sort(key=lambda x: x["score"], reverse=True)
    if use_mmr and len(candidates) > k and len(embs) > 0:
        doc_vecs = np.array([c["embedding"] for c in candidates], dtype=np.float32)
        sel_idx = _mmr(query_vec, doc_vecs, k=k, lambda_mult=0.6)
        final_chunks = [candidates[i] for i in sel_idx]
    else:
        final_chunks = candidates[:k]
    for c in final_chunks:
        c.pop("embedding", None)
    context, sources = _build_context(final_chunks)
    return final_chunks, context, sources

# Enhanced retrieval functions
def retrieve_enhanced(question: str, k: int = 12, **kwargs):
    """Enhanced retrieval using the new system"""
    try:
        from app.services.enhanced_retriever import retrieve_advanced as enhanced_retrieve_advanced
        return enhanced_retrieve_advanced(question, k=k, **kwargs)
    except ImportError:
        # Fallback to legacy retrieval
        return retrieve_advanced(question, k=k, **kwargs)

def retrieve_entities(question: str, k: int = 8):
    """Entity-focused retrieval"""
    try:
        from app.services.enhanced_retriever import retrieve_entities as enhanced_retrieve_entities
        return enhanced_retrieve_entities(question, k=k)
    except ImportError:
        # Fallback to legacy retrieval with entity-focused query
        entity_question = f"companies and parties involved: {question}"
        return retrieve_advanced(entity_question, k=k)

def smart_retrieve(question: str, k: int = 12, debug: bool = False, **kwargs):
    """Smart retrieval using integrated system"""
    try:
        from app.services.integrated_retrieval import SmartRetrievalSystem
        system = SmartRetrievalSystem()
        return system.smart_retrieve(question, k=k, debug=debug, **kwargs)
    except ImportError:
        # Fallback to enhanced retrieval
        chunks, context, sources = retrieve_enhanced(question, k=k, **kwargs)
        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
            "total_chunks_found": len(chunks),
            "context_length": len(context),
        }

# Compat alias
def retrieve(*args, **kwargs):
    return retrieve_advanced(*args, **kwargs)