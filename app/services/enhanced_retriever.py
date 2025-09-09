# app/services/enhanced_retriever.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import logging
import os
import json

# --------------------------------- Configuration ---------------------------------
try:
    from app.core.config import settings
    CHROMA_DIR = getattr(settings, "CHROMA_DIR", "./out/chroma")
    COLLECTION_NAME = getattr(settings, "COLLECTION", "lease_chunks")
    EMBEDDING_MODEL = getattr(settings, "EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
except Exception:
    CHROMA_DIR = os.getenv("CHROMA_DIR", "./out/chroma")
    COLLECTION_NAME = os.getenv("COLLECTION", "lease_chunks")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")

logger = logging.getLogger(__name__)

# --------------------------------- Globals ---------------------------------
_embedder: Optional[SentenceTransformer] = None

def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _embedder = SentenceTransformer(EMBEDDING_MODEL, device=device)
        print(f"[enhanced_retriever] loaded {EMBEDDING_MODEL} device={device}")
    return _embedder

def _get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )

# --------------------------------- Utils ---------------------------------
def _score_from_distance(distance: float) -> float:
    """
    Chroma returns 'distances' whose meaning depends on the collection's metric.
    We map smaller distances -> higher score using a stable 1/(1+d).
    """
    try:
        d = float(distance)
    except Exception:
        d = 0.0
    return 1.0 / (1.0 + max(d, 0.0))

def _embed_texts(texts: List[str]) -> List[List[float]]:
    # Normalize so cosine/IP behave best with BGE
    return _get_embedder().encode(texts, normalize_embeddings=True).tolist()

def _coerce_meta(md: Any) -> Dict[str, Any]:
    if md is None:
        return {}
    if isinstance(md, dict):
        return md
    try:
        return json.loads(md)
    except Exception:
        return {}

# --------------------------------- Retriever ---------------------------------
class EnhancedRetriever:
    """
    Multi-strategy retrieval over Chroma:
    - Query expansion
    - Searches across main/entities/clauses collections
    - Lightweight initial results (no embeddings) + selective backfill for MMR
    - Intent-aware re-ranking + optional MMR diversity
    """

    def __init__(self, collection_prefix: str = COLLECTION_NAME):
        self.client = _get_client()
        self.embedder = _get_embedder()
        self.collection_prefix = collection_prefix

        self.main_collection = self._get_collection(f"{collection_prefix}_main")
        self.entity_collection = self._get_collection(f"{collection_prefix}_entities")
        self.clause_collection = self._get_collection(f"{collection_prefix}_clauses")

        self.query_expansions = {
            "company": ["companies", "corporation", "firm", "organization", "business", "entity", "lessor", "lessee", "party"],
            "parties": ["lessor", "lessee", "company", "companies", "entity", "entities", "organization"],
            "rent": ["rental", "payment", "monthly payment", "lease payment", "amount", "cost"],
            "term": ["duration", "period", "tenure", "lease term", "expiry", "expiration"],
            "space": ["premises", "area", "floor", "suite", "demised premises", "property"],
            "date": ["dates", "when", "time", "period", "effective date", "commencement"],
            "amount": ["money", "cost", "price", "payment", "sum", "charge", "fee"],
            "maintenance": ["service", "upkeep", "repair", "cleaning", "utilities"],
        }

    def _get_collection(self, name: str):
        try:
            return self.client.get_collection(name=name)
        except Exception as e:
            print(f"Warning: Collection {name} not found: {e}")
            return None

    def _expand_query(self, query: str) -> List[str]:
        queries = [query]
        ql = query.lower()
        for key, expansions in self.query_expansions.items():
            if key in ql:
                for e in expansions:
                    if e not in ql:
                        queries.append(ql.replace(key, e))
        queries.append(f"entities and parties: {query}")
        queries.append(f"contract details: {query}")
        if not query.endswith("?"):
            queries.append(f"{query}?")
        if "how many" not in ql and ("companies" in ql or "parties" in ql):
            queries.append(f"how many {query}")
            queries.append(f"list all {query}")
            queries.append(f"count of {query}")
        # dedupe preserve order
        seen, out = set(), []
        for q in queries:
            if q not in seen:
                seen.add(q)
                out.append(q)
        return out

    def _search_collection(
        self,
        collection,
        queries: List[str],
        k: int,
        where: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """
        Fast search: don't ask Chroma to return embeddings yet.
        """
        if not collection or not queries:
            return []

        all_results: List[Dict[str, Any]] = []
        seen_ids: set = set()

        for q in queries:
            try:
                qvec = _embed_texts([q])[0]
                kwargs = {
                    "query_embeddings": [qvec],
                    "n_results": min(k * 2, 100),
                    "include": ["documents", "metadatas", "distances"],  # no embeddings here
                }
                if where:
                    kwargs["where"] = where

                res = collection.query(**kwargs)
                docs = res.get("documents", [[]])[0]
                metas = res.get("metadatas", [[]])[0]
                ids = res.get("ids", [[]])[0]
                dists = res.get("distances", [[]])[0]

                for _id, _doc, _dist, _meta in zip(ids, docs, dists, metas):
                    if _id in seen_ids:
                        continue
                    all_results.append({
                        "id": _id,
                        "text": _doc or "",
                        "score": _score_from_distance(_dist),
                        "metadata": _coerce_meta(_meta),
                        "query_used": q,
                        "collection_name": collection.name,
                        # embedding will be backfilled only if we do MMR
                    })
                    seen_ids.add(_id)
            except Exception as e:
                logging.warning(f"Search failed for query '{q}': {e}")
                continue

        all_results.sort(key=lambda x: x["score"], reverse=True)
        return all_results[:k]

    def _backfill_embeddings(self, results: List[Dict[str, Any]]) -> None:
        """
        Fetch embeddings by ID only for the provided results (in-place), grouped per collection.
        """
        # group by collection
        col_to_items: Dict[str, List[Dict[str, Any]]] = {}
        for r in results:
            cname = r.get("collection_name")
            if cname:
                col_to_items.setdefault(cname, []).append(r)

        for cname, items in col_to_items.items():
            try:
                coll = self.client.get_collection(name=cname)
            except Exception:
                continue
            ids = [r["id"] for r in items]
            got = coll.get(ids=ids, include=["embeddings"])
            embs = got.get("embeddings", []) or []
            for r, emb in zip(items, embs):
                r["embedding"] = emb

    def _rerank_by_query_intent(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        ql = original_query.lower()
        for r in results:
            boost = 0.0
            text = r["text"].lower()
            md = r.get("metadata", {})

            if any(w in ql for w in ["company", "companies", "parties", "who", "lessor", "lessee"]):
                if any(md.get(k) for k in ["companies_extracted", "lessor", "lessee", "people_extracted"]):
                    boost += 0.3
                if md.get("search_type") == "entities":
                    boost += 0.2

            if any(w in ql for w in ["how many", "count", "number", "list"]):
                if "entities" in text or "parties" in text:
                    boost += 0.25
                if md.get("companies_extracted"):
                    boost += 0.2

            for kw in ["rent", "term", "maintenance", "security", "termination"]:
                if kw in ql:
                    if md.get("clause_type") == kw:
                        boost += 0.3
                    if kw in text:
                        boost += 0.1

            r["score"] = min(1.0, float(r.get("score", 0.0)) + boost)

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def _advanced_mmr(self, query_vec: np.ndarray, results: List[Dict[str, Any]],
                      k: int, lambda_mult: float = 0.6) -> List[Dict[str, Any]]:
        if len(results) <= k:
            return results

        embeddings = [r.get("embedding") for r in results]
        if not embeddings or embeddings[0] is None:
            return results[:k]

        V = np.array(embeddings, dtype=np.float32)

        def _norm(v: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
            return v / n

        q = _norm(query_vec.reshape(1, -1))[0]
        V = _norm(V)

        selected: List[int] = []
        remaining = list(range(len(results)))
        rel = V @ q

        while len(selected) < k and remaining:
            if not selected:
                best_idx = remaining[int(np.argmax(rel[remaining]))]
            else:
                selV = V[selected]
                mmr_scores = []
                for idx in remaining:
                    similarity = float(V[idx] @ selV.T).max() if selV.size else 0.0
                    mmr = lambda_mult * rel[idx] - (1 - lambda_mult) * similarity
                    mmr_scores.append(mmr)
                best_idx = remaining[int(np.argmax(mmr_scores))]
            selected.append(best_idx)
            remaining.remove(best_idx)

        return [results[i] for i in selected]

    def _merge_and_deduplicate(self, result_sets: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        seen_key = set()
        merged = []
        flat = [r for rs in result_sets for r in rs]
        flat.sort(key=lambda x: x["score"], reverse=True)
        for r in flat:
            key = (r.get("collection_name"), r["id"])
            if key in seen_key:
                continue
            seen_key.add(key)
            merged.append(r)
        return merged

    def _build_enhanced_context(self, chunks: List[Dict[str, Any]],
                                max_chars: int = 15000) -> Tuple[str, List[Dict[str, Any]]]:
        pieces = []
        sources = []
        total = 0
        by_doc = defaultdict(list)

        for ch in chunks:
            original_id = ch["id"]
            if "_entities" in original_id:
                original_id = original_id.replace("_entities", "")
            elif "_clause" in original_id:
                original_id = original_id.replace("_clause", "")
            doc_id = ch.get("metadata", {}).get("doc_id", original_id.split("_")[0])
            by_doc[doc_id].append(ch)

        for doc_id, doc_chunks in by_doc.items():
            doc_chunks.sort(key=lambda x: (
                x.get("metadata", {}).get("page_start", 0),
                x.get("metadata", {}).get("chunk_index", 0)
            ))
            for ch in doc_chunks:
                if total >= max_chars:
                    break
                md = ch.get("metadata", {}) or {}
                header = (md.get("header", "") or "").strip()
                page_start = md.get("page_start")
                page_end = md.get("page_end", page_start)
                clause_type = md.get("clause_type", "")
                search_type = md.get("search_type", "main")

                parts = [doc_id]
                if clause_type and clause_type != "general":
                    parts.append(f"({clause_type})")
                if header:
                    parts.append(header)
                if page_start:
                    page_info = f"p.{page_start}" + (f"-{page_end}" if page_end and page_end != page_start else "")
                    parts.append(page_info)
                if search_type != "main":
                    parts.append(f"[{search_type}]")
                section_id = " :: ".join(parts)

                text = (ch.get("text") or "").strip()
                formatted = f"[{section_id}]\n{text}\n"
                if total + len(formatted) <= max_chars:
                    pieces.append(formatted)
                    total += len(formatted)
                    sources.append({
                        "id": ch["id"],
                        "doc_id": doc_id,
                        "header": header,
                        "clause_type": clause_type,
                        "page_start": page_start,
                        "page_end": page_end,
                        "score": round(float(ch.get("score", 0.0)), 4),
                        "search_type": search_type,
                        "query_used": ch.get("query_used", ""),
                        "text_preview": text[:150] + "..." if len(text) > 150 else text,
                    })
                else:
                    break

        context = "\n---\n".join(pieces)
        return context, sources

    # ----------------------------- Public APIs -----------------------------
    def retrieve_comprehensive(self, question: str, k: int = 12,
                               where: Optional[Dict[str, Any]] = None,
                               use_mmr: bool = True,
                               **kwargs) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        queries = self._expand_query(question)
        print(f"[enhanced_retriever] Generated {len(queries)} query variations")

        result_sets = []
        if self.main_collection:
            main_results = self._search_collection(self.main_collection, queries, k=k*2, where=where)
            result_sets.append(main_results)
            print(f"[enhanced_retriever] Main collection: {len(main_results)} results")

        if self.entity_collection and any(w in question.lower() for w in
                                          ["company", "companies", "party", "parties", "who", "lessor", "lessee"]):
            entity_results = self._search_collection(self.entity_collection, queries, k=k, where=where)
            result_sets.append(entity_results)
            print(f"[enhanced_retriever] Entity collection: {len(entity_results)} results")

        if self.clause_collection:
            clause_results = self._search_collection(self.clause_collection, queries, k=k, where=where)
            result_sets.append(clause_results)
            print(f"[enhanced_retriever] Clause collection: {len(clause_results)} results")

        if not result_sets:
            return [], "", []

        merged = self._merge_and_deduplicate(result_sets)
        print(f"[enhanced_retriever] Merged results: {len(merged)}")

        reranked = self._rerank_by_query_intent(merged, question)

        if use_mmr and len(reranked) > k:
            # Backfill embeddings for a small pool only
            pool = reranked[: max(50, k * 4)]
            self._backfill_embeddings(pool)
            pool = [p for p in pool if p.get("embedding") is not None]

            qvec = np.array(_embed_texts([question])[0], dtype=np.float32)
            final = self._advanced_mmr(qvec, pool, k, lambda_mult=0.7)
        else:
            final = reranked[:k]

        for r in final:
            r.pop("embedding", None)

        context, sources = self._build_enhanced_context(final)
        print(f"[enhanced_retriever] Final results: {len(final)}")
        return final, context, sources

    def retrieve_entities_focused(self, question: str, k: int = 8) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        q = [
            question,
            f"companies and parties: {question}",
            f"entities mentioned: {question}",
            f"lessor and lessee: {question}",
        ]

        results = []
        if self.entity_collection:
            results.extend(self._search_collection(self.entity_collection, q, k=k*2))
        if self.main_collection:
            results.extend(self._search_collection(self.main_collection, q, k=k))

        merged = self._merge_and_deduplicate([results])
        final = merged[:k]
        for r in final:
            r.pop("embedding", None)
        ctx, src = self._build_enhanced_context(final)
        return final, ctx, src


def get_enhanced_retriever(collection_prefix: str = COLLECTION_NAME) -> EnhancedRetriever:
    return EnhancedRetriever(collection_prefix)


def retrieve_advanced(question: str, k: int = 8, **kwargs):
    retriever = get_enhanced_retriever()
    return retriever.retrieve_comprehensive(question, k=k, **kwargs)


def retrieve_entities(question: str, k: int = 8):
    retriever = get_enhanced_retriever()
    return retriever.retrieve_entities_focused(question, k=k)


def retrieve(*args, **kwargs):
    return retrieve_advanced(*args, **kwargs)
