# app/services/enhanced_retriever_milvus.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import re
from collections import defaultdict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.services.milvus_client import (
    ensure_collection, ensure_loaded, search_vectors
)
from app.services.faiss_cache import FAISSCache
from app.utils.timing import timed


EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_PREFIX = os.getenv("MILVUS_COLLECTION_PREFIX", "lease_chunks")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# hybrid knobs
HY_TOPUP_MIN_UNIQUE = int(os.getenv("HYBRID_TOPUP_MIN_UNIQUE", "6"))
HY_ENTITY_EXTRA_QUERIES = int(os.getenv("HYBRID_ENTITY_EXTRA_QUERIES", "3"))
HY_MAX_QUERY_VARIANTS = int(os.getenv("HYBRID_MAX_QUERY_VARIANTS", "10"))
HY_PER_QUERY_K_MAIN = int(os.getenv("HYBRID_PER_QUERY_K_MAIN", "8"))
HY_PER_QUERY_K_SIDE = int(os.getenv("HYBRID_PER_QUERY_K_SIDE", "4"))

def _score_from_distance(distance: float) -> float:
    try:
        d = float(distance)
    except Exception:
        return 0.0
    return max(0.0, min(d, 1.0))

def _strip_variant_suffix(cid: str) -> str:
    base = cid.replace("_entities", "").replace("_clause", "")
    base = re.sub(r"_w\d+$", "", base)
    return base

def _dedupe_by_base_id(chunks: List[Dict]) -> List[Dict]:
    seen = set()
    out: List[Dict] = []
    for ch in sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True):
        key = _strip_variant_suffix(ch["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out

def _keyword_overlap_boost(q: str, text: str, header: str = "") -> float:
    q_tokens = {t for t in re.findall(r"[a-z0-9]+", q.lower()) if len(t) > 2}
    if not q_tokens: 
        return 0.0
    t_tokens = {t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) > 2}
    h_tokens = {t for t in re.findall(r"[a-z0-9]+", (header or "").lower()) if len(t) > 2}
    overlap = (t_tokens | h_tokens) & q_tokens
    if not overlap:
        return 0.0
    return min(0.12, 0.02 * len(overlap))

def _is_entity_intent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["company", "companies", "party", "parties", "who", "lessor", "lessee", "entity", "entities"])

def _is_temporal_intent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["term", "tenure", "commencement", "effective", "expiration", "expiry", "start date", "end date", "when", "date"])

def _is_financial_intent(q: str) -> bool:
    ql = q.lower()
    return any(w in ql for w in ["rent", "payment", "amount", "price", "cost", "deposit", "security", "service charge", "monthly", "escalation"])


class EnhancedRetrieverMilvus:
    """
    Milvus-backed retriever with Hybrid++ search.
    Now uses FAISSCache for semantic result caching.
    """

    def __init__(self, collection_prefix: str = COLLECTION_PREFIX):
        self.prefix = collection_prefix
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(EMBED_MODEL, device=device)

        self.col_main     = ensure_collection(f"{self.prefix}_main")
        self.col_entities = ensure_collection(f"{self.prefix}_entities")
        self.col_clauses  = ensure_collection(f"{self.prefix}_clauses")
        for c in (self.col_main, self.col_entities, self.col_clauses):
            ensure_loaded(c)

        self.domain_terms = {
            "company": ["companies", "corporation", "firm", "organization", "business", "party", "lessor", "lessee", "entity", "entities"],
            "parties": ["lessor", "lessee", "company", "companies", "entities"],
            "rent": ["base rent", "rental", "payment", "monthly rent", "lease payment", "escalation", "amount", "cost"],
            "term": ["duration", "period", "tenure", "lease term", "expiry", "expiration", "commencement", "effective date", "end date", "start date"],
            "space": ["premises", "area", "floor", "suite", "demised premises", "property"],
            "maintenance": ["service", "upkeep", "repair", "cleaning", "utilities"],
            "deposit": ["security deposit", "refundable deposit", "security"],
        }

        self.cache = FAISSCache(dim=EMBED_DIM, index_type="Flat")

    def _expand_query(self, query: str) -> List[str]:
        ql = query.strip()
        variants = [ql]
        low = ql.lower()

        for key, alts in self.domain_terms.items():
            if key in low:
                for a in alts:
                    v = low.replace(key, a)
                    if v not in variants:
                        variants.append(v)
        adders = [
            f"entities and parties: {ql}",
            f"contract details: {ql}",
            f"clause references: {ql}",
            f"{ql}?",
        ]
        if "how many" not in low and any(x in low for x in ["companies", "company", "parties"]):
            adders += [f"how many {ql}", f"list all {ql}", f"count of {ql}"]

        for a in adders:
            if a not in variants:
                variants.append(a)

        return variants[:HY_MAX_QUERY_VARIANTS]

    def _embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def _search_one(self, col, query_vec: List[float], k: int, where_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        res = search_vectors(col, [query_vec], limit=min(k*2, 100), expr=where_expr)
        hits = res[0] if res else []
        out: List[Dict[str, Any]] = []
        seen = set()
        for h in hits:
            _id = h.entity.get("id")
            if _id in seen:
                continue
            seen.add(_id)
            text = h.entity.get("text") or ""
            meta_raw = h.entity.get("metadata") or "{}"
            try:
                meta = json.loads(meta_raw)
            except Exception:
                meta = {}
            out.append({
                "id": _id,
                "text": text,
                "score": _score_from_distance(h.distance),
                "metadata": meta,
                "collection_name": col.name,
            })
        out.sort(key=lambda x: x["score"], reverse=True)
        return out[:k]

    def _merge_and_dedup(self, lists: List[List[Dict[str, Any]]], k: int) -> List[Dict[str, Any]]:
        flat = [r for L in lists for r in L]
        flat.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        seen_ids = set()
        out = []
        for r in flat:
            if r["id"] in seen_ids:
                continue
            seen_ids.add(r["id"])
            out.append(r)
            if len(out) >= k:
                break
        return out

    def _diversity_filter(self, chunks: List[Dict], want: int) -> List[Dict]:
        if len(chunks) <= want:
            return chunks
        seen_key = {}
        out = []
        for c in chunks:
            md = c.get("metadata", {}) or {}
            key = (md.get("doc_id", ""), md.get("header", ""), md.get("clause_type", ""))
            cnt = seen_key.get(key, 0)
            if cnt < 2:
                out.append(c)
                seen_key[key] = cnt + 1
            if len(out) >= want:
                break
        return out

    def _apply_boosts(self, q: str, items: List[Dict]) -> List[Dict]:
        low = q.lower()
        for r in items:
            md = r.get("metadata", {}) or {}
            header = md.get("header", "") or ""
            boost = _keyword_overlap_boost(q, r.get("text", ""), header)

            ct = (md.get("clause_type") or "").lower()
            if _is_financial_intent(low) and ct in ("rent", "financial", "security"):
                boost += 0.06
            if _is_temporal_intent(low) and (ct == "term" or md.get("dates_extracted")):
                boost += 0.06
            if _is_entity_intent(low) and any(md.get(k) for k in ["companies_extracted", "lessor", "lessee", "people_extracted"]):
                boost += 0.07
            if md.get("header_priority") == 1:
                boost += 0.03

            r["score"] = min(1.0, float(r.get("score", 0.0)) + boost)

        items.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return items

    def _build_enhanced_context(self, chunks: List[Dict[str, Any]], max_chars: int = 15000) -> Tuple[str, List[Dict[str, Any]]]:
        pieces = []
        sources = []
        total = 0
        by_doc = defaultdict(list)

        for ch in chunks:
            original_id = ch["id"]
            doc_id = ch.get("metadata", {}).get("doc_id", _strip_variant_suffix(original_id).split("_")[0])
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
                    page_info = f"p.{page_start}" + (f"-{page_end}" if page_end != page_start else "")
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
                        "text_preview": text[:150] + "..." if len(text) > 150 else text,
                    })
                else:
                    break

        context = "\n---\n".join(pieces)
        return context, sources

    def retrieve_hybrid_robust(self, question: str, k: int = 12) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        qvec = self._embed([question])[0]

        # 1) Try FAISS cache first
        cache_results = self.cache.search([qvec], k=1)
        if cache_results:
            hit_id = cache_results[0]["id"]
            # ID stored as query text
            if hit_id == question:
                print(f"[cache hit] {question}")
                return cache_results[0].get("payload")

        # 2) Otherwise, do full retrieval
        with timed("hybrid.expand_embed"):
            queries = self._expand_query(question)
            qvecs = self._embed(queries)

        entity_intent = _is_entity_intent(question)
        results_sets: List[List[Dict[str, Any]]] = []

        with timed("hybrid.seed_and_topup"):
            if entity_intent:
                for q, v in zip(queries[:1 + HY_ENTITY_EXTRA_QUERIES], qvecs[:1 + HY_ENTITY_EXTRA_QUERIES]):
                    ents = self._search_one(self.col_entities, v, k=max(k, HY_PER_QUERY_K_MAIN))
                    pack = self._apply_boosts(q, ents)
                    results_sets.append(pack)

            for q, v in zip(queries, qvecs):
                main = self._search_one(self.col_main, v, k=HY_PER_QUERY_K_MAIN)
                claus = self._search_one(self.col_clauses, v, k=HY_PER_QUERY_K_SIDE)
                pack = self._apply_boosts(q, main + claus)
                results_sets.append(pack)

        with timed("hybrid.merge"):
            merged = []
            for pack in results_sets:
                merged.extend(pack)
            merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            merged = _dedupe_by_base_id(merged)

        if len(merged) < max(k, HY_TOPUP_MIN_UNIQUE):
            with timed("hybrid.topup"):
                need = max(k, HY_TOPUP_MIN_UNIQUE) - len(merged)
                for q, v in zip(queries, qvecs):
                    if need <= 0:
                        break
                    extra = self._search_one(self.col_main, v, k=need)
                    extra = self._apply_boosts(q, extra)
                    seen = {_strip_variant_suffix(m["id"]) for m in merged}
                    for e in extra:
                        if _strip_variant_suffix(e["id"]) not in seen:
                            merged.append(e)
                            seen.add(_strip_variant_suffix(e["id"]))
                            need -= 1
                            if need <= 0:
                                break

        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        merged = self._diversity_filter(merged, want=max(k*2, k))
        final = merged[:k]

        with timed("hybrid.merge_context"):
            context, sources = self._build_enhanced_context(final)

        # 3) Store results in FAISS cache
        self.cache.add([qvec], [question])
        self.cache.save()

        payload = (final, context, sources)
        print(f"[hybrid] entity_intent={entity_intent} unique~={len(final)} final_k={k}")
        return payload
