# app/services/enhanced_retriever_milvus.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import json
import os
import time
import re
from collections import defaultdict, OrderedDict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.services.milvus_client import (
    ensure_collection, ensure_loaded, search_vectors
)
from app.utils.timing import timed

# ----------------- ENV / Config -----------------
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
COLLECTION_PREFIX = os.getenv("MILVUS_COLLECTION_PREFIX", "lease_chunks")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

# hybrid knobs
HY_TOPUP_MIN_UNIQUE = int(os.getenv("HYBRID_TOPUP_MIN_UNIQUE", "6"))
HY_ENTITY_EXTRA_QUERIES = int(os.getenv("HYBRID_ENTITY_EXTRA_QUERIES", "3"))
HY_MAX_QUERY_VARIANTS = int(os.getenv("HYBRID_MAX_QUERY_VARIANTS", "10"))
HY_PER_QUERY_K_MAIN = int(os.getenv("HYBRID_PER_QUERY_K_MAIN", "8"))
HY_PER_QUERY_K_SIDE = int(os.getenv("HYBRID_PER_QUERY_K_SIDE", "4"))
HY_ENABLE_RESULT_CACHE = os.getenv("HYBRID_ENABLE_CACHE", "true").lower() == "true"
HY_CACHE_MAX = int(os.getenv("HYBRID_CACHE_MAX", "256"))
HY_CACHE_TTL_SEC = int(os.getenv("HYBRID_CACHE_TTL_SEC", "600"))

# ----------------- Helpers -----------------

def _score_from_distance(distance: float) -> float:
    """
    Milvus returns 'distance' where, for IP (cosine with normalized vecs), larger = more similar.
    We clamp into [0,1]. For L2, you could map via 1/(1+d), but we're standardizing on IP+normalized.
    """
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
    """
    Very fast lexical signal to complement embeddings.
    - rewards shared rare-ish tokens; harmless for speed.
    """
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
    Milvus-backed retriever with robust, LLM-free "Hybrid++" path:
      - domain-aware query expansion (not just a few hardcoded swaps)
      - entity-first seeding (when applicable)
      - coverage top-up across main/clauses
      - keyword-overlap boosting + metadata-aware nudges
      - dedupe by base-id; basic diversity guard
      - tiny in-process result cache (fast hot path)
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

        self._cache: OrderedDict[str, Tuple[float, Tuple[List[Dict], str, List[Dict]]]] = OrderedDict()

    def _expand_query(self, query: str) -> List[str]:
        """
        More general expansion:
         - synonym/alias swaps
         - role prompts (entities/clauses/main)
         - question phrasing variants (list/count/who/what)
        Bounded to HY_MAX_QUERY_VARIANTS for speed.
        """
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
        # light metadata-aware boost + keyword overlap will run later
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
        """
        Avoid too many near-duplicates from the same header/section; cheap heuristic.
        """
        if len(chunks) <= want:
            return chunks
        seen_key = {}
        out = []
        for c in chunks:
            md = c.get("metadata", {}) or {}
            key = (md.get("doc_id", ""), md.get("header", ""), md.get("clause_type", ""))
            cnt = seen_key.get(key, 0)
            if cnt < 2:  # allow up to 2 from same section
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

            # domain nudges
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

    # ---------- Public APIs ----------
    def retrieve_comprehensive(
        self,
        question: str,
        k: int = 12,
        where: Optional[Dict[str, Any]] = None,
        use_mmr: bool = True,
        **kwargs
    ) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        # kept for compatibility with previous code paths
        queries = self._expand_query(question)
        expr = None
        results_sets: List[List[Dict[str, Any]]] = []
        qvecs = self._embed(queries)
        for q, v in zip(queries, qvecs):
            main = self._search_one(self.col_main, v, k=HY_PER_QUERY_K_MAIN, where_expr=expr)
            ents = self._search_one(self.col_entities, v, k=HY_PER_QUERY_K_SIDE, where_expr=expr)
            claus = self._search_one(self.col_clauses, v, k=HY_PER_QUERY_K_SIDE, where_expr=expr)
            pack = self._apply_boosts(q, main + ents + claus)
            results_sets.append(pack)

        merged = self._merge_and_dedup(results_sets, k=max(k*3, 24))
        final = merged[:k]
        context, sources = self._build_enhanced_context(final)
        return final, context, sources

    def retrieve_entities_focused(self, question: str, k: int = 8) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        qlist = [
            question,
            f"entities and parties: {question}",
            f"companies and parties: {question}",
            f"lessor and lessee: {question}",
        ]
        vecs = self._embed(qlist)

        results: List[Dict[str, Any]] = []
        for v, q in zip(vecs, qlist):
            ents = self._search_one(self.col_entities, v, k=k*2)
            main = self._search_one(self.col_main, v, k=k)
            pack = self._apply_boosts(q, ents + main)
            results.extend(pack)

        # dedup & top-k
        seen = set()
        out = []
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            if r["id"] not in seen:
                seen.add(r["id"])
                out.append(r)
            if len(out) >= k:
                break

        ctx, src = self._build_enhanced_context(out)
        return out, ctx, src

    # -------- New: Robust, LLM-free Hybrid++ --------
    def retrieve_hybrid_robust(self, question: str, k: int = 12) -> Tuple[List[Dict[str, Any]], str, List[Dict[str, Any]]]:
        """
        Generalized, fast path for CRE questions (entities, dates, rent, terms, etc.)
        No LLM calls. Uses domain-aware expansions, IP+normalized embeddings, lexical boost,
        entity seeding (if applicable), coverage top-up, dedupe+diversity, micro-cache.
        """
        cache_key = None
        if HY_ENABLE_RESULT_CACHE:
            cache_key = f"{question}::{k}"
            now = time.time()
            # evict expired & maintain LRU
            if cache_key in self._cache:
                ts, payload = self._cache.pop(cache_key)
                if now - ts < HY_CACHE_TTL_SEC:
                    # refresh LRU position
                    self._cache[cache_key] = (ts, payload)
                    return payload

        with timed("hybrid.expand_embed"):
            queries = self._expand_query(question)
            qvecs = self._embed(queries)

        entity_intent = _is_entity_intent(question)
        results_sets: List[List[Dict[str, Any]]] = []

        with timed("hybrid.seed_and_topup"):
            # 1) seed: entity-first if likely entity question
            if entity_intent:
                for q, v in zip(queries[:1 + HY_ENTITY_EXTRA_QUERIES], qvecs[:1 + HY_ENTITY_EXTRA_QUERIES]):
                    ents = self._search_one(self.col_entities, v, k=max(k, HY_PER_QUERY_K_MAIN))
                    pack = self._apply_boosts(q, ents)
                    results_sets.append(pack)

            # 2) coverage top-up: always query main/clauses with all variants (capped)
            for q, v in zip(queries, qvecs):
                main = self._search_one(self.col_main, v, k=HY_PER_QUERY_K_MAIN)
                claus = self._search_one(self.col_clauses, v, k=HY_PER_QUERY_K_SIDE)
                pack = self._apply_boosts(q, main + claus)
                results_sets.append(pack)

        # 3) merge & dedupe
        with timed("hybrid.merge"):
            merged = []
            for pack in results_sets:
                merged.extend(pack)
            # sort by boosted score first
            merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
            merged = _dedupe_by_base_id(merged)

        # 4) ensure coverage: if we didn’t reach enough unique after seed+topup, keep adding
        # (this is cheap because each _search_one already limited k tightly)
        if len(merged) < max(k, HY_TOPUP_MIN_UNIQUE):
            with timed("hybrid.topup"):
                need = max(k, HY_TOPUP_MIN_UNIQUE) - len(merged)
                # re-run top variants against main quickly
                for q, v in zip(queries, qvecs):
                    if need <= 0:
                        break
                    extra = self._search_one(self.col_main, v, k=need)
                    extra = self._apply_boosts(q, extra)
                    # add only new base-ids
                    seen = {_strip_variant_suffix(m["id"]) for m in merged}
                    for e in extra:
                        if _strip_variant_suffix(e["id"]) not in seen:
                            merged.append(e)
                            seen.add(_strip_variant_suffix(e["id"]))
                            need -= 1
                            if need <= 0:
                                break

        # 5) final sort & diversity guard
        merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        merged = self._diversity_filter(merged, want=max(k*2, k))
        final = merged[:k]

        with timed("hybrid.merge_context"):
            context, sources = self._build_enhanced_context(final)

        # cache it
        if HY_ENABLE_RESULT_CACHE:
            self._cache[cache_key] = (time.time(), (final, context, sources))
            # enforce size
            while len(self._cache) > HY_CACHE_MAX:
                self._cache.popitem(last=False)

        print(f"[hybrid] entity_intent={entity_intent} unique~={len(final)} final_k={k}")
        return final, context, sources
