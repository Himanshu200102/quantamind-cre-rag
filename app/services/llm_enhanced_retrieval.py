# app/services/llm_enhanced_retrieval.py
from __future__ import annotations

import os
import json
import re
import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass

# from app.services.enhanced_retriever import EnhancedRetriever
from app.services.enhanced_retriever_milvus import EnhancedRetrieverMilvus as EnhancedRetriever
from app.services.local_llm import LocalLlama, coerce_json
from app.utils.timing import timed  # << added: timing blocks

log = logging.getLogger(__name__)

# ---------------- Configuration ----------------

_DEFAULT_MODEL_NAME = "local-llama"  # informational

TERM_ALIASES: Dict[str, List[str]] = {
    "effective date": ["commencement date", "start date", "date of commencement"],
    "lease term": ["tenure", "term", "duration"],
    "end date": ["expiration date", "expiry date"],
    "rent": ["base rent", "monthly rent", "lease rent"],
    "security deposit": ["refundable deposit", "deposit", "security money"],
}

INTENT_CANON = {
    "entity_search": ["company", "companies", "party", "parties", "lessor", "lessee", "who"],
    "financial_terms": ["rent", "payment", "amount", "price", "cost", "deposit", "security", "service charge", "monthly"],
    "temporal_info": ["term", "tenure", "commencement", "effective", "expiration", "expiry", "end date", "start date", "when"],
    "legal_clauses": ["assignment", "sublease", "insurance", "taxes", "use", "improvements", "maintenance", "termination", "notice"],
    "comprehensive_summary": [],
}


@dataclass
class QueryAnalysis:
    intent: str
    entities_needed: List[str]
    query_expansions: List[str]
    search_strategy: str
    expected_answer_type: str
    confidence: float


def _infer_intent_heuristic(q: str) -> str:
    ql = q.lower()
    for intent, kws in INTENT_CANON.items():
        if any(k in ql for k in kws):
            return intent
    return "comprehensive_summary"


def _strip_variant_suffix(cid: str) -> str:
    base = cid.replace("_entities", "").replace("_clause", "")
    base = re.sub(r"_w\d+$", "", base)
    return base


def _merge_unique_by_base_id(chunks: List[Dict]) -> List[Dict]:
    seen: Set[str] = set()
    out: List[Dict] = []
    for ch in sorted(chunks, key=lambda x: x.get("score", 0.0), reverse=True):
        key = _strip_variant_suffix(ch["id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return out


# ---------------- LLM wrapper ----------------

class _LocalLLM:
    def __init__(self):
        self.model_name = _DEFAULT_MODEL_NAME
        self.llm = LocalLlama()

    def gen(self, prompt: str, temp: float = 0.2, max_tokens: int = 1024, retries: int = 1) -> str:
        err: Optional[Exception] = None
        for _ in range(retries + 1):
            try:
                return self.llm.gen(prompt, temp=temp, max_tokens=max_tokens)
            except Exception as e:
                err = e
                time.sleep(0.4)
        raise err if err else RuntimeError("LLM generation failed")


# ---------------- Core class ----------------

class LLMEnhancedRetriever:
    """
    Enterprise-level retrieval using local LLaMA for:
    1) Query analysis & expansion
    2) LLM reranking
    3) Gap detection
    4) Multi-hop retrieval
    """

    def __init__(self, collection_prefix: str = "lease_chunks"):
        self.llm = _LocalLLM()
        self.base_retriever = EnhancedRetriever(collection_prefix)

    # -------- Query analysis --------
    def analyze_query_with_llm(self, question: str) -> QueryAnalysis:
        prompt = f"""You analyze user queries for CRE/lease retrieval.
Analyze: "{question}"
Return ONLY JSON:
{{
  "intent": "entity_search|financial_terms|temporal_info|legal_clauses|comprehensive_summary",
  "entities_needed": ["companies","people","amounts","dates"],
  "query_expansions": ["5-9 semantically related queries"],
  "search_strategy": "broad|specific|multi_hop|entity_focused",
  "expected_answer_type": "count|list|explanation|summary|comparison",
  "confidence": 0.0-1.0
}}"""
        try:
            with timed("llm:analyze_query"):
                raw = self.llm.gen(prompt, temp=0.15, max_tokens=400)
            data = coerce_json(raw)
            return QueryAnalysis(
                intent=str(data.get("intent") or _infer_intent_heuristic(question)),
                entities_needed=list(data.get("entities_needed") or []),
                query_expansions=list(data.get("query_expansions") or [question]),
                search_strategy=str(data.get("search_strategy") or "broad"),
                expected_answer_type=str(data.get("expected_answer_type") or "explanation"),
                confidence=float(data.get("confidence") or 0.5),
            )
        except Exception as e:
            logging.warning(f"LLM query analysis failed: {e}")
            return QueryAnalysis(
                intent=_infer_intent_heuristic(question),
                entities_needed=["companies", "people", "amounts", "dates"],
                query_expansions=[question],
                search_strategy="broad",
                expected_answer_type="explanation",
                confidence=0.35,
            )

    def _add_domain_alias_expansions(self, original_query: str, expansions: List[str]) -> List[str]:
        ql = original_query.lower()
        out = list(expansions)
        for canonical, alts in TERM_ALIASES.items():
            if canonical in ql:
                for a in alts:
                    out.append(ql.replace(canonical, a))
                    out.append(f"{ql} {a}")
            else:
                for a in alts:
                    if a in ql:
                        out.append(ql.replace(a, canonical))
                        out.append(f"{ql} {canonical}")
        # dedupe preserving order
        seen = set()
        final = []
        for q in [original_query, *out]:
            if q not in seen:
                seen.add(q)
                final.append(q)
        return final

    def generate_search_queries(self, original_query: str, analysis: QueryAnalysis) -> List[str]:
        prompt = f"""Generate 8-12 search queries for a CRE/lease retrieval system.
Original: "{original_query}"
Intent: {analysis.intent}
Entities: {analysis.entities_needed}
Expected: {analysis.expected_answer_type}
Output ONLY a JSON array of query strings.
"""
        try:
            with timed("llm:gen_queries"):
                raw = self.llm.gen(prompt, temp=0.2, max_tokens=500)
            queries = coerce_json(raw)
            if not isinstance(queries, list):
                queries = [original_query]
        except Exception as e:
            logging.warning(f"LLM query generation failed: {e}")
            queries = list(analysis.query_expansions or [original_query])
        return self._add_domain_alias_expansions(original_query, queries)

    # -------- Reranking --------
    def rerank_with_llm(self, chunks: List[Dict], original_query: str, analysis: QueryAnalysis) -> List[Dict]:
        if len(chunks) <= 3:
            return chunks

        chunk_summaries = []
        for i, ch in enumerate(chunks[:15]):
            md = ch.get("metadata", {}) or {}
            header = md.get("header", "")
            clause_type = md.get("clause_type", "")
            text_preview = ch.get("text", "")[:360]
            chunk_summaries.append({
                "index": i,
                "header": header,
                "clause_type": clause_type,
                "text_preview": text_preview,
                "score": round(float(ch.get("score", 0.0)), 4),
            })

        prompt = f"""Rerank results for: "{original_query}"
Intent: {analysis.intent}
Expected: {analysis.expected_answer_type}
Chunks: {json.dumps(chunk_summaries, indent=2)}
Return ONLY a JSON array of indices best→worst, e.g. [0,2,1]. Prefer direct answers, completeness, multi-entity coverage when relevant.
"""

        order: Optional[List[int]] = None
        try:
            with timed("llm:rerank"):
                raw = self.llm.gen(prompt, temp=0.0, max_tokens=300)
            data = coerce_json(raw)
            if isinstance(data, list) and all(isinstance(x, int) for x in data):
                order = [x for x in data if 0 <= x < len(chunks)]
        except Exception as e:
            logging.warning(f"LLM rerank failed: {e}")

        if order:
            reordered = [chunks[i] for i in order]
            mentioned = set(order)
            reordered.extend([c for i, c in enumerate(chunks) if i not in mentioned])
        else:
            reordered = chunks

        # light metadata-based boost
        ql = original_query.lower()
        for r in reordered:
            md = r.get("metadata", {}) or {}
            boost = 0.0
            if md.get("header_priority") == 1:
                boost += 0.05

            ct = (md.get("clause_type") or "").lower()
            if any(k in ql for k in ["rent", "payment", "deposit", "service charge"]):
                if ct in ("rent", "financial", "security"):
                    boost += 0.05
                if md.get("amounts_extracted"):
                    boost += 0.05

            if any(k in ql for k in ["term", "tenure", "commencement", "effective", "expiration", "expiry"]):
                if ct == "term":
                    boost += 0.05
                if md.get("dates_extracted"):
                    boost += 0.05

            if any(k in ql for k in ["company", "companies", "party", "parties", "lessor", "lessee", "who"]):
                if any(md.get(k) for k in ["companies_extracted", "lessor", "lessee", "people_extracted"]):
                    boost += 0.06
                if md.get("search_type") == "entities":
                    boost += 0.04

            r["score"] = min(1.0, float(r.get("score", 0.0)) + boost)

        reordered.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return reordered

    # -------- Gap detection --------
    def detect_missing_information(self, question: str, chunks: List[Dict]) -> List[str]:
        ql = question.lower()
        md_list = [c.get("metadata", {}) or {} for c in chunks[:8]]
        out: List[str] = []

        if any(k in ql for k in ["term", "tenure", "commencement", "effective", "expiration", "expiry"]):
            if not any(m.get("dates_extracted") for m in md_list):
                out += ["lease commencement and expiration date section", "renewal/extension and notice period clauses"]

        if any(k in ql for k in ["rent", "payment", "amount", "deposit", "service charge"]):
            if not any(m.get("amounts_extracted") for m in md_list):
                out += ["base rent and escalation schedule", "security deposit and refund terms"]

        chunk_content = "\n\n".join([
            f"[{(c.get('metadata', {}) or {}).get('header','Section')}] {(c.get('text','') or '')[:220]}..."
            for c in chunks[:8]
        ])
        prompt = f"""Original: "{question}"
Retrieved: {chunk_content}
If anything is missing to fully answer, return ONLY a JSON array of follow-up search queries (0-6 items). Else [].
"""
        try:
            with timed("llm:gap_detect"):
                raw = self.llm.gen(prompt, temp=0.1, max_tokens=300)
            data = coerce_json(raw)
            if isinstance(data, list):
                out.extend([q for q in data if isinstance(q, str)])
        except Exception as e:
            logging.warning(f"Gap detection failed: {e}")

        # dedupe, preserve order
        seen = set()
        final: List[str] = []
        for q in out:
            if q not in seen:
                seen.add(q)
                final.append(q)
        return final

    # -------- Pipelines --------
    def enterprise_retrieve(self, question: str, k: int = 15, max_iterations: int = 2) -> Tuple[List[Dict], str, List[Dict]]:
        print(f"[Enterprise Retrieval] Starting for: {question}")
        with timed("pipeline:analyze"):
            analysis = self.analyze_query_with_llm(question)
        print(f"[Enterprise Retrieval] Intent: {analysis.intent}, Strategy: {analysis.search_strategy}")

        with timed("pipeline:gen_search_queries"):
            search_queries = self.generate_search_queries(question, analysis)
        print(f"[Enterprise Retrieval] Generated {len(search_queries)} search queries")

        all_chunks: List[Dict] = []

        if analysis.intent in ["entity_search"] or any(w in question.lower() for w in ["compan", "part", "who"]):
            with timed("retrieval:entities_focused"):
                entity_chunks, _, _ = self.base_retriever.retrieve_entities_focused(question, k=max(4, k // 2))
            all_chunks.extend(entity_chunks)
            print(f"[Enterprise Retrieval] Entity search: {len(entity_chunks)} chunks")

        with timed("retrieval:batch_queries"):
            for q in search_queries[:6]:
                try:
                    q_chunks, _, _ = self.base_retriever.retrieve_comprehensive(q, k=max(6, k // 3))
                    all_chunks.extend(q_chunks)
                except Exception as e:
                    logging.warning(f"Search failed for query '{q}': {e}")

        unique_chunks = _merge_unique_by_base_id(all_chunks)
        unique_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        top_chunks = unique_chunks[: max(k * 2, k)]
        print(f"[Enterprise Retrieval] Found {len(unique_chunks)} unique chunks")

        with timed("pipeline:rerank"):
            reranked = self.rerank_with_llm(top_chunks, question, analysis)
        reranked = _merge_unique_by_base_id(reranked)
        final_chunks = reranked[:k]

        if analysis.search_strategy in ["multi_hop", "broad"] and final_chunks:
            with timed("pipeline:gap_detection"):
                missing = self.detect_missing_information(question, final_chunks)
            if missing:
                print(f"[Enterprise Retrieval] Found {len(missing)} gaps, follow-ups...")
                with timed("retrieval:gap_followups"):
                    seen_ids = {c["id"] for c in final_chunks}
                    for gq in missing[:3]:
                        try:
                            g_chunks, _, _ = self.base_retriever.retrieve_comprehensive(gq, k=max(4, k // 4))
                            for gc in g_chunks:
                                if gc["id"] not in seen_ids:
                                    final_chunks.append(gc)
                                    seen_ids.add(gc["id"])
                        except Exception as e:
                            logging.warning(f"Gap search failed for '{gq}': {e}")

        final_chunks = _merge_unique_by_base_id(final_chunks)[:k]
        with timed("context:build"):
            context, sources = self.base_retriever._build_enhanced_context(final_chunks, max_chars=20000)
        print(f"[Enterprise Retrieval] Final result: {len(final_chunks)} chunks")
        return final_chunks, context, sources

    def hybrid_search(self, question: str, k: int = 12) -> Tuple[List[Dict], str, List[Dict]]:
        """
        Robust, LLM-free hybrid retrieval with timing logs.
        Requires EnhancedRetrieverMilvus.retrieve_hybrid_robust(...)
        """
        with timed("retrieval(hybrid)"):
            # Calls the optimized Milvus-only path (entity-first + coverage top-up)
            final, ctx, src = self.base_retriever.retrieve_hybrid_robust(question, k)
        return final, ctx, src

    def multi_hop_retrieval(self, question: str, k: int = 12, max_hops: int = 3) -> Tuple[List[Dict], str, List[Dict]]:
        all_chunks: List[Dict] = []
        processed: Set[str] = set()
        current: List[str] = [question]
        for hop in range(max_hops):
            print(f"[Multi-hop] Hop {hop + 1}: {len(current)} queries")
            with timed(f"retrieval:multi_hop[{hop+1}]"):
                hop_chunks: List[Dict] = []
                for q in current:
                    if q in processed:
                        continue
                    processed.add(q)
                    chs, _, _ = self.base_retriever.retrieve_comprehensive(q, k=max(4, k // 2))
                    hop_chunks.extend(chs)
            if not hop_chunks:
                break
            all_chunks.extend(hop_chunks)
            if hop < max_hops - 1:
                with timed(f"llm:gen_followups[{hop+1}]"):
                    current = self.generate_followup_queries(hop_chunks, question)

        unique = _merge_unique_by_base_id(all_chunks)
        unique.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        final = unique[:k]
        with timed("context:build"):
            ctx, src = self.base_retriever._build_enhanced_context(final)
        return final, ctx, src

    def generate_followup_queries(self, chunks: List[Dict], original_question: str) -> List[str]:
        chunk_content = "\n".join([
            f"{(c.get('metadata', {}) or {}).get('header', '')}: {(c.get('text','') or '')[:220]}"
            for c in chunks[:5]
        ])
        prompt = f"""Original: "{original_question}"
Retrieved: {chunk_content}
Return ONLY a JSON array of 3-5 follow-up queries to extend coverage.
"""
        try:
            with timed("llm:gen_followups"):
                raw = self.llm.gen(prompt, temp=0.2, max_tokens=300)
            data = coerce_json(raw)
            if isinstance(data, list):
                return [s for s in data if isinstance(s, str)]
        except Exception as e:
            logging.warning(f"Follow-up generation failed: {e}")
        return []


# ---------------- System wrapper ----------------

class EnterpriseRAGSystem:
    def __init__(self, collection_prefix: str = "lease_chunks"):
        self.llm_retriever = LLMEnhancedRetriever(collection_prefix)

    def enterprise_search(self, question: str, k: int = 12, method: str = "comprehensive") -> Dict[str, Any]:
        if method == "hybrid":
            chunks, context, sources = self.llm_retriever.hybrid_search(question, k)
            method_used = "hybrid"
        elif method == "multi_hop":
            chunks, context, sources = self.llm_retriever.multi_hop_retrieval(question, k)
            method_used = "multi_hop"
        else:
            chunks, context, sources = self.llm_retriever.enterprise_retrieve(question, k)
            method_used = "comprehensive"

        entities = self.extract_all_entities(chunks)
        confidence = self.calculate_confidence(question, chunks, entities)

        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
            "entities_found": entities,
            "confidence_score": confidence,
            "method_used": method_used,
            "total_chunks": len(chunks),
            "context_length": len(context),
        }

    def extract_all_entities(self, chunks: List[Dict]) -> Dict[str, List[str]]:
        entities = {"companies": set(), "people": set(), "amounts": set(), "dates": set(), "locations": set()}
        for ch in chunks:
            md = ch.get("metadata", {}) or {}
            if md.get("companies_extracted"):
                companies = [c.strip() for c in str(md["companies_extracted"]).split(";") if c.strip()]
                entities["companies"].update(companies)
            for key in ["lessor", "lessee"]:
                if md.get(key):
                    entities["companies"].add(md[key])
            for et in ["people", "amounts", "dates", "locations"]:
                k = f"{et}_extracted"
                if md.get(k):
                    items = [x.strip() for x in str(md[k]).split(";") if x.strip()]
                    entities[et].update(items)
        return {k: sorted(list(v)) for k, v in entities.items() if v}

    def calculate_confidence(self, question: str, chunks: List[Dict], entities: Dict) -> float:
        conf = 0.0
        conf += min(len(chunks) / 10.0, 0.3)
        if chunks:
            avg = sum(float(c.get("score", 0.0)) for c in chunks) / len(chunks)
            conf += max(0.0, min(avg, 1.0)) * 0.4
        ql = question.lower()
        if "compan" in ql and entities.get("companies"):
            conf += 0.2
        if "amount" in ql and entities.get("amounts"):
            conf += 0.2
        return min(conf, 1.0)


# Factories

def get_enterprise_rag_system(collection_prefix: str = "lease_chunks") -> EnterpriseRAGSystem:
    return EnterpriseRAGSystem(collection_prefix)


def enterprise_retrieve(question: str, k: int = 12, method: str = "comprehensive") -> Dict[str, Any]:
    system = get_enterprise_rag_system()
    return system.enterprise_search(question, k=k, method=method)
