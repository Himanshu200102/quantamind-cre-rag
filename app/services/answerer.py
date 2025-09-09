# app/services/answerer.py
from __future__ import annotations
import re
import json
import logging
from typing import Any, Dict, List, Tuple

from app.services.local_llm import LocalLlama, coerce_json

# ----- Simple intent detection (fast & stable) -----
def detect_intent(q: str) -> str:
    ql = q.lower()
    if any(x in ql for x in ["company", "companies", "party", "parties", "lessor", "lessee", "who"]):
        return "entities"
    if any(x in ql for x in ["rent", "payment", "amount", "price", "cost", "deposit", "security", "service charge"]):
        return "financial"
    if any(x in ql for x in ["term", "duration", "date", "when", "effective", "commencement", "expiration", "expiry"]):
        return "dates"
    return "summary"

# ----- Small helpers -----
def _gather_context(chunks: List[Dict], max_sections: int = 10) -> str:
    parts = []
    for i, c in enumerate(chunks[:max_sections]):
        md = c.get("metadata", {}) or {}
        header = md.get("header", f"Section {i+1}")
        page = md.get("page_start")
        tag = [header]
        if md.get("doc_type"):
            tag.append(f"({md['doc_type']})")
        if md.get("clause_type") and md["clause_type"] != "general":
            tag.append(f"[{md['clause_type']}]")
        if page:
            tag.append(f"p.{page}")
        stamp = " :: ".join(tag)
        parts.append(f"[{stamp}]\n{c.get('text','')}")
    return "\n\n---\n\n".join(parts)

def _render_table(rows: List[Tuple[str, str]], headers: Tuple[str, str]) -> str:
    lines = [f"| {headers[0]} | {headers[1]} |", "| --- | --- |"]
    for a, b in rows:
        lines.append(f"| {a} | {b} |")
    return "\n".join(lines)

# ----- Deterministic fallbacks -----
_COMPANY_RE = re.compile(r"\b([A-Z][A-Za-z0-9&.,\-\s]+?(?:Inc|LLC|Ltd|PLC|Limited|Company|Corporation|Corp)\.?)\b")
_MONEY_RE   = re.compile(r"\b(?:USD\s*\$|\$|Rs\.|INR\s*)\s?[\d,]+(?:\.\d+)?\b")
_DATE_RE    = re.compile(r"\b(?:\d{1,2}\s+[A-Z][a-z]{2,8}\s+\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{2,4})\b")

def _fallback_companies(chunks: List[Dict], entities: Dict[str, List[str]]) -> List[Tuple[str, str]]:
    # Prefer entities map; if missing, regex from text.
    names = set(entities.get("companies", []))
    if not names:
        for c in chunks:
            for m in _COMPANY_RE.findall(c.get("text","")):
                names.add(m.strip())
    # Try roles from metadata
    pairs = []
    for n in sorted(names):
        role = "Unknown"
        for c in chunks:
            md = c.get("metadata", {}) or {}
            if md.get("lessor") and n.lower() in str(md["lessor"]).lower():
                role = "Lessor"
            if md.get("lessee") and n.lower() in str(md["lessee"]).lower():
                role = "Lessee"
        pairs.append((n, role))
    return pairs

def _fallback_money(chunks: List[Dict]) -> List[str]:
    found = set()
    for c in chunks:
        for m in _MONEY_RE.findall(c.get("text","")):
            found.add(m.strip())
    return sorted(found)

def _fallback_dates(chunks: List[Dict]) -> List[str]:
    found = set()
    for c in chunks:
        for m in _DATE_RE.findall(c.get("text","")):
            found.add(m.strip())
    return sorted(found)

# ----- Main: answer generation with strict JSON + fallbacks -----
class Answerer:
    def __init__(self):
        self.llm = LocalLlama()

    def _json_only(self, prompt: str, max_tokens: int = 300) -> Dict[str, Any] | List[Any]:
        """Call local LLM, force JSON, coerce & return."""
        raw = self.llm.gen(prompt, temp=0.1, max_tokens=max_tokens)
        return coerce_json(raw)

    # Entities (companies + roles)
    def answer_entities(self, question: str, chunks: List[Dict], entities: Dict[str, List[str]]) -> str:
        context = _gather_context(chunks, max_sections=12)
        schema = """Return ONLY valid JSON:
{
  "companies": [
    {"name": "Exact legal name", "role": "Lessor|Lessee|Unknown"}
  ]
}"""
        prompt = f"""You extract companies and roles from leases. Answer strictly from DOCUMENT CONTENT.

QUESTION: {question}

DOCUMENT CONTENT:
{context}

{schema}"""
        rows: List[Tuple[str,str]] = []
        try:
            obj = self._json_only(prompt, max_tokens=350)
            items = obj.get("companies", []) if isinstance(obj, dict) else []
            for it in items:
                n = (it.get("name") or "").strip()
                r = (it.get("role") or "Unknown").strip()
                if n:
                    rows.append((n, r if r in ("Lessor","Lessee","Unknown") else "Unknown"))
        except Exception as e:
            logging.warning(f"entity JSON parse failed: {e}")

        if not rows:
            rows = _fallback_companies(chunks, entities)

        if not rows:
            return "No companies identified in the retrieved sections."
        return _render_table(rows, ("Company Name", "Role"))

    # Financial terms
    def answer_financial(self, question: str, chunks: List[Dict]) -> str:
        context = _gather_context(chunks, max_sections=10)
        schema = """Return ONLY valid JSON:
{
  "amounts": [
    {"value": "string like $12,345", "purpose": "rent|deposit|fee|other", "note": "short context"}
  ]
}"""
        prompt = f"""Extract monetary amounts + purposes from the lease. Strictly from DOCUMENT CONTENT.

QUESTION: {question}

DOCUMENT CONTENT:
{context}

{schema}"""
        rows: List[Tuple[str,str]] = []
        try:
            obj = self._json_only(prompt, max_tokens=350)
            items = obj.get("amounts", []) if isinstance(obj, dict) else []
            for it in items:
                val = (it.get("value") or "").strip()
                pur = (it.get("purpose") or "other").strip()
                note = (it.get("note") or "").strip()
                if val:
                    label = pur if note == "" else f"{pur} – {note}"
                    rows.append((val, label))
        except Exception as e:
            logging.warning(f"financial JSON parse failed: {e}")

        if not rows:
            for m in _fallback_money(chunks):
                rows.append((m, "amount"))

        if not rows:
            return "No financial amounts identified in the retrieved sections."
        return _render_table(rows, ("Amount", "Purpose / Note"))

    # Dates / term
    def answer_dates(self, question: str, chunks: List[Dict]) -> str:
        context = _gather_context(chunks, max_sections=10)
        schema = """Return ONLY valid JSON:
{
  "dates": [
    {"type": "commencement|effective|expiration|renewal|notice|other",
     "value": "YYYY-MM-DD or original text",
     "note": "short context"}
  ]
}"""
        prompt = f"""Extract key dates and their type from the lease. Strictly from DOCUMENT CONTENT.

QUESTION: {question}

DOCUMENT CONTENT:
{context}

{schema}"""
        rows: List[Tuple[str,str]] = []
        try:
            obj = self._json_only(prompt, max_tokens=350)
            items = obj.get("dates", []) if isinstance(obj, dict) else []
            for it in items:
                v = (it.get("value") or "").strip()
                t = (it.get("type") or "other").strip()
                note = (it.get("note") or "").strip()
                if v:
                    label = t if note == "" else f"{t} – {note}"
                    rows.append((v, label))
        except Exception as e:
            logging.warning(f"dates JSON parse failed: {e}")

        if not rows:
            for d in _fallback_dates(chunks):
                rows.append((d, "date"))

        if not rows:
            return "No dates identified in the retrieved sections."
        return _render_table(rows, ("Date", "Type / Note"))

    # Summary (fallback general)
    def answer_summary(self, question: str, chunks: List[Dict]) -> str:
        context = _gather_context(chunks, max_sections=8)
        prompt = f"""Summarize the answer to the question using ONLY DOCUMENT CONTENT.
Be concise, bullet the key facts, and cite headers when obvious. Do NOT include instructions or meta text.

QUESTION: {question}

DOCUMENT CONTENT:
{context}

Return a short paragraph or 3–6 bullets."""
        try:
            text = self.llm.gen(prompt, temp=0.2, max_tokens=300).strip()
            return text
        except Exception:
            # metadata-only fallback
            return "Found relevant sections, but could not summarize with the local model."

    # Router
    def answer(self, question: str, chunks: List[Dict], entities: Dict[str, List[str]]) -> str:
        intent = detect_intent(question)
        if intent == "entities":
            return self.answer_entities(question, chunks, entities)
        if intent == "financial":
            return self.answer_financial(question, chunks)
        if intent == "dates":
            return self.answer_dates(question, chunks)
        return self.answer_summary(question, chunks)
