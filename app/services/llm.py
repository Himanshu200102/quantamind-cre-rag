from __future__ import annotations
import os, re
from typing import List, Dict, Any, Tuple

def get_enhanced_system_prompt() -> str:
    return """You are an expert lease analyst with deep knowledge of commercial real estate law.

Your responsibilities:
1. ONLY answer based on the provided lease document context
2. If information is not in the context, clearly state "Not found in the provided documents"
3. Always cite sources using the format [doc_id p.X] or [doc_id p.X-Y]
4. For financial terms, include exact amounts, percentages, and dates
5. For legal provisions, quote relevant language when helpful
6. If multiple clauses apply, organize your response with clear sections
7. Identify any potential conflicts or ambiguities in the lease language
8. Use precise legal terminology appropriate for lease analysis

Answer Format:
- Start with a direct answer to the question
- Provide supporting details with citations
- Note any limitations or missing information
- Flag any unusual or potentially problematic provisions
"""

def estimate_answer_confidence(answer: str, prompt: str) -> float:
    confidence = 1.0
    for phrase in ["not found", "unclear", "appears to", "seems to", "may be", "possibly", "potentially", "likely", "probably"]:
        if phrase in answer.lower():
            confidence *= 0.8
    if len(answer.split()) < 20:
        confidence *= 0.9
    citation_count = len(re.findall(r'\[[\w\s\.-]+\s+p\.\d+[-\d]*\]', answer))
    confidence = min(1.0, confidence + 0.1 * citation_count)
    return round(confidence, 2)

def make_enhanced_prompt(question: str, context: str, retrieval_conf: float | None = None) -> str:
    confidence_note = ""
    if retrieval_conf is not None:
        if retrieval_conf < 0.5:
            confidence_note = "\nNOTE: Retrieved context has low confidence scores. Answer with caution."
        elif retrieval_conf > 0.8:
            confidence_note = "\nNOTE: Retrieved context has high confidence scores."
    return f"""System:
{get_enhanced_system_prompt()}

User:
Question: {question}

Context from lease documents:
{context}

{confidence_note}

Instructions:
- Answer the question directly and concisely (≤ 6 bullet points, ≤ 100 words)
- Include specific dollar amounts, percentages, dates, and terms
- Cite sources using [doc_id p.X] format
- If multiple provisions apply, organize them clearly
- Note any conflicts or ambiguities
- If the answer requires information not in the context, state this clearly

Answer:"""

def call_openai(prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.1) -> str:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a precise legal analyst."},
                  {"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()
