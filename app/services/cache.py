# app/services/cache.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional, List
import os
import hashlib
from cachetools import LRUCache, TTLCache

# Toggle via .env ENABLE_CACHE=1
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "1") not in ("0", "false", "False")

# ---- Small, fast in-process caches ----
# embedding cache: query string -> vector (list[float])
EMBED_CACHE = LRUCache(maxsize=int(os.getenv("EMBED_CACHE_SIZE", "4096")))

# retrieval cache: (method, query, where, k) -> list of chunk dicts
RETRIEVAL_CACHE = TTLCache(
    maxsize=int(os.getenv("RETRIEVAL_CACHE_SIZE", "1024")),
    ttl=int(os.getenv("RETRIEVAL_CACHE_TTL", "600")),  # seconds
)

# query analysis cache: question -> JSON dict
ANALYSIS_CACHE = TTLCache(
    maxsize=int(os.getenv("ANALYSIS_CACHE_SIZE", "1024")),
    ttl=int(os.getenv("ANALYSIS_CACHE_TTL", "1800")),
)

# generated search queries cache: (question,intent) -> list[str]
GENQUERIES_CACHE = TTLCache(
    maxsize=int(os.getenv("GENQUERIES_CACHE_SIZE", "1024")),
    ttl=int(os.getenv("GENQUERIES_CACHE_TTL", "1800")),
)

def _stable_key(*parts: Any) -> str:
    m = hashlib.sha1()
    for p in parts:
        m.update(repr(p).encode("utf-8", errors="ignore"))
        m.update(b"\x1f")
    return m.hexdigest()

def get_embed(key: str) -> Optional[List[float]]:
    if not ENABLE_CACHE: return None
    return EMBED_CACHE.get(key)

def put_embed(key: str, vec: List[float]) -> None:
    if not ENABLE_CACHE: return
    EMBED_CACHE[key] = vec

def get_retrieval(method: str, query: str, where: Optional[Dict], k: int) -> Optional[List[Dict[str, Any]]]:
    if not ENABLE_CACHE: return None
    key = _stable_key(method, query, where or {}, k)
    return RETRIEVAL_CACHE.get(key)

def put_retrieval(method: str, query: str, where: Optional[Dict], k: int, value: List[Dict[str, Any]]) -> None:
    if not ENABLE_CACHE: return
    key = _stable_key(method, query, where or {}, k)
    RETRIEVAL_CACHE[key] = value

def get_analysis(question: str) -> Optional[Dict[str, Any]]:
    if not ENABLE_CACHE: return None
    return ANALYSIS_CACHE.get(question)

def put_analysis(question: str, value: Dict[str, Any]) -> None:
    if not ENABLE_CACHE: return
    ANALYSIS_CACHE[question] = value

def get_genqueries(question: str, intent: str) -> Optional[List[str]]:
    if not ENABLE_CACHE: return None
    return GENQUERIES_CACHE.get((question, intent))

def put_genqueries(question: str, intent: str, value: List[str]) -> None:
    if not ENABLE_CACHE: return
    GENQUERIES_CACHE[(question, intent)] = value
