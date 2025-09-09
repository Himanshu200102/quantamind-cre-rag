# app/services/milvus_embeddings.py
from __future__ import annotations
import os
import json
from typing import List, Dict, Any

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from app.services.milvus_client import ensure_collection, ensure_loaded, insert_rows

DEFAULT_DB_COLLECTION = os.getenv("MILVUS_COLLECTION_PREFIX", "lease_chunks")
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
SCHEMA_EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))  # collection schema dim


def _get_embedder(model_name: str) -> SentenceTransformer:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    return model


def _norm(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12
    return v / n


class MilvusIngestService:
    """
    Ingest *.jsonl chunks into three Milvus collections:
     - {base}_main
     - {base}_entities
     - {base}_clauses
    Each line in JSONL must have: id, text, metadata (dict)
    """
    def __init__(self, base_collection: str = DEFAULT_DB_COLLECTION, embedding_model: str = DEFAULT_EMBED_MODEL):
        self.base = base_collection
        self.embedder = _get_embedder(embedding_model)
        self.model_name = embedding_model

        # Validate model dim against schema dim to avoid silent shape errors
        model_dim = self.embedder.get_sentence_embedding_dimension()
        if model_dim != SCHEMA_EMBED_DIM:
            raise RuntimeError(
                f"[milvus ingest] EMBED_DIM mismatch: schema={SCHEMA_EMBED_DIM} vs model_dim={model_dim} "
                f"(model={embedding_model}). Set EMBED_DIM={model_dim} before creating collections "
                "or switch to a model matching the existing schema."
            )

        # prepare collections
        self.col_main = ensure_collection(f"{self.base}_main")
        self.col_entities = ensure_collection(f"{self.base}_entities")
        self.col_clauses = ensure_collection(f"{self.base}_clauses")
        for c in (self.col_main, self.col_entities, self.col_clauses):
            ensure_loaded(c)

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self.embedder.encode(texts, normalize_embeddings=True)
        return embs.tolist()

    def _choose_target_collection(self, md: Dict[str, Any]) -> str:
        # prefer explicit routing from metadata if present
        st = str(md.get("search_type", "")).lower()
        if st == "entities":
            return "entities"
        if st == "clauses":
            return "clauses"
        # fallbacks: clause_type hints
        ct = str(md.get("clause_type", "")).lower()
        if ct and ct not in ("", "general"):
            return "clauses"
        # default
        return "main"

    def ingest_dir(self, chunks_dir: str, batch_size: int = 128) -> str:
        """
        Reads all *.jsonl in chunks_dir. Lines like:
          {"id":"...", "text":"...", "metadata": {...}}
        """
        import glob
        import tqdm

        files = sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl")))
        if not files:
            raise RuntimeError(f"No jsonl chunk files found in {chunks_dir}")

        total = 0
        for fpath in files:
            lines: List[Dict[str, Any]] = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        # coerce shapes
                        obj["id"] = str(obj.get("id") or obj.get("chunk_id") or "")
                        obj["text"] = obj.get("text", "") or obj.get("content", "")
                        meta = obj.get("metadata", {}) or {}
                        if not obj["id"]:
                            # stable id if missing
                            obj["id"] = f"{os.path.basename(fpath)}_{len(lines)}"
                        obj["metadata"] = meta
                        lines.append(obj)
                    except Exception:
                        continue

            # split by target collection for batch embedding
            buckets = {"main": [], "entities": [], "clauses": []}
            for it in lines:
                tgt = self._choose_target_collection(it["metadata"])
                buckets[tgt].append(it)

            for name, items in buckets.items():
                if not items:
                    continue
                # embed in batches
                for i in range(0, len(items), batch_size):
                    batch = items[i : i + batch_size]
                    vectors = self._embed_texts([b["text"] for b in batch])
                    rows = []
                    for vec, b in zip(vectors, batch):
                        rows.append({
                            "id": b["id"],
                            "text": b["text"],
                            "metadata": json.dumps(b["metadata"], ensure_ascii=False),
                            "embedding": vec,
                        })

                    if name == "main":
                        insert_rows(self.col_main, rows)
                    elif name == "entities":
                        insert_rows(self.col_entities, rows)
                    else:
                        insert_rows(self.col_clauses, rows)
                    total += len(rows)

        return f"[milvus_ingest] model={self.model_name} dim={SCHEMA_EMBED_DIM} inserted={total} into {self.base}_*"
