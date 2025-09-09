# app/services/embeddings.py
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_DB_DIR = os.getenv("CHROMA_DIR", "./out/chroma")
DEFAULT_COLLECTION = os.getenv("COLLECTION", "lease_chunks")
# Smaller = faster, still very strong quality; override via env EMBEDDING_MODEL if you want
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class ChromaService:
    """
    Simple ingestion helper:
      - Reads *.jsonl chunk files produced by your chunker
      - Builds an embedding string that includes header + text + selected metadata
      - Upserts into a Chroma collection
    """

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        collection: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.db_dir = db_dir
        self.collection_name = collection
        self.embedding_model_name = embedding_model

        self.client = chromadb.PersistentClient(
            path=self.db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.embedding_model_name, device=device)
        try:
            dim = self.embedder.get_sentence_embedding_dimension()
        except Exception:
            dim = "?"
        print(f"[ingest] model={self.embedding_model_name} dim={dim} device={device}")
        print(f"[ingest] db_dir={self.db_dir} collection={self.collection_name}")

    # -------- Public API --------
    def ingest_dir(self, *, chunks_dir: str, batch_size: int = 128) -> str:
        """
        Read all *.jsonl files under chunks_dir and upsert to Chroma in batches.
        Each line must be a dict with keys: id, content, metadata.
        """
        root = Path(chunks_dir)
        if not root.exists():
            raise FileNotFoundError(f"chunks_dir not found: {root}")

        files = sorted(list(root.rglob("*.jsonl")))
        if not files:
            raise FileNotFoundError(f"No *.jsonl files under {root}")

        total_upserted = 0
        ids_batch: List[str] = []
        docs_batch: List[str] = []
        metas_batch: List[Dict[str, Any]] = []

        def _flush() -> int:
            nonlocal ids_batch, docs_batch, metas_batch, total_upserted
            if not ids_batch:
                return 0
            embs = self._embed_texts(docs_batch)
            self.collection.upsert(
                ids=ids_batch,
                documents=docs_batch,
                embeddings=embs,
                metadatas=metas_batch,
            )
            count = len(ids_batch)
            total_upserted += count
            ids_batch, docs_batch, metas_batch = [], [], []
            return count

        for fp in files:
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    rid = str(rec.get("id") or "")
                    text = (rec.get("content") or "").strip()
                    meta = rec.get("metadata") or {}

                    if not rid or not text:
                        continue

                    # lightweight context for embedding
                    header = (meta.get("header") or "").strip()
                    lessor = meta.get("lessor") or ""
                    lessee = meta.get("lessee") or ""
                    eff = meta.get("effective_date") or ""
                    term = meta.get("term_duration") or ""
                    clause_type = meta.get("clause_type") or ""

                    meta_snippets = []
                    if lessor:
                        meta_snippets.append(f"LESSOR: {lessor}")
                    if lessee:
                        meta_snippets.append(f"LESSEE: {lessee}")
                    if eff:
                        meta_snippets.append(f"EFFECTIVE DATE: {eff}")
                    if term:
                        meta_snippets.append(f"TERM: {term}")
                    if clause_type:
                        meta_snippets.append(f"CLAUSE: {clause_type}")

                    doc_for_embedding = "\n".join(
                        s for s in [header, text, "\n".join(meta_snippets)] if s
                    )

                    ids_batch.append(rid)
                    docs_batch.append(doc_for_embedding)
                    metas_batch.append(meta)

                    if len(ids_batch) >= batch_size:
                        _flush()

        _flush()

        return (
            f"OK: processed {len(files)} files; upserted {total_upserted} vectors into "
            f"collection '{self.collection_name}' at '{self.db_dir}'"
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()
