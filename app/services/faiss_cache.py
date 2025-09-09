# app/services/faiss_cache.py
import os
import faiss
import numpy as np
import threading
import time
from typing import List, Dict, Any, Tuple

CACHE_DIM = int(os.getenv("EMBED_DIM", "768"))
CACHE_INDEX_TYPE = os.getenv("FAISS_INDEX", "Flat") 
CACHE_PATH = os.getenv("FAISS_CACHE_PATH", "faiss_cache.index")

class CacheMetrics:
    """Thread-safe metrics collector for cache performance."""
    def __init__(self):
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        self.faiss_latency = []
        self.milvus_latency = []

    def record_hit(self, latency: float):
        with self._lock:
            self.hits += 1
            self.faiss_latency.append(latency)

    def record_miss(self, latency: float):
        with self._lock:
            self.misses += 1
            self.milvus_latency.append(latency)

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cache_hits": self.hits,
                "cache_misses": self.misses,
                "hit_rate": self.hits / max(1, self.hits + self.misses),
                "avg_faiss_latency_ms": np.mean(self.faiss_latency) * 1000 if self.faiss_latency else 0,
                "avg_milvus_latency_ms": np.mean(self.milvus_latency) * 1000 if self.milvus_latency else 0,
            }


class FAISSCache:
    """
    FAISS cache for semantic search in front of Milvus.
    Stores embeddings + ids in memory for fast recall.
    """

    def __init__(self, dim: int = CACHE_DIM, index_type: str = CACHE_INDEX_TYPE):
        self.dim = dim
        self.index = self._create_index(index_type, dim)
        self.ids: List[str] = []
        self.metrics = CacheMetrics()

        # Try loading existing cache
        if os.path.exists(CACHE_PATH):
            self.load(CACHE_PATH)

    def _create_index(self, index_type: str, dim: int):
        if index_type.lower() == "flat":
            return faiss.IndexFlatIP(dim)
        elif index_type.lower() == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            return faiss.IndexIVFFlat(quantizer, dim, 256, faiss.METRIC_INNER_PRODUCT)
        elif index_type.lower() == "hnsw":
            return faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unsupported FAISS index type: {index_type}")

    def add(self, embeddings: List[List[float]], ids: List[str]):
        """Add embeddings + ids to the cache.

        Args:
            embeddings: List of embeddings (outer list) with inner lists of floats.
            ids: List of ids corresponding to the embeddings.
        """
        arr = np.array(embeddings).astype("float32")
        if not self.index.is_trained:
            self.index.train(arr)
        self.index.add(arr)
        self.ids.extend(ids)

    def search(self, query: List[List[float]], k: int = 5) -> List[Dict[str, Any]]:
        """
        Perform a search with the given query and return k results.
        
        Args:
            query: List of embeddings (outer list) with inner lists of floats.
            k: Number of results to return.

        Returns:
            List of dictionaries, each containing
                id: str, id of the document
                score: float, the similarity score of the document to the query
        """
        q = np.array(query).astype("float32")
        start = time.time()
        D, I = self.index.search(q, k)
        latency = time.time() - start

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.ids):
                continue
            results.append({
                "id": self.ids[idx],
                "score": float(dist),
            })
        if results:
            self.metrics.record_hit(latency)
        return results

    def save(self, path: str = CACHE_PATH):
        faiss.write_index(self.index, path)

    def load(self, path: str = CACHE_PATH):
        self.index = faiss.read_index(path)

    def get_metrics(self):
        return self.metrics.summary()
