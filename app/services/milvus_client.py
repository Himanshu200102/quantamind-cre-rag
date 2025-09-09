# app/services/milvus_client.py
from __future__ import annotations
import os
from typing import Dict, List, Optional, Any

from pymilvus import (
    connections,
    utility,
    FieldSchema, CollectionSchema, DataType,
    Collection, MilvusException
)

MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
MILVUS_URI  = os.getenv("MILVUS_URI", f"http://{MILVUS_HOST}:{MILVUS_PORT}")

EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
# Defaults used only when creating a brand-new index; searching now auto-detects the existing index metric/type
MILVUS_METRIC = os.getenv("MILVUS_METRIC", "IP")            # "COSINE" | "IP" | "L2"
MILVUS_INDEX_TYPE = os.getenv("MILVUS_INDEX_TYPE", "IVF_FLAT")  # "IVF_FLAT", "HNSW", ...
MILVUS_NLIST = int(os.getenv("MILVUS_NLIST", "1024"))
MILVUS_NPROBE = int(os.getenv("MILVUS_NPROBE", "16"))
MILVUS_HNSW_EF = int(os.getenv("MILVUS_HNSW_EF", "64"))     # used when index_type is HNSW


def _connect() -> None:
    if "default" not in connections.list_connections():
        connections.connect(alias="default", uri=MILVUS_URI)


def _create_index(col: Collection) -> None:
    """
    Create an index using env defaults for a fresh collection.
    """
    index_params: Dict[str, Any] = {
        "index_type": MILVUS_INDEX_TYPE,
        "metric_type": MILVUS_METRIC,
        "params": {},
    }
    if MILVUS_INDEX_TYPE.upper().startswith("IVF"):
        index_params["params"] = {"nlist": MILVUS_NLIST}
    elif MILVUS_INDEX_TYPE.upper() == "HNSW":
        index_params["params"] = {"M": 16, "efConstruction": 200}
    # else (FLAT, etc.) no extra params
    col.create_index(field_name="embedding", index_params=index_params)


def ensure_collection(name: str) -> Collection:
    """
    Create collection if missing. Schema:
      - id (VarChar) PK
      - text (VarChar)
      - metadata (VarChar, store JSON-serialized dict)
      - embedding (FloatVector[EMBED_DIM])
    """
    _connect()
    if utility.has_collection(name):
        col = Collection(name)
    else:
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=512, description="chunk id"),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, description="chunk text"),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535, description="json metadata"),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM)
        ]
        schema = CollectionSchema(fields=fields, description=f"RAG chunks for {name}")
        col = Collection(name, schema=schema)

    # Create index if none
    if not col.indexes:
        _create_index(col)

    return col


def ensure_loaded(col: Collection) -> None:
    try:
        col.load()
    except MilvusException:
        col.release()
        col.load()


def insert_rows(col: Collection, rows: List[Dict[str, Any]]) -> int:
    """
    rows: list of dicts with keys: id, text, metadata(str JSON), embedding(list[float])
    """
    if not rows:
        return 0
    data = [
        [r["id"] for r in rows],
        [r.get("text", "") for r in rows],
        [r.get("metadata", "") for r in rows],
        [r["embedding"] for r in rows],
    ]
    mr = col.insert(data)
    col.flush()
    return mr.insert_count


def get_index_info(col: Collection) -> Dict[str, str]:
    """
    Return the active index's metric_type and index_type. Falls back to env defaults if missing.
    """
    metric = MILVUS_METRIC
    index_type = MILVUS_INDEX_TYPE
    try:
        if col.indexes:
            idx = col.indexes[0]
            p = idx.params or {}
            metric = (p.get("metric_type") or metric).upper()
            index_type = (p.get("index_type") or index_type).upper()
    except Exception:
        pass
    return {"metric_type": metric, "index_type": index_type}


def _build_search_params(col: Collection) -> Dict[str, Any]:
    """
    Build search params consistent with the collection's existing index.
    Avoids metric mismatches like expected=COSINE actual=IP.
    """
    info = get_index_info(col)
    metric_type = info["metric_type"]
    index_type = info["index_type"]

    params: Dict[str, Any] = {"metric_type": metric_type, "params": {}}

    if index_type.startswith("IVF"):
        params["params"] = {"nprobe": MILVUS_NPROBE}
    elif index_type == "HNSW":
        params["params"] = {"ef": MILVUS_HNSW_EF}
    else:
        # FLAT or others: no special params needed
        params["params"] = {}

    # Also pass offset/ignore_growing for completeness
    params["offset"] = 0
    params["ignore_growing"] = False
    return params


def search_vectors(
    col: Collection,
    queries: List[List[float]],
    limit: int = 10,
    output_fields: Optional[List[str]] = None,
    expr: Optional[str] = None,
) -> Any:
    """
    Perform ANN search. Embeddings should already be normalized when using COSINE/IP.
    The metric/index_type are auto-detected from the collection to prevent mismatches.
    """
    search_params = _build_search_params(col)

    if output_fields is None:
        output_fields = ["id", "text", "metadata"]

    res = col.search(
        data=queries,
        anns_field="embedding",
        param=search_params,
        limit=limit,
        expr=expr,
        output_fields=output_fields,
    )
    return res
