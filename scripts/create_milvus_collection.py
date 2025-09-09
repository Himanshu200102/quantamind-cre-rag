import os
import sys
from pymilvus import (
    connections, utility,
    FieldSchema, CollectionSchema, DataType, Collection
)

MILVUS_URI       = os.getenv("MILVUS_URI", "http://localhost:19530")
COLLECTION       = os.getenv("MILVUS_COLLECTION", "leases_v1")
VECTOR_DIM       = int(os.getenv("VECTOR_DIM", "768"))
MILVUS_METRIC    = os.getenv("MILVUS_METRIC", "COSINE")  # COSINE recommended for normalized embeddings
TEXT_MAX_LENGTH  = int(os.getenv("TEXT_MAX_LENGTH", "4096"))

def ensure_collection():
    # Connect
    connections.connect("default", uri=MILVUS_URI)

    # Skip if exists
    if utility.has_collection(COLLECTION):
        print(f"[OK] Collection already exists: {COLLECTION}")
        return

    print(f"[CREATE] Creating `{COLLECTION}` (dim={VECTOR_DIM}, metric={MILVUS_METRIC})")

    # Define schema (legacy ORM API)
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="autoincrement id"
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=TEXT_MAX_LENGTH,
            description="raw chunk text"
        ),
        FieldSchema(
            name="vec",
            dtype=DataType.FLOAT_VECTOR,
            dim=VECTOR_DIM,
            description="embedding"
        ),
    ]
    schema = CollectionSchema(
        fields=fields,
        description="Lease chunks (text + 768-dim embedding)"
    )

    # Create collection
    col = Collection(name=COLLECTION, schema=schema)

    # Create vector index
    index_params = {
        "index_type": "AUTOINDEX",
        "metric_type": MILVUS_METRIC,
        "params": {}
    }
    col.create_index(field_name="vec", index_params=index_params)

    # Load to memory
    col.load()
    print(f"[READY] `{COLLECTION}` created and loaded.")

if __name__ == "__main__":
    try:
        ensure_collection()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
