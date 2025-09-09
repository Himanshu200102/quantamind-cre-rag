from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkRequest(BaseModel):
    """
    Request to chunk PDFs on disk.
    - input_dir: root folder to search for PDFs
    - patterns: optional glob patterns (relative to project or absolute)
    - out_dir: where to write the .jsonl chunk files
    - lease_id_pattern: optional regex with named groups ('lease' or 'lease_id', optionally 'num'/'property')
    - limit: optional safety cap on how many PDFs to process this call
    """
    input_dir: str = Field(default="./data/Lease_data")
    patterns: Optional[List[str]] = None
    out_dir: str = Field(default="./out/chunks")
    lease_id_pattern: Optional[str] = None
    limit: Optional[int] = Field(default=None)


class ChunkRecord(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any]


class ChunkResponse(BaseModel):
    ok: bool
    processed: int
    out_dir: str
    samples: List[ChunkRecord] = []
