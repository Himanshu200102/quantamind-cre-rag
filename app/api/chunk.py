from __future__ import annotations
import re
from pathlib import Path
from typing import List

from fastapi import APIRouter, HTTPException
from app.schemas.chunk import ChunkRequest, ChunkResponse, ChunkRecord
from app.services.chunking import find_pdfs, process_pdf
from pydantic import BaseModel

router = APIRouter(prefix="/chunk", tags=["chunk"])


@router.post("", response_model=ChunkResponse)
def chunk_docs(req: ChunkRequest):
    input_dir = Path(req.input_dir)
    if not input_dir.exists():
        raise HTTPException(status_code=400, detail=f"input_dir does not exist: {input_dir}")

    lease_re = None
    if req.lease_id_pattern:
        try:
            lease_re = re.compile(req.lease_id_pattern)
        except re.error as e:
            raise HTTPException(status_code=400, detail=f"Invalid lease_id_pattern: {e}")

    pdfs: List[Path] = find_pdfs(str(input_dir), req.patterns)
    if not pdfs:
        return ChunkResponse(ok=True, processed=0, out_dir=req.out_dir, samples=[])

    if req.limit and req.limit > 0 and req.limit < len(pdfs):
        pdfs = pdfs[: req.limit]

    out_dir = Path(req.out_dir)
    processed = 0
    samples: List[ChunkRecord] = []

    for pdf in pdfs:
        try:
            records = process_pdf(pdf, out_dir, lease_re)
            processed += 1
            for r in records[:3]:
                if len(samples) < 3:
                    samples.append(ChunkRecord(**r))
        except Exception as e:
            continue

    return ChunkResponse(ok=True, processed=processed, out_dir=str(out_dir), samples=samples)
