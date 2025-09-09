from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel, Field

class DownloadRequest(BaseModel):
    ticker: Optional[str] = Field(None)
    cik: Optional[str] = Field(None)
    forms: List[str] = Field(default_factory=lambda: ["8-K", "10-K", "10-Q"])
    since: Optional[str] = Field(None)
    until: Optional[str] = Field(None)
    limit: int = Field(20, ge=1, le=200)
    out_dir: str = Field("data")

class FilingSaved(BaseModel):
    accession_no: str
    form: str
    filing_date: Optional[str] = None
    report_date: Optional[str] = None
    primary_doc: Optional[str] = None
    exhibits: List[str] = []

class DownloadResponse(BaseModel):
    cik: str
    company: Optional[str] = None
    ticker: Optional[str] = None
    forms: List[str]
    saved_count: int
    saved: List[FilingSaved]
    base_dir: str
    note: Optional[str] = None
