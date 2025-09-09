from __future__ import annotations

from fastapi import APIRouter, HTTPException
from app.schemas.download import DownloadRequest, DownloadResponse, FilingSaved
from app.services.sec_downloader import plan_downloads, execute_downloads

router = APIRouter(prefix="", tags=["download"])

@router.post("/download", response_model=DownloadResponse, summary="Download SEC filings and Exhibit 10.x")
def download_endpoint(req: DownloadRequest):
    try:
        plan = plan_downloads(
            ticker=req.ticker,
            cik=req.cik,
            forms=req.forms,
            since=req.since,
            until=req.until,
            limit=req.limit,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        summary = execute_downloads(plan, req.out_dir)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    saved_models = [FilingSaved(**x) for x in summary["saved"]]
    return DownloadResponse(
        cik=summary["cik"],
        company=summary.get("company"),
        ticker=summary.get("ticker"),
        forms=summary.get("forms", []),
        saved_count=summary.get("saved_count", 0),
        saved=saved_models,
        base_dir=summary.get("base_dir", ""),
        note=summary.get("note"),
    )
