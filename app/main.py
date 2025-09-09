# app/main.py
try:
    import pysqlite3 as sqlite3  # noqa: F401
    import sys
    sys.modules["sqlite3"] = sqlite3
except Exception:
    pass

import os
import time
import logging
from fastapi import FastAPI, Request
from dotenv import load_dotenv

from app.api import api_router

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

app = FastAPI(title="Lease RAG API", version="1.0.0")

@app.middleware("http")
async def time_requests(request: Request, call_next):
    t0 = time.perf_counter()
    resp = await call_next(request)
    dt = time.perf_counter() - t0
    logging.getLogger(__name__).info(f"[TIME] {request.method} {request.url.path} ... {dt:.2f}s")
    return resp

@app.get("/")
def root():
    return {"ok": True, "service": "Lease RAG API"}

app.include_router(api_router)
