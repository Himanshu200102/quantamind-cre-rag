from fastapi import APIRouter
from .chunk import router as chunk_router
from .ingest import router as ingest_router
from .rag import router as rag_router
from .download import router as download_router
from .ask import router as ask_router   # NEW

api_router = APIRouter()
api_router.include_router(chunk_router)
api_router.include_router(ingest_router)
api_router.include_router(rag_router)
api_router.include_router(download_router)
api_router.include_router(ask_router)   # NEW
