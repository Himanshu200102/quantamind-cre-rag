# app/schemas/query.py
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

class IngestRequest(BaseModel):
    chunks_dir: Optional[str] = None
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    batch_size: int = 128

class EnhancedIngestRequest(BaseModel):
    chunks_dir: str
    collection: Optional[str] = None
    embedding_model: Optional[str] = None
    batch_size: int = 64

class RetrieveRequest(BaseModel):
    question: str
    k: int = 8
    candidate_pool: int = 40
    where: Optional[Dict[str, Any]] = None
    use_mmr: bool = True

class EnhancedRetrieveRequest(BaseModel):
    question: str
    k: int = 12
    context_length: int = 15000
    debug: bool = False
    where: Optional[Dict[str, Any]] = None
    use_mmr: bool = True

class RetrievedChunk(BaseModel):
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class RetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]

class EnhancedRetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]
    context: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class QueryAnalysisResponse(BaseModel):
    intent: str
    entities_of_interest: List[str]
    question_type: str
    specificity_score: float
    requires_aggregation: bool
    suggested_collections: List[str]

class SmartRetrieveResponse(BaseModel):
    chunks: List[RetrievedChunk]
    context: str
    sources: List[Dict[str, Any]]
    query_analysis: QueryAnalysisResponse
    retrieval_strategy: Dict[str, Any]
    total_chunks_found: int
    context_length: int

class AnswerRequest(RetrieveRequest):
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.1

class AnswerResponse(BaseModel):
    answer: str
    confidence: float
    sources: List[Dict[str, Any]]