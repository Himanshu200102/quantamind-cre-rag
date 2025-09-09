# app/api/query.py
from fastapi import APIRouter, HTTPException
from app.schemas.query import (
    RetrieveRequest, RetrieveResponse, 
    EnhancedRetrieveRequest, SmartRetrieveResponse,
    QueryAnalysisResponse
)
from app.services.retriever import retrieve_advanced, retrieve_enhanced, smart_retrieve, retrieve_entities

router = APIRouter(prefix="", tags=["query"])

@router.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: RetrieveRequest):
    """
    Legacy retrieval endpoint for backward compatibility
    """
    try:
        where = req.where
        if not where or (isinstance(where, dict) and len(where) == 0):
            where = None

        chunks, context, sources = retrieve_advanced(
            question=req.question,
            k=req.k,
            candidate_pool=req.candidate_pool,
            where=where,
            use_mmr=req.use_mmr,
        )
        return {"chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve-enhanced", response_model=SmartRetrieveResponse)
def retrieve_enhanced_endpoint(req: EnhancedRetrieveRequest):
    """
    Enhanced retrieval with multi-collection search and query analysis
    """
    try:
        where = req.where
        if not where or (isinstance(where, dict) and len(where) == 0):
            where = None

        chunks, context, sources = retrieve_enhanced(
            question=req.question,
            k=req.k,
            where=where,
            use_mmr=req.use_mmr,
        )
        
        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
            "query_analysis": {
                "intent": "general",
                "entities_of_interest": [],
                "question_type": "general",
                "specificity_score": 0.5,
                "requires_aggregation": False,
                "suggested_collections": ["main"]
            },
            "retrieval_strategy": {"primary_method": "enhanced"},
            "total_chunks_found": len(chunks),
            "context_length": len(context)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/smart-retrieve", response_model=SmartRetrieveResponse)
def smart_retrieve_endpoint(req: EnhancedRetrieveRequest):
    """
    Smart retrieval with intent analysis and adaptive strategy selection
    """
    try:
        result = smart_retrieve(
            question=req.question,
            k=req.k,
            context_length=req.context_length,
            debug=req.debug,
            where=req.where,
            use_mmr=req.use_mmr,
        )
        
        # If result is from fallback, format it properly
        if isinstance(result, dict) and "query_analysis" not in result:
            # This is from fallback, add missing fields
            result["query_analysis"] = {
                "intent": "general",
                "entities_of_interest": [],
                "question_type": "general", 
                "specificity_score": 0.5,
                "requires_aggregation": False,
                "suggested_collections": ["main"]
            }
            result["retrieval_strategy"] = {"primary_method": "fallback"}
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/retrieve-entities")
def retrieve_entities_endpoint(req: RetrieveRequest):
    """
    Entity-focused retrieval for finding companies, parties, people, etc.
    """
    try:
        chunks, context, sources = retrieve_entities(
            question=req.question,
            k=req.k
        )
        
        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
            "search_type": "entities",
            "total_chunks_found": len(chunks)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/find-companies")
def find_companies():
    """
    Specialized endpoint to find all companies in the lease documents
    """
    try:
        question = "How many companies are there in this contract? List all companies and parties involved."
        result = smart_retrieve(
            question=question,
            k=20,  # Get more results for comprehensive company search
            debug=True
        )
        
        # Extract company information from sources
        companies_found = set()
        people_found = set()
        
        for source in result.get("sources", []):
            metadata = source.get("metadata", {})
            
            # Extract from metadata
            if "lessor" in metadata and metadata["lessor"]:
                companies_found.add(metadata["lessor"])
            if "lessee" in metadata and metadata["lessee"]:
                companies_found.add(metadata["lessee"])
            if "companies_extracted" in metadata:
                companies_found.update(metadata["companies_extracted"])
        
        return {
            "companies_found": list(companies_found),
            "total_companies": len(companies_found),
            "people_found": list(people_found),
            "total_chunks_searched": result.get("total_chunks_found", 0),
            "sources": result.get("sources", [])[:10],  # Return top 10 sources
            "query_used": question
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/find-financial-terms")
def find_financial_terms():
    """
    Find all financial terms, amounts, and payment details
    """
    try:
        question = "What are all the rent amounts, payments, deposits and financial terms mentioned?"
        result = smart_retrieve(
            question=question,
            k=15,
            debug=True
        )
        
        return {
            "context": result.get("context", ""),
            "sources": result.get("sources", []),
            "total_chunks_found": result.get("total_chunks_found", 0),
            "query_used": question,
            "search_focus": "financial_terms"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test-enhanced")
def test_enhanced():
    """
    Test endpoint to verify enhanced retrieval is working
    """
    try:
        test_questions = [
            "How many companies are there in this contract?",
            "Who are the parties involved?",
            "What is the rent amount?",
            "When does the lease start?"
        ]
        
        results = {}
        
        for question in test_questions:
            try:
                result = smart_retrieve(question=question, k=5, debug=True)
                results[question] = {
                    "chunks_found": result.get("total_chunks_found", 0),
                    "intent": result.get("query_analysis", {}).get("intent", "unknown"),
                    "strategy": result.get("retrieval_strategy", {}).get("primary_method", "unknown"),
                    "success": True
                }
            except Exception as e:
                results[question] = {
                    "error": str(e),
                    "success": False
                }
        
        return {
            "test_results": results,
            "enhanced_system_available": True
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "enhanced_system_available": False
        }