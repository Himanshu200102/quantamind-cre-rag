# app/services/integrated_retrieval.py
from __future__ import annotations
from typing import Dict, List, Optional, Tuple, Any
import logging
from dataclasses import asdict

# Import our enhanced components
# from app.services.query_analyzer import QueryAnalyzer, QueryIntent, QueryAnalysis
# from app.services.enhanced_retriever import EnhancedRetriever

# For demonstration, we'll assume these are available
# In practice, you'd import them from the actual modules

class SmartRetrievalSystem:
    """
    Integrated retrieval system that:
    1. Analyzes query intent
    2. Selects optimal retrieval strategy
    3. Performs multi-stage retrieval
    4. Provides comprehensive results
    """
    
    def __init__(self, collection_prefix: str = "lease_chunks"):
        # self.query_analyzer = QueryAnalyzer()
        # self.retriever = EnhancedRetriever(collection_prefix)
        self.collection_prefix = collection_prefix
        
        # For demo purposes, we'll simulate these
        print(f"[SmartRetrieval] Initialized with collection prefix: {collection_prefix}")
    
    def smart_retrieve(self, question: str, k: int = 12, 
                      context_length: int = 15000, 
                      debug: bool = False) -> Dict[str, Any]:
        """
        Smart retrieval that adapts strategy based on query analysis
        """
        if debug:
            print(f"[SmartRetrieval] Processing question: {question}")
        
        # Step 1: Analyze the query
        # analysis = self.query_analyzer.analyze(question)
        
        # For demo, we'll simulate query analysis
        analysis = self._simulate_query_analysis(question)
        
        if debug:
            print(f"[SmartRetrieval] Query analysis:")
            print(f"  Intent: {analysis['intent']}")
            print(f"  Entities: {analysis['entities_of_interest']}")
            print(f"  Question type: {analysis['question_type']}")
            print(f"  Requires aggregation: {analysis['requires_aggregation']}")
            print(f"  Specificity: {analysis['specificity_score']:.2f}")
        
        # Step 2: Select retrieval strategy based on analysis
        retrieval_strategy = self._select_strategy(analysis)
        
        if debug:
            print(f"[SmartRetrieval] Selected strategy: {retrieval_strategy}")
        
        # Step 3: Execute retrieval
        results = self._execute_retrieval(question, analysis, retrieval_strategy, k)
        
        # Step 4: Post-process and format results
        final_results = self._post_process_results(results, analysis, context_length)
        
        return {
            "chunks": final_results["chunks"],
            "context": final_results["context"],
            "sources": final_results["sources"],
            "query_analysis": analysis,
            "retrieval_strategy": retrieval_strategy,
            "total_chunks_found": len(final_results["chunks"]),
            "context_length": len(final_results["context"]),
        }
    
    def _simulate_query_analysis(self, question: str) -> Dict[str, Any]:
        """Simulate query analysis for demo purposes"""
        question_lower = question.lower()
        
        # Determine intent
        if any(word in question_lower for word in ["how many", "count", "number"]):
            intent = "entity_count"
        elif any(word in question_lower for word in ["who", "what companies", "parties"]):
            intent = "entity_identification"
        elif any(word in question_lower for word in ["rent", "cost", "amount", "payment"]):
            intent = "financial"
        elif any(word in question_lower for word in ["when", "date", "term", "duration"]):
            intent = "temporal"
        else:
            intent = "general"
        
        # Extract entities
        entities = set()
        if any(word in question_lower for word in ["company", "companies"]):
            entities.add("companies")
        if any(word in question_lower for word in ["party", "parties"]):
            entities.add("parties")
        if any(word in question_lower for word in ["rent", "payment"]):
            entities.add("money")
        
        # Question type
        if question_lower.startswith("how many"):
            question_type = "count"
        elif question_lower.startswith("who"):
            question_type = "who"
        elif question_lower.startswith("what"):
            question_type = "what"
        else:
            question_type = "general"
        
        return {
            "intent": intent,
            "entities_of_interest": entities,
            "question_type": question_type,
            "requires_aggregation": question_type == "count",
            "specificity_score": 0.7 if entities else 0.3,
            "suggested_collections": ["entities", "main"] if intent in ["entity_count", "entity_identification"] else ["main"],
        }
    
    def _select_strategy(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Select optimal retrieval strategy based on analysis"""
        strategy = {
            "primary_method": "comprehensive",
            "use_entity_collection": False,
            "use_clause_collection": False,
            "expand_queries": True,
            "mmr_lambda": 0.6,
            "rerank_by_intent": True,
        }
        
        intent = analysis["intent"]
        
        # Entity-focused strategies
        if intent in ["entity_count", "entity_identification"]:
            strategy.update({
                "primary_method": "entity_focused",
                "use_entity_collection": True,
                "mmr_lambda": 0.7,  # More diversity for entity queries
                "expand_queries": True,
            })
        
        # Financial queries
        elif intent == "financial":
            strategy.update({
                "use_clause_collection": True,
                "mmr_lambda": 0.5,  # More relevance for specific info
            })
        
        # High specificity queries
        if analysis["specificity_score"] > 0.7:
            strategy.update({
                "expand_queries": False,  # Don't dilute specific queries
                "mmr_lambda": 0.4,  # Favor relevance over diversity
            })
        
        # Aggregation queries
        if analysis["requires_aggregation"]:
            strategy.update({
                "primary_method": "comprehensive",
                "use_entity_collection": True,
                "mmr_lambda": 0.8,  # Maximum diversity for comprehensive results
            })
        
        return strategy
    
    def _execute_retrieval(self, question: str, analysis: Dict[str, Any], 
                          strategy: Dict[str, Any], k: int) -> Dict[str, Any]:
        """Execute the selected retrieval strategy"""
        
        # For demo purposes, simulate retrieval results
        # In practice, this would use the actual enhanced retriever
        
        # Simulate different retrieval methods
        if strategy["primary_method"] == "entity_focused":
            results = self._simulate_entity_retrieval(question, k)
        else:
            results = self._simulate_comprehensive_retrieval(question, k)
        
        return results
    
    def _simulate_entity_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Simulate entity-focused retrieval"""
        # This would be the actual retriever.retrieve_entities_focused() call
        chunks = []
        
        # Simulate finding company information from the lease
        for i in range(min(k, 6)):  # Simulate fewer but more relevant results for entity queries
            chunk = {
                "id": f"chunk_{i}",
                "text": f"Simulated entity result {i}: Sony Chocolate Industries Ltd. and Augmedix Bangladesh Ltd. are the parties...",
                "score": 0.9 - (i * 0.1),
                "metadata": {
                    "doc_id": "lease_doc",
                    "header": f"Section {i+1}",
                    "clause_type": "parties" if i < 2 else "general",
                    "search_type": "entities" if i % 2 == 0 else "main",
                    "companies_extracted": ["Sony Chocolate Industries Ltd.", "Augmedix Bangladesh Ltd."],
                    "lessor": "Sony Chocolate Industries Ltd.",
                    "lessee": "Augmedix Bangladesh Ltd.",
                }
            }
            chunks.append(chunk)
        
        return {"chunks": chunks}
    
    def _simulate_comprehensive_retrieval(self, question: str, k: int) -> Dict[str, Any]:
        """Simulate comprehensive retrieval"""
        chunks = []
        
        # Simulate comprehensive results
        for i in range(k):
            chunk = {
                "id": f"chunk_{i}",
                "text": f"Simulated comprehensive result {i}: This lease agreement contains various terms and conditions...",
                "score": 0.85 - (i * 0.05),
                "metadata": {
                    "doc_id": "lease_doc",
                    "header": f"Section {i+1}",
                    "clause_type": ["rent", "term", "maintenance", "general"][i % 4],
                    "search_type": "main",
                    "page_start": i + 1,
                }
            }
            chunks.append(chunk)
        
        return {"chunks": chunks}
    
    def _post_process_results(self, results: Dict[str, Any], analysis: Dict[str, Any], 
                             context_length: int) -> Dict[str, Any]:
        """Post-process and format final results"""
        chunks = results["chunks"]
        
        # Build context
        context_parts = []
        sources = []
        total_length = 0
        
        for i, chunk in enumerate(chunks):
            if total_length >= context_length:
                break
            
            # Format chunk for context
            header = chunk["metadata"].get("header", "")
            doc_id = chunk["metadata"].get("doc_id", "document")
            clause_type = chunk["metadata"].get("clause_type", "")
            search_type = chunk["metadata"].get("search_type", "main")
            
            # Build section identifier
            section_parts = [doc_id]
            if clause_type and clause_type != "general":
                section_parts.append(f"({clause_type})")
            if header:
                section_parts.append(header)
            if search_type != "main":
                section_parts.append(f"[{search_type}]")
            
            section_id = " :: ".join(section_parts)
            formatted_chunk = f"[{section_id}]\n{chunk['text']}\n"
            
            if total_length + len(formatted_chunk) <= context_length:
                context_parts.append(formatted_chunk)
                total_length += len(formatted_chunk)
                
                # Add to sources
                sources.append({
                    "id": chunk["id"],
                    "doc_id": doc_id,
                    "header": header,
                    "clause_type": clause_type,
                    "score": round(chunk["score"], 4),
                    "search_type": search_type,
                    "text_preview": chunk["text"][:150] + "..." if len(chunk["text"]) > 150 else chunk["text"],
                })
            else:
                break
        
        context = "\n---\n".join(context_parts)
        
        # If this is an entity count query, add summary information
        if analysis["intent"] == "entity_count" and analysis["entities_of_interest"]:
            context = self._add_entity_summary(context, chunks, analysis)
        
        return {
            "chunks": chunks,
            "context": context,
            "sources": sources,
        }
    
    def _add_entity_summary(self, context: str, chunks: List[Dict[str, Any]], 
                           analysis: Dict[str, Any]) -> str:
        """Add entity summary for count queries"""
        entity_summary = "\n\n=== ENTITY SUMMARY ===\n"
        
        # Extract unique entities from chunks
        companies = set()
        people = set()
        
        for chunk in chunks:
            metadata = chunk["metadata"]
            
            # Extract companies
            if "companies_extracted" in metadata:
                companies.update(metadata["companies_extracted"])
            if "lessor" in metadata and metadata["lessor"]:
                companies.add(metadata["lessor"])
            if "lessee" in metadata and metadata["lessee"]:
                companies.add(metadata["lessee"])
        
        if companies:
            entity_summary += f"COMPANIES FOUND ({len(companies)}):\n"
            for i, company in enumerate(sorted(companies), 1):
                entity_summary += f"  {i}. {company}\n"
        
        if people:
            entity_summary += f"\nPEOPLE MENTIONED ({len(people)}):\n"
            for i, person in enumerate(sorted(people), 1):
                entity_summary += f"  {i}. {person}\n"
        
        return context + entity_summary
    
    # Convenience methods for specific use cases
    def find_all_companies(self, k: int = 20) -> Dict[str, Any]:
        """Specialized method to find all companies in the lease"""
        question = "How many companies are there in this contract? List all companies and parties involved."
        return self.smart_retrieve(question, k=k, debug=True)
    
    def find_financial_terms(self, k: int = 10) -> Dict[str, Any]:
        """Find all financial terms and amounts"""
        question = "What are all the rent amounts, payments, deposits and financial terms mentioned?"
        return self.smart_retrieve(question, k=k)
    
    def find_key_dates(self, k: int = 10) -> Dict[str, Any]:
        """Find all important dates in the lease"""
        question = "What are all the important dates, terms, and deadlines mentioned in the lease?"
        return self.smart_retrieve(question, k=k)
    
    def get_lease_summary(self, k: int = 15) -> Dict[str, Any]:
        """Get comprehensive lease summary"""
        question = "Provide a comprehensive summary of this lease agreement including parties, terms, rent, and key obligations."
        return self.smart_retrieve(question, k=k, context_length=20000)


# Usage examples and testing functions
def test_company_retrieval():
    """Test the company retrieval specifically"""
    system = SmartRetrievalSystem()
    
    test_queries = [
        "How many companies are there in this contract?",
        "Who are all the parties involved in this lease agreement?",
        "List all companies mentioned in the document",
        "What companies are the lessor and lessee?",
    ]
    
    print("=== TESTING COMPANY RETRIEVAL ===")
    
    for query in test_queries:
        print(f"\n--- Query: {query} ---")
        results = system.smart_retrieve(query, debug=True)
        
        print(f"Found {results['total_chunks_found']} chunks")
        print(f"Context length: {results['context_length']} chars")
        print(f"Strategy used: {results['retrieval_strategy']['primary_method']}")
        
        # Show first few sources
        if results['sources']:
            print("\nTop sources:")
            for i, source in enumerate(results['sources'][:3]):
                print(f"  {i+1}. {source['id']} (score: {source['score']}) - {source['text_preview']}")
        
        print("\n" + "="*50)


def demo_usage():
    """Demonstrate the enhanced retrieval system"""
    print("=== ENHANCED RAG RETRIEVAL DEMO ===\n")
    
    # Initialize the system
    system = SmartRetrievalSystem("lease_chunks")
    
    # Test different types of queries
    test_cases = [
        {
            "query": "How many companies are there in this contract?",
            "description": "Entity count query - should find all companies",
        },
        {
            "query": "What is the monthly rent amount?",
            "description": "Financial query - should find rent details",
        },
        {
            "query": "When does the lease start and end?",
            "description": "Temporal query - should find dates",
        },
        {
            "query": "Who is the lessor and lessee?",
            "description": "Entity identification - should find party details",
        },
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{i}. {test_case['description']}")
        print(f"   Query: {test_case['query']}")
        
        results = system.smart_retrieve(test_case['query'], k=8, debug=False)
        
        print(f"   Results: {results['total_chunks_found']} chunks found")
        print(f"   Intent: {results['query_analysis']['intent']}")
        print(f"   Strategy: {results['retrieval_strategy']['primary_method']}")
        print()
    
    # Demonstrate specialized methods
    print("=== SPECIALIZED RETRIEVAL METHODS ===")
    
    print("\n1. Finding all companies:")
    company_results = system.find_all_companies()
    print(f"   Found {company_results['total_chunks_found']} relevant chunks")
    
    print("\n2. Finding financial terms:")
    financial_results = system.find_financial_terms()
    print(f"   Found {financial_results['total_chunks_found']} financial chunks")
    
    print("\n3. Finding key dates:")
    date_results = system.find_key_dates()
    print(f"   Found {date_results['total_chunks_found']} date-related chunks")


# Factory function for easy integration
def get_smart_retrieval_system(collection_prefix: str = "lease_chunks") -> SmartRetrievalSystem:
    """Get configured smart retrieval system"""
    return SmartRetrievalSystem(collection_prefix)


# Main interface function
def smart_retrieve(question: str, k: int = 12, debug: bool = False, **kwargs) -> Dict[str, Any]:
    """
    Main interface for smart retrieval
    
    Args:
        question: User's question
        k: Number of chunks to retrieve
        debug: Enable debug output
        **kwargs: Additional parameters
    
    Returns:
        Dict with chunks, context, sources, and metadata
    """
    system = get_smart_retrieval_system()
    return system.smart_retrieve(question, k=k, debug=debug, **kwargs)


if __name__ == "__main__":
    # Run demo
    demo_usage()
    
    # Run specific company retrieval test
    test_company_retrieval()