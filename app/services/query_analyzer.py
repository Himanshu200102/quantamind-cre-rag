# app/services/query_analyzer.py
from __future__ import annotations
from typing import Dict, List, Set, Optional, Tuple, Any
import re
from dataclasses import dataclass
from enum import Enum

class QueryIntent(Enum):
    ENTITY_COUNT = "entity_count"  # How many companies, parties, etc.
    ENTITY_IDENTIFICATION = "entity_identification"  # Who are the parties, what companies
    FINANCIAL = "financial"  # Rent, costs, amounts
    TEMPORAL = "temporal"  # Dates, terms, duration
    SPATIAL = "spatial"  # Areas, locations, premises
    LEGAL_TERMS = "legal_terms"  # Clauses, obligations, rights
    PROCESS = "process"  # How something works, procedures
    COMPARISON = "comparison"  # Differences, similarities
    GENERAL = "general"

@dataclass
class QueryAnalysis:
    intent: QueryIntent
    entities_of_interest: Set[str]
    keywords: Set[str]
    question_type: str  # what, who, when, where, how, why, count
    specificity_score: float  # 0-1, how specific the query is
    requires_aggregation: bool
    suggested_collections: List[str]
    expanded_queries: List[str]

class QueryAnalyzer:
    """
    Analyzes user queries to determine intent and optimize retrieval strategy
    """
    
    def __init__(self):
        self.intent_patterns = {
            QueryIntent.ENTITY_COUNT: [
                r'\b(?:how many|count|number of|total)\s+(?:companies|parties|entities|lessors|lessees|firms|organizations)\b',
                r'\b(?:list all|all the)\s+(?:companies|parties|entities)\b',
                r'\b(?:companies|parties|entities)\s+(?:are there|involved|mentioned)\b',
            ],
            QueryIntent.ENTITY_IDENTIFICATION: [
                r'\b(?:who|what|which)\s+(?:companies|parties|entities|lessors|lessees)\b',
                r'\b(?:names?|identit(?:y|ies))\s+of\s+(?:companies|parties|entities)\b',
                r'\b(?:lessor|lessee)\s+(?:is|are|name)\b',
            ],
            QueryIntent.FINANCIAL: [
                r'\b(?:rent|payment|cost|amount|money|price|fee|charge|deposit|security)\b',
                r'\b(?:how much|what.*cost|financial|monetary)\b',
                r'\b(?:taka|tk|dollar|usd|bdt)\b',
            ],
            QueryIntent.TEMPORAL: [
                r'\b(?:when|date|time|duration|term|period|expir|start|end|commence)\b',
                r'\b(?:how long|length of time|timeline)\b',
                r'\b(?:lease term|tenure|effective date)\b',
            ],
            QueryIntent.SPATIAL: [
                r'\b(?:where|location|address|premises|space|area|floor|suite)\b',
                r'\b(?:sq\.?\s*ft|square feet|size|dimensions)\b',
                r'\b(?:demised premises|property)\b',
            ],
            QueryIntent.LEGAL_TERMS: [
                r'\b(?:clause|obligation|right|responsibility|covenant|breach|termination)\b',
                r'\b(?:legal|law|contract|agreement|terms|conditions)\b',
                r'\b(?:maintenance|insurance|assignment|sublease)\b',
            ],
            QueryIntent.PROCESS: [
                r'\b(?:how|process|procedure|step|method|way)\b',
                r'\b(?:handover|delivery|installation|setup)\b',
            ],
            QueryIntent.COMPARISON: [
                r'\b(?:difference|similar|compare|versus|vs|between)\b',
                r'\b(?:same|different|alike|unlike)\b',
            ],
        }
        
        self.question_patterns = {
            "what": [r'\bwhat\b', r'\bwhich\b'],
            "who": [r'\bwho\b'],
            "when": [r'\bwhen\b'],
            "where": [r'\bwhere\b'],
            "how": [r'\bhow\b'],
            "why": [r'\bwhy\b'],
            "count": [r'\bhow many\b', r'\bcount\b', r'\bnumber\b', r'\btotal\b'],
        }
        
        self.entity_patterns = {
            "companies": [r'\bcompan(?:y|ies)\b', r'\bcorporation\b', r'\bfirm\b', r'\borganization\b', r'\bbusiness\b'],
            "parties": [r'\bpart(?:y|ies)\b', r'\blessor\b', r'\blessee\b'],
            "people": [r'\bpeople\b', r'\bperson\b', r'\bindividual\b', r'\bdirector\b', r'\bmanager\b'],
            "money": [r'\bmoney\b', r'\brent\b', r'\bpayment\b', r'\bcost\b', r'\bamount\b', r'\bprice\b'],
            "time": [r'\bdate\b', r'\btime\b', r'\bperiod\b', r'\bterm\b', r'\bduration\b'],
            "space": [r'\bspace\b', r'\barea\b', r'\bpremises\b', r'\bfloor\b', r'\broom\b'],
        }
        
        self.aggregation_indicators = [
            r'\b(?:how many|count|total|sum|all|list|enumerate)\b',
            r'\b(?:each|every|per)\b',
        ]
        
        # Collection preferences based on query type
        self.collection_preferences = {
            QueryIntent.ENTITY_COUNT: ["entities", "main"],
            QueryIntent.ENTITY_IDENTIFICATION: ["entities", "main"],
            QueryIntent.FINANCIAL: ["clauses", "main"],
            QueryIntent.TEMPORAL: ["clauses", "main"],
            QueryIntent.SPATIAL: ["main", "clauses"],
            QueryIntent.LEGAL_TERMS: ["clauses", "main"],
            QueryIntent.PROCESS: ["main", "clauses"],
            QueryIntent.COMPARISON: ["main"],
            QueryIntent.GENERAL: ["main", "clauses", "entities"],
        }
    
    def analyze(self, query: str) -> QueryAnalysis:
        """
        Comprehensive query analysis
        """
        query_lower = query.lower().strip()
        
        # Determine intent
        intent = self._determine_intent(query_lower)
        
        # Extract entities of interest
        entities_of_interest = self._extract_entities_of_interest(query_lower)
        
        # Extract keywords
        keywords = self._extract_keywords(query_lower)
        
        # Determine question type
        question_type = self._determine_question_type(query_lower)
        
        # Calculate specificity score
        specificity_score = self._calculate_specificity(query_lower, entities_of_interest, keywords)
        
        # Check if aggregation is required
        requires_aggregation = self._requires_aggregation(query_lower)
        
        # Suggest collections
        suggested_collections = self.collection_preferences.get(intent, ["main"])
        
        # Generate expanded queries
        expanded_queries = self._generate_expanded_queries(query, intent, entities_of_interest)
        
        return QueryAnalysis(
            intent=intent,
            entities_of_interest=entities_of_interest,
            keywords=keywords,
            question_type=question_type,
            specificity_score=specificity_score,
            requires_aggregation=requires_aggregation,
            suggested_collections=suggested_collections,
            expanded_queries=expanded_queries,
        )
    
    def _determine_intent(self, query: str) -> QueryIntent:
        """Determine the primary intent of the query"""
        intent_scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, query, re.IGNORECASE))
                score += matches
            if score > 0:
                intent_scores[intent] = score
        
        if not intent_scores:
            return QueryIntent.GENERAL
        
        return max(intent_scores, key=intent_scores.get)
    
    def _extract_entities_of_interest(self, query: str) -> Set[str]:
        """Extract what entities the user is asking about"""
        entities = set()
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    entities.add(entity_type)
        
        return entities
    
    def _extract_keywords(self, query: str) -> Set[str]:
        """Extract important keywords from the query"""
        # Remove common stop words but keep important lease-related terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being'}
        
        # Important lease terms to always keep
        important_terms = {
            'lessor', 'lessee', 'rent', 'lease', 'term', 'premises', 'agreement', 
            'contract', 'payment', 'deposit', 'security', 'maintenance', 'clause',
            'company', 'companies', 'party', 'parties', 'amount', 'date', 'time'
        }
        
        words = re.findall(r'\b\w+\b', query)
        keywords = set()
        
        for word in words:
            word_lower = word.lower()
            if len(word) > 2 and (word_lower not in stop_words or word_lower in important_terms):
                keywords.add(word_lower)
        
        return keywords
    
    def _determine_question_type(self, query: str) -> str:
        """Determine the type of question being asked"""
        for q_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    return q_type
        return "general"
    
    def _calculate_specificity(self, query: str, entities: Set[str], keywords: Set[str]) -> float:
        """Calculate how specific the query is (0.0 to 1.0)"""
        specificity = 0.0
        
        # Base score from query length
        word_count = len(query.split())
        specificity += min(word_count / 20.0, 0.3)  # Up to 0.3 for length
        
        # Boost for specific entities
        specificity += len(entities) * 0.1  # Up to additional points for entities
        
        # Boost for specific keywords
        specific_keywords = {'date', 'amount', 'name', 'address', 'number', 'count', 'specific'}
        specific_count = sum(1 for kw in keywords if kw in specific_keywords)
        specificity += specific_count * 0.1
        
        # Boost for exact values or names
        if re.search(r'\b(?:\d+|specific|exact|particular)\b', query, re.IGNORECASE):
            specificity += 0.2
        
        return min(specificity, 1.0)
    
    def _requires_aggregation(self, query: str) -> bool:
        """Check if the query requires aggregation of multiple results"""
        for pattern in self.aggregation_indicators:
            if re.search(pattern, query, re.IGNORECASE):
                return True
        return False
    
    def _generate_expanded_queries(self, original_query: str, intent: QueryIntent, entities: Set[str]) -> List[str]:
        """Generate expanded versions of the query"""
        expanded = [original_query]
        query_lower = original_query.lower()
        
        # Intent-specific expansions
        if intent == QueryIntent.ENTITY_COUNT:
            expanded.extend([
                f"list all companies in {original_query}",
                f"parties involved in {original_query}",
                f"entities mentioned in {original_query}",
                "lessor and lessee details",
                "company names and organizations",
            ])
        
        elif intent == QueryIntent.ENTITY_IDENTIFICATION:
            expanded.extend([
                f"names of companies: {original_query}",
                f"party identification: {original_query}",
                "lessor details",
                "lessee details",
                "contracting parties",
            ])
        
        elif intent == QueryIntent.FINANCIAL:
            expanded.extend([
                f"payment details: {original_query}",
                f"financial terms: {original_query}",
                "rent amount",
                "security deposit",
                "monetary obligations",
            ])
        
        elif intent == QueryIntent.TEMPORAL:
            expanded.extend([
                f"timeline: {original_query}",
                f"dates mentioned: {original_query}",
                "lease term duration",
                "effective dates",
                "expiration information",
            ])
        
        # Entity-specific expansions
        if "companies" in entities:
            expanded.extend([
                "corporate entities involved",
                "business organizations mentioned",
                "company details and information",
            ])
        
        if "parties" in entities:
            expanded.extend([
                "contracting parties details",
                "lessor and lessee information",
                "party obligations and rights",
            ])
        
        # Add question variations
        if not original_query.endswith("?"):
            expanded.append(f"{original_query}?")
        
        # Remove duplicates and return
        return list(set(expanded))


# Convenience function
def analyze_query(query: str) -> QueryAnalysis:
    """Quick function to analyze a query"""
    analyzer = QueryAnalyzer()
    return analyzer.analyze(query)