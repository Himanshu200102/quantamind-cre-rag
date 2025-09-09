# app/services/enhanced_embeddings.py
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb
from chromadb.config import Settings
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_DB_DIR = os.getenv("CHROMA_DIR", "./out/chroma")
DEFAULT_COLLECTION = os.getenv("COLLECTION", "lease_chunks")
# Smaller = faster, very competitive quality
DEFAULT_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# Universal CRE clause taxonomies by document type
CRE_CLAUSE_TAXONOMIES = {
    "lease": {
        "rent": ["rent", "rental", "payment", "monthly", "advance rent", "base rent"],
        "term": ["tenure", "lease term", "duration", "expiration", "commencement", "renewal"],
        "premises": ["demised premises", "leased premises", "space", "area", "floor", "suite", "property"],
        "security": ["security deposit", "advance", "refund", "guarantee"],
        "maintenance": ["maintenance", "service charge", "utilities", "cleaning", "repairs"],
        "termination": ["termination", "terminate", "breach", "default", "notice"],
        "parties": ["lessor", "lessee", "landlord", "tenant", "party", "parties"],
        "assignment": ["assignment", "sublet", "sublease", "transfer"],
        "insurance": ["insurance", "liability", "coverage", "policy"],
        "taxes": ["taxes", "property tax", "assessment", "levy"],
        "use": ["permitted use", "restrictions", "prohibited", "compliance"],
        "improvements": ["improvements", "alterations", "modifications", "capital"],
    },
    "purchase": {
        "price": ["purchase price", "sale price", "consideration", "payment"],
        "closing": ["closing", "settlement", "possession", "transfer", "deed"],
        "financing": ["financing", "mortgage", "loan", "down payment"],
        "contingencies": ["contingency", "subject to", "condition", "approval"],
        "inspection": ["inspection", "due diligence", "condition", "defects"],
        "title": ["title", "ownership", "encumbrance", "lien", "clear title"],
        "warranties": ["warranty", "representation", "guarantee", "covenant"],
        "default": ["default", "breach", "remedy", "termination"],
        "environmental": ["environmental", "hazardous", "contamination", "compliance"],
        "zoning": ["zoning", "permits", "approvals", "compliance"],
    },
    "management": {
        "services": ["management services", "property management", "operations"],
        "fees": ["management fee", "compensation", "commission"],
        "responsibilities": ["duties", "obligations", "responsibilities", "scope"],
        "reporting": ["reports", "accounting", "financial statements"],
        "maintenance": ["maintenance", "repairs", "improvements", "capital"],
        "leasing": ["leasing", "marketing", "tenant relations"],
        "termination": ["termination", "notice", "cause"],
        "performance": ["performance standards", "benchmarks", "metrics"],
    },
    "construction": {
        "scope": ["scope of work", "construction", "materials", "specifications"],
        "schedule": ["completion date", "milestone", "schedule", "timeline"],
        "payment": ["progress payments", "retainage", "final payment"],
        "changes": ["change order", "modification", "additional work"],
        "warranties": ["warranty", "defects", "punch list", "completion"],
        "delays": ["delay", "extension", "force majeure", "weather"],
        "default": ["default", "breach", "remedies", "termination"],
        "safety": ["safety", "OSHA", "compliance", "insurance"],
        "permits": ["permits", "approvals", "inspections", "certificates"],
    },
    "financing": {
        "loan_terms": ["principal", "interest rate", "term", "amortization"],
        "security": ["collateral", "security", "mortgage", "pledge"],
        "covenants": ["financial covenants", "maintenance", "restrictions"],
        "default": ["default", "acceleration", "remedies", "foreclosure"],
        "payments": ["payment schedule", "monthly payment", "prepayment"],
        "conditions": ["conditions precedent", "closing conditions"],
        "guarantees": ["guarantee", "guarantor", "recourse"],
    },
    "development": {
        "project": ["development project", "construction", "phases", "completion"],
        "approvals": ["permits", "approvals", "zoning", "environmental"],
        "financing": ["development financing", "construction loan", "funding"],
        "timeline": ["development schedule", "milestones", "completion"],
        "costs": ["development costs", "budget", "cost overruns"],
        "revenue": ["revenue sharing", "profits", "sales proceeds"],
    },
}

class EnhancedChromaService:
    """
    Universal CRE document embedding service with:
    - Dynamic document type detection (metadata provided by your chunker)
    - Entity extraction cues written to metadata
    - Clause identification
    - Multi-collection strategy for optimal retrieval
    """

    def __init__(
        self,
        db_dir: str = DEFAULT_DB_DIR,
        collection: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBED_MODEL,
    ) -> None:
        self.db_dir = db_dir
        self.collection_name = collection
        self.embedding_model_name = embedding_model

        self.client = chromadb.PersistentClient(
            path=self.db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Create multiple collections for different embedding strategies
        self.main_collection = self._get_or_create_collection(f"{self.collection_name}_main")
        self.entity_collection = self._get_or_create_collection(f"{self.collection_name}_entities")
        self.clause_collection = self._get_or_create_collection(f"{self.collection_name}_clauses")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embedder = SentenceTransformer(self.embedding_model_name, device=device)
        try:
            dim = self.embedder.get_sentence_embedding_dimension()
        except Exception:
            dim = "?"
        print(f"[enhanced_ingest] model={self.embedding_model_name} dim={dim} device={device}")
        print(f"[enhanced_ingest] db_dir={self.db_dir} collections={self.collection_name}_*")

    def _get_or_create_collection(self, name: str):
        return self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # --- (entity/metadata helpers unchanged for brevity) ---

    def _extract_universal_entities(self, text: str, doc_type: str) -> Dict[str, Set[str]]:
        entities = {
            "companies": set(),
            "people": set(),
            "addresses": set(),
            "amounts": set(),
            "dates": set(),
            "areas": set(),
            "legal_refs": set(),
        }
        company_patterns = [
            r'\b([A-Z][A-Za-z\s&]+(?:Ltd\.|Limited|Inc\.|Corporation|Corp\.|LLC|L\.L\.C\.|Industries|Company|Co\.|Group|Holdings|Pvt\.|Private Limited|LLP|Partnership|Trust|REIT|Bank|Credit Union))\b',
            r'\b([A-Z][A-Za-z\s&]+Bangladesh\s+(?:Ltd\.|Limited|Inc\.|Corporation))\b',
            r'\b([A-Z][A-Z\s]+(?:COMPANY|CORPORATION|INDUSTRIES|TRUST|FUND|BANK))\b',
        ]
        for pattern in company_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                m = match[0] if isinstance(match, tuple) else match
                name = (m or "").strip()
                if len(name) > 5 and not name.lower().startswith(("the ", "a ", "an ")):
                    entities["companies"].add(name)

        people_patterns = [
            r'(?:Mr\.|Mrs\.|Ms\.|Dr\.|Managing Director|Country Director|Director|President|CEO|CFO|Vice President|General Manager|Project Manager)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'represented by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
            r'(?:signed by|executed by|witnessed by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        ]
        for pattern in people_patterns:
            for match in re.findall(pattern, text):
                if len(match.split()) >= 2:
                    entities["people"].add(match.strip())

        address_patterns = [
            r'\b\d+[A-Za-z]?[\/,\s]+[A-Za-z\s,\-]+(?:Road|Street|Avenue|Lane|Drive|Place|Suite|Floor|Building|Centre|Complex)[A-Za-z\s,\-\d]*',
            r'Suite\s*#?\s*[A-Za-z0-9\-]+[,\s]+[A-Za-z\s,\-\d]+',
            r'(?:Floor|Level)\s*\d+[^,\n]*',
        ]
        for pattern in address_patterns:
            entities["addresses"].update([m.strip() for m in re.findall(pattern, text, re.IGNORECASE) if len(m.strip()) > 10])

        amount_patterns = [
            r'(?:\$|USD\s*|Tk\.?\s*|BDT\s*|€|EUR\s*|£|GBP\s*)[\d,]+\.?\d*(?:\s*(?:\([^)]+\))?)',
            r'\b[\d,]+\.?\d*\s*(?:dollars?|USD|taka|BDT|pounds?|GBP|euros?|EUR)',
            r'(?:million|billion|thousand|lacs?)\s+(?:dollars?|taka|BDT)',
        ]
        for pattern in amount_patterns:
            entities["amounts"].update([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])

        date_patterns = [
            r'\b\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}\b',
            r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b',
        ]
        for pattern in date_patterns:
            entities["dates"].update([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])

        area_patterns = [
            r'\b[\d,]+\.?\d*\s*(?:sq\.?\s*ft\.?|square\s+feet|sqft|acres?|hectares?)',
            r'\b[\d,]+\.?\d*\s*(?:KW|kilowatt|MW|megawatt)',
        ]
        for pattern in area_patterns:
            entities["areas"].update([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])

        legal_patterns = [
            r'\b(?:Section|§)\s*\d+(?:\.\d+)*\b',
            r'\b(?:Article|Art\.)\s*\d+(?:\.\d+)*\b',
            r'\b(?:Clause|Para|Paragraph)\s*\d+(?:\.\d+)*\b',
        ]
        for pattern in legal_patterns:
            entities["legal_refs"].update([m.strip() for m in re.findall(pattern, text, re.IGNORECASE)])

        return {k: v for k, v in entities.items() if v}

    def _identify_clause_type(self, text: str, header: str, doc_type: str) -> str:
        combined = (header + " " + text).lower()
        clause_types = CRE_CLAUSE_TAXONOMIES.get(doc_type, CRE_CLAUSE_TAXONOMIES["lease"])
        scores = {}
        for clause_type, keywords in clause_types.items():
            score = sum(combined.count(keyword) for keyword in keywords)
            if score > 0:
                scores[clause_type] = score
        return max(scores, key=scores.get) if scores else "general"

    def _create_rich_embedding_text(self, text: str, metadata: Dict[str, Any], entities: Dict[str, Set[str]]) -> str:
        header = metadata.get("header", "").strip()
        doc_type = metadata.get("doc_type", "")
        clause_type = metadata.get("clause_type", "")

        context_parts = []
        if doc_type:
            context_parts.append(f"DOCUMENT TYPE: {doc_type}")
        if clause_type:
            context_parts.append(f"CLAUSE TYPE: {clause_type}")
        if header:
            context_parts.append(f"SECTION: {header}")
        for entity_type, entity_set in entities.items():
            if entity_set and len(entity_set) <= 5:
                entities_str = ", ".join(list(entity_set)[:5])
                context_parts.append(f"{entity_type.upper()}: {entities_str}")

        property_address = metadata.get("property_address")
        area_sqft = metadata.get("area_sqft")
        effective_date = metadata.get("effective_date")
        if property_address:
            context_parts.append(f"PROPERTY: {property_address}")
        if area_sqft:
            context_parts.append(f"AREA: {area_sqft} sq ft")
        if effective_date:
            context_parts.append(f"DATE: {effective_date}")

        context_str = "\n".join(context_parts)
        return f"{context_str}\n\nCONTENT:\n{text}"

    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = None
            elif isinstance(value, (str, int, float, bool)):
                sanitized[key] = value
            elif isinstance(value, list):
                sanitized[key] = "; ".join([str(v) for v in value]) if value else ""
            elif isinstance(value, (set, tuple)):
                sanitized[key] = "; ".join([str(v) for v in value]) if value else ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def ingest_dir(self, *, chunks_dir: str, batch_size: int = 64) -> str:
        root = Path(chunks_dir)
        if not root.exists():
            raise FileNotFoundError(f"chunks_dir not found: {root}")

        files = sorted(list(root.rglob("*.jsonl")))
        if not files:
            raise FileNotFoundError(f"No *.jsonl files under {root}")

        total_upserted = 0
        main_batch = {"ids": [], "docs": [], "metas": []}
        entity_batch = {"ids": [], "docs": [], "metas": []}
        clause_batch = {"ids": [], "docs": [], "metas": []}

        def _flush_all():
            nonlocal total_upserted
            if main_batch["ids"]:
                embs = self._embed_texts(main_batch["docs"])
                self.main_collection.upsert(
                    ids=main_batch["ids"],
                    documents=main_batch["docs"],
                    embeddings=embs,
                    metadatas=main_batch["metas"],
                )
                total_upserted += len(main_batch["ids"])
                main_batch["ids"], main_batch["docs"], main_batch["metas"] = [], [], []
            if entity_batch["ids"]:
                embs = self._embed_texts(entity_batch["docs"])
                self.entity_collection.upsert(
                    ids=entity_batch["ids"],
                    documents=entity_batch["docs"],
                    embeddings=embs,
                    metadatas=entity_batch["metas"],
                )
                entity_batch["ids"], entity_batch["docs"], entity_batch["metas"] = [], [], []
            if clause_batch["ids"]:
                embs = self._embed_texts(clause_batch["docs"])
                self.clause_collection.upsert(
                    ids=clause_batch["ids"],
                    documents=clause_batch["docs"],
                    embeddings=embs,
                    metadatas=clause_batch["metas"],
                )
                clause_batch["ids"], clause_batch["docs"], clause_batch["metas"] = [], [], []

        for fp in tqdm(files, desc="Processing files"):
            with fp.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    rid = str(rec.get("id") or "")
                    text = (rec.get("content") or "").strip()
                    meta = rec.get("metadata") or {}
                    if not rid or not text:
                        continue

                    doc_type = meta.get("doc_type", "lease")
                    entities = self._extract_universal_entities(text, doc_type)
                    clause_type = self._identify_clause_type(text, meta.get("header", ""), doc_type)
                    meta["clause_type"] = clause_type

                    for entity_type, entity_set in entities.items():
                        if entity_set:
                            meta[f"{entity_type}_extracted"] = "; ".join(list(entity_set))

                    rich_text = self._create_rich_embedding_text(text, meta, entities)
                    sanitized_meta = self._sanitize_metadata(meta.copy())

                    # main
                    main_batch["ids"].append(rid)
                    main_batch["docs"].append(rich_text)
                    main_batch["metas"].append(sanitized_meta)

                    # entities
                    if entities:
                        entity_text = "ENTITIES FOUND:\n"
                        for entity_type, entity_set in entities.items():
                            if entity_set:
                                entity_text += f"{entity_type.upper()}: {', '.join(entity_set)}\n"
                        entity_text += f"\nCONTEXT:\n{text}"
                        entity_meta = sanitized_meta.copy()
                        entity_meta["search_type"] = "entities"
                        entity_batch["ids"].append(f"{rid}_entities")
                        entity_batch["docs"].append(entity_text)
                        entity_batch["metas"].append(entity_meta)

                    # clause
                    clause_text = f"CLAUSE TYPE: {clause_type}\nDOCUMENT TYPE: {doc_type}\n{meta.get('header', '')}\n\n{text}"
                    clause_meta = sanitized_meta.copy()
                    clause_meta["search_type"] = "clause"
                    clause_batch["ids"].append(f"{rid}_clause")
                    clause_batch["docs"].append(clause_text)
                    clause_batch["metas"].append(clause_meta)

                    if (len(main_batch["ids"]) >= batch_size or
                        len(entity_batch["ids"]) >= batch_size or
                        len(clause_batch["ids"]) >= batch_size):
                        _flush_all()

        _flush_all()

        return (
            f"Universal CRE document ingestion complete: processed {len(files)} files, "
            f"upserted {total_upserted} vectors across 3 collections at '{self.db_dir}'"
        )

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def get_document_stats(self) -> Dict[str, Any]:
        try:
            main_count = self.main_collection.count()
            entity_count = self.entity_collection.count()
            clause_count = self.clause_collection.count()
            try:
                sample = self.main_collection.get(
                    limit=min(100, main_count),
                    include=["metadatas"]
                )
                doc_types = {}
                if sample and "metadatas" in sample and sample["metadatas"]:
                    for metadata in sample["metadatas"]:
                        doc_type = metadata.get("doc_type", "unknown")
                        doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            except:
                doc_types = {"unable_to_analyze": main_count}

            return {
                "total_documents": main_count,
                "entity_chunks": entity_count,
                "clause_chunks": clause_count,
                "document_types": doc_types,
                "collections": {
                    "main": f"{self.collection_name}_main",
                    "entities": f"{self.collection_name}_entities",
                    "clauses": f"{self.collection_name}_clauses"
                }
            }
        except Exception as e:
            return {"error": str(e)}


def get_enhanced_chroma_service(
    db_dir: str = DEFAULT_DB_DIR,
    collection: str = DEFAULT_COLLECTION,
    embedding_model: str = DEFAULT_EMBED_MODEL,
) -> EnhancedChromaService:
    return EnhancedChromaService(db_dir, collection, embedding_model)
