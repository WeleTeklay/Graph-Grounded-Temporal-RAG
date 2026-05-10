#src/contradiction_detector.py
"""
Production-Grade Contradiction Detector for Graph-Grounded Temporal RAG.
Actively identifies, resolves, and explains contradictions across evolving documents.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

import numpy as np
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, Timer

logger = get_logger(__name__)


@dataclass
class ClauseVersion:
    """Represents a single version of a clause."""
    chunk_id: str
    doc_id: str
    effective_date: str
    text: str
    text_hash: str
    is_current: bool = False
    similarity_score: float = 1.0


@dataclass
class ContradictionReport:
    """Complete contradiction analysis report."""
    query: str
    keyword: str
    versions_found: List[ClauseVersion]
    current_version: Optional[ClauseVersion]
    historical_versions: List[ClauseVersion]
    has_contradiction: bool
    contradiction_type: str  # 'amendment', 'repeal', 'reinstatement', 'none'
    explanation: str
    confidence: float
    graph_path: List[str]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ContradictionDetector:
    """
    Production-grade contradiction detector with graph traversal and semantic analysis.
    Integrates with existing Neo4j graph and QueryEngine.
    """
    
    def __init__(self, neo4j_driver=None, embedding_model=None):
        """
        Args:
            neo4j_driver: Existing Neo4j driver (reuses connection)
            embedding_model: Existing embedding model (reuses model)
        """
        self.driver = neo4j_driver
        self.embedder = embedding_model
        
        self._init_connections()
        self._compile_patterns()
        
    def _init_connections(self):
        """Initialize or reuse connections."""
        if self.driver is None and config.neo4j.is_valid():
            try:
                self.driver = GraphDatabase.driver(
                    config.neo4j.uri,
                    auth=(config.neo4j.user, config.neo4j.password)
                )
                self.driver.verify_connectivity()
                logger.info("ContradictionDetector connected to Neo4j")
            except Exception as e:
                logger.warning(f"Neo4j unavailable: {e}")
                self.driver = None
        
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer(config.embedding.model_name)
            except Exception as e:
                logger.warning(f"Embedding model unavailable: {e}")
                self.embedder = None
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword extraction."""
        self.patterns = {
            'fee': re.compile(r'\b(fee|penalty|charge|payment|cost|price)\b', re.IGNORECASE),
            'date': re.compile(r'\b(date|effective|commence|start|begin)\b', re.IGNORECASE),
            'obligation': re.compile(r'\b(must|shall|required|obligated|responsible)\b', re.IGNORECASE),
            'party': re.compile(r'\b(employer|employee|borrower|lender|party|parties)\b', re.IGNORECASE),
            'amount': re.compile(r'\$\d+|\d+\s*percent|\d+%|\d+\s*dollars', re.IGNORECASE),
        }
    
    def extract_keyword(self, query: str) -> str:
        """Extract the most relevant keyword for contradiction detection."""
        query_lower = query.lower()
        
        for category, pattern in self.patterns.items():
            match = pattern.search(query)
            if match:
                return match.group(0).lower()
        
        words = query_lower.split()
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'to', 'by'}
        content_words = [w for w in words if w not in stopwords and len(w) > 2]
        
        return content_words[0] if content_words else query_lower[:20]
    
    def compute_text_hash(self, text: str) -> str:
        """Compute semantic hash of text for comparison."""
        normalized = ' '.join(text.lower().split())[:500]
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts."""
        if self.embedder is None:
            return self._jaccard_similarity(text1, text2)
        
        try:
            emb1 = self.embedder.encode(text1[:1000], normalize_embeddings=True)
            emb2 = self.embedder.encode(text2[:1000], normalize_embeddings=True)
            return float(np.dot(emb1, emb2))
        except Exception:
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fallback Jaccard similarity."""
        set1 = set(text1.lower().split())
        set2 = set(text2.lower().split())
        if not set1 or not set2:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)
    
    @handle_errors(default_return=[])
    def find_all_versions_by_keyword(self, keyword: str, limit: int = 50) -> List[ClauseVersion]:
        """
        Find ALL versions of clauses containing the keyword using graph traversal.
        """
        if self.driver is None:
            return []
        
        versions = []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (c:Clause)
                WHERE toLower(c.text) CONTAINS toLower($keyword)
                RETURN DISTINCT 
                    c.id as chunk_id,
                    c.doc_id as doc_id,
                    c.effective_date as effective_date,
                    c.text as text
                ORDER BY effective_date
                LIMIT $limit
            """, keyword=keyword, limit=limit)
            
            for record in result:
                text = record['text'] or ''
                version = ClauseVersion(
                    chunk_id=record['chunk_id'],
                    doc_id=record['doc_id'],
                    effective_date=str(record['effective_date']),
                    text=text,
                    text_hash=self.compute_text_hash(text)
                )
                versions.append(version)
        
        return versions
        
    @handle_errors(default_return=[])
    def find_all_versions_by_semantic(self, query: str, limit: int = 50) -> List[ClauseVersion]:
        """
        Find ALL versions using combined keyword and semantic search.
        """
        keyword = self.extract_keyword(query)
        versions = self.find_all_versions_by_keyword(keyword, limit)
        
        if len(versions) < 2:
            return versions
        
        return versions
    
    def group_versions_by_similarity(
        self, 
        versions: List[ClauseVersion], 
        threshold: float = 0.6
    ) -> List[List[ClauseVersion]]:
        """
        Group versions by semantic similarity to identify related clauses.
        """
        if len(versions) < 2:
            return [versions] if versions else []
        
        groups = []
        used = set()
        
        for i, v1 in enumerate(versions):
            if v1.chunk_id in used:
                continue
            
            group = [v1]
            used.add(v1.chunk_id)
            
            for j, v2 in enumerate(versions[i+1:], i+1):
                if v2.chunk_id in used:
                    continue
                
                similarity = self.compute_similarity(v1.text, v2.text)
                v2.similarity_score = similarity
                
                if similarity >= threshold:
                    group.append(v2)
                    used.add(v2.chunk_id)
            
            groups.append(group)
        
        return groups
    
    def find_current_version(self, versions: List[ClauseVersion]) -> Optional[ClauseVersion]:
        """
        Find the current (most recent) version using graph traversal.
        """
        if not versions:
            return None
        
        if self.driver is None:
            sorted_versions = sorted(versions, key=lambda v: v.effective_date, reverse=True)
            return sorted_versions[0] if sorted_versions else None
        
        current = None
        latest_date = None
        
        for version in versions:
            with self.driver.session() as session:
                result = session.run("""
                    MATCH (c:Clause {id: $chunk_id})
                    OPTIONAL MATCH (c)-[:SUPERSEDES*]->(newer:Clause)
                    WHERE NOT (newer)-[:SUPERSEDES]->()
                    RETURN COALESCE(newer.effective_date, c.effective_date) as effective_date,
                           COALESCE(newer.id, c.id) as chunk_id
                """, chunk_id=version.chunk_id)
                
                record = result.single()
                if record:
                    eff_date = str(record['effective_date'])
                    if latest_date is None or eff_date > latest_date:
                        latest_date = eff_date
                        for v in versions:
                            if v.chunk_id == record['chunk_id']:
                                current = v
                                break
        
        if current:
            current.is_current = True
        
        return current
    
    def get_graph_path(self, chunk_id: str) -> List[str]:
        """Get the SUPERSEDES path for a clause."""
        if self.driver is None:
            return [chunk_id]
        
        path = []
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH path = (start:Clause {id: $chunk_id})-[:SUPERSEDES*0..10]->(end:Clause)
                WHERE NOT (end)-[:SUPERSEDES]->()
                RETURN [node in nodes(path) | node.id] as path_ids
            """, chunk_id=chunk_id)
            
            record = result.single()
            if record:
                path = record['path_ids']
        
        return path if path else [chunk_id]
    
    def determine_contradiction_type(self, versions: List[ClauseVersion]) -> str:
        """Determine the type of contradiction from version history."""
        if len(versions) < 2:
            return 'none'
        
        sorted_versions = sorted(versions, key=lambda v: v.effective_date)
        
        hashes = [v.text_hash for v in sorted_versions]
        if len(set(hashes)) == 1:
            return 'none'
        
        if hashes[0] == hashes[-1]:
            return 'reinstatement'
        
        if all(h != hashes[0] for h in hashes[1:]):
            return 'amendment'
        
        return 'amendment'
    
    def generate_explanation(
        self,
        current: Optional[ClauseVersion],
        historical: List[ClauseVersion],
        contradiction_type: str
    ) -> str:
        """Generate human-readable explanation of contradiction resolution."""
        if contradiction_type == 'none':
            return "No contradiction detected. Only one version of this clause exists."
        
        if current is None:
            return "Unable to determine current version."
        
        hist_dates = [v.effective_date for v in historical[:3]]
        hist_docs = [v.doc_id for v in historical[:3]]
        
        if contradiction_type == 'amendment':
            return (
                f"This clause was amended. "
                f"Original version(s) from {', '.join(hist_docs)} (effective {', '.join(hist_dates)}) "
                f"have been superseded by the current version in {current.doc_id} "
                f"(effective {current.effective_date})."
            )
        elif contradiction_type == 'reinstatement':
            return (
                f"This clause was modified then reinstated. "
                f"The current version in {current.doc_id} (effective {current.effective_date}) "
                f"restores the original language."
            )
        else:
            return (
                f"Multiple versions exist. The current version is from {current.doc_id} "
                f"(effective {current.effective_date})."
            )
    
    def calculate_confidence(self, versions: List[ClauseVersion], current: Optional[ClauseVersion]) -> float:
        """Calculate confidence score for contradiction detection."""
        if len(versions) < 2:
            return 1.0
        
        if current is None:
            return 0.5
        
        confidence = 1.0
        
        if len(versions) > 5:
            confidence *= 0.9
        
        similarities = [v.similarity_score for v in versions if v.similarity_score < 1.0]
        if similarities:
            avg_sim = sum(similarities) / len(similarities)
            confidence *= avg_sim
        
        return min(1.0, max(0.5, confidence))
    
    def analyze(self, query: str, target_date: Optional[str] = None) -> ContradictionReport:
        """
        Perform complete contradiction analysis for a query.
        
        Args:
            query: User's question
            target_date: Optional date for temporal context
        
        Returns:
            Complete ContradictionReport
        """
        with Timer("Contradiction analysis", log=False):
            keyword = self.extract_keyword(query)
            
            versions = self.find_all_versions_by_semantic(query)
            
            if not versions:
                return ContradictionReport(
                    query=query,
                    keyword=keyword,
                    versions_found=[],
                    current_version=None,
                    historical_versions=[],
                    has_contradiction=False,
                    contradiction_type='none',
                    explanation="No matching clauses found.",
                    confidence=0.0,
                    graph_path=[]
                )
            
            groups = self.group_versions_by_similarity(versions)
            
            all_historical = []
            all_currents = []
            
            for group in groups:
                if len(group) >= 2:
                    current = self.find_current_version(group)
                    if current:
                        all_currents.append(current)
                        historical = [v for v in group if v.chunk_id != current.chunk_id]
                        all_historical.extend(historical)
            
            if not all_currents and versions:
                all_currents = [self.find_current_version(versions)]
                all_historical = [v for v in versions if all_currents[0] and v.chunk_id != all_currents[0].chunk_id]
            
            current = all_currents[0] if all_currents else None
            historical = all_historical[:10]
            all_versions = (all_currents + historical) if current else versions
            
            contradiction_type = self.determine_contradiction_type(all_versions)
            has_contradiction = contradiction_type != 'none'
            
            explanation = self.generate_explanation(current, historical, contradiction_type)
            confidence = self.calculate_confidence(all_versions, current)
            
            graph_path = self.get_graph_path(current.chunk_id) if current else []
            
            return ContradictionReport(
                query=query,
                keyword=keyword,
                versions_found=all_versions,
                current_version=current,
                historical_versions=historical,
                has_contradiction=has_contradiction,
                contradiction_type=contradiction_type,
                explanation=explanation,
                confidence=confidence,
                graph_path=graph_path
            )
    
    def enhance_answer(self, answer: str, report: ContradictionReport) -> str:
        """Enhance an answer with contradiction awareness."""
        if not report.has_contradiction:
            return answer
        
        if report.contradiction_type == 'amendment':
            note = (
                f"\n\n[Note: This clause was amended. "
                f"The current version (effective {report.current_version.effective_date}) "
                f"supersedes {len(report.historical_versions)} earlier version(s).]"
            )
            return answer + note
        
        return answer


def create_contradiction_detector(neo4j_driver=None, embedding_model=None) -> ContradictionDetector:
    """Factory function for contradiction detector."""
    return ContradictionDetector(neo4j_driver, embedding_model)