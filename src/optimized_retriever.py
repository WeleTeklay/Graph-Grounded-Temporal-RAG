#src/optimized_retriever.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from functools import lru_cache

import re
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Generator
from collections import defaultdict
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import chromadb
import ollama
from neo4j import GraphDatabase

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, Timer
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="neo4j")

logger = get_logger(__name__)


@dataclass
class RetrievalCandidate:
    """Enhanced retrieval candidate with multiple scores."""
    chunk_id: str
    doc_id: str
    text: str
    effective_date: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    rerank_score: float = 0.0
    fusion_score: float = 0.0
    is_current: bool = True
    source_type: str = "unknown"


class QueryExpander:
    """Expand queries for better recall."""
    
    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model
        self.expansion_cache: Dict[str, List[str]] = {}
    
    @lru_cache(maxsize=100)
    def expand(self, query: str, num_expansions: int = 5) -> List[str]:
        """
        Generate query expansions using LLM.
        """
        if query in self.expansion_cache:
            return self.expansion_cache[query]
        
        prompt = f"""[INST] You are a legal search expert. Generate {num_expansions} alternative search queries 
for the following question. Include synonyms, legal terminology, and related concepts.

Original: "{query}"

Output as JSON list: ["expansion1", "expansion2", ...] [/INST]"""
        
        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={"temperature": 0.3, "num_predict": 200}
            )
            
            # Extract JSON
            import re
            json_match = re.search(r'\[.*\]', response['response'], re.DOTALL)
            if json_match:
                expansions = json.loads(json_match.group(0))
                expansions = [q for q in expansions if q != query][:num_expansions]
                self.expansion_cache[query] = expansions
                return expansions
        except:
            pass
        
        # Fallback: keyword extraction
        return self._keyword_expansion(query)
    
    def _keyword_expansion(self, query: str) -> List[str]:
        """Simple keyword-based expansion."""
        words = query.lower().split()
        stopwords = {'what', 'is', 'the', 'a', 'an', 'of', 'in', 'on', 'at', 'for', 'to', 'by'}
        keywords = [w for w in words if w not in stopwords]
        
        expansions = []
        for i in range(len(keywords)):
            subset = keywords[:i] + keywords[i+1:]
            if subset:
                expansions.append(' '.join(subset))
        
        return expansions[:5]


class HybridSearcher:
    """Optimized hybrid search with multiple retrieval strategies."""
    
    def __init__(self, collection: chromadb.Collection, embedder: SentenceTransformer):
        self.collection = collection
        self.embedder = embedder
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[str] = []
        self.ids: List[str] = []
        self.metadatas: List[Dict] = []
        self._build_index()
        self._max_cache_size = 1000  # Increased from 200
        self._cache_ttl = 86400  # 24 hours (increased from 3600)
    
    def metadata_boost(self, query: str, candidates: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Boost chunks that contain exact metadata matches."""
        boosted = []
        
        # Check if query is asking for metadata
        metadata_keywords = ['effective date', 'doc_id', 'document id', 'original', 'first document']
        is_metadata_query = any(kw in query.lower() for kw in metadata_keywords)
        
        for chunk_id, score in candidates:
            boost = 1.0
            if is_metadata_query:
                meta = self._get_metadata(chunk_id)
                if meta:
                    text = self._get_text(chunk_id)
                    if text and '[DOCUMENT METADATA]' in text:
                        boost = 2.0  # Double score for metadata chunks
            boosted.append((chunk_id, score * boost))
        
        return boosted
    
    def _build_index(self):
        """Build BM25 index."""
        results = self.collection.get(include=['documents', 'metadatas'])
        self.ids = results.get('ids', [])
        self.documents = results.get('documents', [])
        self.metadatas = results.get('metadatas', [])
        
        if self.documents:
            tokenized = [doc.split() for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized)
            logger.info(f"BM25 index built: {len(self.documents)} documents")
    
    def vector_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Pure vector search."""
        embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=min(top_k, len(self.ids)),
            include=['distances']
        )
        
        if not results or 'ids' not in results:
            return []
        
        ids = results['ids'][0]
        scores = 1.0 - np.array(results.get('distances', [[1.0] * len(ids)])[0])
        
        return list(zip(ids, scores.tolist()))
    
    def bm25_search(self, query: str, top_k: int = 50) -> List[Tuple[str, float]]:
        """Pure BM25 search."""
        if not self.bm25:
            return []
        
        tokenized = query.split()
        scores = self.bm25.get_scores(tokenized)
        
        # Normalize scores
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        return [(self.ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]
    
    def keyword_search(self, query: str, top_k: int = 30) -> List[Tuple[str, float]]:
        """Exact keyword matching with fuzzy fallback."""
        keywords = [w.lower() for w in query.split() if len(w) > 3]
        results = defaultdict(float)
        
        for i, doc in enumerate(self.documents):
            doc_lower = doc.lower()
            for kw in keywords:
                if kw in doc_lower:
                    results[self.ids[i]] += 1.0
        
        # Normalize by keyword count
        for chunk_id in results:
            results[chunk_id] /= len(keywords)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        return sorted_results[:top_k]
    
    def temporal_boost(self, candidates: List[Tuple[str, float]], target_date: Optional[str] = None) -> List[Tuple[str, float]]:
        """Boost scores based on temporal relevance."""
        boosted = []
        
        for chunk_id, score in candidates:
            # Find metadata
            meta = next((m for i, m in enumerate(self.metadatas) if self.ids[i] == chunk_id), {})
            effective_date = meta.get('effective_date', '')
            
            boost = 1.0
            if target_date and effective_date:
                if effective_date <= target_date:
                    boost = 1.2  # Boost current/relevant documents
                else:
                    boost = 0.5  # Penalize future documents
            
            boosted.append((chunk_id, score * boost))
        
        return boosted


class FusionRetriever:
    """
    Multi-strategy fusion retriever for optimal results.
    """
    
    def __init__(self, collection: chromadb.Collection, embedder: SentenceTransformer):
        self.searcher = HybridSearcher(collection, embedder)
        self.expander = QueryExpander()
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        # Fusion weights
        self.weights = {
            'vector': 0.30,    # Reduced - legal text needs exact matches
            'bm25': 0.30,      # Increased - term frequency matters for legal terms
            'keyword': 0.25,   # Increased - exact legal terminology critical
            'expanded': 0.15   # Reduced - expansion can dilute precision
}
    
    def reciprocal_rank_fusion(
        self, 
        result_lists: List[List[Tuple[str, float]]], 
        k: int = 60
    ) -> Dict[str, float]:
        """
        Reciprocal Rank Fusion - combines multiple ranked lists.
        """
        scores = defaultdict(float)
        
        for results in result_lists:
            for rank, (chunk_id, _) in enumerate(results, 1):
                scores[chunk_id] += 1.0 / (k + rank)
        
        return dict(scores)
    
    def weighted_fusion(
        self,
        results: Dict[str, List[float]],
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Weighted score fusion."""
        fused = defaultdict(float)
        
        for strategy, weight in weights.items():
            if strategy in results:
                for chunk_id, score in results[strategy].items():
                    fused[chunk_id] += score * weight
        
        return dict(fused)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 20,
        target_date: Optional[str] = None,
        use_expansion: bool = True,
        use_reranking: bool = True
    ) -> List[RetrievalCandidate]:
        """
        Multi-strategy retrieval with fusion and reranking.
        """
        all_results: Dict[str, Dict[str, float]] = {}
        
        # 1. Vector search
        vector_results = self.searcher.vector_search(query, top_k=50)
        all_results['vector'] = {cid: score for cid, score in vector_results}
        
        # 2. BM25 search
        bm25_results = self.searcher.bm25_search(query, top_k=50)
        all_results['bm25'] = {cid: score for cid, score in bm25_results}
        
        # 3. Keyword search
        keyword_results = self.searcher.keyword_search(query, top_k=30)
        all_results['keyword'] = {cid: score for cid, score in keyword_results}
        
        # 4. Query expansion
        if use_expansion:
            expansions = self.expander.expand(query, num_expansions=3)
            expanded_results = {}
            for exp_query in expansions:
                exp_vec = self.searcher.vector_search(exp_query, top_k=20)
                for cid, score in exp_vec:
                    expanded_results[cid] = max(expanded_results.get(cid, 0), score)
            all_results['expanded'] = expanded_results
        
        # Fuse results
        fused_scores = self.weighted_fusion(all_results, self.weights)
        
        # Temporal boosting
        if target_date:
            boosted = []
            for chunk_id, score in fused_scores.items():
                meta = self._get_metadata(chunk_id)
                if meta:
                    effective_date = meta.get('effective_date', '')
                    if effective_date and effective_date <= target_date:
                        score *= 1.5   # Stronger boost for current (+50%)
                    elif effective_date:
                        score *= 0.3   # Stronger penalty for outdated (-70%)
                boosted.append((chunk_id, score))
            fused_scores = dict(boosted)
        
        # Get top candidates
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k * 2]
        
        # Build candidate objects
        candidates = []
        for chunk_id, fusion_score in sorted_candidates:
            meta = self._get_metadata(chunk_id)
            text = self._get_text(chunk_id)
            
            if text and meta:
                candidate = RetrievalCandidate(
                    chunk_id=chunk_id,
                    doc_id=meta.get('doc_id', 'unknown'),
                    text=text,
                    effective_date=meta.get('effective_date', ''),
                    vector_score=all_results.get('vector', {}).get(chunk_id, 0),
                    bm25_score=all_results.get('bm25', {}).get(chunk_id, 0),
                    fusion_score=fusion_score,
                    is_current=True
                )
                candidates.append(candidate)
        
       # Rerank top candidates - legal docs need deeper reranking
        RERANK_DEPTH = 30  # Increased from 20 for better precision

        if use_reranking and candidates:
            candidates = self._rerank_candidates(query, candidates[:RERANK_DEPTH], top_k)
        
        return candidates[:top_k]
    
    def _rerank_candidates(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_k: int
    ) -> List[RetrievalCandidate]:
        """Rerank using cross-encoder with confidence threshold."""
        if not candidates:
            return []
        
        pairs = [(query, c.text[:500]) for c in candidates]
        scores = self.reranker.predict(pairs)
        
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
            
            # Only trust reranker if score is high enough
            if score > 0.3:  # Confidence threshold
                # High confidence: weight reranker more
                candidate.fusion_score = 0.4 * candidate.fusion_score + 0.6 * float(score)
            else:
                # Low confidence: keep original fusion score
                candidate.fusion_score = candidate.fusion_score
        
        candidates.sort(key=lambda x: x.fusion_score, reverse=True)
        return candidates[:top_k]
    
    def _get_metadata(self, chunk_id: str) -> Optional[Dict]:
        """Get metadata for a chunk."""
        for i, cid in enumerate(self.searcher.ids):
            if cid == chunk_id:
                return self.searcher.metadatas[i] if i < len(self.searcher.metadatas) else {}
        return None
    
    def _get_text(self, chunk_id: str) -> Optional[str]:
        """Get text for a chunk."""
        for i, cid in enumerate(self.searcher.ids):
            if cid == chunk_id:
                return self.searcher.documents[i] if i < len(self.searcher.documents) else None
        return None
    def _should_expand_query(self, query: str) -> bool:
        """
        Determine if query expansion should be used.
        Legal queries benefit LESS from expansion.
        """
        # Don't expand if query contains specific legal citations
        legal_patterns = [
            r'section\s+\d+',      # Section 4.2
            r'doc_\d+',            # doc_001
            r'\d{4}-\d{2}-\d{2}',  # 2020-01-15
            r'§\s*\d+',            # § 4.2
        ]
        
        for pattern in legal_patterns:
            if re.search(pattern, query.lower()):
                return False  # Exact legal citation - don't expand
        
        # Expand only for vague queries
        vague_indicators = ['about', 'tell me', 'explain', 'what is', 'describe']
        return any(ind in query.lower() for ind in vague_indicators)
class ContextOptimizer:
    """Optimize context before sending to LLM."""
    
    def __init__(self, max_tokens: int = 3000):
        self.max_tokens = max_tokens
        self.model = "llama3.2:3b"
    
    def deduplicate(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Remove near-duplicate chunks."""
        seen_hashes = set()
        unique = []
        
        for c in candidates:
            text_hash = hashlib.md5(c.text[:200].encode()).hexdigest()
            if text_hash not in seen_hashes:
                seen_hashes.add(text_hash)
                unique.append(c)
        
        return unique
    
    def prioritize_current(self, candidates: List[RetrievalCandidate]) -> List[RetrievalCandidate]:
        """Prioritize current versions over historical."""
        current = [c for c in candidates if c.is_current]
        historical = [c for c in candidates if not c.is_current]
        
        # Interleave: current first, then historical for context
        result = current[:5] + historical[:3]
        return result
    
    def compress_context(self, candidates: List[RetrievalCandidate], query: str) -> str:
        """
        Intelligently compress context to fit token limit.
        """
        context_parts = []
        total_chars = 0
        char_limit = self.max_tokens * 3  # Rough estimate: 1 token ≈ 3 chars
        
        for c in candidates:
            # Extract most relevant sentences
            relevant_text = self._extract_relevant_sentences(c.text, query, max_chars=400)
            
            part = f"[{c.doc_id} | Effective: {c.effective_date} | Score: {c.fusion_score:.2f}]\n{relevant_text}\n"
            
            if total_chars + len(part) > char_limit:
                # Truncate last part if needed
                remaining = char_limit - total_chars
                if remaining > 100:
                    part = part[:remaining] + "..."
                    context_parts.append(part)
                break
            
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n---\n".join(context_parts)
    
    def _extract_relevant_sentences(self, text: str, query: str, max_chars: int = 400) -> str:
        """Extract sentences most relevant to query."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        if len(text) <= max_chars:
            return text
        
        # Score sentences by keyword overlap
        query_keywords = set(query.lower().split())
        scored = []
        
        for sent in sentences:
            sent_keywords = set(sent.lower().split())
            overlap = len(query_keywords & sent_keywords)
            scored.append((sent, overlap))
        
        # Sort by relevance and take top
        scored.sort(key=lambda x: x[1], reverse=True)
        
        result = []
        total_chars = 0
        for sent, _ in scored:
            if total_chars + len(sent) > max_chars:
                break
            result.append(sent)
            total_chars += len(sent)
        
        return ' '.join(result)

class QueryPreprocessor:
    """Enhance queries for better retrieval."""
    
    def __init__(self):
        self.doc_aliases = {
            'original': 'doc_001',
            'first': 'doc_001',
            'amendment': 'doc_002',
            'restated': 'doc_004',
            'ethiopian': 'doc_001',
            'proclamation': 'doc_001',
        }
    
    def enhance(self, query: str) -> str:
        """Add context hints to query."""
        return query
        
        # Add doc_id hints
        for alias, doc_id in self.doc_aliases.items():
            if alias in query.lower() and doc_id not in query:
                enhanced = f"{enhanced} {doc_id}"
        
        # Add metadata hints
        if any(kw in query.lower() for kw in ['effective date', 'date', 'when']):
            if 'EFFECTIVE_DATE' not in enhanced:
                enhanced = f"{enhanced} EFFECTIVE_DATE metadata"
        
        if any(kw in query.lower() for kw in ['doc_id', 'document id']):
            if 'DOC_ID' not in enhanced:
                enhanced = f"{enhanced} DOC_ID metadata"
        
        return enhanced


class OptimizedQueryEngine:
    """
    Complete optimized query engine with caching and performance enhancements.
    """
    
    # Class-level BM25 cache (shared across instances)
    _bm25_cache = None
    _collection_cache = None
    
    def __init__(self):
        # Initialize components
        self.chroma_client = chromadb.PersistentClient(
            path=str(config.paths.vectors_dir),
            settings=chromadb.Settings(anonymized_telemetry=False)
        )
        self.collection = self.chroma_client.get_collection("legal_clauses")
        self.embedder = SentenceTransformer(config.embedding.model_name)
        
        self.retriever = FusionRetriever(self.collection, self.embedder)
        self.optimizer = ContextOptimizer()
        self.preprocessor = QueryPreprocessor()
        
        # Caches
        self._answer_cache: Dict[str, Dict[str, Any]] = {}
        self._embedding_cache: Dict[str, List[float]] = {}
        self._cache_ttl = 86400  # 24 hour
        self._max_cache_size = 1000
        
        # Graph connection
        self.graph_enabled = False
        if config.neo4j.is_valid():
            try:
                self.neo4j_driver = GraphDatabase.driver(
                    config.neo4j.uri,
                    auth=(config.neo4j.user, config.neo4j.password),
                    max_connection_lifetime=3600,
                    max_connection_pool_size=5
                )
                self.neo4j_driver.verify_connectivity()
                self.graph_enabled = True
                logger.info("Graph features enabled")
            except Exception as e:
                logger.warning(f"Graph unavailable: {e}")
                self.neo4j_driver = None
        
        logger.info("Optimized Query Engine initialized")
    
    def _get_cache_key(self, question: str, target_date: Optional[str]) -> str:
        """Generate deterministic cache key."""
        normalized = f"{question.strip().lower()}|{target_date or 'none'}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _clean_expired_cache(self):
        """Remove expired cache entries."""
        now = datetime.now()
        expired = []
        for key, value in self._answer_cache.items():
            if now > value.get('expires_at', now):
                expired.append(key)
        for key in expired:
            del self._answer_cache[key]
    
    def _evict_lru_if_needed(self):
        """Evict least recently used cache entries if over max size."""
        if len(self._answer_cache) > self._max_cache_size:
            # Remove oldest 20%
            remove_count = int(self._max_cache_size * 0.2)
            sorted_keys = sorted(
                self._answer_cache.keys(),
                key=lambda k: self._answer_cache[k].get('cached_at', datetime.min)
            )
            for key in sorted_keys[:remove_count]:
                del self._answer_cache[key]
    
    @handle_errors(default_return=None)
    def _get_cached_embedding(self, text: str) -> List[float]:
        """Get cached embedding or compute and cache."""
        cache_key = hashlib.md5(text[:500].encode()).hexdigest()
        
        if cache_key in self._embedding_cache:
            return self._embedding_cache[cache_key]
        
        embedding = self.embedder.encode(text, normalize_embeddings=True).tolist()
        
        if len(self._embedding_cache) < 500:
            self._embedding_cache[cache_key] = embedding
        
        return embedding
    
    def answer(self, question: str, target_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Answer question with optimized retrieval and caching.
        """
        # Preprocess query
        enhanced_question = self.preprocessor.enhance(question)
        
        # Check cache
        cache_key = self._get_cache_key(enhanced_question, target_date)
        if cache_key in self._answer_cache:
            cached = self._answer_cache[cache_key]
            if datetime.now() < cached.get('expires_at', datetime.now()):
                logger.info(f"Cache hit: {question[:50]}...")
                result = cached['result'].copy()
                result['cached'] = True
                return result
        
        # Determine if query needs expansion
        complex_keywords = ['compare', 'difference', 'change', 'history', 'evolution', 'amendment']
        use_expansion = False
        
        # Adjust retrieval parameters
        top_k = 12
        
        # Retrieval
        retrieval_start = time.perf_counter()
        candidates = self.retriever.retrieve(
            enhanced_question,
            top_k=top_k,
            target_date=target_date,
            use_expansion=use_expansion,
            use_reranking=True
        )
        retrieval_time = time.perf_counter() - retrieval_start
        
        if not candidates:
            return {
                'answer': "No relevant documents found.",
                'sources': [],
                'confidence': 0.0,
                'retrieval_time_ms': int(retrieval_time * 1000),
                'cached': False
            }
        
        # Resolve current versions via graph
        if self.graph_enabled:
            candidates = self._resolve_current_versions(candidates)
        
        # Deduplicate and prioritize
        candidates = self.optimizer.deduplicate(candidates)
        candidates = self.optimizer.prioritize_current(candidates)
        
        # Compress context
        context = self.optimizer.compress_context(candidates, enhanced_question)
        
        # Build sources
        sources = [
            {
                'chunk_id': c.chunk_id,
                'doc_id': c.doc_id,
                'effective_date': c.effective_date,
                'score': round(c.fusion_score, 3),
                'is_current': c.is_current
            }
            for c in candidates[:5]
        ]
        
        # Generate answer
        generation_start = time.perf_counter()
        prompt = self._build_prompt(enhanced_question, context, target_date)
        
        response = ollama.generate(
            model=config.ollama.model,
            prompt=prompt,
            options={
                "temperature": 0.1,
                "num_predict": 400 if use_expansion else 250,
                "top_p": 0.9
            }
        )
        generation_time = time.perf_counter() - generation_start
        answer = response['response'].strip()
        
        # Calculate confidence
        confidence = self._calculate_confidence(candidates, answer)
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': confidence,
            'retrieval_count': len(candidates),
            'graph_used': self.graph_enabled,
            'retrieval_time_ms': int(retrieval_time * 1000),
            'generation_time_ms': int(generation_time * 1000),
            'total_time_ms': int((retrieval_time + generation_time) * 1000),
            'cached': False,
            'query_enhanced': enhanced_question != question
        }
        
        # Cache result
        self._answer_cache[cache_key] = {
            'result': result.copy(),
            'cached_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(seconds=self._cache_ttl)
        }
        
        # Cleanup cache periodically
        if len(self._answer_cache) % 10 == 0:
            self._clean_expired_cache()
            self._evict_lru_if_needed()
        
        return result
    
    def answer_stream(self, question: str, target_date: Optional[str] = None) -> Generator[str, None, None]:
        """
        Stream answer token by token for real-time UI.
        """
        enhanced_question = self.preprocessor.enhance(question)
        
        with Timer("Optimized retrieval", log=False):
            candidates = self.retriever.retrieve(
                enhanced_question,
                top_k=10,
                target_date=target_date,
                use_expansion=False,
                use_reranking=True
            )
        
        if not candidates:
            yield "No relevant documents found."
            return
        
        if self.graph_enabled:
            candidates = self._resolve_current_versions(candidates)
        
        candidates = self.optimizer.deduplicate(candidates)
        context = self.optimizer.compress_context(candidates, enhanced_question)
        prompt = self._build_prompt(enhanced_question, context, target_date)
        
        # Stream response
        for chunk in ollama.generate(
            model=config.ollama.model,
            prompt=prompt,
            options={"temperature": 0.1, "num_predict": 300},
            stream=True
        ):
            if 'response' in chunk:
                yield chunk['response']
    
    def _resolve_current_versions(self, candidates: List) -> List:
        """Use graph to mark current versions."""
        if not self.neo4j_driver:
            return candidates
        
        with self.neo4j_driver.session() as session:
            for c in candidates:
                result = session.run("""
                    MATCH (d:Document {doc_id: $doc_id})
                    WHERE NOT (d)-[:SUPERSEDES]->()
                    RETURN d.doc_id as current_doc_id, d.effective_date as current_date
            """, doc_id=c.doc_id).single()
                
                if result:
                    c.is_current = (c.doc_id == result['current_doc_id'])
                    if c.is_current:
                        c.effective_date = str(result['current_date'])
            
        return candidates
    
    def _build_prompt(self, question: str, context: str, target_date: Optional[str]) -> str:
        """Build optimized prompt."""
        date_context = ""
        if target_date:
            date_context = f"Today's date is {target_date}. Use information effective on or before this date."
        else:
            date_context = "Use the most current information available."
        
        return f"""[INST] You are a precise legal assistant. Answer based ONLY on the context below.

{date_context}

CONTEXT:
{context}

Question: {question}

Instructions:
- Extract exact information from the context
- If the answer is in metadata (EFFECTIVE_DATE, DOC_ID), use it
- Be concise and specific
- If not found, say "Not found in documents"

Answer: [/INST]"""
    
    def _calculate_confidence(self, candidates: List, answer: str) -> float:
        """Calculate answer confidence."""
        if not candidates:
            return 0.0
        
        # Top candidate scores
        top_scores = [c.fusion_score for c in candidates[:3] if hasattr(c, 'fusion_score')]
        avg_score = np.mean(top_scores) if top_scores else 0.5
        
        # Answer quality heuristics
        length_score = min(1.0, len(answer.split()) / 15)
        not_found_penalty = 0.3 if "not found" in answer.lower() else 1.0
        
        confidence = avg_score * length_score * not_found_penalty
        
        return round(min(1.0, max(0.1, confidence)), 3)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        return {
            'answer_cache_size': len(self._answer_cache),
            'embedding_cache_size': len(self._embedding_cache),
            'max_cache_size': self._max_cache_size,
            'cache_ttl_seconds': self._cache_ttl
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self._answer_cache.clear()
        self._embedding_cache.clear()
        logger.info("All caches cleared")