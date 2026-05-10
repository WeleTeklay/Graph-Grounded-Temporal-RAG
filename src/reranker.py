#src/reranker.py
"""
Production-grade Cross-Encoder Reranker with batching, caching, and error handling.
"""

import hashlib
from typing import List, Tuple, Optional, Dict
from functools import lru_cache

import torch
import numpy as np
from sentence_transformers import CrossEncoder

from src.logger import get_logger
from src.utils import handle_errors

logger = get_logger(__name__)


class ReRanker:
    """
    Production-grade cross-encoder reranker.
    
    Features:
    - Automatic batching for memory efficiency
    - Score normalization
    - Text truncation to prevent OOM
    - Optional caching
    - GPU support with fallback
    - Error handling with graceful degradation
    """
    
    def __init__(
        self,
        model_name: str = 'cross-encoder/ms-marco-MiniLM-L-6-v2',
        max_length: int = 512,
        batch_size: int = 32,
        device: Optional[str] = None,
        use_cache: bool = True,
        normalize_scores: bool = True
    ):
        """
        Args:
            model_name: HuggingFace cross-encoder model
            max_length: Maximum token length (truncates longer texts)
            batch_size: Batch size for prediction
            device: 'cuda', 'cpu', or None (auto-detect)
            use_cache: Enable LRU caching of results
            normalize_scores: Normalize scores to [0, 1] range
        """
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.normalize_scores = normalize_scores
        self.use_cache = use_cache
        
        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Initialize model
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=max_length,
                device=self.device
            )
            logger.info(f"Loaded reranker: {model_name} on {self.device}")
            self.ready = True
        except Exception as e:
            logger.error(f"Failed to load reranker: {e}")
            self.model = None
            self.ready = False
        
        # Cache for query-document pairs
        self._pair_cache: Dict[str, float] = {}
    
    def _compute_cache_key(self, query: str, document: str) -> str:
        """Compute deterministic cache key for a pair."""
        normalized = f"{query[:100]}|{document[:200]}"
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _truncate_documents(self, documents: List[str]) -> List[str]:
        """Truncate documents to approximate max_length tokens."""
        # Rough estimate: 1 token ≈ 4 characters
        max_chars = self.max_length * 4
        return [doc[:max_chars] if len(doc) > max_chars else doc for doc in documents]
    
    def _normalize(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to [0, 1] range using softmax."""
        if not self.normalize_scores:
            return scores
        
        # Softmax normalization
        exp_scores = np.exp(scores - np.max(scores))
        return exp_scores / exp_scores.sum()
    
    @handle_errors(default_return=[])
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
        return_scores: bool = False
    ) -> List:
        """
        Re-rank documents using cross-encoder.
        
        Args:
            query: Search query
            documents: List of document texts
            top_k: Number of top results to return
            return_scores: If True, return (document, score) tuples
        
        Returns:
            List of documents (or tuples) sorted by relevance
        """
        if not self.ready:
            logger.warning("Reranker not ready, returning original order")
            return documents[:top_k]
        
        if not documents:
            return []
        
        # Truncate for memory efficiency
        docs = self._truncate_documents(documents)
        
        # Check cache
        uncached_docs = []
        uncached_indices = []
        scores = np.zeros(len(docs))
        
        if self.use_cache:
            for i, doc in enumerate(docs):
                cache_key = self._compute_cache_key(query, doc)
                if cache_key in self._pair_cache:
                    scores[i] = self._pair_cache[cache_key]
                else:
                    uncached_docs.append(doc)
                    uncached_indices.append(i)
        else:
            uncached_docs = docs
            uncached_indices = list(range(len(docs)))
        
        # Compute scores for uncached pairs
        if uncached_docs:
            pairs = [(query, doc) for doc in uncached_docs]
            
            try:
                # Batch prediction
                raw_scores = self.model.predict(
                    pairs,
                    batch_size=self.batch_size,
                    show_progress_bar=False
                )
                
                # Ensure numpy array
                if not isinstance(raw_scores, np.ndarray):
                    raw_scores = np.array(raw_scores)
                
                # Normalize
                norm_scores = self._normalize(raw_scores)
                
                # Update cache and scores
                for idx, doc, score in zip(uncached_indices, uncached_docs, norm_scores):
                    scores[idx] = float(score)
                    if self.use_cache:
                        cache_key = self._compute_cache_key(query, doc)
                        self._pair_cache[cache_key] = float(score)
                        
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                # Fallback: return original order
                return documents[:top_k]
        
        # Sort by score descending
        ranked_indices = np.argsort(scores)[::-1]
        
        # Return top_k
        if return_scores:
            return [(documents[i], float(scores[i])) for i in ranked_indices[:top_k]]
        else:
            return [documents[i] for i in ranked_indices[:top_k]]
    
    def rerank_batch(
        self,
        query: str,
        documents: List[str],
        batch_size: int = None
    ) -> np.ndarray:
        """
        Get scores for all documents (useful for large-scale reranking).
        
        Returns:
            Numpy array of scores
        """
        if not self.ready:
            return np.ones(len(documents)) / len(documents)
        
        docs = self._truncate_documents(documents)
        pairs = [(query, doc) for doc in docs]
        
        batch_size = batch_size or self.batch_size
        scores = self.model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        
        if self.normalize_scores:
            scores = self._normalize(np.array(scores))
        
        return np.array(scores)
    
    def clear_cache(self):
        """Clear the pair cache."""
        self._pair_cache.clear()
        logger.debug("Reranker cache cleared")
    
    def get_cache_size(self) -> int:
        """Return number of cached pairs."""
        return len(self._pair_cache)
    
    def warmup(self, sample_queries: List[str], sample_documents: List[str]):
        """
        Warmup the model with sample data for faster first inference.
        
        Args:
            sample_queries: List of sample queries
            sample_documents: List of sample documents
        """
        if not self.ready:
            return
        
        logger.info("Warming up reranker...")
        for query in sample_queries[:3]:
            _ = self.rerank(query, sample_documents[:10], top_k=3)
        logger.info("Reranker warmup complete")