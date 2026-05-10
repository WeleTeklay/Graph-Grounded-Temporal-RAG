#src/05_create_index.py
"""
Phase 5: Production-Grade Vector Index Builder
Builds ChromaDB vector index with batching, validation, and metadata enrichment.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

import chromadb
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_json_load, safe_json_dump, Timer

os.environ["CHROMA_TELEMETRY_IMPL"] = "none"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

logger = get_logger(__name__)


@dataclass
class IndexStats:
    """Statistics from index building."""
    total_chunks: int
    chunks_indexed: int
    batches_processed: int
    embedding_model: str
    vector_dimension: int
    build_time_seconds: float
    collection_size_bytes: int
    timestamp: str


class VectorIndexBuilder:
    """
    Production-grade ChromaDB vector index builder.
    
    Features:
    - Batch processing for memory efficiency
    - Duplicate detection and handling
    - Metadata enrichment
    - Progress tracking
    - Validation and reporting
    """
    
    def __init__(self, model_name: str = None, batch_size: int = None):
        self.model_name = model_name or config.embedding.model_name
        self.batch_size = batch_size or config.embedding.batch_size
        self.vector_dimension = config.embedding.dimension
        
        self.client = None
        self.collection = None
        self.embedder = None
        
        self._initialize()
    
    def _initialize(self):
        """Initialize ChromaDB client and embedding model."""
        self.client = chromadb.PersistentClient(
            path=str(config.paths.vectors_dir),
            settings=chromadb.Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.embedder = SentenceTransformer(self.model_name)
        logger.info(f"Loaded embedding model: {self.model_name}")
    
    def _get_or_create_collection(self, reset: bool = False):
        """Get existing collection or create new one."""
        collection_name = "legal_clauses"
        
        if reset:
            try:
                self.client.delete_collection(collection_name)
                logger.info(f"Deleted existing collection: {collection_name}")
            except Exception:
                pass
        
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
        except Exception:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={
                    "description": "Legal document clauses with temporal metadata",
                    "created_at": datetime.now().isoformat(),
                    "embedding_model": self.model_name,
                    "vector_dimension": self.vector_dimension
                }
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def _compute_chunk_hash(self, text: str) -> str:
        """Compute hash for chunk deduplication."""
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:16]
    
    def _enrich_metadata(self, chunk: Dict) -> Dict:
        """Enrich chunk with additional metadata."""
        text = chunk.get('text', '')
        
        return {
            'doc_id': chunk.get('doc_id', 'unknown'),
            'effective_date': chunk.get('effective_date', 'unknown'),
            'chunk_id': chunk.get('chunk_id', 'unknown'),
            'chunk_index': chunk.get('chunk_index', 0),
            'char_count': len(text),
            'word_count': len(text.split()),
            'has_section_header': bool(chunk.get('section_headers')),
            'chunk_hash': self._compute_chunk_hash(text)
        }
    
    def _prepare_batch(self, chunks: List[Dict]) -> Tuple[List[str], List[str], List[List[float]], List[Dict]]:
        """
        Prepare a batch for indexing.
        
        Returns:
            Tuple of (ids, documents, embeddings, metadatas)
        """
        ids = []
        documents = []
        metadatas = []
        texts_to_embed = []
        
        for chunk in chunks:
            chunk_id = chunk.get('chunk_id')
            text = chunk.get('text', '')
            
            if not chunk_id or not text:
                continue
            
            # Enrich text with date context for better retrieval
            effective_date = chunk.get('effective_date', '')
            enriched_text = f"[Effective: {effective_date}] {text}" if effective_date else text
            
            ids.append(chunk_id)
            documents.append(enriched_text)
            texts_to_embed.append(text)
            metadatas.append(self._enrich_metadata(chunk))
        
        if not texts_to_embed:
            return [], [], [], []
        
        # Generate embeddings
        embeddings = self.embedder.encode(
            texts_to_embed,
            normalize_embeddings=True,
            show_progress_bar=False
        ).tolist()
        
        return ids, documents, embeddings, metadatas
    
    @handle_errors(default_return=0)
    def index_chunks(self, chunks: List[Dict], reset: bool = True) -> int:
        """
        Index chunks in batches.
        
        Args:
            chunks: List of chunk dictionaries
            reset: Whether to reset existing collection
            
        Returns:
            Number of chunks indexed
        """
        self._get_or_create_collection(reset=reset)
        
        total_indexed = 0
        duplicate_count = 0
        error_count = 0
        
        # Process in batches
        batches = [chunks[i:i + self.batch_size] 
                  for i in range(0, len(chunks), self.batch_size)]
        
        logger.info(f"Indexing {len(chunks)} chunks in {len(batches)} batches")
        
        with tqdm(total=len(chunks), desc="Indexing chunks") as pbar:
            for batch_idx, batch in enumerate(batches):
                try:
                    ids, documents, embeddings, metadatas = self._prepare_batch(batch)
                    
                    if not ids:
                        pbar.update(len(batch))
                        continue
                    
                    # Check for existing IDs
                    existing = self.collection.get(ids=ids)
                    existing_ids = set(existing['ids']) if existing else set()
                    
                    # Filter out duplicates
                    new_items = [(i, d, e, m) for i, d, e, m in zip(ids, documents, embeddings, metadatas)
                                if i not in existing_ids]
                    
                    duplicate_count += len(ids) - len(new_items)
                    
                    if new_items:
                        new_ids, new_docs, new_embs, new_metas = zip(*new_items)
                        
                        self.collection.add(
                            ids=list(new_ids),
                            documents=list(new_docs),
                            embeddings=list(new_embs),
                            metadatas=list(new_metas)
                        )
                        total_indexed += len(new_ids)
                    
                    pbar.update(len(batch))
                    
                except Exception as e:
                    logger.error(f"Batch {batch_idx} failed: {e}")
                    error_count += len(batch)
                    pbar.update(len(batch))
        
        if duplicate_count > 0:
            logger.info(f"Skipped {duplicate_count} duplicate chunks")
        if error_count > 0:
            logger.warning(f"Failed to index {error_count} chunks")
        
        return total_indexed
    
    def get_collection_stats(self) -> Dict:
        """Get current collection statistics."""
        if not self.collection:
            return {}
        
        count = self.collection.count()
        
        # Get storage size
        vector_path = config.paths.vectors_dir
        total_size = sum(f.stat().st_size for f in vector_path.rglob('*') if f.is_file())
        
        return {
            'total_documents': count,
            'storage_size_bytes': total_size,
            'storage_size_mb': round(total_size / (1024 * 1024), 2),
            'embedding_model': self.model_name,
            'vector_dimension': self.vector_dimension
        }
    
    def validate_index(self) -> Tuple[bool, List[str]]:
        """Validate index integrity."""
        issues = []
        
        if not self.collection:
            issues.append("Collection not initialized")
            return False, issues
        
        count = self.collection.count()
        if count == 0:
            issues.append("Collection is empty")
        
        # Test retrieval
        try:
            results = self.collection.query(query_texts=["test"], n_results=1)
            if not results:
                issues.append("Query returned no results")
        except Exception as e:
            issues.append(f"Query test failed: {e}")
        
        return len(issues) == 0, issues
    
    def build(self, chunks_path: Path = None, reset: bool = True) -> Optional[IndexStats]:
        """
        Build complete vector index.
        
        Args:
            chunks_path: Path to clauses.json
            reset: Whether to reset existing collection
            
        Returns:
            IndexStats if successful, None otherwise
        """
        start_time = datetime.now()
        
        # Load chunks
        chunks_path = chunks_path or config.paths.chunks_dir / "clauses.json"
        
        if not chunks_path.exists():
            logger.error(f"Chunks not found: {chunks_path}")
            return None
        
        chunks = safe_json_load(chunks_path, default=[])
        
        if not chunks:
            logger.error("No chunks to index")
            return None
        
        logger.info(f"Loaded {len(chunks)} chunks")
        
        with Timer("Vector indexing"):
            indexed = self.index_chunks(chunks, reset=reset)
        
        if indexed == 0:
            logger.error("No chunks were indexed")
            return None
        
        # Validate
        is_valid, issues = self.validate_index()
        if not is_valid:
            logger.warning(f"Index validation issues: {issues}")
        
        # Get stats
        stats = self.get_collection_stats()
        build_time = (datetime.now() - start_time).total_seconds()
        
        # Save report
        report = {
            'build_timestamp': datetime.now().isoformat(),
            'total_chunks_loaded': len(chunks),
            'chunks_indexed': indexed,
            'batches_processed': (len(chunks) + self.batch_size - 1) // self.batch_size,
            'embedding_model': self.model_name,
            'vector_dimension': self.vector_dimension,
            'build_time_seconds': build_time,
            'validation_passed': is_valid,
            'validation_issues': issues,
            'collection_stats': stats
        }
        
        report_path = config.paths.vectors_dir / "index_build_report.json"
        safe_json_dump(report, report_path)
        
        logger.info(f"Index build complete in {build_time:.1f}s")
        logger.info(f"Chunks indexed: {indexed}/{len(chunks)}")
        logger.info(f"Collection size: {stats.get('storage_size_mb', 0)} MB")
        logger.info(f"Report: {report_path}")
        
        return IndexStats(
            total_chunks=len(chunks),
            chunks_indexed=indexed,
            batches_processed=report['batches_processed'],
            embedding_model=self.model_name,
            vector_dimension=self.vector_dimension,
            build_time_seconds=build_time,
            collection_size_bytes=stats.get('storage_size_bytes', 0),
            timestamp=datetime.now().isoformat()
        )


def main():
    """Main entry point for vector indexing phase."""
    logger.info("=" * 60)
    logger.info("Phase 5: Production Vector Index Builder")
    logger.info("=" * 60)
    
    builder = VectorIndexBuilder()
    stats = builder.build(reset=True)
    
    if stats:
        logger.info(f"Successfully indexed {stats.chunks_indexed} chunks")
        return 0
    else:
        logger.error("Index build failed")
        return 1


if __name__ == "__main__":
    exit(main())

