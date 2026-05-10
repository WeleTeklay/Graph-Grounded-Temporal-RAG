"""
Phase 2: Production-Grade Document Chunker
Semantic chunking with section awareness, overlap control, and validation.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import re
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

import pandas as pd
from tqdm import tqdm

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_json_dump, Timer

logger = get_logger(__name__)


@dataclass
class Chunk:
    """Structured representation of a document chunk."""
    chunk_id: str
    doc_id: str
    effective_date: str
    chunk_index: int
    total_chunks: int
    text: str
    char_count: int
    word_count: int
    section_headers: List[str]
    chunk_hash: str
    created_at: str


class SemanticChunker:
    """
    Production-grade semantic chunker with section awareness.
    """
    
    def __init__(self, max_chars: int = 1200, overlap_chars: int = 200, 
                 min_chunk_chars: int = 100):
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars
        self.min_chunk_chars = min_chunk_chars
        
        # Legal document section patterns
        self.section_patterns = [
            r'^Section\s+\d+\.?\d*',
            r'^SECTION\s+\d+\.?\d*',
            r'^Article\s+\d+\.?\d*',
            r'^ARTICLE\s+[IVXLCDM]+',
            r'^§\s*\d+\.?\d*',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z][A-Z\s]+:$',
        ]
        
    def _extract_section_headers(self, text: str) -> List[str]:
        """Extract section headers from text."""
        headers = []
        lines = text.split('\n')
        
        for line in lines[:10]:
            line = line.strip()
            if not line:
                continue
            
            for pattern in self.section_patterns:
                if re.match(pattern, line, re.IGNORECASE):
                    headers.append(line[:100])
                    break
                    
        return headers
    
    def _compute_chunk_hash(self, text: str) -> str:
        return hashlib.md5(text.encode('utf-8')).hexdigest()[:12]
    
    def _count_words(self, text: str) -> int:
        return len(text.split())
    
    def _find_semantic_boundary(self, text: str, target_pos: int) -> int:
        if target_pos >= len(text):
            return len(text)
        
        window = 200
        start = max(0, target_pos - window)
        end = min(len(text), target_pos + window)
        search_text = text[start:end]
        
        para_match = re.search(r'\n\s*\n', search_text)
        if para_match:
            return start + para_match.start()
        
        sent_match = re.search(r'[.!?]\s+[A-Z]', search_text)
        if sent_match:
            return start + sent_match.start() + 1
        
        nl_match = re.search(r'\n', search_text)
        if nl_match:
            return start + nl_match.start()
        
        space_match = re.search(r'\s', search_text)
        if space_match:
            return start + space_match.start()
        
        return target_pos
    
    def _validate_chunk(self, chunk_text: str) -> bool:
        if len(chunk_text.strip()) < self.min_chunk_chars:
            return False
        
        garbage_ratio = sum(1 for c in chunk_text if not c.isprintable() and c not in '\n\r\t')
        if garbage_ratio / max(len(chunk_text), 1) > 0.1:
            return False
        
        return True
    
    def chunk_text(self, text: str, preserve_headers: bool = True) -> List[str]:
        if not text or len(text) < self.min_chunk_chars:
            return []
        
        headers = self._extract_section_headers(text)
        header_prefix = '\n'.join(headers) + '\n\n' if preserve_headers and headers else ''
        
        chunks = []
        current_pos = 0
        text_len = len(text)
        
        while current_pos < text_len:
            target_end = min(current_pos + self.max_chars, text_len)
            
            if target_end < text_len:
                boundary = self._find_semantic_boundary(text, target_end)
            else:
                boundary = text_len
            
            chunk_text = text[current_pos:boundary].strip()
            
            if preserve_headers and headers:
                chunk_text = header_prefix + chunk_text
            
            if self._validate_chunk(chunk_text):
                chunks.append(chunk_text)
            
            if boundary >= text_len:
                break
                
            overlap_start = max(current_pos, boundary - self.overlap_chars)
            current_pos = self._find_semantic_boundary(text, overlap_start)
        
        return chunks
    
    # ===== THIS METHOD MUST BE INSIDE THE CLASS (FIXED INDENTATION) =====
    @handle_errors(default_return=[])
    def chunk_document(self, doc_id: str, doc_title: str, effective_date: str, 
                    text: str) -> List[Chunk]:
        """
        Chunk a single document and return structured Chunk objects.
        """
        # Extract ONLY the actual text content
        text_start = text.find('[TEXT CONTENT]')
        if text_start != -1:
            actual_text = text[text_start + 13:]
            actual_text = actual_text.lstrip('\n')
        else:
            actual_text = text
        
        # Remove any remaining metadata markers
        actual_text = re.sub(r'\[DOCUMENT METADATA\].*?\[END METADATA\]\s*', '', actual_text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        actual_text = re.sub(r'\n\s*\n', '\n\n', actual_text)
        
        if len(actual_text.strip()) < 100:
            logger.warning(f"Very little text extracted for {doc_id}")
            return []
        
        # Generate chunks from ACTUAL text
        raw_chunks = self.chunk_text(actual_text)
        
        if not raw_chunks:
            logger.warning(f"No valid chunks generated for {doc_id}")
            return []
        
        # Create structured chunk objects
        chunks = []
        total = len(raw_chunks)
        
        for i, chunk_text in enumerate(raw_chunks):
            chunk = Chunk(
                chunk_id=f"{doc_id}_chunk_{i+1:03d}",
                doc_id=doc_id,
                effective_date=effective_date,
                chunk_index=i,
                total_chunks=total,
                text=chunk_text,
                char_count=len(chunk_text),
                word_count=self._count_words(chunk_text),
                section_headers=self._extract_section_headers(chunk_text),
                chunk_hash=self._compute_chunk_hash(chunk_text),
                created_at=datetime.now().isoformat()
            )
            chunks.append(chunk)
        
        logger.debug(f"{doc_id}: {len(chunks)} chunks generated")
        return chunks


def process_parsed_file(txt_path: Path) -> Tuple[str, str, str]:
    """Extract metadata and text from parsed file."""
    with open(txt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    doc_id_match = re.search(r'DOC_ID:\s*(\S+)', content)
    date_match = re.search(r'EFFECTIVE_DATE:\s*(\S+)', content)
    
    doc_id = doc_id_match.group(1) if doc_id_match else txt_path.stem
    effective_date = date_match.group(1) if date_match else 'unknown'
    
    return doc_id, effective_date, content


def main():
    """Main entry point for chunking phase."""
    logger.info("=" * 60)
    logger.info("Phase 2: Production Semantic Chunker")
    logger.info("=" * 60)
    
    chunker = SemanticChunker(
        max_chars=config.chunking.max_chars,
        overlap_chars=config.chunking.overlap_chars,
        min_chunk_chars=config.chunking.min_chunk_chars
    )
    
    txt_files = list(config.paths.processed_texts_dir.glob("*.txt"))
    
    if not txt_files:
        logger.error("No processed text files found")
        return 1
    
    logger.info(f"Found {len(txt_files)} processed documents")
    
    all_chunks = []
    doc_stats = []
    
    with Timer("Chunking all documents"):
        for txt_path in tqdm(txt_files, desc="Chunking"):
            try:
                doc_id, effective_date, content = process_parsed_file(txt_path)
                chunks = chunker.chunk_document(doc_id, doc_id, effective_date, content)
                
                if chunks:
                    all_chunks.extend([asdict(c) for c in chunks])
                    doc_stats.append({
                        'doc_id': doc_id,
                        'chunks': len(chunks),
                        'total_chars': sum(c.char_count for c in chunks),
                        'total_words': sum(c.word_count for c in chunks)
                    })
            except Exception as e:
                logger.error(f"Failed to chunk {txt_path.name}: {e}")
    
    config.paths.chunks_dir.mkdir(parents=True, exist_ok=True)
    chunks_path = config.paths.chunks_dir / "clauses.json"
    safe_json_dump(all_chunks, chunks_path)
    
    stats = {
        'chunked_at': datetime.now().isoformat(),
        'total_documents': len(txt_files),
        'successful_documents': len(doc_stats),
        'total_chunks': len(all_chunks),
        'config': {
            'max_chars': chunker.max_chars,
            'overlap_chars': chunker.overlap_chars,
            'min_chunk_chars': chunker.min_chunk_chars
        },
        'document_stats': doc_stats
    }
    
    stats_path = config.paths.chunks_dir / "chunking_stats.json"
    safe_json_dump(stats, stats_path)
    
    logger.info(f"Created {len(all_chunks)} chunks from {len(doc_stats)} documents")
    logger.info(f"Chunks: {chunks_path}")
    logger.info(f"Stats: {stats_path}")
    
    if doc_stats:
        avg_chunks = sum(d['chunks'] for d in doc_stats) / len(doc_stats)
        logger.info(f"Average chunks per document: {avg_chunks:.1f}")
    
    return 0


if __name__ == "__main__":
    exit(main())