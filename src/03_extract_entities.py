#src/03_extract_entities.py
"""
Phase 3: Production-Grade Entity Extractor
Extracts legal entities with validation, retry logic, and caching.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

import ollama
from tqdm import tqdm

from src.logger import get_logger
from src.config import config
from src.utils import handle_errors, safe_json_load, safe_json_dump, Timer

logger = get_logger(__name__)


class ExtractionStatus(Enum):
    SUCCESS = "success"
    FAILED = "failed"
    CACHED = "cached"
    SKIPPED = "skipped"


@dataclass
class ExtractedEntities:
    """Extracted legal entities from a chunk."""
    clause_number: Optional[str] = None
    subject_party: Optional[str] = None
    action: Optional[str] = None
    amount: Optional[str] = None
    object_party: Optional[str] = None
    deadline: Optional[str] = None
    condition: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ExtractionResult:
    """Complete extraction result with metadata."""
    chunk_id: str
    doc_id: str
    effective_date: str
    entities: Dict[str, Any]
    status: str
    attempt_count: int
    response_time_ms: int
    model_used: str
    extracted_at: str
    cache_hit: bool = False
    error_message: Optional[str] = None


class EntityExtractor:
    """Production-grade entity extractor with validation and caching."""
    
    def __init__(self, model_name: str = None, cache_dir: Path = None):
        self.model_name = model_name or config.ollama.model
        # FIX: Use paths.extracted_dir instead of EXTRACTED_DIR
        self.cache_dir = cache_dir or config.paths.extracted_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_retries = 2
        self.retry_delay = 1.0
        
        self.cache = self._load_cache()
        self.prompt_template = self._build_prompt_template()
    
    def _build_prompt_template(self) -> str:
        """Build strict prompt template for legal entity extraction."""
        return """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a legal entity extraction system. Output ONLY valid JSON. No explanations.

Extract these fields from the legal clause (use null if missing):
- clause_number: Section number (e.g., "2.1")
- subject_party: Party with obligation (e.g., "Borrower")
- action: Required action (e.g., "pay fee")
- amount: Amount/percentage (e.g., "$100")
- object_party: Party receiving benefit (e.g., "Lender")
- deadline: Time constraint (e.g., "30 days")
- condition: Triggering condition

Output format: {"clause_number": "...", "subject_party": "...", "action": "...", "amount": "...", "object_party": "...", "deadline": "...", "condition": "..."}
<|eot_id|><|start_header_id|>user<|end_header_id|>
CLAUSE TEXT:
{chunk_text}

JSON OUTPUT:
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
    
    def _compute_cache_key(self, chunk_text: str) -> str:
        """Compute cache key from chunk text."""
        normalized = ' '.join(chunk_text.lower().split())[:200]
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def _load_cache(self) -> Dict[str, Dict]:
        """Load cached extractions."""
        cache_file = self.cache_dir / "extraction_cache.json"
        return safe_json_load(cache_file, default={})
    
    def _save_cache(self):
        """Save extraction cache."""
        cache_file = self.cache_dir / "extraction_cache.json"
        safe_json_dump(self.cache, cache_file)
    
    def _clean_json_response(self, response_text: str) -> Optional[Dict]:
        """Clean and parse JSON from LLM response."""
        response_text = response_text.strip()
        
        # Remove markdown code fences
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        elif response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        response_text = response_text.strip()
        
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*\}', response_text)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except json.JSONDecodeError:
                    pass
        return None
    
    def _calculate_confidence(self, data: Dict) -> float:
        """Calculate confidence based on filled fields."""
        filled = sum(1 for v in data.values() if v and v != "null")
        return min(1.0, filled / 5.0)
    
    @handle_errors(default_return=None)
    def _call_llm(self, prompt: str) -> Optional[str]:
        """Call Ollama LLM with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options={"temperature": 0.1, "num_predict": 200}
                )
                return response['response']
            except Exception as e:
                logger.warning(f"LLM attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
        return None
    
    @handle_errors(default_return=None)
    def extract_from_chunk(self, chunk: Dict) -> Optional[ExtractionResult]:
        """Extract entities from a single chunk."""
        chunk_id = chunk.get('chunk_id', 'unknown')
        doc_id = chunk.get('doc_id', 'unknown')
        effective_date = chunk.get('effective_date', 'unknown')
        chunk_text = chunk.get('text', '')
        
        if not chunk_text:
            return None
        
        chunk_text = chunk_text[:1200]
        
        # Check cache
        cache_key = self._compute_cache_key(chunk_text)
        if cache_key in self.cache:
            return ExtractionResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                effective_date=effective_date,
                entities=self.cache[cache_key],
                status=ExtractionStatus.CACHED.value,
                attempt_count=1,
                response_time_ms=0,
                model_used=self.model_name,
                extracted_at=datetime.now().isoformat(),
                cache_hit=True
            )
        
        # Build prompt and call LLM
        prompt = self.prompt_template.format(chunk_text=chunk_text)
        
        start_time = time.perf_counter()
        response_text = self._call_llm(prompt)
        response_time_ms = int((time.perf_counter() - start_time) * 1000)
        
        if not response_text:
            return ExtractionResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                effective_date=effective_date,
                entities={},
                status=ExtractionStatus.FAILED.value,
                attempt_count=self.max_retries,
                response_time_ms=response_time_ms,
                model_used=self.model_name,
                extracted_at=datetime.now().isoformat(),
                error_message="LLM returned no response"
            )
        
        # Parse response
        entities = self._clean_json_response(response_text)
        
        if entities:
            entities['confidence'] = self._calculate_confidence(entities)
            self.cache[cache_key] = entities
            
            return ExtractionResult(
                chunk_id=chunk_id,
                doc_id=doc_id,
                effective_date=effective_date,
                entities=entities,
                status=ExtractionStatus.SUCCESS.value,
                attempt_count=1,
                response_time_ms=response_time_ms,
                model_used=self.model_name,
                extracted_at=datetime.now().isoformat()
            )
        
        return ExtractionResult(
            chunk_id=chunk_id,
            doc_id=doc_id,
            effective_date=effective_date,
            entities={},
            status=ExtractionStatus.FAILED.value,
            attempt_count=1,
            response_time_ms=response_time_ms,
            model_used=self.model_name,
            extracted_at=datetime.now().isoformat(),
            error_message="Failed to parse JSON"
        )
    
    def extract_all(self, chunks: List[Dict], limit: Optional[int] = None) -> List[ExtractionResult]:
        """Extract entities from all chunks."""
        if limit:
            chunks = chunks[:limit]
            logger.info(f"Test mode: processing {limit} chunks")
        
        results = []
        
        with Timer("Entity extraction"):
            for chunk in tqdm(chunks, desc="Extracting entities"):
                result = self.extract_from_chunk(chunk)
                if result:
                    results.append(result)
                time.sleep(0.2)  # Rate limiting
        
        self._save_cache()
        return results


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("Phase 3: Production Entity Extractor")
    logger.info("=" * 60)
    
    # FIX: Use paths.chunks_dir instead of CHUNKS_DIR
    chunks_path = config.paths.chunks_dir / "clauses.json"
    if not chunks_path.exists():
        logger.error(f"Chunks not found: {chunks_path}")
        return 1
    
    chunks = safe_json_load(chunks_path, default=[])
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # TEST MODE: Process only 5 chunks
    TEST_LIMIT = 5
    logger.info(f"TEST MODE: Processing first {TEST_LIMIT} chunks only")
    
    extractor = EntityExtractor()
    results = extractor.extract_all(chunks, limit=TEST_LIMIT)
    
    # FIX: Use paths.extracted_dir instead of EXTRACTED_DIR
    config.paths.extracted_dir.mkdir(parents=True, exist_ok=True)
    output_path = config.paths.extracted_dir / "entities.jsonl"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(asdict(result)) + '\n')
    
    success = sum(1 for r in results if r.status == ExtractionStatus.SUCCESS.value)
    cached = sum(1 for r in results if r.cache_hit)
    failed = sum(1 for r in results if r.status == ExtractionStatus.FAILED.value)
    
    logger.info(f"Extraction complete")
    logger.info(f"   Success: {success}")
    logger.info(f"   Cached: {cached}")
    logger.info(f"   Failed: {failed}")
    logger.info(f"Output: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())