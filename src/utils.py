#src/utils.py
"""
Production-grade utility functions with retry logic, atomic operations, and metrics.
"""

import json
import os
import time
import hashlib
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable, TypeVar, Union, Iterator
from functools import wraps
from datetime import datetime, timedelta
from threading import Lock
from contextlib import contextmanager
import traceback

from src.logger import get_logger

logger = get_logger(__name__)

T = TypeVar('T')


# ============================================================================
# Decorators
# ============================================================================

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: tuple = (Exception,),
    on_failure: Any = None
) -> Callable:
    """
    Decorator to retry a function on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts
        delay: Initial delay between retries (seconds)
        backoff: Multiplier for delay after each attempt
        exceptions: Tuple of exceptions to catch
        on_failure: Value to return if all retries fail
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay
            last_exception = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt < max_attempts:
                        logger.warning(
                            f"{func.__name__} failed (attempt {attempt}/{max_attempts}): {e}. "
                            f"Retrying in {current_delay:.1f}s..."
                        )
                        time.sleep(current_delay)
                        current_delay *= backoff
                    else:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {e}"
                        )
            
            if on_failure is not None:
                return on_failure
            if last_exception:
                raise last_exception
            return None
        return wrapper
    return decorator


def handle_errors(
    default_return: Any = None,
    reraise: bool = False,
    log_level: str = "error"
) -> Callable:
    """
    Decorator to catch and log exceptions gracefully.
    
    Args:
        default_return: Value to return on error
        reraise: If True, re-raise exception after logging
        log_level: "debug", "info", "warning", "error"
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                log_func = getattr(logger, log_level)
                log_func(f"Error in {func.__name__}: {e}")
                logger.debug(traceback.format_exc())
                
                if reraise:
                    raise
                return default_return
        return wrapper
    return decorator


def timeout(seconds: float) -> Callable:
    """
    Decorator to timeout a function after specified seconds.
    
    Note: Uses signals on Unix, threading on Windows.
    """
    import signal
    from functools import partial
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            def timeout_handler(signum, frame):
                raise TimeoutError(f"{func.__name__} timed out after {seconds}s")
            
            # Set signal handler
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(seconds))
            
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
            
            return result
        return wrapper
    return decorator


def memoize(maxsize: int = 128) -> Callable:
    """
    Memoization decorator with LRU cache.
    """
    def decorator(func: Callable) -> Callable:
        cache: Dict[str, Any] = {}
        cache_order: List[str] = []
        lock = Lock()
        
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key
            key_parts = [str(arg) for arg in args]
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
            
            with lock:
                if key in cache:
                    # Move to end (LRU)
                    cache_order.remove(key)
                    cache_order.append(key)
                    return cache[key]
                
                result = func(*args, **kwargs)
                
                cache[key] = result
                cache_order.append(key)
                
                # Evict oldest if over maxsize
                if len(cache_order) > maxsize:
                    oldest = cache_order.pop(0)
                    del cache[oldest]
                
                return result
        
        def clear_cache():
            with lock:
                cache.clear()
                cache_order.clear()
        
        wrapper.clear_cache = clear_cache
        return wrapper
    return decorator


# ============================================================================
# File Operations
# ============================================================================

@retry(max_attempts=3, delay=0.5, exceptions=(IOError, OSError))
def safe_file_read(
    filepath: Union[str, Path],
    encodings: List[str] = None
) -> Optional[str]:
    """
    Read file with automatic encoding detection and retry logic.
    
    Args:
        filepath: Path to file
        encodings: List of encodings to try
    
    Returns:
        File contents as string, or None if all encodings fail
    """
    filepath = Path(filepath)
    
    if encodings is None:
        encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(filepath, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return None
    
    logger.error(f"Could not decode {filepath} with any encoding")
    return None


@retry(max_attempts=3, delay=0.5)
def safe_file_write(
    filepath: Union[str, Path],
    content: str,
    encoding: str = 'utf-8',
    atomic: bool = True
) -> bool:
    """
    Write content to file with atomic write option.
    
    Args:
        filepath: Destination path
        content: String content to write
        encoding: Text encoding
        atomic: If True, write to temp file then rename (prevents corruption)
    
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    if atomic:
        # Write to temp file then rename
        fd, temp_path = tempfile.mkstemp(
            dir=filepath.parent,
            prefix=f".{filepath.name}.",
            suffix=".tmp"
        )
        try:
            with os.fdopen(fd, 'w', encoding=encoding) as f:
                f.write(content)
            shutil.move(temp_path, filepath)
            return True
        except Exception as e:
            logger.error(f"Atomic write failed for {filepath}: {e}")
            try:
                os.unlink(temp_path)
            except:
                pass
            return False
    else:
        try:
            with open(filepath, 'w', encoding=encoding) as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Write failed for {filepath}: {e}")
            return False


@retry(max_attempts=3, delay=0.5)
def safe_json_load(
    filepath: Union[str, Path],
    default: Any = None,
    validate_schema: Optional[Dict] = None
) -> Any:
    """
    Safely load JSON file with validation.
    
    Args:
        filepath: Path to JSON file
        default: Value to return on failure
        validate_schema: Optional schema dict for validation
    
    Returns:
        Parsed JSON data or default
    """
    filepath = Path(filepath)
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if validate_schema:
            if not validate_json_schema(data, validate_schema):
                logger.error(f"JSON schema validation failed for {filepath}")
                return default
        
        return data
        
    except FileNotFoundError:
        logger.error(f"JSON file not found: {filepath}")
        return default
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in {filepath}: {e}")
        return default


@retry(max_attempts=3, delay=0.5)
def safe_json_dump(
    data: Any,
    filepath: Union[str, Path],
    indent: int = 2,
    atomic: bool = True,
    ensure_ascii: bool = False
) -> bool:
    """
    Safely save data as JSON with atomic write.
    
    Args:
        data: Data to serialize
        filepath: Destination path
        indent: JSON indentation
        atomic: Use atomic write
        ensure_ascii: Escape non-ASCII characters
    
    Returns:
        True if successful, False otherwise
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        content = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
        return safe_file_write(filepath, content, atomic=atomic)
    except Exception as e:
        logger.error(f"Failed to serialize JSON for {filepath}: {e}")
        return False


def safe_jsonl_append(
    data: Dict[str, Any],
    filepath: Union[str, Path]
) -> bool:
    """
    Append a JSON object to a JSONL file.
    
    Args:
        data: Dictionary to append
        filepath: Path to JSONL file
    
    Returns:
        True if successful
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
        return True
    except Exception as e:
        logger.error(f"Failed to append to {filepath}: {e}")
        return False


def stream_jsonl(
    filepath: Union[str, Path]
) -> Iterator[Dict[str, Any]]:
    """
    Stream JSONL file line by line (memory efficient).
    
    Args:
        filepath: Path to JSONL file
    
    Yields:
        Parsed JSON objects
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        return
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON at {filepath}:{line_num}: {e}")
                continue


# ============================================================================
# Validation
# ============================================================================

def validate_json_schema(data: Any, schema: Dict) -> bool:
    """
    Validate JSON data against a simple schema.
    
    Schema format:
    {
        "field": {"type": "str", "required": True},
        "field2": {"type": "int", "min": 0},
    }
    """
    if not isinstance(data, dict):
        return False
    
    for field, rules in schema.items():
        if rules.get('required', False) and field not in data:
            logger.error(f"Required field missing: {field}")
            return False
        
        if field in data:
            value = data[field]
            expected_type = rules.get('type')
            
            if expected_type == 'str' and not isinstance(value, str):
                logger.error(f"Field {field} expected str, got {type(value)}")
                return False
            elif expected_type == 'int' and not isinstance(value, int):
                logger.error(f"Field {field} expected int, got {type(value)}")
                return False
            elif expected_type == 'float' and not isinstance(value, (int, float)):
                logger.error(f"Field {field} expected float, got {type(value)}")
                return False
            elif expected_type == 'list' and not isinstance(value, list):
                logger.error(f"Field {field} expected list, got {type(value)}")
                return False
            elif expected_type == 'dict' and not isinstance(value, dict):
                logger.error(f"Field {field} expected dict, got {type(value)}")
                return False
            
            if 'min' in rules and value < rules['min']:
                logger.error(f"Field {field} value {value} < min {rules['min']}")
                return False
            if 'max' in rules and value > rules['max']:
                logger.error(f"Field {field} value {value} > max {rules['max']}")
                return False
            if 'pattern' in rules and not isinstance(value, str):
                import re
                if not re.match(rules['pattern'], value):
                    logger.error(f"Field {field} does not match pattern")
                    return False
    
    return True


def validate_manifest(manifest_path: Path) -> bool:
    """
    Validate document_manifest.csv structure.
    """
    import pandas as pd
    
    try:
        df = pd.read_csv(manifest_path)
        required_cols = {'doc_id', 'doc_title', 'effective_date', 'supersedes_doc_id'}
        
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            logger.error(f"Manifest missing columns: {missing}")
            return False
        
        # Validate date format
        date_pattern = r'^\d{4}-\d{2}-\d{2}$'
        invalid_dates = df[~df['effective_date'].astype(str).str.match(date_pattern)]
        if not invalid_dates.empty:
            logger.error(f"Invalid date format: {invalid_dates['doc_id'].tolist()}")
            return False
        
        # Validate supersedes references
        valid_ids = set(df['doc_id'])
        for _, row in df.iterrows():
            supersedes = row['supersedes_doc_id']
            if pd.notna(supersedes) and supersedes and supersedes != 'None':
                if supersedes not in valid_ids:
                    logger.error(f"supersedes_doc_id '{supersedes}' not found in doc_ids")
                    return False
        
        logger.info(f"Manifest validated: {len(df)} documents")
        return True
        
    except Exception as e:
        logger.error(f"Failed to validate manifest: {e}")
        return False


# ============================================================================
# Path Utilities
# ============================================================================

def ensure_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).parent.parent


def get_relative_path(path: Union[str, Path], base: Path = None) -> Path:
    """Get path relative to project root."""
    if base is None:
        base = get_project_root()
    return Path(path).relative_to(base)


def clean_old_files(directory: Path, pattern: str, days_old: int) -> int:
    """
    Delete files older than specified days.
    
    Returns:
        Number of files deleted
    """
    cutoff = datetime.now() - timedelta(days=days_old)
    deleted = 0
    
    for filepath in directory.glob(pattern):
        if filepath.is_file():
            mtime = datetime.fromtimestamp(filepath.stat().st_mtime)
            if mtime < cutoff:
                try:
                    filepath.unlink()
                    deleted += 1
                    logger.debug(f"Deleted old file: {filepath}")
                except Exception as e:
                    logger.warning(f"Failed to delete {filepath}: {e}")
    
    return deleted


# ============================================================================
# Timer & Metrics
# ============================================================================

class Timer:
    """
    Context manager for timing operations with metrics collection.
    """
    
    _metrics: Dict[str, List[float]] = {}
    _lock = Lock()
    
    def __init__(self, name: str = "Operation", log: bool = True, collect: bool = True):
        self.name = name
        self.log = log
        self.collect = collect
        self.start: Optional[datetime] = None
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        self.start = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.elapsed = (datetime.now() - self.start).total_seconds()
        
        if self.log:
            logger.info(f"{self.name} completed in {self.elapsed:.2f}s")
        
        if self.collect:
            with self._lock:
                if self.name not in self._metrics:
                    self._metrics[self.name] = []
                self._metrics[self.name].append(self.elapsed)
    
    @classmethod
    def get_metrics(cls, name: str = None) -> Dict[str, Dict[str, float]]:
        """Get timing metrics statistics."""
        with cls._lock:
            if name:
                metrics = {name: cls._metrics.get(name, [])}
            else:
                metrics = cls._metrics.copy()
        
        result = {}
        for op_name, times in metrics.items():
            if times:
                result[op_name] = {
                    'count': len(times),
                    'total': sum(times),
                    'mean': sum(times) / len(times),
                    'min': min(times),
                    'max': max(times)
                }
        
        return result
    
    @classmethod
    def clear_metrics(cls):
        """Clear all collected metrics."""
        with cls._lock:
            cls._metrics.clear()


class ProgressTracker:
    """
    Context manager for tracking progress with ETA.
    """
    
    def __init__(self, total: int, description: str = "Processing", log_interval: int = 10):
        self.total = total
        self.description = description
        self.log_interval = log_interval
        self.current = 0
        self.start_time: Optional[datetime] = None
        self.last_log = 0
    
    def __enter__(self):
        self.start_time = datetime.now()
        logger.info(f"Starting: {self.description} ({self.total} items)")
        return self
    
    def update(self, n: int = 1):
        self.current += n
        
        # Log progress at intervals
        if self.current - self.last_log >= self.log_interval or self.current == self.total:
            self.last_log = self.current
            elapsed = (datetime.now() - self.start_time).total_seconds()
            
            if self.current > 0:
                rate = self.current / elapsed if elapsed > 0 else 0
                eta = (self.total - self.current) / rate if rate > 0 else 0
                
                logger.info(
                    f"{self.description}: {self.current}/{self.total} "
                    f"({self.current/self.total*100:.1f}%) - "
                    f"Rate: {rate:.1f}/s - ETA: {eta:.0f}s"
                )
    
    def __exit__(self, *args):
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"Completed: {self.description} ({self.current} items in {elapsed:.1f}s)")


# ============================================================================
# Hash & Security
# ============================================================================

def compute_file_hash(filepath: Union[str, Path], algorithm: str = "sha256") -> str:
    """Compute hash of a file."""
    filepath = Path(filepath)
    hasher = hashlib.new(algorithm)
    
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    
    return hasher.hexdigest()


def compute_text_hash(text: str, algorithm: str = "md5") -> str:
    """Compute hash of text."""
    return hashlib.new(algorithm, text.encode('utf-8')).hexdigest()


# ============================================================================
# Miscellaneous
# ============================================================================

def chunk_list(lst: List[T], chunk_size: int) -> Iterator[List[T]]:
    """Split a list into chunks."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten a nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def safe_import(module_name: str, fallback: Any = None) -> Any:
    """Safely import a module with fallback."""
    try:
        return __import__(module_name, fromlist=[''])
    except ImportError as e:
        logger.warning(f"Could not import {module_name}: {e}")
        return fallback