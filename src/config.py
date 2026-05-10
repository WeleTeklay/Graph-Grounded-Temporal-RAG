#src/config.py
"""
Production-grade configuration management with validation, hot-reload, and multiple sources.
"""

import os
import json
import yaml
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List
from functools import lru_cache
from threading import RLock
import logging

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration Exceptions
# ============================================================================

class ConfigError(Exception):
    """Base configuration error."""
    pass


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""
    pass


class ConfigNotFoundError(ConfigError):
    """Configuration file not found."""
    pass


# ============================================================================
# Path Configuration
# ============================================================================

@dataclass
class PathConfig:
    """Project path configuration."""
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(default=None)
    raw_pdfs_dir: Path = field(default=None)
    processed_texts_dir: Path = field(default=None)
    chunks_dir: Path = field(default=None)
    extracted_dir: Path = field(default=None)
    vectors_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    cache_dir: Path = field(default=None)
    
    def __post_init__(self):
        """Initialize derived paths and create directories."""
        if self.data_dir is None:
            self.data_dir = self.project_root / "data"
        
        self.raw_pdfs_dir = self.raw_pdfs_dir or self.data_dir / "raw_pdfs"
        self.processed_texts_dir = self.processed_texts_dir or self.data_dir / "processed_texts"
        self.chunks_dir = self.chunks_dir or self.data_dir / "chunks"
        self.extracted_dir = self.extracted_dir or self.data_dir / "extracted"
        self.vectors_dir = self.vectors_dir or self.data_dir / "vectors"
        self.logs_dir = self.logs_dir or self.project_root / "logs"
        self.cache_dir = self.cache_dir or self.data_dir / "cache"
    
    def create_directories(self) -> None:
        """Create all required directories."""
        dirs = [
            self.raw_pdfs_dir,
            self.processed_texts_dir,
            self.chunks_dir,
            self.extracted_dir,
            self.vectors_dir,
            self.logs_dir,
            self.cache_dir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def validate(self) -> List[str]:
        """Validate paths and return list of issues."""
        issues = []
        
        if not self.project_root.exists():
            issues.append(f"Project root does not exist: {self.project_root}")
        
        if not self.raw_pdfs_dir.exists():
            issues.append(f"Raw PDFs directory missing: {self.raw_pdfs_dir}")
        
        return issues


# ============================================================================
# Component Configurations
# ============================================================================

@dataclass
class Neo4jConfig:
    """Neo4j graph database configuration."""
    uri: str = "bolt://localhost:7687"
    user: str = "neo4j"
    password: str = ""
    database: str = "neo4j"
    max_connection_pool_size: int = 10
    connection_timeout: int = 30
    max_retry_time: int = 60
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        """Create from environment variables."""
        load_dotenv()
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=int(os.getenv("NEO4J_POOL_SIZE", "10")),
            connection_timeout=int(os.getenv("NEO4J_TIMEOUT", "30")),
            max_retry_time=int(os.getenv("NEO4J_RETRY_TIME", "60")),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if not self.uri:
            issues.append("NEO4J_URI is required")
        if not self.user:
            issues.append("NEO4J_USER is required")
        if not self.password:
            issues.append("NEO4J_PASSWORD is required (or set to empty for no auth)")
        if not self.uri.startswith(("bolt://", "neo4j://", "neo4j+s://", "bolt+s://")):
            issues.append(f"Invalid Neo4j URI scheme: {self.uri}")
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is valid."""
        return len(self.validate()) == 0


@dataclass
class OllamaConfig:
    """Ollama LLM configuration."""
    model: str = "llama3.2:3b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    max_tokens: int = 512
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    timeout: int = 120
    num_ctx: int = 4096
    
    @classmethod
    def from_env(cls) -> "OllamaConfig":
        """Create from environment variables."""
        load_dotenv()
        return cls(
            model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
            host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            max_tokens=int(os.getenv("OLLAMA_MAX_TOKENS", "512")),
            top_p=float(os.getenv("OLLAMA_TOP_P", "0.9")),
            repeat_penalty=float(os.getenv("OLLAMA_REPEAT_PENALTY", "1.1")),
            timeout=int(os.getenv("OLLAMA_TIMEOUT", "120")),
            num_ctx=int(os.getenv("OLLAMA_NUM_CTX", "4096")),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if not self.model:
            issues.append("OLLAMA_MODEL is required")
        if not 0 <= self.temperature <= 2:
            issues.append(f"Temperature must be 0-2, got {self.temperature}")
        if self.max_tokens < 1:
            issues.append(f"max_tokens must be positive, got {self.max_tokens}")
        return issues
    
    @property
    def generate_options(self) -> Dict[str, Any]:
        """Return options dict for ollama.generate()."""
        return {
            "temperature": self.temperature,
            "num_predict": self.max_tokens,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
            "num_ctx": self.num_ctx,
        }


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""
    model_name: str = "all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 100
    normalize: bool = True
    device: str = "auto"  # auto, cpu, cuda
    
    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        """Create from environment variables."""
        load_dotenv()
        return cls(
            model_name=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            dimension=int(os.getenv("EMBEDDING_DIMENSION", "384")),
            batch_size=int(os.getenv("EMBEDDING_BATCH_SIZE", "100")),
            normalize=os.getenv("EMBEDDING_NORMALIZE", "true").lower() == "true",
            device=os.getenv("EMBEDDING_DEVICE", "auto"),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if not self.model_name:
            issues.append("EMBEDDING_MODEL is required")
        if self.dimension < 1:
            issues.append(f"dimension must be positive, got {self.dimension}")
        if self.batch_size < 1:
            issues.append(f"batch_size must be positive, got {self.batch_size}")
        return issues


@dataclass
class ChunkingConfig:
    """Document chunking configuration."""
    max_chars: int = 1200
    overlap_chars: int = 200
    min_chunk_chars: int = 100
    preserve_sections: bool = True
    split_on_headers: bool = True
    
    @classmethod
    def from_env(cls) -> "ChunkingConfig":
        """Create from environment variables."""
        load_dotenv()
        return cls(
            max_chars=int(os.getenv("CHUNK_MAX_CHARS", "1200")),
            overlap_chars=int(os.getenv("CHUNK_OVERLAP", "200")),
            min_chunk_chars=int(os.getenv("CHUNK_MIN_CHARS", "100")),
            preserve_sections=os.getenv("CHUNK_PRESERVE_SECTIONS", "true").lower() == "true",
            split_on_headers=os.getenv("CHUNK_SPLIT_HEADERS", "true").lower() == "true",
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if self.max_chars < self.min_chunk_chars:
            issues.append(f"max_chars ({self.max_chars}) must be >= min_chunk_chars ({self.min_chunk_chars})")
        if self.overlap_chars >= self.max_chars:
            issues.append(f"overlap_chars ({self.overlap_chars}) must be < max_chars ({self.max_chars})")
        return issues


@dataclass
class RetrievalConfig:
    """Retrieval configuration."""
    top_k: int = 10
    rerank_top_k: int = 3
    final_top_k: int = 2
    hybrid_alpha: float = 0.6
    similarity_threshold: float = 0.25
    enable_reranking: bool = True
    enable_graph: bool = True
    max_context_length: int = 2000
    
    @classmethod
    def from_env(cls) -> "RetrievalConfig":
        """Create from environment variables."""
        load_dotenv()
        return cls(
            top_k=int(os.getenv("RETRIEVAL_TOP_K", "20")),
            rerank_top_k=int(os.getenv("RETRIEVAL_RERANK_TOP_K", "5")),
            final_top_k=int(os.getenv("RETRIEVAL_FINAL_TOP_K", "3")),
            hybrid_alpha=float(os.getenv("RETRIEVAL_ALPHA", "0.7")),
            similarity_threshold=float(os.getenv("RETRIEVAL_THRESHOLD", "0.3")),
            enable_reranking=os.getenv("RETRIEVAL_ENABLE_RERANK", "true").lower() == "true",
            enable_graph=os.getenv("RETRIEVAL_ENABLE_GRAPH", "true").lower() == "true",
            max_context_length=int(os.getenv("RETRIEVAL_MAX_CONTEXT", "3000")),
        )
    
    def validate(self) -> List[str]:
        """Validate configuration."""
        issues = []
        if self.top_k < self.rerank_top_k:
            issues.append(f"top_k ({self.top_k}) must be >= rerank_top_k ({self.rerank_top_k})")
        if self.rerank_top_k < self.final_top_k:
            issues.append(f"rerank_top_k ({self.rerank_top_k}) must be >= final_top_k ({self.final_top_k})")
        if not 0 <= self.hybrid_alpha <= 1:
            issues.append(f"hybrid_alpha must be 0-1, got {self.hybrid_alpha}")
        return issues


# ============================================================================
# Master Configuration
# ============================================================================

@dataclass
class AppConfig:
    """Master application configuration."""
    neo4j: Neo4jConfig = field(default_factory=Neo4jConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chunking: ChunkingConfig = field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    paths: PathConfig = field(default_factory=PathConfig)
    
    # Metadata
    version: str = "2.0.0"
    environment: str = "development"  # development, staging, production
    
    @classmethod
    def from_env(cls) -> "AppConfig":
        """Create full configuration from environment variables."""
        return cls(
            neo4j=Neo4jConfig.from_env(),
            ollama=OllamaConfig.from_env(),
            embedding=EmbeddingConfig.from_env(),
            chunking=ChunkingConfig.from_env(),
            retrieval=RetrievalConfig.from_env(),
            paths=PathConfig(),
            environment=os.getenv("APP_ENV", "development"),
        )
    
    @classmethod
    def from_yaml(cls, path: Path) -> "AppConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            raise ConfigNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        return cls(
            neo4j=Neo4jConfig(**data.get('neo4j', {})),
            ollama=OllamaConfig(**data.get('ollama', {})),
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            chunking=ChunkingConfig(**data.get('chunking', {})),
            retrieval=RetrievalConfig(**data.get('retrieval', {})),
            paths=PathConfig(**data.get('paths', {})),
            version=data.get('version', '2.0.0'),
            environment=data.get('environment', 'development'),
        )
    
    @classmethod
    def from_json(cls, path: Path) -> "AppConfig":
        """Load configuration from JSON file."""
        if not path.exists():
            raise ConfigNotFoundError(f"Config file not found: {path}")
        
        with open(path, 'r') as f:
            data = json.load(f)
        
        return cls(
            neo4j=Neo4jConfig(**data.get('neo4j', {})),
            ollama=OllamaConfig(**data.get('ollama', {})),
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            chunking=ChunkingConfig(**data.get('chunking', {})),
            retrieval=RetrievalConfig(**data.get('retrieval', {})),
            paths=PathConfig(**data.get('paths', {})),
            version=data.get('version', '2.0.0'),
            environment=data.get('environment', 'development'),
        )
    
    def validate(self) -> List[str]:
        """Validate entire configuration."""
        issues = []
        issues.extend(self.neo4j.validate())
        issues.extend(self.ollama.validate())
        issues.extend(self.embedding.validate())
        issues.extend(self.chunking.validate())
        issues.extend(self.retrieval.validate())
        issues.extend(self.paths.validate())
        return issues
    
    def is_valid(self) -> bool:
        """Check if configuration is fully valid."""
        return len(self.validate()) == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def to_yaml(self, path: Path) -> None:
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def to_json(self, path: Path) -> None:
        """Save configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    def create_directories(self) -> None:
        """Create all required directories."""
        self.paths.create_directories()
    
    def log_config(self) -> None:
        """Log current configuration (excluding secrets)."""
        safe_config = self.to_dict()
        if 'neo4j' in safe_config:
            safe_config['neo4j']['password'] = '***REDACTED***'
        logger.info(f"Configuration loaded: {safe_config}")


# ============================================================================
# Singleton Configuration Manager
# ============================================================================

class ConfigManager:
    """
    Thread-safe singleton configuration manager with hot-reload support.
    """
    
    _instance: Optional["ConfigManager"] = None
    _lock: RLock = RLock()
    
    def __new__(cls) -> "ConfigManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._config = None
                    cls._instance._load_config()
        return cls._instance
    
    def _load_config(self) -> None:
        """Load configuration from environment and files."""
        # Try YAML first
        yaml_path = Path("config.yaml")
        json_path = Path("config.json")
        
        if yaml_path.exists():
            self._config = AppConfig.from_yaml(yaml_path)
            logger.info(f"Loaded config from {yaml_path}")
        elif json_path.exists():
            self._config = AppConfig.from_json(json_path)
            logger.info(f"Loaded config from {json_path}")
        else:
            self._config = AppConfig.from_env()
            logger.info("Loaded config from environment variables")
        
        # Validate
        issues = self._config.validate()
        if issues:
            for issue in issues:
                logger.warning(f"Config issue: {issue}")
            if self._config.environment == "production":
                raise ConfigValidationError(f"Invalid production config: {issues}")
        
        # Create directories
        self._config.create_directories()
    
    @property
    def config(self) -> AppConfig:
        """Get current configuration."""
        return self._config
    
    def reload(self) -> None:
        """Hot-reload configuration."""
        with self._lock:
            self._load_config()
            logger.info("Configuration reloaded")
    
    def get(self) -> AppConfig:
        """Get configuration (alias for config property)."""
        return self.config


# ============================================================================
# Global Access
# ============================================================================

_config_manager: Optional[ConfigManager] = None


def get_config() -> AppConfig:
    """Get the global application configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager.config


def reload_config() -> None:
    """Hot-reload configuration."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    else:
        _config_manager.reload()


# Singleton instance for backward compatibility
config = get_config()