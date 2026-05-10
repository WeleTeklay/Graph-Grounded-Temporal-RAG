#src/logger.py
"""
Centralized logging configuration for the entire project.
Usage: from src.logger import get_logger; logger = get_logger(__name__)
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path

def get_logger(name: str = "GraphRAG", level: str = "INFO") -> logging.Logger:
    """
    Returns a configured logger with both console and file handlers.
    
    Args:
        name: Logger name (usually __name__ from calling module)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Configured logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.handlers:
        return logger
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler - INFO and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # File handler - DEBUG and above (more detailed for debugging)
    log_file = log_dir / f"graphrag_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(funcName)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


class ProgressTracker:
    """Context manager for tracking progress of long operations."""
    
    def __init__(self, logger, operation_name: str, total_items: int = None):
        self.logger = logger
        self.operation = operation_name
        self.total = total_items
        self.current = 0
    
    def __enter__(self):
        self.logger.info(f"Starting: {self.operation}")
        return self
    
    def update(self, n: int = 1, message: str = None):
        self.current += n
        if self.total:
            pct = (self.current / self.total) * 100
            msg = f"{self.operation}: {self.current}/{self.total} ({pct:.1f}%)"
            if message:
                msg += f" - {message}"
            self.logger.debug(msg)
    
    def __exit__(self, *args):
        self.logger.info(f"Completed: {self.operation} (processed {self.current} items)")


# Singleton logger instance for quick imports
_default_logger = None

def get_default_logger():
    global _default_logger
    if _default_logger is None:
        _default_logger = get_logger("GraphRAG")
    return _default_logger