"""
Centralized logging configuration using Loguru for the Financial Statement Analysis POC.
"""
import sys
from pathlib import Path
from loguru import logger
from typing import Optional
from pathlib import Path


LOG_DIR = Path("logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

def setup_logger(
    log_file: str = "FS_RAG.log",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days",
    compression: str = "zip"
) -> logger:
    """
    Configure and return a loguru logger instance with file and console outputs.
    
    Args:
        log_file: Path to the log file. If None, only console logging is enabled.
        log_level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        rotation: When to rotate the log file (size or time-based)
        retention: How long to keep old log files
        compression: Compression format for rotated logs
        
    Returns:
        Configured logger instance
    """
    # Remove default logger
    logger.remove()

    # Console handler with color and formatting
    logger.add(
        LOG_DIR / log_file,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=log_level,
        rotation=rotation,
        retention=retention,
        compression=compression,
        enqueue=True,  # Thread-safe logging
    )
    
    logger.info(f"Logger initialized with level: {log_level}")
    return logger


def log_execution_time(func):
    """
    Decorator to log function execution time.
    
    Usage:
        @log_execution_time
        def my_function():
            pass
    """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        logger.info(f"Starting execution: {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            elapsed_time = time.perf_counter() - start_time
            logger.info(f"Completed {func.__name__} in {elapsed_time:.2f} seconds")
            return result
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(f"Failed {func.__name__} after {elapsed_time:.2f} seconds: {str(e)}")
            raise
    
    return wrapper


# Initialize default logger
default_logger = setup_logger()