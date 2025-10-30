"""
Unified logging configuration for Urartu.

This module provides a pre-configured logger that automatically sets up
unbuffered, immediately-flushed output for both local and Slurm execution.

Usage:
    from urartu import get_logger
    
    logger = get_logger(__name__)
    logger.info("This will be immediately visible in logs!")
"""

import logging
import sys
import os
from typing import Optional

# Global flag to ensure configuration happens only once
_CONFIGURED = False
_LOGGER_CACHE = {}
_TQDM_CONFIGURED = False


def configure_tqdm_once():
    """
    Configure tqdm to write to stdout instead of stderr.
    This ensures progress bars appear in .out files alongside other logs.
    """
    global _TQDM_CONFIGURED
    
    if _TQDM_CONFIGURED:
        return
    
    try:
        import tqdm as tqdm_module
        # Monkey-patch tqdm to default to stdout
        original_tqdm_init = tqdm_module.tqdm.__init__
        
        def patched_init(self, *args, **kwargs):
            # If 'file' is not explicitly set, use stdout
            if 'file' not in kwargs:
                kwargs['file'] = sys.stdout
            original_tqdm_init(self, *args, **kwargs)
        
        tqdm_module.tqdm.__init__ = patched_init
        _TQDM_CONFIGURED = True
    except ImportError:
        # tqdm not installed, skip configuration
        pass


def configure_logging_once():
    """
    Configure logging to output to stdout/stderr with immediate flushing.
    - INFO, DEBUG, WARNING logs go to stdout (.out files)
    - ERROR, CRITICAL logs go to stderr (.err files)
    
    Also configures tqdm to write to stdout for consistency.
    
    This function is idempotent - it only configures once even if called multiple times.
    """
    global _CONFIGURED
    
    if _CONFIGURED:
        return
    
    # Force unbuffered output at the environment level
    os.environ['PYTHONUNBUFFERED'] = '1'
    
    # Force unbuffered output for existing streams
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(line_buffering=True)
    
    # Create custom handler that flushes immediately
    class ImmediateFlushHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    
    # Remove all existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Handler for stdout: INFO, DEBUG, WARNING
    stdout_handler = ImmediateFlushHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.addFilter(lambda record: record.levelno < logging.ERROR)  # Only < ERROR
    root_logger.addHandler(stdout_handler)
    
    # Handler for stderr: ERROR, CRITICAL
    stderr_handler = ImmediateFlushHandler(sys.stderr)
    stderr_handler.setFormatter(formatter)
    stderr_handler.setLevel(logging.ERROR)  # Only ERROR and above
    root_logger.addHandler(stderr_handler)
    
    # Also configure logging for all child loggers
    logging.captureWarnings(True)
    
    # Configure tqdm to use stdout
    configure_tqdm_once()
    
    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a pre-configured logger that automatically flushes output.
    
    This logger is configured for both local and Slurm execution, ensuring
    all output is immediately visible in log files.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
              If None, returns the root logger
    
    Returns:
        A configured Logger instance
    
    Example:
        from urartu import get_logger
        
        logger = get_logger(__name__)
        logger.info("This message appears immediately!")
    """
    # Configure logging on first use
    configure_logging_once()
    
    # Cache loggers to avoid recreating them
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    
    logger = logging.getLogger(name)
    _LOGGER_CACHE[name] = logger
    
    return logger


# For backward compatibility
configure_logging = configure_logging_once

