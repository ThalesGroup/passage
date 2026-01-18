"""Shared utilities for PASSAGE."""

import functools
import time

from loguru import logger


def _log_timing(func):
    """Decorator to log function entry, exit and execution time at debug level."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug(f"### ENTER {func.__name__}()")
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            logger.debug(f"### EXIT {func.__name__}() duration={elapsed:.4f}s")

    return wrapper
