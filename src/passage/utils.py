# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Shared utilities for PASSAGE."""

import functools
import time
from pathlib import Path

import numpy as np
import zstandard as zstd
from loguru import logger

# Default compression level for tensor blobs
TENSOR_COMPRESSION_LEVEL = 12

# File suffix for compressed tensor files
TENSOR_SUFFIX = ".npy.zst"


def encode_tensor_blob(arr: np.ndarray, level: int = TENSOR_COMPRESSION_LEVEL) -> bytes:
    """Encode a numpy array as zstd-compressed raw bytes.

    Args:
        arr: Numpy array to encode.
        level: Zstd compression level (1-22, default `TENSOR_COMPRESSION_LEVEL` = 12).

    Returns:
        Compressed bytes blob.
    """
    arr = np.ascontiguousarray(arr)
    c = zstd.ZstdCompressor(level=level)
    return c.compress(arr.tobytes())


def decode_tensor_blob(blob: bytes, shape: tuple, dtype: str = "float32") -> np.ndarray:
    """Decode a zstd-compressed tensor blob back to numpy array.

    Args:
        blob: Zstd-compressed raw bytes.
        shape: Target shape for the array.
        dtype: Numpy dtype string (default 'float32').

    Returns:
        Decoded numpy array with specified shape and dtype.
    """
    d = zstd.ZstdDecompressor()
    b = d.decompress(blob)
    return np.frombuffer(b, dtype=np.dtype(dtype)).reshape(shape)


def save_tensor(tensor: np.ndarray, path: Path, level: int = TENSOR_COMPRESSION_LEVEL) -> None:
    """Save a numpy tensor to a compressed file.

    Args:
        tensor: Numpy array to save.
        path: File path (should end with .npy.zst).
        level: Zstd compression level (default `TENSOR_COMPRESSION_LEVEL` = 12).
    """
    path = Path(path)
    blob = encode_tensor_blob(tensor, level=level)
    path.write_bytes(blob)


def load_tensor(path: Path, shape: tuple, dtype: str = "float32") -> np.ndarray:
    """Load a tensor from a compressed file.

    Args:
        path: File path to the compressed tensor.
        shape: Expected shape of the tensor.
        dtype: Numpy dtype string (default 'float32').

    Returns:
        Decoded numpy array.
    """
    path = Path(path)
    blob = path.read_bytes()
    return decode_tensor_blob(blob, shape, dtype)


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
