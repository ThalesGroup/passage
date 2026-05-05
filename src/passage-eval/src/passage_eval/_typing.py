# MIT License - Copyright (c) 2026 THALES
"""Shared lightweight types for passage_eval."""

from __future__ import annotations

from typing import Sequence, Tuple, Union

import numpy as np

PathLike = Union[np.ndarray, Sequence[Tuple[int, int]]]


def _as_xy_array(path: PathLike) -> np.ndarray:
    """Return path as an ``(N, 2)`` float64 ndarray of (row, col) coordinates.

    Empty paths return an array of shape ``(0, 2)``.
    """
    if path is None:
        return np.zeros((0, 2), dtype=np.float64)
    arr = np.asarray(path, dtype=np.float64)
    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected path of shape (N, 2); got shape {arr.shape}.")
    return arr
