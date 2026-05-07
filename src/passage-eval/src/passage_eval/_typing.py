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
