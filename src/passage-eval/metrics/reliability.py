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

"""Reliability and safety guardrails."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from passage_eval._typing import PathLike, _as_xy_array


def _is_connected_path(path: np.ndarray) -> bool:
    """A path connects start to goal if it is non-empty and consecutive
    cells are 8-connected (Chebyshev distance <= 1)."""
    if path.shape[0] == 0:
        return False
    if path.shape[0] == 1:
        return True
    diffs = np.diff(path, axis=0)
    return bool(np.all(np.max(np.abs(diffs), axis=1) <= 1))


def success_rate(
    predicted_paths: Sequence[PathLike],
    *,
    starts: Sequence[tuple[int, int]] | None = None,
    goals: Sequence[tuple[int, int]] | None = None,
) -> float:
    """Fraction of samples whose decoded path connects start to goal.

    A path is "successful" iff:
        (i) it has at least two cells,
       (ii) consecutive cells are 8-connected, and
      (iii) when ``starts`` and ``goals`` are provided, the first / last
            cells match the prescribed endpoints.

    Returns NaN for an empty batch.
    """
    n = len(predicted_paths)
    if n == 0:
        return float("nan")
    if (starts is None) != (goals is None):
        raise ValueError("Provide both `starts` and `goals` together, or neither.")
    if starts is not None and len(starts) != n:
        raise ValueError(f"len(starts)={len(starts)} != len(paths)={n}.")
    if goals is not None and len(goals) != n:
        raise ValueError(f"len(goals)={len(goals)} != len(paths)={n}.")
    n_ok = 0
    for i, p in enumerate(predicted_paths):
        arr = _as_xy_array(p)
        if not _is_connected_path(arr):
            continue
        if starts is not None and goals is not None:
            si, sj = starts[i]
            gi, gj = goals[i]
            if int(arr[0, 0]) != int(si) or int(arr[0, 1]) != int(sj):
                continue
            if int(arr[-1, 0]) != int(gi) or int(arr[-1, 1]) != int(gj):
                continue
        n_ok += 1
    return n_ok / n


def obstacle_violation_rate(
    predicted_paths: Sequence[PathLike],
    obstacles: Sequence[np.ndarray],
) -> float:
    """Fraction of samples whose decoded path crosses any obstacle pixel.

    Each ``obstacles[i]`` is an ``(H, W)`` boolean array (1 = forbidden).
    Returns NaN for an empty batch.
    """
    n = len(predicted_paths)
    if n == 0:
        return float("nan")
    if len(obstacles) != n:
        raise ValueError(f"len(obstacles)={len(obstacles)} != len(paths)={n}.")
    n_violate = 0
    for path, obs in zip(predicted_paths, obstacles):
        arr = _as_xy_array(path)
        if arr.size == 0:
            continue
        obs_arr = np.asarray(obs)
        if obs_arr.size == 0:
            continue
        rr = np.clip(np.round(arr[:, 0]).astype(int), 0, obs_arr.shape[0] - 1)
        cc = np.clip(np.round(arr[:, 1]).astype(int), 0, obs_arr.shape[1] - 1)
        if bool(np.any(obs_arr[rr, cc])):
            n_violate += 1
    return n_violate / n


def obstacle_cells_traversed_per_path(
    predicted_paths: Sequence[PathLike],
    obstacles: Sequence[np.ndarray],
) -> float:
    """Mean obstacle-cell traversal count, conditioned on a violation.

    Distinguishes 'clipped a corner' (1) from 'drove through the building'
    (50). Returns NaN if the batch contains no violation.
    """
    n = len(predicted_paths)
    if n == 0:
        return float("nan")
    if len(obstacles) != n:
        raise ValueError(f"len(obstacles)={len(obstacles)} != len(paths)={n}.")
    counts: list[int] = []
    for path, obs in zip(predicted_paths, obstacles):
        arr = _as_xy_array(path)
        if arr.size == 0:
            continue
        obs_arr = np.asarray(obs)
        if obs_arr.size == 0:
            continue
        rr = np.clip(np.round(arr[:, 0]).astype(int), 0, obs_arr.shape[0] - 1)
        cc = np.clip(np.round(arr[:, 1]).astype(int), 0, obs_arr.shape[1] - 1)
        c = int(np.sum(obs_arr[rr, cc] != 0))
        if c > 0:
            counts.append(c)
    if not counts:
        return float("nan")
    return float(np.mean(counts))
