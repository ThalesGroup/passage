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
    if path.shape[0] < 2 or not np.isfinite(path).all() or not np.equal(path, np.floor(path)).all():
        return False
    diffs = np.diff(path, axis=0)
    return bool(np.all(np.max(np.abs(diffs), axis=1) == 1))


def path_status(
    path: PathLike,
    *,
    start: tuple[int, int] | None = None,
    goal: tuple[int, int] | None = None,
    obstacles: np.ndarray | None = None,
) -> tuple[bool, bool, int]:
    """Return connection, obstacle violation and traversal count for one path.

    Invalid coordinates are never rounded or clipped into a valid grid cell.
    An obstacle crossing is reported separately from start-to-goal connection.
    """
    try:
        points = _as_xy_array(path)
    except (TypeError, ValueError):
        return False, False, 0
    connected = _is_connected_path(points)
    if start is not None and goal is not None and len(points):
        connected = connected and np.array_equal(points[0], start) and np.array_equal(points[-1], goal)
    count = 0
    if obstacles is not None and len(points):
        obs = np.asarray(obstacles)
        if obs.ndim != 2:
            raise ValueError("obstacles must be a 2D grid")
        finite_integer = np.isfinite(points).all(axis=1) & np.equal(points, np.floor(points)).all(axis=1)
        in_bounds = finite_integer & (points >= 0).all(axis=1) & (points < obs.shape).all(axis=1)
        connected = connected and bool(in_bounds.all())
        cells = points[in_bounds].astype(np.int64)
        count = int(np.count_nonzero(obs[cells[:, 0], cells[:, 1]]))
    return bool(connected), count > 0, count


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
    return (
        sum(
            path_status(
                path,
                start=starts[i] if starts is not None else None,
                goal=goals[i] if goals is not None else None,
            )[0]
            for i, path in enumerate(predicted_paths)
        )
        / n
    )


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
    return sum(path_status(path, obstacles=obs)[1] for path, obs in zip(predicted_paths, obstacles)) / n


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
    counts = [path_status(path, obstacles=obs)[2] for path, obs in zip(predicted_paths, obstacles)]
    positive = [count for count in counts if count > 0]
    return float(np.mean(positive)) if positive else float("nan")
