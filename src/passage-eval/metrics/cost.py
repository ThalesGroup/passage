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

"""Analytic path cost under the same metre-valued models as the generator."""

from __future__ import annotations

import numpy as np
from numba import njit

from passage.pathfinding_utils import COST_MODELS, PIXEL_SIZE_M
from passage_eval._typing import PathLike, _as_xy_array


def canonical_cost_model(name: str) -> str:
    """Accept the paper's ``slope`` spelling and the release's ``slope_uphill``."""
    name = "slope_uphill" if name == "slope" else name
    if name not in COST_MODELS:
        raise ValueError(f"Unknown cost model {name!r}")
    return name


MODEL_CODES = {"distance": 0, "elevation": 1, "absolute_elevation": 2, "slope_uphill": 3, "energy": 4}


@njit(cache=True)
def directed_edge_cost(
    src: float,
    dst: float,
    distance: float,
    weight: float,
    model: int,
    minimum: float,
    elevation_range: float,
    absolute_max: float,
) -> float:
    """One directed edge, evaluated exclusively in float64 by callers.

    Destination height uses extrema from the whole crop, including blocked
    cells. Reverse search must call this with predecessor first, current last.
    """
    if model == 0:
        return distance
    if model == 1:
        value = (dst - minimum) / elevation_range if elevation_range > 0 else 0.0
        return distance * (1.0 + weight * value)
    if model == 2:
        value = abs(dst) / absolute_max if absolute_max > 0 else 0.0
        return distance * (1.0 + weight * value)
    gain = max(0.0, dst - src)
    if model == 4:
        return distance + weight * gain
    return distance * (1.0 + weight * (gain / elevation_range)) if elevation_range > 0 else distance


@njit(cache=True)
def _sum_path(elevation, points, weight, model, minimum, elevation_range, absolute_max):
    total = 0.0
    for index in range(1, len(points)):
        r0, c0 = points[index - 1]
        r1, c1 = points[index]
        diagonal = r0 != r1 and c0 != c1
        distance = PIXEL_SIZE_M * np.sqrt(2.0) if diagonal else PIXEL_SIZE_M
        total += directed_edge_cost(
            elevation[r0, c0], elevation[r1, c1], distance, weight, model, minimum, elevation_range, absolute_max
        )
    return total


def path_cost(elevation: np.ndarray, path: PathLike, *, weight: float, cost_model: str) -> float:
    """Sum directed edge costs; require an integer, in-bounds 8-connected path.

    Obstacle and endpoint validity are checked by the scorer. A malformed path
    returns NaN so it can receive the protocol's censored penalty there.
    Valid paths with nonfinite float64 cost arithmetic raise ValueError.
    """
    cost_model = canonical_cost_model(cost_model)
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("cost weight must be finite and nonnegative")
    # Persisted terrain is float32; promote BEFORE differences/normalization.
    elevation = np.asarray(elevation, dtype=np.float64)
    if elevation.ndim != 2 or not elevation.size or not np.isfinite(elevation).all():
        raise ValueError("elevation must be a finite, nonempty 2D grid")
    try:
        points = _as_xy_array(path)
    except (ValueError, TypeError):
        return float("nan")
    if len(points) < 2 or not np.isfinite(points).all() or not np.equal(points, np.floor(points)).all():
        return float("nan")
    if np.any(points < 0) or np.any(points >= elevation.shape):
        return float("nan")
    steps = np.diff(points, axis=0)
    if not np.all(np.max(np.abs(steps), axis=1) == 1):
        return float("nan")
    points = points.astype(np.int64)
    minimum, maximum = float(elevation.min()), float(elevation.max())
    elevation_range = maximum - minimum
    if not np.isfinite(elevation_range):
        raise ValueError("float64 cost arithmetic overflow in elevation range")
    total = float(
        _sum_path(
            elevation,
            points,
            float(weight),
            MODEL_CODES[cost_model],
            minimum,
            elevation_range,
            max(abs(minimum), abs(maximum)),
        )
    )
    if not np.isfinite(total):
        raise ValueError("float64 cost arithmetic overflow in path total")
    return total
