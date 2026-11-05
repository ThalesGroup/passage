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

"""A second, independent implementation of the reference optimum ``c*``: an 8-connected Dijkstra search.

The reference admission inventory qualifies every task with the deterministic A* of ``passage.audit``. This module
recomputes the optimal cost with code that shares none of it: no heuristic, its own priority queue, its own edge
costs and its own reading of the graph. It follows the graph that the release declares and that the manuscript's
cost models define:

* 8-connected moves of 30 m (orthogonal) or 30 m x sqrt(2) (diagonal), corner cutting allowed;
* a move is blocked when its destination is an obstacle; no path exists when the start or the goal is one;
* the edge cost of each model, with the elevation extrema of the whole crop (blocked cells included):
  ``distance`` L; ``elevation`` L (1 + w (z_dst - z_min) / (z_max - z_min)); ``absolute_elevation``
  L (1 + w |z_dst| / max |z|); ``energy`` L + w max(0, z_dst - z_src); ``slope_uphill``
  L (1 + w max(0, z_dst - z_src) / (z_max - z_min)); a zero range or zero maximum leaves L.

Residual risk, stated with the result: both implementations share this definition of the cost (receipts audit,
limit 10). Agreement shows that the A* found the optimum of this graph, not that the graph is the right one.
"""

from __future__ import annotations

import heapq
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

IMPLEMENTATION = "passage-dijkstra8/v1"
PIXEL_SIZE_M = 30.0
MODELS = ("distance", "elevation", "absolute_elevation", "energy", "slope_uphill")
_ALIASES = {"slope": "slope_uphill"}
_MOVES = tuple(
    (dr, dc, PIXEL_SIZE_M * (math.sqrt(2.0) if dr and dc else 1.0))
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if dr or dc
)


def dijkstra8_cost(
    elevation: Any, obstacles: Any, start: Any, goal: Any, *, cost_model: str, weight: float
) -> float | None:
    """The optimal path cost from ``start`` to ``goal`` (row, column), or ``None`` when no path exists."""
    model = _ALIASES.get(cost_model, cost_model)
    if model not in MODELS:
        raise ValueError(f"the second implementation does not define the cost model {cost_model!r}")
    weight = float(weight)
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("the cost weight must be finite and nonnegative")
    z = np.asarray(elevation, dtype=np.float64)
    blocked = np.asarray(obstacles).astype(bool)
    if z.ndim != 2 or blocked.shape != z.shape or not z.size or not np.isfinite(z).all():
        raise ValueError("elevation and obstacles must be finite grids of one shape")
    rows, columns = z.shape
    start, goal = tuple(int(v) for v in start), tuple(int(v) for v in goal)
    for row, column in (start, goal):
        if not (0 <= row < rows and 0 <= column < columns):
            raise ValueError("an endpoint lies outside the grid")
    if blocked[start] or blocked[goal]:
        return None
    low, high = float(z.min()), float(z.max())
    span, magnitude = high - low, max(abs(low), abs(high))
    heights = z.tolist()
    walls = blocked.tolist()
    if model == "elevation":
        factor = [[1.0 + weight * (value - low) / span if span > 0 else 1.0 for value in line] for line in heights]
    elif model == "absolute_elevation":
        factor = [
            [1.0 + weight * abs(value) / magnitude if magnitude > 0 else 1.0 for value in line] for line in heights
        ]
    else:
        factor = None
    best = {start: 0.0}
    queue = [(0.0, start)]
    done = set()
    while queue:
        cost, cell = heapq.heappop(queue)
        if cell in done:
            continue
        if cell == goal:
            return cost
        done.add(cell)
        row, column = cell
        here = heights[row][column]
        for dr, dc, length in _MOVES:
            r, c = row + dr, column + dc
            if not (0 <= r < rows and 0 <= c < columns) or walls[r][c] or (r, c) in done:
                continue
            if factor is not None:
                step = length * factor[r][c]
            elif model == "distance":
                step = length
            else:
                gain = max(0.0, heights[r][c] - here)
                if model == "energy":
                    step = length + weight * gain
                else:
                    step = length * (1.0 + weight * gain / span) if span > 0 else length
            total = cost + step
            if total < best.get((r, c), math.inf):
                best[(r, c)] = total
                heapq.heappush(queue, (total, (r, c)))
    return None


def second_reference_check(scene: Any, qualified: Mapping[str, Any], *, rtol: float, atol: float) -> dict[str, Any]:
    """Compare a qualifier's outcome on ``scene`` with the second implementation.

    A search that ended ``exact`` must give the same optimal cost within the graph tolerance; a search that ended
    ``no_path`` must find no path either. Any other status (an exhausted search) is not compared here.
    """
    status = qualified.get("reference_status")
    if status not in ("exact", "no_path"):
        return {"implementation": IMPLEMENTATION, "status": "not_compared", "reference_status": status}
    cost = dijkstra8_cost(
        scene.elevation, scene.obstacles, scene.start, scene.goal, cost_model=scene.cost_model, weight=scene.cost_weight
    )
    optimum = qualified.get("optimal_cost") if status == "exact" else None
    if status == "exact":
        agrees = (
            cost is not None
            and isinstance(optimum, (int, float))
            and math.isfinite(optimum)
            and bool(np.isclose(cost, optimum, rtol=rtol, atol=atol))
        )
    else:
        agrees = cost is None
    return {
        "implementation": IMPLEMENTATION,
        "status": "agrees" if agrees else "disagrees",
        "reference_status": status,
        "cost": cost,
        "reference_cost": optimum,
        "abs_difference": abs(cost - optimum) if cost is not None and isinstance(optimum, (int, float)) else None,
    }
