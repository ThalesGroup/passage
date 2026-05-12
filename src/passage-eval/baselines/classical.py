# MIT License
#
# Copyright (c) 2026 THALES

"""Two classical reference planners that implement :class:`Planner`.

* :class:`AStarPlanner` — exact A$^{\\star}$ on the analytic cost grid.
  When ``cost_weight = 1.0`` this is the PASSAGE oracle itself; with
  ``cost_weight != 1.0`` it becomes a deliberately mis-calibrated
  baseline that exercises the metric implementation through a
  non-trivial planner.

* :class:`GreedyToGoalPlanner` — Manhattan-distance descent that
  ignores costs entirely, walking diagonally toward the goal. Useful
  as a *weak* baseline that produces sub-optimal but feasible paths,
  letting users sanity-check that the primary metrics actually
  separate good and bad planners.

Both depend only on numpy + the ``heapq`` module — no torch or
external dataset packages — so they are safe to import in a CPU-only
review environment.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from passage_eval.protocol.planner import Planner, PlannerOutput, Scene


__all__ = ["AStarPlanner", "GreedyToGoalPlanner"]


_NEIGHBOUR_OFFSETS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)
_NEIGHBOUR_DISTS_8 = tuple(
    math.sqrt(dr * dr + dc * dc) for dr, dc in _NEIGHBOUR_OFFSETS_8
)


def _heuristic_l2(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _edge_cost(elevation: np.ndarray, u: tuple[int, int], v: tuple[int, int],
               *, weight: float, cost_model: str, dist: float) -> float:
    h_u = float(elevation[u[0], u[1]])
    h_v = float(elevation[v[0], v[1]])
    if cost_model == "distance":
        return dist
    if cost_model == "elevation":
        h_max = float(elevation.max())
        h_min = float(elevation.min())
        if h_max <= h_min:
            return dist
        return dist * (1.0 + weight * (h_v - h_min) / (h_max - h_min))
    if cost_model == "absolute_elevation":
        return dist * (1.0 + weight * abs(h_v))
    if cost_model == "slope_uphill":
        h_max = float(elevation.max())
        h_min = float(elevation.min())
        if h_max <= h_min:
            return dist
        gain = max(0.0, h_v - h_u)
        return dist * (1.0 + weight * gain / (h_max - h_min))
    if cost_model == "energy":
        return dist + weight * max(0.0, h_v - h_u)
    raise ValueError(f"Unknown cost_model='{cost_model}'.")


def _path_cost(elevation: np.ndarray, path: list[tuple[int, int]],
               *, weight: float, cost_model: str) -> float:
    if len(path) < 2:
        return float("nan")
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        dist = math.hypot(u[0] - v[0], u[1] - v[1])
        total += _edge_cost(elevation, u, v,
                            weight=weight, cost_model=cost_model, dist=dist)
    return total


@dataclass
class AStarPlanner(Planner):
    """Exact A$^{\\star}$ on the analytic cost grid.

    Implements :class:`Planner`. Acts as the PASSAGE oracle when
    invoked with the same cost model and weight as the dataset's
    label generation pipeline; with a perturbed weight it becomes a
    sub-optimal but valid candidate planner used as a sanity-check
    baseline.

    The implementation is deliberately straightforward — a
    binary-heap A$^{\\star}$ with the Euclidean heuristic — and does
    not pretend to be the fast PASSAGE C-extension.
    """

    name: str = "astar_classical"

    def predict_path(self, scene: Scene) -> PlannerOutput:
        elevation = scene.elevation
        obstacles = scene.obstacles
        H, W = elevation.shape
        start = scene.start
        goal = scene.goal

        if obstacles[start[0], start[1]] or obstacles[goal[0], goal[1]]:
            return PlannerOutput(predicted_path=[start], predicted_cost=float("inf"))

        open_heap: list[tuple[float, int, tuple[int, int]]] = []
        heapq.heappush(open_heap, (_heuristic_l2(start, goal), 0, start))
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        g_score: dict[tuple[int, int], float] = {start: 0.0}
        counter = 1
        found = False

        while open_heap:
            _f, _id, current = heapq.heappop(open_heap)
            if current == goal:
                found = True
                break
            cr, cc = current
            for (dr, dc), dist in zip(_NEIGHBOUR_OFFSETS_8, _NEIGHBOUR_DISTS_8):
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obstacles[nr, nc]:
                    continue
                neighbour = (nr, nc)
                step = _edge_cost(elevation, current, neighbour,
                                  weight=scene.cost_weight,
                                  cost_model=scene.cost_model,
                                  dist=dist)
                tentative = g_score[current] + step
                if tentative < g_score.get(neighbour, float("inf")):
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative
                    f_score = tentative + _heuristic_l2(neighbour, goal)
                    heapq.heappush(open_heap, (f_score, counter, neighbour))
                    counter += 1

        if not found:
            return PlannerOutput(predicted_path=[start], predicted_cost=float("inf"))

        path: list[tuple[int, int]] = [goal]
        while path[-1] != start:
            path.append(came_from[path[-1]])
        path.reverse()
        cost = _path_cost(elevation, path,
                          weight=scene.cost_weight,
                          cost_model=scene.cost_model)
        return PlannerOutput(predicted_path=path, predicted_cost=cost)


@dataclass
class GreedyToGoalPlanner(Planner):
    """Diagonal-greedy descent toward the goal, ignoring costs.

    Useful as a weak-but-valid baseline. Always reaches the goal as
    long as no obstacle blocks every diagonal-then-axial route, and
    its cost is always $\\geq$ A$^{\\star}$ — so the cost-regret
    metric should always be positive when this planner is scored
    against the oracle.
    """

    name: str = "greedy_classical"
    max_steps_factor: int = 8

    def predict_path(self, scene: Scene) -> PlannerOutput:
        elevation = scene.elevation
        obstacles = scene.obstacles
        H, W = elevation.shape
        start = scene.start
        goal = scene.goal

        path: list[tuple[int, int]] = [start]
        current = start
        max_steps = self.max_steps_factor * (H + W)
        for _ in range(max_steps):
            if current == goal:
                break
            best = None
            best_dist = float("inf")
            cr, cc = current
            for dr, dc in _NEIGHBOUR_OFFSETS_8:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obstacles[nr, nc]:
                    continue
                if (nr, nc) in path:
                    continue
                d = _heuristic_l2((nr, nc), goal)
                if d < best_dist:
                    best_dist = d
                    best = (nr, nc)
            if best is None:
                break
            path.append(best)
            current = best
        cost = _path_cost(elevation, path,
                          weight=scene.cost_weight,
                          cost_model=scene.cost_model)
        return PlannerOutput(predicted_path=path, predicted_cost=cost)
