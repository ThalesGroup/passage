"""Exact, deterministic implicit-grid references and directed costs-to-go.

No explicit edge graph is built. An indexed binary heap holds at most one
entry per cell, with (priority, flattened row-major cell ID) ties. All terrain
arithmetic and accumulation are float64, after decoding persisted terrain.
The existing GraphSpec destination-only diagonal policy is preserved.
Point results are recosted in forward path order; reverse fields accumulate
from the goal. Their costs agree within GraphSpec's rtol/atol, with no bitwise
equality promise between these summation orders.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Integral
import time

import numpy as np
from numba import njit

from passage_eval.metrics.cost import MODEL_CODES, directed_edge_cost, path_cost
from passage_eval.protocol.contracts import GraphSpec


@dataclass(frozen=True)
class ReferenceResult:
    status: str
    path: tuple[tuple[int, int], ...]
    cost: float | None
    expansions: int
    work: dict[str, int]
    elapsed_seconds: float


@dataclass(frozen=True)
class GoalField:
    """Offline field: only settled distances are exposed as finite values.

    For an exact full field, infinity proves unreachability. For a censored
    field, infinity means unproved; it must never become a training negative.
    next_nodes contains flattened successor IDs on routes towards goal.
    """
    status: str
    distances: np.ndarray
    settled: np.ndarray
    next_nodes: np.ndarray
    goal: tuple[int, int]
    expansions: int
    work: dict[str, int]
    elapsed_seconds: float

    def route(self, start: tuple[int, int]) -> ReferenceResult:
        start = _point(start, self.distances.shape)
        if not self.settled[start]:
            status = "no_path" if self.status == "exact" else "censored"
            return ReferenceResult(status, (), None, self.expansions, dict(self.work), self.elapsed_seconds)
        path = _trace(self.next_nodes.ravel(), start, self.goal, self.distances.shape[1], forward=True)
        return ReferenceResult("exact", path, float(self.distances[start]), self.expansions,
                               dict(self.work), self.elapsed_seconds)


def _point(point, shape):
    try:
        valid = len(point) == 2 and all(not isinstance(x, (bool, np.bool_)) and isinstance(x, Integral) for x in point)
    except TypeError:
        valid = False
    if not valid:
        raise ValueError("endpoints must be integer row/column pairs")
    point = tuple(int(x) for x in point)
    if any(x < 0 or x >= size for x, size in zip(point, shape)):
        raise ValueError("endpoint outside terrain")
    return point


def _inputs(elevation, obstacles, graph, max_expansions):
    graph = GraphSpec() if graph is None else graph
    if not isinstance(graph, GraphSpec):
        raise ValueError("graph must be a GraphSpec")
    if max_expansions is not None and (isinstance(max_expansions, bool)
            or not isinstance(max_expansions, Integral) or max_expansions < 0):
        raise ValueError("max_expansions must be a nonnegative integer or None")
    terrain = np.ascontiguousarray(elevation, dtype=np.float64)
    if terrain.ndim != 2 or not terrain.size or not np.isfinite(terrain).all():
        raise ValueError("elevation must be a finite nonempty 2D grid")
    if terrain.size > np.iinfo(np.int32).max:
        raise ValueError("grid exceeds int32 cell indexing")
    obstacles = np.asarray(obstacles)
    if obstacles.shape != terrain.shape or not np.isin(obstacles, [0, 1]).all():
        raise ValueError("obstacles must be a matching binary grid")
    minimum, maximum = float(terrain.min()), float(terrain.max())
    if not np.isfinite(maximum - minimum):
        raise ValueError("elevation range exceeds float64")
    return terrain, np.ascontiguousarray(obstacles, dtype=np.bool_), graph, minimum, maximum


@njit(cache=True)
def _heuristic(node, cols, target, astar):
    if not astar:
        return 0.0
    dr = node // cols - target // cols
    dc = node % cols - target % cols
    # Conservative Euclidean lower bound. This margin absorbs rounding in
    # long path sums without modifying the graph's semantic edge costs.
    return 30.0 * math.sqrt(float(dr * dr + dc * dc)) * (1.0 - 1e-12)


@njit(cache=True)
def _less(a, b, distances, cols, target, astar):
    pa = distances[a] + _heuristic(a, cols, target, astar)
    pb = distances[b] + _heuristic(b, cols, target, astar)
    return pa < pb or (pa == pb and a < b)


@njit(cache=True)
def _sift_up(heap, positions, slot, distances, cols, target, astar):
    node = heap[slot]
    while slot > 0:
        parent = (slot - 1) // 2
        if not _less(node, heap[parent], distances, cols, target, astar):
            break
        heap[slot] = heap[parent]
        positions[heap[slot]] = slot
        slot = parent
    heap[slot] = node
    positions[node] = slot


@njit(cache=True)
def _search(terrain, blocked, origin, target, model, weight, minimum, elevation_range,
            absolute_max, limit, reverse, astar):
    rows, cols = terrain.shape
    n = terrain.size
    distances = np.full(n, np.inf, dtype=np.float64)
    parents = np.full(n, -1, dtype=np.int32)
    settled = np.zeros(n, dtype=np.bool_)
    positions = np.full(n, -1, dtype=np.int32)
    heap = np.empty(n, dtype=np.int32)
    heap[0] = origin
    positions[origin] = 0
    distances[origin] = 0.0
    count, expansions, queue_operations, peak_queue, discovered = 1, 0, 1, 1, 1
    status = 1  # no_path, or completed reverse field
    while count:
        node = heap[0]
        # Goal settlement is a success, not an expansion of its outgoing edges.
        if not reverse and node == target:
            settled[node] = True
            queue_operations += 1
            status = 0
            break
        if expansions >= limit:
            status = 2
            break
        count -= 1
        positions[node] = -1
        queue_operations += 1
        if count:
            last = heap[count]
            slot = 0
            while 2 * slot + 1 < count:
                child = 2 * slot + 1
                if child + 1 < count and _less(heap[child + 1], heap[child], distances, cols, target, astar):
                    child += 1
                if not _less(heap[child], last, distances, cols, target, astar):
                    break
                heap[slot] = heap[child]
                positions[heap[slot]] = slot
                slot = child
            heap[slot] = last
            positions[last] = slot
        settled[node] = True
        expansions += 1
        r, c = node // cols, node % cols
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                if blocked[rr, cc]:
                    continue
                neighbor = rr * cols + cc
                distance = 30.0 * math.sqrt(2.0) if dr != 0 and dc != 0 else 30.0
                if reverse:
                    edge = directed_edge_cost(terrain[rr, cc], terrain[r, c], distance, weight,
                                              model, minimum, elevation_range, absolute_max)
                else:
                    edge = directed_edge_cost(terrain[r, c], terrain[rr, cc], distance, weight,
                                              model, minimum, elevation_range, absolute_max)
                candidate = distances[node] + edge
                if not np.isfinite(edge) or not np.isfinite(candidate):
                    # Infinity already denotes unproved/unreachable cells. It
                    # must never absorb arithmetic overflow as a graph proof.
                    return 3, distances, parents, settled, expansions, queue_operations, peak_queue, discovered
                if candidate < distances[neighbor]:
                    if np.isinf(distances[neighbor]):
                        discovered += 1
                    distances[neighbor] = candidate
                    parents[neighbor] = node
                    settled[neighbor] = False  # permit reopening after floating-point differences
                    slot = positions[neighbor]
                    if slot < 0:
                        slot = count
                        count += 1
                        heap[slot] = neighbor
                    _sift_up(heap, positions, slot, distances, cols, target, astar)
                    queue_operations += 1
                    peak_queue = max(peak_queue, count)
    return status, distances, parents, settled, expansions, queue_operations, peak_queue, discovered


def _trace(parents, start, goal, cols, *, forward=False):
    node, stop = (start[0] * cols + start[1], goal[0] * cols + goal[1]) if forward else (
        goal[0] * cols + goal[1], start[0] * cols + start[1])
    path = []
    for _ in range(len(parents) + 1):
        path.append((node // cols, node % cols))
        if node == stop:
            return tuple(path if forward else reversed(path))
        node = int(parents[node])
        if node < 0:
            break
    raise RuntimeError("reference predecessor chain is incomplete or cyclic")


def _work(expansions, queue_operations, peak_queue, discovered):
    return {"expansions": int(expansions), "queue_operations": int(queue_operations),
            "max_live_states": int(discovered), "peak_queue": int(peak_queue)}


def solve_reference(elevation, obstacles, start, goal, *, graph: GraphSpec | None = None,
                    max_expansions: int | None = None, algorithm: str = "astar") -> ReferenceResult:
    """Return an exact optimum, proved no_path, or an explicitly censored solve.

    Invalid inputs or float64 cost overflow raise ValueError. Blocked endpoints
    yield proved no_path; arithmetic failure never proves infeasibility.
    Zero budget censors a distinct-endpoint request. Budgets count expanded
    outgoing neighborhoods; settling a goal does not spend an expansion.
    JIT compilation is included in elapsed_seconds on the first invocation.
    """
    begun = time.perf_counter()
    if algorithm not in {"astar", "dijkstra"}:
        raise ValueError("algorithm must be astar or dijkstra")
    terrain, blocked, graph, minimum, maximum = _inputs(elevation, obstacles, graph, max_expansions)
    start, goal = _point(start, terrain.shape), _point(goal, terrain.shape)
    if blocked[start] or blocked[goal]:
        return ReferenceResult("no_path", (), None, 0, _work(0, 0, 0, 0), time.perf_counter() - begun)
    cols = terrain.shape[1]
    output = _search(terrain, blocked, start[0] * cols + start[1], goal[0] * cols + goal[1],
                     MODEL_CODES[graph.cost_model], float(graph.cost_weight), minimum, maximum - minimum,
                     max(abs(minimum), abs(maximum)), np.iinfo(np.int64).max if max_expansions is None
                     else int(max_expansions), False, algorithm == "astar")
    code, distances, parents, _, expansions, operations, peak, discovered = output
    if code == 3:
        raise ValueError("reference float64 cost arithmetic overflow")
    status = ("exact", "no_path", "censored")[code]
    path = _trace(parents, start, goal, cols) if code == 0 else ()
    cost = (0.0 if start == goal else recost_legal_path(terrain, path, graph=graph, obstacles=blocked,
                                                      start=start, goal=goal)) if code == 0 else None
    if code == 0 and not np.isfinite(cost):
        raise RuntimeError("exact reference produced an illegal or nonfinite route")
    return ReferenceResult(status, path, cost, int(expansions), _work(expansions, operations, peak, discovered),
                           time.perf_counter() - begun)


@njit(cache=True)
def _reaches(blocked, origin, target):
    rows, cols = blocked.shape
    seen = np.zeros(rows * cols, dtype=np.bool_)
    stack = np.empty(rows * cols, dtype=np.int32)
    stack[0] = origin
    seen[origin] = True
    count = 1
    while count:
        count -= 1
        node = stack[count]
        if node == target:
            return True
        r, c = node // cols, node % cols
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if (dr == 0 and dc == 0) or rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                    continue
                neighbor = rr * cols + cc
                if blocked[rr, cc] or seen[neighbor]:
                    continue
                seen[neighbor] = True
                stack[count] = neighbor
                count += 1
    return False


def endpoints_connected(obstacles, start, goal) -> bool:
    """Return whether the reference graph has any route from start to goal.

    The edge set is the one ``solve_reference`` searches: eight neighbours,
    obstacles checked at the destination cell only, every other edge finite.
    Reachability therefore needs no terrain, and ``False`` is exactly the case
    the solver would prove ``no_path``. Blocked endpoints are disconnected.
    """
    blocked = np.asarray(obstacles)
    if blocked.ndim != 2 or not blocked.size or not np.isin(blocked, [0, 1]).all():
        raise ValueError("obstacles must be a nonempty binary 2D grid")
    if blocked.size > np.iinfo(np.int32).max:
        raise ValueError("grid exceeds int32 cell indexing")
    blocked = np.ascontiguousarray(blocked, dtype=np.bool_)
    start, goal = _point(start, blocked.shape), _point(goal, blocked.shape)
    if blocked[start] or blocked[goal]:
        return False
    cols = blocked.shape[1]
    return bool(_reaches(blocked, start[0] * cols + start[1], goal[0] * cols + goal[1]))


def reverse_goal_field(elevation, obstacles, goal, *, graph: GraphSpec | None = None,
                       max_expansions: int | None = None) -> GoalField:
    """Dijkstra on transposed edges computes directed original costs to goal.

    Relaxing predecessor u from current v adds w(u,v), never w(v,u).
    Only proved/settled costs are published; targets are evaluator-only.
    Float64 cost overflow raises ValueError instead of publishing a field.
    """
    begun = time.perf_counter()
    terrain, blocked, graph, minimum, maximum = _inputs(elevation, obstacles, graph, max_expansions)
    goal = _point(goal, terrain.shape)
    if blocked[goal]:
        raise ValueError("reverse field goal must be traversable")
    cols = terrain.shape[1]
    origin = goal[0] * cols + goal[1]
    output = _search(terrain, blocked, origin, origin, MODEL_CODES[graph.cost_model], float(graph.cost_weight),
                     minimum, maximum - minimum, max(abs(minimum), abs(maximum)),
                     np.iinfo(np.int64).max if max_expansions is None else int(max_expansions), True, False)
    code, distances, parents, settled, expansions, operations, peak, discovered = output
    if code == 3:
        raise ValueError("reference float64 cost arithmetic overflow")
    distances[~settled] = np.inf
    parents[~settled] = -1
    return GoalField("censored" if code == 2 else "exact", distances.reshape(terrain.shape),
                     settled.reshape(terrain.shape), parents.reshape(terrain.shape), goal, int(expansions),
                     _work(expansions, operations, peak, discovered), time.perf_counter() - begun)


def recost_legal_path(elevation, path, *, graph: GraphSpec | None = None, obstacles=None,
                      start=None, goal=None) -> float:
    """Re-cost legal integer 8-connected paths; return NaN for illegal routes."""
    graph = GraphSpec() if graph is None else graph
    if not isinstance(graph, GraphSpec):
        raise ValueError("graph must be a GraphSpec")
    terrain = np.asarray(elevation)
    if terrain.ndim != 2 or not terrain.size or not np.isfinite(terrain).all():
        raise ValueError("elevation must be a finite nonempty 2D grid")
    try:
        points = np.asarray(path, dtype=np.float64)
    except (TypeError, ValueError):
        return float("nan")
    if (points.ndim != 2 or points.shape[1] != 2 or not len(points) or not np.isfinite(points).all()
            or not np.equal(points, np.floor(points)).all() or np.any(points < 0)
            or np.any(points >= terrain.shape)):
        return float("nan")
    if start is not None and not np.array_equal(points[0], start):
        return float("nan")
    if goal is not None and not np.array_equal(points[-1], goal):
        return float("nan")
    indices = points.astype(np.int64)
    if obstacles is not None:
        obstacles = np.asarray(obstacles)
        if obstacles.shape != terrain.shape or not np.isin(obstacles, [0, 1]).all():
            raise ValueError("obstacles must be a matching binary grid")
        if np.any(obstacles[indices[:, 0], indices[:, 1]]):
            return float("nan")
    if len(points) == 1:
        return 0.0 if start is not None and goal is not None and np.array_equal(start, goal) else float("nan")
    return path_cost(terrain, indices, weight=graph.cost_weight, cost_model=graph.cost_model)
