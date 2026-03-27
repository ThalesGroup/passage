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

"""Pathfinding utilities and helpers for PASSAGE."""

import heapq
import math
import time

import networkx as nx
import numpy as np
from loguru import logger

from passage.utils import _log_timing

# Try to import Numba for JIT-compiled solvers; fall back to pure Python if unavailable
try:
    from numba import njit, prange
    from numba.typed import List as NumbaList

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Provide dummy decorator that does nothing
    def njit(*args, **kwargs):
        def decorator(func):
            return func

        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator

    prange = range
    NumbaList = list

# Edge cost models
#   elevation         - distance * (1 + weight * normalized_elevation) [DEFAULT]
#   absolute_elevation- distance * (1 + weight * normalized_abs_elevation)
#   distance          - distance only (flat)
#   slope_uphill      - penalize uphill elevation changes (distance-scaled)
#   energy            - distance + weight * max(0, elevation_gain)
COST_MODELS = {"elevation", "absolute_elevation", "distance", "slope_uphill", "energy"}

# Diagonal movement options
# Controls which neighbors are connected in the graph:
#   always  - 8-connectivity (cardinal + diagonal) [DEFAULT]
#   never   - 4-connectivity (cardinal only: N/S/E/W)
DIAGONAL_OPTIONS = {"always", "never"}

# Pixel size in meters (ALOS AW3D30 = 30m resolution)
PIXEL_SIZE_M = 30.0
DIAGONAL_DISTANCE_M = PIXEL_SIZE_M * math.sqrt(2)  # ~42.43m

# Heuristic functions for A* search (scaled to meters)
HEURISTIC_MAP = {
    "euclidean": lambda u, v: PIXEL_SIZE_M * math.sqrt((u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2),
    "manhattan": lambda u, v: PIXEL_SIZE_M * (abs(u[0] - v[0]) + abs(u[1] - v[1])),
    "chebyshev": lambda u, v: PIXEL_SIZE_M * max(abs(u[0] - v[0]), abs(u[1] - v[1])),
}


class _GraphTemplate:
    """Cached graph topology for a given grid shape and connectivity."""

    def __init__(self, rows: int, cols: int, diagonal_movement: str):
        self.rows = rows
        self.cols = cols
        self.diagonal_movement = diagonal_movement

        # Define neighbor offsets with their physical distances
        if diagonal_movement == "never":
            offsets = [
                (-1, 0, PIXEL_SIZE_M),
                (1, 0, PIXEL_SIZE_M),
                (0, -1, PIXEL_SIZE_M),
                (0, 1, PIXEL_SIZE_M),
            ]
        else:
            offsets = [
                (-1, 0, PIXEL_SIZE_M),
                (1, 0, PIXEL_SIZE_M),
                (0, -1, PIXEL_SIZE_M),
                (0, 1, PIXEL_SIZE_M),
                (-1, -1, DIAGONAL_DISTANCE_M),
                (-1, 1, DIAGONAL_DISTANCE_M),
                (1, -1, DIAGONAL_DISTANCE_M),
                (1, 1, DIAGONAL_DISTANCE_M),
            ]

        i_grid, j_grid = np.mgrid[0:rows, 0:cols]
        src_i_all = []
        src_j_all = []
        dst_i_all = []
        dst_j_all = []
        dist_all = []

        for di, dj, distance_m in offsets:
            dst_i = i_grid + di
            dst_j = j_grid + dj
            valid = (dst_i >= 0) & (dst_i < rows) & (dst_j >= 0) & (dst_j < cols)
            if not np.any(valid):
                continue

            src_i = i_grid[valid]
            src_j = j_grid[valid]
            dst_i_v = dst_i[valid]
            dst_j_v = dst_j[valid]

            src_i_all.append(src_i.ravel())
            src_j_all.append(src_j.ravel())
            dst_i_all.append(dst_i_v.ravel())
            dst_j_all.append(dst_j_v.ravel())
            dist_all.append(np.full(src_i.size, distance_m, dtype=np.float64))

        if len(src_i_all) == 0:
            self.src_i = np.array([], dtype=np.int32)
            self.src_j = np.array([], dtype=np.int32)
            self.dst_i = np.array([], dtype=np.int32)
            self.dst_j = np.array([], dtype=np.int32)
            self.distance_m = np.array([], dtype=np.float64)
        else:
            self.src_i = np.concatenate(src_i_all).astype(np.int32)
            self.src_j = np.concatenate(src_j_all).astype(np.int32)
            self.dst_i = np.concatenate(dst_i_all).astype(np.int32)
            self.dst_j = np.concatenate(dst_j_all).astype(np.int32)
            self.distance_m = np.concatenate(dist_all).astype(np.float64)

        # Build graph once with zero weights (weights updated per crop)
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from((i, j) for i in range(rows) for j in range(cols))
        if len(self.src_i) > 0:
            self.graph.add_weighted_edges_from(
                (
                    (int(self.src_i[k]), int(self.src_j[k])),
                    (int(self.dst_i[k]), int(self.dst_j[k])),
                    0.0,
                )
                for k in range(len(self.src_i))
            )


_GRAPH_TEMPLATE_CACHE: dict[tuple[int, int, str], _GraphTemplate] = {}


def _get_graph_template(rows: int, cols: int, diagonal_movement: str) -> _GraphTemplate:
    key = (rows, cols, diagonal_movement)
    template = _GRAPH_TEMPLATE_CACHE.get(key)
    if template is None:
        template = _GraphTemplate(rows, cols, diagonal_movement)
        _GRAPH_TEMPLATE_CACHE[key] = template
    return template


def _normalize_elevation_arrays(elevation: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return normalized elevation, normalized absolute elevation, and elevation range."""
    elev_min = float(elevation.min())
    elev_max = float(elevation.max())
    elev_range = elev_max - elev_min
    if elev_range > 0:
        normalized = (elevation - elev_min) / elev_range
    else:
        normalized = np.zeros_like(elevation)
    abs_elev = np.abs(elevation)
    abs_max = float(abs_elev.max())
    if abs_max > 0:
        abs_norm = abs_elev / abs_max
    else:
        abs_norm = np.zeros_like(elevation)
    return normalized, abs_norm, elev_range


def _compute_edge_weights(
    template: _GraphTemplate,
    elevation: np.ndarray,
    weight: float,
    cost_model: str,
) -> np.ndarray:
    if cost_model not in COST_MODELS:
        cost_model = "elevation"

    if template.distance_m.size == 0:
        return template.distance_m

    if cost_model == "distance":
        return template.distance_m

    normalized, abs_norm, elev_range = _normalize_elevation_arrays(elevation)
    dst_i = template.dst_i
    dst_j = template.dst_j

    if cost_model == "absolute_elevation":
        return template.distance_m * (1.0 + weight * abs_norm[dst_i, dst_j])

    src_i = template.src_i
    src_j = template.src_j
    if cost_model == "slope_uphill":
        if elev_range > 0:
            delta = elevation[dst_i, dst_j] - elevation[src_i, src_j]
            slope_norm = np.maximum(0.0, delta) / elev_range
        else:
            slope_norm = np.zeros_like(dst_i, dtype=np.float32)
        return template.distance_m * (1.0 + weight * slope_norm)

    if cost_model == "energy":
        delta = elevation[dst_i, dst_j] - elevation[src_i, src_j]
        return template.distance_m + weight * np.maximum(0.0, delta)

    return template.distance_m * (1.0 + weight * normalized[dst_i, dst_j])


def _build_pathfinding_cost_matrix(
    elevation: np.ndarray,
    weight: float,
    cost_model: str,
) -> np.ndarray | None:
    """Build a per-node cost matrix for the pathfinding package.

    Note: pathfinding uses node weights (not directional edge costs), so only
    cost models that depend on destination elevation are supported.
    """
    if cost_model not in COST_MODELS:
        cost_model = "elevation"

    if cost_model in {"slope_uphill", "energy"}:
        return None

    if cost_model == "distance":
        costs = np.ones_like(elevation, dtype=np.float64)
    else:
        normalized, abs_norm, _ = _normalize_elevation_arrays(elevation)
        if cost_model == "absolute_elevation":
            costs = 1.0 + weight * abs_norm
        else:
            costs = 1.0 + weight * normalized

    return np.maximum(costs, 1e-6)


def _pathfinding_path_to_coords(path) -> list[tuple[int, int]]:
    if not path:
        return []
    first = path[0]
    if hasattr(first, "x") and hasattr(first, "y"):
        return [(int(node.y), int(node.x)) for node in path]
    if isinstance(first, (tuple, list)) and len(first) >= 2:
        return [(int(p[1]), int(p[0])) for p in path]
    return []


def _find_path_pathfinding(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    solver_name: str,
    diagonal_movement: str,
    weight: float,
    heuristic_name: str,
    cost_model: str,
) -> tuple[list[tuple[int, int]], float, float]:
    """Pathfinding backend using the pathfinding package.

    Returns (path, prep_time, solve_time).
    """
    import importlib

    DiagonalMovement = importlib.import_module("pathfinding.core.diagonal_movement").DiagonalMovement
    Grid = importlib.import_module("pathfinding.core.grid").Grid
    heuristic_mod = importlib.import_module("pathfinding.core.heuristic")
    AStarFinder = importlib.import_module("pathfinding.finder.a_star").AStarFinder
    BestFirst = importlib.import_module("pathfinding.finder.best_first").BestFirst
    DijkstraFinder = importlib.import_module("pathfinding.finder.dijkstra").DijkstraFinder

    heuristic_map = {
        "euclidean": heuristic_mod.euclidean,
        "manhattan": heuristic_mod.manhattan,
        "chebyshev": heuristic_mod.chebyshev,
    }

    t_prep = time.time()
    cost_matrix = _build_pathfinding_cost_matrix(crop, weight, cost_model)
    if cost_matrix is None:
        return [], 0.0, 0.0

    diag_mode = DiagonalMovement.always if diagonal_movement == "always" else DiagonalMovement.never
    grid = Grid(matrix=cost_matrix.tolist())
    start_node = grid.node(int(start[1]), int(start[0]))
    end_node = grid.node(int(end[1]), int(end[0]))

    if solver_name == "dijkstra":
        finder = DijkstraFinder(diagonal_movement=diag_mode)
    elif solver_name == "best_first":
        heuristic = heuristic_map.get(heuristic_name, heuristic_mod.euclidean)
        finder = BestFirst(diagonal_movement=diag_mode, heuristic=heuristic)
    else:
        heuristic = heuristic_map.get(heuristic_name, heuristic_mod.euclidean)
        finder = AStarFinder(diagonal_movement=diag_mode, heuristic=heuristic)

    prep_time = time.time() - t_prep
    t_solve = time.time()
    path, _ = finder.find_path(start_node, end_node, grid)
    solve_time = time.time() - t_solve
    return _pathfinding_path_to_coords(path), prep_time, solve_time


@_log_timing
def build_graph_from_elevation(
    elevation: np.ndarray,
    weight: float = 1.0,
    diagonal_movement: str = "always",
    cost_model: str = "elevation",
) -> nx.DiGraph:
    """
    Build a directed graph from a 2D elevation array for pathfinding.

    Each pixel becomes a node with coordinates (i, j). Edges connect to
    neighboring pixels with weights based on physical distance and elevation cost.

    Edge weight formula depends on cost_model:
        elevation: distance_m * (1 + weight * normalized_elevation)
        distance:  distance_m
        slope_uphill: distance_m * (1 + weight * max(0, delta_elev) / elev_range)
        energy:    distance_m + weight * max(0, delta_elev)
        absolute_elevation: distance_m * (1 + weight * normalized_abs_elevation)
    where distance_m is 30m for cardinal directions, 30*sqrt(2) for diagonals.

    Args:
        elevation: 2D elevation array (not normalized).
        weight: Multiplier for elevation cost (higher = prefer flat paths).
        diagonal_movement: 'always' for 8-connectivity, 'never' for 4-connectivity.
        cost_model: edge cost model name.

    Returns:
        NetworkX DiGraph with nodes at (i, j) and weighted edges to neighbors.
    """
    t_start = time.time()
    rows, cols = elevation.shape
    logger.debug(
        f"build_graph_from_elevation: start rows={rows}, cols={cols}, "
        f"diagonal={diagonal_movement}, cost_model={cost_model}, weight={weight}, "
        f"graph_backend=edges"
    )

    if cost_model not in COST_MODELS:
        logger.warning(f"Unknown cost_model '{cost_model}', falling back to 'elevation'")
        cost_model = "elevation"

    # Reuse cached graph topology and only update edge weights
    t_template = time.time()
    template = _get_graph_template(rows, cols, diagonal_movement)
    logger.debug(f"build_graph_from_elevation: template fetch took {time.time() - t_template:.4f}s")

    t_weights = time.time()
    weights = _compute_edge_weights(template, elevation, weight, cost_model)
    logger.debug(f"build_graph_from_elevation: weights compute took {time.time() - t_weights:.4f}s")

    if template.distance_m.size == 0:
        return template.graph

    t_update = time.time()
    template.graph.add_weighted_edges_from(
        (
            (int(template.src_i[k]), int(template.src_j[k])),
            (int(template.dst_i[k]), int(template.dst_j[k])),
            float(weights[k]),
        )
        for k in range(len(template.src_i))
    )
    logger.debug(f"build_graph_from_elevation: edge weights update took {time.time() - t_update:.4f}s")
    logger.debug(f"build_graph_from_elevation: total time {time.time() - t_start:.4f}s")
    return template.graph


def _path_cost_from_graph(graph, path_list) -> float:
    """Compute total cost of a path using a prebuilt NetworkX graph's edge weights.

    This iterates over consecutive node pairs in `path_list` and sums the
    'weight' attribute from the graph edges. Falls back to Euclidean pixel
    distance converted to meters using PIXEL_SIZE_M if an edge weight is
    missing or an error occurs.
    """
    if graph is None or not path_list:
        return 0.0
    total = 0.0
    for a, b in zip(path_list, path_list[1:]):
        try:
            u = (int(a[0]), int(a[1]))
            v = (int(b[0]), int(b[1]))
            data = graph.get_edge_data(u, v) or {}
            weight = data.get("weight")
            if weight is not None:
                total += float(weight)
            else:
                di = b[0] - a[0]
                dj = b[1] - a[1]
                total += float(math.hypot(di, dj) * PIXEL_SIZE_M)
        except Exception:
            di = b[0] - a[0]
            dj = b[1] - a[1]
            total += float(math.hypot(di, dj) * PIXEL_SIZE_M)
    return total


# =============================================================================
# Numba-optimized grid-based pathfinding solvers
# =============================================================================
# These solvers operate directly on NumPy arrays and are JIT-compiled for
# maximum performance on high-resolution grids (e.g., 4096x4096).
# =============================================================================

# Cost model constants for Numba (avoid string comparisons in hot loops)
_COST_DISTANCE = 0
_COST_ELEVATION = 1
_COST_ABSOLUTE_ELEVATION = 2
_COST_SLOPE_UPHILL = 3
_COST_ENERGY = 4

_COST_MODEL_MAP = {
    "distance": _COST_DISTANCE,
    "elevation": _COST_ELEVATION,
    "absolute_elevation": _COST_ABSOLUTE_ELEVATION,
    "slope_uphill": _COST_SLOPE_UPHILL,
    "energy": _COST_ENERGY,
}


@njit(cache=True)
def _numba_heuristic_euclidean(i: int, j: int, ei: int, ej: int, pixel_size: float) -> float:
    """Euclidean heuristic scaled to meters."""
    di = float(i - ei)
    dj = float(j - ej)
    return pixel_size * math.sqrt(di * di + dj * dj)


@njit(cache=True)
def _numba_heuristic_manhattan(i: int, j: int, ei: int, ej: int, pixel_size: float) -> float:
    """Manhattan heuristic scaled to meters."""
    return pixel_size * (abs(i - ei) + abs(j - ej))


@njit(cache=True)
def _numba_heuristic_chebyshev(i: int, j: int, ei: int, ej: int, pixel_size: float) -> float:
    """Chebyshev heuristic scaled to meters."""
    return pixel_size * max(abs(i - ei), abs(j - ej))


@njit(cache=True)
def _numba_compute_step_cost(
    cost_model: int,
    dist_m: float,
    weight: float,
    elev_range: float,
    dest_factor_val: float,
    src_elev: float,
    dst_elev: float,
) -> float:
    """Compute step cost based on cost model (Numba-compatible)."""
    if cost_model == _COST_DISTANCE:
        return dist_m
    elif cost_model == _COST_ELEVATION or cost_model == _COST_ABSOLUTE_ELEVATION:
        return dist_m * dest_factor_val
    elif cost_model == _COST_SLOPE_UPHILL:
        if elev_range > 0.0:
            delta = dst_elev - src_elev
            slope_norm = max(0.0, delta) / elev_range
        else:
            slope_norm = 0.0
        return dist_m * (1.0 + weight * slope_norm)
    elif cost_model == _COST_ENERGY:
        delta = dst_elev - src_elev
        return dist_m + weight * max(0.0, delta)
    else:
        # Default to elevation model
        return dist_m * dest_factor_val


@njit(cache=True)
def _numba_heap_push(heap_data: np.ndarray, heap_size: int, f: float, g: float, i: int, j: int) -> int:
    """Push (f, g, i, j) onto heap and return new size. Binary min-heap by f-score."""
    idx = heap_size
    heap_data[idx, 0] = f
    heap_data[idx, 1] = g
    heap_data[idx, 2] = float(i)
    heap_data[idx, 3] = float(j)
    heap_size += 1

    # Bubble up
    while idx > 0:
        parent = (idx - 1) // 2
        if heap_data[parent, 0] <= heap_data[idx, 0]:
            break
        # Swap
        for k in range(4):
            tmp = heap_data[parent, k]
            heap_data[parent, k] = heap_data[idx, k]
            heap_data[idx, k] = tmp
        idx = parent

    return heap_size


@njit(cache=True)
def _numba_heap_pop(heap_data: np.ndarray, heap_size: int) -> tuple:
    """Pop minimum element from heap. Returns (f, g, i, j, new_size)."""
    f = heap_data[0, 0]
    g = heap_data[0, 1]
    i = int(heap_data[0, 2])
    j = int(heap_data[0, 3])

    heap_size -= 1
    if heap_size > 0:
        # Move last element to root
        for k in range(4):
            heap_data[0, k] = heap_data[heap_size, k]

        # Bubble down
        idx = 0
        while True:
            left = 2 * idx + 1
            right = 2 * idx + 2
            smallest = idx

            if left < heap_size and heap_data[left, 0] < heap_data[smallest, 0]:
                smallest = left
            if right < heap_size and heap_data[right, 0] < heap_data[smallest, 0]:
                smallest = right

            if smallest == idx:
                break

            # Swap
            for k in range(4):
                tmp = heap_data[idx, k]
                heap_data[idx, k] = heap_data[smallest, k]
                heap_data[smallest, k] = tmp
            idx = smallest

    return f, g, i, j, heap_size


@njit(cache=True)
def _numba_astar_core(
    crop: np.ndarray,
    dest_factor: np.ndarray,
    si: int,
    sj: int,
    ei: int,
    ej: int,
    weight: float,
    elev_range: float,
    cost_model: int,
    neighbors: np.ndarray,
    heuristic_type: int,
    pixel_size: float,
) -> np.ndarray:
    """
    Core A* algorithm implemented in Numba for maximum performance.

    Returns parent array of shape (rows, cols, 2) where parents[i,j] = (pi, pj).
    Returns array with all -1 if no path found.
    """
    rows, cols = crop.shape
    INF = 1e18

    # g_score: cost to reach each cell
    g_score = np.full((rows, cols), INF, dtype=np.float64)
    g_score[si, sj] = 0.0

    # Parent tracking for path reconstruction
    parents = np.full((rows, cols, 2), -1, dtype=np.int32)

    # Closed set
    closed = np.zeros((rows, cols), dtype=np.uint8)

    # Heap: preallocate for worst case (each cell pushed multiple times)
    # Use 4 columns: (f, g, i, j)
    max_heap_size = rows * cols * 4  # Conservative estimate
    heap_data = np.empty((max_heap_size, 4), dtype=np.float64)
    heap_size = 0

    # Initial heuristic
    if heuristic_type == 0:  # euclidean
        h0 = _numba_heuristic_euclidean(si, sj, ei, ej, pixel_size)
    elif heuristic_type == 1:  # manhattan
        h0 = _numba_heuristic_manhattan(si, sj, ei, ej, pixel_size)
    else:  # chebyshev
        h0 = _numba_heuristic_chebyshev(si, sj, ei, ej, pixel_size)

    heap_size = _numba_heap_push(heap_data, heap_size, h0, 0.0, si, sj)

    n_neighbors = neighbors.shape[0]

    while heap_size > 0:
        f, g, i, j, heap_size = _numba_heap_pop(heap_data, heap_size)

        if closed[i, j]:
            continue

        if i == ei and j == ej:
            return parents

        closed[i, j] = 1

        src_elev = crop[i, j]

        for n in range(n_neighbors):
            di = int(neighbors[n, 0])
            dj = int(neighbors[n, 1])
            dist_m = neighbors[n, 2]

            ni = i + di
            nj = j + dj

            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if closed[ni, nj]:
                continue

            dst_elev = crop[ni, nj]
            dest_factor_val = dest_factor[ni, nj]

            step_cost = _numba_compute_step_cost(
                cost_model, dist_m, weight, elev_range, dest_factor_val, src_elev, dst_elev
            )

            tentative_g = g + step_cost

            if tentative_g < g_score[ni, nj]:
                g_score[ni, nj] = tentative_g
                parents[ni, nj, 0] = i
                parents[ni, nj, 1] = j

                if heuristic_type == 0:
                    h = _numba_heuristic_euclidean(ni, nj, ei, ej, pixel_size)
                elif heuristic_type == 1:
                    h = _numba_heuristic_manhattan(ni, nj, ei, ej, pixel_size)
                else:
                    h = _numba_heuristic_chebyshev(ni, nj, ei, ej, pixel_size)

                f_score = tentative_g + h
                heap_size = _numba_heap_push(heap_data, heap_size, f_score, tentative_g, ni, nj)

    # No path found - return parents with start marked but no valid path
    return parents


@njit(cache=True)
def _numba_dijkstra_core(
    crop: np.ndarray,
    dest_factor: np.ndarray,
    si: int,
    sj: int,
    ei: int,
    ej: int,
    weight: float,
    elev_range: float,
    cost_model: int,
    neighbors: np.ndarray,
) -> np.ndarray:
    """
    Core Dijkstra algorithm implemented in Numba for maximum performance.

    Returns parent array of shape (rows, cols, 2) where parents[i,j] = (pi, pj).
    """
    rows, cols = crop.shape
    INF = 1e18

    g_score = np.full((rows, cols), INF, dtype=np.float64)
    g_score[si, sj] = 0.0

    parents = np.full((rows, cols, 2), -1, dtype=np.int32)
    closed = np.zeros((rows, cols), dtype=np.uint8)

    # Heap with 3 columns: (g, i, j)
    max_heap_size = rows * cols * 4
    heap_data = np.empty((max_heap_size, 3), dtype=np.float64)
    heap_size = 0

    # Push start
    heap_data[0, 0] = 0.0
    heap_data[0, 1] = float(si)
    heap_data[0, 2] = float(sj)
    heap_size = 1

    n_neighbors = neighbors.shape[0]

    while heap_size > 0:
        # Pop minimum
        g = heap_data[0, 0]
        i = int(heap_data[0, 1])
        j = int(heap_data[0, 2])

        heap_size -= 1
        if heap_size > 0:
            for k in range(3):
                heap_data[0, k] = heap_data[heap_size, k]
            # Bubble down
            idx = 0
            while True:
                left = 2 * idx + 1
                right = 2 * idx + 2
                smallest = idx
                if left < heap_size and heap_data[left, 0] < heap_data[smallest, 0]:
                    smallest = left
                if right < heap_size and heap_data[right, 0] < heap_data[smallest, 0]:
                    smallest = right
                if smallest == idx:
                    break
                for k in range(3):
                    tmp = heap_data[idx, k]
                    heap_data[idx, k] = heap_data[smallest, k]
                    heap_data[smallest, k] = tmp
                idx = smallest

        if closed[i, j]:
            continue

        if i == ei and j == ej:
            return parents

        closed[i, j] = 1
        src_elev = crop[i, j]

        for n in range(n_neighbors):
            di = int(neighbors[n, 0])
            dj = int(neighbors[n, 1])
            dist_m = neighbors[n, 2]

            ni = i + di
            nj = j + dj

            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if closed[ni, nj]:
                continue

            dst_elev = crop[ni, nj]
            dest_factor_val = dest_factor[ni, nj]

            step_cost = _numba_compute_step_cost(
                cost_model, dist_m, weight, elev_range, dest_factor_val, src_elev, dst_elev
            )

            tentative_g = g + step_cost

            if tentative_g < g_score[ni, nj]:
                g_score[ni, nj] = tentative_g
                parents[ni, nj, 0] = i
                parents[ni, nj, 1] = j

                # Push to heap
                idx = heap_size
                heap_data[idx, 0] = tentative_g
                heap_data[idx, 1] = float(ni)
                heap_data[idx, 2] = float(nj)
                heap_size += 1
                # Bubble up
                while idx > 0:
                    parent_idx = (idx - 1) // 2
                    if heap_data[parent_idx, 0] <= heap_data[idx, 0]:
                        break
                    for k in range(3):
                        tmp = heap_data[parent_idx, k]
                        heap_data[parent_idx, k] = heap_data[idx, k]
                        heap_data[idx, k] = tmp
                    idx = parent_idx

    return parents


@njit(cache=True)
def _numba_reconstruct_path(parents: np.ndarray, si: int, sj: int, ei: int, ej: int) -> np.ndarray:
    """Reconstruct path from parents array. Returns Nx2 array of (i, j) coordinates."""
    # First pass: count path length
    length = 0
    i, j = ei, ej
    while True:
        length += 1
        pi, pj = parents[i, j, 0], parents[i, j, 1]
        if pi == -1 and pj == -1:
            break
        i, j = pi, pj

    # Check if we reached start (valid path)
    if i != si or j != sj:
        # No valid path
        return np.empty((0, 2), dtype=np.int32)

    # Second pass: fill path array in reverse
    path = np.empty((length, 2), dtype=np.int32)
    i, j = ei, ej
    for idx in range(length - 1, -1, -1):
        path[idx, 0] = i
        path[idx, 1] = j
        pi, pj = parents[i, j, 0], parents[i, j, 1]
        if pi == -1 and pj == -1:
            break
        i, j = pi, pj

    return path


def _get_neighbors_array(diagonal_movement: str) -> np.ndarray:
    """Get neighbor offsets as numpy array for Numba compatibility."""
    if diagonal_movement == "never":
        return np.array(
            [
                [-1, 0, PIXEL_SIZE_M],
                [1, 0, PIXEL_SIZE_M],
                [0, -1, PIXEL_SIZE_M],
                [0, 1, PIXEL_SIZE_M],
            ],
            dtype=np.float64,
        )
    return np.array(
        [
            [-1, 0, PIXEL_SIZE_M],
            [1, 0, PIXEL_SIZE_M],
            [0, -1, PIXEL_SIZE_M],
            [0, 1, PIXEL_SIZE_M],
            [-1, -1, DIAGONAL_DISTANCE_M],
            [-1, 1, DIAGONAL_DISTANCE_M],
            [1, -1, DIAGONAL_DISTANCE_M],
            [1, 1, DIAGONAL_DISTANCE_M],
        ],
        dtype=np.float64,
    )


_HEURISTIC_MAP_NUMBA = {
    "euclidean": 0,
    "manhattan": 1,
    "chebyshev": 2,
}


def _grid_astar_numba(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    weight: float,
    diagonal_movement: str,
    heuristic_name: str,
    cost_model: str,
) -> list[tuple[int, int]]:
    """Numba-accelerated A* pathfinding on grid."""
    rows, cols = crop.shape
    si, sj = int(start[0]), int(start[1])
    ei, ej = int(end[0]), int(end[1])

    if si == ei and sj == ej:
        return [(si, sj)]

    # Prepare cost model
    cost_model_id = _COST_MODEL_MAP.get(cost_model, _COST_ELEVATION)

    # Prepare dest_factor and elev_range
    normalized, abs_norm, elev_range = _normalize_elevation_arrays(crop)
    if cost_model_id == _COST_ELEVATION:
        dest_factor = (1.0 + weight * normalized).astype(np.float64)
    elif cost_model_id == _COST_ABSOLUTE_ELEVATION:
        dest_factor = (1.0 + weight * abs_norm).astype(np.float64)
    else:
        dest_factor = np.ones((rows, cols), dtype=np.float64)

    # Ensure crop is float64 for Numba
    crop_f64 = crop.astype(np.float64)

    neighbors = _get_neighbors_array(diagonal_movement)
    heuristic_type = _HEURISTIC_MAP_NUMBA.get(heuristic_name, 0)

    parents = _numba_astar_core(
        crop_f64,
        dest_factor,
        si,
        sj,
        ei,
        ej,
        weight,
        elev_range,
        cost_model_id,
        neighbors,
        heuristic_type,
        PIXEL_SIZE_M,
    )

    # Check if path exists
    if parents[ei, ej, 0] == -1 and parents[ei, ej, 1] == -1:
        # Verify it's the start (path of length 1 - start==end already handled)
        if ei != si or ej != sj:
            return []

    path_arr = _numba_reconstruct_path(parents, si, sj, ei, ej)
    if path_arr.size == 0:
        return []

    return [(int(path_arr[i, 0]), int(path_arr[i, 1])) for i in range(path_arr.shape[0])]


def _grid_dijkstra_numba(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    weight: float,
    diagonal_movement: str,
    cost_model: str,
) -> list[tuple[int, int]]:
    """Numba-accelerated Dijkstra pathfinding on grid."""
    rows, cols = crop.shape
    si, sj = int(start[0]), int(start[1])
    ei, ej = int(end[0]), int(end[1])

    if si == ei and sj == ej:
        return [(si, sj)]

    cost_model_id = _COST_MODEL_MAP.get(cost_model, _COST_ELEVATION)

    normalized, abs_norm, elev_range = _normalize_elevation_arrays(crop)
    if cost_model_id == _COST_ELEVATION:
        dest_factor = (1.0 + weight * normalized).astype(np.float64)
    elif cost_model_id == _COST_ABSOLUTE_ELEVATION:
        dest_factor = (1.0 + weight * abs_norm).astype(np.float64)
    else:
        dest_factor = np.ones((rows, cols), dtype=np.float64)

    crop_f64 = crop.astype(np.float64)
    neighbors = _get_neighbors_array(diagonal_movement)

    parents = _numba_dijkstra_core(
        crop_f64,
        dest_factor,
        si,
        sj,
        ei,
        ej,
        weight,
        elev_range,
        cost_model_id,
        neighbors,
    )

    if parents[ei, ej, 0] == -1 and parents[ei, ej, 1] == -1:
        if ei != si or ej != sj:
            return []

    path_arr = _numba_reconstruct_path(parents, si, sj, ei, ej)
    if path_arr.size == 0:
        return []

    return [(int(path_arr[i, 0]), int(path_arr[i, 1])) for i in range(path_arr.shape[0])]


# Optimized grid-based solvers that operate directly on numpy arrays to avoid expensive NetworkX graph construction
def _grid_solvers_neighbors(diagonal_movement: str):
    if diagonal_movement == "never":
        return [(-1, 0, PIXEL_SIZE_M), (1, 0, PIXEL_SIZE_M), (0, -1, PIXEL_SIZE_M), (0, 1, PIXEL_SIZE_M)]
    return [
        (-1, 0, PIXEL_SIZE_M),
        (1, 0, PIXEL_SIZE_M),
        (0, -1, PIXEL_SIZE_M),
        (0, 1, PIXEL_SIZE_M),
        (-1, -1, DIAGONAL_DISTANCE_M),
        (-1, 1, DIAGONAL_DISTANCE_M),
        (1, -1, DIAGONAL_DISTANCE_M),
        (1, 1, DIAGONAL_DISTANCE_M),
    ]


def _reconstruct_path(parents, end_i, end_j):
    path = []
    i, j = end_i, end_j
    while True:
        path.append((i, j))
        pi, pj = parents[i, j]
        if pi == -1 and pj == -1:
            break
        i, j = int(pi), int(pj)
    path.reverse()
    return path


def _grid_astar(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    weight: float,
    diagonal_movement: str,
    heuristic_name: str,
    cost_model: str,
) -> list[tuple[int, int]]:
    """A* over grid using arrays; returns path as list of (i,j).

    Uses Numba-accelerated implementation when available for ~20-50x speedup
    on high-resolution grids.
    """
    # Use Numba-accelerated version when available
    if NUMBA_AVAILABLE:
        return _grid_astar_numba(crop, start, end, weight, diagonal_movement, heuristic_name, cost_model)

    # Fallback to pure Python implementation
    rows, cols = crop.shape
    si, sj = int(start[0]), int(start[1])
    ei, ej = int(end[0]), int(end[1])
    if si == ei and sj == ej:
        return [(si, sj)]

    if cost_model not in COST_MODELS:
        logger.warning(f"Unknown cost_model '{cost_model}', falling back to 'elevation'")
        cost_model = "elevation"

    if cost_model == "elevation":
        normalized, abs_norm, elev_range = _normalize_elevation_arrays(crop)
        dest_factor = 1.0 + weight * normalized
    elif cost_model == "absolute_elevation":
        normalized, abs_norm, elev_range = _normalize_elevation_arrays(crop)
        dest_factor = 1.0 + weight * abs_norm
    else:
        dest_factor = None
        _, _, elev_range = _normalize_elevation_arrays(crop)

    neighbors = _grid_solvers_neighbors(diagonal_movement)

    # Scores
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[si, sj] = 0.0
    parents = np.full((rows, cols, 2), -1, dtype=np.int32)

    # Heuristic function (returns meters)
    heuristic = HEURISTIC_MAP.get(heuristic_name, HEURISTIC_MAP["euclidean"])

    open_heap = []  # (f, g, i, j)
    f0 = heuristic((si, sj), (ei, ej))
    heapq.heappush(open_heap, (f0, 0.0, si, sj))

    closed = np.zeros((rows, cols), dtype=np.bool_)

    while open_heap:
        f, g, i, j = heapq.heappop(open_heap)
        if closed[i, j]:
            continue
        if i == ei and j == ej:
            return _reconstruct_path(parents, ei, ej)
        closed[i, j] = True

        for di, dj, dist_m in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if closed[ni, nj]:
                continue
            if cost_model == "distance":
                step_cost = dist_m
            elif cost_model == "absolute_elevation":
                step_cost = dist_m * float(dest_factor[ni, nj])
            elif cost_model == "slope_uphill":
                if elev_range > 0:
                    delta = float(crop[ni, nj] - crop[i, j])
                    slope_norm = max(0.0, delta) / elev_range
                else:
                    slope_norm = 0.0
                step_cost = dist_m * (1.0 + weight * slope_norm)
            elif cost_model == "energy":
                delta = float(crop[ni, nj] - crop[i, j])
                step_cost = dist_m + weight * max(0.0, delta)
            else:  # elevation
                step_cost = dist_m * float(dest_factor[ni, nj])

            tentative_g = g + step_cost
            if tentative_g < g_score[ni, nj]:
                g_score[ni, nj] = tentative_g
                parents[ni, nj, 0] = i
                parents[ni, nj, 1] = j
                f_score = tentative_g + heuristic((ni, nj), (ei, ej))
                heapq.heappush(open_heap, (f_score, tentative_g, ni, nj))

    # No path
    return []


def _grid_dijkstra(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    weight: float,
    diagonal_movement: str,
    cost_model: str,
) -> list[tuple[int, int]]:
    """Dijkstra over grid using arrays; equivalent to A* with zero heuristic.

    Uses Numba-accelerated implementation when available for ~20-50x speedup
    on high-resolution grids.
    """
    # Use Numba-accelerated version when available
    if NUMBA_AVAILABLE:
        return _grid_dijkstra_numba(crop, start, end, weight, diagonal_movement, cost_model)

    # Fallback to pure Python implementation
    rows, cols = crop.shape
    si, sj = int(start[0]), int(start[1])
    ei, ej = int(end[0]), int(end[1])
    if si == ei and sj == ej:
        return [(si, sj)]

    if cost_model not in COST_MODELS:
        cost_model = "elevation"
    normalized, abs_norm, elev_range = _normalize_elevation_arrays(crop)
    if cost_model == "elevation":
        dest_factor = 1.0 + weight * normalized
    elif cost_model == "absolute_elevation":
        dest_factor = 1.0 + weight * abs_norm
    else:
        dest_factor = None

    neighbors = _grid_solvers_neighbors(diagonal_movement)

    # Scores
    g_score = np.full((rows, cols), np.inf, dtype=np.float64)
    g_score[si, sj] = 0.0
    parents = np.full((rows, cols, 2), -1, dtype=np.int32)

    open_heap = []  # (g, i, j)
    heapq.heappush(open_heap, (0.0, si, sj))

    closed = np.zeros((rows, cols), dtype=np.bool_)

    while open_heap:
        g, i, j = heapq.heappop(open_heap)
        if closed[i, j]:
            continue
        if i == ei and j == ej:
            return _reconstruct_path(parents, ei, ej)
        closed[i, j] = True

        for di, dj, dist_m in neighbors:
            ni, nj = i + di, j + dj
            if not (0 <= ni < rows and 0 <= nj < cols):
                continue
            if closed[ni, nj]:
                continue
            if cost_model == "distance":
                step_cost = dist_m
            elif cost_model == "absolute_elevation":
                step_cost = dist_m * float(dest_factor[ni, nj])
            elif cost_model == "slope_uphill":
                if elev_range > 0:
                    delta = float(crop[ni, nj] - crop[i, j])
                    slope_norm = max(0.0, delta) / elev_range
                else:
                    slope_norm = 0.0
                step_cost = dist_m * (1.0 + weight * slope_norm)
            elif cost_model == "energy":
                delta = float(crop[ni, nj] - crop[i, j])
                step_cost = dist_m + weight * max(0.0, delta)
            else:  # elevation
                step_cost = dist_m * float(dest_factor[ni, nj])

            tentative_g = g + step_cost
            if tentative_g < g_score[ni, nj]:
                g_score[ni, nj] = tentative_g
                parents[ni, nj, 0] = i
                parents[ni, nj, 1] = j
                heapq.heappush(open_heap, (tentative_g, ni, nj))

    return []


def check_path_collision(path: list[tuple[int, int]], obstacles_mask: np.ndarray) -> bool:
    """
    Check if a path crosses any obstacles in the provided mask.

    Args:
        path: List of (row, col) tuples representing the path.
        obstacles_mask: 2D numpy array where 1 indicates an obstacle and 0 indicates free space.

    Returns:
        True if any point on the path coincides with an obstacle, False otherwise.
    """
    if not path or obstacles_mask is None:
        return False

    # Attempt fast vectorized path check for common case: sequence of (row, col)
    try:
        arr = np.asarray(path)
        # If object array of tuples/lists, stack into numeric 2D array
        if arr.dtype == np.object_ and arr.ndim == 1:
            arr = np.vstack(arr)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            rows = arr[:, 0].astype(np.intp)
            cols = arr[:, 1].astype(np.intp)
            # Keep only in-bounds points
            valid = (rows >= 0) & (rows < obstacles_mask.shape[0]) & (cols >= 0) & (cols < obstacles_mask.shape[1])
            if not np.any(valid):
                return False
            rows = rows[valid]
            cols = cols[valid]
            # Advanced indexing returns obstacle values for each point
            return bool(np.any(obstacles_mask[rows, cols] != 0))
    except Exception:
        # Fall through to safe scalar fallback on any error
        pass

    # Fallback: robust scalar check (handles objects with attributes, etc.)
    for p in path:
        try:
            i, j = int(p[0]), int(p[1])
        except Exception:
            try:
                i, j = int(p.y), int(p.x)  # support node-like objects if present
            except Exception:
                continue
        if 0 <= i < obstacles_mask.shape[0] and 0 <= j < obstacles_mask.shape[1]:
            if obstacles_mask[i, j] != 0:
                return True
    return False
