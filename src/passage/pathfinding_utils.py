"""Pathfinding utilities and helpers for PASSAGE."""

import math
import time

import networkx as nx
import numpy as np
from loguru import logger

from passage.utils import _log_timing

# Edge cost models
#   elevation         - distance * (1 + weight * normalized_elevation) [DEFAULT]
#   distance          - distance only (flat)
#   slope_uphill      - penalize uphill elevation changes (distance-scaled)
#   energy            - distance + weight * max(0, elevation_gain)
#   absolute_elevation- distance * (1 + weight * normalized_abs_elevation)
COST_MODELS = {"elevation", "distance", "slope_uphill", "energy", "absolute_elevation"}

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
    """A* over grid using arrays; returns path as list of (i,j)."""
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

    # Heuristic function (returns meters)
    heuristic = HEURISTIC_MAP.get(heuristic_name, HEURISTIC_MAP["euclidean"])

    import heapq

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
    """Dijkstra over grid using arrays; equivalent to A* with zero heuristic."""
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

    import heapq

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
    # Use numpy indexing for speed if possible, but path is usually a list of tuples
    for i, j in path:
        if obstacles_mask[i, j] == 1:
            return True
    return False
