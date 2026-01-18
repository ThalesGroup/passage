"""
Generate module for PASSAGE.

This module generates dataset samples from ALOS AW3D30 DSM tiles,
loading center tile and surrounding tiles to create cropped elevation images.

The generation process:
1. Draw a random elevation from the global min/max range (from calibrate.json).
2. Load tile_stats into a pandas DataFrame and filter tiles containing that elevation.
3. Select a random tile from the filtered set.
4. Load surrounding tiles as needed for the requested resolution.
5. Handle latitude bounds and longitude wrapping at -180/180.
"""

import json
import math
import multiprocessing
import os
import random
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import cv2
import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import yaml
from loguru import logger
from rasterio.windows import Window
from tqdm import tqdm

from passage.obstacles import sample_angle_from_config, sample_n_from_config, superellipse_mask
from passage.pathfinding_utils import (
    HEURISTIC_MAP,
    _find_path_pathfinding,
    _grid_astar,
    _grid_dijkstra,
    build_graph_from_elevation,
)
from passage.pathfinding_utils import (
    _get_graph_template as _get_graph_template,
    _build_pathfinding_cost_matrix as _build_pathfinding_cost_matrix,
    _pathfinding_path_to_coords as _pathfinding_path_to_coords,
    _compute_edge_weights as _compute_edge_weights,
    _normalize_elevation_arrays as _normalize_elevation_arrays,
)
from passage.utils import _log_timing
from passage.visualization import tensor_to_image


# Standard ALOS AW3D30 tile size in pixels (1° x 1° at 30m resolution)
TILE_SIZE_PX = 3600

# Latitude bounds
MIN_LAT = -90
MAX_LAT = 90

# Longitude bounds (wraps around)
MIN_LON = -180
MAX_LON = 180

# Solver backends
#   networkx   - NetworkX-based graph solvers [DEFAULT]
#   pathfinding- pathfinding package solvers (grid-based)
#   grid       - internal NumPy grid solvers (grid_astar/grid_dijkstra)

# NetworkX solver mapping
# Available solvers:
#   astar            - A* search with heuristic (fast, optimal) [DEFAULT]
#   astar_bi         - Bidirectional A* (faster for long paths)
#   dijkstra         - Dijkstra's algorithm (optimal, no heuristic)
#   dijkstra_bi      - Bidirectional Dijkstra (faster for long paths)
#   bellman_ford     - Bellman-Ford algorithm (handles negative weights, slowest)
#   unweighted       - BFS-based shortest path (ignores weights)

# Pathfinding package solvers:
#   astar            - A* search with heuristic
#   dijkstra         - Dijkstra's algorithm
#   best_first       - Best-first search (greedy)

# Example config:
#   solver:
#     backend: networkx
#     name: astar
#     diagonal_movement: always
#     weight: 30.0
#     heuristic: euclidean  # only for astar/astar_bi: euclidean, manhattan, chebyshev
NETWORKX_SOLVERS = {"astar", "astar_bi", "dijkstra", "dijkstra_bi", "bellman_ford", "unweighted"}
PATHFINDING_SOLVERS = {"astar", "dijkstra", "best_first"}
GRID_SOLVERS = {"grid_astar", "grid_dijkstra"}


def _parse_solver_backend_and_name(solver_config: dict) -> tuple[str, str]:
    raw_name = solver_config.get("name", "astar")
    backend = solver_config.get("backend")
    solver_name = raw_name

    if isinstance(raw_name, str) and ":" in raw_name:
        prefix, name = raw_name.split(":", 1)
        backend = prefix
        solver_name = name

    if not backend:
        backend = "networkx"

    if backend == "grid":
        if solver_name == "astar":
            solver_name = "grid_astar"
        elif solver_name == "dijkstra":
            solver_name = "grid_dijkstra"

    return backend, solver_name


def _resolve_solver_config(solver_config: dict) -> dict:
    """Merge common + backend-specific solver config sections.

    Accepts either a flat config or a structured config like:
    solver:
      name: networkx:astar
      common: {...}
      networkx: {...}
      pathfinding: {...}
      grid: {...}
    """
    if not isinstance(solver_config, dict):
        return {}

    has_sections = any(key in solver_config for key in ("common", "networkx", "pathfinding", "grid"))
    if not has_sections:
        return solver_config

    backend, _ = _parse_solver_backend_and_name(solver_config)
    resolved = {}
    resolved.update(solver_config.get("common", {}))
    resolved.update(solver_config.get(backend, {}))
    if "name" in solver_config:
        resolved["name"] = solver_config["name"]
    return resolved


@_log_timing
def find_path(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    solver_config: dict,
    prebuilt_graph: nx.DiGraph | None = None,
) -> list[tuple[int, int]]:
    """
    Find a path between two points on the crop using configured solver backend.

    Args:
        crop: 2D elevation array (not normalized).
        start: (i, j) start coordinates.
        end: (i, j) end coordinates.
        solver_config: Solver configuration dict with:
            - backend: 'networkx' (default), 'pathfinding', or 'grid'
            - name: solver name for selected backend
            - diagonal_movement: 'always' (default) or 'never'
            - weight: float multiplier for elevation cost (default 1.0)
            - heuristic: 'euclidean' (default), 'manhattan', 'chebyshev' (for astar/best_first)
            - cost_model: 'elevation' (default), 'distance', 'slope_uphill', 'energy', 'absolute_elevation'
        prebuilt_graph: Optional prebuilt NetworkX DiGraph. If provided and the selected backend is
            'networkx', the prebuilt graph will be used and graph construction will be skipped.

    Returns:
        List of (i, j) tuples representing the path, or empty list if no path found.
    """
    rows, cols = crop.shape
    grid_size = (int(rows), int(cols))

    # Validate start/end coordinates
    si, sj = int(start[0]), int(start[1])
    ei, ej = int(end[0]), int(end[1])
    if not (0 <= si < rows and 0 <= sj < cols and 0 <= ei < rows and 0 <= ej < cols):
        logger.debug(f"find_path: start/end out of bounds for grid_size={grid_size}, start={start}, end={end}")
        return []

    # Extract solver configuration
    solver_config = _resolve_solver_config(solver_config)
    backend, solver_name = _parse_solver_backend_and_name(solver_config)
    if backend not in {"networkx", "pathfinding", "grid"}:
        logger.warning(f"Unknown backend '{backend}', falling back to networkx")
        backend = "networkx"
    diagonal_name = solver_config.get("diagonal_movement", "always")
    weight = solver_config.get("weight", 1.0)
    heuristic_name = solver_config.get("heuristic", "euclidean")
    cost_model = solver_config.get("cost_model", "elevation")
    logger.debug(
        f"find_path: backend={backend}, solver={solver_name}, diagonal={diagonal_name}, weight={weight}, "
        f"heuristic={heuristic_name}, cost_model={cost_model}, graph_backend=edges, "
        f"grid_size={grid_size}, start={start}, end={end}"
    )

    # For grid-based solvers we avoid constructing a NetworkX graph (faster)
    if backend == "grid" or solver_name in GRID_SOLVERS:
        grid_solver = solver_name if solver_name in GRID_SOLVERS else "grid_astar"
        if grid_solver == "grid_dijkstra" or solver_name == "dijkstra":
            grid_solver = "grid_dijkstra"
        t0 = time.time()
        if grid_solver == "grid_astar":
            path = _grid_astar(crop, (si, sj), (ei, ej), weight, diagonal_name, heuristic_name, cost_model)
        else:
            path = _grid_dijkstra(crop, (si, sj), (ei, ej), weight, diagonal_name, cost_model)
        solve_time = time.time() - t0
        logger.debug(
            f"Grid solver finished: path_len={len(path)}, solve_time={solve_time:.4f}s, total={solve_time:.4f}s"
        )
        return list(path)

    # Pathfinding backend
    if backend == "pathfinding":
        if solver_name not in PATHFINDING_SOLVERS:
            logger.warning(f"Unknown pathfinding solver '{solver_name}', falling back to astar")
            solver_name = "astar"

        if cost_model in {"slope_uphill", "energy"}:
            logger.warning("Pathfinding backend does not support directional cost models; falling back to networkx")
        else:
            path, prep_time, solve_time = _find_path_pathfinding(
                crop,
                (si, sj),
                (ei, ej),
                solver_name,
                diagonal_name,
                weight,
                heuristic_name,
                cost_model,
            )
            total_time = prep_time + solve_time
            logger.debug(
                "Pathfinding solver finished: "
                f"path_len={len(path)}, prep_time={prep_time:.4f}s, solve_time={solve_time:.4f}s, total={total_time:.4f}s"
            )
            return list(path)

    # NetworkX backend
    if solver_name not in NETWORKX_SOLVERS:
        logger.warning(f"Unknown networkx solver '{solver_name}', falling back to astar")
        solver_name = "astar"

    if prebuilt_graph is not None:
        G = prebuilt_graph
        graph_time = 0.0
        # Ensure t0 exists so timing calculations later don't raise UnboundLocalError
        t0 = time.time()
        logger.debug("Using prebuilt NetworkX graph; skipping build_graph_from_elevation()")
    else:
        t0 = time.time()
        G = build_graph_from_elevation(
            crop,
            weight=weight,
            diagonal_movement=diagonal_name,
            cost_model=cost_model,
        )
        graph_time = time.time() - t0
    logger.debug(
        f"Graph construction: nodes={G.number_of_nodes()}, edges={G.number_of_edges()}, time={graph_time:.4f}s"
    )

    # Convert coordinates to tuples for NetworkX
    start_node = (si, sj)
    end_node = (ei, ej)

    # Find path using selected solver
    t1 = time.time()
    try:
        if solver_name == "astar":
            heuristic = HEURISTIC_MAP.get(heuristic_name, HEURISTIC_MAP["euclidean"])
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="weight")
        elif solver_name == "astar_bi":
            heuristic = HEURISTIC_MAP.get(heuristic_name, HEURISTIC_MAP["euclidean"])
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="weight")
            # Note: NetworkX doesn't have native bidirectional A*, using standard A*
            # For true bidirectional, consider using bidirectional_dijkstra as alternative
        elif solver_name == "dijkstra":
            path = nx.dijkstra_path(G, start_node, end_node, weight="weight")
        elif solver_name == "dijkstra_bi":
            # Bidirectional Dijkstra - returns (length, path)
            _, path = nx.bidirectional_dijkstra(G, start_node, end_node, weight="weight")
        elif solver_name == "bellman_ford":
            path = nx.bellman_ford_path(G, start_node, end_node, weight="weight")
        elif solver_name == "unweighted":
            path = nx.shortest_path(G, start_node, end_node)
        else:
            heuristic = HEURISTIC_MAP.get(heuristic_name, HEURISTIC_MAP["euclidean"])
            path = nx.astar_path(G, start_node, end_node, heuristic=heuristic, weight="weight")
    except nx.NetworkXNoPath:
        solve_time = time.time() - t1
        logger.debug(f"No path found: solve_time={solve_time:.4f}s")
        return []
    except Exception as e:
        solve_time = time.time() - t1
        logger.debug(f"Pathfinding error after {solve_time:.4f}s: {e}")
        raise

    solve_time = time.time() - t1
    total_time = time.time() - t0
    logger.debug(
        f"Pathfinding finished: path_len={len(path)}, graph_time={graph_time:.4f}s, solve_time={solve_time:.4f}s, total={total_time:.4f}s"
    )

    return list(path)


# @_log_timing
def lat_to_str(lat_deg: int) -> str:
    """
    Convert latitude degree to string format (e.g., N042 or S010).

    Args:
        lat_deg: Latitude in degrees (-90 to +90).

    Returns:
        String in format [N|S]XXX.
    """
    prefix = "N" if lat_deg >= 0 else "S"
    return f"{prefix}{min(abs(lat_deg), 90):03d}"


# @_log_timing
def lon_to_str(lon_deg: int) -> str:
    """
    Convert longitude degree to string format (e.g., E001 or W120).

    Args:
        lon_deg: Longitude in degrees (-180 to +180).

    Returns:
        String in format [E|W]XXX.
    """
    prefix = "E" if lon_deg >= 0 else "W"
    return f"{prefix}{min(abs(lon_deg), 180):03d}"


# @_log_timing
def parse_lat_lon_from_filename(filename: str) -> tuple[int, int] | None:
    """Extract latitude and longitude from a DSM tile filename."""
    pattern = r"([NS])(\d{3})([EW])(\d{3})"
    match = re.search(pattern, filename)
    if not match:
        return None

    lat_char, lat_val, lon_char, lon_val = match.groups()
    lat = int(lat_val)
    lon = int(lon_val)

    if lat_char == "S":
        lat = -lat
    if lon_char == "W":
        lon = -lon

    return lat, lon


# @_log_timing
def wrap_lon(lon: int) -> int:
    """Wrap longitude to [-180, 180) range."""
    if lon < MIN_LON:
        return lon + 360
    elif lon >= MAX_LON:
        return lon - 360
    return lon


@_log_timing
def load_tile(tiles_dirpath: Path, lat: int, lon: int) -> np.ndarray:
    """
    Load a DSM tile for the given lat/lon coordinates.

    Args:
        tiles_dirpath: Directory containing DSM tiles.
        lat: Latitude in degrees.
        lon: Longitude in degrees.

    Returns:
        Numpy array of elevation data (TILE_SIZE_PX x TILE_SIZE_PX).
        Returns zeros array if tile not found.
    """
    lat_str = lat_to_str(lat)
    lon_str = lon_to_str(lon)

    # Search for tile file without glob - try deterministic filenames first, then a single-pass directory scan.
    candidate_names = [
        f"{lat_str}{lon_str}_DSM.tif",
        f"{lat_str}{lon_str}_AW3D30_DSM.tif",
        f"{lat_str}{lon_str}_AW3D30.tif",
        f"{lat_str}{lon_str}.tif",
        f"ALPSMLC30_{lat_str}{lon_str}_DSM.tif",
    ]

    tile_file = None
    # Try exact candidate filenames first
    for name in candidate_names:
        candidate = tiles_dirpath / name
        if candidate.exists():
            tile_file = candidate
            break

    if tile_file is None:
        # Single-pass directory scan: pick the first file that contains the lat/lon key and looks like a DSM TIFF
        key = f"{lat_str}{lon_str}"
        for p in tiles_dirpath.iterdir():
            if p.is_file() and key in p.name and p.name.endswith("_DSM.tif"):
                tile_file = p
                break
        # Fallback: any .tif containing the key
        if tile_file is None:
            for p in tiles_dirpath.iterdir():
                if p.is_file() and key in p.name and p.name.lower().endswith(".tif"):
                    tile_file = p
                    break

    if tile_file is not None:
        logger.debug(f"Using tile file: {tile_file}")
        try:
            with rasterio.open(tile_file) as src:
                data = src.read(1).astype(np.float32)
            if data.shape != (TILE_SIZE_PX, TILE_SIZE_PX):
                try:
                    data = cv2.resize(data, (TILE_SIZE_PX, TILE_SIZE_PX), interpolation=cv2.INTER_NEAREST)
                except Exception:  # pragma: no cover
                    logger.warning(f"Unexpected tile shape {data.shape} for {tile_file}, using zero tile")
                    data = np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32)
            return data
        except Exception as e:  # pragma: no cover
            logger.debug(f"Error reading tile file {tile_file}: {e}")
            logger.warning(f"Failed to load tile {tile_file} with rasterio: {e}")

    # Return flat elevation map if tile not found
    logger.debug(f"Tile not found for lat={lat}, lon={lon} in directory {tiles_dirpath}; returning zeros")
    data = np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32)
    return data


@_log_timing
def load_tile_grid(
    tiles_dirpath: Path,
    center_lat: int,
    center_lon: int,
    half_tiles: int = 1,
) -> np.ndarray:
    """
    Load a (2*half_tiles+1) x (2*half_tiles+1) grid of tiles centered on the given coordinates.

    Handles latitude bounds by substituting flat tiles for out-of-bounds rows.
    Handles longitude wrapping at -180/180.
    """
    import time

    start_time = time.time()
    rows = []
    N = 2 * half_tiles + 1

    for dlat in range(half_tiles, -half_tiles - 1, -1):
        lat = center_lat + dlat
        if lat < MIN_LAT or lat >= MAX_LAT:
            # Substitute zero tiles for out-of-bounds latitudes (consistent with earlier behavior)
            logger.debug(f"Latitude {lat} out of bounds [{MIN_LAT}, {MAX_LAT}); substituting zero tiles")
            row_tiles = [np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32) for _ in range(N)]
        else:
            row_tiles = []
            for dlon in range(-half_tiles, half_tiles + 1):
                lon = wrap_lon(center_lon + dlon)
                tile = load_tile(tiles_dirpath, lat, lon)
                row_tiles.append(tile)
        rows.append(np.concatenate(row_tiles, axis=1))

    grid = np.concatenate(rows, axis=0)  # Vertical concat
    duration = time.time() - start_time
    logger.debug(
        f"load_tile_grid center=({center_lat},{center_lon}) half_tiles={half_tiles} N={N} grid_shape={grid.shape} load_duration={duration:.2f}s"
    )
    return grid


@_log_timing
def load_tile_window(
    tiles_dirpath: Path,
    lat: int,
    lon: int,
    row_off: int,
    col_off: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Load a rectangular window from a tile (row_off, col_off, height, width).

    Returns zeros if tile does not exist or window is empty.
    """
    lat_str = lat_to_str(lat)
    lon_str = lon_to_str(lon)

    glob_pattern = f"*{lat_str}{lon_str}*_DSM.tif"
    tile_files = list(Path(tiles_dirpath).glob(glob_pattern))

    if len(tile_files) == 0:
        return np.zeros((height, width), dtype=np.float32)

    try:
        with rasterio.open(tile_files[0]) as src:
            # Clamp window to tile bounds
            win = Window(col_off, row_off, width, height)
            arr = src.read(1, window=win).astype(np.float32)
            if arr.shape != (height, width):
                # If smaller due to window exceeding bounds, pad
                padded = np.zeros((height, width), dtype=np.float32)
                h, w = arr.shape
                padded[:h, :w] = arr
                arr = padded
        return arr
    except Exception as e:  # pragma: no cover - rare read error
        logger.warning(f"Failed to read window from tile {tile_files[0]}: {e}")
        return np.zeros((height, width), dtype=np.float32)


@_log_timing
def _assemble_crop_from_tiles(
    tiles_dirpath: Path,
    center_lat: int,
    center_lon: int,
    half_tiles: int,
    y1: int,
    x1: int,
    resolution: int,
    center_tile: np.ndarray,
) -> np.ndarray:
    """Assemble a crop of size (resolution, resolution) from tile windows.

    The center tile is provided (full TILE_SIZE_PX x TILE_SIZE_PX) to avoid reloading it.
    """
    N = 2 * half_tiles + 1
    y2 = y1 + resolution
    x2 = x1 + resolution
    crop = np.zeros((resolution, resolution), dtype=np.float32)

    for row_idx in range(N):
        dlat = half_tiles - row_idx
        tile_lat = center_lat + dlat
        tile_top = row_idx * TILE_SIZE_PX
        for col_idx in range(N):
            dlon = -half_tiles + col_idx
            tile_lon = wrap_lon(center_lon + dlon)
            tile_left = col_idx * TILE_SIZE_PX

            # Overlap between tile and desired crop in grid coordinates
            overlap_y1 = max(0, y1 - tile_top)
            overlap_y2 = min(TILE_SIZE_PX, y2 - tile_top)
            overlap_x1 = max(0, x1 - tile_left)
            overlap_x2 = min(TILE_SIZE_PX, x2 - tile_left)

            h = overlap_y2 - overlap_y1
            w = overlap_x2 - overlap_x1
            if h <= 0 or w <= 0:
                continue

            # Destination indices in crop
            dest_y1 = tile_top + overlap_y1 - y1
            dest_y2 = dest_y1 + h
            dest_x1 = tile_left + overlap_x1 - x1
            dest_x2 = dest_x1 + w

            # If this is the center tile, use the provided array
            if dlat == 0 and dlon == 0:
                block = center_tile[int(overlap_y1) : int(overlap_y2), int(overlap_x1) : int(overlap_x2)]
            else:
                block = load_tile_window(
                    tiles_dirpath, tile_lat, tile_lon, int(overlap_y1), int(overlap_x1), int(h), int(w)
                )

            crop[dest_y1:dest_y2, dest_x1:dest_x2] = block

    return crop


def _resolve_obstacle_config(config: dict | None) -> dict:
    """Normalize obstacle settings from config.yaml.

    Returns a dict with keys:
      - enabled
      - ratio_max
      - size_min_ratio
      - size_max_ratio
      - generation_cfg
      - obstacle_penalty
      - config (raw obstacles config)
    """
    config = config or {}
    obstacles_cfg = config.get("obstacles", {}) if isinstance(config, dict) else {}

    enabled = bool(obstacles_cfg.get("enabled", False))
    ratio_max = float(obstacles_cfg.get("ratio_max", 0.0))

    shape_cfg = obstacles_cfg.get("shape", {}) or {}
    size_cfg = shape_cfg.get("size", {}) or {}
    if not size_cfg:
        # Backward compatibility
        size_cfg = {
            "min_ratio": shape_cfg.get("size_min_ratio", obstacles_cfg.get("size_min_ratio", 0.20)),
            "max_ratio": shape_cfg.get("size_max_ratio", obstacles_cfg.get("size_max_ratio", 0.50)),
        }

    size_min_ratio = float(size_cfg.get("min_ratio", 0.20))
    size_max_ratio = float(size_cfg.get("max_ratio", 0.50))

    generation_cfg = obstacles_cfg.get("generation", {}) or {}
    obstacle_penalty = float(obstacles_cfg.get("obstacle_penalty", 1e8))

    return {
        "enabled": enabled,
        "ratio_max": ratio_max,
        "size_min_ratio": size_min_ratio,
        "size_max_ratio": size_max_ratio,
        "generation_cfg": generation_cfg,
        "obstacle_penalty": obstacle_penalty,
        "config": obstacles_cfg,
    }


def _generate_obstacles_mask(
    shape: tuple[int, int],
    ratio_max: float,
    size_min_ratio: float,
    size_max_ratio: float,
    obstacles_cfg: dict,
    marker_positions: list[tuple[int, int]] | None = None,
    rng: np.random.Generator | None = None,
    max_attempts: int | None = None,
    generation_cfg: dict | None = None,
) -> tuple[np.ndarray, float]:
    """Generate a binary obstacle mask using randomized superellipses.

    Returns (mask, target_ratio).
    """
    rng = np.random.default_rng() if rng is None else rng
    height, width = shape
    ratio_max = float(max(0.0, ratio_max))

    marker_positions = [] if marker_positions is None else [tuple(map(int, p)) for p in marker_positions]

    gen = generation_cfg if generation_cfg is not None else obstacles_cfg.get("generation", {})
    max_attempts = int(gen.get("max_attempts", 50)) if max_attempts is None else int(max_attempts)
    candidates_per_mask = int(gen.get("candidates_per_mask", 200))

    for _ in range(max_attempts):
        target_ratio = float(rng.uniform(0.0, ratio_max))
        mask = np.zeros((height, width), dtype=np.uint8)

        if target_ratio <= 0.0:
            return mask, target_ratio

        size_min_ratio = float(max(0.0, size_min_ratio))
        size_max_ratio = float(max(size_min_ratio, size_max_ratio))

        min_w = max(1, int(size_min_ratio * width))
        max_w = max(1, int(size_max_ratio * width))
        min_h = max(1, int(size_min_ratio * height))
        max_h = max(1, int(size_max_ratio * height))

        abort_attempt = False
        for _ in range(candidates_per_mask):
            candidate_retry = 0
            while True:
                w = int(rng.integers(min_w, max_w + 1))
                h = int(rng.integers(min_h, max_h + 1))
                cx = int(rng.integers(0, width))
                cy = int(rng.integers(0, height))
                angle = float(sample_angle_from_config(rng))
                n = float(sample_n_from_config(obstacles_cfg, rng))

                a = max(w / 2.0, 1.0)
                b = max(h / 2.0, 1.0)
                candidate = superellipse_mask(
                    (height, width), (cy, cx), a, b, n, angle=angle, angle_degrees=False
                ).astype(np.uint8)

                if any(candidate[i, j] == 1 for i, j in marker_positions):
                    candidate_retry += 1
                    if candidate_retry < 10:
                        continue
                    abort_attempt = True
                    break

                new_ratio = float(np.mean(mask | candidate))
                if new_ratio > target_ratio:
                    abort_attempt = True
                    break

                mask = np.logical_or(mask, candidate).astype(np.uint8)
                break

            if abort_attempt:
                break

        if any(mask[i, j] == 1 for i, j in marker_positions):
            continue

        return mask, target_ratio

    return np.zeros((height, width), dtype=np.uint8), 0.0


@_log_timing
def tiles_needed_for_resolution(resolution: int) -> tuple[int, int, int]:
    """
    Compute how many tiles are concatenated to produce a crop for the given resolution.

    Returns a tuple of:
        - N: number of tiles along one axis (2*half_tiles + 1)
        - total_tiles: total tiles concatenated (N*N)
        - half_tiles: number of tiles on each side of the center tile

    Example: for small resolutions half_tiles==1 -> N=3 -> total_tiles=9
    """
    half_res = resolution // 2
    half_tiles = max(1, math.ceil(half_res / TILE_SIZE_PX))
    N = 2 * half_tiles + 1
    total_tiles = N * N
    return N, total_tiles, half_tiles


@_log_timing
def generate_sample(
    args: tuple[int, Path, Path, int, pd.DataFrame, float, float, bool, dict, dict],
) -> tuple[int, bool, str]:
    """
    Generate a single dataset sample.

    Args:
          args: Tuple of (index, tiles_dirpath, outputs_dirpath, resolution,
              tile_stats_df, global_min, global_max, disable_images, solver_config, obstacle_settings).

    Returns:
        Tuple of (index, success, message).
    """
    (
        index,
        tiles_dirpath,
        outputs_dirpath,
        resolution,
        tile_stats_df,
        global_min,
        global_max,
        disable_images,
        solver_config,
        obstacle_settings,
    ) = args

    solver_config = _resolve_solver_config(solver_config)
    obstacle_settings = obstacle_settings or {}
    max_marker_retries = solver_config.get("max_marker_retries", 10)
    backend, solver_name = _parse_solver_backend_and_name(solver_config)
    max_elevation_retries = 10  # Total attempts for different elevations

    start_time = datetime.now()
    elevation_attempts = 0

    half_res = resolution // 2
    half_tiles = max(1, math.ceil(half_res / TILE_SIZE_PX))

    for elevation_attempt in range(max_elevation_retries):
        elevation_attempts += 1
        try:
            # New sampling strategy: pick a random tile first, then draw target elevation from that tile's range.
            # This reduces failed elevation attempts caused by drawing elevations not covered by any tile.
            candidates = tile_stats_df

            target_elev = random.uniform(float(global_min), float(global_max))

            candidates = candidates[(candidates["min"] <= target_elev) & (candidates["max"] >= target_elev)]

            if candidates.empty:
                continue  # No valid tiles for this resolution

            # Select a random tile and draw a target elevation from its local range
            selected = candidates.sample(n=1).iloc[0]
            center_lat = int(selected["lat"])
            center_lon = int(selected["lon"])

            # 5. Load the full grid of tiles (may be larger but simplifies processing)
            logger.debug(f"Loading full grid for center=({center_lat},{center_lon}) half_tiles={half_tiles}")
            grid = load_tile_grid(tiles_dirpath, center_lat, center_lon, half_tiles=half_tiles)
            logger.debug(f"grid min/max: {grid.min()}/{grid.max()}")

            # Offset (in pixels) from grid origin to start of center tile
            center_offset = TILE_SIZE_PX * half_tiles
            grid_h, grid_w = grid.shape

            # 6. Find the pixel with the closest elevation to target_elev within the center tile region
            center_tile_y1 = center_offset
            center_tile_y2 = center_offset + TILE_SIZE_PX
            center_tile_x1 = center_offset
            center_tile_x2 = center_offset + TILE_SIZE_PX

            center_tile_region = grid[center_tile_y1:center_tile_y2, center_tile_x1:center_tile_x2]

            # Find pixel with minimum absolute difference to target_elev
            abs_diff = np.abs(center_tile_region - target_elev)
            min_idx = np.argmin(abs_diff)
            target_row, target_col = divmod(min_idx, TILE_SIZE_PX)

            # Convert to grid coordinates
            target_grid_y = center_tile_y1 + target_row
            target_grid_x = center_tile_x1 + target_col

            # 7. Generate a random crop containing the target pixel with random offsets
            crop_y1_min = max(0, target_grid_y - resolution + 1)
            crop_y1_max = min(grid_h - resolution, target_grid_y)
            crop_x1_min = max(0, target_grid_x - resolution + 1)
            crop_x1_max = min(grid_w - resolution, target_grid_x)

            if crop_y1_max < crop_y1_min or crop_x1_max < crop_x1_min:
                continue  # Try another elevation

            # Randomly select crop position
            y1 = random.randint(crop_y1_min, crop_y1_max)
            x1 = random.randint(crop_x1_min, crop_x1_max)
            y2 = y1 + resolution
            x2 = x1 + resolution

            # Extract the crop directly from the loaded grid
            crop = grid[y1:y2, x1:x2]

            # 8. Calculate marker 1 position (the target pixel) in crop coordinates
            i1 = target_grid_y - y1
            j1 = target_grid_x - x1

            # 9. Try to find a path with retry logic
            h, w = crop.shape
            total_pixels = h * w
            marker1_flat_idx = i1 * w + j1

            # Optionally prebuild NetworkX graph once for this crop when networkx solver is selected
            prebuilt_graph = None
            if backend == "networkx" or solver_name in NETWORKX_SOLVERS:
                try:
                    prebuilt_graph = build_graph_from_elevation(
                        crop,
                        weight=solver_config.get("weight", 1.0),
                        diagonal_movement=solver_config.get("diagonal_movement", "always"),
                        cost_model=solver_config.get("cost_model", "elevation"),
                    )
                except Exception as e:
                    logger.warning(f"Failed to prebuild NetworkX graph for crop: {e}")
                    prebuilt_graph = None

            path_without = []
            marker_attempts_used = 0
            for marker_attempt in range(max_marker_retries):
                marker_attempts_used += 1
                # Generate marker 2 randomly, distinct from marker 1
                idx2 = random.randint(0, total_pixels - 2)
                if idx2 >= marker1_flat_idx:
                    idx2 += 1
                i2, j2 = divmod(idx2, w)

                # Try to find path between markers (without obstacles)
                path_without = find_path(crop, (i1, j1), (i2, j2), solver_config, prebuilt_graph=prebuilt_graph)
                if len(path_without) > 0:
                    break  # Found a valid path

            if len(path_without) == 0:
                logger.debug(f"No path found after {marker_attempts_used} marker attempts; retrying elevation")
                continue  # Try another elevation

            # Generate obstacles mask (if enabled) and compute path with obstacles
            obstacles_cfg = obstacle_settings.get("config", {})
            obstacles_enabled = bool(obstacle_settings.get("enabled", False))
            obstacle_ratio_max = float(obstacle_settings.get("ratio_max", 0.0))
            size_min_ratio = float(obstacle_settings.get("size_min_ratio", 0.20))
            size_max_ratio = float(obstacle_settings.get("size_max_ratio", 0.50))
            generation_cfg = obstacle_settings.get("generation_cfg", {})
            obstacle_penalty = float(obstacle_settings.get("obstacle_penalty", 1e8))

            marker_positions = [(i1, j1), (i2, j2)]
            if obstacles_enabled:
                obstacles_mask, obstacle_target_ratio = _generate_obstacles_mask(
                    crop.shape,
                    obstacle_ratio_max,
                    size_min_ratio,
                    size_max_ratio,
                    obstacles_cfg,
                    marker_positions=marker_positions,
                    generation_cfg=generation_cfg,
                )
            else:
                obstacles_mask = np.zeros_like(crop, dtype=np.uint8)
                obstacle_target_ratio = 0.0

            obstacle_ratio = float(np.mean(obstacles_mask))

            masked_elevation = crop.copy()
            if obstacles_enabled and obstacle_penalty > 0:
                masked_elevation[obstacles_mask == 1] = masked_elevation.max() + obstacle_penalty

            path_with = find_path(masked_elevation, (i1, j1), (i2, j2), solver_config)

            # Successfully found a path - proceed with sample generation
            # Normalize to float32 in [0, 1] using global_min/global_max
            crop_f = crop.astype(np.float32)
            logger.debug(f"crop_f min/max before normalization: {crop_f.min()}/{crop_f.max()}")
            crop_norm = (crop_f - global_min) / (global_max - global_min)
            logger.debug(f"crop_norm min/max after normalization: {crop_norm.min()}/{crop_norm.max()}")
            crop_norm = np.clip(crop_norm, 0.0, 1.0).astype(np.float32)
            logger.debug(f"crop_norm min/max after clipping: {crop_norm.min()}/{crop_norm.max()}")

            # Create marker channel: zeros with two pixels set to 1
            marker_channel = np.zeros_like(crop_norm, dtype=np.float32)
            marker_channel[i1, j1] = 1.0
            marker_channel[i2, j2] = 1.0

            # Extract elevation values before normalization for metadata
            tile_min_elev = float(selected["min"])
            tile_max_elev = float(selected["max"])
            crop_min_elev = float(np.min(crop))
            crop_max_elev = float(np.max(crop))
            marker_1_elev = float(crop[i1, j1])
            marker_2_elev = float(crop[i2, j2])

            # Stack elevation, marker, and obstacle channels: shape (H, W, 3)
            obstacles_channel = obstacles_mask.astype(np.float32)
            crop_with_markers = np.stack([crop_norm, marker_channel, obstacles_channel], axis=-1)

            # Save as .npy (3-channel float32) into 'tensors'
            tensors_dir = outputs_dirpath / str(resolution) / "tensors"
            tensors_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = tensors_dir / f"{index:010d}.npy"
            try:
                np.save(str(tensor_path), crop_with_markers)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed saving tensor {tensor_path}: {e}")

            # Save metadata into metadata folder *before* images so visualization can read it
            metadata_dir = outputs_dirpath / str(resolution) / "metadata"
            metadata = {
                "index": index,
                "resolution": resolution,
                "target_elevation": float(target_elev),
                "tile_min_elevation": tile_min_elev,
                "tile_max_elevation": tile_max_elev,
                "crop_min_elevation": crop_min_elev,
                "crop_max_elevation": crop_max_elev,
                "center": {
                    "lat": float(locals().get("geo_center_lat"))
                    if locals().get("geo_center_lat") is not None
                    else None,
                    "lon": float(locals().get("geo_center_lon"))
                    if locals().get("geo_center_lon") is not None
                    else None,
                },
                "top_left": {
                    "lat": float(locals().get("geo_top_left_lat"))
                    if locals().get("geo_top_left_lat") is not None
                    else None,
                    "lon": float(locals().get("geo_top_left_lon"))
                    if locals().get("geo_top_left_lon") is not None
                    else None,
                },
                "bottom_right": {
                    "lat": float(locals().get("geo_bottom_right_lat"))
                    if locals().get("geo_bottom_right_lat") is not None
                    else None,
                    "lon": float(locals().get("geo_bottom_right_lon"))
                    if locals().get("geo_bottom_right_lon") is not None
                    else None,
                },
                "source_tile": {
                    "lat": center_lat,
                    "lon": center_lon,
                    "name": f"{lat_to_str(center_lat)}{lon_to_str(center_lon)}",
                },
                "markers": [
                    {"i": int(i1), "j": int(j1), "elevation": marker_1_elev},
                    {"i": int(i2), "j": int(j2), "elevation": marker_2_elev},
                ],
                "path": {
                    "length": len(path_without),
                    "pixels": [{"i": int(pi), "j": int(pj)} for pi, pj in path_without],
                },
                "paths": {
                    "without_obstacles": {
                        "length": len(path_without),
                        "pixels": [{"i": int(pi), "j": int(pj)} for pi, pj in path_without],
                    },
                    "with_obstacles": {
                        "length": len(path_with),
                        "pixels": [{"i": int(pi), "j": int(pj)} for pi, pj in path_with],
                    },
                },
                "obstacles": {
                    "enabled": obstacles_enabled,
                    "ratio_max": obstacle_ratio_max,
                    "ratio_target": float(obstacle_target_ratio),
                    "ratio_actual": obstacle_ratio,
                    "size_min_ratio": size_min_ratio,
                    "size_max_ratio": size_max_ratio,
                    "penalty": obstacle_penalty,
                },
                "solver": {
                    "backend": backend,
                    "name": solver_name,
                    "diagonal_movement": solver_config.get("diagonal_movement", "always"),
                    "weight": solver_config.get("weight", 1.0),
                    "heuristic": solver_config.get("heuristic", "euclidean"),
                    "cost_model": solver_config.get("cost_model", "elevation"),
                },
                "timestamp": datetime.now().isoformat(),
            }
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"{index:010d}.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

            # Save visual image at full resolution in 'images' unless disabled
            if not disable_images:
                images_dir = outputs_dirpath / str(resolution) / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                # Convert tensor to BGR image using helper function (provide path pixels and optional elev bounds)
                path_for_image = path_with if obstacles_enabled else path_without
                # elevation channel is crop_norm in [0, 1], so no need to pass elev_range unless desired
                thumb_rgb = tensor_to_image(crop_with_markers, path_pixels=path_for_image)
                image_path = images_dir / f"{index:010d}.png"
                try:
                    cv2.imwrite(str(image_path), thumb_rgb)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed saving image {image_path}: {e}")
            # Log per-sample diagnostics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            logger.info(
                f"Sample {index}: success=True, elevation_attempts={elevation_attempts}, "
                f"marker_retries={marker_attempt + 1 if 'marker_attempt' in locals() else 0}, "
                f"path_length_no_obstacles={len(path_without)}, "
                f"path_length_with_obstacles={len(path_with)}, "
                f"grid_shape=({grid_h},{grid_w}), duration={duration:.2f}s"
            )

            # Calculate geographic coordinates
            px_per_deg = TILE_SIZE_PX
            crop_center_y = y1 + resolution // 2
            crop_center_x = x1 + resolution // 2
            center_px_in_center_tile_y = crop_center_y - center_offset
            center_px_in_center_tile_x = crop_center_x - center_offset
            geo_center_lat = center_lat + 1 - (center_px_in_center_tile_y / px_per_deg)
            geo_center_lon = center_lon + (center_px_in_center_tile_x / px_per_deg)
            half_res_deg = (resolution // 2) / px_per_deg
            geo_top_left_lat = geo_center_lat + half_res_deg
            geo_top_left_lon = geo_center_lon - half_res_deg
            geo_bottom_right_lat = geo_center_lat - half_res_deg
            geo_bottom_right_lon = geo_center_lon + half_res_deg

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return index, True, f"Generated sample {index}", duration

        except Exception as e:
            logger.debug(f"Elevation attempt {elevation_attempt} failed: {e}")
            continue

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Sample {index}: success=False, elevation_attempts={elevation_attempts}, "
        f"marker_retries={marker_attempts_used if 'marker_attempts_used' in locals() else 0}, duration={duration:.2f}s"
    )
    return index, False, f"Failed to generate sample {index} after {max_elevation_retries} elevation attempts", duration


@_log_timing
def _generate_sample_wrapper(args):
    """Wrapper for multiprocessing."""
    return generate_sample(args)


@_log_timing
def count_valid_samples(res_dir: Path, disable_images: bool = False) -> int:
    """
    Count consecutive valid samples starting from index 0.

    A sample is valid if it has all required files:
    - tensors/{index:010d}.npy
    - metadata/{index:010d}.yaml
    - images/{index:010d}.png (if disable_images is False)

    Stops counting at the first incomplete sample and deletes any partial files.

    Args:
        res_dir: Resolution directory containing tensors/, metadata/, images/ subdirs.
        disable_images: If True, don't check for images.

    Returns:
        Number of consecutive valid samples from index 0.
    """
    tensors_dir = res_dir / "tensors"
    metadata_dir = res_dir / "metadata"
    images_dir = res_dir / "images"

    if not tensors_dir.exists() or not metadata_dir.exists():
        return 0

    index = 0
    while True:
        tensor_file = tensors_dir / f"{index:010d}.npy"
        metadata_file = metadata_dir / f"{index:010d}.yaml"
        image_file = images_dir / f"{index:010d}.png"

        tensor_exists = tensor_file.exists()
        metadata_exists = metadata_file.exists()
        image_exists = image_file.exists() if not disable_images else True

        if tensor_exists and metadata_exists and image_exists:
            index += 1
        else:
            # Check for partial files and delete them with warning
            partial_files = []
            if tensor_exists:
                partial_files.append(tensor_file)
            if metadata_exists:
                partial_files.append(metadata_file)
            if not disable_images and image_exists:
                partial_files.append(image_file)

            if partial_files:
                logger.warning(
                    f"Found incomplete sample at index {index}, deleting partial files: "
                    f"{[str(f.name) for f in partial_files]}"
                )
                for f in partial_files:
                    try:
                        f.unlink()
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Failed to delete partial file {f}: {e}")
            break

    return index


@_log_timing
def run_generate(
    outputs_dirpath: str,
    config_filepath: str,
    tiles_dirpath: str,
    resolution: int | list[int] | None = None,
    num_samples: int | None = None,
    num_cpus: int | None = None,
    disable_images: bool = False,
    force: bool = False,
) -> None:
    """
    Main orchestration function for generating dataset samples.

    Args:
        outputs_dirpath: Directory path for output files.
        config_filepath: Path to config.yaml file.
        tiles_dirpath: Directory containing DSM tiles.
        resolution: Image resolution (single int) or list of resolutions.
        num_samples: Number of samples to generate.
        num_cpus: Number of CPUs to use for multiprocessing.
        disable_images: If True, skip saving PNG images.
        force: If True, delete existing directories and regenerate all samples.
    """
    # Load config
    config_path = Path(config_filepath)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Get parameters from config or arguments
    if isinstance(resolution, list):
        resolutions = resolution
    elif resolution is not None:
        resolutions = [resolution]
    else:
        resolutions = config.get("resolutions", [256])

    try:
        resolutions = [int(r) for r in resolutions]
    except Exception:
        logger.error(f"Invalid resolution values: {resolutions}")
        return

    if any(r <= 0 for r in resolutions):
        logger.error(f"Resolution values must be positive integers: {resolutions}")
        return

    samples_count = num_samples if num_samples is not None else config.get("num_samples", 100)

    outputs_path = Path(outputs_dirpath)
    tiles_path = Path(tiles_dirpath)

    if not tiles_path.exists():
        logger.error(f"Tiles directory not found: {tiles_path}")
        return

    # Load calibrate.json to get global min/max and tile_stats
    calibrate_path = Path("outputs/calibrate/calibrate.json")
    if not calibrate_path.exists():
        logger.error(f"Calibration file not found: {calibrate_path}. Run 'passage calibrate' first.")
        return

    try:
        with open(calibrate_path) as f:
            calibration = json.load(f)
        global_min = calibration["global_min"]
        global_max = calibration["global_max"]
        tile_stats = calibration["tile_stats"]
        logger.info(f"Loaded calibration: global_min={global_min}, global_max={global_max}, tiles={len(tile_stats)}")
    except Exception as e:
        logger.error(f"Failed to load calibration file: {e}")
        return

    # Load tile_stats into pandas DataFrame
    tile_stats_df = pd.DataFrame(tile_stats)
    logger.info(f"Loaded {len(tile_stats_df)} tiles into DataFrame")

    # Load solver configuration
    solver_config = _resolve_solver_config(config.get("solver", {}))
    backend, solver_name = _parse_solver_backend_and_name(solver_config)
    logger.info(
        f"Solver config: backend={backend}, "
        f"name={solver_name}, "
        f"diagonal={solver_config.get('diagonal_movement', 'always')}, "
        f"weight={solver_config.get('weight', 1.0)}, "
        f"cost_model={solver_config.get('cost_model', 'elevation')}, "
        f"max_marker_retries={solver_config.get('max_marker_retries', 10)}"
    )

    # Load obstacle configuration
    obstacle_settings = _resolve_obstacle_config(config)
    logger.info(
        "Obstacles: enabled=%s, ratio_max=%.3f, size_min_ratio=%.3f, size_max_ratio=%.3f, penalty=%.1e",
        obstacle_settings.get("enabled", False),
        obstacle_settings.get("ratio_max", 0.0),
        obstacle_settings.get("size_min_ratio", 0.0),
        obstacle_settings.get("size_max_ratio", 0.0),
        obstacle_settings.get("obstacle_penalty", 0.0),
    )

    # Harmonized startup log
    cpus = num_cpus if num_cpus is not None else os.cpu_count()
    logger.info(
        f"Starting generate: outputs={outputs_path}, tiles={tiles_path}, resolution_override={resolution}, "
        f"num_samples={samples_count}, cpus={cpus}, disable_images={disable_images}"
    )

    # Generate samples for each resolution
    for res in resolutions:
        res_dir = outputs_path / str(res)

        # Determine start index based on force flag and existing samples
        start_index = 0
        if res_dir.exists():
            if force:
                logger.warning(f"--force specified: Deleting existing directory {res_dir} and regenerating all samples")
                try:
                    shutil.rmtree(res_dir)
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed to remove existing directory {res_dir}: {e}")
            else:
                # Count valid samples and resume from there
                start_index = count_valid_samples(res_dir, disable_images)
                if start_index >= samples_count:
                    logger.info(f"Resolution {res} already complete ({start_index}/{samples_count} samples), skipping")
                    continue
                if start_index > 0:
                    logger.info(
                        f"Resuming generation for resolution {res}: found {start_index} valid samples, "
                        f"generating {samples_count - start_index} more"
                    )
                    logger.warning(
                        "Note: Calibration data may have changed since previous generation — "
                        "samples may have inconsistent normalization"
                    )

        samples_to_generate = samples_count - start_index
        # Compute and log how many tiles will be concatenated to generate the crop for this resolution
        N, total_tiles, half_tiles = tiles_needed_for_resolution(res)
        grid_pixels = N * TILE_SIZE_PX
        logger.info(
            f"Resolution {res}: using {N}x{N} tiles (total {total_tiles}) -> grid ~{grid_pixels}x{grid_pixels} px "
            f"(half_tiles={half_tiles})"
        )
        logger.info(f"Generating {samples_to_generate} samples at resolution {res}x{res}...")

        # Prepare output directories
        res_dir.mkdir(parents=True, exist_ok=True)
        (res_dir / "tensors").mkdir(exist_ok=True)
        if not disable_images:
            (res_dir / "images").mkdir(exist_ok=True)
        (res_dir / "metadata").mkdir(exist_ok=True)

        # Prepare arguments for each sample (starting from start_index)
        args_list = [
            (
                i,
                tiles_path,
                outputs_path,
                res,
                tile_stats_df,
                global_min,
                global_max,
                disable_images,
                solver_config,
                obstacle_settings,
            )
            for i in range(start_index, samples_count)
        ]

        # Process samples
        durations = []
        successful = 0
        if num_cpus == 0:
            logger.info("Running in sequential mode (no multiprocessing)")
            for args in tqdm(args_list, total=samples_to_generate):
                idx, success, msg, duration = generate_sample(args)
                durations.append(duration)
                if not success:
                    logger.warning(msg)
                else:
                    successful += 1
        else:
            pool_size = num_cpus if num_cpus is not None else os.cpu_count()
            logger.info(f"Running with {pool_size} CPUs")
            with multiprocessing.Pool(pool_size) as pool:
                for idx, success, msg, duration in tqdm(
                    pool.imap_unordered(_generate_sample_wrapper, args_list),
                    total=samples_to_generate,
                ):
                    durations.append(duration)
                    if not success:
                        logger.warning(msg)
                    else:
                        successful += 1

        logger.info(f"Completed generation for resolution {res}")

        # Save per-resolution summary inside the resolution directory
        mean_time = float(sum(durations) / len(durations)) if len(durations) > 0 else 0.0
        res_summary = {
            "resolution": res,
            "requested_samples": samples_count,
            "start_index": start_index,
            "generated_samples": samples_to_generate,
            "successful_samples": successful,
            "mean_time_per_sample_seconds": mean_time,
            "tiles_count": len(tile_stats_df),
            "tiles_dirpath": str(tiles_path),
            "global_min": global_min,
            "global_max": global_max,
            "timestamp": datetime.now().isoformat(),
        }
        res_summary_path = res_dir / "summary.json"
        try:
            with open(res_summary_path, "w") as f:
                json.dump(res_summary, f, indent=2)
            logger.info(f"Per-resolution summary saved to {res_summary_path}")
        except Exception as e:  # pragma: no cover
            logger.warning(f"Failed to write per-resolution summary to {res_summary_path}: {e}")
        # Accumulate for overall mean calculation
        if "overall_durations" not in locals():
            overall_durations = []
        overall_durations.extend(durations)

    # Save generation summary
    mean_time_overall = (
        float(sum(overall_durations) / len(overall_durations))
        if "overall_durations" in locals() and len(overall_durations) > 0
        else 0.0
    )
    summary = {
        "resolutions": resolutions,
        "num_samples": samples_count,
        "tiles_count": len(tile_stats_df),
        "tiles_dirpath": str(tiles_path),
        "global_min": global_min,
        "global_max": global_max,
        "mean_time_per_sample_seconds": mean_time_overall,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = outputs_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Generation completed. Summary saved to {summary_path}")
