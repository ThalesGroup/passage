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

"""
Generate module for PASSAGE.

`run_generate` runs the generator that built PASSAGE v1.0 when the
configuration sets `generation.dataset_version`, as the shipped
`config/config.yaml` does. Otherwise it runs a sample-by-sample path that the
v1.0 release does not use.

Tiles inside `config/hold_out_mask.png` (181x361 pixels, one pixel per degree,
latitude -90 to +90, longitude -180 to +180) are used only by the `test` split.
"""

import csv
import hashlib
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

import networkx as nx
import numpy as np
import pandas as pd
import rasterio
import yaml
from loguru import logger
from PIL import Image
from rasterio.enums import Resampling
from rasterio.windows import Window
from tqdm import tqdm

from passage.obstacles import sample_angle_from_config, sample_n_from_config, superellipse_mask
from passage.pathfinding_utils import (
    DIAGONAL_DISTANCE_M,
    HEURISTIC_MAP,
    PIXEL_SIZE_M,
    _find_path_pathfinding,
    _grid_astar,
    _grid_astar_with_details,
    _grid_dijkstra,
    build_graph_from_elevation,
    check_path_collision,
)
from passage.pathfinding_utils import (
    _build_pathfinding_cost_matrix as _build_pathfinding_cost_matrix,
)
from passage.pathfinding_utils import (
    _compute_edge_weights as _compute_edge_weights,
)
from passage.pathfinding_utils import (
    _get_graph_template as _get_graph_template,
)
from passage.pathfinding_utils import (
    _normalize_elevation_arrays as _normalize_elevation_arrays,
)
from passage.pathfinding_utils import (
    _pathfinding_path_to_coords as _pathfinding_path_to_coords,
)
from passage.utils import TENSOR_SUFFIX, _log_timing, save_tensor

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

SPLIT_ORDER = ["train", "calibration", "validation", "test"]

# Default path for hold-out mask (relative to config directory)
DEFAULT_HOLD_OUT_MASK_PATH = "config/hold_out_mask.png"

# Conservative auto-caps for large crops when worker count is left to auto-detection.
# Large resolutions create multiple full-resolution arrays per worker, so using every
# visible CPU on an HPC node can exhaust RAM long before CPU becomes the bottleneck.
AUTO_CPU_CAPS_BY_RESOLUTION: tuple[tuple[int, int], ...] = (
    (4096, 8),
    (2048, 16),
)

# Global cache for hold-out mask (loaded once at startup)
_HOLD_OUT_MASK: np.ndarray | None = None
_HOLD_OUT_MASK_PATH: str | None = None


def tensor_to_image(*args, **kwargs) -> np.ndarray:
    """Lazily import visualization helpers so `--no-images` can run without OpenCV."""
    from passage.visualization import tensor_to_image as _tensor_to_image

    return _tensor_to_image(*args, **kwargs)


def _recommended_generate_cpu_cap(resolution: int) -> int | None:
    """Return a conservative worker cap for memory-heavy resolutions."""
    for min_resolution, cap in AUTO_CPU_CAPS_BY_RESOLUTION:
        if resolution >= min_resolution:
            return cap
    return None


def _resolve_generation_pool_size(num_cpus: int | None, resolution: int, num_tasks: int) -> int:
    """Choose a worker count that avoids obvious oversubscription.

    The generator defaults to all visible CPUs when ``num_cpus`` is omitted, but that
    can be unsafe for large crops on HPC nodes. We therefore:
    1. cap workers to the number of remaining samples,
    2. auto-cap only the default/all-CPU case for large resolutions,
    3. preserve explicit ``--num-cpus`` choices while warning when they look risky.
    """
    if num_cpus == 0:
        return 0

    available_cpus = max(1, os.cpu_count() or 1)
    requested_cpus = available_cpus if num_cpus is None else max(1, int(num_cpus))
    effective_cpus = min(requested_cpus, max(1, int(num_tasks)))

    if effective_cpus < requested_cpus:
        logger.info(
            "Limiting generation workers to %s because only %s sample(s) remain for resolution %s",
            effective_cpus,
            num_tasks,
            resolution,
        )

    recommended_cap = _recommended_generate_cpu_cap(resolution)
    if recommended_cap is not None and effective_cpus > recommended_cap:
        if num_cpus is None:
            logger.warning(
                "Auto-capping generation workers from %s to %s for resolution %s. "
                "Large crops are memory-heavy and the all-CPU default can OOM on shared or HPC nodes. "
                "Pass --num-cpus explicitly to override if your node can sustain more workers.",
                effective_cpus,
                recommended_cap,
                resolution,
            )
            effective_cpus = recommended_cap
        else:
            logger.warning(
                "Using %s generation workers for resolution %s. Large crops are memory-heavy; "
                "consider --num-cpus <= %s if you see OOM kills or BrokenPipeError traces.",
                effective_cpus,
                resolution,
                recommended_cap,
            )

    if effective_cpus <= 1:
        return 0

    return effective_cpus


def _save_png(image_path: Path, image_bgr: np.ndarray) -> None:
    """Write an OpenCV-style BGR image as a PNG without requiring OpenCV."""
    rgb_image = np.ascontiguousarray(image_bgr[:, :, ::-1])
    Image.fromarray(rgb_image).save(image_path, format="PNG", compress_level=0)


def load_hold_out_mask(mask_path: str | Path | None = None) -> np.ndarray | None:
    """Load the hold-out mask from a PNG file.

    The mask is a 181x361 pixel image (1 degree per pixel) where:
    - Row 0 = latitude +90, Row 180 = latitude -90
    - Col 0 = longitude -180, Col 360 = longitude +180
    - Non-zero RGB values indicate hold-out (test-only) regions

    Args:
        mask_path: Path to the mask PNG file. If None, uses DEFAULT_HOLD_OUT_MASK_PATH.

    Returns:
        2D numpy array (181, 361) with 0/1 values, or None if mask doesn't exist.
    """
    global _HOLD_OUT_MASK, _HOLD_OUT_MASK_PATH

    if mask_path is None:
        mask_path = DEFAULT_HOLD_OUT_MASK_PATH

    mask_path = Path(mask_path)

    # Return cached mask if same path
    if _HOLD_OUT_MASK is not None and _HOLD_OUT_MASK_PATH == str(mask_path):
        return _HOLD_OUT_MASK

    if not mask_path.exists():
        logger.debug(f"Hold-out mask not found at {mask_path}, hold-out filtering disabled")
        _HOLD_OUT_MASK = None
        _HOLD_OUT_MASK_PATH = None
        return None

    try:
        img = Image.open(mask_path)
        arr = np.array(img)

        # Handle different image formats (RGBA, RGB, grayscale)
        if arr.ndim == 3:
            # Use R channel (all RGB channels should be identical)
            mask_2d = arr[:, :, 0]
        else:
            mask_2d = arr

        # Convert to binary mask (0 or 1)
        binary_mask = (mask_2d > 0).astype(np.uint8)

        # Validate expected dimensions
        expected_shape = (181, 361)  # lat: -90 to +90, lon: -180 to +180
        if binary_mask.shape != expected_shape:
            logger.warning(
                f"Hold-out mask has unexpected shape {binary_mask.shape}, expected {expected_shape}. "
                f"Mask will be resized."
            )
            # Resize if needed (shouldn't happen with correct mask)
            binary_mask = np.array(Image.fromarray(binary_mask).resize((361, 181), Image.NEAREST))

        _HOLD_OUT_MASK = binary_mask
        _HOLD_OUT_MASK_PATH = str(mask_path)

        hold_out_count = np.sum(binary_mask > 0)
        logger.info(f"Loaded hold-out mask from {mask_path}: {hold_out_count} tiles in hold-out region")

        return _HOLD_OUT_MASK

    except Exception as e:
        logger.warning(f"Failed to load hold-out mask from {mask_path}: {e}")
        _HOLD_OUT_MASK = None
        _HOLD_OUT_MASK_PATH = None
        return None


def is_tile_in_hold_out(lat: int, lon: int, mask: np.ndarray | None = None) -> bool:
    """Check if a tile at (lat, lon) is in the hold-out region.

    Args:
        lat: Tile latitude (integer, -90 to +89)
        lon: Tile longitude (integer, -180 to +179)
        mask: Hold-out mask array (181x361). If None, uses global cached mask.

    Returns:
        True if the tile is in the hold-out region, False otherwise.
    """
    if mask is None:
        mask = _HOLD_OUT_MASK

    if mask is None:
        return False

    # Convert lat/lon to mask pixel coordinates
    # Row: lat +90 -> row 0, lat -90 -> row 180
    # Col: lon -180 -> col 0, lon +180 -> col 360
    row = 90 - lat
    col = lon + 180

    # Clamp to valid range
    row = max(0, min(180, row))
    col = max(0, min(360, col))

    return bool(mask[row, col] > 0)


def filter_tiles_by_hold_out(
    tile_stats_df: pd.DataFrame,
    split_name: str,
    mask: np.ndarray | None = None,
) -> pd.DataFrame:
    """Filter tile_stats DataFrame based on hold-out mask and split type.

    For 'test' split: only include tiles IN the hold-out region.
    For other splits (train, validation, calibration): EXCLUDE tiles in the hold-out region.

    Args:
        tile_stats_df: DataFrame with 'lat' and 'lon' columns.
        split_name: Name of the split ('train', 'validation', 'calibration', 'test').
        mask: Hold-out mask array. If None, uses global cached mask.

    Returns:
        Filtered DataFrame.
    """
    if mask is None:
        mask = _HOLD_OUT_MASK

    if mask is None:
        logger.debug(f"No hold-out mask loaded, returning all {len(tile_stats_df)} tiles for split '{split_name}'")
        return tile_stats_df

    # Create boolean mask for each tile
    in_hold_out = tile_stats_df.apply(
        lambda row: is_tile_in_hold_out(int(row["lat"]), int(row["lon"]), mask),
        axis=1,
    )

    if split_name == "test":
        # For test: only use tiles IN the hold-out region
        filtered_df = tile_stats_df[in_hold_out]
        logger.debug(
            f"Split '{split_name}': filtered to {len(filtered_df)}/{len(tile_stats_df)} tiles (hold-out region only)"
        )
    else:
        # For train/validation/calibration: EXCLUDE tiles in hold-out region
        filtered_df = tile_stats_df[~in_hold_out]
        logger.debug(
            f"Split '{split_name}': filtered to {len(filtered_df)}/{len(tile_stats_df)} tiles "
            f"(excluding hold-out region)"
        )

    return filtered_df


def _split_sort_key(name: str) -> tuple[int, str]:
    try:
        return (SPLIT_ORDER.index(name), name)
    except ValueError:
        return (len(SPLIT_ORDER), name)


def _order_tile_candidates(tile_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable tile ordering for deterministic sampling."""
    if tile_stats_df.empty:
        return tile_stats_df

    sort_columns = [column for column in ("lat", "lon", "filename") if column in tile_stats_df.columns]
    if not sort_columns:
        return tile_stats_df.reset_index(drop=True)

    return tile_stats_df.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _derive_sample_seed(generation_seed: int, resolution: int, index: int, split_name: str) -> int:
    """Derive a stable per-sample seed from run-level provenance."""
    payload = f"{int(generation_seed)}:{int(resolution)}:{int(index)}:{split_name}".encode("ascii", "strict")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _resolve_preview_images_config(config: dict | None, active_resolutions: list[int]) -> dict:
    """Resolve deterministic preview-image generation settings."""
    config = config or {}
    export_cfg = config.get("export", {}) if isinstance(config, dict) else {}
    generation_cfg = config.get("generation", {}) if isinstance(config, dict) else {}
    preview_cfg = generation_cfg.get("preview_images", {}) if isinstance(generation_cfg, dict) else {}

    enabled = bool(preview_cfg.get("enabled", True))

    configured_resolutions = preview_cfg.get("resolutions")
    if configured_resolutions in (None, "", []):
        preview_resolutions = {int(resolution) for resolution in active_resolutions}
    elif isinstance(configured_resolutions, (list, tuple, set)):
        preview_resolutions = {int(resolution) for resolution in configured_resolutions}
    else:
        preview_resolutions = {int(configured_resolutions)}

    num_samples_per_split = preview_cfg.get("num_samples_per_split")
    if num_samples_per_split in (None, ""):
        num_samples_per_split = export_cfg.get("sample", {}).get("num_samples", 0)
    num_samples_per_split = max(int(num_samples_per_split), 0)

    return {
        "enabled": enabled and num_samples_per_split > 0,
        "resolutions": preview_resolutions,
        "num_samples_per_split": num_samples_per_split,
    }


def _select_preview_image_indices(
    split_assignments: dict[int, str],
    num_samples_per_split: int,
) -> dict[str, set[int]]:
    """Select the deterministic preview subset: first N sample indices for each split."""
    if num_samples_per_split <= 0:
        return {}

    split_indices: dict[str, list[int]] = {}
    for index, split_name in split_assignments.items():
        split_indices.setdefault(split_name, []).append(int(index))

    return {split_name: set(sorted(indices)[:num_samples_per_split]) for split_name, indices in split_indices.items()}


def _parse_solver_backend_and_name(solver_config: dict) -> tuple[str, str]:
    """Parse backend and solver algorithm name from config.

    Handles multiple formats:
    1. name: "backend:solver" (e.g., "grid:astar")
    2. backend: "grid", solver: "astar" or solver_algorithm: "astar"
    3. Defaults to networkx backend and astar solver
    """
    backend = solver_config.get("backend")

    # Check for solver algorithm specified via solver_algorithm or solver key
    solver_name = solver_config.get("solver_algorithm") or solver_config.get("solver", "astar")

    # Check if name contains backend:solver format
    raw_name = solver_config.get("name", "")
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
    """Resolve solver config to a flat dictionary with solver_algorithm key.

    Accepts a flat config like:
    solver:
      name: astar_elevation
      backend: grid
      solver: astar
      cost_model: elevation
      ...

    The 'solver' key specifies the actual algorithm (astar, dijkstra, etc.).
    """
    if not isinstance(solver_config, dict):
        return {}

    resolved = dict(solver_config)

    # If the config has a 'solver' key, use it as the actual solver algorithm name
    if "solver" in resolved:
        resolved["solver_algorithm"] = resolved["solver"]

    # Default backend to 'grid' if not specified
    if "backend" not in resolved:
        resolved["backend"] = "grid"

    return resolved


@_log_timing
def find_path(
    crop: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    solver_config: dict,
    prebuilt_graph: nx.DiGraph | None = None,
    blocked_mask: np.ndarray | None = None,
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
        blocked_mask: Optional boolean mask where True marks impassable cells.

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
    if blocked_mask is not None:
        blocked = np.asarray(blocked_mask, dtype=np.bool_)
        if blocked.shape != crop.shape:
            raise ValueError(f"blocked_mask shape {blocked.shape} does not match crop shape {crop.shape}")
        if blocked[si, sj] or blocked[ei, ej]:
            logger.debug(f"find_path: start/end blocked for grid_size={grid_size}, start={start}, end={end}")
            return []
    else:
        blocked = None

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
            path = _grid_astar(crop, (si, sj), (ei, ej), weight, diagonal_name, heuristic_name, cost_model, blocked)
        else:
            path = _grid_dijkstra(crop, (si, sj), (ei, ej), weight, diagonal_name, cost_model, blocked)
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
                blocked,
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

    if prebuilt_graph is not None and blocked is None:
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
            blocked_mask=blocked,
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
                if src.height != TILE_SIZE_PX or src.width != TILE_SIZE_PX:
                    data = src.read(
                        1,
                        out_shape=(TILE_SIZE_PX, TILE_SIZE_PX),
                        resampling=Resampling.nearest,
                    ).astype(np.float32)
                else:
                    data = src.read(1).astype(np.float32)
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
    return_loaded_tiles: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int]]]:
    """
    Load a ``(2*half_tiles+1) x (2*half_tiles+1)`` grid of tiles centered on the given coordinates.

    Handles latitude bounds by substituting flat tiles for out-of-bounds rows.
    Handles longitude wrapping at -180/180.

    Returns:
        - grid: concatenated numpy array of shape (N*TILE_SIZE_PX, N*TILE_SIZE_PX)
        - loaded_tiles: list of (lat, lon) tuples in row-major order corresponding to each tile block
    """
    import time

    start_time = time.time()
    rows = []
    loaded_tiles: list[tuple[int, int]] = []
    N = 2 * half_tiles + 1

    for dlat in range(half_tiles, -half_tiles - 1, -1):
        lat = center_lat + dlat
        if lat < MIN_LAT or lat >= MAX_LAT:
            # Substitute zero tiles for out-of-bounds latitudes (consistent with earlier behavior)
            logger.debug(f"Latitude {lat} out of bounds [{MIN_LAT}, {MAX_LAT}); substituting zero tiles")
            row_tiles = [np.zeros((TILE_SIZE_PX, TILE_SIZE_PX), dtype=np.float32) for _ in range(N)]
            # still record the lat/lon positions for these substituted tiles
            for dlon in range(-half_tiles, half_tiles + 1):
                lon = wrap_lon(center_lon + dlon)
                loaded_tiles.append((lat, lon))
        else:
            row_tiles = []
            for dlon in range(-half_tiles, half_tiles + 1):
                lon = wrap_lon(center_lon + dlon)
                tile = load_tile(tiles_dirpath, lat, lon)
                row_tiles.append(tile)
                loaded_tiles.append((lat, lon))
        rows.append(np.concatenate(row_tiles, axis=1))

    grid = np.concatenate(rows, axis=0)  # Vertical concat
    duration = time.time() - start_time
    logger.debug(
        f"load_tile_grid center=({center_lat},{center_lon}) half_tiles={half_tiles} N={N} "
        f"grid_shape={grid.shape} tiles_loaded={len(loaded_tiles)} load_duration={duration:.2f}s"
    )
    if return_loaded_tiles:
        return grid, loaded_tiles
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
    return_loaded_tiles: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Assemble a crop of size (resolution, resolution) from tile windows.

    The center tile is provided (full TILE_SIZE_PX x TILE_SIZE_PX) to avoid reloading it.
    """
    N = 2 * half_tiles + 1
    y2 = y1 + resolution
    x2 = x1 + resolution
    crop = np.zeros((resolution, resolution), dtype=np.float32)

    loaded_tiles: list[tuple[int, int, int, int]] = []

    row_start = max(0, y1 // TILE_SIZE_PX)
    row_end = min(N - 1, (y2 - 1) // TILE_SIZE_PX)
    col_start = max(0, x1 // TILE_SIZE_PX)
    col_end = min(N - 1, (x2 - 1) // TILE_SIZE_PX)

    for row_idx in range(row_start, row_end + 1):
        dlat = half_tiles - row_idx
        tile_lat = center_lat + dlat
        tile_top = row_idx * TILE_SIZE_PX
        for col_idx in range(col_start, col_end + 1):
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

            loaded_tiles.append((tile_lat, tile_lon, row_idx, col_idx))

            # Destination indices in crop
            dest_y1 = tile_top + overlap_y1 - y1
            dest_y2 = dest_y1 + h
            dest_x1 = tile_left + overlap_x1 - x1
            dest_x2 = dest_x1 + w

            # If latitude is out of bounds, use zero tiles (consistent with load_tile_grid)
            if tile_lat < MIN_LAT or tile_lat >= MAX_LAT:
                block = np.zeros((int(h), int(w)), dtype=np.float32)
            # If this is the center tile, use the provided array
            elif dlat == 0 and dlon == 0:
                block = center_tile[int(overlap_y1) : int(overlap_y2), int(overlap_x1) : int(overlap_x2)]
            else:
                block = load_tile_window(
                    tiles_dirpath, tile_lat, tile_lon, int(overlap_y1), int(overlap_x1), int(h), int(w)
                )

            crop[dest_y1:dest_y2, dest_x1:dest_x2] = block

    if return_loaded_tiles:
        return crop, loaded_tiles
    return crop


def _resolve_obstacle_config(config: dict | None) -> dict:
    """Normalize obstacle settings from config.yaml.

    Returns a dict with keys:
      - enabled
      - ratio_max
      - size_min_ratio
      - size_max_ratio
      - generation_cfg
            - impassable
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
    impassable = bool(obstacles_cfg.get("impassable", True))
    obstacle_penalty = float(obstacles_cfg.get("obstacle_penalty", 1e8))

    return {
        "enabled": enabled,
        "ratio_max": ratio_max,
        "size_min_ratio": size_min_ratio,
        "size_max_ratio": size_max_ratio,
        "generation_cfg": generation_cfg,
        "impassable": impassable,
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
    args: tuple[int, Path, Path, int, pd.DataFrame, float, float, bool, list, dict, str]
    | tuple[int, Path, Path, int, pd.DataFrame, float, float, bool, list, dict, str, int]
    | tuple[int, Path, Path, int, pd.DataFrame, float, float, bool, list, dict, str, int, bool],
) -> tuple[int, bool, str, float]:
    """
    Generate a single dataset sample.

    Args:
        args: Tuple of (index, tiles_dirpath, outputs_dirpath, resolution,
            tile_stats_df, global_min, global_max, disable_images, solvers_configs,
            obstacle_settings, split_name[, generation_seed[, save_preview_images]]).

    Returns:
        Tuple of (index, success, message, duration_seconds).
    """
    # Unpack args tuple
    if len(args) not in {11, 12, 13}:
        raise ValueError("generate_sample expects an 11-, 12-, or 13-element args tuple")
    if len(args) == 11:
        (
            index,
            tiles_dirpath,
            outputs_dirpath,
            resolution,
            tile_stats_df,
            global_min,
            global_max,
            disable_images,
            solvers_configs,
            obstacle_settings,
            split_name,
        ) = args
        generation_seed = 0
        save_preview_images = not disable_images
    elif len(args) == 12:
        (
            index,
            tiles_dirpath,
            outputs_dirpath,
            resolution,
            tile_stats_df,
            global_min,
            global_max,
            disable_images,
            solvers_configs,
            obstacle_settings,
            split_name,
            generation_seed,
        ) = args
        save_preview_images = not disable_images
    else:
        (
            index,
            tiles_dirpath,
            outputs_dirpath,
            resolution,
            tile_stats_df,
            global_min,
            global_max,
            disable_images,
            solvers_configs,
            obstacle_settings,
            split_name,
            generation_seed,
            save_preview_images,
        ) = args

    # Validate solvers_configs is a non-empty list
    if not solvers_configs or not isinstance(solvers_configs, list):
        raise ValueError("solvers_configs must be a non-empty list of solver configurations")

    generation_seed = int(generation_seed)
    save_preview_images = bool(save_preview_images)
    sample_seed = _derive_sample_seed(generation_seed, resolution, index, split_name)
    py_rng = random.Random(sample_seed)
    np_rng = np.random.default_rng(sample_seed)

    obstacle_settings = obstacle_settings or {}
    # Use max_marker_retries from the first solver's common config (they should all be the same)
    first_solver_resolved = _resolve_solver_config(solvers_configs[0])
    max_marker_retries = first_solver_resolved.get("max_marker_retries", 10)
    max_elevation_retries = 10  # Total attempts for different elevations

    start_time = datetime.now()
    elevation_attempts = 0
    # timing diagnostics
    tile_loading_time = 0.0
    marker_search_time = 0.0
    tensor_save_time = 0.0
    metadata_write_time = 0.0
    image_conversion_time = 0.0
    image_write_time = 0.0
    obstacle_gen_total = 0.0

    half_res = resolution // 2
    half_tiles = max(1, math.ceil(half_res / TILE_SIZE_PX))

    for elevation_attempt in range(max_elevation_retries):
        elevation_attempts += 1
        try:
            # New sampling strategy: pick a random tile first, then draw target elevation from that tile's range.
            # This reduces failed elevation attempts caused by drawing elevations not covered by any tile.
            # Apply hold-out mask filtering based on split type
            candidates = filter_tiles_by_hold_out(tile_stats_df, split_name)

            if candidates.empty:
                logger.warning(
                    f"No tiles available for split '{split_name}' after hold-out filtering. "
                    f"Check hold_out_mask.png configuration."
                )
                continue

            target_elev = py_rng.uniform(float(global_min), float(global_max))

            candidates = candidates[(candidates["min"] <= target_elev) & (candidates["max"] >= target_elev)]

            if candidates.empty:
                continue  # No valid tiles for this resolution

            candidates = _order_tile_candidates(candidates)

            # Select a random tile and draw a target elevation from its local range
            selected = candidates.iloc[py_rng.randrange(len(candidates))]
            center_lat = int(selected["lat"])
            center_lon = int(selected["lon"])

            # 5. Load the center tile first; only intersecting tiles will be loaded for the crop later
            logger.debug(f"Loading center tile for center=({center_lat},{center_lon}) half_tiles={half_tiles}")
            t_tile = time.time()
            center_tile = load_tile(tiles_dirpath, center_lat, center_lon)

            # Offset (in pixels) from grid origin to start of center tile
            center_offset = TILE_SIZE_PX * half_tiles
            grid_h = (2 * half_tiles + 1) * TILE_SIZE_PX
            grid_w = grid_h

            # 6. Find the pixel with the closest elevation to target_elev within the center tile
            abs_diff = np.abs(center_tile - target_elev)
            min_idx = np.argmin(abs_diff)
            target_row, target_col = divmod(min_idx, TILE_SIZE_PX)

            # Convert to grid coordinates
            target_grid_y = center_offset + target_row
            target_grid_x = center_offset + target_col

            # 7. Generate a random crop containing the target pixel with random offsets
            crop_y1_min = max(0, target_grid_y - resolution + 1)
            crop_y1_max = min(grid_h - resolution, target_grid_y)
            crop_x1_min = max(0, target_grid_x - resolution + 1)
            crop_x1_max = min(grid_w - resolution, target_grid_x)

            if crop_y1_max < crop_y1_min or crop_x1_max < crop_x1_min:
                continue  # Try another elevation

            # Randomly select crop position
            y1 = py_rng.randint(crop_y1_min, crop_y1_max)
            x1 = py_rng.randint(crop_x1_min, crop_x1_max)
            y2 = y1 + resolution
            x2 = x1 + resolution

            # Assemble the crop using only intersecting tiles
            crop, loaded_tiles = _assemble_crop_from_tiles(
                tiles_dirpath,
                center_lat,
                center_lon,
                half_tiles,
                y1,
                x1,
                resolution,
                center_tile,
                return_loaded_tiles=True,
            )
            tile_loading_time = time.time() - t_tile
            logger.debug(
                f"crop_shape={crop.shape} grid_shape=({grid_h},{grid_w}) tiles_loaded={len(loaded_tiles)} "
                f"tile_load_seconds={tile_loading_time:.3f}"
            )

            # 8. Calculate marker 1 position (the target pixel) in crop coordinates
            i1 = target_grid_y - y1
            j1 = target_grid_x - x1

            # 9. Try to find valid markers with retry logic (use first solver for validation)
            h, w = crop.shape
            total_pixels = h * w
            marker1_flat_idx = i1 * w + j1

            # Extract obstacle settings
            obstacles_cfg = obstacle_settings.get("config", {})
            obstacles_enabled = bool(obstacle_settings.get("enabled", False))
            obstacle_ratio_max = float(obstacle_settings.get("ratio_max", 0.0))
            size_min_ratio = float(obstacle_settings.get("size_min_ratio", 0.20))
            size_max_ratio = float(obstacle_settings.get("size_max_ratio", 0.50))
            generation_cfg = obstacle_settings.get("generation_cfg", {})
            obstacles_impassable = bool(obstacle_settings.get("impassable", True))
            obstacle_penalty = float(obstacle_settings.get("obstacle_penalty", 1e8))

            # Use first solver for marker validation
            first_solver_config = _resolve_solver_config(solvers_configs[0])

            obstacles_mask = None
            obstacle_target_ratio = 0.0
            obstacle_ratio = 0.0
            i2, j2 = 0, 0

            t_marker = time.time()
            marker_attempts_used = 0
            found_valid_markers = False
            for marker_attempt in range(max_marker_retries):
                marker_attempts_used += 1
                # Generate marker 2 randomly, distinct from marker 1
                idx2 = py_rng.randrange(total_pixels - 1)
                if idx2 >= marker1_flat_idx:
                    idx2 += 1
                i2, j2 = divmod(idx2, w)

                # Try to find path between markers using first solver (without obstacles) to validate
                test_path = find_path(crop, (i1, j1), (i2, j2), first_solver_config)
                if len(test_path) == 0:
                    continue  # No path possible, try other markers

                # Generate obstacles mask (if enabled)
                marker_positions = [(i1, j1), (i2, j2)]
                obstacle_gen_time = 0.0
                if obstacles_enabled:
                    t_obs = time.time()
                    obstacles_mask, obstacle_target_ratio = _generate_obstacles_mask(
                        crop.shape,
                        obstacle_ratio_max,
                        size_min_ratio,
                        size_max_ratio,
                        obstacles_cfg,
                        marker_positions=marker_positions,
                        rng=np_rng,
                        generation_cfg=generation_cfg,
                    )
                    obstacle_gen_time = time.time() - t_obs
                    obstacle_gen_total += obstacle_gen_time
                else:
                    obstacles_mask = np.zeros_like(crop, dtype=np.uint8)
                    obstacle_target_ratio = 0.0

                obstacle_ratio = float(np.mean(obstacles_mask))

                # Validate that a path with obstacles exists using first solver
                blocked_obstacles = (
                    obstacles_mask.astype(np.bool_) if obstacles_enabled and obstacles_impassable else None
                )
                masked_elevation = crop.copy()
                if obstacles_enabled and blocked_obstacles is None and obstacle_penalty > 0:
                    masked_elevation[obstacles_mask == 1] = masked_elevation.max() + obstacle_penalty

                test_path_with = find_path(
                    crop if blocked_obstacles is not None else masked_elevation,
                    (i1, j1),
                    (i2, j2),
                    first_solver_config,
                    blocked_mask=blocked_obstacles,
                )

                # Check if path_with crosses obstacles (should not happen if penalty is high enough)
                if obstacles_enabled and check_path_collision(test_path_with, obstacles_mask):
                    logger.debug(f"test path_with crosses obstacles (attempt {marker_attempt}), retrying markers")
                    continue

                found_valid_markers = True
                break  # Found valid markers

            marker_search_time = time.time() - t_marker
            if not found_valid_markers:
                logger.debug(
                    f"No valid markers found after {marker_attempts_used} attempts; retrying elevation (marker_search_seconds={marker_search_time:.3f})"
                )
                continue  # Try another elevation

            # Now compute paths for ALL solvers
            # Structure: solver_results[solver_name] = {config, path_free, path_obstacles, costs, timing, ...}
            solver_results: dict[str, dict] = {}

            for solver_cfg in solvers_configs:
                solver_cfg_resolved = _resolve_solver_config(solver_cfg)
                solver_name = solver_cfg.get("name", "unknown")
                backend, parsed_solver_name = _parse_solver_backend_and_name(solver_cfg_resolved)
                supports_considered_area = backend == "grid" and parsed_solver_name in {"astar", "grid_astar"}

                t_solver_start = time.time()

                # Compute path without obstacles (free)
                t_path_free = time.time()
                if supports_considered_area:
                    path_free_result = _grid_astar_with_details(
                        crop,
                        (i1, j1),
                        (i2, j2),
                        float(solver_cfg_resolved.get("weight", 1.0)),
                        str(solver_cfg_resolved.get("diagonal_movement", "always")),
                        str(solver_cfg_resolved.get("heuristic", "euclidean")),
                        str(solver_cfg_resolved.get("cost_model", "elevation")),
                    )
                    path_free = path_free_result.path
                    considered_area_free = int(path_free_result.considered_area)
                else:
                    path_free = find_path(crop, (i1, j1), (i2, j2), solver_cfg_resolved)
                    considered_area_free = -1
                t_path_free = time.time() - t_path_free

                # Compute path with obstacles
                t_path_obstacles = time.time()
                obstacle_surface = crop if blocked_obstacles is not None else masked_elevation
                if supports_considered_area:
                    path_obstacles_result = _grid_astar_with_details(
                        obstacle_surface,
                        (i1, j1),
                        (i2, j2),
                        float(solver_cfg_resolved.get("weight", 1.0)),
                        str(solver_cfg_resolved.get("diagonal_movement", "always")),
                        str(solver_cfg_resolved.get("heuristic", "euclidean")),
                        str(solver_cfg_resolved.get("cost_model", "elevation")),
                        blocked_obstacles,
                    )
                    path_obstacles = path_obstacles_result.path
                    considered_area_obstacles = int(path_obstacles_result.considered_area)
                else:
                    path_obstacles = find_path(
                        obstacle_surface,
                        (i1, j1),
                        (i2, j2),
                        solver_cfg_resolved,
                        blocked_mask=blocked_obstacles,
                    )
                    considered_area_obstacles = -1
                t_path_obstacles = time.time() - t_path_obstacles

                # Check if free path crosses obstacles
                path_free_crosses_obstacles = check_path_collision(path_free, obstacles_mask)

                # Compute costs using the solver's cost model
                def _sum_costs_for_solver(elevation_arr, path_list, cfg):
                    """Compute path cost using solver's cost model."""
                    if not path_list:
                        return 0.0

                    normalized, abs_norm, elev_range = _normalize_elevation_arrays(elevation_arr)
                    weight_param = float(cfg.get("weight", 1.0))
                    cost_model = cfg.get("cost_model", "elevation")

                    def _edge_weight(a, b):
                        ai, aj = int(a[0]), int(a[1])
                        bi, bj = int(b[0]), int(b[1])
                        di = bi - ai
                        dj = bj - aj

                        if abs(di) == 1 and abs(dj) == 1:
                            dist_m = DIAGONAL_DISTANCE_M
                        else:
                            dist_m = PIXEL_SIZE_M * math.hypot(di, dj)

                        if cost_model == "distance":
                            return float(dist_m)
                        if cost_model == "absolute_elevation":
                            dst_val = float(abs_norm[bi, bj])
                            return float(dist_m * (1.0 + weight_param * dst_val))
                        if cost_model == "slope_uphill":
                            if elev_range > 0:
                                delta = float(elevation_arr[bi, bj] - elevation_arr[ai, aj])
                                slope_norm = max(0.0, delta) / elev_range
                            else:
                                slope_norm = 0.0
                            return float(dist_m * (1.0 + weight_param * slope_norm))
                        if cost_model == "energy":
                            delta = float(elevation_arr[bi, bj] - elevation_arr[ai, aj])
                            return float(dist_m + weight_param * max(0.0, delta))
                        # default: 'elevation'
                        dst_val = float(normalized[bi, bj])
                        return float(dist_m * (1.0 + weight_param * dst_val))

                    total = 0.0
                    for a, b in zip(path_list, path_list[1:]):
                        try:
                            total += _edge_weight(a, b)
                        except Exception:
                            total += float(math.hypot(b[0] - a[0], b[1] - a[1]))
                    return total

                t_cost = time.time()
                cost_free = _sum_costs_for_solver(crop, path_free, solver_cfg_resolved)
                obstacle_cost_surface = crop if blocked_obstacles is not None else masked_elevation
                cost_obstacles = _sum_costs_for_solver(obstacle_cost_surface, path_obstacles, solver_cfg_resolved)
                t_cost = time.time() - t_cost

                # Compute cost difference percentage
                if cost_free > 0:
                    cost_diff_percent = float((cost_obstacles - cost_free) / cost_free * 100.0)
                else:
                    cost_diff_percent = 0.0

                solver_total_time = time.time() - t_solver_start

                solver_results[solver_name] = {
                    "config": {
                        "backend": backend,
                        "solver": parsed_solver_name,
                        "diagonal_movement": solver_cfg_resolved.get("diagonal_movement", "always"),
                        "weight": solver_cfg_resolved.get("weight", 1.0),
                        "heuristic": solver_cfg_resolved.get("heuristic", "euclidean"),
                        "cost_model": solver_cfg_resolved.get("cost_model", "elevation"),
                    },
                    "path_free": path_free,
                    "path_obstacles": path_obstacles,
                    "path_free_crosses_obstacles": path_free_crosses_obstacles,
                    "considered_area_free": considered_area_free,
                    "considered_area_obstacles": considered_area_obstacles,
                    "cost_free": cost_free,
                    "cost_obstacles": cost_obstacles,
                    "cost_diff_percent": cost_diff_percent,
                    "timing": {
                        "path_free_seconds": float(t_path_free),
                        "path_obstacles_seconds": float(t_path_obstacles),
                        "cost_compute_seconds": float(t_cost),
                        "total_seconds": float(solver_total_time),
                    },
                }

            # Successfully found a path - proceed with sample generation
            # Normalize to float32 in [0, 1] using global_min/global_max
            crop_f = crop.astype(np.float32)
            logger.debug(f"crop_f min/max before normalization: {crop_f.min()}/{crop_f.max()}")
            crop_norm = (crop_f - global_min) / (global_max - global_min)
            logger.debug(f"crop_norm min/max after normalization: {crop_norm.min()}/{crop_norm.max()}")
            crop_norm = np.clip(crop_norm, 0.0, 1.0).astype(np.float32)
            logger.debug(f"crop_norm min/max after clipping: {crop_norm.min()}/{crop_norm.max()}")

            # Create marker channel: -1 everywhere, start_point=0, end_point=1
            marker_channel = np.full_like(crop_norm, -1.0, dtype=np.float32)
            marker_channel[i1, j1] = 0.0
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
            base_res_dir = outputs_dirpath / str(resolution)
            split_root = base_res_dir / split_name
            tensors_dir = split_root / "tensors" / f"{lat_to_str(center_lat)}" / f"{lon_to_str(center_lon)}"
            tensors_dir.mkdir(parents=True, exist_ok=True)
            tensor_path = tensors_dir / f"{index:010d}{TENSOR_SUFFIX}"
            try:
                t_np = time.time()
                save_tensor(crop_with_markers, tensor_path)
                tensor_save_time = time.time() - t_np
            except Exception as e:  # pragma: no cover
                tensor_save_time = time.time() - t_np if "t_np" in locals() else 0.0
                logger.warning(f"Failed saving tensor {tensor_path}: {e}")

            def _wrap_lon_float(lon_val: float) -> float:
                return ((lon_val + 180.0) % 360.0) - 180.0

            # Build list of tiles used to generate the elevation crop from loaded_tiles
            tiles_used = []
            tiles_overlapping = []
            for tile_lat, tile_lon, row_idx, col_idx in loaded_tiles:
                tile_top = row_idx * TILE_SIZE_PX
                tile_left = col_idx * TILE_SIZE_PX
                # Skip out-of-bounds latitudes (consistent with earlier behavior)
                if tile_lat < MIN_LAT or tile_lat >= MAX_LAT:
                    continue
                tile_record = {
                    "lat": int(tile_lat),
                    "lon": int(tile_lon),
                    "name": f"{lat_to_str(tile_lat)}{lon_to_str(tile_lon)}",
                }
                tiles_used.append(tile_record)

                # Check overlap with crop bounds
                tile_bottom = tile_top + TILE_SIZE_PX
                tile_right = tile_left + TILE_SIZE_PX
                overlaps = not (tile_right <= x1 or tile_left >= x2 or tile_bottom <= y1 or tile_top >= y2)
                if overlaps:
                    tiles_overlapping.append(tile_record)

            # Calculate geographic coordinates for crop bounds/center
            px_per_deg = TILE_SIZE_PX
            crop_center_y = y1 + resolution / 2.0
            crop_center_x = x1 + resolution / 2.0
            center_px_in_center_tile_y = crop_center_y - center_offset
            center_px_in_center_tile_x = crop_center_x - center_offset
            geo_center_lat = center_lat + 1.0 - (center_px_in_center_tile_y / px_per_deg)
            geo_center_lon = _wrap_lon_float(center_lon + (center_px_in_center_tile_x / px_per_deg))

            top_left_px_in_center_tile_y = y1 - center_offset
            top_left_px_in_center_tile_x = x1 - center_offset
            geo_top_left_lat = center_lat + 1.0 - (top_left_px_in_center_tile_y / px_per_deg)
            geo_top_left_lon = _wrap_lon_float(center_lon + (top_left_px_in_center_tile_x / px_per_deg))

            bottom_right_px_in_center_tile_y = y2 - center_offset
            bottom_right_px_in_center_tile_x = x2 - center_offset
            geo_bottom_right_lat = center_lat + 1.0 - (bottom_right_px_in_center_tile_y / px_per_deg)
            geo_bottom_right_lon = _wrap_lon_float(center_lon + (bottom_right_px_in_center_tile_x / px_per_deg))

            # Build metadata with per-solver paths and configs
            # Structure: solvers[solver_name] = {config, paths: {free, obstacles}, timing}
            solvers_metadata: dict[str, dict] = {}
            paths_metadata: dict[str, dict] = {}

            for solver_name, result in solver_results.items():
                solvers_metadata[solver_name] = result["config"]
                paths_metadata[solver_name] = {
                    "free": {
                        "length": len(result["path_free"]),
                        "crosses_obstacles": bool(result["path_free_crosses_obstacles"]),
                        "considered_area": int(result.get("considered_area_free", -1)),
                        "cost": float(result["cost_free"]),
                    },
                    "obstacles": {
                        "length": len(result["path_obstacles"]),
                        "crosses_obstacles": False,
                        "considered_area": int(result.get("considered_area_obstacles", -1)),
                        "cost": float(result["cost_obstacles"]),
                        "cost_diff_percent": float(result["cost_diff_percent"]),
                    },
                    "timing": result["timing"],
                }

            # Save metadata into metadata folder *before* images so visualization can read it
            base_res_dir = outputs_dirpath / str(resolution)
            split_root = base_res_dir / split_name
            metadata_dir = split_root / "metadata" / f"{lat_to_str(center_lat)}" / f"{lon_to_str(center_lon)}"
            metadata = {
                "index": index,
                "resolution": resolution,
                "split": split_name,
                "calibration": {
                    "min": float(global_min),
                    "max": float(global_max),
                },
                "tile": {
                    "source": {
                        "lat": center_lat,
                        "lon": center_lon,
                        "name": f"{lat_to_str(center_lat)}{lon_to_str(center_lon)}",
                    },
                    "stats": {
                        "min_elevation": tile_min_elev,
                        "max_elevation": tile_max_elev,
                    },
                    "used": tiles_used,
                    "overlapping": tiles_overlapping,
                },
                "crop": {
                    "min_elevation": crop_min_elev,
                    "max_elevation": crop_max_elev,
                    "target_elevation": float(target_elev),
                    "center": {"lat": float(geo_center_lat), "lon": float(geo_center_lon)},
                    "top_left": {"lat": float(geo_top_left_lat), "lon": float(geo_top_left_lon)},
                    "bottom_right": {"lat": float(geo_bottom_right_lat), "lon": float(geo_bottom_right_lon)},
                },
                "markers": {
                    "start_point": {"i": int(i1), "j": int(j1), "elevation": marker_1_elev},
                    "end_point": {"i": int(i2), "j": int(j2), "elevation": marker_2_elev},
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
                "provenance": {
                    "generation_seed": generation_seed,
                    "sample_seed": sample_seed,
                },
                "solvers": solvers_metadata,
                "paths": paths_metadata,
                "timing": {
                    "tile_loading_seconds": float(tile_loading_time),
                    "marker_search_seconds": float(marker_search_time),
                    "obstacle_generation_seconds": float(obstacle_gen_total),
                    "tensor_save_seconds": float(tensor_save_time),
                    "metadata_write_seconds": float(metadata_write_time),
                    "image_conversion_seconds": float(image_conversion_time),
                    "image_write_seconds": float(image_write_time),
                },
                "timestamp": datetime.now().isoformat(),
            }
            metadata_dir.mkdir(parents=True, exist_ok=True)
            metadata_path = metadata_dir / f"{index:010d}.yaml"
            with open(metadata_path, "w") as f:
                yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)

            # Save pixel-index paths as CSV files under paths/<solver_name>/free and paths/<solver_name>/obstacles
            try:
                lat_dir = f"{lat_to_str(center_lat)}"
                lon_dir = f"{lon_to_str(center_lon)}"
                base_res_dir = outputs_dirpath / str(resolution)

                for solver_name, result in solver_results.items():
                    # Free path (without obstacles)
                    free_dir = base_res_dir / split_name / "paths" / solver_name / "free" / lat_dir / lon_dir
                    free_dir.mkdir(parents=True, exist_ok=True)
                    free_csv_file = free_dir / f"{index:010d}.csv"
                    with open(free_csv_file, "w", newline="") as ff:
                        writer = csv.writer(ff)
                        for pi, pj in result["path_free"]:
                            writer.writerow([int(pi), int(pj)])

                    # Obstacles path (with obstacles)
                    obs_dir = base_res_dir / split_name / "paths" / solver_name / "obstacles" / lat_dir / lon_dir
                    obs_dir.mkdir(parents=True, exist_ok=True)
                    obs_csv_file = obs_dir / f"{index:010d}.csv"
                    with open(obs_csv_file, "w", newline="") as ff:
                        writer = csv.writer(ff)
                        for pi, pj in result["path_obstacles"]:
                            writer.writerow([int(pi), int(pj)])
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed saving pixel-index paths for sample {index}: {e}")

            # Save visual image at full resolution in 'images' unless disabled
            # Use paths from first solver for visualization
            first_solver_name = list(solver_results.keys())[0]
            first_result = solver_results[first_solver_name]
            path_without = first_result["path_free"]
            path_with = first_result["path_obstacles"]

            if not disable_images and save_preview_images:
                # No generic sample image is written any more; only per-solver images are produced for export.
                base_res_dir = outputs_dirpath / str(resolution)
                split_root = base_res_dir / split_name
                # Provide default timing variables so metadata fields remain present
                image_conversion_time = 0.0
                image_write_time = 0.0

                # Save per-solver images (image_<solver_name>.png) under images/<solver_name>/<lat>/<lon> for export
                try:
                    for solver_name, result in solver_results.items():
                        solver_images_dir = (
                            split_root
                            / "images"
                            / solver_name
                            / f"{lat_to_str(center_lat)}"
                            / f"{lon_to_str(center_lon)}"
                        )
                        solver_images_dir.mkdir(parents=True, exist_ok=True)
                        s_path_without = result.get("path_free", [])
                        s_path_with = result.get("path_obstacles", [])
                        if (s_path_with == s_path_without) or (not obstacles_enabled):
                            solver_thumb = tensor_to_image(crop_with_markers, path_pixels=s_path_without)
                        else:
                            solver_thumb = tensor_to_image(
                                crop_with_markers, path_pixels=s_path_without, alt_path_pixels=s_path_with
                            )
                        solver_image_path = solver_images_dir / f"{index:010d}.png"
                        try:
                            _save_png(solver_image_path, solver_thumb)
                        except Exception as e:  # pragma: no cover
                            logger.warning(f"Failed saving solver image {solver_image_path}: {e}")
                except Exception as e:  # pragma: no cover
                    logger.warning(f"Failed saving per-solver images for sample {index}: {e}")
            # Update metadata with measured timings and rewrite metadata file
            try:
                t_meta = time.time()
                metadata["timing"].update(
                    {
                        "tensor_save_seconds": float(tensor_save_time),
                        "image_conversion_seconds": float(image_conversion_time),
                        "image_write_seconds": float(image_write_time),
                        "tile_loading_seconds": float(tile_loading_time),
                        "marker_search_seconds": float(marker_search_time),
                        "obstacle_generation_seconds": float(obstacle_gen_total),
                    }
                )
                metadata_write_time = time.time() - t_meta
                metadata["timing"]["metadata_write_seconds"] = float(metadata_write_time)
                with open(metadata_path, "w") as f:
                    yaml.dump(metadata, f, default_flow_style=False, sort_keys=False)
            except Exception as e:  # pragma: no cover
                logger.warning(f"Failed updating metadata {metadata_path}: {e}")

            # Log per-sample diagnostics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            num_solvers = len(solver_results)
            first_solver_name = list(solver_results.keys())[0]
            # logger.info(
            #     f"Resolution: {resolution} / Sample {index}: success=True, elevation_attempts={elevation_attempts}, "
            #     f"marker_retries={marker_attempt + 1 if 'marker_attempt' in locals() else 0}, "
            #     f"solvers={num_solvers}, first_solver_path_len={len(solver_results[first_solver_name]['path_free'])}, "
            #     f"grid_shape=({grid_h},{grid_w}), duration={duration:.2f}s"
            # )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            return index, True, f"Generated sample {index}", duration

        except Exception as e:
            logger.debug(f"Elevation attempt {elevation_attempt} failed: {e}")
            continue

    duration = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Resolution: {resolution} / Sample {index}: success=False, elevation_attempts={elevation_attempts}, "
        f"marker_retries={marker_attempts_used if 'marker_attempts_used' in locals() else 0}, duration={duration:.2f}s"
    )
    return index, False, f"Failed to generate sample {index} after {max_elevation_retries} elevation attempts", duration


@_log_timing
def _generate_sample_wrapper(args):
    """Wrapper for multiprocessing."""
    return generate_sample(args)


@_log_timing
def count_valid_samples(
    res_dir: Path,
    disable_images: bool = False,
    split_assignments: dict[int, str] | None = None,
    preview_image_indices_by_split: dict[str, set[int]] | None = None,
    solver_names: list[str] | None = None,
) -> int:
    """
    Count consecutive valid samples starting from index 0.

    A sample is valid if it has all required files under the expected split/lat/lon subdirectories:
    - {split}/tensors/{lat}/{lon}/{index:010d}.npy.zst
    - {split}/metadata/{lat}/{lon}/{index:010d}.yaml
    - {split}/images/{solver}/{lat}/{lon}/{index:010d}.png for preview-subset rows
      when images are enabled

    Stops counting at the first incomplete sample and deletes any partial files.

    Args:
        res_dir: Resolution directory containing split subdirs.
        disable_images: If True, don't check for images.
        split_assignments: Mapping from sample index -> split name. If None, assumes a layout without split directories.
        preview_image_indices_by_split: Optional mapping from split name to the set of
            sample indices that are expected to have preview images.
        solver_names: Optional solver names expected under ``images/<solver>/...``

    Returns:
        Number of consecutive valid samples from index 0.
    """

    def _build_index_map(dir_path: Path, suffix: str) -> dict[int, Path]:
        index_map: dict[int, Path] = {}
        if not dir_path.exists():
            return index_map
        for path in dir_path.rglob(f"*{suffix}"):
            if not path.is_file() or not str(path).endswith(suffix):
                continue
            # Handle multi-part extensions like .npy.zst
            stem = path.name
            if stem.endswith(suffix):
                stem = stem[: -len(suffix)]
            if not stem.isdigit():
                continue
            idx = int(stem)
            if idx in index_map and index_map[idx] != path:
                logger.warning(
                    f"Duplicate sample index {idx} under {dir_path}; keeping {index_map[idx]} and ignoring {path}"
                )
                continue
            index_map[idx] = path
        return index_map

    if not split_assignments:
        tensors_dir = res_dir / "tensors"
        metadata_dir = res_dir / "metadata"
        images_dir = res_dir / "images"

        if not tensors_dir.exists() or not metadata_dir.exists():
            return 0

        tensors_map = _build_index_map(tensors_dir, TENSOR_SUFFIX)
        metadata_map = _build_index_map(metadata_dir, ".yaml")
        images_map = _build_index_map(images_dir, ".png") if not disable_images else {}
        index = 0
        while True:
            tensor_file = tensors_map.get(index)
            metadata_file = metadata_map.get(index)
            image_file = images_map.get(index) if not disable_images else None

            tensor_exists = tensor_file is not None
            metadata_exists = metadata_file is not None
            image_exists = image_file is not None if not disable_images else True

            if tensor_exists and metadata_exists and image_exists:
                index += 1
            else:
                partial_files = []
                if tensor_exists and tensor_file is not None:
                    partial_files.append(tensor_file)
                if metadata_exists and metadata_file is not None:
                    partial_files.append(metadata_file)
                if not disable_images and image_exists and image_file is not None:
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

    # Split-aware layout
    split_names = sorted(set(split_assignments.values()))
    tensors_map_by_split: dict[str, dict[int, Path]] = {}
    metadata_map_by_split: dict[str, dict[int, Path]] = {}
    image_maps_by_split: dict[str, dict[str, dict[int, Path]] | dict[int, Path]] = {}

    solver_names = list(solver_names or [])

    for split in split_names:
        tensors_map_by_split[split] = _build_index_map(res_dir / split / "tensors", TENSOR_SUFFIX)
        metadata_map_by_split[split] = _build_index_map(res_dir / split / "metadata", ".yaml")
        if not disable_images:
            if solver_names:
                image_maps_by_split[split] = {
                    solver_name: _build_index_map(res_dir / split / "images" / solver_name, ".png")
                    for solver_name in solver_names
                }
            else:
                image_maps_by_split[split] = _build_index_map(res_dir / split / "images", ".png")

    index = 0
    max_index = max(split_assignments.keys()) + 1 if split_assignments else 0
    while index < max_index:
        split = split_assignments.get(index, "train")
        tensor_file = tensors_map_by_split.get(split, {}).get(index)
        metadata_file = metadata_map_by_split.get(split, {}).get(index)

        requires_images = False
        if not disable_images:
            if preview_image_indices_by_split is None:
                requires_images = True
            else:
                requires_images = index in preview_image_indices_by_split.get(split, set())

        tensor_exists = tensor_file is not None
        metadata_exists = metadata_file is not None
        image_exists = True
        existing_image_files: list[Path] = []
        if not disable_images:
            if solver_names:
                solver_image_maps = image_maps_by_split.get(split, {})
                for solver_name in solver_names:
                    image_file = solver_image_maps.get(solver_name, {}).get(index)
                    if image_file is not None:
                        existing_image_files.append(image_file)
                image_exists = (not requires_images) or len(existing_image_files) == len(solver_names)
            else:
                image_file = image_maps_by_split.get(split, {}).get(index)
                if image_file is not None:
                    existing_image_files.append(image_file)
                image_exists = (not requires_images) or image_file is not None

        if tensor_exists and metadata_exists and image_exists:
            index += 1
        else:
            partial_files = []
            if tensor_exists and tensor_file is not None:
                partial_files.append(tensor_file)
            if metadata_exists and metadata_file is not None:
                partial_files.append(metadata_file)
            partial_files.extend(existing_image_files)

            if partial_files:
                logger.warning(
                    f"Found incomplete sample at index {index} (split={split}), deleting partial files: "
                    f"{[str(f.name) for f in partial_files]}"
                )
                for f in partial_files:
                    try:
                        f.unlink()
                    except Exception as e:  # pragma: no cover
                        logger.warning(f"Failed to delete partial file {f}: {e}")
            break

    return index


def _assign_splits(
    num_samples: int,
    ratios: dict[str, float] | None,
    seed: int = 42,
) -> dict[int, str]:
    """Assign each sample index to a split using deterministic rounding + shuffle.

    Ensures the exact number of samples matches num_samples, distributing leftovers by largest fractional remainder.
    """

    if num_samples <= 0:
        return {}

    default_order = ["train", "calibration", "validation", "test"]
    ratios = ratios or {"train": 1.0}

    # Keep only positive ratios
    ratios = {k: float(v) for k, v in ratios.items() if float(v) > 0.0}
    if not ratios:
        ratios = {"train": 1.0}

    # Preserve deterministic order using default_order first, then any extra keys sorted
    ordered_splits = [s for s in default_order if s in ratios] + [s for s in sorted(ratios) if s not in default_order]
    total_ratio = sum(ratios.values())
    if total_ratio <= 0:
        ratios = {"train": 1.0}
        ordered_splits = ["train"]
        total_ratio = 1.0

    counts: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for split in ordered_splits:
        weight = ratios.get(split, 0.0) / total_ratio
        target = weight * num_samples
        base = int(math.floor(target))
        counts[split] = base
        remainders.append((target - base, split))
        assigned += base

    leftover = num_samples - assigned
    remainders.sort(
        key=lambda x: (-x[0], ordered_splits.index(x[1]) if x[1] in ordered_splits else len(ordered_splits))
    )
    for i in range(leftover):
        _, split = remainders[i % len(remainders)]
        counts[split] += 1

    assignments: list[str] = []
    for split in ordered_splits:
        assignments.extend([split] * counts.get(split, 0))

    rng = random.Random(seed)
    rng.shuffle(assignments)

    return {i: assignments[i] for i in range(num_samples)}


@_log_timing
def run_generate(
    outputs_dirpath: str,
    config_filepath: str,
    tiles_dirpath: str,
    calibration_filepath: str = "outputs/calibrate/calibrate.json",
    resolution: int | list[int] | None = None,
    num_samples: int | None = None,
    num_cpus: int | None = None,
    disable_images: bool = False,
    force: bool = False,
    scene_indices: range | list[int] | None = None,
) -> dict | None:
    """
    Main orchestration function for generating dataset samples.

    Args:
        outputs_dirpath: Directory path for output files.
        config_filepath: Path to config.yaml file.
        tiles_dirpath: Directory containing DSM tiles.
        calibration_filepath: Path to calibration JSON file.
        resolution: Image resolution (single int) or list of resolutions.
        num_samples: Number of samples to generate.
        num_cpus: Number of CPUs to use for multiprocessing.
        disable_images: If True, skip saving PNG images.
        force: If True, delete existing directories and regenerate all samples; the release generator refuses to overwrite an existing output root.
        scene_indices: Global indices for a bounded release batch; preserve the full sample count.
    """
    # Load config
    config_path = Path(config_filepath)
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return

    with open(config_path) as f:
        config = yaml.safe_load(f)

    dataset_version = config.get("generation", {}).get("dataset_version") if isinstance(config, dict) else None
    if dataset_version is not None:
        from passage.successor_generation import DATASET_VERSION, run_successor_generation

        if dataset_version != DATASET_VERSION:
            raise ValueError(f"Unsupported generation dataset version: {dataset_version}")
        return run_successor_generation(
            outputs_dirpath, config_filepath, tiles_dirpath, calibration_filepath,
            resolution=resolution, num_samples=num_samples, num_cpus=num_cpus,
            disable_images=disable_images, force=force, scene_indices=scene_indices,
        )
    if scene_indices is not None:
        raise ValueError("Bounded scene indices require an explicit release dataset version")

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

    # num_samples can be either a single integer (global) or a mapping per-resolution
    cfg_num_samples = num_samples if num_samples is not None else config.get("num_samples", 100)

    splits_cfg = config.get("splits", {}) if isinstance(config, dict) else {}
    splits_enabled = bool(splits_cfg.get("enabled", True))
    split_ratios = splits_cfg.get("ratios", {}) if splits_enabled else {"train": 1.0}
    split_seed = int(splits_cfg.get("seed", 42) if splits_enabled else 42)
    generation_cfg = config.get("generation", {}) if isinstance(config, dict) else {}
    generation_seed = int(generation_cfg.get("seed", split_seed))
    preview_images_cfg = _resolve_preview_images_config(config, resolutions)
    # Note: split assignments will be computed per-resolution below using the per-resolution sample count
    split_assignments = None  # placeholder (computed per-resolution)

    outputs_path = Path(outputs_dirpath)
    tiles_path = Path(tiles_dirpath)

    if not tiles_path.exists():
        logger.warning(f"Tiles directory not found: {tiles_path}; continuing without tile files")
        # Tiles directory is optional for some workflows (e.g., tests that only need calibration and mocked generation)
        # Do not return early; downstream code (and tests) may rely on generate_sample being mocked.

    # Load calibrate.json to get global min/max and tile_stats
    calibration_path = Path(calibration_filepath)
    if not calibration_path.exists():
        logger.error(f"Calibration file not found: {calibration_path}. Run 'passage calibrate' first.")
        return

    try:
        with open(calibration_path) as f:
            calibration = json.load(f)
        global_min = calibration["global_min"]
        global_max = calibration["global_max"]
        tile_stats = calibration["tile_stats"]
        logger.info(
            f"Loaded calibration from {calibration_path}: "
            f"global_min={global_min}, global_max={global_max}, tiles={len(tile_stats)}"
        )
    except Exception as e:
        logger.error(f"Failed to load calibration file: {e}")
        return

    # Load tile_stats into pandas DataFrame
    tile_stats_df = pd.DataFrame(tile_stats)
    logger.info(f"Loaded {len(tile_stats_df)} tiles into DataFrame")

    # Load hold-out mask for geographic split filtering
    # The mask path can be specified in config or use default
    hold_out_cfg = config.get("hold_out", {}) if isinstance(config, dict) else {}
    hold_out_enabled = hold_out_cfg.get("enabled", True)  # Enabled by default if mask exists
    hold_out_mask_path = hold_out_cfg.get("mask_path", DEFAULT_HOLD_OUT_MASK_PATH)

    if hold_out_enabled:
        hold_out_mask = load_hold_out_mask(hold_out_mask_path)

        if hold_out_mask is not None:
            # Log statistics about tile distribution
            test_tiles = filter_tiles_by_hold_out(tile_stats_df, "test", hold_out_mask)
            train_tiles = filter_tiles_by_hold_out(tile_stats_df, "train", hold_out_mask)
            logger.info(
                f"Hold-out mask applied: {len(test_tiles)} tiles for test (hold-out region), "
                f"{len(train_tiles)} tiles for train/val/calibration (outside hold-out)"
            )
    else:
        logger.info("Hold-out geographic filtering is disabled in config")
        # Clear any cached mask to ensure no filtering happens
        global _HOLD_OUT_MASK, _HOLD_OUT_MASK_PATH
        _HOLD_OUT_MASK = None
        _HOLD_OUT_MASK_PATH = None

    # Load solvers configuration from 'solvers' key (list of solver configs)
    solvers_configs = config.get("solvers")
    # If solvers not present in provided config, fall back to package default config
    if not solvers_configs:
        try:
            default_cfg_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
            if default_cfg_path.exists():
                default_cfg = yaml.safe_load(default_cfg_path.read_text())
                solvers_configs = default_cfg.get("solvers", [])
                if solvers_configs:
                    logger.info(f"Loaded solvers from default config at {default_cfg_path}")
        except Exception as e:
            logger.debug(f"Failed loading default config for solvers: {e}")

    if not solvers_configs or not isinstance(solvers_configs, list):
        logger.error("No 'solvers' configuration found in config file. Expected a list of solver configs.")
        return

    logger.info(f"Loaded {len(solvers_configs)} solver configurations:")
    for solver_cfg in solvers_configs:
        solver_name = solver_cfg.get("name", "unknown")
        resolved = _resolve_solver_config(solver_cfg)
        backend, parsed_name = _parse_solver_backend_and_name(resolved)
        logger.info(
            f"  - {solver_name}: backend={backend}, solver={parsed_name}, "
            f"cost_model={resolved.get('cost_model', 'elevation')}, "
            f"weight={resolved.get('weight', 1.0)}"
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
        f"num_samples={cfg_num_samples}, requested_cpus={cpus}, disable_images={disable_images}, "
        f"split_seed={split_seed}, generation_seed={generation_seed}, "
        f"preview_images_enabled={preview_images_cfg['enabled']}, "
        f"preview_images_per_split={preview_images_cfg['num_samples_per_split']}"
    )

    # Generate samples for each resolution
    for res in resolutions:
        res_dir = outputs_path / str(res)

        # Resolve per-resolution sample count (accept dict keys as int or str, support 'default' or '*' fallback)
        if isinstance(cfg_num_samples, dict):
            samples_for_res = (
                cfg_num_samples.get(res)
                or cfg_num_samples.get(str(res))
                or cfg_num_samples.get("default")
                or cfg_num_samples.get("*")
                or 100
            )
        else:
            samples_for_res = int(cfg_num_samples)

        try:
            samples_for_res = int(samples_for_res)
        except Exception:
            logger.error(f"Invalid num_samples value for resolution {res}: {samples_for_res}")
            return

        if samples_for_res <= 0:
            logger.error(f"num_samples for resolution {res} must be a positive integer")
            return

        # Compute split assignments for this resolution
        split_assignments = (
            _assign_splits(samples_for_res, split_ratios, seed=split_seed)
            if splits_enabled
            else {i: "train" for i in range(samples_for_res)}
        )
        split_names = sorted(set(split_assignments.values()), key=_split_sort_key)
        preview_image_indices_by_split = (
            _select_preview_image_indices(split_assignments, preview_images_cfg["num_samples_per_split"])
            if preview_images_cfg["enabled"] and res in preview_images_cfg["resolutions"]
            else {}
        )

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
                start_index = count_valid_samples(
                    res_dir,
                    disable_images,
                    split_assignments,
                    preview_image_indices_by_split=preview_image_indices_by_split,
                    solver_names=[solver_cfg.get("name", "unknown") for solver_cfg in solvers_configs],
                )
                if start_index >= samples_for_res:
                    logger.info(
                        f"Resolution {res} already complete ({start_index}/{samples_for_res} samples), skipping"
                    )
                    continue
                if start_index > 0:
                    logger.info(
                        f"Resuming generation for resolution {res}: found {start_index} valid samples, "
                        f"generating {samples_for_res - start_index} more"
                    )
                    logger.warning(
                        "Note: Calibration data may have changed since previous generation — "
                        "samples may have inconsistent normalization"
                    )

        samples_to_generate = samples_for_res - start_index
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
        for split in split_names:
            split_root = res_dir / split
            (split_root / "tensors").mkdir(parents=True, exist_ok=True)
            (split_root / "metadata").mkdir(parents=True, exist_ok=True)
            if not disable_images and preview_image_indices_by_split.get(split):
                (split_root / "images").mkdir(parents=True, exist_ok=True)

        # Prepare arguments for each sample (starting from start_index)
        args_list = []
        split_requested = {s: 0 for s in split_names}
        split_success = {s: 0 for s in split_names}
        for i in range(start_index, samples_for_res):
            split = split_assignments.get(i, "train")
            split_requested[split] = split_requested.get(split, 0) + 1
            save_preview_images = bool(
                preview_image_indices_by_split.get(split) and i in preview_image_indices_by_split[split]
            )
            args_list.append(
                (
                    i,
                    tiles_path,
                    outputs_path,
                    res,
                    tile_stats_df,
                    global_min,
                    global_max,
                    disable_images,
                    solvers_configs,
                    obstacle_settings,
                    split,
                    generation_seed,
                    save_preview_images,
                )
            )

        # Process samples
        durations = []
        successful = 0
        pool_size = _resolve_generation_pool_size(num_cpus, res, samples_to_generate)
        if pool_size == 0:
            logger.info("Running in sequential mode (no multiprocessing)")
            for args in tqdm(args_list, total=samples_to_generate):
                idx, success, msg, duration = generate_sample(args)
                durations.append(duration)
                if not success:
                    logger.warning(msg)
                else:
                    successful += 1
                    split_success[split_assignments.get(idx, "train")] = (
                        split_success.get(split_assignments.get(idx, "train"), 0) + 1
                    )
        else:
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
                        split_success[split_assignments.get(idx, "train")] = (
                            split_success.get(split_assignments.get(idx, "train"), 0) + 1
                        )

        logger.info(f"Completed generation for resolution {res}")

        # Save per-resolution summary inside the resolution directory
        mean_time = float(sum(durations) / len(durations)) if len(durations) > 0 else 0.0
        res_summary = {
            "resolution": res,
            "requested_samples": samples_for_res,
            "num_samples": samples_for_res,
            "start_index": start_index,
            "generated_samples": samples_to_generate,
            "successful_samples": successful,
            "mean_time_per_sample_seconds": mean_time,
            "tiles_count": len(tile_stats_df),
            "tiles_dirpath": str(tiles_path),
            "global_min": global_min,
            "global_max": global_max,
            "splits": {
                split: {
                    "requested": split_requested.get(split, 0),
                    "successful": split_success.get(split, 0),
                }
                for split in split_names
            },
            "split_seed": split_seed,
            "generation": {
                "seed": generation_seed,
                "sample_seeding": "blake2b(seed,resolution,index,split)",
                "timestamps_are_runtime_dependent": True,
            },
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
        "num_samples": cfg_num_samples,
        "tiles_count": len(tile_stats_df),
        "tiles_dirpath": str(tiles_path),
        "global_min": global_min,
        "global_max": global_max,
        "splits": {
            "enabled": splits_enabled,
            "ratios": split_ratios,
            "seed": split_seed,
        },
        "generation": {
            "seed": generation_seed,
            "sample_seeding": "blake2b(seed,resolution,index,split)",
            "timestamps_are_runtime_dependent": True,
        },
        "mean_time_per_sample_seconds": mean_time_overall,
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = outputs_path / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Generation completed. Summary saved to {summary_path}")
