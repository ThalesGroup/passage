"""Offline supervision under the released directed grid (never candidate oracle input)."""

from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

MOVES = tuple((r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if (r, c) != (0, 0))


def reverse_targets(
    terrain: np.ndarray, obstacles: np.ndarray, start: tuple[int, int], goal: tuple[int, int], weight: float = 62.4
) -> tuple[np.ndarray, np.ndarray]:
    """All costs to goal and an exact route; reverse each *directed* edge correctly."""
    terrain = np.asarray(terrain, dtype=np.float64)
    obstacles = np.asarray(obstacles, dtype=bool)
    if (
        terrain.ndim != 2
        or obstacles.shape != terrain.shape
        or not np.isfinite(terrain).all()
        or not np.isfinite(weight)
        or weight < 0
    ):
        raise ValueError("finite terrain, matching obstacles and nonnegative weight required")
    height, width = terrain.shape
    for point in (start, goal):
        if len(point) != 2 or not (0 <= point[0] < height and 0 <= point[1] < width) or obstacles[tuple(point)]:
            raise ValueError("endpoints must be free and inside the grid")
    minimum, span = terrain.min(), np.ptp(terrain)
    normalized = (terrain - minimum) / span if span else np.zeros_like(terrain)
    ids = np.arange(terrain.size).reshape(terrain.shape)
    sources, destinations, weights = [], [], []
    for dr, dc in MOVES:
        sr, sc = slice(max(0, -dr), min(height, height - dr)), slice(max(0, -dc), min(width, width - dc))
        tr, tc = slice(max(0, dr), min(height, height + dr)), slice(max(0, dc), min(width, width + dc))
        valid = ~obstacles[sr, sc] & ~obstacles[tr, tc]
        sources.append(ids[sr, sc][valid])
        destinations.append(ids[tr, tc][valid])
        weights.append((30 * np.hypot(dr, dc) * (1 + weight * normalized[tr, tc]))[valid])
    # Search from goal on reverse graph: c_reverse(v,u) = c_original(u,v).
    graph = csr_matrix(
        (np.concatenate(weights), (np.concatenate(destinations), np.concatenate(sources))),
        shape=(terrain.size, terrain.size),
    )
    distances, parents = dijkstra(graph, directed=True, indices=int(ids[tuple(goal)]), return_predecessors=True)
    current, target = int(ids[tuple(start)]), int(ids[tuple(goal)])
    route = []
    if np.isfinite(distances[current]):
        for _ in range(terrain.size):
            route.append(divmod(current, width))
            if current == target:
                break
            current = int(parents[current])
            if current < 0:
                raise RuntimeError("shortest-path predecessor chain is broken")
        else:
            raise RuntimeError("shortest-path predecessor cycle")
    return distances.reshape(terrain.shape), np.asarray(route, dtype=np.int64).reshape(-1, 2)
