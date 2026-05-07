---
title: Solvers & Heuristics
---

# Solvers & Heuristics

PASSAGE supports **three solver backends** and **two search algorithms**, combined with **three admissible heuristics**. This modular architecture enables fair comparison of classical pathfinding algorithms on real terrain data.

---

## Solver Backends

### 1. NetworkX Backend

Uses the [NetworkX](https://networkx.org/) `DiGraph` with pre-computed edge weights:

- **Pros:** Mature library, full-featured, good for moderate grid sizes
- **Cons:** Higher memory usage, slower on large grids
- **Graph construction:** each pixel is a node; edges connect to 4 or 8 neighbors with weights from the chosen [cost model](cost-models.md)

```python
# Internal: builds a directed graph with weighted edges
graph = build_graph_from_elevation(elevation, weight, diagonal, cost_model)
path = nx.astar_path(graph, start, end, heuristic=h)
```

### 2. Pathfinding Backend

Uses the [`python-pathfinding`](https://github.com/brean/python-pathfinding) library:

- **Pros:** Purpose-built for grid pathfinding, moderate performance
- **Cons:** Node-weighted (not edge-weighted), less flexible cost models
- **Grid construction:** a weight matrix is built where each cell contains the cost to enter that cell

### 3. Grid Backend (Numba-JIT)

PASSAGE's custom implementation — a **pure-Python A\*/Dijkstra** with optional **Numba JIT compilation** for near-C performance:

- **Pros:** Fastest backend, full control over cost model, no external graph library overhead
- **Cons:** Requires Numba for best performance (graceful fallback to pure Python)
- **Implementation:** binary heap priority queue, inline cost computation, direct array access

!!! tip "Recommended backend"
    The `grid` backend is the **default and recommended** choice. With Numba available, it's the fastest option by a significant margin.

---

## Search Algorithms

### A\* Search

The classic optimal pathfinding algorithm. Expands nodes by minimizing:

$$
f(s) = g(s) + h(s)
$$

where $g(s)$ is the cost-so-far from start to node $s$, and $h(s)$ is the **heuristic estimate** of the remaining cost to the goal.

!!! success "Optimality guarantee"
    With an **admissible** heuristic ($h(s) \leq h^*(s)$ for all $s$), A\* is guaranteed to find the optimal path.

### Dijkstra's Algorithm

Equivalent to A\* with $h(s) = 0$ — explores all directions equally:

$$
f(s) = g(s)
$$

Dijkstra always finds the optimal path but explores more nodes than A\* with a good heuristic.

---

## Heuristics

PASSAGE implements three admissible heuristics. All use the physical pixel spacing of $\Delta = 30\,\text{m}$:

### Euclidean Heuristic

$$
h_{\text{euclidean}}(s, g) = \Delta \cdot \sqrt{(i_g - i_s)^2 + (j_g - j_s)^2}
$$

The straight-line distance. Tightest possible admissible heuristic for grids with diagonal movement.

### Manhattan Heuristic

$$
h_{\text{manhattan}}(s, g) = \Delta \cdot \left(|i_g - i_s| + |j_g - j_s|\right)
$$

Sum of horizontal and vertical distances. Admissible for 4-connected grids (no diagonals).

!!! warning
    Manhattan heuristic **overestimates** on 8-connected grids (with diagonal movement), making it **inadmissible** in that setting. Use `euclidean` or `chebyshev` instead.

### Chebyshev Heuristic

$$
h_{\text{chebyshev}}(s, g) = \Delta \cdot \max\left(|i_g - i_s|,\; |j_g - j_s|\right)
$$

The "chessboard" distance. Admissible for 8-connected grids and often tighter than Euclidean in practice.

---

## Heuristic Comparison

| Heuristic | Formula | Best for | Tightness |
|-----------|---------|----------|-----------|
| Euclidean | $\Delta\sqrt{\Delta i^2 + \Delta j^2}$ | 8-connected grids | Tightest |
| Manhattan | $\Delta(|\Delta i| + |\Delta j|)$ | 4-connected grids | Loose on 8-connected |
| Chebyshev | $\Delta \max(|\Delta i|, |\Delta j|)$ | 8-connected grids | Good |

---

## Diagonal Movement

PASSAGE supports three diagonal movement policies:

| Policy | Description | Neighbor count |
|--------|-------------|:--------------:|
| `always` | Diagonal moves always allowed | 8 |
| `never` | Only cardinal moves (N/S/E/W) | 4 |
| `if_at_most_one_obstacle` | Diagonal allowed unless both adjacent cells are blocked | 8 (conditional) |

The default configuration uses `always` with 8-connected movement.

---

## Performance Characteristics

Approximate timings for a single path computation on different resolutions (grid backend, Numba-JIT, single core):

| Resolution | A\* (typical) | Dijkstra (typical) |
|-----------|:------------:|:-----------------:|
| 64×64 | ~1 ms | ~2 ms |
| 256×256 | ~10 ms | ~50 ms |
| 1024×1024 | ~200 ms | ~2 s |
| 4096×4096 | ~5 s | ~30 s |

!!! note
    Actual timings depend heavily on terrain complexity, obstacle density, and start/end positions. The values above are rough order-of-magnitude estimates.

---

## Implementation Details

### Graph Template Caching

For the **NetworkX backend**, PASSAGE uses a `_GraphTemplate` class that caches the graph **topology** (nodes and edge indices) for a given grid shape and connectivity. Only the edge **weights** are recomputed per sample — avoiding the cost of rebuilding the full graph structure.

### Numba JIT Compilation

The grid backend's core algorithms (`_numba_astar_core`, `_numba_dijkstra_core`) are decorated with `@numba.njit`:

- First call triggers JIT compilation (~2-5 seconds)
- Subsequent calls run at native speed
- Graceful fallback to pure Python if Numba is not installed

### Path Cost Computation

After a path is found, its **total cost** is computed by summing edge weights along the path:

$$
C(\text{path}) = \sum_{k=0}^{|\text{path}|-2} w(p_k, p_{k+1})
$$

This cost is stored in the sample metadata alongside the path coordinates and solver timing information.
