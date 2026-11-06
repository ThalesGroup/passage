---
title: Solvers & Heuristics
---

# Solvers & Heuristics

All routes of **PASSAGE v1.0** come from one exact solver, the **reference solver** `passage.reference.solve_reference`. The generator also contains **three alternative backends** (NetworkX, pathfinding and a Numba grid solver), which offer A\*, Dijkstra and a few other searches, with three heuristics. The v1.0 release does not use them for its labels: `passage generate` runs them only when `generation.dataset_version` is absent from the configuration. Keep the value shipped in `config/config.yaml` to use the reference solver.

---

## Reference Solver (v1.0)

The reference solver is an **A\* search on an implicit grid**: it computes neighbors on the fly and never builds an edge graph. It is compiled with Numba.

- **Terrain.** Before each solve, the generator decodes the stored float32 tensor and turns the elevation channel back into meters. The solver then does all arithmetic in **float64**. Labels are therefore solved on the same terrain that users load.
- **Graph.** The graph is eight-connected, with steps of $30$ m (orthogonal) and $30\sqrt{2}$ m (diagonal). Obstacle cells are blocked. A diagonal step is checked against its **destination cell only**, so a route may pass between two obstacle cells that touch at a corner. Cells are given as `[row, col]`. The step costs are those of the [cost models](cost-models.md).
- **Heuristic.** The solver uses the Euclidean distance in meters, multiplied by $1 - 10^{-12}$. Every step costs at least its length, so this is a lower bound on the remaining cost for all five cost models. The small margin absorbs floating-point rounding in long sums.
- **Ties.** The queue orders cells by priority, then by row-major cell index. A cell's cost changes only on a strict improvement.
- **Outcomes.** `exact` means an optimal route was found; its cost is then recomputed along the route. `no_path` means the search proved that no route exists; the generator rejects start and goal cells that the obstacles disconnect before it solves (see [Blocked Cells and Connected Endpoints](obstacles.md#blocked-cells-and-connected-endpoints)), so its labels never take this outcome. `censored` means an expansion budget stopped the search; the variant is stored as `failed`. PASSAGE v1.0 sets no budget (`generation.max_reference_expansions: null`), so every variant is `exact` and stored with the status `ok`.

The graph is fixed by a **graph specification** (`GraphSpec` in `passage_eval.protocol.contracts`), stored with every scene:

| Field | v1.0 value |
|-------|-----------|
| `connectivity` | 8 |
| `diagonal_policy` | `destination_only` |
| `obstacle_policy` | `blocked` |
| `pixel_size_m` | 30.0 |
| `coordinate_order` | `row_col` |
| `cost_model`, `cost_weight` | Per objective, see [Cost Models](cost-models.md) |
| `rtol`, `atol` | $10^{-7}$, $10^{-6}$ (tolerances for comparing costs) |

### Which Configuration Keys Apply

The v1.0 generator reads three keys from each entry of `solvers` in `config/config.yaml`: `name`, `cost_model` and `weight`. It requires exactly three entries, one for each of `elevation`, `energy` and `slope_uphill`, and it refuses any `diagonal_movement` other than `always`.

The keys `backend`, `solver`, `heuristic` and `max_marker_retries` are copied into each scene's metadata, but they only affect the alternative backends below.

---

## Alternative Backends (not used by the v1.0 release)

### 1. NetworkX Backend

Uses a [NetworkX](https://networkx.org/) `DiGraph` with precomputed edge weights:

- **Pros:** mature, full-featured library; fine for moderate grid sizes.
- **Cons:** higher memory use; slower on large grids.
- **Graph construction:** each cell is a node; edges join it to its 4 or 8 neighbors, with weights from the chosen [cost model](cost-models.md).
- **Algorithms:** `astar`, `astar_bi` (runs standard A\*), `dijkstra`, `dijkstra_bi`, `bellman_ford`, `unweighted`.

```python
# Internal: builds a directed graph with weighted edges
graph = build_graph_from_elevation(elevation, weight, diagonal, cost_model)
path = nx.astar_path(graph, start, end, heuristic=h)
```

### 2. Pathfinding Backend

Uses the [`python-pathfinding`](https://github.com/brean/python-pathfinding) library:

- **Pros:** built for grid pathfinding.
- **Cons:** costs are attached to cells, not to steps, so only models that depend on the destination cell work (`distance`, `elevation`, `absolute_elevation`). For `slope_uphill` and `energy`, the generator falls back to NetworkX.
- **Grid construction:** a matrix holds, for each cell, the cost of entering it.
- **Algorithms:** `astar`, `dijkstra`, `best_first`. Best-first search is greedy and does not guarantee an optimal route.

### 3. Grid Backend (Numba)

PASSAGE's own A\*/Dijkstra on arrays, compiled with Numba:

- **Pros:** fastest of the three backends, full control over the cost model, no graph library overhead.
- **Implementation:** binary-heap priority queue, inline cost computation, direct array access.
- **Algorithms:** `astar` and `dijkstra`.

The classical A\* baseline of `passage-eval` also calls this backend's A\*, with the Euclidean heuristic. Numba is a required dependency of the `passage` package.

---

## Search Algorithms

### A\* Search

A\* expands cells in order of:

$$
f(s) = g(s) + h(s)
$$

where $g(s)$ is the cost from the start to cell $s$, and $h(s)$ is the **heuristic**, an estimate of the remaining cost to the goal.

!!! success "Optimality guarantee"
    An **admissible** heuristic never overestimates the remaining cost: $h(s) \leq h^*(s)$ for every $s$, where $h^*$ is the true remaining cost. For A\* graph search that never reopens a closed cell, a **consistent** heuristic guarantees optimality: $h(u) \leq c(u,v) + h(v)$ for every step. On an 8-connected grid, the Euclidean and Chebyshev heuristics are both admissible and consistent.

### Dijkstra's Algorithm

Dijkstra is A\* with $h(s) = 0$; it explores in order of accumulated cost:

$$
f(s) = g(s)
$$

With nonnegative step costs, Dijkstra finds an optimal route when one exists. A useful heuristic can reduce the number of cells A\* needs to expand.

---

## Heuristics

The alternative backends offer three heuristics, set by the `heuristic` key. All use the cell spacing $\Delta = 30\,\text{m}$. Each step costs at least its length under every cost model, so a heuristic is admissible when it never exceeds the length of the shortest grid route to the goal.

### Euclidean Heuristic

$$
h_{\text{euclidean}}(s, g) = \Delta \cdot \sqrt{(i_g - i_s)^2 + (j_g - j_s)^2}
$$

The straight-line distance. It is admissible on 4- and 8-connected grids, and it is the tightest of the three on 8-connected grids. The octile distance (the length of the shortest 8-connected route) would be tighter still; PASSAGE does not implement it.

### Manhattan Heuristic

$$
h_{\text{manhattan}}(s, g) = \Delta \cdot \left(\lvert i_g - i_s \rvert + \lvert j_g - j_s \rvert\right)
$$

The sum of the vertical and horizontal distances. It is admissible on 4-connected grids (no diagonals).

!!! warning
    On 8-connected grids, the Manhattan heuristic **overestimates**: one diagonal step of $30\sqrt{2} \approx 42.4$ m covers a Manhattan distance of 60 m. It is then **inadmissible**, and A\* may return a route that is not optimal. Use `euclidean` or `chebyshev` instead.

### Chebyshev Heuristic

$$
h_{\text{chebyshev}}(s, g) = \Delta \cdot \max\left(\lvert i_g - i_s \rvert,\; \lvert j_g - j_s \rvert\right)
$$

The "chessboard" distance. It is admissible on 4- and 8-connected grids. It is never larger than the Euclidean distance, so it is never tighter.

---

## Heuristic Comparison

| Heuristic | Formula | Best for | On 8-connected grids |
|-----------|---------|----------|-----------|
| Euclidean | $\Delta\sqrt{\Delta i^2 + \Delta j^2}$ | 8-connected grids | Admissible; tightest of the three |
| Manhattan | $\Delta(\lvert \Delta i \rvert + \lvert \Delta j \rvert)$ | 4-connected grids | Inadmissible |
| Chebyshev | $\Delta \max(\lvert \Delta i \rvert, \lvert \Delta j \rvert)$ | 8-connected grids | Admissible; never tighter than Euclidean |

---

## Diagonal Movement

The `diagonal_movement` key accepts two policies:

| Policy | Description | Neighbor count |
|--------|-------------|:--------------:|
| `always` | Diagonal steps allowed | 8 |
| `never` | Orthogonal steps only (N/S/E/W) | 4 |

Use only these two values: other values are not checked, and the backends do not treat them the same way. The v1.0 generator requires `always`, and its reference solver applies the destination-only rule above: a diagonal step is refused only when its destination cell is an obstacle.

---

## Performance

Solve time depends on the resolution, the terrain, the obstacles and the distance between the endpoints. The [`solver_runtime_scaling.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/solver_runtime_scaling.ipynb) notebook on the Hub reports the search effort and the measured solve time of the reference solver across resolutions.

---

## Implementation Details

### Reference Solver

`src/passage/reference.py` holds:

- `solve_reference()`, the exact A\* (or Dijkstra) from start to goal used for every v1.0 label;
- `reverse_goal_field()`, a Dijkstra search on reversed steps that gives the exact cost-to-go from every reachable cell to the goal;
- `recost_legal_path()`, which recomputes the cost of a route and returns NaN for an illegal one.

### Graph Template Caching

For the **NetworkX backend**, the `_GraphTemplate` class caches the graph **topology** (nodes and edge indices) for a given grid shape and connectivity. Only the edge **weights** are recomputed for each scene, which avoids rebuilding the whole graph.

### Numba JIT Compilation

The reference solver's search and the grid backend's kernels (`_numba_astar_core`, `_numba_dijkstra_core`) are compiled with `numba.njit`, with on-disk caching. The first call in a fresh environment includes the compilation time; later calls run at native speed.

### Path Cost Computation

After a route is found, its **total cost** is the sum of its step costs:

$$
C(\text{path}) = \sum_{k=0}^{\lvert\text{path}\rvert-2} c(p_k, p_{k+1})
$$

This cost is stored in the scene metadata (`metadata.paths.<solver>.<free|obstacles>.cost`), next to the route's status, its length and the number of cells the search expanded.
