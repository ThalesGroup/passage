---
title: Cost Models
icon: material/scale-balance
---

# Cost Models

PASSAGE implements **five distinct cost models**, each defining a different edge-weight function for the pathfinding graph. These models capture different real-world navigation priorities — from minimizing distance to avoiding uphill climbs.

---

## Common Notation

| Symbol | Definition |
|--------|-----------|
| $d$ | Euclidean distance between adjacent pixels ($30\,\text{m}$ cardinal, $30\sqrt{2}\,\text{m}$ diagonal) |
| $\alpha$ | Cost model weight (configurable per solver) |
| $e(i,j)$ | Raw elevation at pixel $(i,j)$ |
| $\bar{e}(i,j)$ | Normalized elevation $\in [0,1]$ |
| $\|\bar{e}\|(i,j)$ | Absolute normalized elevation |
| $\Delta e$ | Elevation difference $e_{\text{dst}} - e_{\text{src}}$ |
| $\Delta e_{\text{range}}$ | Global elevation range $e_{\max} - e_{\min}$ |

---

## 1. Elevation Cost

Prefer low-elevation pixels. The cost increases proportionally with the **destination's normalized elevation**:

$$
w_{\text{elevation}} = d \cdot \left(1 + \alpha \cdot \bar{e}_{\text{dst}}\right)
$$

!!! tip "When to use"
    Useful when operating in mountainous terrain where **higher altitude = higher risk** (weather, oxygen, turbulence). A helicopter might prefer lower valleys.

---

## 2. Distance Cost

Pure shortest-path — ignores elevation entirely:

$$
w_{\text{distance}} = d
$$

This is the baseline cost model. Combined with A\* and a consistent heuristic, it produces geometrically shortest paths.

---

## 3. Slope-Uphill Cost

Penalizes **uphill movement** while treating downhill as free (cost = 0 extra). Models energy expenditure for ground vehicles or the difficulty of gaining altitude:

$$
w_{\text{slope}} = d \cdot \left(1 + \alpha \cdot \frac{\max(0,\; \Delta e)}{\Delta e_{\text{range}}}\right)
$$

The $\max(0, \Delta e)$ term ensures only positive elevation changes (climbing) contribute to the cost. Descending is weighted at the base distance.

!!! example "Helicopter analogy"
    A HEMS helicopter climbing from a valley floor to a mountain ridge expends significantly more energy than descending. This model captures that asymmetry.

---

## 4. Energy Cost

An additive energy penalty for climbing:

$$
w_{\text{energy}} = d + \alpha \cdot \max(0,\; \Delta e)
$$

Unlike the slope model (which is multiplicative), the energy model adds a fixed penalty per meter of elevation gain, independent of horizontal distance.

---

## 5. Absolute Cost

Penalizes pixels with **high absolute elevation** regardless of direction of travel:

$$
w_{\text{absolute}} = d \cdot \left(1 + \alpha \cdot |\bar{e}_{\text{dst}}|\right)
$$

This forces paths to stick to regions near sea level, avoiding both mountain peaks and deep depressions.

---

## Comparison Table

| Cost Model | Formula | Directional? | Penalizes |
|-----------|---------|:------------:|-----------|
| **Elevation** | $d \cdot (1 + \alpha \bar{e}_{\text{dst}})$ | No | High altitude |
| **Distance** | $d$ | No | Long paths only |
| **Slope-uphill** | $d \cdot (1 + \alpha \cdot \max(0,\Delta e)/\Delta e_{\text{range}})$ | Yes (uphill) | Climbing |
| **Energy** | $d + \alpha \cdot \max(0, \Delta e)$ | Yes (uphill) | Elevation gain |
| **Absolute** | $d \cdot (1 + \alpha \cdot |\bar{e}_{\text{dst}}|)$ | No | Extreme elevations |

---

## Effect of the Weight $\alpha$

The weight parameter $\alpha$ controls the **balance between distance and terrain awareness**:

- $\alpha = 0$: all models reduce to pure distance (shortest path)
- $\alpha \gg 1$: terrain features dominate, potentially producing longer but terrain-optimal paths
- The default weights are **calibrated** to produce visually distinct paths across cost models

| Solver Config | Cost Model | Default $\alpha$ |
|--------------|-----------|:---------:|
| `astar_elevation` | Elevation | 5,878 |
| `astar_energy` | Energy | 3,888 |
| `astar_slope` | Slope-uphill | 70,170 |

!!! info "Why are weights so large?"
    Since normalized elevation values are in $[0, 1]$ and pixel distances are $30\,\text{m}$, large weights are needed to make the terrain component comparable to the distance component. The values were empirically tuned to produce meaningfully different paths.

---

## Visual Comparison

On the same terrain and markers, different cost models produce distinctly different paths:

```
Elevation model → avoids mountain peaks, prefers valleys
Slope model     → avoids steep climbs, may take longer routes
Energy model    → minimizes total elevation gain
Distance model  → straight-line shortest path (baseline)
Absolute model  → stays near sea level
```

See the [**Cost Model notebook**](notebooks.md) for interactive visualizations.

---

## Implementation

Cost models are implemented in `src/passage/pathfinding_utils.py`:

- **NetworkX backend:** edge weights computed via `_compute_edge_weights()` using vectorized NumPy operations
- **Grid/Numba backend:** step costs computed via `_numba_compute_step_cost()` — a JIT-compiled function dispatching on cost model ID

Each cost model is identified by a string constant:

```python
COST_MODELS = ["elevation", "distance", "slope_uphill", "energy", "absolute"]
```
