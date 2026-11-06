---
title: Cost Models
---

# Cost Models

A **cost model** (or objective) sets the cost of each step of a route on the routing graph. The generator implements **five cost models**. PASSAGE v1.0 uses three of them: `elevation`, `slope_uphill` and `energy`. The other two, `distance` and `absolute_elevation`, are options not used by the v1.0 release.

---

## Common Notation

| Symbol | Definition |
|--------|-----------|
| $u, v$ | Source cell and destination cell of one step |
| $d_{uv}$ | Step length: $30\,\text{m}$ for an orthogonal step, $30\sqrt{2}\,\text{m}$ for a diagonal step |
| $w$ | Weight of the objective (set per solver in `config/config.yaml`) |
| $h_u, h_v$ | Elevations of $u$ and $v$, in meters |
| $h_{\min}, h_{\max}$ | Lowest and highest elevation of the **crop**, in meters, obstacle cells included |

The reference solver decodes the stored elevation channel $\bar{e}$ to
meters with the global calibration range (see [Sampling Procedure](sampling.md#step-6-normalize-elevation)),
then evaluates these formulas in float64. The `elevation` and
`slope_uphill` terms divide by the crop range $h_{\max} - h_{\min}$; both
reduce to distance on a flat crop.

---

## 1. Elevation Cost

Prefers low cells of the crop. The cost grows with the **destination's** elevation, relative to the crop:

$$
c_{\mathrm{elevation}}(u,v) = d_{uv}\left(1 + w\,\frac{h_v - h_{\min}}{h_{\max} - h_{\min}}\right)
$$

This objective favours lower terrain within a crop. It does not estimate
flight risk or altitude clearance.

---

## 2. Distance Cost

Pure shortest path; elevation is ignored:

$$
c_{\mathrm{distance}}(u,v) = d_{uv}
$$

This is the baseline model. With A\* and a consistent heuristic, it gives the geometrically shortest 8-connected route. It is not used by the v1.0 release.

---

## 3. Slope-Uphill Cost

Penalizes **climbing**. A descending or flat step costs only its length:

$$
c_{\mathrm{slope}}(u,v) = d_{uv}\left(1 + w\,\frac{\max(0,\,h_v - h_u)}{h_{\max} - h_{\min}}\right)
$$

The $\max(0, h_v - h_u)$ term keeps only elevation gains. The gain is divided by the crop's elevation range and multiplies the step length.

The same edge can therefore cost more uphill than downhill.

---

## 4. Energy Cost

Adds a penalty for each meter climbed:

$$
c_{\mathrm{energy}}(u,v) = d_{uv} + w\,\max(0,\,h_v - h_u)
$$

Unlike the slope model, which multiplies the step length, the energy model **adds** a penalty per meter of gain. The gain is in meters, not normalized, and the penalty does not depend on the step length. The objective is a routing
proxy; estimating fuel or battery use requires a vehicle model.

---

## 5. Absolute-Elevation Cost

Penalizes cells far from sea level, whatever the direction of travel:

$$
c_{\mathrm{absolute\_elevation}}(u,v) = d_{uv}\left(1 + w\,\frac{\lvert h_v \rvert}{\max(\lvert h_{\min} \rvert,\, \lvert h_{\max} \rvert)}\right)
$$

The absolute elevation term is zero when every cell is at sea level. This
model is not used by the v1.0 release.

---

## Comparison Table

Every model except `distance` gives **directed** costs: reversing a step can change its cost, so the routing graph is directed.

| Cost model | Step cost $c(u,v)$ | Depends on | Penalizes | Used by v1.0 |
|-----------|---------|------------|-----------|:---:|
| `elevation` | $d_{uv}\,(1 + w\,(h_v - h_{\min})/(h_{\max} - h_{\min}))$ | Destination elevation | High cells of the crop | Yes |
| `distance` | $d_{uv}$ | Step length only | Long routes | No |
| `slope_uphill` | $d_{uv}\,(1 + w\,\max(0, h_v - h_u)/(h_{\max} - h_{\min}))$ | Climb from $u$ to $v$ | Climbing | Yes |
| `energy` | $d_{uv} + w\,\max(0, h_v - h_u)$ | Climb from $u$ to $v$ | Elevation gain | Yes |
| `absolute_elevation` | $d_{uv}\,(1 + w\,\lvert h_v \rvert / \max(\lvert h_{\min} \rvert, \lvert h_{\max} \rvert))$ | Destination elevation | Distance from sea level | No |

---

## Effect of the Weight $w$

The weight $w$ sets the **balance between distance and terrain**:

- $w = 0$: every model reduces to pure distance (shortest route).
- $w \gg 1$: terrain dominates, which can give longer routes that are cheaper under the objective.

The v1.0 weights are fixed in `config/config.yaml`. Each scene also records them, as float64, in its metadata (`metadata.solvers.<solver>.weight` and its graph specification).
The paper selected each weight by comparing its routes with distance-only
routes, using Hausdorff divergence with a penalty for following crop borders.

| Solver | Cost model | Weight $w$ |
|--------------|-----------|:---------:|
| `astar_elevation` | `elevation` | 62.4 |
| `astar_energy` | `energy` | 1125.3 |
| `astar_slope` | `slope_uphill` | 8886.2 |

---

## Visual Comparison

On the same terrain and markers, the objectives give different routes:

```text
elevation           → prefers the low cells of the crop (valleys)
slope_uphill        → avoids climbs, even at the price of a longer route
energy              → balances distance and total climb in meters
distance            → shortest 8-connected route (baseline)
absolute_elevation  → stays close to sea level
```

The [three objectives](notebooks/costmodel.md) notebook shows the three v1.0 objectives on released scenes; all notebooks are listed on the [Notebooks](notebooks/index.md) page.

---

## Implementation

The reference solver that labels v1.0 and the `passage-eval` scorer share one step-cost function: `directed_edge_cost()` in `src/passage-eval/metrics/cost.py`. It is compiled with Numba and evaluated in float64.

The alternative solver backends, not used by the v1.0 release, compute the same formulas in `src/passage/pathfinding_utils.py`:

- **NetworkX backend:** edge weights from `_compute_edge_weights()`, vectorized with NumPy.
- **Grid backend:** step costs from `_numba_compute_step_cost()`, a Numba-compiled function that dispatches on a cost-model code.

Each cost model is identified by a string:

```python
COST_MODELS = {"elevation", "absolute_elevation", "distance", "slope_uphill", "energy"}
```

The release graph specification (`GraphSpec`) rejects any other name. It also accepts `slope` as another spelling of `slope_uphill`.
