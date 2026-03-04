---
title: Configuration Reference
icon: material/cog
---

# Configuration Reference

PASSAGE is configured through `config/config.yaml`. This page documents every option in detail.

---

## Resolutions

```yaml
resolutions: [64, 128, 256, 512, 1024, 2048, 4096]
```

List of pixel resolutions to generate. Each resolution $R$ produces samples of shape $(R \times R)$, covering a geographic area of approximately $(R \times 30)$ meters per side.

| Resolution | Coverage | Tile grid |
|-----------|----------|-----------|
| 64 | ~1.9 km | 1×1 |
| 128 | ~3.8 km | 1×1 |
| 256 | ~7.7 km | 1×1 |
| 512 | ~15.4 km | 1×1 |
| 1024 | ~30.7 km | 1×1 |
| 2048 | ~61.4 km | 1×1 |
| 4096 | ~122.9 km | 2×2 |

---

## Sample Counts

```yaml
num_samples:
  64: 10000
  128: 2500
  256: 1000
  512: 500
  1024: 100
  2048: 20
  4096: 10
```

Number of samples to generate per resolution. Can be a single integer (applied to all) or a per-resolution mapping.

!!! tip
    CLI `--num-samples` overrides the config value for all resolutions.

---

## Splits

```yaml
splits:
  train: 0.90
  calibration: 0.02
  validation: 0.04
  test: 0.04
seed: 42
```

Deterministic split assignment using the configured seed. Ratios must sum to 1.0.

---

## Hold-Out Region

```yaml
hold_out:
  enabled: true
  mask: assets/hold_out_mask.png
```

A geographic hold-out mask (181×361 pixel binary image) that excludes certain regions (e.g., South America) from training data. Tiles in held-out regions are reserved for test/validation splits only.

---

## Solvers

```yaml
solvers:
  astar_elevation:
    backend: grid
    algorithm: astar
    cost_model: elevation
    weight: 5878.0
    heuristic: euclidean
    diagonal_movement: always

  astar_energy:
    backend: grid
    algorithm: astar
    cost_model: energy
    weight: 3888.0
    heuristic: euclidean
    diagonal_movement: always

  astar_slope:
    backend: grid
    algorithm: astar
    cost_model: slope_uphill
    weight: 70170.0
    heuristic: euclidean
    diagonal_movement: always
```

Each solver entry defines:

| Key | Options | Description |
|-----|---------|-------------|
| `backend` | `grid`, `networkx`, `pathfinding` | Solver implementation |
| `algorithm` | `astar`, `dijkstra` | Search algorithm |
| `cost_model` | `elevation`, `distance`, `slope_uphill`, `energy`, `absolute` | Edge weight function |
| `weight` | float | Cost model scaling factor $\alpha$ |
| `heuristic` | `euclidean`, `manhattan`, `chebyshev` | A\* heuristic |
| `diagonal_movement` | `always`, `never`, `if_at_most_one_obstacle` | Diagonal connectivity |

See [Cost Models](../generation/cost-models.md) for the mathematical definitions.

---

## Obstacles

```yaml
obstacles:
  enabled: true
  ratio_max: 0.30
  shape:
    size:
      min_ratio: 0.02
      max_ratio: 0.25
    n:
      distribution: log_uniform
      min: 1.0
      max: 80.0
  max_attempts: 50
  candidates_per_mask: 200
  obstacle_penalty: 1.0e+08
```

| Key | Description |
|-----|-------------|
| `enabled` | Toggle obstacle generation |
| `ratio_max` | Maximum fraction of the grid covered by obstacles |
| `shape.size.min_ratio` / `max_ratio` | Obstacle semi-axis as fraction of grid size |
| `shape.n.distribution` | Distribution for superellipse exponent (`uniform`, `log_uniform`, `beta`, `exponential`) |
| `shape.n.min` / `max` | Exponent range ($n=1$: diamond, $n=2$: ellipse, $n \gg 2$: rectangle) |
| `max_attempts` | Max placement attempts before giving up |
| `candidates_per_mask` | Candidate positions per obstacle |
| `obstacle_penalty` | Elevation penalty added to blocked cells ($10^8$) |

See [Obstacle Generation](../generation/obstacles.md) for the superellipse math.

---

## Export

```yaml
export:
  default: 1024x1024_sample
  sample:
    num_samples: 64
  compression_level: 12
  chunk_sizes:
    64: 2048
    128: 512
    256: 128
    512: 64
    1024: 32
    2048: 8
    4096: 4
```

| Key | Description |
|-----|-------------|
| `default` | Default HF dataset config name |
| `sample.num_samples` | Number of samples in `_sample` configs |
| `compression_level` | Zstandard compression level (1–22) |
| `chunk_sizes` | Rows per Parquet file, per resolution |

---

## Full Example

??? example "Complete `config/config.yaml`"

    ```yaml
    resolutions: [64, 128, 256, 512, 1024, 2048, 4096]

    num_samples:
      64: 10000
      128: 2500
      256: 1000
      512: 500
      1024: 100
      2048: 20
      4096: 10

    splits:
      train: 0.90
      calibration: 0.02
      validation: 0.04
      test: 0.04
    seed: 42

    hold_out:
      enabled: true
      mask: assets/hold_out_mask.png

    solvers:
      astar_elevation:
        backend: grid
        algorithm: astar
        cost_model: elevation
        weight: 5878.0
        heuristic: euclidean
        diagonal_movement: always
      astar_energy:
        backend: grid
        algorithm: astar
        cost_model: energy
        weight: 3888.0
        heuristic: euclidean
        diagonal_movement: always
      astar_slope:
        backend: grid
        algorithm: astar
        cost_model: slope_uphill
        weight: 70170.0
        heuristic: euclidean
        diagonal_movement: always

    obstacles:
      enabled: true
      ratio_max: 0.30
      shape:
        size:
          min_ratio: 0.02
          max_ratio: 0.25
        n:
          distribution: log_uniform
          min: 1.0
          max: 80.0
      max_attempts: 50
      candidates_per_mask: 200
      obstacle_penalty: 1.0e+08

    export:
      default: 1024x1024_sample
      sample:
        num_samples: 64
      compression_level: 12
      chunk_sizes:
        64: 2048
        128: 512
        256: 128
        512: 64
        1024: 32
        2048: 8
        4096: 4
    ```
