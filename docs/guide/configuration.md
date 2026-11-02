---
title: Configuration Reference
---

# Configuration Reference

PASSAGE is configured through `config/config.yaml`. The shipped file is the
configuration that built PASSAGE v1.0. This page documents every key, in the
order of the generation workflow.

---

## How `generate` Reads This File

The key `generation.dataset_version` selects the generator that built
PASSAGE v1.0 (the *release generator*). Keep the value shipped in
`config/config.yaml`: another value stops `passage generate` with an error, and
without the key `passage generate` runs another generation path, which the v1.0
release does not use.

The release generator also checks three rules before it starts:

- exactly three solvers, with unique names and the objectives `elevation`,
  `energy` and `slope_uphill`;
- `diagonal_movement: always` for every solver;
- `obstacles.impassable: true`.

Some keys stay in the file but are **not read by the v1.0 release generator**:

| Key | Why it is not used |
|-----|--------------------|
| `splits.enabled` | The release generator always uses `splits.ratios` and `splits.seed` |
| `generation.preview_images.*` | Preview rows come from `export.sample.num_samples`; `passage compress` renders their images |
| `solvers[].backend`, `solver`, `heuristic`, `max_marker_retries` | Copied into each scene's metadata, but the labels come from the exact reference A\* solver |
| `obstacles.obstacle_penalty` | Applies only when `impassable: false`, which the release generator refuses |

`solvers[].diagonal_movement` is read only to check that it is `always`.

---

## Resolutions

```yaml
resolutions:
  - 64
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
```

List of crop sizes to generate. A resolution $R$ produces scenes of
$R \times R$ cells. With 30 m cells, the nominal side is $R \times 30$ m; the
actual ground spacing varies with latitude.

| Resolution | Nominal side |
|-----------|--------------|
| 64 | 1.92 km |
| 128 | 3.84 km |
| 256 | 7.68 km |
| 512 | 15.36 km |
| 1024 | 30.72 km |
| 2048 | 61.44 km |
| 4096 | 122.88 km |

A source tile holds 3600 × 3600 cells. A crop is cut wherever it falls, so it
can span two tiles per axis; a 4096 crop, wider than one tile, can span three.

---

## Sample Counts

```yaml
num_samples:
  64: 1000000
  128: 1000000
  256: 1000000
  512: 1000000
  1024: 250000
  2048: 100000
  4096: 50000
```

Number of scenes to generate per resolution: 4,400,000 in total. The value can
be a single integer (applied to every resolution) or a per-resolution mapping.

!!! tip
    CLI `--num-samples` overrides the config value for all selected resolutions.

---

## Splits

```yaml
splits:
  enabled: true
  ratios:
    train: 0.9
    calibration: 0.02
    validation: 0.04
    test: 0.04
  seed: 42
```

| Key | Description |
|-----|-------------|
| `ratios` | Share of scenes per split, rounded to whole scenes. Also the share of 5° blocks among `train`, `calibration` and `validation` |
| `seed` | Seeds both the scene-to-split assignment and the geographic block assignment |
| `enabled` | Not read by the v1.0 release generator |

See [Splits](generate.md#splits) for the geographic rules.

---

## Hold-Out Region

```yaml
hold_out:
  enabled: true
  mask_path: assets/hold_out_mask.png
```

The mask is a 181 × 361 pixel PNG with one pixel per degree: row 0 is latitude
+90°, column 0 is longitude −180°. Non-zero pixels mark hold-out tiles. Tiles
inside the mask belong to the `test` split only, and every `test` scene comes
from them. The shipped mask covers South America.

---

## Generation

```yaml
generation:
  dataset_version: ...   # keep the value shipped in config/config.yaml
  seed: 42
  max_scene_attempts: 100
  max_reference_expansions: null
  geographic_policy:
    block_degrees: 5
    buffer_tiles: 1
  preview_images:
    enabled: true
    resolutions: [64, 128, 256, 512, 1024, 2048, 4096]
    num_samples_per_split: 64
```

| Key | Description |
|-----|-------------|
| `dataset_version` | Selects the release generator. Keep the shipped value |
| `seed` | Seeds each scene: `blake2b(seed, resolution, index, split)` sets its tile, crop, markers and obstacles |
| `max_scene_attempts` | Crops tried for one scene before it is recorded as `rejected` |
| `max_reference_expansions` | Search budget of the reference solver. `null` means no budget; with a budget, an unfinished search is recorded as `failed` |
| `geographic_policy.block_degrees` | Side of the geographic blocks, in degrees. It must divide 180 and exceed twice `buffer_tiles` |
| `geographic_policy.buffer_tiles` | Width of the guard ring around a crop's tiles, in tiles |
| `preview_images.*` | Not read by the v1.0 release generator |

---

## Solvers

```yaml
solvers:
  - name: astar_elevation
    backend: grid
    diagonal_movement: always
    max_marker_retries: 10
    solver: astar
    cost_model: elevation
    weight: 62.4
    heuristic: euclidean
  - name: astar_energy
    backend: grid
    diagonal_movement: always
    max_marker_retries: 10
    solver: astar
    cost_model: energy
    weight: 1125.3
    heuristic: euclidean
  - name: astar_slope
    backend: grid
    diagonal_movement: always
    max_marker_retries: 10
    solver: astar
    cost_model: slope_uphill
    weight: 8886.2
    heuristic: euclidean
```

Each entry of the list defines one objective:

| Key | Options | Description |
|-----|---------|-------------|
| `name` | text | Solver name, used in column names (`path_<name>_free`, `path_<name>_obstacles`) and folder names |
| `cost_model` | `elevation`, `slope_uphill`, `energy` | Edge cost function. `distance` and `absolute_elevation` also exist but are not used by the v1.0 release |
| `weight` | float | Weight $w$ of the objective |
| `diagonal_movement` | `always` | Eight-connected grid; a diagonal step is checked against its destination cell only |
| `backend`, `solver`, `heuristic`, `max_marker_retries` | | Recorded in the metadata; not used by the v1.0 release generator |

Every label comes from an exact A\* reference that computes in float64. See
[Cost Models](../generation/cost-models.md) for the formulas and
[Solvers & Heuristics](../generation/solvers.md) for the solvers.

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
      min: 1.0
      max: 80.0
      distribution:
        type: log_uniform
  generation:
    max_attempts: 50
    candidates_per_mask: 200
  impassable: true
  obstacle_penalty: 1e8
```

| Key | Description |
|-----|-------------|
| `enabled` | Add obstacles to each scene |
| `ratio_max` | Upper bound of the obstacle coverage. Each mask draws its target coverage uniformly between 0 and `ratio_max` |
| `shape.size.min_ratio` / `max_ratio` | Full width and height of a superellipse, as a fraction of the crop side; its semi-axes are half of these |
| `shape.n.min` / `max` | Exponent range ($n=1$: diamond, $n=2$: ellipse, large $n$: close to a rectangle) |
| `shape.n.distribution.type` | Distribution of the exponent: `log_uniform` (shipped), `uniform`, `beta` (parameters `a`, `b`) or `exponential` (parameter `lambda`) |
| `generation.max_attempts` | Masks tried before giving up; an empty mask is then used |
| `generation.candidates_per_mask` | Largest number of superellipses tried for one mask |
| `impassable` | `true`: obstacle cells are blocked, and a crop whose obstacles disconnect the start from the goal is rejected. The release generator requires `true` |
| `obstacle_penalty` | Elevation penalty added to obstacle cells when `impassable: false`; not used by the v1.0 release |

Obstacles never cover the start or the goal. See
[Obstacle Generation](../generation/obstacles.md) for the superellipse math.

---

## Export

```yaml
export:
  default: 1024x1024_samples
  sample:
    num_samples: 64
  compression_level: 12
  chunk_size:
    64: 20000
    128: 10000
    256: 2500
    512: 1500
    1024: 250
    2048: 75
    4096: 25
```

| Key | Description |
|-----|-------------|
| `default` | Default Hugging Face configuration named in the dataset card |
| `sample.num_samples` | Rows per split in each `<R>x<R>_samples` configuration: the lowest scene IDs of the split, with rendered preview images |
| `compression_level` | Zstandard level of the Parquet files written by `passage compress` |
| `chunk_size` | Rows per Parquet shard, per resolution |

---

## Full Example

??? example "Complete `config/config.yaml` (comments removed)"

    ```yaml
    resolutions: [64, 128, 256, 512, 1024, 2048, 4096]

    export:
      default: 1024x1024_samples
      sample:
        num_samples: 64
      compression_level: 12
      chunk_size:
        64: 20000
        128: 10000
        256: 2500
        512: 1500
        1024: 250
        2048: 75
        4096: 25

    num_samples:
      64: 1000000
      128: 1000000
      256: 1000000
      512: 1000000
      1024: 250000
      2048: 100000
      4096: 50000

    splits:
      enabled: true
      ratios:
        train: 0.9
        calibration: 0.02
        validation: 0.04
        test: 0.04
      seed: 42

    generation:
      dataset_version: ...   # keep the value shipped in config/config.yaml
      seed: 42
      max_scene_attempts: 100
      max_reference_expansions: null
      geographic_policy:
        block_degrees: 5
        buffer_tiles: 1
      preview_images:
        enabled: true
        resolutions: [64, 128, 256, 512, 1024, 2048, 4096]
        num_samples_per_split: 64

    hold_out:
      enabled: true
      mask_path: assets/hold_out_mask.png

    solvers:
      - name: astar_elevation
        backend: grid
        diagonal_movement: always
        max_marker_retries: 10
        solver: astar
        cost_model: elevation
        weight: 62.4
        heuristic: euclidean
      - name: astar_energy
        backend: grid
        diagonal_movement: always
        max_marker_retries: 10
        solver: astar
        cost_model: energy
        weight: 1125.3
        heuristic: euclidean
      - name: astar_slope
        backend: grid
        diagonal_movement: always
        max_marker_retries: 10
        solver: astar
        cost_model: slope_uphill
        weight: 8886.2
        heuristic: euclidean

    obstacles:
      enabled: true
      ratio_max: 0.30
      shape:
        size:
          min_ratio: 0.02
          max_ratio: 0.25
        n:
          min: 1.0
          max: 80.0
          distribution:
            type: log_uniform
      generation:
        max_attempts: 50
        candidates_per_mask: 200
      impassable: true
      obstacle_penalty: 1e8
    ```
