---
title: Data Format
---

# Data Format

This page describes the data formats used throughout PASSAGE, from the files the generator writes to the compressed Parquet shards on Hugging Face. The [dataset card](../dataset/index.md) is the reference for the released columns.

---

## Tensor Format

Each scene is stored as a **3-channel tensor** of shape $(R, R, 3)$, where $R$ is the resolution:

| Channel | Name | Dtype | Range | Description |
|:-------:|------|-------|-------|-------------|
| 0 | Elevation | `float32` | $[0, 1]$ | Elevation normalized with the global calibration range ($\bar{e}$) |
| 1 | Markers | `float32` | $\{-1, 0, 1\}$ | Background / start / goal |
| 2 | Obstacles | `float32` | $\{0, 1\}$ | Obstacle mask (1 = blocked cell) |

### Compression

A tensor blob holds the **raw bytes** of the array, compressed with Zstandard:

1. The array is laid out in C order (row-major) as little-endian float32 values.
2. The bytes are compressed with **Zstandard** at level 12.
3. The generator saves the blob with the `.npy.zst` suffix; in Parquet it is the `tensor` column.

The blob has **no NumPy `.npy` header**, so `np.load` cannot read it. Decode it with `np.frombuffer` and reshape it:

```python
import numpy as np
import zstandard

# Encoding (what passage.utils.encode_tensor_blob does)
blob = zstandard.ZstdCompressor(level=12).compress(np.ascontiguousarray(tensor).tobytes())

# Decoding
x = np.frombuffer(zstandard.ZstdDecompressor().decompress(blob), dtype="<f4").reshape(R, R, 3)

# The same with the package helper
from passage.utils import decode_tensor_blob
x = decode_tensor_blob(blob, shape=(R, R, 3))
```

To get elevations in meters, use the calibration range stored in the metadata: `x[..., 0] * (max - min) + min`. PASSAGE v1.0 uses `min = -430.0` and `max = 8767.0`.

---

## Metadata Format

Each scene has a YAML metadata file. The skeleton below shows its keys; `<…>` marks a value that changes from scene to scene.

```yaml
index: <int>                     # scene index; the Parquet `id` is this index, zero-padded to 10 digits
resolution: <int>
split: <train | calibration | validation | test>
dataset_version: <…>             # the value of generation.dataset_version in config/config.yaml
units: metres
calibration:
  min: -430.0                    # global calibration range, in meters
  max: 8767.0
markers:
  start_point: {i: <row>, j: <col>}
  end_point: {i: <row>, j: <col>}
tile:
  source: {lat: <int>, lon: <int>}    # center tile, named by its south-west corner
  used: [<source window>, …]          # one window per tile the crop overlaps
  overlapping: [<source window>, …]
crop:
  top_left: {lat: <float>, lon: <float>}
  bottom_right: {lat: <float>, lon: <float>}
  center: {lat: <float>, lon: <float>}
obstacles:
  enabled: true
  ratio_target: <float>          # target coverage t
  ratio_actual: <float>          # final coverage, at most t
provenance:
  generation_seed: 42
  sample_seed: <int>
  provenance_id: <sha256>
  calibration_sha256: <sha256>
  source_windows: [<source window>, …]
solvers:
  astar_elevation:
    name: astar_elevation
    cost_model: elevation
    weight: 62.4
    # … the other keys of this solvers entry in config/config.yaml …
    graph_spec:
      connectivity: 8
      diagonal_policy: destination_only
      obstacle_policy: blocked
      pixel_size_m: 30.0
      coordinate_order: row_col
      cost_model: elevation
      cost_weight: 62.4
      rtol: 1.0e-07
      atol: 1.0e-06
    reference_engine: passage.reference.solve_reference
    arithmetic_precision: float64
  astar_energy: {…}              # cost_model: energy, weight: 1125.3
  astar_slope: {…}               # cost_model: slope_uphill, weight: 8886.2
paths:
  astar_elevation:
    free:
      status: <ok | infeasible>  # ok for every released variant
      cost: <float or null>
      length: <int>              # number of cells in the route
      expansions: <int>          # cells expanded by the reference search
      considered_area: <int>
      reference_status: <exact | no_path>  # exact for every released variant
      graph_id: <sha256>
      crosses_obstacles: false
    obstacles: {…}               # same keys, obstacles blocked
  astar_energy: {…}
  astar_slope: {…}
```

A **source window** records the tile (`lat`, `lon`, file `name`, `sha256`), the rows and columns read from it, and where they go in the crop.

---

## Path CSV Format

Each solver produces two route files per scene: one on free terrain and one with obstacles blocked. Each line holds one cell as a `row,col` pair, from the start to the goal, with no header:

```csv
44,160
45,159
46,158
...
```

Every released variant has the status `ok`, so every file holds a route. The generator writes the files to `paths/<solver>/free/…` and `paths/<solver>/obstacles/…` (see [File Naming Conventions](#file-naming-conventions)).

---

## Parquet Schema

During export, scenes are packaged into **Apache Parquet** shards with Hugging Face `Features` typing. The columns are those of the dataset card's [Columns](../dataset/index.md#columns) table:

| Column | Type | Content |
| --- | --- | --- |
| `id` | string | Scene index, zero-padded; unique within a configuration and split |
| `tensor` | bytes | Zstd-compressed float32 array $(R, R, 3)$: normalized elevation, markers, obstacles |
| `metadata` | struct | Split, calibration, source tiles, crop, markers, obstacles, seeds, solver parameters, and per-variant status and cost |
| `metadata_json` | string | Complete metadata as JSON text |
| `scene_receipt` | string | JSON receipt: seeds, source windows and hashes, attempts, terminal records |
| `path_<solver>_free` | list of [row, col] | Reference route on free terrain |
| `path_<solver>_obstacles` | list of [row, col] | Reference route with obstacles blocked |
| `image_<solver>` | image | Rendered preview, `_samples` configurations only |

`<solver>` is `astar_elevation`, `astar_energy` or `astar_slope` (the `slope_uphill` objective). Every released variant has the status `ok` in `metadata.paths.<solver>.<free|obstacles>.status`, so no route is an empty list.

The `metadata` struct has a fixed schema. Fields that the v1.0 generator does not write are null there; `metadata_json` holds the complete metadata.

### Parquet Configuration

Shards are compressed with zstd at level 12 (`export.compression_level`). The maximum number of rows per shard is set by `export.chunk_size` in `config/config.yaml`:

| Resolution | Rows per shard (max) |
|-----------|:----------------------:|
| 64 | 20,000 |
| 128 | 10,000 |
| 256 | 2,500 |
| 512 | 1,500 |
| 1024 | 250 |
| 2048 | 75 |
| 4096 | 25 |

Shard sizes decrease with resolution to keep each Parquet file at a manageable size.

---

## File Naming Conventions

### Generator output

With the shipped configuration, `passage generate` writes one directory per resolution and split under its output root (`--outputs-dirpath`, shown as `<root>`). `<R>` is the resolution (for example `256`), `<index>` is the scene index on 10 digits, and `<lat>/<lon>` names the center tile, for example `N061/E081`:

```text
<root>/
├── generation_manifest.json               # provenance of the run
├── generation-input-snapshot/             # copies of the configuration, calibration and hold-out mask
├── generation-source-snapshot/            # copy of the generating source code
└── <R>/
    ├── roster.json                        # frozen split of every scene index
    ├── summary.json                       # outcome counts of the last run
    └── <split>/
        ├── tensors/<lat>/<lon>/<index>.npy.zst
        ├── metadata/<lat>/<lon>/<index>.yaml
        ├── paths/<solver>/<free|obstacles>/<lat>/<lon>/<index>.csv
        └── receipts/<index>.json          # written last; marks the scene as complete
```

| File | Description |
|------|-------------|
| `<index>.npy.zst` | Compressed 3-channel tensor |
| `<index>.yaml` | Full scene metadata |
| `<index>.csv` | One reference route, one `row,col` pair per line |
| `<index>.json` | Scene receipt: seeds, attempts, source windows and the hash of every file of the scene |

The generator writes no preview images. The Parquet export renders them for the `_samples` configurations.

### Export structure

The main entries of the exported tree, as published on the Hub:

```text
<R>x<R>/<split>/data-*.parquet           core scenes, no images (7 configurations)
<R>x<R>_samples/<split>/data-*.parquet   64 preview rows per split, with images (7 configurations)
<R>x<R>/*.ipynb                          executed analysis notebooks for that resolution
calibration/calibrate.json               elevation range and authenticated tile inventory
release-receipts/                        generation, verification and publication receipts
```

The published shards were written by `passage regenerate`, which generates and exports the scenes in batches. Their names follow `data-<R>-<first>-<last>-<shard>.parquet`, where `<first>` and `<last>` are the first and last scene indices of the batch (10 digits) and `<shard>` numbers the shards of that batch (5 digits). `passage compress` instead numbers the shards of each split from zero: `data-<shard>.parquet`.

---

## Hugging Face Dataset Configs

Each resolution has two Hugging Face configurations:

| Config | Content |
|--------|---------|
| `<R>x<R>` | All scenes at resolution $R$, without images |
| `<R>x<R>_samples` | 64 preview rows per split, with rendered images; they repeat scenes of the core configuration |

The default configuration is `1024x1024_samples`. Core configurations are large (57 GB at 256×256, 533 GB at 4096×4096), so pin the verified revision and stream them:

```python
from datasets import load_dataset

REVISION = "2ab44d899652c7fb99c425488e84a7d565963547"  # verified PASSAGE v1.0 upload

# Preview rows: small, 64 per split, with rendered images.
rows = load_dataset("thalesgroup/passage", "256x256_samples",
                    split="validation", revision=REVISION)

# Core configuration: stream it instead of downloading it whole.
core = load_dataset("thalesgroup/passage", "256x256", split="train",
                    streaming=True, revision=REVISION)
for row in core.take(4):
    print(row["id"], row["metadata"]["split"])
```
