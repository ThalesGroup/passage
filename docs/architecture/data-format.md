---
title: Data Format
---

# Data Format

This page describes the data formats used throughout PASSAGE — from raw GeoTIFF tiles to the final compressed Parquet shards on Hugging Face.

---

## Tensor Format

The core data structure is a **3-channel tensor** of shape $(R, R, 3)$ where $R$ is the resolution:

| Channel | Name | Dtype | Range | Description |
|:-------:|------|-------|-------|-------------|
| 0 | Elevation | `float32` | $[0, 1]$ | Normalized elevation ($\bar{e}$) |
| 1 | Markers | `float32` | $\{-1, 0, 1\}$ | Background / start / goal |
| 2 | Obstacles | `float32` | $\{0, 1\}$ | Obstacle binary mask |

### Compression

Tensors are stored as compressed NumPy arrays:

1. Tensor is serialized to NumPy `.npy` format (in memory)
2. The bytes are compressed with **Zstandard** at level 12
3. Saved with the `.npy.zst` extension

```python
# Encoding (internal)
blob = zstd.compress(numpy_bytes, level=12)

# Decoding (user-facing)
from passage.utils import decode_tensor_blob
tensor = decode_tensor_blob(blob, shape=(256, 256, 3), dtype="float32")
```

!!! info "Compression ratio"
    Typical compression ratios are **3–8×** depending on terrain complexity. A 256×256×3 float32 tensor (~768 KB raw) typically compresses to ~100–250 KB.

---

## Metadata Format

Each sample includes a YAML metadata file with full provenance:

```yaml
tile:
  lat: 45
  lon: 5
  filename: N45E005_AVE_DSM.tif
crop:
  row_start: 1200
  col_start: 800
  height: 256
  width: 256
markers:
  start:
    row: 42
    col: 187
  end:
    row: 230
    col: 15
calibration:
  global_min: -415.0
  global_max: 8848.0
obstacles:
  enabled: true
  count: 7
  coverage_ratio: 0.23
solvers:
  - name: astar_elevation
    backend: grid
    algorithm: astar
    cost_model: elevation
    weight: 5878.0
    heuristic: euclidean
    time_free: 0.0123
    time_obstacles: 0.0156
    cost_free: 45231.7
    cost_obstacles: 52187.3
```

---

## Path CSV Format

Each solver produces two CSV files per sample — one for obstacle-free terrain, one with obstacles:

```
path_astar_elevation_free.csv
path_astar_elevation_obstacles.csv
```

Format: one point per line, `x,y` coordinates (column, row):

```csv
187,42
186,43
185,44
...
15,230
```

---

## Parquet Schema

During export, samples are packaged into **Apache Parquet** files with Hugging Face `Features` typing:

```
Root
├── tensor              bytes           # zstd-compressed (R,R,3) numpy array
├── metadata
│   ├── tile
│   │   ├── lat         int32
│   │   ├── lon         int32
│   │   └── filename    string
│   ├── crop
│   │   ├── row_start   int32
│   │   ├── col_start   int32
│   │   ├── height      int32
│   │   └── width       int32
│   ├── markers
│   │   ├── start       {row: int32, col: int32}
│   │   └── end         {row: int32, col: int32}
│   ├── obstacles
│   │   ├── enabled     bool
│   │   ├── count       int32
│   │   └── coverage    float32
│   └── solvers[]
│       ├── name        string
│       ├── backend     string
│       ├── algorithm   string
│       ├── cost_model  string
│       ├── weight      float64
│       ├── heuristic   string
│       ├── time_free   float64
│       └── time_obs    float64
├── path_<solver>_free       string     # CSV-encoded path
├── path_<solver>_obstacles  string     # CSV-encoded path
└── image                    bytes      # (Optional) PNG image
```

### Parquet Configuration

| Resolution | Chunk size (rows/file) | Compression |
|-----------|:----------------------:|:-----------:|
| 64 | 2,048 | zstd |
| 128 | 512 | zstd |
| 256 | 128 | zstd |
| 512 | 64 | zstd |
| 1024 | 32 | zstd |
| 2048 | 8 | zstd |
| 4096 | 4 | zstd |

Chunk sizes decrease with resolution to keep individual Parquet files at manageable sizes.

---

## File Naming Conventions

### Sample directories

```
outputs/dataset/{resolution}/{split}/{sample_id:06d}/
```

Example: `outputs/dataset/256/train/000042/`

### Sample files

| File | Description |
|------|-------------|
| `tensor.npy.zst` | Compressed 3-channel tensor |
| `metadata.yaml` | Full sample metadata |
| `sample.png` | Visualization image (optional) |
| `path_{solver}_free.csv` | Path on free terrain |
| `path_{solver}_obstacles.csv` | Path with obstacles |

### Export structure

```
outputs/export/{resolution}x{resolution}/
├── {split}/
│   ├── 0000000000.parquet
│   ├── 0000000001.parquet
│   └── ...
└── dataset.ipynb   # Executed per-resolution notebook
```

---

## HuggingFace Dataset Configs

Each resolution has two HF configs:

| Config | Content |
|--------|---------|
| `{R}x{R}` | Full dataset at resolution $R$ |
| `{R}x{R}_sample` | Subset of 64 samples for quick exploration |

Default config: `1024x1024_sample`

```python
# Load full dataset at 256×256
ds = load_dataset("geoffrey-g-delhomme/passage", name="256x256")

# Load quick sample
ds = load_dataset("geoffrey-g-delhomme/passage", name="256x256_sample")
```
