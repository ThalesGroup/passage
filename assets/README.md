<!-- PASSAGE Hugging Face Dataset Card -->

<p align="center">
  <img src="assets/banner.png" alt="PASSAGE — Path Solving Dataset & Generator" width="100%"/>
</p>

<h1 align="center">🌍 PASSAGE — Real-Terrain Benchmark & Generator</h1>

<p align="center">
  <strong>Trustworthiness-oriented real-terrain benchmark and generation pipeline for pathfinding and surrogate modeling.</strong>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://github.com/ThalesGroup/passage/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/ci.yml?label=CI&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="https://github.com/ThalesGroup/passage/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white" alt="License"></a>
  <a href="https://huggingface.co/datasets/geoffrey-g-delhomme/passage"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/package-uv-DE5FE9?logo=uv&logoColor=white" alt="uv"></a>
</p>

<p align="center">
  <a href="https://thalesgroup.github.io/passage/">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/geoffrey-g-delhomme/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage">💻 GitHub</a> •
  <a href="#-citation">📝 Cite</a>
</p>

---

## 🚁 Why PASSAGE?

> *Every year, Helicopter Emergency Medical Services (HEMS) crews across Europe perform over 300,000 rescue missions — racing against time through mountain passes, across valleys, and over ridgelines to reach patients in cardiac arrest or severe trauma. For every minute of delay, survival rates drop by 7–10 %. The path the helicopter takes matters — yet pathfinding algorithms are still benchmarked on flat synthetic grids.*

**PASSAGE** was created to change that. Built from **real-world satellite elevation data** (JAXA ALOS AW3D30 at 30 m/pixel), it provides the research community with large-scale, multi-resolution pathfinding benchmarks that capture the true complexity of terrain navigation.

We open-source this toolkit to make terrain-routing methods easier to compare, reproduce, and audit under a shared benchmark surface.

---

## ✨ Highlights

<table>
<tr>
  <td align="center" width="25%">🌐<br><strong>Multi-Resolution</strong><br><small>64×64 → 4096×4096 px</small></td>
  <td align="center" width="25%">🏔️<br><strong>Real Elevation</strong><br><small>JAXA ALOS AW3D30 DSM</small></td>
  <td align="center" width="25%">🔷<br><strong>Rich Obstacles</strong><br><small>Superellipse shapes</small></td>
  <td align="center" width="25%">⚡<br><strong>5 Cost Models</strong><br><small>Elevation, slope, energy…</small></td>
</tr>
<tr>
  <td align="center">🧮<br><strong>3 Solver Backends</strong><br><small>NetworkX, pathfinding, Numba</small></td>
  <td align="center">🤗<br><strong>HuggingFace-Ready</strong><br><small>Parquet + one-command upload</small></td>
  <td align="center">🔁<br><strong>Fully Resumable</strong><br><small>No wasted compute</small></td>
  <td align="center">🛡️<br><strong>Certification-Aware</strong><br><small>ARP 6983 / ED-324</small></td>
</tr>
</table>

<table>
<tr>
  <td align="center"><strong>14,130</strong><br><small>Default Samples</small></td>
  <td align="center"><strong>7</strong><br><small>Resolution Levels</small></td>
  <td align="center"><strong>5</strong><br><small>Cost Models</small></td>
  <td align="center"><strong>3</strong><br><small>Solver Backends</small></td>
  <td align="center"><strong>30 m/px</strong><br><small>Spatial Resolution</small></td>
</tr>
</table>

---

## 🗺️ Data Overview

### Representative Sample

<p align="center">
  <img src="assets/sample.png" alt="Representative PASSAGE sample" width="720"/>
</p>

<p align="center"><em>Normalized elevation with start/end markers, procedural superellipse obstacles, and two computed paths (free terrain vs. obstacle-aware).</em></p>

### Dataset World Coverage

The dataset leverages the worldwide elevation data publicly provided by JAXA. Each cell below represents a 5×5 grid of 1°×1° tiles (3600×3600 pixels at 30 m resolution). Black cells indicate regions with no available tile — these are treated as flat tiles at sea level during dataset generation.

<p align="center">
  <img src="assets/download.png" alt="Download Coverage Map" width="720"/>
</p>

### Maximum Elevation Map

<p align="center">
  <img src="calibration/max_elevation_map.png" alt="Maximum Elevation Map" width="720"/>
</p>

This visualization shows the **normalized maximum elevation** within each downloaded tile. Brighter regions indicate tiles containing higher terrain (mountain peaks, elevated plateaus); darker regions correspond to low-lying areas (coastal zones, river basins).

> **📈 Elevation Range**
>
> According to the calibration across all tiles:
>
> - Global minimum elevation: **−430.0 m**
> - Global maximum elevation: **8 767.0 m**
>
> Detailed statistics are available in [calibrate.json](calibration/calibrate.json).

### Peak-to-Peak Elevation Map

<p align="center">
  <img src="calibration/ptp_elevation_map.png" alt="Peak-to-Peak Elevation Map" width="720"/>
</p>

The **peak-to-peak (PTP) elevation map** displays the elevation range (max − min) within each tile. High PTP values (bright regions) indicate significant topographic variation — suitable for challenging pathfinding scenarios. Low PTP values suggest relatively flat terrain.

### Hold-Out Region

To assess generalization of trained models, a region containing **South America** is held out. This region was chosen because it concentrates a wide range of elevations representative of all other places in the world map.

<p align="center">
  <img src="assets/hold_out_mask.png" alt="Hold-Out Mask" width="720"/>
</p>

---

## 🚀 Quick Start

### Loading the Dataset

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob

# Load a specific resolution from Hugging Face
ds = load_dataset("your-username/passage-dataset", name="256x256")
sample = ds["train"][0]

# Decode the compressed tensor → (256, 256, 3) numpy array
tensor = decode_tensor_blob(sample["tensor"], (256, 256, 3))
elevation = tensor[:, :, 0]  # Normalized [0, 1]
markers   = tensor[:, :, 1]  # -1 background, 0 start, 1 goal
obstacles = tensor[:, :, 2]  # Binary mask (0/1)

# Access solver paths
path_cols = [k for k in sample if k.startswith("path_")]
print("Solver paths:", path_cols)
```

### Manual Tensor Decompression

```python
import numpy as np
import zstandard as zstd

resolution = sample["metadata"]["resolution"]

# Decompress the tensor
d = zstd.ZstdDecompressor()
b = d.decompress(sample["tensor"])
tensor = np.frombuffer(b, dtype=np.float32).reshape((resolution, resolution, 3))

# Unravel channels
elevation = tensor[:, :, 0]
markers   = tensor[:, :, 1]
obstacles = tensor[:, :, 2]

# Denormalize elevation to meters
global_min = sample["metadata"]["calibration"]["min"]
global_max = sample["metadata"]["calibration"]["max"]
elevation_m = (elevation * (global_max - global_min)) + global_min
```

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a complete loading & visualization example.

---

## 📦 Dataset Structure

Each sample consists of:

| Field | Type | Description |
|-------|------|-------------|
| `tensor` | `bytes` | zstd-compressed `(H, W, 3)` float32 tensor payload |
| `path_<solver>_free` | `list[[i, j], ...]` | Path without obstacles for each configured solver |
| `path_<solver>_obstacles` | `list[[i, j], ...]` | Path with obstacles for each configured solver |
| `image_<solver>` | `image` | Optional per-solver visualization image (sample configs only) |
| `metadata` | `dict` | Sample metadata (coordinates, elevations, stats) |

### Tensor Channels

1. **Channel 0 (Elevation)**: Normalized elevation in $[0, 1]$
2. **Channel 1 (Markers)**: $-1$ everywhere, $0$ at start, $1$ at goal
3. **Channel 2 (Obstacles)**: Binary mask ($0$ = free, $1$ = obstacle)

> **Note**: Tensors are stored as compressed `.npy.zst` blobs.

---

## 🔁 Pipeline Overview

<table>
<tr>
  <td align="center"><strong>1. Download</strong><br><small>JAXA DSM tiles</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>2. Calibrate</strong><br><small>min / max elevation</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>3. Generate</strong><br><small>samples + paths</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>4. Export</strong><br><small>Parquet + HF Hub</small></td>
</tr>
</table>

| Step | Command | What it does |
|:----:|---------|--------------|
| 1 | `passage download` | Download JAXA ALOS AW3D30 5°×5° tile archives with resume |
| 2 | `passage calibrate` | Scan tiles for global min/max elevation |
| 3 | `passage generate` | Create multi-resolution samples with paths, obstacles, tensors |
| 4 | `passage export` | Write Parquet shards, execute notebooks, push to HF Hub |

> 📖 **Full walkthrough:** see the [Documentation](https://thalesgroup.github.io/passage/guide/).

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| [`demo.ipynb`](notebooks/demo.ipynb) | Quick-start: load, decode, and visualize a sample |
| [`costmodel.ipynb`](notebooks/costmodel.ipynb) | Compare all 5 cost models on the same terrain |
| [`obstacles.ipynb`](notebooks/obstacles.ipynb) | Explore superellipse obstacle generation |
| [`solve.ipynb`](notebooks/solve.ipynb) | Solver timing and quality comparison |
| `_dataset.ipynb` | Template — auto-executed per resolution during export |

Per-resolution analysis notebooks are included with each configuration:

- **64×64**: In-depth analysis [notebook](64x64/dataset.ipynb)
- **128×128**: In-depth analysis [notebook](128x128/dataset.ipynb)
- **256×256**: In-depth analysis [notebook](256x256/dataset.ipynb)
- **512×512**: In-depth analysis [notebook](512x512/dataset.ipynb)
- **1024×1024**: In-depth analysis [notebook](1024x1024/dataset.ipynb)
- **2048×2048**: In-depth analysis [notebook](2048x2048/dataset.ipynb)
- **4096×4096**: In-depth analysis [notebook](4096x4096/dataset.ipynb)

---

## 🎯 Sampling Procedure

Each sample is generated through a carefully designed stochastic process ensuring diversity across elevations, geographic locations, and terrain types.

### Step 1 — Target Elevation Sampling

A target elevation $e_{\text{target}}$ is sampled uniformly from the global calibrated range:

$$e_{\text{target}} \sim \mathcal{U}(e_{\min}, e_{\max})$$

where $e_{\min}$ and $e_{\max}$ are computed during calibration across all available tiles.

### Step 2 — Tile Selection

From the set of tiles $\mathcal{T}$ containing the target elevation, a tile $T$ is selected uniformly at random:

$$T \sim \mathcal{U}\left(\{t \in \mathcal{T} : e_{\min}^{(t)} \leq e_{\text{target}} \leq e_{\max}^{(t)}\}\right)$$

### Step 3 — Pixel Localization

Within the selected tile, the pixel $(i^*, j^*)$ closest to $e_{\text{target}}$ is identified:

$$(i^*, j^*) = \arg\min_{(i,j)} \left| \text{elevation}(i,j) - e_{\text{target}} \right|$$

### Step 4 — Crop Extraction

A crop of size $R \times R$ (where $R$ is the resolution) is randomly positioned such that the target pixel $(i^*, j^*)$ lies within the crop. The crop position $(y_1, x_1)$ is sampled as:

$$y_1 \sim \mathcal{U}\left[\max(0, i^* - R + 1), \min(H - R, i^*)\right]$$
$$x_1 \sim \mathcal{U}\left[\max(0, j^* - R + 1), \min(W - R, j^*)\right]$$

### Step 5 — Marker Placement

Two markers define the pathfinding problem:
- **Marker 1** (start): The target pixel $(i^*, j^*)$ within crop coordinates
- **Marker 2** (goal): A uniformly sampled discrete pixel from the $R^2-1$ remaining cells:

$$\text{Marker}_2 \sim \mathcal{U}_{\text{discrete}}\left(\{(i,j) \in [0,R)^2 : (i,j) \neq \text{Marker}_1\}\right)$$

### Step 6 — Elevation Normalization

The elevation crop is normalized to $[0, 1]$ using global calibration statistics:

$$\hat{e}(i,j) = \frac{e(i,j) - e_{\min}}{e_{\max} - e_{\min}}$$

---

## 🧩 Obstacle Generation

Obstacles are procedurally generated using **superellipse** shapes (Lamé curves), allowing a rich variety of geometries — from circular to rectangular and everything in between.

### Superellipse Equation

A superellipse centered at $(c_x, c_y)$ with semi-axes $a$, $b$ and exponent $n$ is defined by:

$$\left(\frac{|x - c_x|}{a}\right)^n + \left(\frac{|y - c_y|}{b}\right)^n \leq 1$$

The exponent $n$ controls the shape:
- $n = 2$: Standard ellipse (circle when $a = b$)
- $n < 2$: Diamond/star shapes ($n = 1$ gives a rhombus)
- $n > 2$: Rounded rectangles (approaches a rectangle as $n \to \infty$)

### Obstacle Parameters

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| **Position** $(c_x, c_y)$ | $\mathcal{U}([0, W) \times [0, H))$ | Center coordinates |
| **Size** $(w, h)$ | $\mathcal{U}([s_{\min} \cdot R, s_{\max} \cdot R])$ | Width and height based on size ratios |
| **Exponent** $n$ | Configurable (default: log-uniform) | Shape parameter |
| **Angle** $\theta$ | $\mathcal{U}([0, 2\pi))$ | Rotation angle |

where $s_{\min}$ and $s_{\max}$ are configurable size ratios relative to resolution $R$.

### Target Coverage

A target obstacle ratio $r_{\text{target}}$ is sampled uniformly:

$$r_{\text{target}} \sim \mathcal{U}(0, r_{\max})$$

Obstacles are iteratively added until coverage approaches $r_{\text{target}}$, while ensuring no obstacle overlaps with the start or goal markers.

---

## ⚖️ Path Cost Models

Each edge weight uses physical step distance $d$ (30 m cardinal, $30\sqrt{2}$ diagonal), weight $\alpha$, and elevation values from the current crop:

- **elevation**

$$w = d \cdot \left(1 + \alpha \cdot \bar{e}_{\text{dst}}\right), \quad \bar{e}_{\text{dst}} = \frac{e_{\text{dst}} - e_{\min}}{e_{\max} - e_{\min}}$$

- **absolute_elevation**

$$w = d \cdot \left(1 + \alpha \cdot \bar{e}^{\text{abs}}_{\text{dst}}\right), \quad \bar{e}^{\text{abs}}_{\text{dst}} = \frac{|e_{\text{dst}}|}{\max |e|}$$

- **distance**

$$w = d$$

- **slope_uphill**

$$w = d \cdot \left(1 + \alpha \cdot \frac{\max(0, e_{\text{dst}} - e_{\text{src}})}{e_{\max} - e_{\min}}\right)$$

- **energy**

$$w = d + \alpha \cdot \max(0, e_{\text{dst}} - e_{\text{src}})$$

These equations match the runtime implementation used by NetworkX and grid solvers.

---

## 🔬 Research Applications

| Domain | Use Case | Relevant Features |
|--------|----------|-------------------|
| **Classical AI** | A\*, Dijkstra benchmarking | 5 cost models, timing data, multi-resolution |
| **Reinforcement Learning** | Terrain navigation agents | Grid observations, reward shaping with cost models |
| **Graph Neural Networks** | Learned heuristics | Graph structure from elevation grids, path labels |
| **Surrogate Models** | Fast approximation of optimal paths | Input/output pairs at scale (14 K+ samples) |
| **Computer Vision** | Path prediction from elevation images | Tensor format, multi-channel images |
| **Certified ML** | ARP 6983 / ED-324 compliance | Bounded ODD, deterministic generation, traceability |

---

## 📝 Citation

If you use PASSAGE in your research, please cite:

```bibtex
@article{delhomme2026passage,
  title   = {PASSAGE: A Multi-Resolution Pathfinding Dataset Mixing Complex
             Obstacles and Elevation-Based Tiles Towards Certified Machine
             Learning Systems},
  author  = {Delhomme, Geoffrey and Trap, Fran{\c{c}}ois and Kaakai, Fateh},
  year    = {2026},
  url     = {https://github.com/ThalesGroup/passage}
}
```

---

## 🙏 Acknowledgments

- **[JAXA](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)** 🇯🇵 for the ALOS AW3D30 Digital Surface Model
- **[Hugging Face](https://huggingface.co/)** for the Datasets library and Hub infrastructure
- **[Thales Group](https://www.thalesgroup.com/)** for supporting open-source research in safety-critical AI

---

## 📄 License

This dataset is released under the **MIT License**. The underlying elevation data is sourced from JAXA's ALOS AW3D30 product, freely available for research purposes.

---

<p align="center">
  <a href="https://thalesgroup.github.io/passage/">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/geoffrey-g-delhomme/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage">💻 GitHub</a> •
  <a href="https://github.com/ThalesGroup/passage/issues">🐛 Issues</a>
</p>

<p align="center"><em>Made with ❤️ for the pathfinding research community</em></p>
