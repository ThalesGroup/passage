# 🌍 PASSAGE

> **PASSAGE** stands for **PA**th **S**olving Data**S**et **A**nd **GE**nerator

PASSAGE is a professionally maintained toolkit for generating elevation-based datasets from the ALOS AW3D30 DSM (Digital Surface Model). It facilitates reproducible dataset creation and benchmarking for machine learning models that address terrain-based pathfinding challenges.

---

## 🗺️ Data Visualization

### Tile download coverage

<p align="center">
  <img src="assets/download.png" alt="Download Coverage Map" />
</p>

This map shows the tiles downloaded from the [JAXA ALOS AW3D30 DSM](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm) website. Each cell represents a 1°×1° tile (3600×3600 pixels at 30m resolution). The dataset leverages this globally distributed elevation data to ensure diverse terrain samples spanning mountains, valleys, plains, and coastal regions.

### Maximum elevation map

<p align="center">
  <img src="assets/max_elevation_map.png" alt="Maximum Elevation Map" />
</p>

This visualization shows the **normalized maximum elevation** within each downloaded tile. Brighter regions indicate tiles containing higher terrain (mountain peaks, elevated plateaus), while darker regions correspond to low-lying areas (coastal zones, river basins). This map helps understand the global elevation distribution used for sampling.

### Peak-to-Peak elevation map

<p align="center">
  <img src="assets/ptp_elevation_map.png" alt="Peak-to-Peak Elevation Map" />
</p>

The **peak-to-peak (PTP) elevation map** displays the elevation range (max − min) within each tile. High PTP values (bright regions) indicate tiles with significant topographic variation—suitable for challenging pathfinding scenarios. Low PTP values suggest relatively flat terrain. This metric is useful for understanding terrain complexity.

### Representative sample

<p align="center">
  <img src="assets/sample.png" alt="Representative Sample" />
</p>

The image above provides a representative visualization of a single generated sample. It typically combines normalized elevation, marker locations (start/goal), and the obstacle mask, and is intended for quick inspection and demonstration of the sample encoding.

---

## 🎯 Dataset sampling procedure

Each sample in the dataset is generated through a carefully designed stochastic process that ensures diversity across elevations, geographic locations, and terrain types.

### Step 1: Target Elevation Sampling

A target elevation $e_{\text{target}}$ is sampled uniformly from the global calibrated range:

$$e_{\text{target}} \sim \mathcal{U}(e_{\min}, e_{\max})$$

where $e_{\min}$ and $e_{\max}$ are computed during calibration across all available tiles.

### Step 2: Tile Selection

From the set of tiles $\mathcal{T}$ containing the target elevation, a tile $T$ is selected uniformly at random:

$$T \sim \mathcal{U}\left(\{t \in \mathcal{T} : e_{\min}^{(t)} \leq e_{\text{target}} \leq e_{\max}^{(t)}\}\right)$$

### Step 3: Pixel Localization

Within the selected tile, the pixel $(i^*, j^*)$ closest to $e_{\text{target}}$ is identified:

$$(i^*, j^*) = \arg\min_{(i,j)} \left| \text{elevation}(i,j) - e_{\text{target}} \right|$$

### Step 4: Crop Extraction

A crop of size $R \times R$ (where $R$ is the resolution) is randomly positioned such that the target pixel $(i^*, j^*)$ lies within the crop. The crop position $(y_1, x_1)$ is sampled as:

$$y_1 \sim \mathcal{U}\left[\max(0, i^* - R + 1), \min(H - R, i^*)\right]$$
$$x_1 \sim \mathcal{U}\left[\max(0, j^* - R + 1), \min(W - R, j^*)\right]$$

### Step 5: Marker Placement

Two markers define the pathfinding problem:
- **Marker 1** (start): The target pixel $(i^*, j^*)$ within crop coordinates
- **Marker 2** (goal): A randomly selected pixel distinct from Marker 1:

$$\text{Marker}_2 \sim \mathcal{U}\left(\{(i,j) : (i,j) \neq \text{Marker}_1\}\right)$$

### Step 6: Elevation Normalization

The elevation crop is normalized to $[0, 1]$ using global calibration statistics:

$$\hat{e}(i,j) = \frac{e(i,j) - e_{\min}}{e_{\max} - e_{\min}}$$

---

## 🧩 Obstacle generation

Obstacles are procedurally generated using **superellipse** shapes (also known as Lamé curves). This approach allows for a rich variety of obstacle geometries—from circular to rectangular and everything in between.

### Superellipse Equation

A superellipse centered at $(c_x, c_y)$ with semi-axes $a$, $b$ and exponent $n$ is defined by:

$$\left(\frac{|x - c_x|}{a}\right)^n + \left(\frac{|y - c_y|}{b}\right)^n \leq 1$$

The exponent $n$ controls the shape:
- $n = 2$: Standard ellipse (circle when $a = b$)
- $n < 2$: Diamond/star shapes ($n = 1$ gives a rhombus)
- $n > 2$: Rounded rectangles (approaches a rectangle as $n \to \infty$)

### Obstacle Parameters

For each obstacle, the following parameters are sampled:

| Parameter | Distribution | Description |
|-----------|-------------|-------------|
| **Position** $(c_x, c_y)$ | $\mathcal{U}([0, W) \times [0, H))$ | Center coordinates |
| **Size** $(w, h)$ | $\mathcal{U}([s_{\min} \cdot R, s_{\max} \cdot R])$ | Width and height based on size ratios |
| **Exponent** $n$ | Log-uniform on $[n_{\min}, n_{\max}]$ | Shape parameter |
| **Angle** $\theta$ | $\mathcal{U}([0, 2\pi))$ | Rotation angle |

where $s_{\min}$ and $s_{\max}$ are configurable size ratios (relative to resolution $R$).

### Target Coverage

A target obstacle ratio $r_{\text{target}}$ is sampled uniformly:

$$r_{\text{target}} \sim \mathcal{U}(0, r_{\max})$$

Obstacles are iteratively added until the coverage approaches $r_{\text{target}}$, while ensuring no obstacle overlaps with the start or goal markers.

---

## 📦 Dataset structure

Each sample consists of:

| Field | Type | Description |
|-------|------|-------------|
| `tensor` | `float32[H, W, 3]` | 3-channel tensor: elevation, markers, obstacles |
| `image` | `bytes` | PNG visualization of the sample |
| `path_free` | `list[[i, j], ...]` | Optimal path without obstacles |
| `path_obstacles` | `list[[i, j], ...]` | Optimal path with obstacles |
| `metadata` | `dict` | Sample metadata (coordinates, elevations, stats) |

### Tensor Channels

1. **Channel 0 (Elevation)**: Normalized elevation in $[0, 1]$
2. **Channel 1 (Markers)**: $-1$ everywhere, $0$ at start, $1$ at goal
3. **Channel 2 (Obstacles)**: Binary mask ($0$ = free, $1$ = obstacle)

---

## ⚡ Quick start

```bash
# Install dependencies
uv sync

# Generate the dataset
uv run passage generate --resolution 256 --num-samples 1000

# Prepare for export
uv run passage prepare --repo-id your-username/passage-dataset

# Upload to Hugging Face
cd outputs/export && uv run hf upload . your-username/passage-dataset --repo-type dataset
```

### Loading the Dataset

```python
from datasets import load_dataset

# Load a specific resolution
dataset = load_dataset("your-username/passage-dataset", "256")

# Access a sample
sample = dataset["train"][0]
print(f"Sample ID: {sample['id']}")
print(f"Path length (free): {len(sample['path_free'])}")
print(f"Path length (obstacles): {len(sample['path_obstacles'])}")
```

---

## 📓 Notebooks

Detailed analysis and visualization are available in the `notebooks/` folder:

- **`dataset.ipynb`** — Comprehensive dataset exploration, statistics, and visualizations
- **`costmodel.ipynb`** — Pathfinding cost model analysis
- **`display.ipynb`** — Sample visualization utilities
- **`metrics.ipynb`** — Path quality metrics
- **`obstacles.ipynb`** — Obstacle generation analysis
- **`solve.ipynb`** — Pathfinding solver comparisons

For in-depth documentation of the sampling procedure, obstacle generation, and dataset statistics, please refer to the **`dataset.ipynb`** notebook included with each resolution export.

---

## 📜 License

This dataset is released under the MIT License. The underlying elevation data is sourced from JAXA's ALOS AW3D30 product, which is freely available for research purposes.

---

## 🙏 Acknowledgments

- **JAXA** 🇯🇵 — provided the ALOS AW3D30 Digital Surface Model
- Open-source community 🌍 — contributions to pathfinding algorithms and geospatial tooling

---
