---
title: Interactive Notebooks
icon: material/notebook
---

# Interactive Notebooks

PASSAGE includes several **Jupyter notebooks** for exploring the dataset, visualizing results, and understanding the generation process. These notebooks serve both as documentation and as analysis tools.

---

## Available Notebooks

| Notebook | Purpose | Key outputs |
|----------|---------|-------------|
| **`demo.ipynb`** | Quick-start: load a sample and visualize it | Elevation heatmap, markers, obstacles, paths |
| **`costmodel.ipynb`** | Compare all 5 cost models on the same terrain | Side-by-side path visualizations |
| **`obstacles.ipynb`** | Explore obstacle generation & distributions | Superellipse shape gallery, coverage histograms |
| **`solve.ipynb`** | Solver timing & quality comparison | Timing bar charts, path cost scatter plots |
| **`_dataset.ipynb`** | Template — executed per resolution during export | Per-resolution statistics and samples |

!!! note "Template notebooks"
    Notebooks prefixed with `_` (like `_dataset.ipynb`, `_display.ipynb`, `_metrics.ipynb`) are internal templates used by the export pipeline. They are parameterized and executed automatically via [papermill](https://papermill.readthedocs.io/).

---

## Demo Notebook

The demo notebook is the best starting point for exploring PASSAGE data:

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob
import matplotlib.pyplot as plt

# Load dataset
ds = load_dataset("geoffrey-g-delhomme/passage", name="256x256")
sample = ds["train"][0]

# Decode tensor
tensor = decode_tensor_blob(sample["tensor"], (256, 256, 3))
elevation = tensor[:, :, 0]
markers = tensor[:, :, 1]
obstacles = tensor[:, :, 2]

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(elevation, cmap="terrain")
axes[0].set_title("Elevation")
axes[1].imshow(markers, cmap="RdYlBu")
axes[1].set_title("Markers")
axes[2].imshow(obstacles, cmap="Reds")
axes[2].set_title("Obstacles")
plt.tight_layout()
plt.show()
```

---

## Cost Model Notebook

Compares how different cost models produce different paths for the **same terrain and markers**:

- **Elevation model** → paths that avoid high-altitude regions
- **Slope-uphill model** → paths that minimize climbing
- **Energy model** → paths that minimize total elevation gain
- **Distance model** → shortest geometric path (baseline)

This notebook is especially useful for understanding the practical impact of cost model selection.

---

## Obstacles Notebook

Explores the obstacle generation process:

- Gallery of superellipse shapes across different exponents $n$
- Distribution of obstacle coverage ratios across the dataset
- Effect of `ratio_max` on path difficulty
- Visual comparison of paths with and without obstacles

---

## Solve Notebook

Analyzes solver performance:

- Timing comparison across backends (NetworkX vs. pathfinding vs. grid)
- Path cost distributions per cost model
- Resolution scaling analysis
- Solver convergence characteristics

---

## Running Notebooks Locally

```bash
# Install Jupyter kernel
uv sync --group dev

# Launch Jupyter
uv run jupyter lab

# Or execute a specific notebook
uv run jupyter execute notebooks/demo.ipynb
```

---

## Notebook Exports on Hugging Face

When you run `passage export`, notebooks are automatically:

1. **Copied** to the export directory under `notebooks/`
2. **Executed** via `papermill` (for `_dataset.ipynb` — parameterized per resolution)
3. **Uploaded** to the Hugging Face Hub alongside the dataset

This means users can view pre-executed notebooks directly on the HF dataset page without needing to run them locally.

!!! tip "Viewing on Hugging Face"
    Navigate to the `notebooks/` directory in your [HF dataset repository](https://huggingface.co/datasets/geoffrey-g-delhomme/passage) to view rendered notebooks with all outputs.
