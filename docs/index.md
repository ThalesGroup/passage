---
title: PASSAGE — Path Solving Dataset & Generator
description: Open-source elevation-based pathfinding benchmarks to advance research and save lives
icon: material/home
hide:
  - navigation
  - toc
---

<link rel="stylesheet" href="https://unpkg.com/katex@0.16.11/dist/katex.min.css">

<!-- Hero Section -->
<div class="hero-section" markdown>

# PASSAGE

<p class="hero-tagline">
  <strong>PA</strong>th <strong>S</strong>olving Data<strong>S</strong>et <strong>A</strong>nd <strong>GE</strong>nerator<br>
  Real-world elevation-based pathfinding benchmarks to advance research — and save lives.
</p>

<div class="hero-buttons" markdown>

[:material-rocket-launch: **Get Started**](guide/index.md){ .btn-primary }
[:fontawesome-brands-github: **View on GitHub**](https://github.com/ThalesGroup/passage){ .btn-secondary }
[:simple-huggingface: **Dataset on HF**](https://huggingface.co/datasets/thalesgroup/passage){ .btn-secondary }

</div>

</div>

<!-- Badges -->
<div class="badges" markdown>

![Python](https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green?logo=opensourceinitiative&logoColor=white)
![Tests](https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/ci.yml?label=tests&logo=githubactions&logoColor=white)
![Coverage](https://img.shields.io/badge/coverage-≥80%25-brightgreen?logo=pytest&logoColor=white)
![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow)
![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white)
![uv](https://img.shields.io/badge/package-uv-DE5FE9?logo=uv&logoColor=white)
![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)

</div>

<!-- show banner image -->
<p align="center">
  <img src="assets/banner.png" alt="PASSAGE banner" width="100%" />
</p>

<!-- Stats Bar -->
<div class="stats-bar" markdown>

<div class="stat-item">
  <span class="stat-value">141,300</span>
  <span class="stat-label">Default Samples</span>
</div>
<div class="stat-item">
  <span class="stat-value">7</span>
  <span class="stat-label">Resolution Levels</span>
</div>
<div class="stat-item">
  <span class="stat-value">5</span>
  <span class="stat-label">Cost Models</span>
</div>
<div class="stat-item">
  <span class="stat-value">3</span>
  <span class="stat-label">Solver Backends</span>
</div>
<div class="stat-item">
  <span class="stat-value">30 m/px</span>
  <span class="stat-label">Spatial Resolution</span>
</div>

</div>

---

<!-- Story Section -->
<div class="story-section" markdown>

## Context

**Helicopter Emergency Medical Services (HEMS)** operate in some of the most challenging navigation environments on Earth. When a rescue crew takes off toward a patient in mountainous terrain, the flight path must account for elevation changes, ridgelines, valleys, and no-fly zones — all in real time. Selecting an efficient route across complex 3D landscapes is a genuine operational challenge, and the underlying pathfinding algorithms play an important role in mission planning.

Despite this, most pathfinding research still relies on **synthetic flat grids** or small hand-crafted maps that do not reflect the complexity of real terrain. This gap motivated the creation of PASSAGE.

**PASSAGE was built to bridge that gap.**

Using real-world elevation data from the [JAXA ALOS AW3D30](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm) Digital Surface Model (30 m/pixel), PASSAGE generates **large-scale, multi-resolution pathfinding benchmarks** grounded in actual geography — from coastal plains to high-altitude mountain ranges.

We open-source this toolkit to give the research community access to realistic terrain data for developing and evaluating pathfinding algorithms. Whether you work on classical planners (A\*), reinforcement learning, graph neural networks, or certified surrogate models for safety-critical systems — PASSAGE provides the data to support meaningful progress.

</div>

---

## :compass: What is PASSAGE?

**PASSAGE** is a complete, reproducible pipeline that transforms raw satellite elevation data into ready-to-use pathfinding datasets:

<div class="pipeline-steps" markdown>

<div class="pipeline-step" markdown>
  <div class="step-number">1</div>

#### Download

Acquire JAXA ALOS AW3D30 DSM tiles covering the entire globe with geographic tracking and resume support.
</div>

<div class="pipeline-step" markdown>
  <div class="step-number">2</div>

#### Calibrate

Scan all tiles to compute global min / max elevation values, producing calibration maps.
</div>

<div class="pipeline-step" markdown>
  <div class="step-number">3</div>

#### Generate

Create multi-resolution samples with start/end markers, superellipse obstacles, and paths from multiple solvers.
</div>

<div class="pipeline-step" markdown>
  <div class="step-number">4</div>

#### Compress

Rewrite the selected resolutions as Parquet shards in the release tree.
</div>

<div class="pipeline-step" markdown>
  <div class="step-number">5</div>

#### Export

Refresh the upload-ready export tree: README, notebooks, calibration assets, and per-resolution analyses.
</div>

<div class="pipeline-step" markdown>
  <div class="step-number">6</div>

#### Upload

Publish the prepared export tree to Hugging Face Hub and verify it with the release manifest.
</div>

</div>

---

## :star2: Key Features

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
  <span class="feature-icon">🌐</span>

### Multi-Resolution

Seven resolution levels from **64×64** to **4096×4096** pixels, covering terrain patches from 1.9 km to 123 km across.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🏔️</span>

### Real Elevation Data

Every sample sourced from **JAXA ALOS AW3D30** — 30 m/pixel Digital Surface Model tiles covering the entire globe.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🔷</span>

### Rich Obstacles

Procedurally generated **superellipse** obstacles with configurable shape, size, rotation, and coverage — from circles to near-rectangles.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">⚡</span>

### 5 Cost Models

Elevation, distance, slope-uphill, energy, and absolute cost functions — each capturing different navigation priorities.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🧮</span>

### 3 Solver Backends

NetworkX graphs, `python-pathfinding` grids, and a custom **Numba-JIT** grid engine for high-performance A\* and Dijkstra.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🤗</span>

### HuggingFace-Ready

Split release flow with `compress`, `export`, and `upload`, plus `manifest.sha256` for release verification.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🔁</span>

### Fully Resumable

The generation and release-preparation stages can be rerun independently — no need to rebuild the whole dataset after an interruption.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">✅</span>

### Tested & Reproducible

23 test suites, 80 %+ coverage, deterministic seeding, pre-commit hooks, and full CI/CD pipeline.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🛡️</span>

### Certification-Aware

Designed with aerospace certification standards (ARP 6983 / ED-324) in mind — bounded operational design domain, traceable samples.
</div>

</div>

---

## :zap: Quick Start

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/ThalesGroup/passage.git
cd passage
uv sync

# Run the full pipeline
passage download                        # Step 1 — fetch JAXA tiles
passage calibrate                       # Step 2 — compute elevation stats
passage generate --resolution 256       # Step 3 — generate samples
passage compress --resolution 256       # Step 4 — write parquet shards
passage export --resolution 256         # Step 5 — refresh export tree
passage upload                          # Step 6 — publish to Hugging Face
```

!!! tip "Explore a sample in 3 lines"

    ```python
    from datasets import load_dataset
    from passage.utils import decode_tensor_blob

    ds = load_dataset("thalesgroup/passage", name="256x256")
    sample = ds["train"][0]
    tensor = decode_tensor_blob(sample["tensor"], (256, 256, 3))
    ```

---

## :microscope: Research Applications

PASSAGE is designed to support a wide spectrum of pathfinding research:

| Domain | Use Case | Relevant Features |
|--------|----------|-------------------|
| **Classical AI** | A\*, Dijkstra benchmarking | 5 cost models, timing data, multi-resolution |
| **Reinforcement Learning** | Terrain navigation agents | Grid observations, reward shaping with cost models |
| **Graph Neural Networks** | Learned heuristics | Graph structure from elevation grids, path labels |
| **Surrogate Models** | Fast approximation of optimal paths | Input/output pairs at scale (141K+ samples) |
| **Computer Vision** | Path prediction from elevation images | Tensor format, multi-channel images |
| **Certified ML** | ARP 6983 / ED-324 compliance | Bounded ODD, deterministic generation, traceability |

---

## :book: Learn More

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
  <span class="feature-icon">📖</span>

### [User Guide](guide/index.md)

Step-by-step walkthrough of the complete pipeline, from installation to Hugging Face export.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🔬</span>

### [Generation Process](generation/index.md)

Deep dive into sampling, cost models, obstacle generation, and solver algorithms — with full math.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">🏗️</span>

### [Code Architecture](architecture/index.md)

Module-by-module tour of the codebase, data formats, and test strategy.
</div>

<div class="feature-card" markdown>
  <span class="feature-icon">📚</span>

### [API Reference](api/index.md)

Auto-generated documentation from source code docstrings.
</div>

</div>

---

## :handshake: Acknowledgments

PASSAGE is built on the shoulders of outstanding open data and open-source communities:

- **[JAXA](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)** for the ALOS AW3D30 Digital Surface Model
- **[Hugging Face](https://huggingface.co/)** for the Datasets library and Hub infrastructure
- **[Thales Group](https://www.thalesgroup.com/)** for supporting open-source research in safety-critical AI

---

<p align="center" style="color: var(--md-default-fg-color--light); font-size: 0.9rem;">
  Made with :heart: for the pathfinding research community<br>
  <strong>MIT License</strong> — Free to use, modify, and distribute
</p>
