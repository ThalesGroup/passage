---
title: Dataset Generation
icon: material/book-open-page-variant
---

# Dataset Generation

Welcome to the PASSAGE Dataset Generation section. It walks you through
the **complete generator workflow** — from installing the toolkit to
publishing your dataset on Hugging Face Hub — and includes the deeper
[**Generation Process**](../generation/index.md) reference covering
sampling, cost models, obstacles, solvers, and configuration.

To **score a planner** on the resulting benchmark (decoded-path
metrics, statistical discipline, deployment footprint), see the
[Benchmark](../benchmark/index.md) section.

!!! info "JAXA AW3D30 — license obligations"
    Every PASSAGE sample is built on the JAXA ALOS World 3D 30 m DSM
    (AW3D30). Any user of PASSAGE or its derived artefacts must
    respect [JAXA's terms of
    use](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)
    — cite the AW3D30 source, propagate the attribution downstream,
    and read the [JAXA license notice](../index.md#jaxa-license-and-attribution)
    before redistributing.

---

## Overview

PASSAGE follows a linear, six-step pipeline. Each step is a CLI subcommand that can be run independently, supports resume, and produces well-defined outputs:

```mermaid
flowchart LR
    A["🌐 Download\nJAXA tiles"] --> B["📏 Calibrate\nElevation stats"]
    B --> C["⚙️ Generate\nDataset samples"]
    C --> D["🗜️ Compress\nParquet shards"]
    D --> E["🧾 Export\nREADME + notebooks + assets"]
    E --> F["🚀 Upload\nHF Hub publish"]

    style A fill:#1565C0,color:#fff,stroke:#0D47A1
    style B fill:#42A5F5,color:#fff,stroke:#1565C0
    style C fill:#2E7D32,color:#fff,stroke:#1B5E20
    style D fill:#00897B,color:#fff,stroke:#00695C
    style E fill:#6D4C41,color:#fff,stroke:#4E342E
    style F fill:#D32F2F,color:#fff,stroke:#B71C1C
```

| Step | Command | What it does | Typical runtime |
|------|---------|-------------|-----------------|
| 1 | `passage download` | Downloads JAXA ALOS AW3D30 5°×5° tile archives | Hours (network-bound) |
| 2 | `passage calibrate` | Scans all tiles for global min/max elevation | Minutes (I/O-bound) |
| 3 | `passage generate` | Creates samples with paths, obstacles, tensors | Hours (CPU-bound) |
| 4 | `passage compress` | Rewrites Parquet shards for the selected resolutions | Minutes to hours |
| 5 | `passage export` | Refreshes the upload-ready export tree | Minutes |
| 6 | `passage upload` | Publishes the prepared export tree to Hugging Face | Minutes to hours |

!!! tip "Each step is resumable"
    If a step is interrupted, simply re-run the same command. PASSAGE will detect existing progress and pick up where it left off.

---

## Prerequisites

- **Python 3.11+**
- **`uv`** package manager (recommended) or `pip`
- ~50 GB free disk space for full tile download
- Hugging Face token (only for `upload`)

---

## What's Next?

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [Installation](installation.md)
Set up your environment, install dependencies, and configure credentials.
</div>

<div class="feature-card" markdown>
### [Step 1 — Download](download.md)
Acquire JAXA elevation tiles with geographic tracking.
</div>

<div class="feature-card" markdown>
### [Step 2 — Calibrate](calibrate.md)
Compute global elevation statistics across all tiles.
</div>

<div class="feature-card" markdown>
### [Step 3 — Generate](generate.md)
Create multi-resolution pathfinding samples.
</div>

<div class="feature-card" markdown>
### [Steps 4-6 — Release](export.md)
Compress Parquet shards, refresh the export tree, and publish to Hugging Face Hub.
</div>

<div class="feature-card" markdown>
### [Configuration](configuration.md)
Full reference for `config/config.yaml`.
</div>

</div>
