---
title: User Guide
icon: material/book-open-page-variant
---

# User Guide

Welcome to the PASSAGE User Guide. This section walks you through the **complete workflow** — from installing the toolkit to publishing your dataset on Hugging Face Hub.

---

## Overview

PASSAGE follows a linear, four-step pipeline. Each step is a CLI subcommand that can be run independently, supports resume, and produces well-defined outputs:

```mermaid
flowchart LR
    A["🌐 Download\nJAXA tiles"] --> B["📏 Calibrate\nElevation stats"]
    B --> C["⚙️ Generate\nDataset samples"]
    C --> D["📦 Export\nParquet + HF Hub"]

    style A fill:#1565C0,color:#fff,stroke:#0D47A1
    style B fill:#42A5F5,color:#fff,stroke:#1565C0
    style C fill:#2E7D32,color:#fff,stroke:#1B5E20
    style D fill:#D32F2F,color:#fff,stroke:#B71C1C
```

| Step | Command | What it does | Typical runtime |
|------|---------|-------------|-----------------|
| 1 | `passage download` | Downloads JAXA ALOS AW3D30 5°×5° tile archives | Hours (network-bound) |
| 2 | `passage calibrate` | Scans all tiles for global min/max elevation | Minutes (I/O-bound) |
| 3 | `passage generate` | Creates samples with paths, obstacles, tensors | Hours (CPU-bound) |
| 4 | `passage export` | Writes Parquet shards, executes notebooks, uploads | Minutes to hours |

!!! tip "Each step is resumable"
    If a step is interrupted, simply re-run the same command. PASSAGE will detect existing progress and pick up where it left off.

---

## Prerequisites

- **Python 3.11+**
- **`uv`** package manager (recommended) or `pip`
- ~50 GB free disk space for full tile download
- Hugging Face account + token (only for `export` with upload)

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
### [Step 4 — Export](export.md)
Package and publish to Hugging Face Hub.
</div>

<div class="feature-card" markdown>
### [Configuration](configuration.md)
Full reference for `config/config.yaml`.
</div>

</div>
