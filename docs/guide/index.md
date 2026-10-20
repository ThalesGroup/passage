---
title: Dataset Generation
icon: material/book-open-page-variant
---

# Dataset Generation

Welcome to the PASSAGE Dataset Generation section. It walks you through
the **complete generator workflow** — from installing the toolkit to
publishing your dataset on Hugging Face Hub — and includes the deeper
[**Generation Process**](../generation/index.md) reference covering
sampling, cost models, obstacles and solvers.

To **score a planner** on the resulting benchmark (decoded-path
metrics, statistical discipline, deployment footprint), see the
[Benchmark](../benchmark/index.md) section.

!!! info "JAXA AW3D30 — license obligations"
    Every PASSAGE scene is built on the JAXA ALOS World 3D–30 m DSM
    (AW3D30). Any user of PASSAGE or its derived artifacts must
    respect [JAXA's terms of use](https://earth.jaxa.jp/en/data/policy/)
    — cite the AW3D30 source, propagate the attribution downstream,
    and read the [JAXA license notice](../index.md#jaxa-license-and-attribution)
    before redistributing.

---

## Overview

PASSAGE follows a linear, six-step pipeline. Each step is a CLI subcommand that can be run on its own and produces well-defined outputs:

```mermaid
flowchart LR
    A["🌐 Download<br>JAXA tiles"] --> B["📏 Calibrate<br>Elevation range"]
    B --> C["⚙️ Generate<br>Scenes and routes"]
    C --> D["🗜️ Compress<br>Parquet shards"]
    D --> E["🧾 Export<br>README + notebooks + assets"]
    E --> F["🚀 Upload<br>HF Hub publish"]

    style A fill:#1565C0,color:#fff,stroke:#0D47A1
    style B fill:#42A5F5,color:#fff,stroke:#1565C0
    style C fill:#2E7D32,color:#fff,stroke:#1B5E20
    style D fill:#00897B,color:#fff,stroke:#00695C
    style E fill:#6D4C41,color:#fff,stroke:#4E342E
    style F fill:#D32F2F,color:#fff,stroke:#B71C1C
```

| Step | Command | What it does | Main output |
|------|---------|-------------|-------------|
| 1 | `passage download` | Downloads the 5°×5° ZIP archives of 1°×1° AW3D30 tiles | `tiles/*_DSM.tif`, `download.csv` |
| 2 | `passage calibrate` | Scans all tiles for the global min/max elevation | `calibrate.json` |
| 3 | `passage generate` | Creates scenes: tensors, metadata and six reference routes | Dataset tree with one receipt per scene |
| 4 | `passage compress` | Writes Parquet shards for the selected resolutions | `<R>x<R>` and `<R>x<R>_samples` configurations |
| 5 | `passage export` | Prepares the upload-ready export tree | README, executed notebooks, assets |
| 6 | `passage upload` | Publishes the prepared export tree to Hugging Face | Hub dataset repository |

For the full release, `passage regenerate` runs steps 3 to 5 in bounded,
resumable batches (see [Steps 4-6 — Release](export.md#all-in-one-passage-regenerate)).

!!! tip "Interrupted runs"
    Download, calibration and generation keep receipts: re-run the same command and they resume where they stopped.
    `compress` replaces configurations only once the new ones are complete, and `export` skips steps whose outputs are already valid.

---

## Prerequisites

- **Python 3.11+** on Linux
- **`uv`** package manager (recommended) or `pip`
- About 463 GB of disk for the 23,993 AW3D30 DSM tiles of the release, plus room for the generated scenes
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
Acquire JAXA elevation tiles, with a hashed receipt per archive.
</div>

<div class="feature-card" markdown>
### [Step 2 — Calibrate](calibrate.md)
Compute the global elevation range and record the tile inventory.
</div>

<div class="feature-card" markdown>
### [Step 3 — Generate](generate.md)
Create multi-resolution scenes with exact reference routes.
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
