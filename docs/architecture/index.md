---
title: Code Architecture
icon: material/sitemap
---

# Code Architecture

This section describes the internal organization of the PASSAGE codebase — module responsibilities, data flows, design patterns, and testing strategy.

---

## High-Level Architecture

```mermaid
graph TB
    subgraph CLI["CLI Entry Point"]
        MAIN["__init__.py<br>argparse + dispatch"]
    end

    subgraph Pipeline["Pipeline Modules"]
        DL["download.py<br>JAXA tile acquisition"]
        CAL["calibrate.py<br>Elevation statistics"]
        GEN["generate.py<br>Sample generation"]
        EXP["export.py<br>Parquet + HF upload"]
    end

    subgraph Core["Core Libraries"]
        PF["pathfinding_utils.py<br>Solvers & graphs"]
        OBS["obstacles.py<br>Superellipse masks"]
        VIZ["visualization.py<br>Image rendering"]
        UTL["utils.py<br>Tensor compression"]
    end

    subgraph External["External Dependencies"]
        RASTERIO["rasterio<br>GeoTIFF I/O"]
        NX["networkx<br>Graph algorithms"]
        NUMBA["numba<br>JIT compilation"]
        HF["huggingface_hub<br>Dataset upload"]
        PM["papermill<br>Notebook execution"]
    end

    MAIN --> DL
    MAIN --> CAL
    MAIN --> GEN
    MAIN --> EXP

    GEN --> PF
    GEN --> OBS
    GEN --> VIZ
    GEN --> UTL

    EXP --> UTL
    EXP --> PM

    DL --> RASTERIO
    CAL --> RASTERIO
    GEN --> RASTERIO
    PF --> NX
    PF --> NUMBA
    EXP --> HF

    style MAIN fill:#D32F2F,color:#fff
    style GEN fill:#1565C0,color:#fff
    style PF fill:#2E7D32,color:#fff
    style EXP fill:#42A5F5,color:#fff
```

---

## Design Principles

### Lazy Imports

The CLI entry point (`__init__.py`) uses **lazy imports** — pipeline modules are only imported when their subcommand is invoked. This avoids loading heavy dependencies (NumPy, rasterio, NetworkX, Numba) for simple operations like `--help`.

### Multiprocessing

CPU-intensive operations (download, calibrate, generate) use Python's `multiprocessing.Pool` for parallelization. Each worker operates on independent data (a tile, a sample) with no shared state.

### Resume Support

All pipeline steps track progress persistently:

- **Download:** `download.csv` records per-tile status
- **Calibrate:** `calibrate.json` stores per-tile statistics
- **Generate:** sample directories on disk serve as checkpoints

### Numba JIT

Performance-critical pathfinding code uses **optional Numba JIT compilation**. The codebase detects Numba availability at import time and falls back to pure Python implementations if unavailable.

---

## Sections

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [Module Reference](modules.md)
Detailed walkthrough of each source module — purpose, key functions, and interdependencies.
</div>

<div class="feature-card" markdown>
### [Data Format](data-format.md)
Tensor encoding, metadata schema, Parquet features, and file naming conventions.
</div>

<div class="feature-card" markdown>
### [Testing Strategy](testing.md)
23 test suites, pytest markers, coverage targets, and CI integration.
</div>

</div>
