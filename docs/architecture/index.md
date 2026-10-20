---
title: Code Architecture
icon: material/sitemap
---

# Code Architecture

This section describes the internal organization of the PASSAGE codebase: module responsibilities, data flows, design patterns and testing strategy. One distribution ships two packages: `passage` (`src/passage/`, the data pipeline) and `passage_eval` (`src/passage-eval/`, the evaluator).

---

## High-Level Architecture

```mermaid
graph TB
    subgraph CLI["CLI Entry Point"]
        MAIN["__init__.py<br>argparse + dispatch<br>10 subcommands"]
    end

    subgraph Pipeline["Pipeline Modules"]
        DL["download.py<br>JAXA tile acquisition"]
        CAL["calibrate.py<br>Elevation calibration"]
        GEN["generate.py<br>Scene generation"]
        EXP["export.py<br>Parquet + HF upload"]
        REG["regenerate.py<br>Full rebuild in batches"]
    end

    subgraph Release["Release Checks"]
        AUD["audit.py<br>Bounded audit"]
        QUA["qualification modules<br>Snapshot qualification"]
        NB["notebook_export.py<br>Notebook execution"]
    end

    subgraph Core["Core Libraries"]
        REF["reference.py<br>Exact reference solver"]
        OBS["obstacles.py<br>Superellipse masks"]
        DS["dataset.py<br>Dataset access"]
        PF["pathfinding_utils.py<br>Alternative backends"]
        VIZ["visualization.py + plotting.py<br>Image rendering"]
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
    MAIN --> REG
    MAIN --> AUD
    MAIN --> QUA
    MAIN --> NB

    GEN --> REF
    GEN --> OBS
    GEN --> DS
    GEN --> UTL
    GEN --> PF
    REG --> GEN
    REG --> EXP

    EXP --> UTL
    EXP --> VIZ
    EXP --> PM
    AUD --> DS
    QUA --> DS

    DL --> RASTERIO
    CAL --> RASTERIO
    GEN --> RASTERIO
    PF --> NX
    PF --> NUMBA
    REF --> NUMBA
    EXP --> HF

    style MAIN fill:#D32F2F,color:#fff
    style GEN fill:#1565C0,color:#fff
    style REF fill:#2E7D32,color:#fff
    style EXP fill:#42A5F5,color:#fff
```

---

## Design Principles

### Lazy Imports

The CLI entry point (`__init__.py`) uses **lazy imports**: a pipeline module is imported only when its subcommand runs. Simple operations such as `--help` therefore do not load heavy dependencies (NumPy, rasterio, NetworkX, Numba).

### Multiprocessing

CPU-intensive steps run in parallel worker processes: download and calibration use a `multiprocessing` pool, and the release generator keeps one process pool per resolution. Each worker handles independent data (an archive, a tile or a scene).

### Resume Support

All pipeline steps record their progress on disk:

- **Download:** a hashed receipt per archive, and `download.csv` with the status of each archive.
- **Calibrate:** `calibrate.json` stores the per-tile statistics and the tile inventory.
- **Generate:** each scene's receipt is written last and marks the scene as complete; on resume, the generator checks every existing receipt and skips complete scenes.
- **Regenerate:** a journal records the committed batches.

### Numba JIT

The exact reference solver, the evaluator's cost functions and the grid backend are compiled with **Numba**, which is a required dependency.

---

## Sections

<div class="feature-grid" markdown>

<div class="feature-card" markdown>
### [Module Reference](modules.md)
The role of each source module, its main functions and the ten CLI subcommands.
</div>

<div class="feature-card" markdown>
### [Data Format](data-format.md)
Tensor encoding, metadata keys, route files, Parquet columns and file naming conventions.
</div>

<div class="feature-card" markdown>
### [Testing Strategy](testing.md)
65 test files and 860 tests, pytest markers, the 80 % coverage gate and CI integration.
</div>

</div>
