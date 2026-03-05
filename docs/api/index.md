---
title: API Reference
icon: material/api
---

# API Reference

Auto-generated documentation from PASSAGE source code docstrings.

---

## Package Overview

::: passage
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      show_source: false

---

## Core Modules

### `passage.generate` — Dataset Generation

::: passage.generate
    options:
      show_root_heading: true
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "^run_generate"
        - "^generate_sample"
        - "^find_path"
        - "^load_tile"
        - "^load_tile_grid"
        - "^tiles_needed_for_resolution"
        - "^count_valid_samples"

---

### `passage.pathfinding_utils` — Solvers & Graphs

::: passage.pathfinding_utils
    options:
      show_root_heading: true
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "^build_graph_from_elevation"
        - "^check_path_collision"

---

### `passage.obstacles` — Superellipse Generation

::: passage.obstacles
    options:
      show_root_heading: true
      members_order: source
      show_source: true

---

### `passage.calibrate` — Elevation Calibration

::: passage.calibrate
    options:
      show_root_heading: true
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "^run_calibrate"

---

### `passage.download` — Tile Download

::: passage.download
    options:
      show_root_heading: true
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "^run_download"
        - "^download_tile"
        - "^generate_status_map"

---

### `passage.export` — Parquet Export & HF Upload

::: passage.export
    options:
      show_root_heading: true
      members_order: source
      show_source: true
      filters:
        - "!^_"
        - "^export_and_upload"
        - "^parse_path_csv"

---

### `passage.utils` — Shared Utilities

::: passage.utils
    options:
      show_root_heading: true
      members_order: source
      show_source: true

---

### `passage.visualization` — Image Rendering

::: passage.visualization
    options:
      show_root_heading: true
      members_order: source
      show_source: true
