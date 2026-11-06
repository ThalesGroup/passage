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
      heading_level: 3
      members: false
      show_source: false

---

## Core Modules

### `passage.generate` — Dataset Generation

With the shipped `config/config.yaml`, `run_generate` runs the generator that
built PASSAGE v1.0; its labels come from `passage.reference` below. The other
functions listed here form a sample-by-sample generation path that the v1.0
release does not use.

::: passage.generate
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - find_path
        - load_tile
        - load_tile_grid
        - tiles_needed_for_resolution
        - generate_sample
        - count_valid_samples
        - run_generate

---

### `passage.reference` — Exact Reference Solver

The exact solver that labels every PASSAGE v1.0 route variant with an optimal
route, and the connectivity check (`endpoints_connected`) with which the
generator rejects start and goal cells that the obstacles disconnect.

::: passage.reference
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - solve_reference
        - endpoints_connected
        - reverse_goal_field
        - recost_legal_path

---

### `passage.dataset` — Bounded Dataset Access

::: passage.dataset
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - DatasetSource
        - decode_scene_inputs

---

### `passage.pathfinding_utils` — Solvers & Graphs

Graph and collision helpers for the generator's optional solver backends
(`networkx`, `pathfinding` and `grid`), which the v1.0 release does not use.

::: passage.pathfinding_utils
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - build_graph_from_elevation
        - check_path_collision

---

### `passage.obstacles` — Superellipse Generation

::: passage.obstacles
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - superellipse_mask
        - sample_n_from_config
        - sample_angle_from_config

---

### `passage.calibrate` — Elevation Calibration

::: passage.calibrate
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - run_calibrate

---

### `passage.download` — Tile Download

::: passage.download
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - download_tile
        - generate_status_map
        - run_download

---

### `passage.export` — Parquet Export & HF Upload

::: passage.export
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - ExportSteps
        - parse_path_csv
        - compress_release
        - prepare_export
        - upload_export

---

### `passage.utils` — Shared Utilities

::: passage.utils
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - encode_tensor_blob
        - decode_tensor_blob
        - save_tensor
        - load_tensor

---

### `passage.visualization` — Image Rendering

::: passage.visualization
    options:
      show_root_heading: true
      heading_level: 4
      members_order: source
      show_source: false
      members:
        - tensor_to_image
