---
title: Changelog
icon: material/history
---

# Changelog

All notable changes to the PASSAGE project are documented here. This project follows [Semantic Versioning](https://semver.org/).

---

## [0.1.0] — 2025-XX-XX

### :tada: Initial Release

The first public release of PASSAGE — **PA**th **S**olving Data**S**et **A**nd **GE**nerator.

#### Features

- **Download pipeline** — automated JAXA ALOS AW3D30 tile download with URL validation, resume support, and world-coverage status map
- **Calibration** — parallel scanning of GeoTIFF tiles for global min/max elevation with visualization maps
- **Dataset generation** — multi-resolution sample creation (64×64 to 4096×4096) with:
    - Uniform elevation-based sampling across the globe
    - Configurable start/end marker placement
    - Superellipse obstacle generation with rich shape distributions
    - 5 cost models (elevation, distance, slope-uphill, energy, absolute)
    - 3 solver backends (NetworkX, pathfinding, Numba-JIT grid)
    - A\* and Dijkstra search algorithms with 3 heuristics
    - Deterministic split assignment (train/calibration/validation/test)
    - Geographic hold-out region support
- **Export pipeline** — Parquet shard creation with Hugging Face Features schema, zstd compression, notebook execution via papermill, and full-folder upload to HF Hub
- **CLI** — 4 subcommands (`download`, `calibrate`, `generate`, `export`) with comprehensive argument parsing
- **Tensor format** — 3-channel (elevation, markers, obstacles) numpy arrays with zstd compression
- **Visualization** — terrain colormap rendering with obstacle overlays, markers, and path polylines
- **Analysis notebooks** — demo, cost model comparison, obstacle exploration, solver analysis
- **Testing** — 23 test files covering all modules, 80%+ branch coverage, pytest markers
- **Quality** — Ruff linting/formatting, pre-commit hooks, editorconfig
- **Documentation** — MkDocs Material site with full user guide, generation process deep-dive, code architecture reference, and auto-generated API docs

#### Configuration

- Default: 7 resolution levels, 141,300 total samples
- 3 pre-configured A\* solvers with calibrated weights
- Log-uniform superellipse exponent distribution ($n \in [1, 80]$)
- 30% maximum obstacle coverage
- 90/2/4/4% train/calibration/validation/test split

---

## [Unreleased]

### Planned

- [ ] Additional cost models (wind-aware, fuel consumption)
- [ ] GPU-accelerated solvers
- [ ] Interactive 3D terrain visualization
- [ ] Multi-agent pathfinding support
- [ ] Ordered waypoint sequences
- [ ] Extended elevation data sources
- [ ] ML baselines (U-Net, GNN, PPO)
