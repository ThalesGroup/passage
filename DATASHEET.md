# PASSAGE Datasheet

## Overview

PASSAGE is a multi-resolution terrain-routing benchmark generated from JAXA ALOS AW3D30 digital surface model tiles.
The default configuration produces 14,130 samples across seven resolutions: 64, 128, 256, 512, 1024, 2048, and 4096.

Each sample contains:

- a compressed `(H, W, 3)` tensor with normalized elevation, start/end markers, and an obstacle mask
- metadata describing calibration bounds, geographic provenance, crop bounds, markers, solver configuration, timing, and deterministic seed provenance
- exported path annotations for `astar_elevation`, `astar_energy`, and `astar_slope`, each with free-terrain and obstacle-aware variants

## Data Sources

- Raw terrain source: JAXA ALOS AW3D30 DSM archives distributed as 5x5 degree downloads.
- Generation unit: PASSAGE operates on 1x1 degree 3600x3600 sub-tiles at approximately 30 meters per pixel.
- Additional synthetic content: superellipse-based obstacle masks sampled procedurally from the repository configuration.

## Collection And Generation Process

1. Download AW3D30 archives and extract the DSM tiles.
2. Calibrate the global elevation minimum and maximum over the local tile cache.
3. For each requested resolution, deterministically derive a per-sample seed from `generation.seed`, resolution, sample index, and split.
4. Sample a target elevation, select a candidate tile, assemble the crop, place start and end markers, and optionally synthesize obstacles.
5. Solve reference paths with the configured solver set and export tensors, metadata, CSV paths, PNG overlays, and parquet shards.

## Splits And Coverage

- Default splits: train 90%, calibration 2%, validation 4%, test 4%.
- Geographic hold-out: test samples are drawn only from the configured hold-out mask; other splits exclude those tiles.
- The benchmark is limited by AW3D30 coverage and quality, as well as the chosen local download footprint.

## Recommended Uses

- Benchmarking terrain-aware shortest-path solvers.
- Training or evaluating learned surrogates for routing on terrain tensors, graphs, or reinforcement-learning environments.
- Studying cross-resolution generalization and geographic hold-out performance.

## Out-Of-Scope Uses

- Operational flight planning, rescue dispatch, or any safety-critical deployment without independent validation.
- Conclusions about weather, airspace, vegetation, urban structures, or regulations: PASSAGE models terrain and procedural obstacles only.

## Biases And Limitations

- Terrain observations inherit artifacts and missingness from the AW3D30 product.
- Procedural obstacles are abstractions, not measured hazards.
- Exported reference paths depend on the configured solver list and obstacle penalty.
- Runtime timestamps are informational; deterministic generation is defined over sample selection and content, not wall-clock metadata.

## Sensitive Data

PASSAGE contains no personal data, biometric data, or directly identifying information.

## Distribution

- Code repository: `https://github.com/ThalesGroup/passage`
- Dataset repository: `https://huggingface.co/datasets/geoffrey-g-delhomme/passage`
- Release metadata: `croissant.json`, `CITATION.cff`, `CHANGELOG.md`, and `outputs/export/manifest.sha256`

## Maintenance

PASSAGE releases should be regenerated with the `Makefile` targets in this repository, versioned in `CHANGELOG.md`, and accompanied by an updated `outputs/export/manifest.sha256` checksum manifest.
