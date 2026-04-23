# PASSAGE Datasheet

This Datasheet follows the Gebru et al. [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010) template and is mirrored in the companion Croissant metadata file (`croissant.json`, conforming to MLCommons Croissant 1.1 and Croissant RAI 1.0). It accompanies the NeurIPS 2026 Evaluations & Datasets Track submission of the PASSAGE benchmark.

## Motivation

### For what purpose was the dataset created?

PASSAGE was created to close a reproducibility gap in *constrained path planning on real terrain*: existing benchmarks either synthesize grids that strip geographic structure, rely on ad-hoc GIS workflows without machine-learning-ready splits, or evaluate embodied perception rather than cost-aware long-horizon routing. PASSAGE provides a real-terrain, multi-resolution benchmark with an enumerable Operational Design Domain (ODD), configurable terrain cost models, procedural obstacles, and exact A* ground-truth labels — packaged as ML-ready Parquet shards with a manifest-checked release tree.

### Who created this dataset and on behalf of which entity?

The dataset was created by Geoffrey Delhomme, François Trap, and Fateh Kaakai on behalf of Thales Group research. See `CITATION.cff` for full attribution. Maintenance is owned by the Thales Group open-source research SIG; community contact via GitHub Issues, Discussions, or `oss@thalesgroup.com`.

### Who funded the creation of the dataset?

Internal research funding (Thales Group). No external grant is associated with this release.

## Composition

### What do the instances represent?

Each instance is a single path-planning problem on a terrain crop: a three-channel input tensor (elevation, start/goal markers, obstacle mask) at a chosen resolution `N`, plus path annotations (ordered `(row, col)` waypoint lists under each configured cost model), a binary `N×N` path mask per cost model, and structured metadata (source tile, crop bounds, resolution, cost model, weight, obstacle configuration, split, solver backend, path cost, path length, solver wall-clock, seeds).

### How many instances are there?

The default release emits **14,130 samples** across seven resolutions (64, 128, 256, 512, 1024, 2048, 4096). Per-resolution, per-split counts are recorded in `outputs/paper/results/dataset_statistics.csv` and reported in Table 2 of the paper.

### Does the dataset contain all possible instances or is it a sample?

PASSAGE is a deterministically seeded *sample* from an enumerable parameter space. Without obstacles, the ODD is finite and fully enumerable: every `(tile, crop, resolution, cost model, weight, start, goal)` maps deterministically to a unique sample. Additional samples can be generated at will with `make rebuild`.

### What data does each instance consist of?

- `elevation`: real-valued, normalized against globally calibrated min/max.
- `markers`: binary start/goal channels.
- `obstacles`: binary forbidden-cell mask (optional; the free-terrain variant is `obstacles=0`).
- `path_waypoints_<cost_model>`: ordered `(row, col)` pairs.
- `path_mask_<cost_model>`: binary `N×N`.
- `metadata`: structured provenance (see Croissant `record_set.metadata`).

### Is there a label or target associated with each instance?

Yes — reference paths under each cost model. Paths are *computational annotations* produced by the A* solver (grid backend, Numba-JIT), not human annotations. There are no ambiguous or missing labels by construction.

### Is any information missing from individual instances?

No; every sample is fully populated by the generator. Upstream AW3D30 tiles may carry sensor artifacts (voids, striping); affected samples are flagged in metadata via the source-tile identifier and, where relevant, the elevation calibration bounds.

### Are relationships between individual instances made explicit?

Yes: samples inherit the `source_tile_id` and `split` fields from their source AW3D30 tile. The geographic hold-out split policy (`split_mask` at 1°/pixel) is versioned alongside the generator configuration.

### Are there recommended data splits?

Yes — the default split is 90% train / 2% calibration / 4% validation / 4% test, with a *geographic hold-out* for test (entire 1°×1° source tiles are reserved for test; no source tile appears in both train and test).

### Are there any errors, sources of noise, or redundancies?

The elevation product inherits JAXA ALOS AW3D30 sensor characteristics (DSM error bounds documented upstream). Path annotations are exact under the documented cost model — no stochastic error. Procedural obstacles are abstractions of real hazards and are not validated against real-world obstacle maps.

### Is the dataset self-contained, or does it link to other resources?

Fully self-contained at the export stage. Raw AW3D30 tiles are re-derivable via `make download` but not shipped in the release.

### Does the dataset contain confidential, personally identifiable, sensitive, or offensive information?

No. PASSAGE contains no personal data, no biometric data, and no human subjects. Offensive content is not possible in this domain.

## Collection Process

### How was the data acquired?

Raw elevation tiles are downloaded from the public JAXA ALOS AW3D30 archive (5°×5° packaging) and extracted into 1°×1° sub-tiles at native 30 m/pixel resolution. A global calibration pass computes elevation min/max across the local tile cache. For each requested `(resolution, sample_idx, split)` triple, a deterministic `blake2b` seed drives tile selection, crop placement, start/goal placement, obstacle synthesis, and cost-weight draw. Reference paths are solved with the Numba-JIT grid-backend A* under each configured cost model (see `src/passage/pathfinding_utils.py`).

### What mechanisms or procedures were used to collect the data?

Code-driven pipeline end-to-end (see `Makefile` and `config/config.yaml`). No crowdsourcing, no human annotators, no subject interaction.

### Was any preprocessing, cleaning, or labeling done?

- Elevation normalised against globally calibrated min/max.
- Markers placed subject to obstacle-exclusion and minimum separation.
- Obstacle masks synthesised from log-uniform super-ellipse parameters, rejected if coverage exceeds 30% after up to 50 placement attempts.
- Paths solved with exact A* under the configured cost model and the 8-connected grid (step lengths 30 m and 30√2 m).

### Was the "raw" data saved in addition to the preprocessed data?

Yes — the upstream AW3D30 archive and the calibrated tile cache are persisted locally under `/data/PASSAGE/download/` and `/data/PASSAGE/calibrate/` respectively. Only the processed export (Parquet shards) is distributed.

### Is the software available for the preprocessing?

Yes — the full generator is open-source in this repository (`src/passage/`), with CI coverage and an end-to-end rebuild via `make rebuild`.

## Uses

### Has the dataset been used for any tasks already?

Yes — PASSAGE is the primary benchmark for the downstream `passage-vision` workspace, which reports dense-field segmentation and cost-to-go regression baselines at the 256×256 operating point, plus a 64→4096 resolution-frontier study.

### What (other) tasks could the dataset be used for?

- Benchmarking classical path-planning algorithms (Dijkstra, Theta*, D\*, JPS) on real terrain.
- Training and evaluating graph-neural planners and RL planners in bridge comparisons.
- Studying multi-resolution scaling, geographic distribution shift, and cost-function transfer in dense prediction.
- Reproducible cost-model design studies with physically interpretable terms.

### Is there anything about the composition of the dataset or the way it was collected that might impact future uses?

- Samples are grid-based; they do not support full 3D motion planning or dynamic environments.
- Obstacles are synthetic. Conclusions about real-world obstacle distributions cannot be drawn directly.
- Geographic coverage is bounded by AW3D30 availability.

### Are there tasks for which the dataset should not be used?

- **Operational airborne flight planning or search-and-rescue dispatch**: PASSAGE is an advisory research benchmark. It does not by itself validate safety-critical deployment and does not substitute for EASA/FAA certification processes.
- **Conclusions about weather, airspace, vegetation, urban structures, or regulations**: out of scope.
- **Safety-of-life decisions**: predicted paths are not safety-of-life outputs without an in-the-loop deterministic verifier and fallback solver.

## Distribution

### Will the dataset be distributed to third parties?

Yes — the dataset is distributed publicly via Hugging Face (`https://huggingface.co/datasets/thalesgroup/passage`). For the NeurIPS 2026 anonymous submission, a reviewer-accessible anonymous mirror is provided and documented in the submission package.

### How will it be distributed?

Parquet shards, zstandard-compressed, one file per resolution × split. Accompanied by:

- `croissant.json` (MLCommons 1.1 + RAI 1.0).
- `outputs/export/manifest.sha256` checksum manifest.
- README on Hugging Face mirroring the GitHub README.
- `CITATION.cff`, `CHANGELOG.md`, `LICENSE`, this Datasheet.

### When will it be distributed?

Version 0.1.0 is released with the NeurIPS 2026 submission (May 2026). Future versions follow semantic versioning in `CHANGELOG.md`.

### Will the dataset be distributed under a license? Will it have an associated DOI?

- Code: MIT (see `LICENSE`).
- Dataset: distributed under `DATASET_LICENSE.md`, which documents the inherited JAXA ALOS AW3D30 obligations for the derived artifacts.
- A Zenodo DOI is planned for the v1.0 release (camera-ready gate) and will be recorded in `CITATION.cff`.

### Have any third parties imposed IP-based or other restrictions?

JAXA ALOS AW3D30 usage terms are inherited by the derived elevation channel. Users of PASSAGE must comply with the upstream JAXA terms, including attribution.

### Do any export controls or other regulatory restrictions apply?

No export controls apply; AW3D30 is a public global elevation product.

## Maintenance

### Who is supporting / hosting / maintaining the dataset?

Thales Group open-source and critical AI research contributors. Contact: `oss@thalesgroup.com`; issues and discussions on the GitHub repository.

### How can the owner be contacted?

GitHub Issues and Discussions at <https://github.com/ThalesGroup/passage>, or email `oss@thalesgroup.com`.

### Is there an erratum?

Errata will be tracked in `CHANGELOG.md`. Breaking schema changes bump the minor or major version and invalidate prior manifest checksums.

### Will the dataset be updated?

Yes. Updates are published via Hugging Face with a corresponding semantic-version tag in this repository and an entry in `CHANGELOG.md`.

### If the dataset relates to people: are there applicable limits on the retention of the data associated with the instances?

Not applicable — no human subjects.

### Will older versions of the dataset continue to be supported/hosted/maintained?

Prior releases remain accessible via their Hugging Face dataset-repo commit SHA and the git tag on GitHub. Deprecations are announced in `CHANGELOG.md`. The deprecation policy: one-minor-version notice, with `rai:dataReleaseMaintenancePlan` documenting active supported versions.

### If others want to extend / augment / build on / contribute to the dataset, is there a mechanism for them to do so?

Yes — pull requests are welcomed (see `CONTRIBUTING.md` and `CODE_OF_CONDUCT.md`). Contributions that change data semantics require a Croissant update and a Datasheet-section review.

---

## Lifecycle Statement (NeurIPS 2026 E&D track requirement)

- **Status**: active.
- **Versioning**: semantic; authoritative source is `CITATION.cff` `version` and `croissant.json` `version`.
- **Maintenance owner**: Thales Group open-source research SIG; issues/PRs on GitHub.
- **Deprecation policy**: prior minor version remains supported for at least one subsequent minor release; deprecations announced in `CHANGELOG.md` and the Hugging Face dataset README.
- **Security contact**: see `SECURITY.md`.
