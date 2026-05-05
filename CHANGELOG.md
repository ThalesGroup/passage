# Changelog

## Unreleased

### Added

- Deterministic per-sample generation seeding via `generation.seed`, with recorded `generation_seed` and `sample_seed` provenance in sample metadata.
- Repository release metadata files: `CITATION.cff`, `DATASHEET.md`, `croissant.json`, and `Makefile`.
- `make manifest` workflow for writing `outputs/export/manifest.sha256` from a prepared export tree.
- `DATASET_LICENSE.md` to document the dataset-side JAXA AW3D30 obligations separately from the MIT code license.

### Changed

- Corrected public documentation to reflect the default 4,400,000-sample (≈4.4 M) benchmark size — 1,000,000 per resolution at 64–512, 250,000 at 1024, 100,000 at 2048, 50,000 at 4096 — and the current reproducibility contract.
- Exposed deterministic-generation provenance in the export schema used for Hugging Face parquet preparation.
- Restored `config/config.yaml` to the 4.4 M-sample paper benchmark and aligned preview-image defaults with the sample export settings.
- Made `passage export` stage release metadata files into the export tree, with anonymous submission copies available via `--anonymous`.
