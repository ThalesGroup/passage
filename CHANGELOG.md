# Changelog

## Unreleased

### Added

- Deterministic per-sample generation seeding via `generation.seed`, with recorded `generation_seed` and `sample_seed` provenance in sample metadata.
- Repository release metadata files: `CITATION.cff`, `DATASHEET.md`, `croissant.json`, and `Makefile`.
- `make manifest` workflow for writing `outputs/export/manifest.sha256` from a prepared export tree.

### Changed

- Corrected public documentation to reflect the default 14,130-sample benchmark size and the current reproducibility contract.
- Exposed deterministic-generation provenance in the export schema used for Hugging Face parquet preparation.
