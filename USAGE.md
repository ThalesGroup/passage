# PASSAGE Usage

PASSAGE provides a small CLI with six main subcommands: `download`, `generate`, `calibrate`, `compress`, `export`, and `upload`.
Run `passage --help` or `passage <subcommand> --help` to see all available options for a subcommand.

For a local end-to-end rebuild, the repository also ships a `Makefile` with `make rebuild RESOLUTION=256` and
`make manifest` targets.

---

## download ✅

Download ALOS DSM tiles and track URL status. By default outputs are written to `outputs/download`.

Examples:

```bash
# Default download
passage download

# Specify an outputs directory and disable resume
passage download --outputs-dirpath outputs/my_download --no-resume

# Parallelize with multiple CPUs
passage download --num-cpus 8
```

Key flags: `--outputs-dirpath`, `--no-resume`, `--num-cpus`, `--log-level`.

---

## generate 🔧

Generate dataset samples from DSM tiles. By default it reads configuration from `config/config.yaml` and uses
`outputs/download/tiles` as the tiles source, `outputs/calibrate/calibrate.json` as the calibration file, and writes to `outputs/dataset`.

Examples:

```bash
# Generate using resolutions defined in config
passage generate

# Override resolutions (generate multiple sizes)
passage generate --resolution 64 128 256

# Generate a fixed number of samples and disable saving PNG images
passage generate --num-samples 10000 --no-images

# Use a specific calibration file
passage generate --calibration-filepath /path/to/calibrate.json

# Force recreation (delete existing outputs and regenerate)
passage generate --force
```

Important flags: `--config` (path to `config.yaml`), `--tiles-dirpath`, `--calibration-filepath`, `--resolution`, `--num-samples`, `--no-images`, `--force`, `--num-cpus`, `--log-level`.

Note: If `--resolution` is not provided, resolutions are taken from `config/config.yaml` (the `resolutions` key). Obstacle generation is controlled by the `obstacles` section in `config/config.yaml` — when enabled, generated samples include an obstacle channel and metadata contains paths with and without obstacles.

Determinism note: `splits.seed` controls split assignment, while `generation.seed` controls per-sample sampling,
crop placement, marker placement, and obstacle synthesis. Sample metadata records both the run seed and the derived
per-sample seed under `metadata.provenance`.

---

## calibrate 📏

Scan DSM tiles to find min/max elevation values (used by generation). Results are written to `outputs/calibrate` by default.

```bash
# Default calibration
passage calibrate

# Recompute everything (disable resume)
passage calibrate --no-resume
```

Key flags: `--tiles-dirpath`, `--outputs-dirpath`, `--no-resume`, `--num-cpus`, `--log-level`.

---

## compress 📦

Generate Parquet shards under `outputs/export`. Existing `<resolution>x<resolution>` and `<resolution>x<resolution>_samples`
directories for the selected resolutions are deleted before new shards are written.

```bash
# Compress every discovered dataset resolution
passage compress

# Compress a single resolution
passage compress --resolution 256

# Compress against a /data-backed dataset tree
passage compress --dataset-dirpath /data/PASSAGE/dataset --calibrate-dirpath /data/PASSAGE/calibrate --resolution 256
```

Useful flags: `--config`, `--dataset-dirpath`, `--calibrate-dirpath`, `--calibration-filepath`, `--export-dirpath`, `--resolution`, `--num-cpus`, `--log-level`.

If both `--calibrate-dirpath` and `--calibration-filepath` are provided to `compress`, the explicit calibration file takes precedence when deriving the export directory.

---

## export 🧾

Refresh the upload-ready export tree. This stage copies assets and calibration files, copies notebooks, executes
`notebooks/_dataset.ipynb` per selected resolution, and rebuilds the root `README.md`.

```bash
# Refresh the full prepared export tree
passage export

# Refresh one prepared resolution
passage export --resolution 256

# Refresh a /data-backed export tree
passage export --dataset-dirpath /data/PASSAGE/dataset --calibrate-dirpath /data/PASSAGE/calibrate --resolution 256
```

Useful flags: `--config`, `--dataset-dirpath`, `--calibrate-dirpath`, `--calibration-filepath`, `--notebooks-dirpath`, `--export-dirpath`, `--resolution`, `--repo-id`, `--max-notebook-retries`, `--notebook-retry-delay`, `--num-cpus`, `--log-level`.

The `export` command expects Parquet shards to already exist for the requested resolutions. Run `passage compress` first.

---

## upload 🚀

Upload the prepared export tree to the Hugging Face Hub. By default, uploads target `thalesgroup/passage` and read
`HF_TOKEN` from `.env.secret` (with optional `.env` loading before that).

```bash
# Upload the full prepared tree to the default dataset repo
passage upload

# Upload a single prepared resolution
passage upload --resolution 256

# Override the target repo explicitly
passage upload --repo-id alice/passage-dataset --env-path .env --secret-path .env.secret
```

Useful flags: `--calibrate-dirpath`, `--calibration-filepath`, `--export-dirpath`, `--resolution`, `--repo-id`, `--env-path`, `--secret-path`, `--private`, `--no-create-repo`, `--num-cpus`, `--log-level`.

After preparing an export, run `make manifest` to write `outputs/export/manifest.sha256`. Repository-level release
metadata also lives in `croissant.json`, `DATASHEET.md`, `CITATION.cff`, and `CHANGELOG.md`.

Uploaded artifacts include:
- dataset parquets under `<resolution>x<resolution>` and `<resolution>x<resolution>_samples`
- generated `<resolution>x<resolution>/dataset.ipynb` notebooks (from `notebooks/_dataset.ipynb`)
- root `README.md`, `assets/`, `calibration/`, and `notebooks/`

---

## Notes & tips 💡

- Default output directories: `outputs/download`, `outputs/dataset`, `outputs/calibrate`.
- See `config/config.yaml` for dataset settings (resolutions, obstacle generation, and other options).
- Use `passage <subcommand> --help` for the full list of options for any subcommand.
- Use `make rebuild RESOLUTION=<n>` when you want a single entrypoint for local download, calibration, generation,
  Parquet compression, export-tree refresh, and checksum manifest refresh.

---

For more details and examples, check the project `README.md` and the `notebooks/` directory.

### Loading the Dataset (see `notebooks/demo.ipynb`)

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob

RESOLUTION = 64
# Load a specific resolution
dataset = load_dataset("thalesgroup/passage", name=f"{RESOLUTION}x{RESOLUTION}")

# Access a sample and decode compressed tensor
sample = dataset["train"][0]
tensor = decode_tensor_blob(sample["tensor"], (RESOLUTION, RESOLUTION, 3))

# Channel meanings:
# - elevation: normalized float32 [0,1] (use sample['metadata']['calibration'] to get real meters)
# - markers: -1 background, 0 start, 1 goal
# - obstacles: binary mask (0 free, 1 obstacle)

# See `notebooks/demo.ipynb` for visualization examples (using `tensor_to_image`).
```
