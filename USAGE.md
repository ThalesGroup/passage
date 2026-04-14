# PASSAGE Usage

PASSAGE provides a small CLI with four main subcommands: `download`, `generate`, `calibrate`, and `export`.
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
`outputs/download/tiles` as the tiles source and writes to `outputs/dataset`.

Examples:

```bash
# Generate using resolutions defined in config
passage generate

# Override resolutions (generate multiple sizes)
passage generate --resolution 64 128 256

# Generate a fixed number of samples and disable saving PNG images
passage generate --num-samples 10000 --no-images

# Force recreation (delete existing outputs and regenerate)
passage generate --force
```

Important flags: `--config` (path to `config.yaml`), `--tiles-dirpath`, `--resolution`, `--num-samples`, `--no-images`, `--force`, `--num-cpus`, `--log-level`.

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

## export 📦

Prepare parquet shards under `outputs/export` and upload the full export folder to the Hugging Face Hub. If `--repo-id` is not
provided, the CLI reads `HF_USER_NAME`/`HF_REPO_NAME` from `.env` and `HF_TOKEN` from `.env.secret` by default.

```bash
# Push export folder with executed per-resolution dataset notebooks (default)
passage export --repo-id alice/passage-dataset

# Dry run: prepare export folder but skip push
passage export --no-upload

# Export a specific resolution without uploading
passage export --resolution 128 --no-upload
```

Useful flags: `--config`, `--dataset-dirpath`, `--calibrate-dirpath`, `--resolution`, `--repo-id`, `--no-exec-notebooks`, `--no-upload`, `--private`, `--no-create-repo`, `--log-level`.

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
  parquet export, and checksum manifest refresh.

---

For more details and examples, check the project `README.md` and the `notebooks/` directory.

### Loading the Dataset (see `notebooks/demo.ipynb`)

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob

RESOLUTION = 64
# Load a specific resolution
dataset = load_dataset("your-username/passage-dataset", name=f"{RESOLUTION}x{RESOLUTION}")

# Access a sample and decode compressed tensor
sample = dataset["train"][0]
tensor = decode_tensor_blob(sample["tensor"], (RESOLUTION, RESOLUTION, 3))

# Channel meanings:
# - elevation: normalized float32 [0,1] (use sample['metadata']['calibration'] to get real meters)
# - markers: -1 background, 0 start, 1 goal
# - obstacles: binary mask (0 free, 1 obstacle)

# See `notebooks/demo.ipynb` for visualization examples (using `tensor_to_image`).
```
