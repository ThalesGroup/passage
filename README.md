# 🌍 PASSAGE — Path Solving Dataset & Generator 🔧

**PASSAGE** (PAth S olving Data S et And G enerator) is a toolkit to build reproducible elevation-based datasets from the JAXA ALOS AW3D30 DSM tiles.
It is targeted at researchers and engineers working on terrain-based pathfinding (A*, RL, GNNs, surrogate models) and provides the full pipeline: download → calibrate → generate → export (push to the Hugging Face Hub).

---

## 🎯 Highlights

- Multi-resolution dataset generation (64 → 4096 px and more)
- Resume support for long-running processes (download / calibrate / generate)
- Configurable obstacle generation using superellipses (rich shapes)
- HF-ready direct push using `datasets` with zstd-compressed shards (no parquet staging)
- Fully tested and geared for reproducible experiments

---

## Quick visual summary

<p align="center">
  <img src="assets/sample.png" alt="Representative Sample" width="640"/>
</p>

This sample shows normalized elevation, start/end **markers**, and procedurally generated **obstacles** with two computed paths (free vs with obstacles).

---

## Install

We recommend using the `uv` package manager for a reproducible environment:

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repo and install dependencies
git clone <repository-url>
cd passage
uv sync            # installs runtime dependencies
uv sync --group dev  # add dev dependencies for testing and linting
```

Alternatively, you can use your system Python and `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
```

---

## Configuration (.env & .env.secret) 🔐

PASSAGE supports uploading prepared exports to Hugging Face. Store credentials in project-root files (never commit secrets):

- `.env` — public-ish settings
  - HF_USER_NAME=your-hf-username
  - HF_REPO_NAME=your-repo-name

- `.env.secret` — private token
  - HF_TOKEN=hf_xxx

Examples:

```bash
# .env
HF_USER_NAME=alice
HF_REPO_NAME=passage-dataset

# .env.secret (keep this private)
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
```

Notes:
- `passage export` reads credentials from `.env` / `.env.secret` by default.
- You can override via `--repo-id username/repo` or pass `--env-path` / `--secret-path` if you store them elsewhere.

---

## Usage (CLI summary)

PASSAGE provides these main subcommands: `download`, `calibrate`, `generate`, `export` (direct HF push).
Run `passage --help` or `passage <subcommand> --help` to see full options for any subcommand. See `USAGE.md` for a longer guide and examples.

### Examples

```bash
# Download tiles (default outputs to outputs/download)
passage download

# Calibrate (scan tiles to compute global min/max elevation)
passage calibrate --no-resume

# Generate datasets (reads config/config.yaml by default)
passage generate --resolution 64 128 256 --num-samples 1000 --num-cpus 8

# Push to Hugging Face (direct) with executed notebooks
passage export --repo-id alice/passage-dataset

# Dry-run export (execute notebooks, skip push)
passage export --no-exec-notebooks --no-upload
```

Important flags include:
- `--outputs-dirpath`, `--tiles-dirpath`, `--config` (for `generate`), `--resolution`, `--num-samples`, `--no-images`, `--force` (for `generate`)
- `--no-resume` (for `download` & `calibrate`), `--num-cpus`, `--log-level`
- `--no-exec-notebooks`, `--max-notebook-retries`, `--no-upload`, `--repo-id`, `--env-path`, `--secret-path` (for `export`)
## Example workflow — end-to-end 🔁

1. Download ALOS tiles:

```bash
passage download --outputs-dirpath outputs/download
```

2. Calibrate elevations across tiles:

```bash
passage calibrate --tiles-dirpath outputs/download/tiles --outputs-dirpath outputs/calibrate
```

3. Generate datasets:

```bash
passage generate --tiles-dirpath outputs/download/tiles --outputs-dirpath outputs/dataset --resolution 256 --num-samples 10000
```

4. Export directly to the Hugging Face Hub (notebooks executed by default). Use `--no-upload` for a dry run that only executes notebooks:

```bash
passage export --dataset-dirpath outputs/dataset --calibrate-dirpath outputs/calibrate --repo-id alice/passage-dataset
# dry-run (no push): passage export --no-upload --no-exec-notebooks
```

---

## Configuration file (`config/config.yaml`)

Core settings live in `config/config.yaml`. Example keys:

```yaml
resolutions: [64, 128, 256]
# num_samples can be a single integer or a mapping per-resolution, e.g.:
# num_samples: {64: 1000, 128: 500, 256: 200}
num_samples: 10
obstacles:
  enabled: true
  ratio_max: 0.30
  ratio_target: 0.12
```

CLI options override config file values.

---

## Notebooks 📓

The `notebooks/` directory contains interactive analysis and documentation including:
- `dataset.ipynb` — full dataset exploration and statistics (executed for each resolution during export unless `--no-exec-notebooks` is passed)
- `demo.ipynb` — quick demo: load a sample for a chosen resolution/split and visualize elevation, obstacles and markers
- `parquet.ipynb` — examples loading parquet shards with `datasets`
- `display.ipynb`, `metrics.ipynb`, `obstacles.ipynb`, `solve.ipynb`, `costmodel.ipynb`

Loading & decoding tensors (from `notebooks/demo.ipynb`):

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob

RESOLUTION = 256
# load the HF dataset for a given resolution
ds = load_dataset("your-username/passage-dataset", name=f"{RESOLUTION}x{RESOLUTION}")
sample = ds["train"][0]

# tensors are stored compressed ('.npy.zst') — decode before use:
tensor = decode_tensor_blob(sample["tensor"], (RESOLUTION, RESOLUTION, 3))

# channels:
# 0: elevation (normalized in [0,1]) — denormalize with metadata 'calibration' if needed
# 1: markers (-1 background, 0 start, 1 goal)
# 2: obstacles (binary mask)
elevation = tensor[:, :, 0]
marker = tensor[:, :, 1]
obstacles = tensor[:, :, 2]
```

Open them locally or view rendered versions in the prepared export on Hugging Face if uploaded.

---

## Output structure

Standard output folders:

```
outputs/
├── download/       # tile archives, download.csv, extracted tiles
├── calibrate/      # calibrate.json and visual stats
└── dataset/        # generated samples (tensors/images/metadata) per resolution
```

Each tensor is stored as `.npy.zst` with shape `(H, W, 3)` — channels: elevation, markers, obstacles.
Metadata is stored alongside as YAML files with tile and path information.

---

## Development & Tests 🧪

Run tests and quality checks:

```bash
uv run pytest
uv run ruff check src/
uv run ruff format src/
```

Use `uv sync --group dev` earlier to ensure dev dependencies are available.

---

## Contributing & License

Contributions welcome — please open issues or pull requests. The project is MIT licensed (see `LICENSE`).

---

## Further reading & resources

- Full CLI help: `passage --help` and `passage <subcommand> --help`
- Usage guide: `USAGE.md` (examples and deeper explanations)
- Notebooks: `notebooks/` for analysis and reproducible documentation

---

If you'd like, I can also add a short `README` snippet that describes how to create `.env` / `.env.secret` safely and add a `env.example` template. Would you like that? ✅
