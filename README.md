# PASSAGE: Path Solving Dataset and Generator

> **PASSAGE** stands for **PA**th **S**olving Data**S**et **A**nd **GE**nerator

PASSAGE is a tool for generating elevation-based datasets from [ALOS AW3D30 DSM](https://www.eorc.jaxa.jp/ALOS/en/aw3d30/) (Digital Surface Model) tiles. It is designed to create training datasets for machine learning models that solve path-finding problems on terrain data.

## Features

- **Download**: Automatically download ALOS AW3D30 DSM tiles from JAXA servers
- **Calibrate**: Scan all tiles to determine global min/max elevation values for normalization
- **Generate**: Create normalized elevation tensor datasets with configurable resolutions
- **Prepare**: Build an HF-ready export with parquet shards per resolution/split
- **Resume Support**: All commands support resuming from interrupted runs
- **Multi-resolution**: Generate datasets at multiple resolutions in a single run
- **Marker Channels**: Each sample includes a marker channel with two random points for path-finding tasks

## Installation

### Python Installation with uv

[uv](https://github.com/astral-sh/uv) is the recommended package manager:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone <repository-url>
cd passage

# Install dependencies and create virtual environment
uv sync

# Install with development dependencies (for testing)
uv sync --group dev

Note: The Hugging Face `datasets` package is included in the project dependencies (used by `notebooks/parquet.ipynb` and HF-related helpers). Run `uv sync` to install it into your environment.
```

## Usage

PASSAGE provides three main commands: `download`, `calibrate`, and `generate`.

### 1. Download Tiles

Download ALOS AW3D30 DSM tiles from JAXA servers:

```bash
# Download all available tiles (default)
uv run passage download

# Custom output directory
uv run passage download --outputs-dirpath /path/to/downloads

# Start fresh (ignore existing progress)
uv run passage download --no-resume

# Use specific number of CPUs
uv run passage download --num-cpus 4
```

### 2. Calibrate

Scan downloaded tiles to find global min/max elevation values:

```bash
# Calibrate using default paths
uv run passage calibrate

# Custom paths
uv run passage calibrate \
    --outputs-dirpath outputs/calibrate \
    --tiles-dirpath outputs/download/tiles

# Recompute all tiles (ignore previous results)
uv run passage calibrate --no-resume
```

This creates `outputs/calibrate/calibrate.json` with:
- Global min/max elevation values
- Per-tile statistics (min, max, lat, lon)
- Elevation visualization maps

### 3. Generate Dataset

Generate normalized elevation datasets from calibrated tiles:

```bash
# Generate with default settings (from config.yaml)
uv run passage generate

# Generate at specific resolution(s)
uv run passage generate --resolution 256
uv run passage generate --resolution 64 128 256 512

# Generate specific number of samples
uv run passage generate --num-samples 1000

# Skip PNG image generation (tensors only)
uv run passage generate --no-images

# Force regeneration (delete existing and start fresh)
uv run passage generate --force

# Sequential processing (useful for debugging)
uv run passage generate --num-cpus 0

# Full example with all options
uv run passage generate \
    --resolution 64 128 \
    --num-samples 500 \
    --tiles-dirpath outputs/download/tiles \
    --outputs-dirpath outputs/dataset \
    --num-cpus 8
```

#### Resume Behavior

By default, `generate` resumes from existing samples:
- Counts valid samples (tensor + metadata + image files)
- Generates only missing samples to reach `--num-samples`
- Logs a warning about potential calibration inconsistency
- Deletes any partial/incomplete sample files

Use `--force` to delete existing directories and regenerate all samples.

### Prepare for Hugging Face (local, no upload)

Generate an HF-friendly export under `outputs/export` with parquet shards per resolution and split:

```bash
# Prepare all resolutions
uv run passage prepare

# Prepare specific resolutions
uv run passage prepare --resolution 256 512 1024
```

This creates `outputs/export/` with:

```
outputs/export/
├── README.md
├── notebooks/                 # Documentation notebooks (copied)
├── 256/
│   ├── dataset.ipynb          # Executed with RESOLUTION=256
│   ├── train/                 # data-00000.parquet, data-00001.parquet, ...
│   ├── calibration/
│   ├── validation/
│   └── test/
└── ...
```

#### Using the prepared parquet shards

```python
from datasets import load_dataset

dataset = load_dataset("username/passage", "256")
print(dataset["train"][0].keys())
```

#### Exporting Notebooks as Documentation

To publish notebooks to Hugging Face, use the `huggingface_hub` Python client or the web UI. For example, use `huggingface_hub` to create a dataset-or-docs repo and upload the `notebooks/` folder.

Notebooks on Hugging Face:
- Are rendered as interactive documentation
- Can be downloaded and run locally
- Provide examples of how to use the dataset

## Configuration

Default settings are in `config/config.yaml`:

```yaml
resolutions:
  - 64
  - 128
  - 256
  - 512
  - 1024
  - 2048
  - 4096
num_samples: 10
```

CLI arguments override these values.

## Output Structure

```
outputs/
├── download/
│   ├── download.csv       # URL status tracking
│   ├── download.png       # Visual status map
│   ├── archives/          # Downloaded zip files (temporary)
│   └── tiles/             # Extracted DSM tiles
├── calibrate/
│   ├── calibrate.json     # Global stats and per-tile data
│   ├── max_elevation_map.png
│   └── ptp_elevation_map.png
└── dataset/
    ├── summary.json       # Generation summary
    ├── 64/
    │   ├── tensors/       # .npy files (H×W×2 float32)
    │   ├── images/        # .png visualization
    │   └── metadata/      # .yaml sample info
    ├── 128/
    │   └── ...
    └── ...
```

### Tensor Format

Each `.npy` tensor is a `(H, W, 3)` float32 array:
- **Channel 0**: Normalized elevation (0-1 range using global min/max)
- **Channel 1**: Marker channel (two pixels set to 1.0, rest 0.0)
- **Channel 2**: Obstacles mask (1.0 for blocked, 0.0 for free)

### Metadata Format

Each `.yaml` metadata file contains:

```yaml
index: 0
resolution: 64
target_elevation: 1073.07
tile_min_elevation: 500.0
tile_max_elevation: 2000.0
crop_min_elevation: 800.0
crop_max_elevation: 1200.0
center:
  lat: 45.123
  lon: 10.456
top_left:
  lat: 45.132
  lon: 10.447
bottom_right:
  lat: 45.114
  lon: 10.465
source_tile:
  lat: 45
  lon: 10
  name: N045E010
markers:
  - {i: 10, j: 20, elevation: 950.5}
  - {i: 50, j: 40, elevation: 1100.2}
paths:
  without_obstacles:
    length: 120
    pixels:
      - {i: 10, j: 20}
      - {i: 11, j: 21}
  with_obstacles:
    length: 135
    pixels:
      - {i: 10, j: 20}
      - {i: 10, j: 21}
obstacles:
  enabled: true
  ratio_max: 0.30
  ratio_target: 0.12
  ratio_actual: 0.11
  size_min_ratio: 0.10
  size_max_ratio: 0.25
  penalty: 100000000.0
timestamp: 2026-01-13T10:30:00
```

### Image Format

PNG images use BGR color mapping:
- **Blue channel**: Elevation (normalized to 0-255)
- **Green channel**: Markers (255 at marker positions, 0 elsewhere)
- **Red channel**: Obstacles and path overlay (obstacles up to 180, path at 255)

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov

# Run tests in parallel
uv run pytest -n auto

# Run specific test markers
uv run pytest -m unit           # Fast unit tests only
uv run pytest -m generate       # Generate module tests
uv run pytest -m download       # Download module tests
uv run pytest -m calibrate      # Calibrate module tests
uv run pytest -m integration    # Integration tests

# Generate HTML coverage report
uv run pytest --cov --cov-report=html

# View test report
open outputs/tests/report.html
```

### Test Markers

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.integration` - Tests requiring external resources
- `@pytest.mark.download` - Download module tests
- `@pytest.mark.calibrate` - Calibrate module tests
- `@pytest.mark.generate` - Generate module tests

### Code Quality

```bash
# Run linter
uv run ruff check src/

# Run formatter
uv run ruff format src/

# Run pre-commit hooks
uv run pre-commit run --all-files
```

## License

See [LICENSE](LICENSE) for details.
