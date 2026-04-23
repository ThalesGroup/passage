---
title: Installation
---

# Installation

This page covers how to set up PASSAGE on your machine, install dependencies, and configure credentials for Hugging Face uploads.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.11+ | 3.12 |
| Disk space | 10 GB (subset) | 50+ GB (full tiles) |
| RAM | 4 GB | 16+ GB (large resolutions) |
| CPU cores | 1 | 8+ (multiprocessing) |
| OS | Linux, macOS | Linux |

!!! note "OpenCV dependency"
    On headless Linux servers, you may need:
    ```bash
    sudo apt install -y libgl1
    ```

---

## Install with `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager that ensures reproducible environments:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/ThalesGroup/passage.git
cd passage

# Install runtime dependencies
uv sync

# (Optional) Install dev dependencies for testing & linting
uv sync --group dev
```

---

## Install with `pip`

If you prefer the standard Python toolchain:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode
pip install -e .

# (Optional) With dev dependencies
pip install -e ".[dev]"
```

---

## Verify Installation

```bash
# Check the CLI is available
passage --help
```

You should see output listing the four subcommands: `download`, `calibrate`, `generate`, `export`.

---

## Configure Hugging Face Credentials :lock:

The `upload` command can publish your prepared export tree directly to the [Hugging Face Hub](https://huggingface.co/). By default it targets `thalesgroup/passage` and reads `HF_TOKEN` from `.env.secret`.

=== "`.env.secret` — Private token"

    ```bash
    HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
    ```

!!! warning "Never commit secrets"
    Both `.env` and `.env.secret` are listed in `.gitignore`. Never commit your Hugging Face token to version control.

You can also point the CLI at custom env files or override the repo explicitly:

```bash
passage upload --env-path .env --secret-path .env.secret
passage upload --repo-id username/passage-dataset --env-path .env --secret-path .env.secret
```

---

## Project Structure

After cloning, the repository is organized as follows:

```
passage/
├── config/
│   └── config.yaml          # Main configuration file
├── src/passage/              # Source code
│   ├── __init__.py           # CLI entry point
│   ├── download.py           # Tile download module
│   ├── calibrate.py          # Elevation calibration
│   ├── generate.py           # Dataset generation
│   ├── export.py             # Release pipeline (compress/export/upload)
│   ├── obstacles.py          # Superellipse generation
│   ├── pathfinding_utils.py  # Solver implementations
│   ├── utils.py              # Tensor compression
│   └── visualization.py      # Image rendering
├── tests/                    # Test suites (23 files)
├── notebooks/                # Analysis notebooks
├── assets/                   # Banner, sample images, masks
├── outputs/                  # Generated artifacts (gitignored)
├── mkdocs.yml                # Documentation config
└── pyproject.toml            # Project metadata & dependencies
```

---

## Next Step

Once installed, proceed to **[Step 1 — Download Tiles](download.md)** to begin acquiring satellite elevation data.
