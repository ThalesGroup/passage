---
title: Installation
---

# Installation

This page covers how to set up PASSAGE on your machine, install dependencies, and configure credentials for Hugging Face uploads.

---

## System Requirements

| Requirement | Details |
|-------------|---------|
| Python | 3.11 or later; the repository pins 3.12 in `.python-version`, and the tests run on 3.11 and 3.12 |
| Operating system | Linux. The calibration and release tools use POSIX file locks (`fcntl`), so Windows is not supported |
| Disk space | About 463 GB for the 23,993 AW3D30 DSM tiles of the release, plus room for the scenes you generate. The published Parquet files alone take about 1.27 TB |
| CPU and memory | Generation runs one worker process per CPU by default. Large crops need much more memory per worker: `generate` caps its default at 16 workers for 2048 and 8 for 4096 |

!!! note "OpenCV dependency"
    On headless Linux servers, you may need:
    ```bash
    sudo apt install -y libgl1
    ```

---

## Install with `uv` (Recommended)

[`uv`](https://docs.astral.sh/uv/) is a fast Python package manager. With the
repository's `uv.lock` file, it installs the exact locked versions:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone the repository
git clone https://github.com/ThalesGroup/passage.git
cd passage

# Install the locked runtime environment (passage and passage-eval)
uv sync --frozen

# (Optional) Add the test and lint tools
uv sync --frozen --group dev
```

Each `uv sync` installs exactly what that command names, so combine options in
one call, for example `uv sync --frozen --group dev --extra torch` for the
development tools plus PyTorch.

Run the tools with `uv run passage …`, or activate the environment
(`source .venv/bin/activate`) and call `passage` directly. This guide uses the
short form.

---

## Preview the Documentation Site

The documentation site is built with MkDocs Material. To expose it only on
your local machine:

```bash
make docs-serve
```

Then open <http://127.0.0.1:8000/passage/>.

To build the static site into `public/`, run:

```bash
make pages-local
```

---

## Install with `pip`

If you prefer the standard Python toolchain:

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode (passage and passage-eval)
pip install -e .

# (Optional) Add PyTorch, used by passage-eval for GPU timing
pip install -e ".[torch]"

# (Optional) Add the test and lint tools (pip 25.1 or later)
pip install --group dev
```

The development tools are a dependency group, not an extra, so
`pip install -e ".[dev]"` does not install them. `pip` resolves versions on its
own; use `uv sync --frozen` to reproduce the locked environment.

---

## Verify Installation

```bash
# Check the CLI is available
passage --help
```

You should see ten subcommands:

| Subcommand | What it does |
|------------|--------------|
| `download` | Download AW3D30 archives and track their status ([Step 1](download.md)) |
| `calibrate` | Find the global elevation range and record the tile inventory ([Step 2](calibrate.md)) |
| `generate` | Generate scenes from the DSM tiles ([Step 3](generate.md)) |
| `compress` | Write Parquet shards for the selected resolutions ([Steps 4-6](export.md)) |
| `export` | Prepare the export directory: notebooks, assets, calibration and README |
| `upload` | Upload the prepared export directory to Hugging Face |
| `regenerate` | Generate, export and verify the dataset in bounded, resumable batches; never uploads |
| `qualify` | Authenticate a pinned local snapshot, take a metadata census and audit stored routes |
| `audit` | Audit a released dataset without modifying its files |
| `analyze` | Prepare or execute the release notebooks in a separate staging folder |

The evaluator has its own command, `passage-eval --help`.

---

## Configure Hugging Face Credentials :lock:

The `upload` command can publish your prepared export tree directly to the
[Hugging Face Hub](https://huggingface.co/). It reads `HF_TOKEN` from
`.env.secret` at the repository root; `.env.secret.template` shows the format.
By default it targets `thalesgroup/passage`, so pass `--repo-id` to publish to
your own repository.

=== "`.env.secret` — Private token"

    ```bash
    HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
    ```

!!! warning "Never commit secrets"
    Both `.env` and `.env.secret` are listed in `.gitignore`. Never commit your Hugging Face token to version control.

You can also point the CLI at custom env files:

```bash
passage upload --repo-id <user>/<dataset>
passage upload --repo-id <user>/<dataset> --env-path .env --secret-path .env.secret
```

---

## Project Structure

After cloning, the repository is organized as follows:

```text
passage/
├── config/
│   └── config.yaml           # Configuration that built PASSAGE v1.0
├── src/
│   ├── passage/              # Generator and release tools
│   │   ├── __init__.py       # CLI entry point
│   │   ├── download.py       # Archive download and receipts
│   │   ├── calibrate.py      # Elevation range and tile inventory
│   │   ├── generate.py       # The generate command and shared helpers
│   │   ├── reference.py      # Exact float64 A* reference solver
│   │   ├── export.py         # compress, export and upload
│   │   ├── regenerate.py     # Bounded, resumable full-release run
│   │   ├── dataset.py        # DatasetSource loader for released data
│   │   ├── obstacles.py      # Superellipse obstacles
│   │   └── ...               # Release generator, audit, qualification, plotting
│   └── passage-eval/         # Evaluator, imported as passage_eval
├── tests/                    # Test suite; evaluator tests in tests/passage_eval/
├── notebooks/                # Notebook sources and per-resolution templates
├── assets/                   # Banner, sample images, hold-out mask
├── scripts/                  # Install and documentation helpers
├── docs/                     # This documentation site (with mkdocs.yml)
├── outputs/                  # Generated files (ignored by git)
├── pyproject.toml            # Project metadata and dependencies
└── uv.lock                   # Locked dependency versions
```

---

## Optional: use `passage-eval`

The evaluation contract, `passage-eval` (imported as `passage_eval`), ships in
the same `passage` distribution; its source is in `src/passage-eval/`. The
installs above already include it. It computes the PASSAGE primary metrics,
the statistical-discipline rules, trustworthiness diagnostics and
deployment-footprint utilities for a planner. **You do not need it to generate
the dataset** — only to score a planner on it.

Sanity check:

```bash
uv run --frozen python -c "import passage_eval; print(passage_eval.__version__)"
# -> 1.0.0

# Evaluator tests (needs the dev group)
uv run --frozen pytest tests/passage_eval -q
```

The walkthrough of the public API — scoring, paired bootstrap, three-seed
aggregation and the pairwise decision rule — is at
**[Benchmark / `passage-eval`](../benchmark/passage-eval.md)**.

---

## Next Step

Once installed, proceed to **[Step 1 — Download Tiles](download.md)** to begin acquiring satellite elevation data.
