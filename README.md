<!-- PASSAGE README — renders on both GitHub and Hugging Face -->

<p align="center">
  <img src="assets/banner.png" alt="PASSAGE — Path Solving Dataset & Generator" width="100%"/>
</p>

<h1 align="center">🌍 PASSAGE — Real-Terrain Benchmark & Generator</h1>

<p align="center">
  <strong>Trustworthiness-oriented real-terrain benchmark and generation pipeline for pathfinding and surrogate modeling.</strong>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://github.com/ThalesGroup/passage/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/ci.yml?label=CI&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://github.com/ThalesGroup/passage/actions/workflows/docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/docs.yml?label=docs&logo=materialformkdocs&logoColor=white" alt="Docs"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="DATASET_LICENSE.md"><img src="https://img.shields.io/badge/dataset%20terms-AW3D30%20%2B%20MIT%20code-orange?logo=opensourceinitiative&logoColor=white" alt="Dataset Terms"></a>
  <a href="https://huggingface.co/datasets/thalesgroup/passage"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Dataset-yellow" alt="HuggingFace Dataset"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/package-uv-DE5FE9?logo=uv&logoColor=white" alt="uv"></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="Pre-commit"></a>
</p>

<p align="center">
  <a href="https://thalesgroup.github.io/passage/">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/thalesgroup/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage">💻 GitHub</a> •
  <a href="#-citation">📝 Cite</a>
</p>

---

## Table of Contents

- [Table of Contents](#table-of-contents)
- [Get started](#get-started)
- [Documentation](#documentation)
- [🚁 Why PASSAGE?](#-why-passage)
- [✨ Highlights](#-highlights)
- [🗺️ Visual Summary](#️-visual-summary)
- [🚀 Quick Start](#-quick-start)
- [🔁 Pipeline Overview](#-pipeline-overview)
- [🧭 Reproducibility](#-reproducibility)
- [📦 Loading the Dataset](#-loading-the-dataset)
- [⚙️ Configuration](#️-configuration)
- [📓 Notebooks](#-notebooks)
- [🔬 Research Applications](#-research-applications)
- [📁 Output Structure](#-output-structure)
- [🧪 Development \& Testing](#-development--testing)
- [📝 Citation](#-citation)
- [Contributing](#contributing)
- [Security](#security)
- [💬 Support](#-support)
- [👤 Maintainers](#-maintainers)
- [🙏 Acknowledgments](#-acknowledgments)
- [🧾 Disclosures](#-disclosures)
- [License](#license)

---

## Get started

PASSAGE is an open-source pathfinding dataset and generation toolkit built on real-world elevation data to benchmark and improve terrain-aware routing algorithms.

- **Project goals**: provide reproducible, multi-resolution benchmarks for pathfinding research, ML surrogates, and safety-critical AI evaluation.
- **Sponsoring SIG**: Thales Group open-source and critical AI research contributors.
- **Community contact**: use [GitHub Issues](https://github.com/ThalesGroup/passage/issues), [GitHub Discussions](https://github.com/ThalesGroup/passage/discussions), or email `oss@thalesgroup.com`.

For installation and first run commands, see [🚀 Quick Start](#-quick-start).

## Documentation

Documentation is available at [thalesgroup.github.io/passage](https://thalesgroup.github.io/passage/).

The documentation site is published with GitHub Pages and includes user guides, API references, and architecture notes.

Please also set the repository **About** metadata:
- **Description**: short project purpose statement for PASSAGE.
- **Website**: `https://thalesgroup.github.io/passage/`.

---

## 🚁 Why PASSAGE?

> *Every year, Helicopter Emergency Medical Services (HEMS) crews across Europe perform over 300,000 rescue missions — racing against time through mountain passes, across valleys, and over ridgelines to reach patients in cardiac arrest or severe trauma. For every minute of delay, survival rates drop by 7–10%. The path the helicopter takes matters — yet pathfinding algorithms are still benchmarked on flat synthetic grids.*

**PASSAGE** was created to change that. Built from **real-world satellite elevation data** (JAXA ALOS AW3D30 at 30 m/pixel), it provides the research community with large-scale, multi-resolution pathfinding benchmarks that capture the true complexity of terrain navigation.

We open-source this toolkit to make terrain-routing methods easier to compare, reproduce, and audit under a shared benchmark surface.

---

## ✨ Highlights

<table>
<tr>
  <td align="center" width="25%">🌐<br><strong>Multi-Resolution</strong><br><small>64×64 → 4096×4096 px</small></td>
  <td align="center" width="25%">🏔️<br><strong>Real Elevation</strong><br><small>JAXA ALOS AW3D30 DSM</small></td>
  <td align="center" width="25%">🔷<br><strong>Rich Obstacles</strong><br><small>Superellipse shapes</small></td>
  <td align="center" width="25%">⚡<br><strong>3 A* Cost Variants</strong><br><small><code>elevation</code>, <code>slope_uphill</code>, <code>energy</code></small></td>
</tr>
<tr>
  <td align="center">🧮<br><strong>Grid A* Oracle</strong><br><small>Exact labels + timings</small></td>
  <td align="center">🤗<br><strong>HuggingFace-Ready</strong><br><small>Parquet + one-command upload</small></td>
  <td align="center">🔁<br><strong>Fully Resumable</strong><br><small>No wasted compute</small></td>
  <td align="center">✅<br><strong>Tested & CI/CD</strong><br><small>80%+ coverage, pre-commit</small></td>
</tr>
</table>

<table>
<tr>
  <td align="center"><strong>4.4 M</strong><br><small>Default Samples</small></td>
  <td align="center"><strong>7</strong><br><small>Resolution Levels</small></td>
  <td align="center"><strong>3</strong><br><small>A* Configurations</small></td>
  <td align="center"><strong>1</strong><br><small>Solver Backend</small></td>
  <td align="center"><strong>30 m/px</strong><br><small>Spatial Resolution</small></td>
</tr>
</table>

---

## 🗺️ Visual Summary

<p align="center">
  <img src="assets/sample.png" alt="Representative PASSAGE sample" width="680"/>
</p>

<p align="center"><em>Normalized elevation with start/end markers, procedural superellipse obstacles, and two computed paths (free terrain vs. obstacle-aware).</em></p>

---

## 🚀 Quick Start

> **CPU-only by default.** No GPU is required to download, generate, or evaluate
> the dataset, and `uv sync` does **not** install any proprietary NVIDIA CUDA
> runtime libraries. The optional `passage-eval[torch]` extra is wired to the
> CPU-only PyTorch index — see [🧾 Disclosures](#-disclosures) for the GPU
> install path.

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install
git clone https://github.com/ThalesGroup/passage.git
cd passage && uv sync

# One-command local rebuild (download -> calibrate -> generate -> export -> manifest)
make rebuild RESOLUTION=256

# Or run each stage manually
passage download                       # 1️⃣  Fetch JAXA elevation tiles
passage calibrate                      # 2️⃣  Compute global elevation stats
passage generate --resolution 256      # 3️⃣  Generate pathfinding samples
passage export --no-upload             # 4️⃣  Package as Parquet shards
make manifest                          # 5️⃣  Write outputs/export/manifest.sha256

# Point generate/export at a specific calibration file when needed
passage generate --calibration-filepath /path/to/calibrate.json --resolution 256
passage export --calibration-filepath /path/to/calibrate.json --no-upload
```

<details>
<summary>💡 Alternative: install with pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```
</details>

---

## 🔁 Pipeline Overview

<table>
<tr>
  <td align="center"><strong>1. Download</strong><br><small>JAXA DSM tiles</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>2. Calibrate</strong><br><small>min / max elevation</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>3. Generate</strong><br><small>samples + paths</small></td>
  <td align="center"><strong>→</strong></td>
  <td align="center"><strong>4. Export</strong><br><small>Parquet + HF Hub</small></td>
</tr>
</table>

| Step | Command | What it does |
|:----:|---------|--------------|
| 1 | `passage download` | Download JAXA ALOS AW3D30 5°×5° tile archives with resume |
| 2 | `passage calibrate` | Scan tiles for global min/max elevation |
| 3 | `passage generate` | Create multi-resolution samples with paths, obstacles, tensors |
| 4 | `passage export` | Write Parquet shards, execute notebooks, push to HF Hub |

> 📖 **Full walkthrough:** see the [User Guide](https://thalesgroup.github.io/passage/guide/) or the local [USAGE.md](USAGE.md) reference.

---

## 🧭 Reproducibility

- `splits.seed` controls deterministic split assignment.
- `generation.seed` controls deterministic per-sample tile selection, crop placement, marker placement, and obstacle synthesis.
- Each sample metadata file records `provenance.generation_seed` and `provenance.sample_seed`.
- Runtime timestamps remain informational metadata and are not part of the deterministic sampling contract.
- `make manifest` writes `outputs/export/manifest.sha256` so a released export tree can be checked for completeness.

Repository-root release metadata is tracked in [`CITATION.cff`](CITATION.cff), [`DATASHEET.md`](DATASHEET.md), [`CHANGELOG.md`](CHANGELOG.md), and [`croissant.json`](croissant.json).

---

## 📦 Loading the Dataset

```python
from datasets import load_dataset
from passage.utils import decode_tensor_blob

# Load a specific resolution from Hugging Face
ds = load_dataset("thalesgroup/passage", name="256x256")
sample = ds["train"][0]

# Decode the compressed tensor → (256, 256, 3) numpy array
tensor = decode_tensor_blob(sample["tensor"], (256, 256, 3))
elevation = tensor[:, :, 0]  # Normalized [0, 1]
markers   = tensor[:, :, 1]  # -1 background, 0 start, 1 goal
obstacles = tensor[:, :, 2]  # Binary mask (0/1)

# Access solver paths
path_cols = [k for k in sample if k.startswith("path_")]
print("Solver paths:", path_cols)
```

---

## ⚙️ Configuration

Core settings live in [`config/config.yaml`](config/config.yaml):

```yaml
resolutions: [64, 128, 256, 512, 1024, 2048, 4096]
num_samples:
  64:    1000000
  128:   1000000
  256:   1000000
  512:   1000000
  1024:   250000
  2048:   100000
  4096:    50000
splits:
  ratios: {train: 0.90, calibration: 0.02, validation: 0.04, test: 0.04}
  seed: 42
generation:
  seed: 42
  preview_images:
    enabled: true
    resolutions: [64, 128, 256, 512, 1024, 2048, 4096]
    num_samples_per_split: 64
obstacles:
  enabled: true
  ratio_max: 0.30
  shape:
    n: { distribution: log_uniform, min: 1.0, max: 80.0 }
solvers:
  - name: astar_elevation
    backend: grid
    solver: astar
    cost_model: elevation
    weight: 5878.0
```

CLI options override config values. See the [Configuration Reference](https://thalesgroup.github.io/passage/guide/configuration/) in the documentation.

<details>
<summary>🔐 Hugging Face credentials</summary>

```bash
# .env
HF_USER_NAME=your-hf-username
HF_REPO_NAME=passage-dataset

# .env.secret (never commit!)
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
```
</details>

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| [`demo.ipynb`](notebooks/demo.ipynb) | Quick-start: load, decode, and visualize a sample |
| [`costmodel.ipynb`](notebooks/costmodel.ipynb) | Compare the configured terrain-cost variants on the same terrain |
| [`obstacles.ipynb`](notebooks/obstacles.ipynb) | Explore superellipse obstacle generation |
| [`solve.ipynb`](notebooks/solve.ipynb) | Solver timing and quality comparison |
| `_dataset.ipynb` | Template — auto-executed per resolution during export |

---

## 🔬 Research Applications

| Domain | Use Case |
|--------|----------|
| **Classical AI** | A\*, Dijkstra benchmarking across cost models |
| **Reinforcement Learning** | Grid-based terrain navigation agents |
| **Graph Neural Networks** | Learned heuristics on elevation graphs |
| **Surrogate Models** | Fast path prediction from 4.4M+ multi-resolution terrain-routing examples |
| **Computer Vision** | Path segmentation from elevation images |
| **Certified ML** | ARP 6983 / ED-324 — bounded ODD, deterministic data |

---

## 📁 Output Structure

```
outputs/
├── download/           # Tile archives + download.csv + extracted GeoTIFFs
├── calibrate/          # calibrate.json + elevation maps
├── dataset/            # Per-resolution samples
│   └── 256/
│       └── train/
│           ├── tensors/N045/E010/0000000000.npy.zst
│           ├── metadata/N045/E010/0000000000.yaml
│           ├── images/astar_energy/N045/E010/0000000000.png
│           └── paths/astar_energy/free/N045/E010/0000000000.csv
└── export/             # Hugging Face-ready parquet shards + release checksums
  ├── 256x256/train/data-00000.parquet
  ├── 256x256_samples/train/data-00000.parquet
  ├── README.md
  └── manifest.sha256
```

---

## 🧪 Development & Testing

```bash
# Install dev dependencies
uv sync --group dev

# Run tests (80%+ coverage required)
uv run pytest --cov

# Lint & format
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## 📝 Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff). If you prefer BibTeX, use:

```bibtex
@inproceedings{delhomme2026passage,
  title     = {{PASSAGE}: A Real-Terrain Multi-Resolution Benchmark for
               Trustworthy Constrained Path Planning},
  author    = {Delhomme, Geoffrey and Trap, Fran{\c{c}}ois and Kaakai, Fateh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS),
               Datasets and Benchmarks Track},
  year      = {2026},
  url       = {https://github.com/ThalesGroup/passage}
}
```

---

## Contributing

If you are interested in contributing to PASSAGE, start by reading the [Contributing guide](/CONTRIBUTING.md).

Contributions are welcome; the guide covers development setup, coding standards, pull requests, and issue reporting.

In brief:

```bash
# Fork & clone, then:
uv sync --group dev
uv run pre-commit install
git checkout -b feature/your-feature-name
# ... make changes ...
uv run pytest --cov
uv run ruff check src/ tests/
```

---

## 💬 Support

Need help? Here's where to go:

| Channel | Link |
|---------|------|
| **Documentation** | [thalesgroup.github.io/passage](https://thalesgroup.github.io/passage/) |
| **Bug Reports** | [GitHub Issues](https://github.com/ThalesGroup/passage/issues) |
| **Discussions & Q&A** | [GitHub Discussions](https://github.com/ThalesGroup/passage/discussions) |
| **Email** | geoffrey.delhomme@thalesgroup.com |

Before opening an issue, please search existing issues and discussions to avoid duplicates.

---

## 👤 Maintainers

| Name | Affiliation | Contact |
|------|------------|--------|
| **Geoffrey Delhomme** | [Thales Group](https://www.thalesgroup.com/) | [geoffrey.delhomme@thalesgroup.com](mailto:geoffrey.delhomme@thalesgroup.com) |
| **François Trap** | [Thales Group](https://www.thalesgroup.com/) | [francois.trap@fr.thalesgroup.com](mailto:francois.trap@fr.thalesgroup.com) |

---

## 🙏 Acknowledgments

- **[JAXA](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)** for the ALOS AW3D30 Digital Surface Model
- **[Hugging Face](https://huggingface.co/)** for the Datasets library and Hub infrastructure
- **[Thales Group](https://www.thalesgroup.com/)** for supporting open-source research in safety-critical AI

---

## 🧾 Disclosures

### AI-assistance disclosure

Portions of this repository's source code, documentation, and notebooks were
drafted with the assistance of an AI coding assistant. All AI-generated
material was reviewed, edited, and validated by the human authors listed in
[`CITATION.cff`](CITATION.cff), who take full responsibility for the released
content. No AI-system co-authorship is asserted.

### Third-party Python dependencies

Python packages declared in `pyproject.toml` and resolved in `uv.lock` are
released by their respective upstreams under their own licenses, which may
differ from the MIT license that covers PASSAGE's own source code. Users are
responsible for reviewing and complying with each upstream's terms — in
particular, GPU acceleration via the optional `passage-eval[torch]` extra
relies on PyTorch and, when used with CUDA wheels, on NVIDIA CUDA runtime
libraries (e.g. `nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cudnn`)
distributed under the proprietary
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/),
**not** under MIT. PASSAGE itself does not redistribute these libraries.

### Optional GPU / CUDA support

PASSAGE runs end-to-end on CPU — no GPU is required to download, generate,
or evaluate the dataset. The `uv.lock` shipped with this repository resolves
PyTorch from the official CPU-only index
(`https://download.pytorch.org/whl/cpu`) and contains **no** `nvidia-*` /
`cuda-*` packages. To enable GPU acceleration (only required by the optional
surrogate-baseline tooling under `passage-eval[torch]`), install the CUDA
build of PyTorch from the appropriate NVIDIA index after `uv sync`, for
example:

```bash
uv sync --extra eval                          # installs CPU PyTorch via the eval extra
uv pip install --reinstall \
  --index-url https://download.pytorch.org/whl/cu128 \
  torch                                        # opt in to the CUDA build
```

By doing so you accept the NVIDIA license terms cited above for the CUDA
runtime libraries pulled in transitively.

---

## Security

Security reporting and handling procedures are documented in [SECURITY.md](SECURITY.md).

If you discover a vulnerability, please use private reporting channels described in that file.

---

## License

PASSAGE ships **two distinct licensing surfaces** and you should
recognise which one applies to what you are doing:

| Surface | License | What it covers |
| --- | --- | --- |
| Generator and evaluator code | [MIT License](LICENSE) | every Python file in this repository — the dataset generator (`passage`), the evaluation contract (`passage-eval`), and the documentation source. |
| Generated elevation data | [JAXA AW3D30 terms of use](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm) | every elevation tile that PASSAGE ingests, every sample whose elevation channel is derived from those tiles, and any model whose training data carries that channel. The full text mirrored at release time is in [`DATASET_LICENSE.md`](DATASET_LICENSE.md). |

> [!WARNING]
> **Respect JAXA's licensing terms.** PASSAGE is built on the
> [JAXA ALOS World 3D 30 m DSM (AW3D30)](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm).
> Any user of PASSAGE — or of any artefact derived from it — **must
> respect the JAXA terms of use**:
>
> 1. **Cite the JAXA AW3D30 source** in publications using PASSAGE
>    (recommended: Takaku *et al.*, *"Updates of 'AW3D30' ALOS Global
>    DSM with Other Open Access Datasets,"* ISPRS Archives, 2020).
> 2. **Reproduce the AW3D30 attribution** in any redistribution of
>    the elevation channel or of any artefact directly derived from
>    it.
> 3. **Read and abide by the JAXA terms-of-use page** linked above
>    for the authoritative wording.
>
> We sincerely thank JAXA for openly sharing AW3D30 with the research
> community.

---

<p align="center">
  <a href="https://thalesgroup.github.io/passage/">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/thalesgroup/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage/issues">🐛 Issues</a> •
  <a href="https://github.com/ThalesGroup/passage/discussions">💬 Discussions</a>
</p>
