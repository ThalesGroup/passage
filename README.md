<!-- PASSAGE code repository README. The title, tagline, executive summary, key facts and citation match the Hugging Face dataset card. -->

<h1 align="center">🌍 PASSAGE: real-terrain benchmark for constrained path planning</h1>

<p align="center">
  <strong>Real terrain, exact reference routes: 4.4 million scenes and 26.4 million route variants for constrained path planning and surrogate modelling.</strong>
</p>

<!-- Badges -->
<p align="center">
  <a href="https://github.com/ThalesGroup/passage/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/ci.yml?label=CI&logo=githubactions&logoColor=white" alt="CI"></a>
  <a href="https://github.com/ThalesGroup/passage/actions/workflows/docs.yml"><img src="https://img.shields.io/github/actions/workflow/status/ThalesGroup/passage/docs.yml?label=docs&logo=materialformkdocs&logoColor=white" alt="Docs"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-≥3.11-3776AB?logo=python&logoColor=white" alt="Python"></a>
  <a href="DATASET_LICENSE.md"><img src="https://img.shields.io/badge/dataset%20terms-AW3D30%20%2B%20MIT%20code-orange?logo=opensourceinitiative&logoColor=white" alt="Dataset Terms"></a>
  <a href="#-citation"><img src="https://img.shields.io/badge/NeurIPS%202026-Evaluations%20%26%20Datasets-6f42c1" alt="NeurIPS 2026, Track on Evaluations and Datasets"></a>
  <a href="https://huggingface.co/datasets/thalesgroup/passage"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20dataset-v1.0-yellow" alt="Hugging Face dataset, version 1.0"></a>
  <a href="https://docs.astral.sh/ruff/"><img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff"></a>
  <a href="https://docs.astral.sh/uv/"><img src="https://img.shields.io/badge/package-uv-DE5FE9?logo=uv&logoColor=white" alt="uv"></a>
  <a href="https://pre-commit.com/"><img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white" alt="Pre-commit"></a>
</p>

<p align="center">
  <a href="docs/index.md">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/thalesgroup/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage">💻 GitHub</a> •
  <a href="#-citation">📝 Cite</a>
</p>

---

## 📌 Executive summary

PASSAGE is a dataset and benchmark for constrained path planning on real
terrain: each scene is a square crop of the JAXA ALOS World 3D–30 m (AW3D30)
digital surface model with a start cell, a goal cell and procedural obstacles.
Each scene has six route variants: three cost objectives (`elevation`,
`slope_uphill`, `energy`), each on free terrain and with the obstacles blocked.
Each variant carries an exact minimum-cost reference route for the stated
graph (eight-connected, 30 m cells); start and goal cells that the obstacles
disconnect are rejected by the generator. PASSAGE v1.0 holds 4,400,000 core
scenes at seven resolutions, from 64×64 to 4096×4096 cells, and 26,400,000
route variants, all exact routes. Its test split is a geographic hold-out
(South America), and the MIT-licensed generator and the `passage-eval`
evaluator are in the [code repository](https://github.com/ThalesGroup/passage).

---

## 📑 Table of Contents

- [📌 Executive summary](#-executive-summary)
- [🌱 Get started](#-get-started)
- [📖 Documentation](#-documentation)
- [🚁 Why PASSAGE?](#-why-passage)
- [✨ Highlights](#-highlights)
- [🧪 Harness an existing algorithm](#-harness-an-existing-algorithm)
- [🏆 Paper results and leaderboard](#-paper-results-and-leaderboard)
- [🗺️ Visual Summary](#️-visual-summary)
- [📦 The Dataset](#-the-dataset)
- [🚀 Quick Start](#-quick-start)
- [🔁 Pipeline Overview](#-pipeline-overview)
- [🧭 Reproducibility](#-reproducibility)
  - [🔄 Sequential full dataset generation and structural analysis](#-sequential-full-dataset-generation-and-structural-analysis)
- [⚙️ Configuration](#️-configuration)
- [📓 Notebooks](#-notebooks)
- [🔬 Research Applications](#-research-applications)
- [📁 Output Structure](#-output-structure)
- [🧪 Development \& Testing](#-development--testing)
- [📝 Citation](#-citation)
- [🤝 Contributing](#-contributing)
- [🔒 Security](#-security)
- [💬 Support](#-support)
- [👤 Maintainers](#-maintainers)
- [🙏 Acknowledgments](#-acknowledgments)
- [🧾 Disclosures](#-disclosures)
- [📜 License](#-license)

---

## 🌱 Get started

This repository holds the MIT-licensed PASSAGE code: the generator that builds
the dataset from JAXA AW3D30 elevation tiles, and the `passage-eval` evaluator
that scores planners against its reference routes. The dataset itself is on
[Hugging Face](https://huggingface.co/datasets/thalesgroup/passage) under the
JAXA AW3D30 terms.

- **Project goals**: a reproducible, multi-resolution benchmark for path-planning research and learned routing surrogates, with an evaluation protocol designed for research toward trustworthy learned planners.
- **Sponsoring SIG**: Thales Group open-source and critical AI research contributors.
- **Community contact**: use [GitHub Issues](https://github.com/ThalesGroup/passage/issues), [GitHub Discussions](https://github.com/ThalesGroup/passage/discussions), or email `oss@thalesgroup.com`.

For installation and first run commands, see [🚀 Quick Start](#-quick-start).

## 📖 Documentation

The documentation sources are in [`docs/`](docs/index.md): user guides, API
references, architecture notes and the benchmark protocol. They build with
MkDocs (`mkdocs.yml`). [USAGE.md](USAGE.md) summarises the command-line
interface.

---

## 🚁 Why PASSAGE?

Constrained path planning on real terrain underlies off-network routing in
robotics, search and rescue and airborne advisory systems: rescue-helicopter
crews, for example, fly through mountain passes, valleys and ridgelines, where
the route the aircraft takes matters. As the NeurIPS 2026 paper reviews,
existing public benchmarks cover this problem only in part: synthetic grids
offer algorithmic control without geographic realism, while geographic
information system (GIS) resources provide real terrain without controllable
obstacles or standard baselines.

**PASSAGE** combines real terrain from the JAXA ALOS AW3D30 digital surface
model (nominal 1-arcsecond sampling, routed as 30 m cells), procedural
obstacles, three cost objectives, exact reference routes, a geographic hold-out
and a shared evaluator. We open-source the code to make terrain-routing methods
easier to compare, reproduce and audit on the same benchmark.

---

## ✨ Highlights

<table>
<tr>
  <td align="center" width="25%">🌐<br><strong>Multi-Resolution</strong><br><small>64×64 → 4096×4096 cells</small></td>
  <td align="center" width="25%">🏔️<br><strong>Real Elevation</strong><br><small>JAXA ALOS AW3D30 DSM</small></td>
  <td align="center" width="25%">🔷<br><strong>Procedural Obstacles</strong><br><small>Super-ellipse shapes</small></td>
  <td align="center" width="25%">⚡<br><strong>3 Cost Objectives</strong><br><small><code>elevation</code>, <code>slope_uphill</code>, <code>energy</code></small></td>
</tr>
<tr>
  <td align="center">🧮<br><strong>Exact A* Reference</strong><br><small>Exact optimal route for every variant</small></td>
  <td align="center">🤗<br><strong>Hugging Face-Ready</strong><br><small>Parquet + one-command upload</small></td>
  <td align="center">🔁<br><strong>Incremental Resume</strong><br><small>Audit input identity before reuse</small></td>
  <td align="center">✅<br><strong>Tested & CI</strong><br><small>865 tests, 80 % coverage gate, pre-commit</small></td>
</tr>
</table>

<table>
<tr>
  <td align="center"><strong>4.4 M</strong><br><small>Core scenes</small></td>
  <td align="center"><strong>26.4 M</strong><br><small>Exact routes</small></td>
  <td align="center"><strong>7</strong><br><small>Resolutions</small></td>
  <td align="center"><strong>3</strong><br><small>Cost objectives</small></td>
  <td align="center"><strong>30 m</strong><br><small>Nominal cell size</small></td>
</tr>
</table>

---

## 🧪 Harness an existing algorithm

`passage-eval` evaluates classical planners, graph methods, reinforcement-learning
policies and vision models through the same ordinary inputs and ordered output
route. Wrap your existing algorithm; include its preprocessing, decoding, repair
and fallback in the wrapper so the harness measures the complete procedure.
Install the [locked environment](#-quick-start) before running these commands.

### 1. Adapt the ordinary inputs

Expose an importable `module:Class` with `plan(request) -> PlanResult`, or a
`module:function` accepting the same request. A thin adapter can look like this:

```python
from passage_eval.protocol.contracts import PlanRequest, PlanResult
from your_algorithm import solve  # your existing implementation

class MyPlanner:
    def plan(self, request: PlanRequest) -> PlanResult:
        path, work = solve(
            elevation=request.elevation, obstacles=request.obstacles,
            start=request.start, goal=request.goal, graph=request.graph,
            budget=request.budget, deadline=request.deadline_monotonic,
        )
        return PlanResult(path=path, work=work, precision="fp32")
```

Adapt the `solve` call to your algorithm's actual interface and report its actual
precision. Requests contain metre elevation, binary obstacles, endpoints,
graph/objective parameters, allowed metadata and an optional normalized
`(H,W,3)` tensor. They exclude reference paths, optimum costs, reference
reachability and cost-to-go targets. Return integer `(row, column)` cells in
start-to-goal order; coordinates are never rounded or clipped. Report a terminal
status when no route is produced. Existing `Planner.predict_path(scene)` and
tensor callables also work; `inference_format: request` selects this interface.

### 2. Freeze the method and task specification

```bash
uv run --frozen passage-eval init-spec --out outputs/my-method.yaml
```

Edit the generated YAML before running. Declare `inference: my_module:MyPlanner`
and `inference_format: request`; make that module importable in the locked
environment. Set the data root, resolution, objective, split, sample cap, seeds,
decoder, budgets and warm-repeat settings. For a local copy of the dataset, set
`dataset_path` to its root, for example `outputs/release/dataset`. Its core shards must be under
`<N>x<N>/<split>/*.parquet`; `_samples` holds previews. Remote mode instead needs
an explicit `dataset_repo` and immutable 40-character `dataset_revision`.
Missing local data stops execution without a network fallback.

Freeze development/calibration choices before the final test. Learned methods
declare existing checkpoint files, using `inference_kwargs_by_seed` for separately
trained replicas; classical methods need no checkpoint. Entry-point source bytes
are hashed. Use `runtime_bundle_manifest` to include helpers or native binaries
in the recorded procedure identity. A device label records provenance; configure
the algorithm's actual runtime separately.

### 3. Check, run and score the routes that were timed

```bash
uv run --frozen passage-eval conform --spec outputs/my-method.yaml --output outputs/eval/check
uv run --frozen passage-eval run --spec outputs/my-method.yaml --output outputs/eval/candidate
uv run --frozen passage-eval score --run outputs/eval/candidate
```

`conform` checks the analytic graph, ordinary-input isolation, failure penalties,
shared deadlines, stage work and coordinates, plus one admitted candidate request.
It does not certify candidate optimality or device behavior. `run` independently qualifies the
entire selected reference roster before constructing the candidate. With
`reference_policy: released_labels`, defective or censored source references
block admission. A separately named `independent_analytic_reference` cohort
recomputes exact references and retains source-label defects; it changes the
cohort rather than validating old labels.

Each seed preserves `run_manifest.json`, `qualification.json`, terminal
`requests.jsonl` and numeric `paths.npz`. `score` reads those routes, verifies the
admitted inputs and writes `per_sample.npz`, `metrics.json` and
`score_receipt.json` without rerunning inference. It scores cold, warmup and warm
series separately. Keep these artifacts together in a fresh output directory.

### 4. Compare identical tasks and read the report

Create `outputs/baseline.yaml` for a second algorithm on the same data, task roster,
reference policy and frozen repeat settings. For a bounded CPU reference method,
use `inference: passage_eval.baselines.classical:BoundedExactPlanner` with
`inference_format: request`; disclose its work limit as part of the method.
The following example assumes both specifications include seed `1234`:

```bash
uv run --frozen passage-eval conform --spec outputs/baseline.yaml --output outputs/eval/baseline-check
uv run --frozen passage-eval run --spec outputs/baseline.yaml --output outputs/eval/baseline
uv run --frozen passage-eval score --run outputs/eval/baseline
uv run --frozen passage-eval compare --score outputs/eval/candidate/seed-1234 --score outputs/eval/baseline/seed-1234 --output outputs/eval/comparison.json
uv run --frozen passage-eval report --compare outputs/eval/comparison.json --output outputs/eval/report.md
```

Comparison requires matching task, source, input, graph and reference identities.
Repeat per trained seed and report seed variation separately. The report retains
every candidate failure in all-task regret and latency: failed requests receive
cost `1,000,000 × qualified optimum`, giving regret `999,999`. It distinguishes
endpoint connectivity from legal successful routes, reports each method's valid
geometry denominator and compares geometry on the both-valid intersection.
Geometry measures agreement with one selected optimum; tied optima may differ.

### 5. Interpret timing and geographic evidence fairly

The complete-request timer includes input preparation, transfers, inference,
decoding, repair/fallback, legality, physical recosting and completion
synchronization. Reference solving, offline geometry/scoring, persistence and
model construction are separate. `cold` means the first request after model
construction. This runner admits batch size one. Freeze `quality_repeat_index`
before execution; all repeats retain their scored routes and failure durations.
Work budgets require reported counters and cover repair/fallback too. Python
deadlines are cooperative: late returns remain timeouts, and safe drain time is
recorded separately. Qualify synchronization on the actual device before making
device latency claims.

Geographic intervals resample whole declared geographic or source-footprint
blocks, recomputing paired statistics. Missing blocks leave uncertainty unknown;
one block cannot estimate it. `compare --odd-spec outputs/odd-bins.json` accepts a
frozen joint Operational Design Domain (ODD) bin specification and reports missing
support. Occupied bins or seed repetition do not establish independence or a
causal geographic effect. Same-device quality versus latency comparisons require
the same workload and complete procedure boundary.

For a claim about learned prediction, declare `component_control` and compare
direct decoding, bounded repair, the same repair machinery/budget with a
prediction-free guide, exact fallback, and exact/budgeted classical procedures.
The attribution table reports work and fallback and marks incomplete controls;
oracle-guided decoding stays a labelled diagnostic. A learned-versus-exact row
alone does not isolate the prediction's contribution.

`bench --spec outputs/my-method.yaml` remains a convenience entry point that uses
this workflow and writes a `Submission` record. Keep each measurement
campaign in its own results root. See [Evaluate your planner](docs/benchmark/contributing-a-method.md)
for a CPU example, manifest fields, budget counters and report interpretation.

### CPU example from a clean checkout

The checked-in external breadth-first-search adapter demonstrates the complete
workflow without training or downloads. Its two analytic scenes check interfaces
and artifact handling; they do not measure real-terrain performance or device
speed. Run once into an unused `outputs/eval-example` directory:

```bash
uv run --frozen python -m examples.make_fixture
uv run --frozen passage-eval conform --spec examples/external_algorithm.json --output outputs/eval-example/conformance
uv run --frozen passage-eval run --spec examples/external_algorithm.json --output outputs/eval-example/bfs
uv run --frozen passage-eval score --run outputs/eval-example/bfs
uv run --frozen passage-eval run --spec examples/exact_algorithm.json --output outputs/eval-example/exact
uv run --frozen passage-eval score --run outputs/eval-example/exact
uv run --frozen passage-eval compare --score outputs/eval-example/bfs/seed-1234 --score outputs/eval-example/exact/seed-1234 --output outputs/eval-example/comparison.json --bootstrap-resamples 100
uv run --frozen passage-eval report --compare outputs/eval-example/comparison.json --output outputs/eval-example/report.md
```

## 🏆 Paper results and leaderboard

The paper reports two dense-vision finalists at 256² under `elevation`:
`convnextv2_atto+BCE` segmentation and `repvit_m1+Bellman, V_bridge` regression,
compared with the analytic A* oracle. Their per-task results are released under
[`src/passage-eval/results/paper-results/`](src/passage-eval/results/paper-results/README.md).
The [evaluation protocol](docs/benchmark/protocol.md)
explains how `passage-eval` implements the paper's task and metrics.

The [leaderboard](docs/benchmark/leaderboard.md)
(`src/passage-eval/results/leaderboard.csv`) has no entries yet, and no
learned-model result on PASSAGE v1.0 is published yet. Its first entries will be
the same two finalists, retrained on PASSAGE v1.0 and scored once on the test
split with `passage-eval`; their weights, decoders and frozen parameters will be
published with their entries (see [First entries](docs/benchmark/candidates.md)).

## 🗺️ Visual Summary

Illustrative validation previews can be generated from an authorised dataset and presentation assets using the [trajectory style](docs/TRAJECTORY_STYLE.md). They are not included in every source distribution.

---

## 📦 The Dataset

The dataset, **PASSAGE v1.0**, is published on Hugging Face as
[thalesgroup/passage](https://huggingface.co/datasets/thalesgroup/passage). Its
[dataset card](https://huggingface.co/datasets/thalesgroup/passage), the
[datasheet](DATASHEET.md) and [croissant.json](croissant.json) describe it.

*Table 1. PASSAGE v1.0 at a glance.*

| Item | Value |
| --- | --- |
| Version | 1.0, release tag [`v1.0`](https://huggingface.co/datasets/thalesgroup/passage/tree/v1.0) |
| Core scenes | 4,400,000 at seven resolutions, from 64×64 to 4096×4096 cells |
| Route variants | 26,400,000, all exact routes (six per scene) |
| Splits | train / calibration / validation / test = 90 / 2 / 4 / 4 % |
| Test geography | South America hold-out; other tiles split by 5°×5° blocks |
| Preview rows | 1,792 (64 per split and resolution), with rendered images |
| Size | about 1.27 TB of Parquet (5,997 shards); 1.31 TB with receipts and notebooks |
| Intended tasks | Classical planning, path-mask segmentation, cost-to-go regression, graph and reinforcement-learning planners |
| Licence | Code: MIT. Data: JAXA AW3D30 terms, see [DATASET_LICENSE.md](DATASET_LICENSE.md) |
| Paper | *PASSAGE: A Real-Terrain Benchmark for Trustworthy Constrained Path Planning*, NeurIPS 2026, Track on Evaluations and Datasets |

Every scene records its source windows. Splits follow source-tile blocks with a
guard ring, crops that touch missing or nodata tiles and start/goal pairs that
the obstacles disconnect are rejected, and every route variant has an exact
reference label.
Learned results on PASSAGE v1.0 will appear on the
[leaderboard](docs/benchmark/leaderboard.md).

Load preview rows and decode a scene (`pip install datasets pillow zstandard numpy`):

```python
import numpy as np
import zstandard
from datasets import load_dataset

REVISION = "v1.0"  # PASSAGE data and publication metadata

# Preview rows: small, 64 per split, with rendered images.
rows = load_dataset("thalesgroup/passage", "256x256_samples",
                    split="validation", revision=REVISION)
row = rows[0]

# The tensor is zstd-compressed float32 with shape (N, N, 3).
n = row["metadata"]["resolution"]
x = np.frombuffer(zstandard.ZstdDecompressor().decompress(row["tensor"]),
                  dtype="<f4").reshape(n, n, 3)
cal = row["metadata"]["calibration"]
elevation_m = x[..., 0] * (cal["max"] - cal["min"]) + cal["min"]
markers = x[..., 1]    # -1 background, 0 start, 1 goal
obstacles = x[..., 2]  # 1 = blocked cell

# One reference route: ordered [row, col] cells from start to goal.
route = np.asarray(row["path_astar_energy_obstacles"])
outcome = row["metadata"]["paths"]["astar_energy"]["obstacles"]
print(outcome["status"], outcome["cost"], len(route))  # "ok" for every released variant
```

Core configurations (`<N>x<N>`, no image columns) are large: `256x256` holds
57 GB and `4096x4096` holds 533 GB. Stream them with
`load_dataset("thalesgroup/passage", "256x256", split="train", streaming=True, revision=REVISION)`,
or use the bounded loader `passage.dataset.DatasetSource`, which reads a local
snapshot or a Hub source pinned to a 40-character commit hash. Preview rows
(`<N>x<N>_samples`) repeat core scenes; do not add them to a training set.
Fix a method's free settings on the `calibration` split before the single test
pass. The dataset card documents the folder layout, columns, graph and label
contract, split policy and limitations.

---

## 🚀 Quick Start

> **CPU-only by default.** No GPU is required to download, generate, or evaluate
> the dataset, and `uv sync --frozen` installs neither PyTorch nor any NVIDIA
> CUDA package. The optional `torch` extra installs PyTorch, which on Linux
> brings NVIDIA CUDA runtime packages — see [🧾 Disclosures](#-disclosures).

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone & install the locked environment
git clone --branch develop https://github.com/ThalesGroup/passage.git
cd passage
uv sync --frozen

# Invoke the installed entry points from the repository root
uv run --frozen passage --help
uv run --frozen passage-eval --help
```

Generation and publication workflows that write provenance receipts require a
source checkout installed with `uv sync --frozen` or `pip install -e .`. Keep
`src/passage/`, `src/passage-eval/`, `pyproject.toml` and `uv.lock` together in
that checkout so the receipts can identify the producer code and locked
environment. Wheel installations support dataset loading and evaluation.

`uv sync` creates `.venv` but does not activate it. Use `uv run --frozen` for
commands below, or run `source .venv/bin/activate` once in each shell before
invoking `passage` or `passage-eval` directly. See the [sequential reproduction
recipe](#-sequential-full-dataset-generation-and-structural-analysis) for the complete pipeline.
`regenerate` combines bounded generation and verified local export. The individual
`compress` and `export` stages remain available; `upload` handles publication.
`export` has no `--no-upload` option.

<details>
<summary>💡 Alternative: install with pip</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .          # runtime dependencies
pip install --group dev   # development tools (pip 25.1 or later)
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
| 1 | `uv run --frozen passage download` | Acquire JAXA ALOS AW3D30 archives and extract tiles |
| 2 | `uv run --frozen passage calibrate` | Scan tiles for global min/max elevation |
| 3 | `uv run --frozen passage generate` | Create multi-resolution tensors, metadata, paths and previews |
| 4 | `uv run --frozen passage compress` | Write full and preview Parquet configurations |
| 5 | `uv run --frozen passage export` | Prepare local metadata, execute the four analysis notebooks per selected resolution and, at 256×256, the paper notebooks |
| 6 | `uv run --frozen passage upload --repo-id <user>/<dataset>` | Publish a prepared export to Hugging Face when explicitly requested |

`uv run --frozen passage regenerate` orchestrates bounded generation, verified
Parquet export and scratch cleanup. The
sequential recipe below uses that command after download and calibration.
Inspect logs and receipts: a zero exit status alone does not establish stage
success or scientific qualification.

> 📖 **Full walkthrough:** see the [User Guide](docs/guide/index.md) or the [USAGE.md](USAGE.md) reference.

---

## 🧭 Reproducibility

- `splits.seed` controls deterministic split assignment.
- `generation.seed` controls deterministic per-sample tile selection, crop placement, marker placement, and obstacle synthesis.
- Each sample metadata file records `provenance.generation_seed` and `provenance.sample_seed`.
- Runtime timestamps remain informational metadata and are not part of the deterministic sampling contract.
- `make manifest` writes `outputs/export/manifest.sha256` to check bytes of the listed export files; a separate ID/component census establishes completeness.

Repository-root release metadata is tracked in [`CITATION.cff`](CITATION.cff), [`DATASHEET.md`](DATASHEET.md), [`CHANGELOG.md`](CHANGELOG.md), and [`croissant.json`](croissant.json).

### 🔬 Data qualification trace

Follow the [data qualification guide](docs/DATA_QUALIFICATION.md) from official terrain inventory and frozen calibration through transactional generation, verified export, cohort qualification and final release gates.
PASSAGE holds the data evidence and the supervision rosters that training code consumes.

### 🔄 Sequential full dataset generation and structural analysis

`regenerate` builds the dataset sequentially across all seven resolutions. The
default configuration requests **4.4 million core scenes** and
**14 configurations**: seven core `<N>x<N>` configurations plus seven `_samples`
configurations with **64 scenes per split**. Only `_samples` contains preview
images; core shards have no image columns.

```bash
uv run --frozen passage download
uv run --frozen passage calibrate
uv run --frozen passage regenerate
```

The final default root is `outputs/release/dataset`; raw staging uses
`outputs/release/scratch`. For each resolution, persistent workers generate
the admitted scenes, then compress/export them in parallel, verify the shards
and remove the exported raw files. Larger resolutions use storage-limited stages;
generation stages and Parquet shard sizes are independent. Completed final shards
remain available for training. Override the
two roots with `--dataset-dirpath` and `--scratch-dirpath` when using separate
storage volumes. `--num-cpus` is optional and defaults to the integer floor of
75% of CPUs in the process affinity; choose a lower value when memory or I/O
limits concurrency. `--dry-run` inspects
the planned work; `--pilot 1` requests one scene per resolution for a small check.
`--max-raw-bytes` controls how much raw data can be staged before export;
`--min-free-bytes` keeps a disk reserve. Neither command uploads anything.

Labels are solved from decoded persisted float32 terrain under the canonical
graph and recorded with explicit terminal statuses. The source-cell split policy
and the label contract are recorded in the release receipts. Freeze and qualify
the task roster before training or headline evaluation.

Publication is a separate action, and regeneration never publishes to the Hub.
Name the destination repository explicitly when an upload is requested:

```bash
# Publish only when explicitly requested, to a repository you name:
# uv run --frozen passage upload --export-dirpath outputs/release/dataset --repo-id <user>/<dataset>
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
    n: { min: 1.0, max: 80.0, distribution: { type: log_uniform } }
solvers:
  - name: astar_elevation
    backend: grid
    solver: astar
    cost_model: elevation
    weight: 62.4
```

CLI options override config values. See the [Configuration Reference](docs/guide/configuration.md) in the documentation.

<details>
<summary>🔐 Hugging Face credentials</summary>

```bash
# .env.secret (never commit!)
HF_TOKEN=hf_XXXXXXXXXXXXXXXXXXXXXXXX
```

The publication destination is set in the sequential generation recipe above.
</details>

---

## 📓 Notebooks

| Notebook | What it shows |
|---|---|
| [`_dataset.ipynb`](notebooks/_dataset.ipynb) | Census of every scene of one configuration: splits, relief, route outcomes and an exact re-solve |
| [`_display.ipynb`](notebooks/_display.ipynb) | One scene: its input channels, six routes and their costs |
| [`_metrics.ipynb`](notebooks/_metrics.ipynb) | How far obstacles and objectives move the optimal route |
| [`_passage_eval_metrics.ipynb`](notebooks/_passage_eval_metrics.ipynb) | Three classical planners scored with `passage-eval` |
| [`demo.ipynb`](notebooks/demo.ipynb) | Getting started: load a scene, decode it, build a training target |
| [`costmodel.ipynb`](notebooks/costmodel.ipynb) | The three objectives: routes, elevation profiles, cost of the wrong objective |
| [`obstacles.ipynb`](notebooks/obstacles.ipynb) | Obstacle masks, coverage and detours over the validation split |
| [`solve.ipynb`](notebooks/solve.ipynb) | Solving labels again with the exact reference; why the search budget matters |
| [`astar_inference_timing.ipynb`](notebooks/astar_inference_timing.ipynb) | Timing a planner fairly: warm-up, repetitions, percentiles |
| [`passage-eval/`](notebooks/passage-eval) | Evaluating a planner step by step, classical baselines on the validation split, the paper's per-task results with their stated evaluation conditions (`02_paper_results`) and the leaderboard record format |
| [`paper/notebooks/`](paper/notebooks) | Five release-wide reports: dataset statistics, receipt checks, route divergence between objectives, an obstacle showcase, and search effort and solve time across resolutions |

The four underscore templates export as `dataset.ipynb`, `display.ipynb`,
`metrics.ipynb` and `passage_eval_metrics.ipynb` at each of the seven resolutions.
Use a declared local snapshot or explicit pinned Hugging Face inputs. Executed
outputs and their producing identities are recorded in the export manifest.

---

## 🔬 Research Applications

| Domain | Use Case |
|--------|----------|
| **Classical AI** | A\*, Dijkstra benchmarking across cost models |
| **Reinforcement Learning** | Grid-based terrain navigation agents |
| **Graph Neural Networks** | Learned heuristics on elevation graphs |
| **Surrogate Models** | Route prediction trained on 4.4 M scenes (26.4 M exact reference routes) at seven resolutions |
| **Computer Vision** | Path segmentation from elevation images |
| **Evaluation research** | Geographic hold-out, seed stability and ODD-binned reporting with `passage-eval` |

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
├── export/             # Hugging Face-ready parquet shards + release checksums
│   ├── 256x256/train/data-00000.parquet
│   ├── 256x256_samples/train/data-00000.parquet
│   ├── README.md
│   └── manifest.sha256
└── release/dataset/      # `passage regenerate` root, same layout as the published dataset
    ├── 256x256/train/data-256-<first>-<last>-<part>.parquet
    ├── 256x256_samples/train/data-256-<first>-<last>-<part>.parquet
    └── release-receipts/
```

---

## 🧪 Development & Testing

```bash
# Install dev dependencies
uv sync --group dev

# Run the 865 tests (coverage gate: 80 %)
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

PASSAGE is accepted as a poster at the 40th Conference on Neural Information
Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets. Please
cite the paper and name the dataset version you used (PASSAGE v1.0):

```bibtex
@inproceedings{delhomme2026passage,
  title     = {{PASSAGE}: A Real-Terrain Benchmark for Trustworthy Constrained Path Planning},
  author    = {Delhomme, Geoffrey and Trap, Fran{\c{c}}ois and Kaakai, Fateh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026},
  url       = {https://github.com/ThalesGroup/passage}
}
```

The NeurIPS 2026 proceedings are not published yet. This entry will be
completed with the proceedings details (volume, pages and paper URL) once
they are available; the title, authors, venue and year will not change.

Machine-readable metadata: [CITATION.cff](CITATION.cff) and
[croissant.json](croissant.json) (Croissant 1.1 with Responsible AI fields).

---

## 🤝 Contributing

If you are interested in contributing to PASSAGE, start by reading the [Contributing guide](CONTRIBUTING.md).

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
| **Documentation** | [`docs/`](docs/index.md) and [USAGE.md](USAGE.md) |
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

### 📚 Third-party Python dependencies

Python packages declared in `pyproject.toml` and resolved in `uv.lock` are
released by their respective upstreams under their own licenses, which may
differ from the MIT license that covers PASSAGE's own source code. Users are
responsible for reviewing and complying with each upstream's terms — in
particular, the optional `torch` extra (`passage[torch]`) installs PyTorch,
which on Linux depends on NVIDIA CUDA runtime libraries (e.g.
`nvidia-cublas`, `nvidia-cuda-runtime`, `nvidia-cudnn-cu13`) distributed
under the proprietary
[NVIDIA Software License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-software-license-agreement/),
**not** under MIT. PASSAGE itself does not redistribute these libraries.

### ⚡ Optional PyTorch and CUDA

PASSAGE runs end-to-end on CPU — no GPU is required to download, generate,
or evaluate the dataset. The default environment (`uv sync --frozen`)
contains no PyTorch and no `nvidia-*` or `cuda-*` package. PyTorch is used
only by the optional device-timing helpers of `passage-eval`; install it
with the `torch` extra:

```bash
uv sync --frozen --extra torch   # adds the locked PyTorch from PyPI
```

On Linux, the locked PyTorch wheel comes from PyPI with its NVIDIA CUDA
runtime packages, so by installing it you accept the NVIDIA license terms
cited above.

---

## 🔒 Security

Security reporting and handling procedures are documented in [SECURITY.md](SECURITY.md).

If you discover a vulnerability, please use private reporting channels described in that file.

---

## 📜 License

PASSAGE ships **two distinct licensing surfaces** and you should
recognise which one applies to what you are doing:

| Surface | License | What it covers |
| --- | --- | --- |
| Generator and evaluator code | [MIT License](LICENSE) | every Python file in this repository — the dataset generator (`passage`), the evaluation contract (`passage-eval`), and the documentation source. |
| Generated elevation data | [JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/) | AW3D30-derived terrain; [`DATASET_LICENSE.md`](DATASET_LICENSE.md) summarizes the official conditions and links to their full text. Model weights require their own licensing and provenance review. |

> [!WARNING]
> **Respect JAXA's terms of use.** PASSAGE is built on the
> [JAXA ALOS World 3D–30m DSM (AW3D30)](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm),
> whose section 5 links to the
> [JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/).
> Anyone who uses or redistributes PASSAGE data, or publishes derivatives
> of them, must follow those terms:
>
> 1. **Credit JAXA and AW3D30**, including in any redistribution or
>    published derivative, with the line: *"The original data used for
>    this product have been supplied by JAXA's ALOS World 3D–30m
>    (AW3D30)."* Keep the third-party credits listed in
>    [`DATASET_LICENSE.md`](DATASET_LICENSE.md).
> 2. **Notify JAXA before any commercial use.** JAXA also asks to be
>    notified, where possible, of publications that use the data.
> 3. **Read the JAXA terms of use** linked above for the authoritative
>    wording.
>
> When you publish results on PASSAGE, we recommend citing the AW3D30
> source with the PASSAGE paper: J. Takaku, T. Tadono, M. Doutsu,
> F. Ohgushi and H. Kai, "Updates of 'AW3D30' ALOS Global Digital Surface
> Model with Other Open Access Datasets", *ISPRS Archives* XLIII-B4-2020,
> 183–189, 2020,
> [doi:10.5194/isprs-archives-XLIII-B4-2020-183-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-183-2020).
>
> We sincerely thank JAXA for openly sharing AW3D30 with the research
> community.

---

<p align="center">
  <a href="docs/index.md">📖 Documentation</a> •
  <a href="https://huggingface.co/datasets/thalesgroup/passage">🤗 Dataset</a> •
  <a href="https://github.com/ThalesGroup/passage/issues">🐛 Issues</a> •
  <a href="https://github.com/ThalesGroup/passage/discussions">💬 Discussions</a>
</p>
