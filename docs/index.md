---
title: PASSAGE
description: Real-terrain benchmark for constrained path planning — NeurIPS 2026, Track on Evaluations and Datasets.
template: home.html
icon: material/home
hide:
  - navigation
  - toc
tagline: "Real terrain, exact reference routes: 4.4 million scenes and 26.4 million route variants for constrained path planning and surrogate modelling."
pins:
  - k: "🤗 Dataset"
    v: "thalesgroup/passage · v1.0"
    link: "https://huggingface.co/datasets/thalesgroup/passage"
  - k: "Data terms"
    v: "JAXA AW3D30"
    link: "https://earth.jaxa.jp/en/data/policy/"
  - k: "Code"
    v: "MIT"
    link: "https://github.com/ThalesGroup/passage/blob/main/LICENSE"
  - k: "Format"
    v: "Parquet · 1.27 TB"
  - k: "Metadata"
    v: "Croissant 1.1 + RAI"
    link: "https://github.com/ThalesGroup/passage/blob/main/croissant.json"
  - k: "Python"
    v: "≥ 3.11"
stats:
  - label: "Core scenes"
    text: "4.4 M"
    count: 4.4
    decimals: 1
    suffix: " M"
  - label: "Exact routes"
    text: "26.4 M"
    count: 26.4
    decimals: 1
    suffix: " M"
  - label: "Grid sizes"
    text: "7"
    count: 7
  - label: "Cost objectives"
    text: "3"
    count: 3
  - label: "Cell size"
    text: "30 m"
    count: 30
    suffix: " m"
---

## :mountain_snow: Why PASSAGE

<p class="pg-lede">Constrained path planning on real terrain underlies routing for
search-and-rescue, robotics and airborne advisory systems, where rescue-helicopter
crews fly through mountain passes, valleys and ridgelines. Existing public
benchmarks cover this only in part: synthetic grids give algorithmic control
without geographic realism, and geographic data give real terrain without
controllable obstacles or shared baselines (the
<a href="#cite-passage">NeurIPS 2026 paper</a>, introduction).</p>

PASSAGE is a dataset and benchmark for constrained path planning on real
terrain: each scene is a square crop of the
[JAXA ALOS World 3D–30 m (AW3D30)](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)
digital surface model with a start cell, a goal cell and procedural obstacles.
Each scene has six route variants: three cost objectives (`elevation`,
`slope_uphill`, `energy`), each on free terrain and with the obstacles blocked.
Each variant carries an exact minimum-cost reference route for the stated graph
(eight-connected, 30 m cells); start and goal cells that the obstacles
disconnect are rejected by the generator. The dataset,
the generator and the `passage-eval` evaluator are publicly released (code under
MIT, data under the JAXA terms), so that classical planners, learned surrogates,
graph networks and reinforcement learning can be scored on the same tasks.

## :compass: From terrain to benchmark

<div class="pipeline-steps" markdown>

<div class="pipeline-step" markdown>
<div class="step-number">1</div>

#### Download

Fetch the AW3D30 tiles (1° × 1°, grouped in 5° × 5° archives), with resume support.
</div>

<div class="pipeline-step" markdown>
<div class="step-number">2</div>

#### Calibrate

Scan every tile for the global elevation range used to normalise the tensors.
</div>

<div class="pipeline-step" markdown>
<div class="step-number">3</div>

#### Generate

Draw scenes, place start, goal and super-ellipse obstacles, solve the exact routes.
</div>

<div class="pipeline-step" markdown>
<div class="step-number">4</div>

#### Compress

Write the full and preview configurations as Parquet shards.
</div>

<div class="pipeline-step" markdown>
<div class="step-number">5</div>

#### Export

Prepare the release tree: card, notebooks, calibration assets and analyses.
</div>

<div class="pipeline-step" markdown>
<div class="step-number">6</div>

#### Upload

Publish the prepared tree to a Hugging Face dataset repository that you name.
</div>

</div>

## :package: PASSAGE v1.0 at a glance

| Item | Value |
| --- | --- |
| Version | 1.0, Hub tag `v1.0` at commit `c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae` |
| Core scenes | 4,400,000 at seven resolutions, from 64×64 to 4096×4096 cells |
| Route variants | 26,400,000, all exact routes (six per scene) |
| Splits | train / calibration / validation / test = 90 / 2 / 4 / 4 % |
| Test geography | South America hold-out; other tiles split by 5°×5° blocks |
| Preview rows | 1,792 (64 per split and resolution), with rendered images |
| Size | about 1.27 TB of Parquet (5,997 shards); 1.31 TB with receipts and notebooks |
| Intended tasks | Classical planning, path-mask segmentation, cost-to-go regression, graph and reinforcement-learning planners |
| Licence | Code: MIT. Data: JAXA AW3D30 terms, see [`DATASET_LICENSE.md`](https://github.com/ThalesGroup/passage/blob/main/DATASET_LICENSE.md) |
| Paper | *PASSAGE: A Real-Terrain Benchmark for Trustworthy Constrained Path Planning*, NeurIPS 2026, Track on Evaluations and Datasets |

Illustrative validation previews are optional and are not included in every source distribution.

## :sparkles: What you get

<div class="feature-grid pg-cols-4" markdown>

<div class="feature-card" markdown>
<span class="feature-icon">🌐</span>

### Seven grid sizes

From 64 × 64 to 4096 × 4096 cells: about 1.9 km to 123 km across at the nominal 30 m.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🏔️</span>

### Real elevation

Every scene is a crop of AW3D30, JAXA's global digital surface model at a nominal 1-arcsecond (about 30 m) sampling ([Takaku et al., 2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-183-2020)).
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🔷</span>

### Procedural obstacles

Super-ellipses of varied shape, size and rotation, up to 30 % of a crop, never on the start or the goal.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">⚡</span>

### Three cost objectives

`elevation`, `slope_uphill` and `energy`, each solved without and with obstacles.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🧮</span>

### Exact references

A Numba-compiled float64 grid A\* labels every variant with an exact optimum; the generator rejects start and goal cells that the obstacles disconnect. A published notebook solves a sample again.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🧭</span>

### Bounded domain

The scene parameters form an operational design domain, a term the paper borrows from the [EASA AI Concept Paper, Issue 02](https://www.easa.europa.eu/en/document-library/general-publications/easa-artificial-intelligence-concept-paper-issue-2); the paper claims no compliance with EASA or FAA guidance.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">✅</span>

### Tested

865 tests with an 80 % coverage gate, deterministic seeding, pre-commit hooks and continuous integration.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🔁</span>

### Resumable

Each stage can run again on its own; input identities are checked before any reuse.
</div>

</div>

## :material-trophy-outline: Leaderboard

{% include-markdown "_partials/leaderboard_main_index.md" %}

## :zap: Quick start

Most users only need the released data. Install `datasets`, `zstandard` and
`numpy`, pin the dataset revision and decode one scene:

```python
import numpy as np
import zstandard
from datasets import load_dataset

REVISION = "c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae"  # verified PASSAGE v1.0 upload

rows = load_dataset("thalesgroup/passage", "256x256_samples",
                    split="validation", revision=REVISION)
row = rows[0]
n = row["metadata"]["resolution"]
x = np.frombuffer(zstandard.ZstdDecompressor().decompress(row["tensor"]),
                  dtype="<f4").reshape(n, n, 3)  # elevation, markers, obstacles
route = np.asarray(row["path_astar_energy_obstacles"])  # ordered [row, col] cells
```

The **Dataset** tab of this site shows how to recover metres, read the route
status and stream the large configurations.

??? example "Rebuild the dataset from the JAXA tiles"

    The generator rebuilds the dataset from the source terrain. This is a large
    job: the AW3D30 inventory holds 23,993 tiles, and the shipped
    `config/config.yaml` asks for 1,000,000 scenes at 256 × 256. Read the
    [user guide](guide/index.md) first. `passage regenerate`, the path whose
    receipts ship with v1.0, accepts `--dry-run` and `--pilot 1` for a trial
    run; the steps below run the same generator one stage at a time.

    ```bash
    # Install uv (if needed), then the locked environment
    curl -LsSf https://astral.sh/uv/install.sh | sh
    git clone --branch develop https://github.com/ThalesGroup/passage.git
    cd passage
    uv sync --frozen

    uv run --frozen passage download                  # Step 1 — fetch JAXA tiles
    uv run --frozen passage calibrate                 # Step 2 — compute elevation range
    uv run --frozen passage generate --resolution 256 # Step 3 — generate scenes and routes
    uv run --frozen passage compress --resolution 256 # Step 4 — write Parquet shards
    uv run --frozen passage export --resolution 256   # Step 5 — prepare the release tree
    uv run --frozen passage upload --repo-id <your-user>/<your-dataset>  # Step 6
    ```

## :microscope: Research uses

| Domain | Use | What PASSAGE provides |
|--------|-----|-----------------------|
| Classical search | A\*, Dijkstra and anytime planners against exact references | Three cost objectives, obstacles, seven grid sizes |
| Reinforcement learning | Terrain navigation agents | Grid observations; exact optimal costs for rewards and regret |
| Graph neural networks | Learned heuristics | Eight-connected graphs from elevation grids, route labels |
| Surrogate models | Fast approximations of optimal routes | 4.4 M scenes and 26.4 M route variants |
| Computer vision | Route prediction from elevation images | (N, N, 3) tensors and route masks |

## :memo: Cite PASSAGE { #cite-passage }

<div class="pg-cite" markdown>

PASSAGE is accepted as a poster at the **40th Conference on Neural Information
Processing Systems (NeurIPS 2026), Track on Evaluations and Datasets**. Please
cite the paper, and name the dataset version you used (PASSAGE v1.0):

```bibtex
@inproceedings{delhomme2026passage,
  title     = {{PASSAGE}: A Real-Terrain Benchmark for Trustworthy Constrained Path Planning},
  author    = {Delhomme, Geoffrey and Trap, Fran{\c{c}}ois and Kaakai, Fateh},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS), Track on Evaluations and Datasets},
  year      = {2026},
  url       = {https://github.com/ThalesGroup/passage}
}
```

</div>

!!! note "Future reference"

    The NeurIPS 2026 proceedings are not published yet. This entry will be
    completed with the proceedings details (volume, pages and paper URL) once
    they are available; the title, authors, venue and year will not change.
    Machine-readable metadata is in
    [`CITATION.cff`](https://github.com/ThalesGroup/passage/blob/main/CITATION.cff).

## :book: Learn more

<div class="feature-grid pg-cols-5" markdown>

<div class="feature-card" markdown>
<span class="feature-icon">📖</span>

### [User guide](guide/index.md)

The pipeline step by step, from installation to the release tree.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🔬</span>

### [Generation process](generation/index.md)

Sampling, cost models, obstacles and the reference solver, with the mathematics.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🏁</span>

### [Benchmark](benchmark/index.md)

The evaluation protocol, the `passage-eval` package and how to evaluate a planner.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">🏗️</span>

### [Code architecture](architecture/index.md)

Modules, data formats and test strategy.
</div>

<div class="feature-card" markdown>
<span class="feature-icon">📚</span>

### [API reference](api/index.md)

Documentation generated from the docstrings.
</div>

</div>

## :handshake: Acknowledgments

- **[JAXA](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm)** for the ALOS World 3D–30m (AW3D30) digital surface model.
- **[Hugging Face](https://huggingface.co/)** for the Datasets library and Hub infrastructure.
- **[Thales Group](https://www.thalesgroup.com/)** for supporting this open research.

## JAXA license and attribution

The PASSAGE code is released under the [MIT License](https://github.com/ThalesGroup/passage/blob/main/LICENSE).
**The elevation data every PASSAGE scene is built on are provided by JAXA under
the [JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/)**;
the dataset is neither MIT-licensed nor public domain. Anyone who uses or
redistributes PASSAGE data, or publishes derivatives of them, must follow those
terms:

!!! warning "JAXA AW3D30 — user obligations"
    1. **Credit JAXA and AW3D30**, including in any redistribution or published
       derivative, with the line: *"The original data used for this product have
       been supplied by JAXA's ALOS World 3D–30m (AW3D30)."*
    2. **Notify JAXA before any commercial use.** JAXA also asks to be notified,
       where possible, of publications that use the data.
    3. **Read the JAXA terms of use** linked above for the authoritative wording.

When you publish results on PASSAGE, we recommend citing the AW3D30 source with the
PASSAGE paper: J. Takaku, T. Tadono, M. Doutsu, F. Ohgushi and H. Kai, "Updates of
'AW3D30' ALOS Global Digital Surface Model with Other Open Access Datasets",
*ISPRS Archives* XLIII-B4-2020, 183–189, 2020,
[doi:10.5194/isprs-archives-XLIII-B4-2020-183-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-183-2020).

A summary of these terms is in
[`DATASET_LICENSE.md`](https://github.com/ThalesGroup/passage/blob/main/DATASET_LICENSE.md);
the JAXA terms of use remain the authoritative text. The **Dataset** tab lists the
third-party elevation sources that AW3D30 uses to fill voids.
