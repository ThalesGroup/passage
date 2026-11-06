---
title: Dataset card
description: PASSAGE v1.0 on the Hugging Face Hub — contents, loading, graph and label contract, evaluation, splits, generation, notebooks, licence and citation.
icon: simple/huggingface
---

# Dataset card

This guide describes **PASSAGE v1.0**. The data, frozen
[dataset card](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/README.md)
and release files are on the Hugging Face Hub.

## Executive summary

PASSAGE is a dataset and benchmark for constrained path planning on real
terrain: each scene is a square crop of the JAXA ALOS World 3D–30 m (AW3D30)
digital surface model with a start cell, a goal cell and procedural obstacles.
Each scene has six route variants: three cost objectives (`elevation`,
`slope_uphill`, `energy`), each on free terrain and with the obstacles blocked.
Each variant carries an exact minimum-cost reference route for the stated
eight-connected graph, with 30 m cardinal steps by convention. The generator
rejects start and goal pairs that the obstacles disconnect. PASSAGE v1.0
contains 4,400,000 core scenes at seven resolutions, from 64×64 to 4096×4096,
and 26,400,000 reference routes. Its test split is a geographic hold-out
(South America), and the MIT-licensed generator and the `passage-eval`
evaluator are in the [code repository](https://github.com/ThalesGroup/passage).

> **Release receipt.** The
> [publication facts](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/publication-facts.json) record
> 4,400,000 core scenes, 26,400,000 objective/obstacle variants and 1,792
> preview rows; preview rows repeat core scenes and add no training
> observations. Labels are solved in float64 from the same float32 terrain
> that users load. Source crops with missing tiles or nodata are rejected.
> All 26,400,000 variants are exact routes; start and goal cells that the
> obstacles disconnect are rejected by the generator.

The guarantees below concern the computed reference labels. Model scores
require their own declared data, decoder and evaluation conditions.

**How to read this guide.** [At a glance](#at-a-glance) and the
[Quick start](#quick-start) get you to a loaded scene in a few lines. The
next three sections describe the files, what a row holds, and the cost model
and label guarantees behind each route.
[Evaluating a planner](#evaluating-a-planner) gives the six metrics of the
benchmark. [Geography and splits](#geography-and-splits) and
[How the scenes were made](#how-the-scenes-were-made) explain the split
policy and the generation steps. [Notebooks](#notebooks) links the executed
reports you can open on the Hub. The card ends with uses and limitations,
licence terms, maintenance and the citation.

---

## At a glance

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
| Licence | Code: MIT. Data: JAXA AW3D30 terms, see [DATASET_LICENSE.md](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/DATASET_LICENSE.md) |
| Paper | *PASSAGE: A Real-Terrain Benchmark for Trustworthy Constrained Path Planning*, NeurIPS 2026, Track on Evaluations and Datasets |

<p align="center">
  <img src="https://huggingface.co/datasets/thalesgroup/passage/resolve/v1.0/notebook-artifacts/256x256/display/display-256-0000000024.png" alt="Validation scene 0000000024 at 256×256: the three cost objectives, without and with obstacles" width="85%"/>
</p>

*Figure 1. One released scene (validation, 256×256, ID `0000000024`) with its
six stored reference routes. Columns are the three cost objectives; the top
row is free terrain (solid green routes) and the bottom row has the obstacles
blocked (dashed blue routes). Rendered by the executed
[`256x256/display.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/256x256/display.ipynb) notebook.*

---

## Quick start

Install `datasets`, `pillow`, `zstandard` and `numpy` (for example
`pip install datasets pillow zstandard numpy`). The examples use the
`v1.0` release tag to load the data and their release metadata together.

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

Every route converts directly into a path-mask target for segmentation:

```python
path_mask = np.zeros((n, n), dtype=np.uint8)
path_mask[route[:, 0], route[:, 1]] = 1
```

Core configurations are large: `256x256` holds 57 GB and `4096x4096` holds
533 GB. Stream them instead of downloading a whole configuration:

```python
core = load_dataset("thalesgroup/passage", "256x256", split="train",
                    streaming=True, revision=REVISION)
for row in core.take(4):
    print(row["id"], row["metadata"]["split"])
```

The core configurations have no image columns. The `passage` package in the
code repository provides a bounded loader, `passage.dataset.DatasetSource`,
which requires a 40-character commit hash for remote use. To browse rows
without code, open the [dataset viewer](https://huggingface.co/datasets/thalesgroup/passage/viewer);
to see a loaded scene step by step, open [`notebooks/demo.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/demo.ipynb).

---

## Files and configurations

```text
<N>x<N>/<split>/data-*.parquet           core scenes, no images (7 configurations)
<N>x<N>_samples/<split>/data-*.parquet   64 preview rows per split, with images (7 configurations)
<N>x<N>/*.ipynb                          executed analysis notebooks for that resolution
calibration/calibrate.json               elevation range and authenticated tile inventory
release-receipts/                        generation, verification and publication receipts
notebooks/, notebook-artifacts/          notebook sources and their outputs
notebook-export-manifest.json            hashes binding notebooks to their inputs and outputs
assets/                                  card images
```

`<N>` is 64, 128, 256, 512, 1024, 2048 or 4096, and `<split>` is `train`,
`calibration`, `validation` or `test`. Each folder is one Hub configuration;
the default Hub configuration is `1024x1024_samples`, a data preview. The
paper's full benchmark comparison uses `256x256`.

*Table 2. Scenes per configuration and split. The nominal side assumes 30 m
cells; actual ground spacing varies with latitude.*

| Resolution | Nominal side | train | calibration | validation | test | Total | Preview rows per split |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64×64 | 1.92 km | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 128×128 | 3.84 km | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 256×256 | 7.68 km | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 512×512 | 15.36 km | 900,000 | 20,000 | 40,000 | 40,000 | 1,000,000 | 64 |
| 1024×1024 | 30.72 km | 225,000 | 5,000 | 10,000 | 10,000 | 250,000 | 64 |
| 2048×2048 | 61.44 km | 90,000 | 2,000 | 4,000 | 4,000 | 100,000 | 64 |
| 4096×4096 | 122.88 km | 45,000 | 1,000 | 2,000 | 2,000 | 50,000 | 64 |
| **All** | | 3,960,000 | 88,000 | 176,000 | 176,000 | 4,400,000 | 1,792 rows in total |

!!! note "Validation and calibration"

    Use `validation` (`valid` in the paper) for model and recipe selection,
    and `calibration` (`calib` in the paper) for decoder settings. Freeze the
    procedure before evaluation on `test`.

!!! note "Preview rows repeat core scenes"

    Preview rows have the lowest admitted scene IDs of each split. Exclude
    them when counting independent scenes or building a training set.

---

## Columns

*Table 3. Parquet columns. Exact Arrow types are in the `dataset_info` block of
the [Hub dataset card](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/README.md).*

| Column | Type | Content |
| --- | --- | --- |
| `id` | string | Scene index, zero-padded; unique within a configuration and split |
| `tensor` | bytes | zstd-compressed float32 array (N, N, 3): normalised elevation, markers, obstacles |
| `metadata` | struct | Split, calibration, source tiles, crop, markers, obstacles, seeds, solver parameters, and per-variant status and cost |
| `metadata_json` | string | Complete metadata as JSON text |
| `scene_receipt` | string | JSON receipt: seeds, source windows and hashes, attempts, terminal records |
| `path_<solver>_free` | list of [row, col] | Reference route on free terrain |
| `path_<solver>_obstacles` | list of [row, col] | Reference route with obstacles blocked |
| `image_<solver>` | image | Rendered preview, `_samples` configurations only |

`<solver>` is `astar_elevation`, `astar_energy` or `astar_slope` (the
`slope_uphill` objective). Every route joins start to goal, and its status in
`metadata.paths.<solver>.<free|obstacles>.status` is `ok`.

---

## Graph and label contract

The routing graph is eight-connected, and obstacle cells are blocked. A step
from cell \(u\) to cell \(v\) has length \(d_{uv} = 30\) m when it is
orthogonal and \(d_{uv} = 30\sqrt{2}\) m when it is diagonal; a diagonal step
is blocked only when its destination cell is an obstacle, so a route may
pass between two obstacle cells that touch at a corner. Let \(h_u\) and \(h_v\) be
the elevations in metres, \(h_{\min}\) and \(h_{\max}\) the crop's minimum
and maximum, and \(w\) the weight of the objective. The step costs of the
`elevation`, `slope_uphill` and `energy` objectives are, in that order:

$$
\begin{aligned}
c_{\mathrm{elevation}}(u,v) &= d_{uv}\left(1 + w\,\frac{h_v - h_{\min}}{h_{\max} - h_{\min}}\right) \\
c_{\mathrm{slope}}(u,v) &= d_{uv}\left(1 + w\,\frac{\max(0,\,h_v - h_u)}{h_{\max} - h_{\min}}\right) \\
c_{\mathrm{energy}}(u,v) &= d_{uv} + w\,\max(0,\,h_v - h_u)
\end{aligned}
$$

A route \(\pi\) from start to goal costs the sum of its steps, and the
reference label is a route of minimum cost:

$$
C(\pi) = \sum_{(u,v)\in\pi} c(u,v), \qquad c^{\star} = \min_{\pi} C(\pi)
$$

The weights are \(w = 62.4\) for `elevation`, \(w = 8886.2\) for
`slope_uphill` and \(w = 1125.3\) for `energy`. These costs are routing
proxies on a digital surface model that includes canopy and building
surfaces. The graph assumes 30 m cells; the actual ground spacing varies with
latitude. Calibration bounds and objective weights are
stored as float64 in the metadata.

Every released variant has the status `ok`: its path is legal and exact
under the frozen graph specification. The generator rejects start and goal
cells that the obstacles disconnect, so every variant has a route. Each
resolution also has a supervision roster, the list of its route variants.
Tied optima may have different geometry.

The release receipts record six exact routes per scene at every resolution:
6,000,000 at each resolution from 64×64 to 512×512, 1,500,000 at
1024×1024, 600,000 at 2048×2048 and 300,000 at 4096×4096.

> **See it on one task.** [`notebooks/costmodel.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/costmodel.ipynb)
> and [`notebooks/paper/cost_model_path_divergence.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/cost_model_path_divergence.ipynb)
> show how the three objectives change the route on the same terrain.

<details markdown="1">
<summary><strong>Frozen graph specifications (JSON)</strong></summary>

```json
{
  "astar_elevation": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "elevation",
    "cost_weight": 62.4,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  },
  "astar_energy": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "energy",
    "cost_weight": 1125.3,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  },
  "astar_slope": {
    "atol": 1e-06,
    "connectivity": 8,
    "coordinate_order": "row_col",
    "cost_model": "slope_uphill",
    "cost_weight": 8886.2,
    "diagonal_policy": "destination_only",
    "obstacle_policy": "blocked",
    "pixel_size_m": 30.0,
    "rtol": 1e-07
  }
}
```

</details>

---

## Evaluating a planner

The `passage-eval` evaluator in the code repository scores a planner with the
six primary metrics of the paper. The planner receives terrain, obstacles and
endpoints; it never sees the reference route or its cost. The scorer checks
each returned route and recomputes its cost with the step costs above.

**Cost regret.** For task \(i\), let \(\hat{c}_i\) be the cost of the
returned route and \(c_i^{\star}\) the reference cost. A route that does not
connect start to goal, or that crosses an obstacle, receives the censored
cost \(\hat{c}_i = 10^{6}\,c_i^{\star}\), hence a regret of 999,999:

$$
r_i = \frac{\hat{c}_i - c_i^{\star}}{c_i^{\star}}
$$

**Geometry.** Let \(\hat{P}_i\) be the cells of the returned route,
\(P_i^{\star}\) those of the reference route, and \(d(p, P)\) the
Euclidean distance in cells from \(p\) to the nearest cell of \(P\). The
average symmetric surface distance is normalised by the reference length, and
the Hausdorff distance keeps the 95th percentile \(Q_{95}\):

$$
\mathrm{ASSD}_i = \frac{1}{2\,|P_i^{\star}|}\left(\frac{1}{|\hat{P}_i|}\sum_{p\in\hat{P}_i} d(p, P_i^{\star}) + \frac{1}{|P_i^{\star}|}\sum_{q\in P_i^{\star}} d(q, \hat{P}_i)\right)
$$

$$
H_{95,i} = \max\left(Q_{95}\big\{d(p, P_i^{\star})\big\}_{p\in\hat{P}_i},\; Q_{95}\big\{d(q, \hat{P}_i)\big\}_{q\in P_i^{\star}}\right)
$$

*Table 4. The six primary metrics.*

| Metric | Aggregate | Better |
| --- | --- | :-: |
| `cost_regret_median` | Median regret over tasks with an exact reference route | lower |
| `cost_regret_p90` | 90th percentile of the same regrets | lower |
| `path_assd` | Mean ASSD over valid routes (connected and obstacle-free) | lower |
| `path_hausdorff_p95` | Mean 95th-percentile Hausdorff distance over valid routes, in cells | lower |
| `success_rate` | For path scoring, share of tasks whose route has at least two cells, is 8-connected and joins start to goal; for complete requests, share with terminal status `ok` | higher |
| `obstacle_violation_rate` | Share of tasks whose route crosses an obstacle cell | lower |

Report the six values together; no composite score replaces them. A
different route can have zero regret, because tied optima exist. A complete
request is `ok` only if it returns a legal route within the declared budget.
Report results per resolution, objective and obstacle condition. The paper
times the whole procedure from raw arrays in memory to the returned route,
including decoding and fallbacks; independent host scoring is separate.
See the [evaluation protocol](../benchmark/protocol.md) for local runner
boundaries and [paper results](../benchmark/candidates.md) for the full
40,000-scene 256×256 comparison. The
[`passage-eval` notebooks](https://huggingface.co/datasets/thalesgroup/passage/tree/v1.0/notebooks/passage-eval)
show the evaluator on released tasks.

---

## Geography and splits

The [South America hold-out mask](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/assets/hold_out_mask.png)
reserves 1,988 grid cells for `test`. Outside the mask, source tiles are
assigned to `train`, `calibration` or `validation` by seeded 5°×5° blocks.
A crop is admitted only if its source tiles and one-tile guard ring belong
to its split. Each scene receipt records its source windows, including
neighboring tiles and wrapped longitude. Missing source tiles, nodata and
incomplete windows are rejected.

Across the seven resolutions, the paper reports 1,682 AW3D30 tiles used by
test crops (7.0% of the 23,993-tile source inventory) and 20,091 distinct
tiles used by the other splits, with no overlap. These are source-tile counts;
the 90/2/4/4 ratios describe scenes.

<details markdown="1">
<summary><strong>Frozen split policy (JSON)</strong></summary>

```json
{
  "block_degrees": 5,
  "buffer_tiles": 1,
  "holdout_mask_hash": "b35306e3f32969c10608e16f556fe2038930b0496482ca9d4a8a7f5de3350ab0",
  "ratios": {
    "calibration": 0.02,
    "test": 0.04,
    "train": 0.9,
    "validation": 0.04
  },
  "seed": 42,
  "version": "source-tile-blocks-v1"
}
```

</details>

---

## How the scenes were made

1. **Source.** The local inventory matches all 23,993 tiles of JAXA's official
   AW3D30 list (checked on 11 October 2026). Tiles are normalised to 3600×3600
   cells by nearest angular-centre sampling.
2. **Calibration.** The complete calibration range is **-430.0 m to 8767.0 m**.
   [calibrate.json](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/calibration/calibrate.json) holds the source-tile
   inventory, its local SHA-256 identities and the validity policy.
3. **Scene.** Each scene uses its own seed, `blake2b(seed, resolution, index,
   split)`. It draws a centre tile and a target elevation, takes the nearest
   cell as the start, places the crop around it and draws the goal uniformly
   among the other cells. Super-ellipse obstacles cover up to 30 % of the crop
   and avoid both markers. A crop whose obstacles disconnect start and goal is
   rejected, and the scene draws its next candidate crop. A scene may take up
   to 100 attempts.
4. **Labels.** An exact float64 A\* reference solves all six variants from the
   decoded float32 tensor, with no expansion budget.
5. **Release.** Shards are verified against the per-resolution cohort and
   supervision hashes, and 41 executed notebooks are bound in
   `notebook-artifacts/manifest.json`. The notebook solver checks use stated,
   bounded samples; they do not independently re-solve every graph.

Normalisation `nearest-angular-center-v1`, tie policy
`float64-priority-then-row-major-node-strict-improvement-v1` and endpoint
policy `one-distinct-pair-per-accepted-crop-disconnected-endpoints-rejected` are
recorded in [publication-facts.json](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/publication-facts.json).

> **Where the claims come from.** The scientific claims of this card derive
> from [release verification](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/release-verification.json),
> [generation provenance](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/generation-provenance.json),
> [publication facts](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/publication-facts.json) and
> [metadata hashes](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/publication-metadata.json). The receipts
> bind the scientific inputs and the published document hashes. Local file
> paths, user names and host names were redacted from the published receipts,
> calibration files, producer source archive, executed notebooks and one
> Parquet scene receipt; [privacy-redactions.json](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/release-receipts/privacy-redactions.json)
> lists every changed file with its original and published SHA-256. The
> release tag binds the dataset files, the Hub dataset card and the executed notebooks
> (see [Notebooks](#notebooks)). You do not need the receipts to use the data.

The images in `assets/` are illustrations; for the released data, inspect the
`_samples` configurations and the executed `display.ipynb` notebooks listed in
[Notebooks](#notebooks).

---

## Notebooks

The repository ships executed notebooks. Each one states the question it
answers, explains every step, and ends with a summary computed from its own
results. They open on the Hub with their outputs; their figures and a JSON
record of their numbers are also saved under
[`notebook-artifacts/`](https://huggingface.co/datasets/thalesgroup/passage/tree/v1.0/notebook-artifacts).
[`notebook-artifacts/manifest.json`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebook-artifacts/manifest.json) binds
each notebook to its source, the code commit, the data revision, its
parameters and the hashes of its outputs.
[`notebook-export-manifest.json`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebook-export-manifest.json) records
the producer execution evidence; the artifact manifest binds the published notebooks.

*Table 5. Executed reports per resolution.*

| Report | 64 | 128 | 256 | 512 | 1024 | 2048 | 4096 |
| --- | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Census of every scene, map of the splits, relief, route outcomes and an exact re-solve (`dataset`) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/64x64/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/128x128/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/256x256/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/512x512/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/1024x1024/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/2048x2048/dataset.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/4096x4096/dataset.ipynb) |
| One scene: its input channels, six routes and their costs (`display`) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/64x64/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/128x128/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/256x256/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/512x512/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/1024x1024/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/2048x2048/display.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/4096x4096/display.ipynb) |
| How far obstacles and objectives move the optimal route (`metrics`) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/64x64/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/128x128/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/256x256/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/512x512/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/1024x1024/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/2048x2048/metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/4096x4096/metrics.ipynb) |
| Three classical planners scored with `passage-eval` (`passage_eval_metrics`) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/64x64/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/128x128/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/256x256/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/512x512/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/1024x1024/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/2048x2048/passage_eval_metrics.ipynb) | [open](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/4096x4096/passage_eval_metrics.ipynb) |

*Table 6. Topic notebooks.*

| Notebook | What it shows |
| --- | --- |
| [`notebooks/demo.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/demo.ipynb) | Getting started: load a scene, decode it, build a training target |
| [`notebooks/costmodel.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/costmodel.ipynb) | The three objectives: routes, elevation profiles, cost of the wrong objective |
| [`notebooks/obstacles.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/obstacles.ipynb) | Obstacle masks, coverage and detours over the validation split |
| [`notebooks/solve.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/solve.ipynb) | Solving labels again with the exact reference; why the search budget matters |
| [`notebooks/astar_inference_timing.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/astar_inference_timing.ipynb) | Timing a planner fairly: warm-up, repetitions, percentiles |
| [`notebooks/paper/dataset_statistics.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/dataset_statistics.ipynb) | Scenes, route outcomes, route length and search effort across resolutions |
| [`notebooks/paper/dataset_qualification.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/dataset_qualification.ipynb) | Checking the release receipts against the published files |
| [`notebooks/paper/cost_model_path_divergence.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/cost_model_path_divergence.ipynb) | Distance between the optima of two objectives at every resolution |
| [`notebooks/paper/obstacles_showcase.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/obstacles_showcase.ipynb) | One scene per resolution where the obstacles move the route |
| [`notebooks/paper/solver_runtime_scaling.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/paper/solver_runtime_scaling.ipynb) | Search effort and measured solve time across resolutions |
| [`notebooks/passage-eval/01_astar_walkthrough.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/passage-eval/01_astar_walkthrough.ipynb) | Evaluating a planner step by step: the contract and the six metrics |
| [`notebooks/passage-eval/02_score_released_baselines.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/passage-eval/02_score_released_baselines.ipynb) | Classical baselines (A\*, greedy, random walk) on the validation split |
| [`notebooks/passage-eval/03_leaderboard.ipynb`](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/notebooks/passage-eval/03_leaderboard.ipynb) | Recording results in the `passage-eval` format |

> **Rerunning a notebook.** The notebooks run in the frozen environment of
> the [code repository](https://github.com/ThalesGroup/passage)
> (`uv run --frozen jupyter lab`). Each has one parameters cell: set
> `dataset_path` to a local copy of the dataset, or to `None` to read from the
> Hub at the pinned `dataset_revision`. A per-resolution notebook runs at
> another resolution by changing its `resolution` parameter.

---

## Intended uses and limitations

Use PASSAGE to benchmark classical planners (for example Dijkstra and A\*)
and learned planners against exact reference routes, to train routing surrogates
(path-mask segmentation, cost-to-go regression, graph or reinforcement-learning
planners), and to study resolution scaling, geographic transfer and transfer
between cost models.

> **Limitations.**
>
> - Elevation comes from a digital surface model: it includes canopy and
>   buildings and may contain sensor artefacts.
> - Scenes are static 2D grids. They do not cover 3D motion planning or
>   dynamic environments.
> - Obstacles, starts and goals are procedural; they do not reflect real
>   hazard or mission distributions. Starts favour elevation extremes of their
>   tile, and coverage is not proportional to ground area.
> - The costs are routing proxies. They ignore vehicle dynamics, clearance,
>   weather, airspace and regulation.
> - The six variants of a scene, and scenes from the same block, are
>   correlated.
> - PASSAGE is a research benchmark. Do not use it for operational flight
>   planning or other safety-of-life decisions; it does not certify deployment.

> **Personal and sensitive information.** None. The data hold public
> elevation data, procedural obstacles and markers, and computed routes. They
> contain no people, personal data or biometric data.

See the [datasheet](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/DATASHEET.md) and the Responsible AI fields of
[croissant.json](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/croissant.json) for details.

---

## Licence and attribution

- Code (generator, evaluator, notebooks): [MIT](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/LICENSE).
- Data: derived from JAXA AW3D30 and subject to the
  [JAXA Terms of Use of Research Data](https://earth.jaxa.jp/en/data/policy/),
  linked from section 5 of the
  [AW3D30 page](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/aw3d30_e.htm).
  Use, modification and redistribution are free of charge with credit to
  JAXA, including for distributed derivatives; commercial use requires prior
  notification to JAXA, and JAXA asks to be notified, where possible, of
  publications that use the data. The dataset is neither MIT-licensed nor
  public domain. See [DATASET_LICENSE.md](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/DATASET_LICENSE.md).
- Credit line: **The original data used for this product have been supplied by JAXA's ALOS World 3D–30m (AW3D30).**
- Source reference: J. Takaku, T. Tadono, M. Doutsu, F. Ohgushi and H. Kai,
  "Updates of 'AW3D30' ALOS Global Digital Surface Model with Other Open
  Access Datasets", *ISPRS Archives* XLIII-B4-2020, 183–189, 2020,
  [doi:10.5194/isprs-archives-XLIII-B4-2020-183-2020](https://doi.org/10.5194/isprs-archives-XLIII-B4-2020-183-2020).
  Cite it with the PASSAGE paper when you publish results on PASSAGE.
- AW3D30 fills voids with third-party elevation data. JAXA's [AW3D30 Version 4.1 Product Description](https://www.eorc.jaxa.jp/ALOS/en/dataset/aw3d30/data/aw3d30v4.1_product_e_1.0.pdf) (checked on 19 October 2026; Table 1, mask file; Section 4, Related URLs) lists GSI Digital Topographic Map 5 m and 10 m mesh data (Geospatial Information Authority of Japan), SRTM-1 v3 (NASA/JPL), Viewfinder Panoramas DEM (Jonathan de Ferranti), ASTER GDEM v2 and v3 (NASA/METI), ArcticDEM v2 and v3 (NGA/NSF) and ArcticDEM v4, TanDEM-X 90m DEM (DLR), REMA v1.1 (PGC, University of Minnesota) and Copernicus DEM GLO-30 (ESA). For the Copernicus DEM it states: "AW3D30 is produced using Copernicus DEM GLO-30-F © DLR e.V. 2014-2018 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA, all rights reserved". Retain these credits in derived works.

---

## Maintenance and versions

- **Maintainers.** Thales open-source and critical-AI research contributors,
  reachable at oss@thalesgroup.com. Hugging Face hosts the data.
- **Versions.** Any change to the data's meaning creates a new version, with a
  new Hub tag, regenerated receipts, an updated card, datasheet and Croissant
  file, and an entry in the code repository's `CHANGELOG.md`. Earlier versions
  stay available.
- **Errata and deprecations.** They are announced in `CHANGELOG.md` and on
  this card.
- **Contributions.** Use pull requests and issues on the code repository (see
  its `CONTRIBUTING.md`). Security reports follow its `SECURITY.md`.

---

## Citation

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

Machine-readable metadata: [CITATION.cff](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/CITATION.cff) and
[croissant.json](https://huggingface.co/datasets/thalesgroup/passage/blob/v1.0/croissant.json) (Croissant 1.1 with Responsible AI fields).

---

## Authors and contact

Geoffrey Delhomme, François Trap and Fateh Kaakai (Thales). Ask questions or
report issues in the
[Hub discussions](https://huggingface.co/datasets/thalesgroup/passage/discussions)
or on [GitHub](https://github.com/ThalesGroup/passage/issues), or write to
geoffrey.delhomme@thalesgroup.com.
