# PASSAGE Dataset Paper Roadmap

This document is the canonical dataset-side roadmap for the PASSAGE paper.
It covers four notebooks, one per dataset-facing artifact the paper needs:

1. `dataset_statistics.ipynb` — App. E (split counts, geographic-coverage
   map).
2. `obstacles_showcase.ipynb` — §3.3 (Operational Design Domain) obstacle
   gallery.
3. `solver_runtime_scaling.ipynb` — App. C (A$^{\star}$ Oracle Solver
   Comparison).
4. `cost_model_path_divergence.ipynb` — App. D (Cost-Model Examples).

Paper-side figures driven by the model-vision program — **Fig. 1
(teaser)**, **Fig. 2 (per-edge-device latency)**, the headline **Table 4**,
and every appendix figure under App. F (ODD Coverage Diagnostics), App. H
(Detailed Vision Experiments), and App. I (Edge-Device Inference) — live
in [`../../vision/paper/`](../../vision/paper/) and are **not** reproduced
here. Do not duplicate them in this workspace.

## Paper Section Mapping

Source of truth:
[`../../paper/neurips/main.tex`](../../paper/neurips/main.tex). Any change
to section ordering, figure / table identifiers, or appendix layout starts
in `main.tex` and propagates here.

| Stage | Notebook | Paper anchor | Figure / Table identifier |
| --- | --- | --- | --- |
| `D0` | `dataset_statistics.ipynb` | App. E (ODD Task-Instance-Space Cardinality) | `tab:split-counts`, `fig:geo-coverage` |
| `D1` | `obstacles_showcase.ipynb` | §3.3 (Operational Design Domain), Table~`tab:odd` | obstacle-envelope visual gallery |
| `D2` | `solver_runtime_scaling.ipynb` | App. C (A$^{\star}$ Oracle Solver Comparison) | `tab:solver-comparison` |
| `D3` | `cost_model_path_divergence.ipynb` | App. D (Cost-Model Examples) | `fig:cost-model-paths`, `tab:cost-model-examples` |

## Stage D0 — Dataset Statistics

- Notebook: `notebooks/dataset_statistics.ipynb`.
- Paper anchors: App. E (`tab:split-counts`, `fig:geo-coverage`); also
  feeds the Datasheet appendix (App. G).
- Owner: dataset generation lead.
- Inputs: one release-ready Parquet shard per resolution, the geographic
  hold-out mask (`passage/config/split_mask.npy`), and the current dataset
  manifest (`passage/outputs/export/manifest.sha256`) produced after
  `passage compress`, `passage export`, and `make manifest`.
- Outputs:
  - `figures/geo_coverage.pdf` (+ `.png`) — `fig:geo-coverage` in
    `main.tex`.
  - `figures/split_counts.csv` — `tab:split-counts` in `main.tex`.
  - `figures/dataset_statistics.manifest.json` — provenance.
- Gate: do not promote new split counts until the release tree and manifest
  have been refreshed for the same dataset snapshot.
- Promotes to: `../../paper/neurips/main.tex` `tab:split-counts`,
  `fig:geo-coverage`, and the Datasheet appendix (App. G).
- Reminder: update Datasheet (`../DATASHEET.md`) and Croissant
  (`../croissant.json`) whenever the numbers change, following the public
  release synchronization rules.

## Stage D1 — Obstacles Showcase

- Notebook: `notebooks/obstacles_showcase.ipynb`.
- Paper anchors: §3.3 (Operational Design Domain), Table~`tab:odd`.
- Owner: obstacle-generation maintainer.
- Inputs: the super-ellipse generator in
  `passage/src/passage/obstacles.py`, a small fixed list of obstacle-config
  seeds, and a reference elevation tile.
- Outputs:
  - `figures/obstacles_showcase.pdf` (+ `.png`).
  - `figures/obstacles_showcase.manifest.json` — seeds and parameters used.
- Promotes to: `../../paper/neurips/main.tex` §3.3 obstacle-envelope
  description and the boundary-axis discussion in §3.3.2.
- Reminder: the original developer walkthrough
  `../notebooks/obstacles.ipynb` stays untouched; this paper-facing
  notebook is a skeleton that imports the same generator.

## Stage D2 — Solver Runtime Scaling

- Notebook: `notebooks/solver_runtime_scaling.ipynb`.
- Paper anchors: App. C (A$^{\star}$ Oracle Solver Comparison) /
  Table~`tab:solver-comparison`.
- Owner: pathfinding backend maintainer.
- Inputs: one representative tile at each resolution
  (`{64, 128, 256, 512, 1024, 2048, 4096}`), the three documented backends
  (`networkx`, `pathfinding`, `passage_grid`), and the three heuristics
  (`euclidean`, `manhattan`, `chebyshev`). Uses
  `helpers.run_astar_bench(...)`.
- Outputs:
  - `figures/solver_runtime_scaling.pdf` (+ `.png`).
  - `figures/solver_runtime_scaling.csv` — per-run wall-clock table.
  - `figures/solver_runtime_scaling.manifest.json` — provenance.
- Promotes to: `../../paper/neurips/main.tex` `tab:solver-comparison` and
  the Numba-A$^{\star}$ feasibility claim at $4096^{2}$.
- Reminder: the original developer walkthrough
  `../notebooks/solve.ipynb` stays untouched; the paper-facing notebook is
  a skeleton that timeboxes the reported runs.

## Stage D3 — Cost-Model Path Divergence

- Notebook: `notebooks/cost_model_path_divergence.ipynb`.
- Paper anchors: App. D (Cost-Model Examples) /
  `fig:cost-model-paths`, `tab:cost-model-examples`.
- Owner: cost-model maintainer.
- Inputs: five `val`-split samples picked to expose distinct
  divergence regimes (one per cost-model outlier plus a three-way
  divergent case and a balanced case) under the three declared cost
  models (`elevation`, `slope_uphill`, `energy`). Reuses
  `helpers.run_astar_bench(...)`.
- Outputs:
  - `figures/cost_model_path_divergence.pdf` (+ `.png`) — populates
    `fig:cost-model-paths`.
  - `figures/cost_model_path_divergence.csv` — pairwise symmetric
    Hausdorff $p_{95}$ table populating `tab:cost-model-examples`.
  - `figures/cost_model_path_divergence.manifest.json` — provenance.
- Promotes to: `../../paper/neurips/main.tex` App. D figure + table and
  the "cost models induce qualitatively different paths" claim.
- Reminder: the original developer walkthrough
  `../notebooks/costmodel.ipynb` stays untouched; the paper-facing
  notebook is a skeleton.

## Claim-To-Evidence Map (Dataset Track)

| Claim | Evidence pointer |
| --- | --- |
| Split counts and 90/2/4/4 partition are reproducible from the release manifest | `passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/split_counts.csv` |
| Geographic hold-out covers the South-American tile inventory | `passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/geo_coverage.pdf` |
| Obstacle generator produces the declared super-ellipse family | `passage/paper/notebooks/obstacles_showcase.ipynb -> passage/paper/figures/obstacles_showcase.pdf` |
| Numba-JIT A$^{\star}$ is the only backend feasible at $4096^{2}$ | `passage/paper/notebooks/solver_runtime_scaling.ipynb -> passage/paper/figures/solver_runtime_scaling.csv` |
| The three physical cost models induce qualitatively different optimal paths | `passage/paper/notebooks/cost_model_path_divergence.ipynb -> passage/paper/figures/cost_model_path_divergence.csv` |

## Global Rules

- Treat `../../paper/neurips/main.tex` as the authoritative source for
  figure and table identifiers; start there when a dataset-facing
  assumption changes.
- Graph-neural-network and reinforcement-learning planner families are
  named in the paper Conclusion and in App. A only as future-work
  targets. Do not introduce GNN / RL notebooks, references, or evidence
  in this workspace.
- Do not edit the developer walkthroughs under `../notebooks/`; clone
  their logic into the paper-facing notebook instead.
- Notebooks must be runnable top-to-bottom in a fresh kernel even on a
  host without the dataset share mounted. Use the graceful-skip patterns
  in `helpers.py`.
