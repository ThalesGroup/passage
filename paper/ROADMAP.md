# PASSAGE Dataset Paper Roadmap

This document is the canonical dataset-side roadmap for the PASSAGE paper. It
covers four notebooks, one per dataset-facing artifact the paper needs:

1. `dataset_statistics.ipynb` — Sec. 3.3 sample grid (Fig. 2) and split
   counts (Table 2).
2. `obstacles_showcase.ipynb` — Sec. 4.4 / App. A obstacle gallery (Fig. 3).
3. `solver_runtime_scaling.ipynb` — Sec. 5.1 solver scaling (Fig. 4).
4. `cost_model_path_divergence.ipynb` — Sec. 5.2 cost-model divergence
   (Fig. 5).

Paper-side figures driven by the model-vision program — **Fig. 1 (teaser)**
and **Table 5 (paradigm headline at 256x256)** — live in
[`../../passage-vision/paper/`](../../passage-vision/paper/) and are out of
scope here. Do not duplicate them in this workspace.

## Paper Section Mapping

Source of truth: [`../../passage-paper/neurips/skeleton.md`](../../passage-paper/neurips/skeleton.md) (v3).
Any change to section ordering, figure or table identifiers, or appendix
layout starts in the skeleton and propagates here.

| Stage | Notebook | Paper section | Figure / Table | Appendix | Representative `\evidence{}` pointer |
| --- | --- | --- | --- | --- | --- |
| `D0` | `dataset_statistics.ipynb` | Sec. 3.3 (sample grid and splits) | Fig. 2, Table 2 | App. A, App. F | `\evidence{passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/sample_grid.pdf}` and `\evidence{... -> dataset_statistics.csv}` |
| `D1` | `obstacles_showcase.ipynb` | Sec. 4.4 (obstacle generation) | Fig. 3 | App. A | `\evidence{passage/paper/notebooks/obstacles_showcase.ipynb -> passage/paper/figures/obstacles_showcase.pdf}` |
| `D2` | `solver_runtime_scaling.ipynb` | Sec. 5.1 (solver backends) | Fig. 4 | App. A | `\evidence{passage/paper/notebooks/solver_runtime_scaling.ipynb -> passage/paper/figures/solver_runtime_scaling.pdf}` and `\evidence{... -> solver_runtime_scaling.csv}` |
| `D3` | `cost_model_path_divergence.ipynb` | Sec. 5.2 (cost-model divergence) | Fig. 5 | App. A | `\evidence{passage/paper/notebooks/cost_model_path_divergence.ipynb -> passage/paper/figures/cost_model_path_divergence.pdf}` and `\evidence{... -> cost_model_path_divergence.csv}` |

## Stage D0 — Dataset Statistics

- Notebook: `notebooks/dataset_statistics.ipynb`.
- Paper anchors: Sec. 3.3 (Fig. 2 sample grid, Table 2 split counts), App. A
  preview, App. F Datasheet mirror.
- Owner: dataset generation lead.
- Inputs: one release-ready Parquet shard per resolution, the geographic
  hold-out mask (`passage/config/split_mask.npy`), and the current dataset
  manifest (`passage/outputs/export/manifest.sha256`) produced after
  `passage compress`, `passage export`, and `make manifest`.
- Outputs:
  - `figures/sample_grid.pdf` (+ `.png`) — Fig. 2 in `main.tex`.
  - `figures/dataset_statistics.csv` — Table 2 in `main.tex`.
  - `figures/dataset_statistics.manifest.json` — provenance.
- Gate: do not promote new split counts until the release tree and manifest
  have been refreshed for the same dataset snapshot.
- Promotes to: `../../passage-paper/neurips/main.tex` Fig. 2 and Table 2
  cells, and Appendix A / Appendix F text.
- Reminder: update Datasheet (`../DATASHEET.md`) and Croissant
  (`../croissant.json`) whenever the numbers change, following the public
  release synchronization rules.

## Stage D1 — Obstacles Showcase

- Notebook: `notebooks/obstacles_showcase.ipynb`.
- Paper anchors: Sec. 4.4 (Fig. 3), Appendix A gallery.
- Owner: obstacle-generation maintainer.
- Inputs: the super-ellipse generator in
  `passage/src/passage/obstacles.py`, a small fixed list of obstacle-config
  seeds, and a reference elevation tile.
- Outputs:
  - `figures/obstacles_showcase.pdf` (+ `.png`) — Fig. 3 in `main.tex`.
  - `figures/obstacles_showcase.manifest.json` — seeds and parameters used.
- Promotes to: `../../passage-paper/neurips/main.tex` Fig. 3 and Appendix A.
- Reminder: the original developer walkthrough
  `../notebooks/obstacles.ipynb` stays untouched; this paper-facing
  notebook is a skeleton that imports the same generator.

## Stage D2 — Solver Runtime Scaling

- Notebook: `notebooks/solver_runtime_scaling.ipynb`.
- Paper anchors: Sec. 5.1 (Fig. 4), Appendix A.
- Owner: pathfinding backend maintainer.
- Inputs: one representative tile at each resolution
  (`{64, 128, 256, 512, 1024, 2048, 4096}`), the three documented backends
  (`networkx`, `pathfinding`, `passage_grid`), and the three heuristics
  (`euclidean`, `manhattan`, `chebyshev`). Uses
  `helpers.run_astar_bench(...)`.
- Outputs:
  - `figures/solver_runtime_scaling.pdf` (+ `.png`) — Fig. 4 in `main.tex`.
  - `figures/solver_runtime_scaling.csv` — per-run wall-clock table.
  - `figures/solver_runtime_scaling.manifest.json` — provenance.
- Promotes to: `../../passage-paper/neurips/main.tex` Fig. 4 and the
  Numba-A* speedup claim at 4096x4096.
- Reminder: the original developer walkthrough
  `../notebooks/solve.ipynb` stays untouched; the paper-facing notebook is
  a skeleton that timeboxes the reported runs.

## Stage D3 — Cost-Model Path Divergence

- Notebook: `notebooks/cost_model_path_divergence.ipynb`.
- Paper anchors: Sec. 5.2 (Fig. 5), Appendix A.
- Owner: cost-model maintainer.
- Inputs: one reference tile and one fixed `(start, goal)` pair; one run
  per cost model in `{distance, elevation, absolute_elevation,
  slope_uphill, energy}` with a per-model log-uniform weight draw; reuses
  `helpers.run_astar_bench(...)`.
- Outputs:
  - `figures/cost_model_path_divergence.pdf` (+ `.png`) — Fig. 5 in
    `main.tex`.
  - `figures/cost_model_path_divergence.csv` — pairwise normalised
    Hausdorff matrix.
  - `figures/cost_model_path_divergence.manifest.json` — provenance.
- Promotes to: `../../passage-paper/neurips/main.tex` Fig. 5 and the
  "cost models induce qualitatively different paths" claim.
- Reminder: the original developer walkthrough
  `../notebooks/costmodel.ipynb` stays untouched; the paper-facing
  notebook is a skeleton.

## Claim-To-Evidence Map (Dataset Track)

| Claim | Evidence pointer |
| --- | --- |
| Seven-resolution sample grid exists and matches dataset shape | `passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/sample_grid.pdf` |
| Split counts (Table 2) are reproducible from the release manifest | `passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/dataset_statistics.csv` |
| Obstacle generator produces the documented super-ellipse family | `passage/paper/notebooks/obstacles_showcase.ipynb -> passage/paper/figures/obstacles_showcase.pdf` |
| Numba-JIT A* is the only backend feasible at 4096x4096 | `passage/paper/notebooks/solver_runtime_scaling.ipynb -> passage/paper/figures/solver_runtime_scaling.csv` |
| The five physical cost models induce qualitatively different optimal paths | `passage/paper/notebooks/cost_model_path_divergence.ipynb -> passage/paper/figures/cost_model_path_divergence.csv` |

## Global Rules

- Treat the paper skeleton `../../passage-paper/neurips/skeleton.md` as the
  authoritative source for figure and table identifiers; start there when
  a dataset-facing assumption changes.
- Do not introduce GNN or RL notebooks or references in this workspace.
- Do not edit the developer walkthroughs under `../notebooks/`; clone their
  logic into the paper-facing notebook instead.
- Notebooks must be runnable top-to-bottom in a fresh kernel even on a host
  without the dataset share mounted. Use the graceful-skip patterns in
  `helpers.py`.
