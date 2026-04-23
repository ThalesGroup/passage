# PASSAGE Dataset Paper Workspace

This directory is the dataset-side paper workspace for PASSAGE. It produces the
figures and tables that the NeurIPS 2026 Evaluations and Datasets Track
submission cites when it needs evidence about the dataset itself: sample grid,
obstacle gallery, solver-runtime scaling, cost-model path divergence, and
split counts.

The companion model-side workspace lives at
[`../../passage-vision/paper/`](../../passage-vision/paper/) and carries the
segmentation and regression baseline evidence. The paper source of truth lives
at
[`../../passage-paper/neurips/skeleton.md`](../../passage-paper/neurips/skeleton.md)
and [`../../passage-paper/neurips/main.tex`](../../passage-paper/neurips/main.tex).

## Directory Layout

| Path | Role |
| --- | --- |
| `README.md` | workspace index, paper-section mapping, stage-interface contract. |
| `ROADMAP.md` | four notebook stages, inputs, outputs, evidence pointers, owners. |
| `REFERENCE_TRACKER.md` | dataset-track citation staging. |
| `helpers.py` | shared Python helpers for the four notebooks. |
| `notebooks/` | one notebook per paper study (Fig. 2, Fig. 3, Fig. 4, Fig. 5, Table 2). |
| `figures/` | output artifacts: `<study>.pdf`, `<study>.png`, `<study>.manifest.json`. |

## Paper Section Mapping

Source of truth: `../../passage-paper/neurips/skeleton.md` (v3). Every row
below lines up with an `\evidence{}` tag in
`../../passage-paper/neurips/main.tex`.

| Notebook | Paper section | Figure / Table | Evidence pointer |
| --- | --- | --- | --- |
| `dataset_statistics.ipynb` | Sec. 3.3 | Fig. 2, Table 2 | `passage/paper/notebooks/dataset_statistics.ipynb -> passage/paper/figures/sample_grid.pdf`; `-> dataset_statistics.csv` |
| `obstacles_showcase.ipynb` | Sec. 4.4, App. A | Fig. 3 | `passage/paper/notebooks/obstacles_showcase.ipynb -> passage/paper/figures/obstacles_showcase.pdf` |
| `solver_runtime_scaling.ipynb` | Sec. 5.1 | Fig. 4 | `passage/paper/notebooks/solver_runtime_scaling.ipynb -> passage/paper/figures/solver_runtime_scaling.pdf`; `-> solver_runtime_scaling.csv` |
| `cost_model_path_divergence.ipynb` | Sec. 5.2 | Fig. 5 | `passage/paper/notebooks/cost_model_path_divergence.ipynb -> passage/paper/figures/cost_model_path_divergence.pdf`; `-> cost_model_path_divergence.csv` |

Claim-to-evidence detail is duplicated in `ROADMAP.md` so the notebooks, the
paper, and this workspace cannot drift silently.

## Stage Interface Contract (light)

Each notebook is expected to:

1. Import from `passage.paper.helpers` (the module in this directory) and from
   `passage` itself. Fail gracefully if data is not mounted.
2. Pin every input (resolutions, tile ids, backends, seeds) in a single
   setup cell.
3. Produce a paper-ready artifact under `figures/` with a matching
   `<study>.manifest.json` that records those inputs plus the exact output
   paths.
4. Not mutate state outside `figures/`. Do not write to `../notebooks/` or
   elsewhere in the dataset repository.

## Release Dependency

`dataset_statistics.ipynb` is the release-gated notebook in this workspace.
Before promoting Fig. 2, Table 2, Datasheet, or Croissant numbers into the
paper, refresh the dataset release surfaces and checksum them:

```bash
passage compress
passage export
make manifest
```

Treat `../outputs/export/manifest.sha256` as the checksum anchor for the
current release tree referenced by `dataset_statistics.ipynb`.

## Quickstart

From a fresh clone, on a host with the `/data/PASSAGE` share mounted:

```bash
cd passage
pip install -e '.[dev]'
jupyter lab paper/notebooks/
```

Then run, in any order:

- `dataset_statistics.ipynb` — produces `figures/sample_grid.pdf` and
  `figures/dataset_statistics.csv`.
- `obstacles_showcase.ipynb` — produces `figures/obstacles_showcase.pdf`.
- `solver_runtime_scaling.ipynb` — produces
  `figures/solver_runtime_scaling.pdf` and
  `figures/solver_runtime_scaling.csv`.
- `cost_model_path_divergence.ipynb` — produces
  `figures/cost_model_path_divergence.pdf` and
  `figures/cost_model_path_divergence.csv`.

Every notebook is runnable top-to-bottom in a fresh kernel even without the
real data share: the setup cell detects missing inputs and short-circuits to
a stub manifest so the artifact contract is still honoured.

## Cross-Links

- Dataset sheet (Appendix F): [`../DATASHEET.md`](../DATASHEET.md).
- Croissant metadata (Appendix G): [`../croissant.json`](../croissant.json).
- Paper skeleton: [`../../passage-paper/neurips/skeleton.md`](../../passage-paper/neurips/skeleton.md).
- Paper source: [`../../passage-paper/neurips/main.tex`](../../passage-paper/neurips/main.tex).
- Developer walkthroughs (unchanged, not paper-facing):
  [`../notebooks/solve.ipynb`](../notebooks/solve.ipynb),
  [`../notebooks/costmodel.ipynb`](../notebooks/costmodel.ipynb),
  [`../notebooks/obstacles.ipynb`](../notebooks/obstacles.ipynb).

## Scope

- This workspace produces dataset-side figures and one dataset table only.
- Paper-side figures driven by the model-vision program (Fig. 1 teaser,
  Table 5 paradigm headline) live in `../../passage-vision/paper/` and are
  **not** reproduced here.
- GNN and RL families are explicitly out of scope for this submission and
  must not be introduced in these notebooks.
