# PASSAGE Dataset Paper Workspace

This directory is the dataset-side paper workspace for PASSAGE. It produces
the figures and tables that the paper cites when it needs evidence about the
dataset itself: solver-runtime scaling, cost-model path divergence,
obstacle showcase, and split / coverage statistics.

The companion model-side workspace lives at
[`../../vision/paper/`](../../vision/paper/) and carries the segmentation,
regression, and edge-inference baseline evidence. The paper source of
truth lives at
[`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).

## Directory Layout

| Path | Role |
| --- | --- |
| `README.md` | workspace index, paper-section mapping, stage-interface contract. |
| `ROADMAP.md` | four notebook stages, inputs, outputs, evidence pointers, owners. |
| `REFERENCE_TRACKER.md` | dataset-track citation staging. |
| `helpers.py` | shared Python helpers for the four notebooks. |
| `notebooks/` | one notebook per dataset-side study (figures and tables). |
| `figures/` | output artifacts: `<study>.pdf`, `<study>.png`, `<study>.manifest.json`. |

## Paper Section Mapping

Source of truth: [`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).
Every row below points at the section / figure / table identifier it
populates in the current paper.

| Notebook | Paper anchor | Figure / Table |
| --- | --- | --- |
| `dataset_statistics.ipynb` | App. E (ODD Task-Instance-Space Cardinality) | `tab:split-counts`, `fig:geo-coverage` |
| `obstacles_showcase.ipynb` | §3.3 (Operational Design Domain), Table~`tab:odd` | obstacle-envelope visual gallery |
| `solver_runtime_scaling.ipynb` | App. C (A$^{\star}$ Oracle Solver Comparison) | `tab:solver-comparison` |
| `cost_model_path_divergence.ipynb` | App. D (Cost-Model Examples) | `fig:cost-model-paths`, `tab:cost-model-examples` |

The model-side workspace (`../../vision/paper/`) owns the body
**Fig. 1 (teaser)** assembly, **Fig. 2 (per-edge-device latency)**, the
headline **Table 4** (seg vs. reg vs. A$^{\star}$), and every appendix
figure under App. F (ODD Coverage Diagnostics), App. H (Detailed Vision
Experiments), and App. I (Edge-Device Inference Validation).

## Stage Interface Contract (light)

Each notebook is expected to:

1. Import from `passage.paper.helpers` (the module in this directory) and
   from `passage` itself. Fail gracefully if data is not mounted.
2. Pin every input (resolutions, tile ids, backends, seeds) in a single
   setup cell.
3. Produce a paper-ready artifact under `figures/` with a matching
   `<study>.manifest.json` recording those inputs plus the exact output
   paths.
4. Not mutate state outside `figures/`. Do not write to `../notebooks/` or
   elsewhere in the dataset repository.

## Release Dependency

`dataset_statistics.ipynb` is the release-gated notebook in this workspace.
Before promoting `tab:split-counts`, `fig:geo-coverage`, the Datasheet
appendix (App. G), or any release-tree numbers into the paper, refresh the
release surfaces and checksum them:

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

- `dataset_statistics.ipynb` — produces split-count tables and a
  geographic-coverage figure under `figures/`.
- `obstacles_showcase.ipynb` — produces an obstacle-envelope gallery
  under `figures/`.
- `solver_runtime_scaling.ipynb` — produces the cross-backend solver
  scaling plot and CSV.
- `cost_model_path_divergence.ipynb` — produces the cost-model path
  divergence panel and CSV (Fig. 3 / Table~`tab:cost-model-examples`).

Every notebook is runnable top-to-bottom in a fresh kernel even without
the real data share: the setup cell detects missing inputs and
short-circuits to a stub manifest so the artifact contract is still
honoured.

## Cross-Links

- Dataset sheet (App. G of the paper): [`../DATASHEET.md`](../DATASHEET.md).
- Croissant metadata: [`../croissant.json`](../croissant.json) (validated
  against MLCommons 1.1 / RAI 1.0; ships with the dataset release tree
  but is not referenced from the paper body).
- Paper source: [`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).
- Developer walkthroughs (unchanged, not paper-facing):
  [`../notebooks/solve.ipynb`](../notebooks/solve.ipynb),
  [`../notebooks/costmodel.ipynb`](../notebooks/costmodel.ipynb),
  [`../notebooks/obstacles.ipynb`](../notebooks/obstacles.ipynb).

## Scope

- This workspace produces dataset-side figures and tables only.
- The body teaser (Fig. 1), the edge-latency body figure (Fig. 2), and the
  paradigm headline (Table 4) are owned by `../../vision/paper/` and are
  **not** reproduced here.
- Graph-neural-network and reinforcement-learning planner families are
  named in the paper Conclusion and in App. A only as **future-work
  targets** for the cross-approach evaluation protocol; the present
  release ships only the two dense-vision example baselines, and these
  notebooks do not attempt to introduce GNN / RL evidence.
