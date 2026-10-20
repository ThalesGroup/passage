# PASSAGE Dataset Paper Workspace

This directory is the dataset-side paper workspace for PASSAGE. It produces
the figures and tables that the paper cites when it needs evidence about the
dataset itself: source and dataset qualification, solver-runtime scaling,
cost-model path divergence, obstacle showcase, and split / coverage statistics.

The current qualification work reads the pinned source without altering its
Parquets. All 5,620 source files are content-authenticated. The complete metadata
census covers 4.4 million core rows in 28 resolution/split cells, and the frozen
196-scene audit has findings for all 1,176 route variants. The qualification
notebook consumes those receipts and exposes missing prerequisites when a bundle
is absent. Its output is not a learned-planner comparison; learned training and
publication of the updated bundle remain pending.

The companion model-side workspace lives at
[`../../vision/paper/`](../../vision/paper/) and carries the segmentation,
regression, and edge-inference baseline evidence. The submitted reference is
[`../../paper/tmp/875_PASSAGE_A_Real_Terrain_Ben.pdf`](../../paper/tmp/875_PASSAGE_A_Real_Terrain_Ben.pdf).
It differs from the older local `../../paper/neurips/main.pdf` and manuscript
sources. See [the alignment record](../docs/benchmark/protocol.md#1-reference-paper-and-alignment)
for the reference checksum and substantive differences. The editable manuscript
lives at [`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).

## Directory Layout

| Path | Role |
| --- | --- |
| `README.md` | workspace index, paper-section mapping, stage-interface contract. |
| `ROADMAP.md` | historical four-study plan; the current [project plan](../tmp/next.md) governs qualification and implementation scope. |
| `REFERENCE_TRACKER.md` | dataset-track citation staging. |
| `helpers.py` | shared Python helpers for the original four studies. |
| `notebooks/` | one notebook per dataset-side study (figures and tables). |
| `figures/` | output artifacts: `<study>.pdf`, `<study>.png`, `<study>.manifest.json`. |

## Paper Section Mapping

Source of truth: [`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).
The qualification row maps to the current editable manuscript. The remaining
rows preserve the earlier four-study mapping; their anchors and result claims
must be reconciled before publication under the revised S1–S4 scope.

| Notebook | Paper anchor | Figure / Table |
| --- | --- | --- |
| `dataset_qualification.ipynb` | §`sec:qualification`, §`sec:results`, App. `app:qualification` | receipt-backed source/census/route/geographic-support panels; pending inputs are explicit |
| `dataset_statistics.ipynb` | App. E (ODD Task-Instance-Space Cardinality) | `tab:split-counts`, `fig:geo-coverage` |
| `obstacles_showcase.ipynb` | §3.3 (Operational Design Domain), Table~`tab:odd` | obstacle-envelope visual gallery |
| `solver_runtime_scaling.ipynb` | App. C (A$^{\star}$ Oracle Solver Comparison) | `tab:solver-comparison` |
| `cost_model_path_divergence.ipynb` | App. D (Cost-Model Examples) | `fig:cost-model-paths`, `tab:cost-model-examples` |

The historical mapping assigned the model-side workspace (`../../vision/paper/`) the body
**Fig. 1 (teaser)** assembly, **Fig. 2 (per-edge-device latency)**, the
headline **Table 3** (seg vs. reg vs. A$^{\star}$), and every appendix
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
4. Write only to a declared separate output directory. The qualification report
   uses `output_dir` and consumes an explicit `qualification_root` containing
   `source-authentication/`, `metadata/` and `route-audit/`. Never write to the
   source snapshot or infer a result bundle from a private dataset path.

## Release Dependency

Promote counts, geographic support and label-quality statements only from the
identified population and completed receipts. Header counts establish 4.4 million
core rows and 1,792 separate previews; source authentication establishes file
identity. Neither establishes qualified labels or geographic isolation.
Publication staging and any upload are later steps with their own manifests;
qualification does not regenerate, compress or replace the downloaded source.

## Qualification Workflow

From this repository, the phases below reproduce the campaign against the
identified source. They read its Parquets and write separate, resumable outputs.
Authentication, core census and the diagnostic route campaign completed on
2026-10-10. These commands reproduce their declared phases; they do not launch
training or publish the outputs.
The authentication phase reads pinned object identities from Hugging Face.
Set `PASSAGE_DATASET_PATH` to your local download of the identified source, and
`PASSAGE_SOURCE_REPO` and `PASSAGE_SOURCE_REVISION` to its Hugging Face dataset
repository and immutable 40-hex commit.

```bash
uv run --frozen passage qualify \
  --dataset-path "$PASSAGE_DATASET_PATH" \
  --dataset-repo "$PASSAGE_SOURCE_REPO" \
  --dataset-revision "$PASSAGE_SOURCE_REVISION" \
  --output outputs/qualification --phase authenticate --workers 8

uv run --frozen passage qualify \
  --dataset-path "$PASSAGE_DATASET_PATH" \
  --dataset-repo "$PASSAGE_SOURCE_REPO" \
  --dataset-revision "$PASSAGE_SOURCE_REVISION" \
  --output outputs/qualification --phase census --batch-size 2048

uv run --frozen passage qualify \
  --dataset-path "$PASSAGE_DATASET_PATH" \
  --dataset-repo "$PASSAGE_SOURCE_REPO" \
  --dataset-revision "$PASSAGE_SOURCE_REVISION" \
  --output outputs/qualification --phase routes --workers 4
```

The census scanned all four core splits at seven resolutions and froze the
diagnostic roster before tensor/reference outcomes: 16 validation scenes and four
train/calibration/test scenes per resolution, with shortfalls explicit. All three
objectives and both obstacle conditions were attempted with a fixed 100,000-expansion
reference budget. Failures and censoring remain visible. This diverse selection
has unequal inclusion probabilities and cannot estimate population defect rates.
Recorded geographic rectangles and source-tile support do not certify strict
source-cell independence; missing windows remain unknown.

The [bound qualification report](../../paper/neurips/figures/generated/qualification_report.json)
records completed authentication, census and route-audit phases. Producer receipts
are under `../outputs/qualification/`. The census reports zero within-split
duplicate-ID excess rows, zero declared-resolution/split mismatches and consistent
required schemas. It admits 4,399,878 recorded rectangles for intersection checks
and excludes 122 wrapped/reversed-longitude rows. Primary train/test tile names
are disjoint, but positive rectangle-overlap row incidences are 1 train / 2 test
at 256, 5 / 2 at 512 and 1 / 1 at 2048; other resolutions have zero among valid
rectangles. These counts do not establish strict source-cell independence.

The 1,176 diagnostic variants include 1,173 legal routes, 433 stored-cost
agreements and 237 qualified released-label optima. Reference outcomes are 632
exact, 541 budget-censored, 3 `no_path` and 0 `not_run`. The three non-legal route
records are empty obstacle-aware variants of 1024-resolution validation scene
`0000022368`, independently confirmed infeasible. The 740 strict cost-disagreement
and 40 suboptimality flags refer to the decoded float32 graph at unchanged
tolerance; representational differences from generation arithmetic may explain
some discrepancies. They are diagnostic findings with potentially overlapping
reasons, not population defect rates or proof of source corruption.

Open `notebooks/dataset_qualification.ipynb` with `qualification_root` set to the
receipt bundle and `output_dir` set to a separate derived-output directory. The
notebook reports completed phases, and does not launch these expensive producers.

## Quickstart

From a fresh clone, on a host with the `/data/PASSAGE` share mounted:

```bash
cd passage
pip install -e '.[dev]'
jupyter lab paper/notebooks/
```

Then run, in any order:

- `dataset_qualification.ipynb` — consumes the explicit qualification bundle and
  reports authenticated files, census coverage, route findings and unknown support.
- `dataset_statistics.ipynb` — produces split-count tables and a
  geographic-coverage figure under `figures/`.
- `obstacles_showcase.ipynb` — produces an obstacle-envelope gallery
  under `figures/`.
- `solver_runtime_scaling.ipynb` — produces the cross-backend solver
  scaling plot and CSV.
- `cost_model_path_divergence.ipynb` — produces the cost-model path
  divergence panel and CSV (Fig. 3 / Table~`tab:cost-model-examples`).

Missing data or result bundles produce explicit prerequisite status. Pending
panels and stub manifests do not satisfy scientific evidence gates.

## Cross-Links

- Dataset sheet (App. G of the paper): [`../DATASHEET.md`](../DATASHEET.md).
- Croissant metadata: [`../croissant.json`](../croissant.json) (validated
  declaring MLCommons 1.1 and RAI terms; final semantic validation and publication
  of the updated local metadata remain separate release checks).
- Paper source: [`../../paper/neurips/main.tex`](../../paper/neurips/main.tex).
- Developer walkthroughs (unchanged, not paper-facing):
  [`../notebooks/solve.ipynb`](../notebooks/solve.ipynb),
  [`../notebooks/costmodel.ipynb`](../notebooks/costmodel.ipynb),
  [`../notebooks/obstacles.ipynb`](../notebooks/obstacles.ipynb).

## Scope

- This workspace produces dataset-side figures and tables only.
- The body teaser (Fig. 1), the edge-latency body figure (Fig. 2), and the
  paradigm headline (Table 3) are owned by `../../vision/paper/` and are
  **not** reproduced here.
- Graph-neural-network and reinforcement-learning planner families are
  named in the paper Conclusion and in App. A only as **future-work
  targets** for the cross-approach evaluation protocol; the original
  study used two dense-vision families. Learned results on PASSAGE v1.0
  and their publication artifacts are pending; these notebooks do not introduce
  GNN / RL evidence.
