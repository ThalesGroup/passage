# Changelog

## Unreleased

- Restyle the documentation site and the README header with the PASSAGE Neon
  theme, built on the Thales colours of the app icon; the helicopter banner
  stays the centre of the home hero.
- Animate the home hero with a night flight over a neon wireframe terrain
  (aurora, summit beacons, a route along the valley and search waves) that
  steers with the pointer and stops under reduced motion; open each tab's first
  sub-level in a menu on hover or keyboard focus; indent the sidebar levels.
- Reorganise the documentation into Home, Dataset and Benchmark. Release,
  presentation and policies are under Home, generation and the data format
  under Dataset, and code architecture and API under Benchmark.
- Replace the presentation schemes with the NeurIPS 2026 talk: the recorded
  video, then every slide with its speaker script.
- Define the leaderboard submission process: Seg and Reg as the main candidates
  and the model for new ones, a contestant guide with a local check and a
  GitHub submission form, and the Thales runbook from intake to publication.
- Refresh the ImageIO, Loguru, Plotly, Pymdown Extensions and python-dotenv
  dependencies. Update Mkdocstrings alongside Pymdown Extensions to preserve
  documentation highlighting, and remove an unsupported Hugging Face Hub extra.
- Update the pinned GitHub Actions, preserving cache pruning and Pages hidden
  files. Group weekly dependency proposals by ecosystem for consolidated review.
- Reject ML image builds whose installed runtime cannot satisfy the supplied
  core wheel's declared requirements, before running the existing CPU probe.

## 1.0.0 — first official code release

The code version is 1.0.0; the dataset retains its separate v1.0 identity.
Historical private wheels and runtime receipts, including those labelled 1.1.0,
retain their original versions and hashes. They are not renamed release artifacts.

- **Dataset PASSAGE v1.0** on Hugging Face
  ([`thalesgroup/passage`](https://huggingface.co/datasets/thalesgroup/passage)):
  4,400,000 core scenes at seven resolutions, from 64×64 to 4096×4096 cells,
  built from JAXA ALOS AW3D30 terrain. Each scene carries 6 route variants
  (three objectives, with and without obstacles), 26,400,000 exact A* routes
  in all; start and goal cells that the obstacles disconnect are rejected by
  the generator. Train / calibration / validation / test
  splits are 90 / 2 / 4 / 4 %, with a South America hold-out and 5°×5° source
  blocks elsewhere; 1,792 preview rows carry rendered images.
- **Generator** (`passage`): `download`, `calibrate`, `generate`, `compress`,
  `export` and `upload`, plus `regenerate` for bounded, resumable generation
  with verified Parquet export. Generation is deterministic from
  `generation.seed`, and every sample records its provenance. `audit` and
  `qualify` check stored routes and released files against recorded receipts.
- **Evaluator** (`passage-eval` 1.0.0): a complete-request harness
  (`conform`, `run`, `score`, `compare`, `report`) for classical, graph,
  reinforcement-learning and vision planners. Planner inputs exclude reference
  labels. Costs are recomputed with 30-metre edges and the released weights;
  failed requests are censored at `1e6 × optimum`. ASSD is normalized by the
  reference waypoint count and Hausdorff p95 is reported in pixels. Paired
  comparisons resample geographic blocks, and every score has a reproducibility
  sidecar. Classical baselines and leaderboard storage are included.
- **Common candidate interface**: `passage-candidate-package/v1` manifests,
  artifact hashing with `candidate freeze`, validation without planner imports,
  and immutable local registration. The common worker supports CPU/CUDA
  execution, rootless OCI sessions, input-only transport, retained execution
  receipts and independent host scoring. Runtime images and device compatibility
  remain operator prerequisites; registration does not grant admission.
- **Security**: require Python 3.11.5 or later, Soup Sieve 4.0.0 or later and
  Tornado 6.6.1 or later. The refreshed lock uses Soup Sieve 4.1.1 and Tornado
  6.6.1 to address the five dependency alerts reported in November 2026.
  Recurring audits and dependency-update proposals cover both publication and
  source integration workflows.
- **Presentation schemes**: unchanged final talk v17 slides and poster v9,
  with source checksums and separate qualitative TRAIN versus TEST context.
- **Leaderboard evidence**: the private Space's active SEG/REG view displays
  27 TEST and three separate VALID pilot observations marked
  `measured_unadmitted`. Historical observations retain their provenance. These are distinct from the
  repository's local storage and empty checked-in TEST entry table. Per-task
  paper results remain separate under `src/passage-eval/results/paper-results/`.
  Public model/runtime publication and leaderboard admission are pending.
- **Documentation and metadata**: user guides, benchmark protocol, datasheet,
  Croissant metadata, citation file, dataset terms (`DATASET_LICENSE.md`),
  notebooks and a test suite. The [release guide](docs/release.md) documents local
  builds, exact-version installation and the remaining publication checks.
