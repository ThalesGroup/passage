# Changelog

## 1.0.0

First public release of PASSAGE, published with the NeurIPS 2026 Evaluations
and Datasets Track paper. The repository and the dataset share version 1.0.

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
- **Leaderboard**: released empty. Its first entries will be the two
  dense-vision finalists of the paper, retrained on PASSAGE v1.0. The per-task
  paper results are released separately under
  `src/passage-eval/results/paper-results/`.
- **Documentation and metadata**: user guides, benchmark protocol, datasheet,
  Croissant metadata, citation file, dataset terms (`DATASET_LICENSE.md`),
  notebooks and a test suite with at least 80 % coverage.
