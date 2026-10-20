---
title: Evaluation Protocol
icon: material/format-list-numbered
---

# Evaluation protocol

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

The evaluator implements the submitted paper's terrain task and decoded-path
metrics. This page identifies the reference, explains the executable contract,
and records contradictions that prevent claiming exact reproduction of every
published number. Candidate training and hardware measurements require their own
artifacts; a successful CLI run alone does not certify a paper headline.

## 1. Reference paper and alignment

The reference is **Passage: A Real-Terrain Benchmark for Trustworthy Constrained
Path Planning**, the 47-page file `875_PASSAGE_A_Real_Terrain_Ben.pdf` supplied for
this alignment. Its SHA-256 is
`5dcb79a2aa1345dcddaac71c6e9e522e2f22525f1c1a899cbea27b546fe5b4af`.
The older local `paper/neurips/main.pdf` has 46 pages and SHA-256
`8689e2deb02bb8ef8177ad1bc17fce74d888ac93c4a6874034a1bce44730ccbd`.

Besides the title and release links, the submission changes Table 3's segmentation
geographic delta from −0.014 to −0.007, removes the former latency/memory rows,
updates Figure 2's accelerator comparison, and reports real 4096² Thor inference
in Appendix I (478 ms segmentation, 317 ms regression, five real samples).
Those figures are paper claims, not new measurements from this checkout. The
editable local manuscript predates the submitted PDF and has not been rewritten.

The evaluator labels new arrays and sidecars `neurips-2026-submission-v1`.
Results produced under another metric contract require re-evaluation before
comparison.

> **Current execution contract**
>
> The new complete-request workflow qualifies references, persists each timed
> route, then scores every repeat offline. It uses a separate qualified contract
> and result root; historical arrays keep `neurips-2026-submission-v1`. The
> [candidate guide](contributing-a-method.md) documents the runnable commands,
> deadline limitations, primary-repeat selection and reference-cohort policies.
> A local audit or revised workflow does not authenticate historical paper values.

### 1.A. Explicit resolutions of paper inconsistencies

- **Cost failures:** §4.1 assigns failed or obstacle-crossing paths a censored
  cost. Appendix A.3 instead describes cost and geometry on successful paths.
  The evaluator follows §4.1 for costs and retains every failure in regret;
  geometry uses the valid subset, with `n_path_valid` reporting its coverage.
- **Geometry units:** §4.1 divides average symmetric surface distance (ASSD)
  by oracle waypoint count. Appendix B states percentile-Hausdorff in pixels.
  The evaluator uses those definitions. The paper's table and NPZ values may
  reflect the study's earlier normalization and must not be compared directly.
- **Worked arithmetic:** Appendix B's displayed paths have two predicted
  waypoints one pixel from the oracle and one oracle waypoint one pixel from
  the prediction. Their ASSD is `(2/7 + 1/6)/2`; dividing by six oracle
  waypoints gives **0.0376984**, rather than the text's approximately 0.012.
  Hausdorff p95 is **1 pixel**. Tests use the displayed coordinates and formula.
- **Generation wording:** Table 1 abbreviates sample seeding as a hash of split
  seed and ID. Release generation actually hashes
  `(generation_seed, resolution, index, split)` with BLAKE2b. Its obstacle
  defaults permit 50 mask attempts with 200 candidate placements per attempt;
  50 is not a maximum shape count. The release algorithm is preserved so
  regenerating existing IDs does not silently produce different samples.

## 2. Released scenes and analytic costs

The paper's learned results come from the original study; they were not
recomputed on the published dataset (see [paper results](leaderboard.md#paper-results)).
The loader reads PASSAGE v1.0 from
[thalesgroup/passage](https://huggingface.co/datasets/thalesgroup/passage) by default.
Use the full `<N>x<N>` configuration and pin a dataset commit, or supply a local
export root containing `<N>x<N>/<split>/*.parquet`. `val` aliases `validation`.
The default cap of 1024 samples selects a deterministic prefix, not the entire
validation split. Data errors propagate; evaluation never substitutes synthetic
terrain or manufactures replacement labels.

Decode the Zstandard float32 `(H,W,3)` tensor as globally normalized elevation,
start/goal markers, and obstacles. The loader reconstructs metre elevation using
the sample's global calibration, selects its released solver and weight, and
validates the oracle's endpoints, connectivity, and obstacle avoidance. A requested
weight may confirm the release value; changing it while retaining that oracle is
rejected. The `free` task variant clears obstacle inputs and selects free labels.

On the 8-connected grid, cardinal edges are 30 metres and diagonals are
`30 * sqrt(2)`. Let `d` denote that distance, `u` the source, `v` the destination,
`z` raw elevation, and `w` the released weight. The release cost formulas are:

| Model | Directed edge cost |
|---|---|
| `distance` | `d` |
| `elevation` | `d * (1 + w * (z[v] - crop_min) / crop_range)` |
| `slope_uphill` | `d * (1 + w * max(z[v] - z[u], 0) / crop_range)` |
| `energy` | `d + w * max(z[v] - z[u], 0)` |
| `absolute_elevation` | `d * (1 + w * abs(z[v]) / crop_max_abs)` |

Zero normalization ranges use the generation backend's zero-valued normalized
arrays. The first three terrain models are the paper's main configured models;
distance and absolute elevation remain available for release-compatible tasks.

## 3. Six primary decoded-path metrics

Every candidate emits an ordered list of `(row, column)` integer cells. Scoring
recomputes costs from that path, ignoring candidate-reported costs. Connection
requires at least two cells, exact prescribed endpoints, in-bounds coordinates,
and an 8-connected move at every step. Duplicate consecutive cells are invalid;
loops with legal moves are allowed. Coordinates are never rounded or clipped.

| Metric | Contract |
|---|---|
| `cost_regret_median` | Median of `predicted_cost / oracle_cost - 1` |
| `cost_regret_p90` | 90th percentile of the same per-sample regrets |
| `path_assd` | Mean of each valid pair's symmetric nearest-waypoint distance divided by oracle waypoint count |
| `path_hausdorff_p95` | Mean of each valid pair's maximum directional 95th-percentile nearest-waypoint distance, in pixels |
| `success_rate` | Fraction connecting the prescribed endpoints with legal grid moves |
| `obstacle_violation_rate` | Fraction containing at least one forbidden cell |

An obstacle crossing can be connected but remains invalid for cost/geometry
scoring. Failed connection, invalid coordinates, or an obstacle violation receive
`predicted_cost = 1e6 * oracle_cost`, hence regret `999999`. Geometry is NaN
for invalid paths and aggregated over valid paths. Report both reliability rates
and `n_path_valid`; never hide the reduced geometry coverage.

Use `score_planner` or `evaluate_planner` to evaluate paths. The lower-level
`compute_primary_metrics` assumes its supplied costs already use the analytic
scene model. See [the API guide](passage-eval.md).

## 4. Calibration and statistical discipline

Declare the canonical decoder and its parameters on the calibration split before
the test pass. Record those decisions and their deterministic hash. Freeze the
headline operating point at 256×256 under `elevation`, the six primary metrics,
and training seeds **1234, 2026, 3407**. The test pass is performed once per seed;
repeated inference seeds do not replace three independently trained checkpoints.

Pair candidate and baseline arrays by sample ID, split, dataset hash, task
variant, cost model, and weight. For the paper's mean paired-difference test,
bootstrap each seed with 10,000 resamples and pool the 30,000 draws. Correct the
six empirical two-sided p-values with Holm–Bonferroni at family-wise alpha 0.05.
CI overlap alone does not establish a win. Use the API's returned draws for
empirical p-values; its CI-only fallback is an approximation.

Report per-seed estimates, confidence intervals, and the across-seed mean.
Different reductions of regret, such as medians and p90s, need their own explicitly
declared estimator. The qualified `compare` CLI recomputes each selected statistic
under paired whole-geographic-block resampling. Missing cluster identities leave
uncertainty unknown; seeds remain separate descriptive conditions. These intervals
are distinct from the paper's pooled mean-difference procedure.
[The walkthrough](walkthrough.md) shows the paper's mean-difference API.

## 5. Trustworthiness and deployment

Report stability (three-seed coefficient of variation), geographic hold-out versus
matched random split, Operational Design Domain boundary behavior, and
same-device oracle-relative efficiency. The geographic test uses the South
America hold-out; each evaluated cohort needs its own identified split policy and
support analysis. Validation results alone do not establish those axes.

Time the full input-to-path pipeline at batch size one on the executing hardware.
The qualified runner defaults to five warmup passes and one warm measured pass
over the selected tasks, with a separate cold pass after candidate construction.
Declare `latency_repeats` and freeze the primary `quality_repeat_index` before
execution. Every timed repeat retains its route, status, complete latency and
offline quality. These settings differ from the paper's
resolution-dependent repetition campaign. Device completion precedes the CPU
route snapshot, legality check and recosting inside the request timer; imports,
model loading, persistence, reference solving and offline geometry remain outside.
Qualify synchronization on the measured device. An accelerator label does not
select a device or create a latency measurement. Cooperative timeout/drain
limitations remain explicit in the candidate guide.

Parameter count, MACs, peak memory, forward/decoder decomposition, and the
trustworthiness campaign require additional method-specific measurements. The
CLI's result bundle is evaluation evidence, not a claim that these studies ran.

## 6. Reproducibility evidence

Each qualified run seed writes a frozen manifest, terminal request records,
numeric ordered paths and reference qualifications. Offline scoring adds
`metrics.json`, `per_sample.npz` and `score_receipt.json` for every series;
`bench` also emits a compatible `sidecar.json`. Arrays
include sample IDs, costs, validity, obstacle counts, geometry, and contract version.
The sidecar records the evaluated subset hash, declared dataset revision, seed set,
inference kwargs, calibration hash when supplied, hardware, timing settings,
precision, evaluator commit and source hash, and a declared container digest.
Absent calibration/container declarations remain absent. Determinism defaults to
`non_deterministic`; a stronger claim requires the submitter's evidence.

Use a fresh result directory for this contract. Frozen data and evaluator hashes
help auditing, but a reproducible learned-method headline also needs the candidate
source, checkpoint artifacts, runtime/container, calibration record, and the full
statistical and trustworthiness analysis.
