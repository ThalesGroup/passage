---
title: Evaluation Protocol
icon: material/format-list-numbered
---

# Evaluation protocol

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

The evaluator implements the terrain task and the decoded-path metrics of the
accepted paper. This page identifies the paper, explains the executable
contract, and states how the evaluator reads points that the paper leaves open.
Training a candidate and measuring hardware produce their own records; a
successful CLI run alone does not reproduce a paper headline.

## 1. Reference paper and alignment

The reference is the accepted paper *PASSAGE: A Real-Terrain Benchmark for
Trustworthy Constrained Path Planning*, NeurIPS 2026, Track on Evaluations and
Datasets ([citation](../index.md#cite-passage)). Learned-model figures quoted
from the paper are historical results of the original study, reported as
obtained; not re-run on the released dataset.

Every per-sample array carries a metric-contract label. `evaluate_planner` and
`score_planner` label their arrays `neurips-2026-submission-v1`. The
complete-request workflow (`passage-eval run`, then `passage-eval score`) labels
its scored arrays `passage-qualified-complete-v1`, or
`passage-stored-diagnostic-v1` when the scoring API is explicitly told to use
unqualified stored labels. Results produced under another metric contract must
be evaluated again before comparison.

> **Complete-request workflow**
>
> The `passage-eval` CLI first qualifies the reference route of every task,
> then times each planner call and saves its route, then scores every saved
> route offline. It writes to its own result folder. The terms *complete
> request*, *qualified* and *cohort* are defined on the
> [Benchmark overview](index.md).
> [Evaluate your planner](contributing-a-method.md) gives the commands, the
> deadline rules, how the primary repeat is chosen and the reference policies.

### 1.A. How the evaluator reads the paper

- **Censored costs:** a route that does not connect start and goal, or that
  crosses an obstacle, receives the censored cost of 10⁶ times the reference
  cost. The evaluator keeps every such failure in the regret statistics.
  Geometry uses the valid routes only, and `n_path_valid` reports how many
  there are.
- **Geometry units:** ASSD is divided by the number of reference waypoints;
  percentile-Hausdorff is in pixels (grid cells). The geometry values in the
  released paper results follow the original study's normalization; do not
  compare them directly with new results.
- **Worked geometry example:** for the worked example in the paper's appendix,
  the reference has six waypoints and the route seven. Two route waypoints lie
  one pixel from the reference, and one reference waypoint lies one pixel from
  the route. The evaluator returns ASSD `(2/7 + 1/6)/2/6` = **0.0376984** and
  Hausdorff p95 = **1 pixel**; the test suite checks both values.
- **Sample seeding and obstacles:** each sample's random seed is a BLAKE2b
  hash of `(generation_seed, resolution, index, split)`. Obstacle placement
  makes up to 50 mask attempts, each with up to 200 candidate shapes; 50 counts
  attempts, not shapes. With the same configuration, regenerating a sample
  index gives the same sample.

## 2. Released scenes and analytic costs

The paper's learned-model results are historical results of the original
study, reported as obtained; not re-run on the released dataset (see
[paper results](leaderboard.md#paper-results)).
The loader reads PASSAGE v1.0 from
[thalesgroup/passage](https://huggingface.co/datasets/thalesgroup/passage) by default.
Use the full `<N>x<N>` configuration and pin a dataset commit, or supply a local
export root containing `<N>x<N>/<split>/*.parquet`. `val` aliases `validation`.
The default cap of 1024 samples selects a deterministic prefix, not the entire
validation split. Data errors propagate; evaluation never substitutes synthetic
terrain or manufactures replacement labels.

Decode the Zstandard float32 `(H,W,3)` tensor as globally normalized elevation,
start/goal markers, and obstacles. The loader reconstructs elevation in meters
from the sample's global calibration, selects its released solver and weight, and
validates the oracle's endpoints, connectivity, and obstacle avoidance. A requested
weight may confirm the release value; changing it while retaining that oracle is
rejected. The `free` task variant clears obstacle inputs and selects free labels.

On the 8-connected grid, cardinal edges are 30 meters and diagonals are
`30 * sqrt(2)`. Let `d` denote that distance, `u` the source, `v` the destination,
`z` raw elevation, and `w` the released weight. The release cost formulas are:

| Model | Directed edge cost |
|---|---|
| `distance` | `d` |
| `elevation` | `d * (1 + w * (z[v] - crop_min) / crop_range)` |
| `slope_uphill` | `d * (1 + w * max(z[v] - z[u], 0) / crop_range)` |
| `energy` | `d + w * max(z[v] - z[u], 0)` |
| `absolute_elevation` | `d * (1 + w * abs(z[v]) / crop_max_abs)` |

`crop_min`, `crop_range` and `crop_max_abs` are the minimum, the range and the
largest absolute value of `z` over the whole crop. On a flat crop (zero range
or zero maximum), the normalized term is zero. PASSAGE v1.0 uses `elevation`,
`slope_uphill` and `energy`. The evaluator also implements `distance` and
`absolute_elevation`, which the v1.0 release does not use.

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
declared estimator. The `passage-eval compare` command recomputes each selected statistic
under paired whole-geographic-block resampling. Missing cluster identities leave
uncertainty unknown; seeds remain separate descriptive conditions. These intervals
are distinct from the paper's pooled mean-difference procedure.
[Paired statistics (paper)](walkthrough.md) shows the paper's mean-difference
functions.

## 5. Trustworthiness and deployment

Report stability (three-seed coefficient of variation), geographic hold-out versus
matched random split, Operational Design Domain boundary behavior, and
same-device oracle-relative efficiency. The geographic test uses the South
America hold-out; each evaluated cohort needs its own identified split policy and
support analysis. Validation results alone do not establish those axes.

Time the full input-to-path pipeline at batch size one on the executing hardware.
The CLI runner defaults to five warmup passes and one warm measured pass
over the selected tasks, with a separate cold pass after candidate construction.
Declare `latency_repeats` and freeze the primary `quality_repeat_index` before
execution. Every timed repeat retains its route, status, complete latency and
offline quality. These settings differ from the paper's
resolution-dependent repetition campaign. Device completion precedes the CPU
route snapshot, legality check and recosting inside the request timer; imports,
model loading, persistence, reference solving and offline geometry remain outside.
On an accelerator, the timer must stop only after the device work has
finished. The CLI synchronizes CUDA work when the planner uses PyTorch; a
planner on another runtime must finish its device work before `plan` returns.
An accelerator label does not select a device or measure latency. A Python
planner cannot be stopped in the middle of a call: the runner waits for it to
return and records a late result as a timeout (see
[Evaluate your planner](contributing-a-method.md)).

Parameter count, MACs, peak memory, forward/decoder decomposition, and the
trustworthiness campaign require additional method-specific measurements. The
CLI's result bundle is evaluation evidence, not a claim that these studies ran.

## 6. Reproducibility evidence

For each seed, `passage-eval run` writes a frozen manifest, terminal request
records, numeric ordered paths and reference qualifications. Offline scoring adds
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
