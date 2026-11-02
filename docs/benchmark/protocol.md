---
title: Evaluation Protocol
icon: material/format-list-numbered
---

# Evaluation protocol

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-24 | 1.1 |

The evaluator implements the paper's terrain task and decoded-path metrics.
This page defines the executable contract, reference costs, path metrics,
calibration procedure and statistical comparisons.

## 1. Reference paper and evaluation

The reference is **Passage: A Real-Terrain Benchmark for Trustworthy Constrained
Path Planning**. The benchmark scores a candidate's final route under the
specified terrain, obstacle and cost model. Each result identifies its inputs,
procedure, reference paths and metric contract.

The complete-request workflow qualifies references, persists each timed route,
then scores every repeat offline. The
[candidate guide](contributing-a-method.md) documents the runnable commands,
deadline handling, primary-repeat selection and reference-cohort policies.

## 2. Released scenes and analytic costs

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
| `success_rate` | Fraction connecting the prescribed endpoints with legal grid moves; complete requests also require terminal status `ok` |
| `obstacle_violation_rate` | Fraction containing at least one forbidden cell |

An obstacle crossing can be connected but remains invalid for cost/geometry
scoring. Failed connection, invalid coordinates, or an obstacle violation receive
`predicted_cost = 1e6 * oracle_cost`, hence regret `999999`. Complete requests
with a non-`ok` terminal status also fail, even if their recorded route connects;
`connected_path` preserves the connection-only outcome separately. Geometry is
NaN for invalid or failed requests and aggregated over valid paths. Report both
reliability rates and `n_path_valid`; never hide the reduced geometry coverage.

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
variant, cost model, and weight. For each metric, the contrast is
`T(candidate) - T(baseline)`: median and p90 for the two regret metrics, and
mean for geometry and reliability. Recompute each method's aggregate inside
every resample. Failed paths remain in regret and reliability; geometry uses
each method's own valid-path subset, with both retained counts reported.

`compare_primary_metrics` requires all six primary metrics, exactly the three
training seeds, identical ordered task IDs, and known geographic blocks. It
uses 10,000 shared whole-block draws by default. Each draw selects the same
blocks for both methods and all three seeds, computes the per-seed contrasts,
then averages those three contrasts. The confidence interval therefore uses
10,000 draws of the seed mean. Per-seed estimates and intervals are also
returned. If a geometry draw has no valid path for either method, its interval
is unavailable; undefined draws are not discarded. These intervals describe
geographic sampling uncertainty conditional on the three fitted checkpoints.

Confirmatory decisions additionally need six p-values from a predeclared valid
paired test of the same block-resampled, seed-mean contrast. Supply the test
name and assumptions with those measured p-values. The API applies
Holm–Bonferroni at family-wise alpha 0.05, but cannot establish the validity of
the caller's test from its name. Bootstrap tail fractions and confidence-interval
proxies do not supply this evidence. Without the test, estimates and intervals
remain descriptive and no confirmatory win is returned. An unavailable test
stays in the six-test family. More method pairs require a prespecified larger
correction family.

The decision distinguishes demonstrated wins on every primary metric,
demonstrated trade-offs in both directions, partial evidence in one direction,
and no demonstrated difference. No demonstrated difference does not establish
equivalence. Primary-metric decisions also do not establish deployment speed
or footprint. [Paired statistics](walkthrough.md) shows the complete API flow.
The `passage-eval compare` command reports matched-task geographic-block
intervals for each selected statistic and condition; it does not perform this
three-checkpoint confirmatory campaign. Missing block identities leave
uncertainty unknown. Released paper results are reported separately and are
unchanged by these API examples.

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
