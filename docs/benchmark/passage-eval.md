---
title: passage-eval package
icon: material/package-variant-closed
---

# `passage-eval` package

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-24 | 1.1 |

`passage_eval` loads released scenes, scores decoded paths under their analytic
terrain task, and preserves the evidence needed for statistical comparison. It
ships in the root `passage` distribution, version `1.0.0`. This guide uses the
public API; [the protocol](protocol.md) gives the metric definitions and how the
evaluator reads the paper.

To evaluate your own planner, start with
[Evaluate your planner](contributing-a-method.md). Its CLI separates `conform`,
`run`, offline `score`, matched `compare` and `report`, and keeps every timed
route. The Python functions below are lighter: they score paths and compare
identified per-task outcomes. The CLI workflow additionally produces timed
results on qualified references.

## 1. Install and load a frozen subset

From the repository root:

```bash
uv sync --frozen            # installs passage and passage_eval
# or: pip install -e .
uv run --frozen python -c "import passage_eval; print(passage_eval.__version__)"  # 1.0.0
```

PyTorch planners need a separately installed
[runtime](../guide/installation.md#optional-pytorch-and-edge-runtimes).
The Python timing helpers use CPU by default. Pass `device="cuda:0"` to
select a compatible GPU explicitly. The evaluator tests live in `tests/passage_eval`;
run them with `uv run --frozen pytest tests/passage_eval -q` (`uv sync`
installs the `dev` group, which holds pytest, by default).

Remote loading needs a pinned 40-character dataset commit. The example below
uses the verified PASSAGE v1.0 upload:

```python
from passage_eval.io.dataset import load_val_scenes

scenes = load_val_scenes(
    resolution=64, cost_model="elevation", split="validation", n_samples=8,
    dataset_repo="thalesgroup/passage",
    dataset_revision="c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae",
)
# Local alternative: dataset_path="<root>/export"
```

The loader validates compressed tensors, calibration, released solver parameters,
and oracle paths. It does not silently fall back to generated data. Sample count
is a prefix cap. The three input channels are elevation, start/goal markers, and
obstacles; reconstructed `Scene.elevation` is in meters.

## 2. Score a planner

```python
from passage_eval import AStarPlanner, evaluate_planner, score_planner

metrics, arrays = evaluate_planner(AStarPlanner(), scenes, batch_size=1)
print(metrics)
# score_planner(...) returns just the metric dictionary.
```

The evaluator gives the candidate isolated input copies without oracle labels,
then validates its path and recomputes costs. Failed paths retain censored regret;
geometry uses valid paths. New arrays include `sample_id`, `regret`, `assd`, `h95`,
`success`, `obstacle_violation`, `valid_path`, `obstacle_cells`, `pred_costs`,
`oracle_costs`, and `metric_contract`.

ASSD divides symmetric nearest-waypoint distance by oracle waypoint count.
Hausdorff p95 uses raw pixels. Arrays without a metric-contract label require an explicit
`allow_legacy=True`; callers must declare their normalization before comparison.

```python
import numpy as np
from passage_eval import load_per_sample_npz, primary_metrics_from_arrays

np.savez_compressed("current.npz", **arrays)
restored = primary_metrics_from_arrays(load_per_sample_npz("current.npz"))
```

The lower-level `compute_primary_metrics` accepts `predicted_paths`,
`oracle_paths`, `predicted_costs`, `oracle_costs`, and optional `starts`, `goals`,
`obstacles`. It assumes costs already use the analytic task; prefer the planner
API when candidate costs are untrusted.

## 3. Compare paired methods

For a complete comparison, supply measured per-task outcomes for both methods
at each training seed. Each mapping contains `task_id`, `regret`, `assd`, `h95`,
`success`, `obstacle_violation`, and `valid_path` in the same frozen task order.
Use the primary quality repeat, keep failed-path censored regret, and check
matching data, task conditions, work limits and checkpoint provenance first.

```python
from passage_eval import compare_primary_metrics

comparison = compare_primary_metrics(
    A_by_seed, B_by_seed,  # keys: 1234, 2026, 3407
    blocks=geographic_blocks,  # aligned with the shared ordered task roster
    n_bootstrap=10_000, seed=1234,
)
for metric, record in comparison.deltas.items():
    print(metric, record.delta, record.ci_low, record.ci_high,
          record.n_a, record.n_b)
print(comparison.inference_status)  # test_not_provided
```

The API compares all six declared primary metrics using `T(A) - T(B)`, with
median and p90 reductions for regret and means for the other metrics. Each
whole-geographic-block draw is shared across methods and training seeds; the
three seed contrasts are averaged inside that draw. Geometry uses each
method's own valid-path subset. An empty subset in any draw makes that
geometry interval unavailable. The returned record includes per-seed estimates
and intervals in `record.per_seed`.

For confirmatory decisions, also supply `p_values`, `paired_test`, and
`test_assumptions`. These must identify a valid predeclared paired test of the
same contrast and all six p-values (`NaN` when unavailable). The API applies
Holm–Bonferroni; it does not validate the scientific assumptions of a supplied
test. Without that evidence, it returns descriptive intervals and no
confirmatory win. Bootstrap tail fractions and CI-only proxies do not replace
the test. [Paired statistics](walkthrough.md) shows loading and reporting.

The lower-level `paired_bootstrap_delta` and `multiseed_paired_bootstrap` accept
an explicit `statistic` callable. They assume correct pairing; the multiseed
helper also assumes an identical task roster across seeds. With no `blocks`,
they resample individual tasks, so they do not establish geographic
uncertainty. Missing geometry needs explicit `nan_policy="omit"`, which reduces
each method's finite subset separately. Prefer `compare_primary_metrics` for
the complete primary-metric family. The CLI `compare` command instead reports
matched-task geographic-block intervals for each selected statistic and
condition; its output alone does not establish a three-seed confirmatory win.

## 4. Deployment and diagnostics

Public timing entry points are `benchmark_forward`, `benchmark_decoder`, and
`benchmark_end_to_end`; each returns a `BenchmarkRecord` with percentile latency
and runtime information. The CLI workflow times ordinary-input preparation,
inference/search, decoding, repair/fallback, device completion, route snapshot,
legality and physical recosting. It scores the persisted routes offline.
Constructor/model-loading time remains separate; the cold series is the first
pass over the tasks after construction.
Device labels are declarations, and candidates configure their actual runtime.

Public decoder diagnostics include `boundary_f_score`,
`mask_endpoints_in_same_component`, `descent_direction_agreement`,
`bellman_residual_reachable`, `local_ordinal_concordance`, and `field_ssi_mae`.
Trustworthiness helpers include `stability_cv`, `geographic_generalisation`,
`odd_boundary_delta`, and `oracle_relative_efficiency`. These APIs require the
appropriate measured inputs; they do not infer missing studies from a single run.

## 5. Sidecars and result storage

`Submission` and `submit` write metric rows and optional evidence. By default,
a `Submission` records the local host name and user name; remove them before
you share results. The CLI workflow writes a run manifest, terminal request
records, numeric route arrays and reference qualifications. Offline scoring adds
per-sample arrays, metrics and content-bound score receipts. `bench`
additionally writes a `Submission` record and a `ReproducibilitySidecar`. Calibration declarations
are hashed with `compute_calibration_decisions_hash`. See
[Evaluate your planner](contributing-a-method.md) for checkpoint mapping, pinned
data, device provenance, and artifact paths.

A results root rejects mixed metric contracts. Keep results produced under
another contract in their own directory and write corrected evaluations to a
fresh root. Compare
methods only after checking matching sample IDs, data/task provenance, and units.
The software uses the MIT license; source terrain obligations remain documented
in [DATASET_LICENSE.md](https://github.com/ThalesGroup/passage/blob/main/DATASET_LICENSE.md).
