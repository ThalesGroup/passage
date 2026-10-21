---
title: passage-eval package
icon: material/package-variant-closed
---

# `passage-eval` package

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

`passage_eval` loads released scenes, scores decoded paths under their analytic
terrain task, and preserves the evidence needed for statistical comparison. It
ships in the root `passage` distribution, version `1.0.0`. This guide uses the
public API; [the protocol](protocol.md) gives the metric definitions and how the
evaluator reads the paper.

To evaluate your own planner, start with
[Evaluate your planner](contributing-a-method.md). Its CLI separates `conform`,
`run`, offline `score`, matched `compare` and `report`, and keeps every timed
route. The Python functions below are lighter: they score paths and run the
paper's paired statistics, but they do not produce the timed results on
qualified references that the CLI workflow produces. The paper's learned-model
results are historical results of the original study, reported as obtained;
not re-run on the released dataset.

## 1. Install and load a frozen subset

From the repository root:

```bash
uv sync --frozen            # installs passage and passage_eval
# or: pip install -e .
uv run --frozen python -c "import passage_eval; print(passage_eval.__version__)"  # 1.0.0
```

Add `--extra torch` to `uv sync` (or `pip install -e ".[torch]"`) to time
PyTorch planners on a GPU. The evaluator tests live in `tests/passage_eval`;
run them with `uv run --frozen pytest tests/passage_eval -q` (`uv sync`
installs the `dev` group, which holds pytest, by default).

Remote loading needs a pinned 40-character dataset commit. The example below
uses the verified PASSAGE v1.0 upload:

```python
from passage_eval.io.dataset import load_val_scenes

scenes = load_val_scenes(
    resolution=64, cost_model="elevation", split="validation", n_samples=8,
    dataset_repo="thalesgroup/passage",
    dataset_revision="2ab44d899652c7fb99c425488e84a7d565963547",
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
Hausdorff p95 uses raw pixels. Arrays without a metric-contract label, such as
the released paper results, use the original study's units; scoring them
requires an explicit `allow_legacy=True`.

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

```python
from passage_eval import paired_bootstrap_delta, holm_bonferroni, decide_pairwise

# A and B must refer to the same frozen samples in the same order.
assert np.array_equal(arrays_A["sample_id"], arrays_B["sample_id"])
record, draws = paired_bootstrap_delta(
    arrays_A["regret"], arrays_B["regret"], metric="regret_mean",
    n_bootstrap=10_000, seed=1234, return_draws=True,
)
p_value = record.two_sided_p_value(draws)
print(record.delta, record.ci_low, record.ci_high, p_value)
```

For three seeds use `multiseed_paired_bootstrap(A_by_seed, B_by_seed,
metric=..., n_bootstrap_per_seed=10_000, return_draws=True)`. Input dictionaries
are keyed by 1234, 2026, and 3407. Geometry compares jointly finite pairs and
reports their retained count. Regret failures remain finite through censorship.
Apply `holm_bonferroni` to the declared family and call
`decide_pairwise(records, holm=adjusted)`. A full Pareto claim requires every
metric to favor the same method; wins with unconfirmed metrics are a trade-off.

[Paired statistics (paper)](walkthrough.md) gives the complete paired
mean-difference example. These functions estimate paired mean differences, as
in the paper. The `passage-eval compare` command instead produces
geographic-block intervals for its reported statistics and marks missing block
information. Choose the estimand and
resampling unit before inspecting results.

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
