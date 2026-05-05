---
title: Evaluation (`passage-eval`)
icon: material/clipboard-check-outline
---

# Computing PASSAGE evaluation metrics

The [`passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval)
sub-package implements the **canonical evaluation contract** of the
PASSAGE benchmark — every primary metric, statistical-discipline rule,
trustworthiness diagnostic, and inference-footprint utility used in the
PASSAGE paper. It is **decoupled** from the dataset-generation package
(`passage`) so the evaluation stack can be patched without invalidating
released data artefacts. This page is the single entry point for
researchers who want to score a learned planner on PASSAGE.

!!! tip "Why it lives in this repo"
    The dataset and the evaluation contract evolve in lock-step but at
    different cadences (data is mostly frozen post-release; evaluation
    rules can iterate). Keeping them as **sibling sub-packages** under
    one repository makes the lock-step explicit while letting you `pip
    install` either side independently.

---

## Install

`passage-eval` ships as a self-contained sub-package under
`src/passage-eval/`. It depends only on `numpy`, `scipy`, and `pandas`;
the optional `[torch]` extra adds GPU-side throughput utilities, and
`[notebook]` adds matplotlib / tqdm / ipykernel for the interactive walk-through.

=== "From the cloned repo (recommended)"

    ```bash
    git clone https://github.com/ThalesGroup/passage.git
    cd passage
    pip install -e ./src/passage-eval               # core only
    pip install -e ./src/passage-eval[notebook]     # + walk-through
    pip install -e ./src/passage-eval[torch]        # + GPU throughput
    pip install -e ./src/passage-eval[dev]          # + pytest, ruff
    ```

=== "With `uv`"

    ```bash
    cd passage
    uv pip install -e ./src/passage-eval
    # or, to use the project workspace's existing environment:
    uv sync --extra passage-eval
    ```

=== "Sanity check"

    ```bash
    python -c "import passage_eval; print(passage_eval.__version__)"
    # -> 0.2.0
    pytest src/passage-eval/tests -q
    # -> 49 passed
    ```

A stand-alone wheel on PyPI is planned post-camera-ready; for the
NeurIPS submission anchor, we recommend pinning to the repo state via
`pip install -e ./src/passage-eval` against a tagged commit so the
evaluation contract is reproducible.

---

## At a glance

`passage-eval` exposes four orthogonal modules, each mapped to a
section of the PASSAGE paper (`paper/neurips/main.tex`):

| Module                        | Paper section                | What you get |
|-------------------------------|------------------------------|--------------|
| `passage_eval.metrics`        | §5.2 / App. J Primary Metrics | Six headline decoded-path metrics |
| `passage_eval.protocol`       | §5.3 / App. J Statistical Discipline | Bootstrap CI, Holm--Bonferroni, three-seed aggregation, decision rule |
| `passage_eval.diagnostics`    | §5.4 / Trustworthiness       | Family-specific decoder-readiness checks |
| `passage_eval.deployment`     | §5.5 / Inference Footprint   | Forward + decoder + end-to-end timing, FLOPs, throughput |
| `passage_eval.sidecar`        | App. J Reproducibility       | Manifest writer + calibration-decisions hash |

The notebook
[`notebooks/_passage_eval_metrics.ipynb`](https://github.com/ThalesGroup/passage/blob/main/notebooks/_passage_eval_metrics.ipynb)
walks through every module against synthetic ground truth so you can
validate the contract on your own machine before hooking it up to a
real model.

---

## Minimum viable example

```python
import numpy as np
from passage_eval import (
    DEFAULT_SEEDS,                # (1234, 2026, 3407) — pre-registered seeds
    compute_primary_metrics,
    paired_bootstrap_ci,
    apply_censored_penalty,
    multiseed_paired_bootstrap,
    holm_bonferroni,
    decide_pairwise,
)

# Per-sample arrays produced by your evaluation harness.
# Shapes: predicted/oracle path costs (N,), success masks (N,),
# obstacle-violation masks (N,), per-sample ASSD/H95 (N,).
pred_cost   = np.load("baseline/cost_pred.npy")
oracle_cost = np.load("baseline/cost_oracle.npy")
success     = np.load("baseline/success.npy")
violation   = np.load("baseline/violation.npy")
assd        = np.load("baseline/assd.npy")
h95         = np.load("baseline/hausdorff_p95.npy")

# Apply the protocol's censored-penalty rule on failed decodes
# (paper §5.3 / App. J Statistical Discipline).
pred_cost = apply_censored_penalty(pred_cost, success, oracle_cost,
                                    multiplier=2.0)

# Six headline metrics in one call.
metrics = compute_primary_metrics(
    pred_cost=pred_cost, oracle_cost=oracle_cost,
    success=success, violation=violation,
    assd=assd, h95=h95,
)
print(metrics)
# -> {'cost_regret_median': ..., 'cost_regret_p90': ...,
#     'path_assd': ..., 'path_hausdorff_p95': ...,
#     'success_rate': ..., 'obstacle_violation_rate': ...}

# Paired-bootstrap CI on the mean of paired differences vs. a baseline.
ci = paired_bootstrap_ci(
    pred_cost=pred_cost, oracle_cost=oracle_cost,
    n_resamples=10_000, seed=1234,
)
print(ci.point, ci.low, ci.high)
```

Three-seed aggregation, Holm--Bonferroni correction across the six
primary metrics, and the
**Pareto-dominant / Trade-off / Indistinguishable** decision rule are
single-line calls described in
[`notebooks/_passage_eval_metrics.ipynb`](#walk-through-notebook).

---

## What you get — module by module

### `passage_eval.metrics` — Primary decoded-path metrics

Six headline metrics under the Holm--Bonferroni umbrella in §5.2 of the
PASSAGE paper:

- `cost_regret_median(pred_cost, oracle_cost)` — median per-sample cost
  ratio minus 1.
- `cost_regret_p90(pred_cost, oracle_cost)` — 90th-percentile equivalent.
- `path_assd(pred_path, oracle_path)` — average symmetric surface
  distance between predicted and oracle paths.
- `path_hausdorff_p95(pred_path, oracle_path)` — 95th-percentile
  Hausdorff distance.
- `success_rate(decoded)` — fraction of test instances where the
  canonical decoder reaches the goal.
- `obstacle_violation_rate(path, obstacles)` — fraction of decoded
  paths intersecting an obstacle.

```python
from passage_eval.metrics import compute_primary_metrics
```

### `passage_eval.protocol` — Statistical discipline

Pre-registered seed set, paired-bootstrap CI, Holm--Bonferroni step-down
correction, and the family-wise decision rule.

```python
from passage_eval.protocol import (
    DEFAULT_SEEDS,                 # (1234, 2026, 3407)
    apply_censored_penalty,        # failed-decode handling
    paired_bootstrap_delta,        # per-seed paired bootstrap on Δ-mean
    multiseed_paired_bootstrap,    # three-seed pooled bootstrap
    holm_bonferroni,               # step-down family-wise correction
    decide_pairwise,               # Pareto/Trade-off/Indistinguishable
)
```

### `passage_eval.diagnostics` — Trustworthiness diagnostics

Family-specific decoder-readiness checks. **Never headline** (paper
§5.4 — these are diagnostics, not pass/fail criteria).

- `mask_endpoints_in_same_component` — segmentation: does the predicted
  mask reach the goal?
- `descent_direction_agreement` — regression: does ∇V align with the
  optimal-policy direction at training cells?
- `bellman_residual_reachable` — regression: $|V_{i,j} - \min_n (c_{ij,n} + V_n)|$
  on the reachable set.
- `local_ordinal_concordance`, `field_ssi_mae`, `boundary_f_score` —
  additional shape / topology diagnostics.

### `passage_eval.deployment` — Inference footprint

Wall-clock latency / throughput utilities used to populate the
deployment-footprint matrix (paper §5.5 / Table 5).

```python
from passage_eval.deployment import (
    forward_timing,        # CUDA event timer with warm-up
    decoder_timing,        # CPU/GPU decoder timer
    end_to_end_percentiles,
    flops_and_params,
    decoder_recovery_rate,
    astar_expansions_ratio,
)
```

GPU is detected at runtime via `torch.cuda.is_available()`; on CPU-only
hosts the throughput cells fall back gracefully and the matrix reports
CPU-side numbers.

### `passage_eval.sidecar` — Reproducibility manifest

Records every degree of freedom that can affect a headline number, and
emits a deterministic content-hash so reviewers can map any reported
score back to the exact dataset version + seeds + calibration decisions
+ hardware stack + dtype + determinism flag.

```python
from passage_eval.sidecar import (
    ReproducibilitySidecar, compute_calibration_decisions_hash,
)
sidecar = ReproducibilitySidecar(
    dataset_version="passage-1.1",
    seeds=(1234, 2026, 3407),
    calibration_decisions={"threshold": 0.5, "bridge_to_goal": True},
    hardware={"gpu": "H100", "cpu": "Xeon Platinum 8480"},
    determinism=True,
)
print(sidecar.calibration_hash)
```

---

## Walk-through notebook

`notebooks/_passage_eval_metrics.ipynb` (local-only — the leading
underscore excludes it from the Hugging Face dataset card export
pipeline) is a complete, executable walk-through. It synthesises a
toy ground-truth path planner, runs every `passage-eval` API against
it, and produces the same diagnostic plots that back the figures in
the paper. Run it with:

```bash
pip install -e ./src/passage-eval[notebook]
jupyter notebook notebooks/_passage_eval_metrics.ipynb
```

The notebook is mapped 1:1 to the paper sections — every cell
references the section of `paper/neurips/main.tex` it implements.

---

## Reproducing the headline table

To reproduce a row of the PASSAGE deployment-footprint table on a
locally-trained model, run on each of the three pre-registered seeds:

```python
from passage_eval import (
    DEFAULT_SEEDS, multiseed_paired_bootstrap,
    holm_bonferroni, decide_pairwise, PRIMARY_METRICS,
)

per_seed_metrics = []           # list of dicts, one per seed
for seed in DEFAULT_SEEDS:
    # Train your planner with this seed, then run passage-eval on the test split.
    metrics = compute_primary_metrics(...)
    per_seed_metrics.append(metrics)

# Pool over seeds with a paired bootstrap, then apply Holm--Bonferroni
# across the six primary metrics with family-wise α = 0.05.
agg = multiseed_paired_bootstrap(per_seed_metrics, n_resamples=10_000)
hb = holm_bonferroni(agg.p_values, alpha=0.05)
decision = decide_pairwise(agg, baseline=oracle_metrics, hb=hb)
print(decision)
# -> 'pareto_dominant' | 'trade_off' | 'indistinguishable'
```

The notebook contains the full version of this snippet, including
the censored-penalty rule for failed decodes, the
forward-pass / decoder-time / end-to-end percentiles row, and the
reproducibility-sidecar emission.

---

## Cross-references

- Paper: §5.2 (Performance Metrics), §5.3 (Statistical Reporting),
  §5.4 (Trustworthiness Evaluation), §5.5 (Inference Metrics),
  App. J (Cross-Approach Evaluation Protocol).
- API reference: see [`API → Evaluation`](../api/evaluation.md) for
  auto-generated docstrings of every public symbol.
- Source: [`src/passage-eval/`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval).
- Tests: 49 unit tests in `src/passage-eval/tests/` — run with
  `pytest src/passage-eval/tests -q`.
