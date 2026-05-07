---
title: passage-eval Package
icon: material/package-variant-closed
---

# `passage-eval` — Evaluation package

`passage-eval` is a self-contained Python sub-package that
implements the [evaluation protocol](protocol.md) of the
**PASSAGE** paper. It is shipped under
[`src/passage-eval/`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval)
in this repository and is the **only** dependency you need to score a
planner on the **PASSAGE** benchmark.

---

## Install

`passage-eval` ships as a self-contained sub-package. Its core
dependencies are `numpy`, `scipy`, and `pandas`; the optional
`[torch]` extra adds GPU-side throughput utilities, and `[notebook]`
adds matplotlib / tqdm / ipykernel for the interactive walk-through.

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
    pytest src/passage-eval/tests -q
    ```

A stand-alone wheel on PyPI is planned post-camera-ready; for
reproducibility, pin to the repo state via
`pip install -e ./src/passage-eval` against a tagged commit so the
evaluation contract is reproducible.

---

## Module map

The package exposes five orthogonal modules, each mapped to a
section of the [evaluation protocol](protocol.md):

| Module | Protocol section | What it provides |
|---|---|---|
| `passage_eval.metrics`     | Primary metrics              | Six headline decoded-path metrics |
| `passage_eval.protocol`    | Statistical discipline       | Bootstrap CI, Holm–Bonferroni, three-seed aggregation, decision rule |
| `passage_eval.diagnostics` | Trustworthiness diagnostics  | Family-specific decoder-readiness checks |
| `passage_eval.deployment`  | Deployment footprint         | Forward + decoder + end-to-end timing, FLOPs, throughput |
| `passage_eval.sidecar`     | Reproducibility sidecar      | Manifest writer + calibration-decisions hash |

The notebook
[`notebooks/_passage_eval_metrics.ipynb`](https://github.com/ThalesGroup/passage/blob/main/notebooks/_passage_eval_metrics.ipynb)
is a complete walk-through against synthetic ground truth — every
cell maps 1:1 to a paper section.

---

## Primary metrics

`passage_eval.metrics` exposes the six headline decoded-path metrics
under the Holm–Bonferroni umbrella in the protocol:

```python
from passage_eval.metrics import (
    cost_regret_median,
    cost_regret_p90,
    path_assd,
    path_hausdorff_p95,
    success_rate,
    obstacle_violation_rate,
    compute_primary_metrics,        # all six in one call
)
```

The single-call helper:

```python
metrics = compute_primary_metrics(
    pred_cost=pred_cost,            # (N,) costs of decoded paths
    oracle_cost=oracle_cost,        # (N,) costs of A* oracle paths
    success=success,                # (N,) bool, 1 if decoder reached goal
    violation=violation,            # (N,) bool, 1 if path crosses obstacle
    assd=assd,                      # (N,) per-sample ASSD vs oracle
    h95=h95,                        # (N,) per-sample Hausdorff p95 vs oracle
)
print(metrics)
# {'cost_regret_median': 0.012,
#  'cost_regret_p90':    0.041,
#  'path_assd':          0.011,
#  'path_hausdorff_p95': 0.036,
#  'success_rate':       0.997,
#  'obstacle_violation_rate': 0.003}
```

Each metric is also available as a stand-alone function for
fine-grained use.

---

## Statistical discipline

`passage_eval.protocol` exposes the pre-registered seed set, the
paired-bootstrap CI on the mean of paired differences, the
three-seed pooled aggregation, the Holm–Bonferroni step-down
correction, and the family-wise decision rule:

```python
from passage_eval.protocol import (
    DEFAULT_SEEDS,                  # (1234, 2026, 3407)
    PRIMARY_METRICS,                # canonical name list
    apply_censored_penalty,         # failed-decode handling
    paired_bootstrap_delta,         # per-seed paired bootstrap on Δ-mean
    paired_bootstrap_ci,            # CI on a single absolute metric
    multiseed_paired_bootstrap,     # three-seed pooled bootstrap
    holm_bonferroni,                # step-down family-wise correction
    decide_pairwise,                # Pareto / Trade-off / Indistinguishable
)
```

### Paired-bootstrap CI on the paired difference

```python
ci = paired_bootstrap_delta(
    metric_A=metrics_A,             # (N,) per-sample metric values for method A
    metric_B=metrics_B,             # (N,) per-sample metric values for method B
    n_resamples=10_000, seed=1234,
)
print(ci.point, ci.low, ci.high)
```

### Three-seed pooled bootstrap

```python
agg = multiseed_paired_bootstrap(
    per_seed_metric_pairs=[(A_seed1, B_seed1),
                            (A_seed2, B_seed2),
                            (A_seed3, B_seed3)],
    n_resamples=10_000,
)
print(agg.point, agg.low, agg.high, agg.p_value)
```

### Censored-penalty rule

```python
pred_cost = apply_censored_penalty(
    pred_cost, success, oracle_cost, multiplier=2.0,
)
```

### Family-wise decision

```python
hb = holm_bonferroni(p_values=agg.p_values, alpha=0.05)
decision = decide_pairwise(agg, baseline=oracle_metrics, hb=hb)
# -> 'pareto_dominant' | 'trade_off' | 'indistinguishable'
```

---

## Trustworthiness diagnostics

Family-specific decoder-readiness checks. **Never headline** —
these are diagnostics, not pass/fail criteria.

```python
from passage_eval.diagnostics.segmentation import (
    mask_endpoints_in_same_component,   # is the predicted mask connected s -> g?
    boundary_f_score,                   # boundary localisation
)
from passage_eval.diagnostics.regression import (
    descent_direction_agreement,        # ∇V agrees with optimal-policy direction?
    bellman_residual_reachable,         # |V - min_n (c + V_n)| on reachable set
    local_ordinal_concordance,          # local order of V matches order of V*
    field_ssi_mae,                      # mean absolute error on the SSI field
)
```

These appear in the per-approach appendix in the paper and in the
`docs/architecture` section here. They never enter the headline
table.

---

## Inference footprint

`passage_eval.deployment` populates the end-to-end latency report:

```python
from passage_eval.deployment import (
    forward_timing,           # CUDA event timer with warm-up
    decoder_timing,           # CPU/GPU decoder timer
    end_to_end_percentiles,   # p50/p95/p99 over warm replicates
    flops_and_params,         # MACs and parameter count
    decoder_recovery_rate,    # Δ-success after decoder vs argmax
    astar_expansions_ratio,   # search-effort ratio vs A* oracle
)
```

GPU is detected at runtime via `torch.cuda.is_available()`; on
CPU-only hosts the throughput cells fall back gracefully and the
matrix reports CPU-side numbers.

```python
fwd_ms = forward_timing(model, sample_input, n_warmup=10, n_iters=100)
dec_ms = decoder_timing(decoder_fn, model_output, n_warmup=10, n_iters=100)
e2e    = end_to_end_percentiles(pipeline_fn, sample_input,
                                 n_warmup=10, n_iters=100)
print(e2e.p50, e2e.p95, e2e.p99)
```

End-to-end latency is what enters the headline table; forward and
decoder times are reported as a decomposition.

---

## Reproducibility sidecar

Every headline row carries a minimal sidecar that pins every degree
of freedom that can affect the reported number, and emits a
deterministic content-hash so reviewers can map any score back to
the exact artefact and configuration:

```python
from passage_eval.sidecar import (
    ReproducibilitySidecar,
    compute_calibration_decisions_hash,
)

sidecar = ReproducibilitySidecar(
    dataset_version="passage-1.1",
    seeds=(1234, 2026, 3407),
    calibration_decisions={"threshold": 0.5, "bridge_to_goal": True},
    hardware={"gpu": "H100", "cpu": "Xeon Platinum 8480"},
    inference={"batch_size": 1, "precision": "fp16"},
    determinism="algorithmically_deterministic",
)
print(sidecar.calibration_hash)        # e.g. 'a3f7…'
sidecar.write("results/seg_baseline.sidecar.json")
```

The deterministic hash is what an auditor reproduces to verify the
test pass used calibration-time values.

---

## Worked example

A complete worked example — from a trained planner's predictions to
a paper-comparable headline row — is in the
[end-to-end walkthrough](walkthrough.md).

The notebook
[`notebooks/_passage_eval_metrics.ipynb`](https://github.com/ThalesGroup/passage/blob/main/notebooks/_passage_eval_metrics.ipynb)
runs every API in this module against synthetic ground truth and
produces the diagnostic plots that back the figures in the paper.

```bash
pip install -e ./src/passage-eval[notebook]
jupyter notebook notebooks/_passage_eval_metrics.ipynb
```

---

## Source-data acknowledgement

`passage-eval` is licensed under the project's MIT License. The
benchmark itself is built on the **JAXA ALOS World 3D 30 m DSM**;
any user of **PASSAGE** or its derived artefacts must respect
JAXA's licensing terms — see the
[JAXA license notice](../index.md#jaxa-license-and-attribution) on
the home page and `DATASET_LICENSE.md` in the repository root.

---

## Cross-references

- [Evaluation protocol](protocol.md) — formal rules.
- [End-to-end walkthrough](walkthrough.md) — code from prediction to headline.
- [API reference](api.md) — auto-generated docstrings.
- Source: [`src/passage-eval/`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval).
- Tests: 49 unit tests in `src/passage-eval/tests/` — run with
  `pytest src/passage-eval/tests -q`.
