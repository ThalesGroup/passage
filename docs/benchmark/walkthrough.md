---
title: End-to-End Walkthrough
icon: material/rocket-launch
---

# End-to-end walkthrough — evaluating a planner

This page is a complete worked example. We assume you have:

1. A **trained planner** (any family — vision, GNN, RL, classical).
2. A **canonical decoder** that turns the planner's intermediate
   output into a path of grid cells.
3. A way to run that pipeline on each **PASSAGE** test sample.

By the end of this page you will have:

- A six-row headline metric block.
- A paired-bootstrap $95\%$ CI on the mean of paired differences
  vs. a baseline.
- The Holm–Bonferroni-adjusted decision (Pareto-dominant /
  Trade-off / Indistinguishable).
- A reproducibility sidecar JSON file pinning every degree of
  freedom.

---

## 0. Prerequisites

```bash
pip install -e ./src/passage-eval[notebook]
```

The walkthrough below runs from a notebook or a stand-alone Python
script — both are equivalent.

---

## 1. Run your planner on the test split

For each of the three pre-registered seeds
$\{1234,\,2026,\,3407\}$, run your planner end-to-end on the
**PASSAGE** test split (the South-America geographic hold-out
slice; see [Dataset Generation](../generation/index.md)).

For every sample $i$, your harness produces:

- `pred_path[i]` — a list of $(r, c)$ waypoints from $s$ to $g$.
- `pred_cost[i]` — the cost of `pred_path` under the same cost model.
- `oracle_path[i]`, `oracle_cost[i]` — already in the dataset shard.
- `success[i]` $\in \{0, 1\}$ — did the decoder reach $g$?
- `violation[i]` $\in \{0, 1\}$ — did `pred_path` cross any obstacle
  cell?

Persist these as one `.npz` per seed.

!!! tip "Decoder freezing (Rule 3)"
    Every decoder hyperparameter (threshold, step, schedule, …) was
    fixed on the calibration split before this run. Record the
    config under a deterministic hash so [Rule 6](protocol.md#rule-6-no-test-split-leakage)
    can be enforced.

---

## 2. Compute the six primary metrics per seed

```python
import numpy as np
from passage_eval.metrics import compute_primary_metrics
from passage_eval.protocol import apply_censored_penalty

per_seed_metrics = []
per_seed_arrays  = []

for seed in (1234, 2026, 3407):
    z = np.load(f"results/test_seed{seed}.npz")
    pred_cost   = z["pred_cost"]
    oracle_cost = z["oracle_cost"]
    success     = z["success"].astype(bool)
    violation   = z["violation"].astype(bool)
    assd        = z["assd"]
    h95         = z["hausdorff_p95"]

    # Censored-penalty rule on failed decodes (paper Rule, m = 2.0).
    pred_cost_p = apply_censored_penalty(pred_cost, success,
                                          oracle_cost, multiplier=2.0)

    per_seed_metrics.append(
        compute_primary_metrics(
            pred_cost=pred_cost_p, oracle_cost=oracle_cost,
            success=success, violation=violation,
            assd=assd, h95=h95,
        )
    )
    per_seed_arrays.append({
        "pred_cost": pred_cost_p, "oracle_cost": oracle_cost,
        "success":   success,     "violation":   violation,
        "assd":      assd,        "h95":         h95,
    })

# Three-seed mean for the headline display
import pandas as pd
df = pd.DataFrame(per_seed_metrics).agg(["mean", "std"])
print(df.round(4))
```

You now have the six headline metrics (mean over three seeds).

---

## 3. Paired bootstrap vs a baseline

Pre-register the baseline (e.g. the dense-vision regression
finalist `repvit_m1+Bellman+V_bridge`) and run it on the same test
split with the same three seeds.

```python
from passage_eval.protocol import multiseed_paired_bootstrap

# For each metric, we need (per_seed_pairs) where each entry is a
# tuple (per-sample metric for A, per-sample metric for B).
pairs_per_metric = {
    "cost_regret_median":      [],
    "cost_regret_p90":         [],
    "path_assd":               [],
    "path_hausdorff_p95":      [],
    "success_rate":            [],
    "obstacle_violation_rate": [],
}
# ... fill from per_seed_arrays for A (your planner) and B (baseline)

agg = {}
for m, pairs in pairs_per_metric.items():
    agg[m] = multiseed_paired_bootstrap(pairs, n_resamples=10_000)
```

Each `agg[m]` carries the pooled point estimate, $95\%$ CI on
$\Delta_m$, and the two-sided $p$-value for $\Delta_m = 0$.

---

## 4. Holm–Bonferroni and the decision

```python
from passage_eval.protocol import holm_bonferroni, decide_pairwise

p_values = {m: agg[m].p_value for m in agg}
hb = holm_bonferroni(p_values, alpha=0.05)

decision = decide_pairwise(agg, hb=hb)
print(decision)
# -> 'pareto_dominant' | 'trade_off' | 'indistinguishable'
```

If your planner is `pareto_dominant`, you may report it as a win on
every primary metric. If `trade_off`, you must report it with the
paired CI bands and a one-sentence narrative noting the lost
metrics. If `indistinguishable`, the claim is **rejected, not
rounded**.

---

## 5. Trustworthiness axes

Beyond the primary headline, run the four axes:

```python
import numpy as np

# 5.1 Stability — three-seed CV of regret_p90
regret_p90_seeds = np.array([m["cost_regret_p90"] for m in per_seed_metrics])
stability_cv = regret_p90_seeds.std() / regret_p90_seeds.mean()

# 5.2 Geographic generalisation — paired delta against a matched random split
#    (run a separate evaluation on the matched-marginal random split, then
#     compute the paired Δ over the same per-sample arrays).

# 5.3 ODD-boundary behaviour — re-run the six metrics on the boundary subset
#    (extremal obstacle coverage, super-ellipse exponents in {1, 80},
#     elevation-band edges).

# 5.4 End-to-end efficiency — fall through to step 6 below.
```

---

## 6. Deployment footprint

```python
from passage_eval.deployment import (
    forward_timing, decoder_timing, end_to_end_percentiles,
    flops_and_params,
)

fwd_ms = forward_timing(model, sample_input, n_warmup=10, n_iters=100)
dec_ms = decoder_timing(decoder_fn, output, n_warmup=10, n_iters=100)
e2e    = end_to_end_percentiles(pipeline_fn, sample_input,
                                 n_warmup=10, n_iters=100)
flops, params = flops_and_params(model, sample_input)

print(f"forward p50:        {fwd_ms.p50:.3f} ms")
print(f"decoder p50:        {dec_ms.p50:.3f} ms")
print(f"end-to-end p50/p95: {e2e.p50:.3f} / {e2e.p95:.3f} ms")
print(f"params: {params:,}; MACs: {flops:.2e}")
```

End-to-end latency is the **headline** number for the deployment row;
forward and decoder times are reported as a decomposition.

---

## 7. Reproducibility sidecar

```python
from passage_eval.sidecar import ReproducibilitySidecar

sidecar = ReproducibilitySidecar(
    dataset_version="passage-1.1",
    seeds=(1234, 2026, 3407),
    calibration_decisions={
        "threshold":      0.5,
        "bridge_to_goal": True,
        "max_steps":      512,
    },
    hardware={"gpu": "H100", "cpu": "Xeon Platinum 8480"},
    inference={"batch_size": 1, "precision": "fp16"},
    determinism="algorithmically_deterministic",
)
sidecar.write("results/my_planner.sidecar.json")
print(sidecar.calibration_hash)        # e.g. 'a3f7…'
```

The sidecar must accompany every headline row. An auditor can
recompute the deterministic hash and verify the test pass used
calibration-time values.

---

## 8. What you submit

A complete **PASSAGE** headline submission carries:

- The **six primary metrics** with paired-bootstrap $95\%$ CIs (per
  seed and pooled).
- The **decision** vs the declared baseline (pareto-dominant /
  trade-off / indistinguishable).
- The **four trustworthiness axes** as point statistics.
- The **deployment-footprint row** for stack R1 (server) and stack
  R2 (edge), per [Rule 7](protocol.md#rule-7-decoding-is-part-of-inference)
  — end-to-end including the decoder.
- The **reproducibility sidecar** JSON.

If any of the above is missing, the entry is **separated from the
reproducible rows in the headline table** ([Rule 8](protocol.md#rule-8-tool-transparency)).

---

## Cross-references

- [Evaluation protocol](protocol.md) — formal rules.
- [`passage-eval` package](passage-eval.md) — module-by-module reference.
- [API reference](api.md) — auto-generated docstrings.
- Notebook walkthrough: [`notebooks/_passage_eval_metrics.ipynb`](https://github.com/ThalesGroup/passage/blob/main/notebooks/_passage_eval_metrics.ipynb).
- Tests as specification: 49 unit tests in `src/passage-eval/tests/`.
