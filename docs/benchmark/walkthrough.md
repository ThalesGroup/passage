---
title: Paired statistics (paper)
icon: material/rocket-launch
---

# Paired statistics (paper)

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-20 | 1.0 |

This page applies the paper's statistical comparison to two planners. Each
planner has three trained checkpoints (seeds 1234, 2026 and 3407), and both
were scored on the same test tasks. The example checks that the two result
sets pair up task by task, bootstraps the mean of the paired differences for
each metric, pools the three seeds, applies the Holm–Bonferroni correction and
reads the decision: Pareto-dominant, trade-off or indistinguishable. Use it to
run the paper's mean-difference analysis on your own results;
[the protocol](protocol.md) explains why one CLI row alone is not a paper
headline. To evaluate a planner and compare it with others, use
[Evaluate your planner](contributing-a-method.md) instead: its `compare` command
uses a different estimator, with intervals that resample whole geographic
blocks.

## 1. Freeze and run the candidates

Install from the root with `uv sync --frozen`. For each candidate, declare
three trained checkpoints in `inference_kwargs_by_seed`, pin the dataset commit,
freeze decoder decisions on calibration, and run its test spec once:

```bash
uv run --frozen passage-eval bench --spec candidate-A.test.yaml
uv run --frozen passage-eval bench --spec candidate-B.test.yaml
```

Each spec uses `split: test`, `cost_model: elevation`, `resolution: 256`, the same
sample cap, and seeds `[1234, 2026, 3407]`. Use separate fresh result roots.
A cap of 1024 is a subset campaign, not the full geographic test. Keep the
sidecars and all checkpoint/runtime artifacts.

## 2. Verify pairing and load the measured evidence

```python
import json
import numpy as np
from passage_eval import DEFAULT_SEEDS, load_per_sample_npz, primary_metrics_from_arrays

# Fill these with the three actual run directories for each candidate.
runs_A = {1234: "...", 2026: "...", 3407: "..."}
runs_B = {1234: "...", 2026: "...", 3407: "..."}
A, B = {}, {}
for seed in DEFAULT_SEEDS:
    for runs, destination in ((runs_A, A), (runs_B, B)):
        destination[seed] = load_per_sample_npz(f"{runs[seed]}/per_sample.npz")
    side_A = json.load(open(f"{runs_A[seed]}/sidecar.json"))
    side_B = json.load(open(f"{runs_B[seed]}/sidecar.json"))
    assert side_A["dataset_content_hash"] == side_B["dataset_content_hash"]
    assert side_A["operating_point"] == side_B["operating_point"]
    assert side_A["extra"]["metric_contract"] == side_B["extra"]["metric_contract"]
    assert np.array_equal(A[seed].extras["sample_id"], B[seed].extras["sample_id"])
    print(seed, primary_metrics_from_arrays(A[seed]))
```

The point metric dictionary includes median and p90 regret. Failed path costs
were already censored by the evaluator; their finite regret stays in the sample
arrays. Geometry is NaN for invalid paths.

## 3. Bootstrap and correct the mean paired differences

```python
from passage_eval import multiseed_paired_bootstrap, holm_bonferroni, decide_pairwise

# This example tests paired means, as stated by the paper's Δ_m definition.
# Median and p90 headline estimators need a separately declared bootstrap plan.
fields = {
    "regret_mean": ("regret", False),
    "path_assd": ("assd", False),
    "path_hausdorff_p95": ("h95", False),
    "success_rate": ("success", True),
    "obstacle_violation_rate": ("obstacle_violation", False),
}
records, p_values = [], {}
for metric, (field, higher) in fields.items():
    def values(bundle):
        return getattr(bundle, field) if hasattr(bundle, field) else bundle.extras[field]
    rec, draws = multiseed_paired_bootstrap(
        {s: values(A[s]) for s in DEFAULT_SEEDS},
        {s: values(B[s]) for s in DEFAULT_SEEDS},
        metric=metric, higher_is_better=higher,
        n_bootstrap_per_seed=10_000, return_draws=True,
    )
    records.append(rec)
    p_values[metric] = rec.two_sided_p_value(draws)
    print(metric, rec.delta, rec.ci_low, rec.ci_high, "paired n:", rec.n)
adjusted = holm_bonferroni(p_values, alpha=.05)
decision, votes = decide_pairwise(records, holm=adjusted)
print(decision.value, votes)
```

This five-test example is explicitly a mean-difference analysis, not the complete
six-metric paper family. Publication comparisons must use the six preregistered
estimators and their declared resampling plan. Geometry retains only jointly
finite candidate/baseline pairs; report those counts alongside each method's
success and violation rates. Do not replace unavailable comparisons with wins.

## 4. Complete the reporting campaign

Report the across-seed mean, per-seed estimates and intervals, and Holm decisions.
Add the stability, geographic versus matched-random, domain-boundary, and
same-device efficiency studies. Deploy on each real target to measure latency,
precision sensitivity, memory, parameter count, and forward/decoder decomposition.

Sidecars record the supplied calibration decisions and container digest, while
missing declarations remain visible. Publish a headline only with candidate
source, the three checkpoint artifacts, frozen data/runtime identities, the
analysis plan, and the measured campaign evidence. Paper results cannot supply
missing measurements for a new method.
