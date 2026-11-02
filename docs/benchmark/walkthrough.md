---
title: Paired statistics
icon: material/rocket-launch
---

# Paired statistics

| Author | Date | Version |
|---|---|---|
| Geoffrey Delhomme | 2026-10-24 | 1.1 |

This page compares two planners on the six primary decoded-path metrics. Each
planner has three trained checkpoints (seeds 1234, 2026 and 3407), scored on the
same frozen test tasks. It checks task pairing, estimates each metric contrast
under shared geographic-block draws, and explains the separate evidence needed
for a confirmatory decision. [The protocol](protocol.md) defines the estimators
and reporting scope. To produce the timed, qualified-reference results used
here, start with [Evaluate your planner](contributing-a-method.md).

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
A cap of 1024 is a subset campaign. Keep the sidecars, manifests, score receipts,
checkpoint artifacts and runtime identities. Check that the methods share the
data, task conditions and declared work limits required by the analysis plan.

## 2. Load the measured evidence and verify pairing

The following example reads the warm scored run directory for each checkpoint.
It selects the frozen primary quality repeat from the complete-request arrays;
repeated timed routes must not become extra independent tasks.

```python
from pathlib import Path
import numpy as np
from passage_eval import DEFAULT_SEEDS, compare_primary_metrics

# Replace these with the actual warm run directories containing per_sample.npz.
runs_A = {1234: "...", 2026: "...", 3407: "..."}
runs_B = {1234: "...", 2026: "...", 3407: "..."}


def load_primary(run_directory):
    with np.load(Path(run_directory) / "per_sample.npz", allow_pickle=False) as data:
        assert data["metric_contract"].item() == "passage-qualified-complete-v1"
        assert data["reference_qualified"].all()
        selected = data["primary_quality"]
        assert selected.dtype.kind == "b"
        assert np.array_equal(selected, data["repeat_index"] == data["quality_repeat_index"].item())
        columns = {
            field: data[field][selected].copy()
            for field in ("regret", "assd", "h95", "success",
                          "obstacle_violation", "valid_path")
        }
        columns["task_id"] = data["sample_id"][selected].copy()
        blocks = data["cluster_id"][selected].copy()
    return columns, blocks


A, B, geographic_blocks = {}, {}, None
for training_seed in DEFAULT_SEEDS:
    A[training_seed], blocks_A = load_primary(runs_A[training_seed])
    B[training_seed], blocks_B = load_primary(runs_B[training_seed])
    assert np.array_equal(A[training_seed]["task_id"], B[training_seed]["task_id"])
    assert np.array_equal(blocks_A, blocks_B)
    if geographic_blocks is None:
        geographic_blocks = blocks_A
    else:
        assert np.array_equal(geographic_blocks, blocks_A)
```

These checks supplement the manifest and sidecar review; equal sample IDs alone
do not establish equal data or task conditions. `compare_primary_metrics` also
checks the exact ordered task roster across both methods and all seeds. It
requires at least two known geographic blocks. Unknown block identities cannot
be replaced with task IDs to create an uncertainty estimate.

Failed paths retain their finite censored regret. Reliability metrics use every
task. Geometry uses each method's own valid-path subset, so report both methods'
retained counts as well as their success and obstacle-violation rates.

## 3. Estimate all six metric contrasts

```python
comparison = compare_primary_metrics(
    A, B, blocks=geographic_blocks,
    n_bootstrap=10_000, alpha=0.05, seed=1234,
)
for metric, record in comparison.deltas.items():
    print(metric, record.delta, record.ci_low, record.ci_high,
          "finite outcomes A/B:", record.n_a, record.n_b)
    for training_seed, seed_record in record.per_seed.items():
        print(training_seed, seed_record.delta,
              seed_record.ci_low, seed_record.ci_high,
              seed_record.n_a, seed_record.n_b)
print(comparison.inference_status)  # test_not_provided
```

For each metric, the contrast is `T(A) - T(B)`. The prescribed reduction `T`
is median or p90 for regret, and mean for geometry and reliability. Each draw
resamples whole geographic blocks, identically for both methods, every training
seed and every metric. The API computes each seed's contrast inside that draw,
then averages the three contrasts. The interval uses 10,000 draws of this seed
mean and is conditional on the three fitted checkpoints.

A geometry draw with an empty valid subset makes its interval unavailable;
the API retains undefined draws. Its per-seed and aggregate outcome counts make
that coverage visible. These descriptive estimates do not supply a confirmatory
p-value or establish superiority.

## 4. Supply the declared paired-test evidence

If the analysis plan includes a valid paired test of the same geographic-block,
seed-mean contrast, supply its six measured p-values and the declared test and
assumptions. The file below is an input produced by that separate analysis,
not by the bootstrap interval helper:

```python
import json

# Required fields: p_values (all six primary metric names),
# paired_test (the declared test), and test_assumptions (its justification).
test_evidence = json.loads(Path("paired-test-evidence.json").read_text())
confirmed = compare_primary_metrics(
    A, B, blocks=geographic_blocks,
    n_bootstrap=10_000, alpha=0.05, seed=1234,
    p_values=test_evidence["p_values"],
    paired_test=test_evidence["paired_test"],
    test_assumptions=test_evidence["test_assumptions"],
)
print(confirmed.inference_status, confirmed.decision.value, confirmed.votes)
```

The API applies Holm–Bonferroni across all six tests at family-wise alpha 0.05.
An unavailable test is `NaN` and keeps its place in that family. The API checks
availability and input shape; a test name does not establish its validity.
Bootstrap tail fractions and confidence-interval proxies cannot replace the
paired-test evidence. Claims over additional method pairs need their own
prespecified correction family.

A Pareto decision requires demonstrated wins on every primary metric. Wins in
both directions are a trade-off; wins in one direction with inconclusive metrics
are partial evidence. No demonstrated difference does not establish equivalence.

## 5. Complete the reporting campaign

Report per-seed estimates and intervals, the mean of the three seed contrasts,
method-specific geometry coverage, the block definition, resampling seed, and
all test assumptions and Holm decisions. Add the stability, geographic versus
matched-random, domain-boundary, and same-device efficiency studies. Deploy on
each real target to measure latency, precision sensitivity, memory, parameter
count, and forward/decoder decomposition.

Sidecars record the hash of supplied calibration decisions and the container
digest. Retain the original calibration record separately. Publish a headline
with candidate source, the three checkpoint artifacts, frozen data and runtime
identities, the analysis plan, and the measured campaign evidence. The separately released paper results remain unchanged and cannot
supply missing measurements for a new method.
