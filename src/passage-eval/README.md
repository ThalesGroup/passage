# passage-eval

Canonical evaluation contract for the
[PASSAGE](https://huggingface.co/datasets/thalesgroup/passage) benchmark.
Pip-installable, decoupled from the dataset-generation package
(`passage`) so the evaluation stack can be patched without invalidating
released data artefacts.

📚 **Full usage walk-through**:
<https://thalesgroup.github.io/passage/guide/evaluation/>  
🧪 **Notebook**: [`notebooks/_passage_eval_metrics.ipynb`](https://github.com/ThalesGroup/passage/blob/main/notebooks/_passage_eval_metrics.ipynb)  
🧰 **API reference**: <https://thalesgroup.github.io/passage/api/evaluation/>

## Quick install (from the repo)

```bash
git clone https://github.com/ThalesGroup/passage.git
cd passage
pip install -e ./src/passage-eval               # core only
pip install -e ./src/passage-eval[notebook]     # + matplotlib + tqdm + ipykernel
pip install -e ./src/passage-eval[torch]        # + GPU throughput utilities
pip install -e ./src/passage-eval[dev]          # + pytest + ruff
python -c "import passage_eval; print(passage_eval.__version__)"   # -> 0.2.0
pytest src/passage-eval/tests -q                                   # -> 49 passed
```


## What it provides

- **Primary decoded-path metrics** (six headline metrics under the
  Holm--Bonferroni umbrella in §5.2 of the PASSAGE paper):
  `cost_regret_median`, `cost_regret_p90`, `path_assd`,
  `path_hausdorff_p95`, `success_rate`, `obstacle_violation_rate`.
- **Statistical discipline** (§5.3 / App. J): pre-registered seed set
  `DEFAULT_SEEDS = (1234, 2026, 3407)`, censored-penalty helper for
  failed decodes (`apply_censored_penalty`), paired-bootstrap CI on the
  mean of paired differences (`paired_bootstrap_delta`), three-seed
  pooled aggregation (`multiseed_paired_bootstrap`), Holm--Bonferroni
  step-down correction (`holm_bonferroni`), and the
  Pareto-dominant / Trade-off / Indistinguishable decision rule
  (`decide_pairwise`).
- **Reproducibility sidecar** (App. J Rule 8): `ReproducibilitySidecar`
  manifest + deterministic `compute_calibration_decisions_hash` so every
  headline row can be mapped back to its dataset version, seeds,
  calibration decisions, hardware stack, batch size, dtype, and
  determinism flag.
- **Decoder-readiness diagnostics** (family-specific, never headline):
  `mask_endpoints_in_same_component`, `descent_direction_agreement`,
  `bellman_residual_reachable`, `local_ordinal_concordance`,
  `field_ssi_mae`, `boundary_f_score`.
- **Cost-model evaluation** for the five PASSAGE physical cost models
  (`distance`, `elevation`, `absolute_elevation`, `slope_uphill`,
  `energy`).
- **Inference benchmark utilities**: forward-pass timing, decoder timing,
  end-to-end percentile latency, throughput, peak GPU memory,
  parameter and FLOPs accounting, A\*-expansions ratio,
  decoder-recovery rate. The `throughput` module assumes a CUDA host but
  falls back to CPU timing when no GPU is visible.

## Install

```bash
pip install passage-eval
# or, with PyTorch helpers:
pip install passage-eval[torch]
```

## Usage

```python
import numpy as np
from passage_eval import (
    DEFAULT_SEEDS,
    PRIMARY_METRICS,
    apply_censored_penalty,
    compute_primary_metrics,
    cost_regret,
    decide_pairwise,
    holm_bonferroni,
    multiseed_paired_bootstrap,
    ReproducibilitySidecar,
    compute_calibration_decisions_hash,
)

# 1. Per-method, per-sample metric arrays come out of your decoder.
predicted_paths_a = [...]  # method A
predicted_paths_b = [...]  # method B
oracle_paths      = [...]
oracle_costs      = np.array([...])

# 2. Censor failed decodes with the canonical K * c* penalty.
predicted_costs_a = apply_censored_penalty(
    predicted_costs_a_raw, oracle_costs,
    decoder_succeeded=succ_a, obstacle_violated=viol_a, k_factor=1e6,
)

# 3. Compute the primary-metric bundle for each method.
bundle_a = compute_primary_metrics(
    predicted_paths=predicted_paths_a,
    oracle_paths=oracle_paths,
    predicted_costs=predicted_costs_a,
    oracle_costs=oracle_costs,
    obstacles=obstacles,
)

# 4. Three-seed paired bootstrap on every primary metric.
deltas, draws = [], {}
for name, family, higher_is_better in PRIMARY_METRICS:
    rec, d = multiseed_paired_bootstrap(
        per_sample_a_by_seed[name],   # {seed: per-sample array}
        per_sample_b_by_seed[name],
        metric=name, higher_is_better=higher_is_better,
        seeds=DEFAULT_SEEDS, n_bootstrap_per_seed=10_000,
        return_draws=True,
    )
    deltas.append(rec)
    draws[name] = d

# 5. Holm--Bonferroni across the six primary metrics.
p_values = {d.metric: d.two_sided_p_value(draws[d.metric]) for d in deltas}
holm = holm_bonferroni(p_values, alpha=0.05)

# 6. Reduce to the Pareto-dominant / Trade-off / Indistinguishable verdict.
verdict, votes = decide_pairwise(deltas, holm=holm)
print(verdict)

# 7. Reproducibility sidecar (one per headline row).
sidecar = ReproducibilitySidecar(
    method_name="dense_regression_finalist",
    family="dense_regression",
    dataset_version="v2.0.0",
    dataset_content_hash="sha256:...",
    operating_point="256x256/elevation",
    seeds=list(DEFAULT_SEEDS),
    calibration_decisions_hash=compute_calibration_decisions_hash({
        "threshold": 0.5, "step_size": 1, "schedule": "cosine",
    }),
    hardware_stack="R1",
    batch_size=1,
    dtype="fp16",
    determinism_flag="algorithmically_deterministic",
)
sidecar.write("results/headline/dense_regression_finalist.json")
```

## Why is this a separate package from `passage`?

The PASSAGE dataset on Hugging Face is pinned to a frozen `passage`
release. The evaluation contract — what counts as a "headline number" —
must be patchable across submissions without invalidating the released
data. Decoupling lets reviewers and downstream users install
`passage-eval` to score their own checkpoints without pulling in the
data-generation toolchain (rasterio, networkx, pathfinding, JAXA tile
download paths).
