---
title: Evaluation Protocol
icon: material/format-list-numbered
---

# Evaluation Protocol

This page distils the **eight rules** that govern the
\textsc{Passage} cross-approach evaluation protocol, plus the
operational contract for each primary metric, the statistical
discipline, the trustworthiness axes, and the deployment-footprint
section. The content is the dataset-side mirror of Appendix A of the
paper (`paper/neurips/main.tex`); the [`passage-eval`
package](passage-eval.md) and the [end-to-end
walkthrough](walkthrough.md) operationalise every rule below.

!!! abstract "Pre-registration"
    The rules and seeds below are **pre-registered**: they are fixed
    *before* any test-split number is observed. If you re-tune them
    after observing test-split results you have effectively rerun a
    different protocol, and reported numbers must be retracted.

---

## The eight rules

### Rule 1 — Common anchor

All headline metrics are computed on the **decoded path** $\hat{P}$
versus the exact A$^{\star}$ oracle path $P^{\star}$ on the same
sample under the same cost model. Different intermediate outputs
(logits, value fields, node policies, rollout policies) are allowed;
the **decoder** is what reconciles them.

The output is the path. The intermediate is not scored.

### Rule 2 — Declare the decoder in advance

Each entry declares its **canonical decoder** before any test-split
pass. Other decoders may be reported as ablations alongside the
canonical one, but the canonical decoder is what the headline number
quotes.

### Rule 3 — No test-split tuning

Every decoder hyperparameter (thresholds, step sizes, temperatures,
rollout horizons, …) is fixed on the **calibration split** (the
$2\%$ slice of the ODD declared in §3.3) and frozen for the test
pass. The test split is run **exactly once per seed**.

### Rule 4 — Diagnostics never carry the narrative

Approach-specific diagnostic metrics — segmentation BCE/Dice,
Bellman residual, considered-area ratio, value MSE, etc. —
**never** carry a headline claim. They live in the per-approach
appendix (or in a companion document).

### Rule 5 — CI overlap is not a win

If the $95\%$ CI on the paired-difference $\Delta_{m}$ contains zero,
$A$ and $B$ are *not distinguishable on $m$ at the declared level*.
Point-estimate rounding is **not accepted**.

### Rule 6 — No test-split leakage

Every calibration-split decision is stored under a deterministic hash
so an auditor can verify the test pass used calibration-time values.
A test-split number whose configuration can be traced back to a
validation-split tuning run is **retracted**.

### Rule 7 — Decoding is part of inference

Latency is measured **end-to-end including the decoder**. Model-only
latency is not accepted as an inference-footprint claim.

### Rule 8 — Tool transparency

Entries whose oracle, decoder, or inference runtime cannot be rebuilt
from a published commit and container digest are **separated** from
reproducible rows in the headline table. To enter the headline table,
an entry must declare its canonical decoder, calibration-split
decisions and their pinning hash, the per-row sidecar, and the
seed-replicated paired-bootstrap CIs.

---

## Primary metrics

Six **decoded-path** metrics under the Holm–Bonferroni umbrella:

| Metric | Symbol | Definition |
|---|---|---|
| Cost-regret median | `cost_regret_median` | $\text{median}_i\,(c(\hat{P}_i)/c(P^{\star}_i) - 1)$ |
| Cost-regret p90    | `cost_regret_p90` | 90$^{\text{th}}$ percentile equivalent |
| Average symmetric surface distance | `path_assd` | mean of mean-symmetric distances between $\hat{P}$ and $P^{\star}$ |
| Hausdorff p95 | `path_hausdorff_p95` | 95$^{\text{th}}$-percentile Hausdorff between $\hat{P}$ and $P^{\star}$ |
| Success rate | `success_rate` | fraction of decodes that connect $s$ to $g$ |
| Obstacle-violation rate | `obstacle_violation_rate` | fraction of decodes crossing at least one forbidden cell |

A **successful decode** has `success=1` *and* `obstacle_violation=0`.
The cost-regret, ASSD, and Hausdorff metrics are reported on the
successful subset alongside the success-rate value, so coverage stays
visible.

`passage_eval.metrics.compute_primary_metrics(...)` returns all six
in a single call; see the
[passage-eval package guide](passage-eval.md#primary-metrics).

---

## Statistical discipline

Pre-register, **before any test-split pass**, in this order:

1. **Operating point** — single headline resolution and cost model.
   \textsc{Passage} pins $256{\times}256$ and `elevation`.
2. **Seeds** — $\{1234,\ 2026,\ 3407\}$, with no "best of $k$".
   Available from `passage_eval.protocol.DEFAULT_SEEDS`.
3. **The six metrics above**, in declared order.
4. **The screened backbones / loss recipes** eligible to enter the
   test pass.
5. **The decision rules** (below).
6. **The analysis plan** (below).

### Paired bootstrap

For each pair $(A, B)$ of methods and each primary metric $m$, draw
$10^{4}$ resamples over the test index set, compute the per-resample
mean of the paired difference
$\Delta_{m} = \mathrm{mean}(m_{A} - m_{B})$, and report the $2.5\%$ /
$97.5\%$ quantiles as the $95\%$ CI on $\Delta_{m}$. The pairing
cancels sample-difficulty variance because both methods are evaluated
on the same indices.

### Three-seed aggregation

Each seed produces its own bootstrap CI; the reported CI is the
**composite** over the pooled $3 \times 10^{4}$ draws. Per-seed
point estimates and per-seed CIs are also reported.

### Holm–Bonferroni across the six primary metrics

Family-wise $\alpha = 0.05$ over the primary-metric set. A method
that wins on `cost_regret_median` but loses on `path_assd` or on the
obstacle-violation guardrail is reported as a **trade-off**, not a
win.

### Censored-penalty rule

Failed decodes (success $=$ 0) inherit a censored penalty:

$$\tilde{c}(\hat{P}_i) \;=\; m \cdot c(P^{\star}_i),\quad m = 2.0$$

before the cost-regret metrics are computed. This forbids a planner
from "winning" on the easy successes while silently dropping the
hard cases. `passage_eval.protocol.apply_censored_penalty(...)`
applies the rule.

---

## Decision rules

Method $A$ **beats** method $B$ on metric $m$ iff the $95\%$ CI on
$\Delta_{m}$ does not contain zero **and** the Holm–Bonferroni
adjusted $p$-value is below $\alpha = 0.05$ within the primary-metric
set. Three possible overall outcomes:

| Outcome | Condition |
|---|---|
| **Pareto-dominant** | $A$ wins on every primary metric; deployment-footprint axes relevant to the narrative are not strictly worse |
| **Trade-off** | $A$ wins on some metrics and loses on others; reported as a trade-off with paired-bootstrap $95\%$ CI bands |
| **Indistinguishable** | No metric crosses the Holm–Bonferroni bar; the claim is rejected, not rounded |

Deployment-footprint axes are reported alongside the primary metrics
with their own $95\%$ CIs but **do not** sit under the same
Holm–Bonferroni umbrella; a Pareto-dominant primary-metric winner
that is $10\times$ more expensive in latency is still reported as a
trade.

`passage_eval.protocol.decide_pairwise(...)` returns one of
`'pareto_dominant'`, `'trade_off'`, `'indistinguishable'`.

---

## Trustworthiness axes

Beyond the primary headline, a planner is also reported on four
**trustworthiness axes**. These re-apply the six primary metrics
under conditions the headline does not capture:

| Axis | What it measures |
|---|---|
| **Stability** | Coefficient of variation $\sigma/\mu$ of $\mathrm{regret}_{p_{90}}$ across the three seeds (above $25\%$ flags a training-unstable planner) |
| **Geographic generalisation** | Paired delta between the geographic hold-out (South-America test split) and a matched-marginal random split |
| **ODD-boundary behaviour** | Performance at the declared envelope edge (extremal obstacle coverage, super-ellipse exponents, elevation bands) |
| **End-to-end efficiency** | Same-device latency relative to the A$^{\star}$ oracle when that comparison is hardware-comparable |

Trustworthiness axes are **point-statistic** reports — they do not
sit under the Holm–Bonferroni umbrella but they are mandatory for any
entry to the headline table.

---

## Deployment footprint

Reported on a declared reference hardware stack, one row per method
per stack:

- Reference stack **R1**: a single H100-class GPU (server class).
- Reference stack **R2**: an embedded edge-GPU stack (AGX Thor, AGX
  Orin, or Tiger Lake-LP) for operational-relevance claims.

Per row:

- **Parameter count**, **MACs at the headline operating point**.
- **Peak runtime memory** (single-sample forward).
- **End-to-end latency** at $p_{50}$, $p_{95}$, $p_{99}$ over
  $\geq 100$ warm replicates at batch size 1.
- **Determinism flag** in $\{\text{bit-exact}, \text{algorithmically
  deterministic}, \text{non-deterministic}\}$.

`passage_eval.deployment.forward_timing(...)`,
`passage_eval.deployment.decoder_timing(...)`, and
`passage_eval.deployment.end_to_end_percentiles(...)` populate the
row; see the [passage-eval package
guide](passage-eval.md#inference-footprint).

---

## Reproducibility sidecar

Every headline row carries a minimal **sidecar** pinning:

1. Dataset version and content hash.
2. Seed set and split seed.
3. Deterministic hash of every calibration-split decision.
4. Reference hardware stack.
5. Inference batch size and numerical precision.
6. Determinism flag of the deployment-footprint section.

`passage_eval.sidecar.ReproducibilitySidecar(...)` writes the
sidecar; the deterministic hash is computed by
`passage_eval.sidecar.compute_calibration_decisions_hash(...)`.

---

## Cross-references

- Paper anchors:
  §3.3 (Operational Design Domain),
  §4.1 (Performance Metrics),
  §4.2 (Inference Metrics),
  §4.3 (Statistical Reporting),
  §4.4 (Trustworthiness Evaluation),
  Appendix A (Cross-Approach Evaluation Protocol),
  Appendix B (Worked Example).
- Implementation: [`passage-eval` package](passage-eval.md).
- Step-by-step example: [Walkthrough](walkthrough.md).
- API reference: [Auto-generated docstrings](api.md).
