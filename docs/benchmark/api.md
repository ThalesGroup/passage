---
title: API Reference
icon: material/api
---

# `passage-eval` API reference

Auto-generated documentation for every public symbol exported by the
[`passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval)
sub-package. The narrative usage guide lives at the
[`passage-eval` package page](passage-eval.md); the formal protocol
sits at [Evaluation protocol](protocol.md); the confirmatory paired analysis API
is described at [Paired statistics](walkthrough.md). This page is the
canonical docstring view.

---

## Ordinary-input requests — `passage_eval.protocol.contracts`

Existing algorithms implement `plan(PlanRequest) -> PlanResult`. The request
contains candidate-visible terrain, endpoints, graph and shared budget; reference
labels stay evaluator-owned. See
[Evaluate your planner](contributing-a-method.md) for runnable specifications.

::: passage_eval.protocol.contracts.GraphSpec
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.protocol.contracts.BudgetSpec
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.protocol.contracts.PlanRequest
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.protocol.contracts.PlanResult
    options:
      show_root_heading: true
      show_source: false

---

## Complete requests and offline scores

::: passage_eval.execution.runner.run_requests
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.execution.scoring.score_run_artifact
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.metrics.comparison.compare_artifacts
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.metrics.comparison.paired_cluster_bootstrap
    options:
      show_root_heading: true
      show_source: false

---

## Timing profiles, latency censoring and batch bounds

The censoring rule and the re-warm profile are proposals, not sealed rules. The
developer guide `docs/benchmark/supervised-execution.md` in the source
checkout (sections 5.G to 5.I, 6.I and 6.J) explains them, including the separate report of validation and
recost time and the request limits that strict comparison requires; the [evaluation protocol](protocol.md) states the
proposed censoring rule.

::: passage_eval.execution.timing
    options:
      show_root_heading: true
      members: false
      show_source: false

::: passage_eval.execution.timing.manifest_warmup
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.metrics.latency.censored_latency_quantiles
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.metrics.latency.validation_recost_summary
    options:
      show_root_heading: true
      show_source: false

::: passage_eval.protocol.batching.check_batch_plan
    options:
      show_root_heading: true
      show_source: false

---

## Top-level package

::: passage_eval
    options:
      show_root_heading: true
      show_root_full_path: false
      members: false
      show_source: false

---

## Primary metrics — `passage_eval.metrics`

The paper's six headline decoded-path metrics, tested together under the
Holm–Bonferroni correction.

::: passage_eval.metrics.regret
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

::: passage_eval.metrics.geometric
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

::: passage_eval.metrics.reliability
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

::: passage_eval.metrics.summary
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Statistical discipline — `passage_eval.protocol`

Six primary metrics and three declared training seeds. The canonical
comparison estimates contrasts of metric aggregates with shared geographic-block
draws, then applies Holm–Bonferroni only to separately supplied paired-test
p-values. It reports partial evidence when remaining metrics are inconclusive.

::: passage_eval.protocol.discipline
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Decoder-readiness diagnostics — `passage_eval.diagnostics`

Checks that a predicted path mask or cost-to-go field is ready for its
decoder. They are diagnostics, **never headline metrics**.

::: passage_eval.diagnostics
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Trustworthiness axes — `passage_eval.trustworthiness`

Stability across training seeds, geographic generalization, behavior at the
edge of the operational design domain (ODD), and latency relative to the exact
A$^{\star}$ oracle on the same device. Each function returns one
`TrustworthinessAxis` record.

::: passage_eval.trustworthiness
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Deployment footprint — `passage_eval.deployment`

Wall-clock latency, decoder timing, end-to-end percentiles, parameter
count, A$^{\star}$-expansions ratio, and decoder-recovery rate.

::: passage_eval.deployment
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Reproducibility sidecar — `passage_eval.sidecar`

Manifest writer and deterministic calibration-decisions hash so every
reported number can be mapped back to its dataset version, seeds,
calibration decisions, hardware stack, batch size, dtype, and
determinism flag.

::: passage_eval.sidecar
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]
