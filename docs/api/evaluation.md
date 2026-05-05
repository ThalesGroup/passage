---
title: API — `passage-eval`
icon: material/clipboard-check-outline
---

# `passage-eval` API reference

Auto-generated documentation for every public symbol exported by the
[`passage-eval`](https://github.com/ThalesGroup/passage/tree/main/src/passage-eval)
sub-package. Usage walk-throughs live on the
[Evaluation guide](../guide/evaluation.md); this page is the canonical
docstring view.

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

Six headline decoded-path metrics under the Holm--Bonferroni umbrella
in §5.2 of the PASSAGE paper.

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

Pre-registered seed set, paired-bootstrap CI on the mean of paired
differences, three-seed pooled aggregation, Holm--Bonferroni step-down
correction, and the family-wise decision rule.

::: passage_eval.protocol.discipline
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Trustworthiness diagnostics — `passage_eval.diagnostics`

Family-specific decoder-readiness checks. **Never headline.**

::: passage_eval.diagnostics
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Deployment footprint — `passage_eval.deployment`

Wall-clock latency, decoder timing, end-to-end percentiles, FLOPs +
parameter count, A$^{\star}$-expansions ratio, and decoder-recovery
rate.

::: passage_eval.deployment
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]

---

## Reproducibility sidecar — `passage_eval.sidecar`

Manifest writer + deterministic calibration-decisions hash so every
reported number can be mapped back to its dataset version, seeds,
calibration decisions, hardware stack, batch size, dtype, and
determinism flag.

::: passage_eval.sidecar
    options:
      show_root_heading: true
      members_order: source
      show_source: false
      filters: ["!^_"]
