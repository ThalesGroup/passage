# MIT License - Copyright (c) 2026 THALES
"""Statistical-discipline primitives for the PASSAGE evaluation protocol.

This subpackage groups the rules of App. J ("Cross-Family Evaluation
Protocol") that operate on top of the per-sample primary metrics:

* default seed set and primary-metric ordering (§5.3 pre-registration);
* censored cost-regret penalty for failed decodes (§5.2);
* per-metric paired bootstrap on the mean of paired differences
  ($\\Delta_m = \\mathrm{mean}(m_A - m_B)$);
* three-seed aggregation by pooling resamples;
* Holm--Bonferroni correction across the six primary metrics; and
* the Pareto-dominant / Trade-off / Indistinguishable decision rule.

Each primitive is small, dependency-free (numpy only) and pure so a
submission can be re-scored from raw per-sample arrays without re-running
the network.
"""

from passage_eval.protocol.discipline import (
    DEFAULT_SEEDS,
    PRIMARY_METRICS,
    BootstrapDelta,
    HolmBonferroniResult,
    MetricFamily,
    PairwiseDecision,
    apply_censored_penalty,
    decide_pairwise,
    holm_bonferroni,
    multiseed_paired_bootstrap,
    paired_bootstrap_delta,
)

__all__ = [
    "DEFAULT_SEEDS",
    "PRIMARY_METRICS",
    "MetricFamily",
    "BootstrapDelta",
    "HolmBonferroniResult",
    "PairwiseDecision",
    "apply_censored_penalty",
    "decide_pairwise",
    "holm_bonferroni",
    "multiseed_paired_bootstrap",
    "paired_bootstrap_delta",
]
