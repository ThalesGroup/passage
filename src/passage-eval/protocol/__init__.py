# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
    PrimaryMetricComparison,
    apply_censored_penalty,
    compare_primary_metrics,
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
    "PrimaryMetricComparison",
    "compare_primary_metrics",
    "apply_censored_penalty",
    "decide_pairwise",
    "holm_bonferroni",
    "multiseed_paired_bootstrap",
    "paired_bootstrap_delta",
]
