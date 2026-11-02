# MIT License
#
# Copyright (c) 2026 THALES

"""Four trustworthiness axes of the PASSAGE paper.

A planner that is on average accurate can still fail operationally. The
PASSAGE protocol re-applies the six primary metrics under four
conditions the headline does not capture:

* ``stability``: coefficient of variation $\\sigma / \\mu$ of
  ``cost_regret_p90`` across the three pre-registered training seeds.
  Above 25% flags a training-unstable planner.
* ``geographic``: difference in mean between the geographic hold-out
  and a matched-marginal random split.
* ``odd_boundary``: performance at the declared envelope edge
  (extremal obstacle coverage, super-ellipse exponents, elevation
  bands).
* ``efficiency``: same-device latency relative to the
  A$^{\\star}$ oracle when the comparison is hardware-comparable.

Each module exposes a small, numpy-only function that takes the
per-sample arrays produced by the released baselines (see
``passage_eval.io``) and emits a single
``TrustworthinessAxis`` row.
"""

from passage_eval.trustworthiness.axis import TrustworthinessAxis
from passage_eval.trustworthiness.stability import stability_cv
from passage_eval.trustworthiness.geographic import geographic_generalisation
from passage_eval.trustworthiness.odd_boundary import odd_boundary_delta
from passage_eval.trustworthiness.efficiency import oracle_relative_efficiency

__all__ = [
    "TrustworthinessAxis",
    "stability_cv",
    "geographic_generalisation",
    "odd_boundary_delta",
    "oracle_relative_efficiency",
]
