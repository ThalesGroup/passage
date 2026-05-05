# MIT License - Copyright (c) 2026 THALES
"""Primary decoded-path metrics for the PASSAGE benchmark.

The six metrics are split across three families:

* ``regret``      cost-regret distribution and its robust summaries.
* ``geometric``   ASSD, p95-Hausdorff, plus path IoU as a diagnostic.
* ``reliability`` success rate and obstacle-violation rate guardrails.

The ``summary`` module aggregates them and runs paired-bootstrap CIs.
"""

from passage_eval.metrics.geometric import (
    path_assd,
    path_hausdorff_p95,
    path_iou_diagnostic,
)
from passage_eval.metrics.regret import (
    cost_regret,
    cost_regret_summary,
)
from passage_eval.metrics.reliability import (
    obstacle_violation_rate,
    obstacle_cells_traversed_per_path,
    success_rate,
)
from passage_eval.metrics.summary import (
    compute_primary_metrics,
    paired_bootstrap_ci,
)

__all__ = [
    "cost_regret",
    "cost_regret_summary",
    "path_assd",
    "path_hausdorff_p95",
    "path_iou_diagnostic",
    "success_rate",
    "obstacle_violation_rate",
    "obstacle_cells_traversed_per_path",
    "compute_primary_metrics",
    "paired_bootstrap_ci",
]
