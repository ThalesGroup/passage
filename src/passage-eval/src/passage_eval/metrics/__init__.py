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
