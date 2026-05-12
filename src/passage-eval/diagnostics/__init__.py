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

"""Family-symmetric decoder-readiness diagnostics.

Segmentation diagnostics live in ``segmentation``; regression
diagnostics in ``regression``. None of them carries a headline claim.
"""

from passage_eval.diagnostics.segmentation import (
    boundary_f_score,
    mask_connected_components_total,
    mask_endpoints_in_same_component,
)
from passage_eval.diagnostics.regression import (
    bellman_residual_reachable,
    descent_direction_agreement,
    field_ssi_mae,
    local_ordinal_concordance,
)

__all__ = [
    "mask_connected_components_total",
    "mask_endpoints_in_same_component",
    "boundary_f_score",
    "descent_direction_agreement",
    "bellman_residual_reachable",
    "local_ordinal_concordance",
    "field_ssi_mae",
]
