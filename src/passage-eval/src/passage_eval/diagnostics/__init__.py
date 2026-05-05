# MIT License - Copyright (c) 2026 THALES
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
