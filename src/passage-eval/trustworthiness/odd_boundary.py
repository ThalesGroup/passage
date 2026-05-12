# MIT License
#
# Copyright (c) 2026 THALES

"""ODD-boundary axis: performance regression at the envelope edge."""
from __future__ import annotations

import numpy as np

from passage_eval.trustworthiness.axis import TrustworthinessAxis


__all__ = ["odd_boundary_delta"]


def odd_boundary_delta(
    metric_per_sample: np.ndarray,
    stratifier_per_sample: np.ndarray,
    *,
    top_quantile: float = 0.10,
    metric_name: str = "cost_regret_p90",
    stratifier_name: str = "obstacle_coverage",
    higher_is_better: bool = False,
) -> TrustworthinessAxis:
    """Difference between the top-quantile (boundary) slice and the
    full-split mean of a primary metric (§5.4 / App. K, axis "ODDe").

    ``stratifier_per_sample`` carries one of the four declared per-sample
    ODD axes (obstacle coverage, super-ellipse exponent, elevation
    band, start-goal distance). Samples whose stratifier sits in the
    top ``top_quantile`` are designated the "boundary" slice. The axis
    value is

        $\\Delta = \\mathrm{mean}(\\text{boundary}) - \\mathrm{mean}(\\text{full})$.

    For smaller-is-better metrics, a positive $\\Delta$ means the
    planner *worsens* at the envelope edge — the axis fires.
    """
    metric_arr = np.asarray(metric_per_sample, dtype=np.float64)
    strat_arr = np.asarray(stratifier_per_sample, dtype=np.float64)
    if metric_arr.shape != strat_arr.shape:
        raise ValueError(
            f"metric/stratifier shape mismatch: {metric_arr.shape} vs {strat_arr.shape}"
        )
    valid = np.isfinite(metric_arr) & np.isfinite(strat_arr)
    metric_arr = metric_arr[valid]
    strat_arr = strat_arr[valid]
    if metric_arr.size == 0:
        return TrustworthinessAxis(
            name="ODDe",
            metric=metric_name,
            value=float("nan"),
            flag=False,
            n_samples=0,
        )
    cutoff = float(np.quantile(strat_arr, 1.0 - top_quantile))
    boundary_mask = strat_arr >= cutoff
    boundary = metric_arr[boundary_mask]
    full_mean = float(metric_arr.mean())
    if boundary.size == 0:
        return TrustworthinessAxis(
            name="ODDe",
            metric=metric_name,
            value=float("nan"),
            flag=False,
            n_samples=0,
            extras={"stratifier": stratifier_name, "top_quantile": top_quantile},
        )
    boundary_mean = float(boundary.mean())
    delta = boundary_mean - full_mean
    # Numerical tolerance: flag only when the drift exceeds float64 noise.
    eps = 1e-12 * max(1.0, abs(full_mean))
    worse = (delta > eps) if not higher_is_better else (delta < -eps)
    return TrustworthinessAxis(
        name="ODDe",
        metric=metric_name,
        value=delta,
        flag=bool(worse),
        n_samples=int(boundary.size),
        extras={
            "stratifier": stratifier_name,
            "top_quantile": top_quantile,
            "boundary_mean": boundary_mean,
            "full_mean": full_mean,
            "cutoff": cutoff,
            "higher_is_better": higher_is_better,
        },
    )
