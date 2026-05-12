# MIT License
#
# Copyright (c) 2026 THALES

"""Geographic generalisation axis: paired delta hold-out vs random split."""
from __future__ import annotations

import numpy as np

from passage_eval.trustworthiness.axis import TrustworthinessAxis


__all__ = ["geographic_generalisation"]


def geographic_generalisation(
    holdout_per_sample: np.ndarray,
    matched_random_per_sample: np.ndarray,
    *,
    metric_name: str = "cost_regret_p90",
    higher_is_better: bool = False,
) -> TrustworthinessAxis:
    """Paired difference of a primary metric between the geographic
    hold-out split and a matched-marginal random split (§5.4 / App. K,
    axis "Geo").

    Both inputs are per-sample arrays for the *same metric*. The output
    ``value`` is ``mean(holdout) - mean(matched_random)`` (i.e.\\ the
    sign matches the metric's "better-is-..." convention as stated by
    ``higher_is_better``).

    The axis fires when the hold-out is meaningfully worse than the
    matched random sample (below the bootstrap CI on the difference is
    a stricter check the user can run separately via
    :func:`passage_eval.protocol.discipline.paired_bootstrap_delta`).
    """
    a = np.asarray(holdout_per_sample, dtype=np.float64)
    b = np.asarray(matched_random_per_sample, dtype=np.float64)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return TrustworthinessAxis(
            name="Geo",
            metric=metric_name,
            value=float("nan"),
            flag=False,
            n_samples=int(min(a.size, b.size)),
        )
    delta = float(a.mean() - b.mean())
    # Worse-side: positive when smaller-is-better metrics regress on hold-out.
    worse = (delta > 0.0) if not higher_is_better else (delta < 0.0)
    return TrustworthinessAxis(
        name="Geo",
        metric=metric_name,
        value=delta,
        flag=bool(worse),
        n_samples=int(min(a.size, b.size)),
        extras={
            "holdout_mean": float(a.mean()),
            "matched_random_mean": float(b.mean()),
            "higher_is_better": higher_is_better,
        },
    )
