# MIT License - Copyright (c) 2026 THALES
"""Surrogate-acceleration metrics."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def astar_expansions_ratio(
    expansions_pred: Sequence[int] | np.ndarray,
    expansions_oracle: Sequence[int] | np.ndarray,
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """expansions(decoded planner on V_hat) / expansions(plain A* on true cost).

    Reported as median + p90 + geomean per the v2 metric set. ``< 1``
    means the surrogate accelerates A*; ``≈ 1`` means no acceleration;
    ``> 1`` means the surrogate adds search effort.
    """
    pred = np.asarray(expansions_pred, dtype=np.float64)
    orac = np.asarray(expansions_oracle, dtype=np.float64)
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape} vs oracle {orac.shape}.")
    valid = np.isfinite(pred) & np.isfinite(orac) & (orac > eps)
    if not bool(valid.any()):
        return {"median": float("nan"), "p90": float("nan"), "geomean": float("nan"), "n_valid": 0}
    ratio = pred[valid] / orac[valid]
    log_ratio = np.log(np.clip(ratio, a_min=eps, a_max=None))
    return {
        "median": float(np.median(ratio)),
        "p90": float(np.percentile(ratio, 90.0)),
        "geomean": float(np.exp(float(np.mean(log_ratio)))),
        "n_valid": int(valid.sum()),
    }


def decoder_recovery_rate(
    decoder_succeeded: Sequence[bool] | np.ndarray,
    fallback_triggered: Sequence[bool] | np.ndarray,
) -> float:
    """Fraction of samples whose canonical decoder produced a path
    *without* triggering the A*-recovery fallback.

    A regression model whose `decoder_recovery_rate` is high is
    "field-ready" — its descent decoder works on its own. A low value
    means the network needed help from A*.
    """
    succ = np.asarray(decoder_succeeded, dtype=bool)
    fb = np.asarray(fallback_triggered, dtype=bool)
    if succ.shape != fb.shape:
        raise ValueError(f"Shape mismatch: succeeded {succ.shape} vs fallback {fb.shape}.")
    if succ.size == 0:
        return float("nan")
    return float(np.mean(succ & ~fb))
