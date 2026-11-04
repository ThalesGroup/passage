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

    Reported as median, p90 and geometric mean. ``< 1``
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

    This legacy measure excludes only the explicitly reported ``fallback``
    stage. It does not receive a bounded-repair flag and therefore cannot
    establish that a successful decoder ran without repair. Use the scored
    ``direct_success_rate`` and ``repair_rate`` to distinguish that case.
    """
    succ = np.asarray(decoder_succeeded, dtype=bool)
    fb = np.asarray(fallback_triggered, dtype=bool)
    if succ.shape != fb.shape:
        raise ValueError(f"Shape mismatch: succeeded {succ.shape} vs fallback {fb.shape}.")
    if succ.size == 0:
        return float("nan")
    return float(np.mean(succ & ~fb))
