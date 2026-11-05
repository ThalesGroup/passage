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

"""Stability axis: coefficient of variation of CR-p90 across seeds."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from passage_eval.trustworthiness.axis import TrustworthinessAxis


__all__ = ["stability_cv"]


def stability_cv(
    cost_regret_p90_per_seed: Sequence[float],
    *,
    cv_threshold: float = 0.25,
) -> TrustworthinessAxis:
    """Coefficient of variation $\\sigma / \\mu$ of the per-seed
    ``cost_regret_p90`` (axis "Stab").

    A planner whose three-seed CR-p90 has a CV above ``cv_threshold``
    is flagged as training-unstable. The default ``cv_threshold`` is
    0.25.
    """
    arr = np.asarray(list(cost_regret_p90_per_seed), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return TrustworthinessAxis(
            name="Stab",
            metric="cost_regret_p90",
            value=float("nan"),
            flag=False,
            n_samples=0,
        )
    mean = float(arr.mean())
    std = float(arr.std(ddof=0))
    if mean == 0.0:
        cv = float("nan") if std == 0.0 else float("inf")
    else:
        cv = std / mean
    flag = bool(np.isfinite(cv) and cv > cv_threshold)
    return TrustworthinessAxis(
        name="Stab",
        metric="cost_regret_p90",
        value=cv,
        flag=flag,
        n_samples=int(arr.size),
        extras={"mean": mean, "std": std, "cv_threshold": cv_threshold},
    )
