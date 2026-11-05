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
    """Difference in the mean of a primary metric between the geographic
    hold-out split and a matched-marginal random split (axis "Geo").

    Both inputs are per-sample arrays for the *same metric*. The output
    ``value`` is ``mean(holdout) - mean(matched_random)``;
    ``higher_is_better`` sets which sign counts as worse.

    The axis fires when the hold-out mean is worse than the matched
    random mean. The flag uses the sign of the difference only; it is
    not a significance test.
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
