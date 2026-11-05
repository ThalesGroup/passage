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

"""Oracle-relative efficiency axis."""

from __future__ import annotations

import numpy as np

from passage_eval.trustworthiness.axis import TrustworthinessAxis


__all__ = ["oracle_relative_efficiency"]


def oracle_relative_efficiency(
    planner_latency_ms_per_sample: float | np.ndarray,
    oracle_latency_ms_per_sample: float | np.ndarray,
    *,
    speedup_threshold: float = 1.0,
) -> TrustworthinessAxis:
    """Same-device latency ratio between planner and A$^{\\star}$ oracle
    (axis "Eff").

    The axis value is the *speed-up*
    $\\mathrm{speedup} = T_{\\text{oracle}} / T_{\\text{planner}}$.

    A speed-up below ``speedup_threshold = 1.0`` means the planner is
    slower than the oracle that produced its labels — the axis fires.

    Inputs may be scalars or arrays; arrays are summarised by their
    median.
    """
    pl = np.asarray(np.atleast_1d(planner_latency_ms_per_sample), dtype=np.float64)
    orc = np.asarray(np.atleast_1d(oracle_latency_ms_per_sample), dtype=np.float64)
    pl = pl[np.isfinite(pl) & (pl > 0)]
    orc = orc[np.isfinite(orc) & (orc > 0)]
    if pl.size == 0 or orc.size == 0:
        return TrustworthinessAxis(
            name="Eff",
            metric="speedup_vs_oracle",
            value=float("nan"),
            flag=False,
            n_samples=0,
        )
    pl_med = float(np.median(pl))
    orc_med = float(np.median(orc))
    speedup = orc_med / pl_med
    return TrustworthinessAxis(
        name="Eff",
        metric="speedup_vs_oracle",
        value=speedup,
        flag=bool(speedup < speedup_threshold),
        n_samples=int(min(pl.size, orc.size)),
        extras={
            "planner_latency_ms_p50": pl_med,
            "oracle_latency_ms_p50": orc_med,
            "speedup_threshold": speedup_threshold,
        },
    )
