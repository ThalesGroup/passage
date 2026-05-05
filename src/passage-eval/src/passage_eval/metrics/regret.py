# MIT License - Copyright (c) 2026 THALES
"""Cost-regret family.

Per-sample cost regret is

    regret_i = (cost_pred_i - cost_oracle_i) / cost_oracle_i

under the same cost model and ODD sample. Headline statistics are the
median and the 90-th percentile of the per-sample distribution; the
geometric mean and the 95-th percentile are kept as diagnostics.

The mean of cost_pred / cost_oracle is dominated by a long right tail
(under the canonical PASSAGE cost models a single obstacle violation
yields a 1e6 penalty, which ruins the mean). The robust statistics here
are the headline numbers for the paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


def cost_regret(
    predicted_costs: Sequence[float] | np.ndarray,
    oracle_costs: Sequence[float] | np.ndarray,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Per-sample cost regret. Returns NaN where the oracle cost is non-positive.

    Parameters
    ----------
    predicted_costs : array_like, shape (N,)
        Cost of the decoded predicted path under the cost model. Use ``np.nan``
        for samples where the decoder failed to produce a path.
    oracle_costs : array_like, shape (N,)
        Cost of the exact A* oracle path under the same cost model.
    eps : float
        Numerical floor for the denominator.
    """
    pred = np.asarray(predicted_costs, dtype=np.float64)
    orac = np.asarray(oracle_costs, dtype=np.float64)
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: predicted_costs {pred.shape} vs oracle_costs {orac.shape}.")
    out = np.full(pred.shape, np.nan, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(orac) & (orac > eps)
    out[valid] = (pred[valid] - orac[valid]) / orac[valid]
    return out


@dataclass(frozen=True)
class CostRegretSummary:
    median: float
    p90: float
    p95: float
    mean: float
    geomean: float
    n_valid: int
    n_nan: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "cost_regret_median": self.median,
            "cost_regret_p90": self.p90,
            "cost_regret_p95": self.p95,
            "cost_regret_mean": self.mean,
            "cost_regret_geomean": self.geomean,
            "cost_regret_n_valid": self.n_valid,
            "cost_regret_n_nan": self.n_nan,
        }


def cost_regret_summary(regret: np.ndarray) -> CostRegretSummary:
    """Return median / p90 / p95 / mean / geomean of a regret distribution.

    The geometric mean is computed on ``1 + regret_i`` and reported as
    ``exp(mean(log(1 + r))) - 1``; clipped at ``-1 + 1e-12`` to keep the
    log finite when an unusually-decoded path is sub-zero in regret
    (which can happen at numerically degenerate corners of the ODD).
    """
    regret = np.asarray(regret, dtype=np.float64)
    nan_mask = ~np.isfinite(regret)
    valid = regret[~nan_mask]
    if valid.size == 0:
        return CostRegretSummary(
            median=float("nan"),
            p90=float("nan"),
            p95=float("nan"),
            mean=float("nan"),
            geomean=float("nan"),
            n_valid=0,
            n_nan=int(nan_mask.sum()),
        )
    median = float(np.median(valid))
    p90 = float(np.percentile(valid, 90.0))
    p95 = float(np.percentile(valid, 95.0))
    mean = float(np.mean(valid))
    one_plus = np.clip(1.0 + valid, a_min=1e-12, a_max=None)
    geomean = float(math.exp(float(np.mean(np.log(one_plus)))) - 1.0)
    return CostRegretSummary(
        median=median,
        p90=p90,
        p95=p95,
        mean=mean,
        geomean=geomean,
        n_valid=int(valid.size),
        n_nan=int(nan_mask.sum()),
    )
