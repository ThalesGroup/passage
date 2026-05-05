# MIT License - Copyright (c) 2026 THALES
"""Aggregation: primary-metric bundle and paired-bootstrap CIs."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from passage_eval._typing import PathLike
from passage_eval.metrics.geometric import path_assd, path_hausdorff_p95
from passage_eval.metrics.regret import cost_regret, cost_regret_summary
from passage_eval.metrics.reliability import (
    obstacle_cells_traversed_per_path,
    obstacle_violation_rate,
    success_rate,
)


def compute_primary_metrics(
    *,
    predicted_paths: Sequence[PathLike],
    oracle_paths: Sequence[PathLike],
    predicted_costs: Sequence[float] | np.ndarray,
    oracle_costs: Sequence[float] | np.ndarray,
    obstacles: Sequence[np.ndarray] | None = None,
    starts: Sequence[tuple[int, int]] | None = None,
    goals: Sequence[tuple[int, int]] | None = None,
) -> dict[str, float]:
    """Compute the six primary decoded-path metrics + diagnostic side.

    Returns a flat dict ready to write into a paper-facing CSV row.
    """
    if not (len(predicted_paths) == len(oracle_paths) == len(predicted_costs) == len(oracle_costs)):
        raise ValueError("Mismatched batch lengths across paths/costs.")
    if obstacles is not None and len(obstacles) != len(predicted_paths):
        raise ValueError("len(obstacles) must equal batch size.")

    regret = cost_regret(predicted_costs, oracle_costs)
    regret_stats = cost_regret_summary(regret).as_dict()

    assd = path_assd(predicted_paths, oracle_paths)
    haus95 = path_hausdorff_p95(predicted_paths, oracle_paths)

    succ = success_rate(predicted_paths, starts=starts, goals=goals)
    if obstacles is not None:
        oviol = obstacle_violation_rate(predicted_paths, obstacles)
        obstacle_cells = obstacle_cells_traversed_per_path(predicted_paths, obstacles)
    else:
        oviol = float("nan")
        obstacle_cells = float("nan")

    return {
        # primary headline
        "cost_regret_median": regret_stats["cost_regret_median"],
        "cost_regret_p90": regret_stats["cost_regret_p90"],
        "path_assd": float(np.nanmean(assd)) if np.any(np.isfinite(assd)) else float("nan"),
        "path_hausdorff_p95": float(np.nanmean(haus95)) if np.any(np.isfinite(haus95)) else float("nan"),
        "success_rate": succ,
        "obstacle_violation_rate": oviol,
        # diagnostic-side
        "cost_regret_p95": regret_stats["cost_regret_p95"],
        "cost_regret_mean": regret_stats["cost_regret_mean"],
        "cost_regret_geomean": regret_stats["cost_regret_geomean"],
        "obstacle_cells_traversed_per_path": obstacle_cells,
        # bookkeeping
        "n_samples": int(len(predicted_paths)),
        "n_regret_valid": int(regret_stats["cost_regret_n_valid"]),
        "n_regret_nan": int(regret_stats["cost_regret_n_nan"]),
    }


def paired_bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    *,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Paired-bootstrap CI on the difference of a metric across paired samples.

    Both arrays are indexed identically (sample-paired), so the
    sample-difficulty variance cancels under the resample. ``metric_fn``
    is applied to each resampled pair and the difference recorded.
    """
    samples_a = np.asarray(samples_a)
    samples_b = np.asarray(samples_b)
    if samples_a.shape[0] != samples_b.shape[0]:
        raise ValueError("Paired samples must have equal length.")
    n = samples_a.shape[0]
    if n == 0:
        return {"point": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    point = float(metric_fn(samples_a, samples_b))
    diffs = np.empty(n_bootstrap, dtype=np.float64)
    for k in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        diffs[k] = float(metric_fn(samples_a[idx], samples_b[idx]))
    diffs.sort()
    lo = float(diffs[int(np.floor((alpha / 2.0) * n_bootstrap))])
    hi = float(diffs[int(np.ceil((1.0 - alpha / 2.0) * n_bootstrap)) - 1])
    return {"point": point, "ci_low": lo, "ci_high": hi}
