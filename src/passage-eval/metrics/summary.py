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

"""Aggregation: primary-metric bundle and paired-bootstrap CIs."""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from passage_eval._typing import PathLike
from passage_eval.metrics.geometric import path_assd, path_hausdorff_p95
from passage_eval.metrics.regret import cost_regret, cost_regret_summary
from passage_eval.metrics.reliability import (
    path_status,
)
from passage_eval.protocol.discipline import apply_censored_penalty


def per_sample_metrics(
    *,
    predicted_paths: Sequence[PathLike],
    oracle_paths: Sequence[PathLike],
    predicted_costs: Sequence[float] | np.ndarray,
    oracle_costs: Sequence[float] | np.ndarray,
    obstacles: Sequence[np.ndarray] | None = None,
    starts: Sequence[tuple[int, int]] | None = None,
    goals: Sequence[tuple[int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Validated per-sample evidence; censored costs retain failed decodes (§4.1)."""
    n = len(predicted_paths)
    if not (n == len(oracle_paths) == len(predicted_costs) == len(oracle_costs)):
        raise ValueError("Mismatched batch lengths across paths/costs.")
    if (starts is None) != (goals is None):
        raise ValueError("Provide both starts and goals")
    for values in (obstacles, starts, goals):
        if values is not None and len(values) != n:
            raise ValueError("Per-sample fields must align with paths")
    statuses = [
        path_status(
            p,
            start=starts[i] if starts is not None else None,
            goal=goals[i] if goals is not None else None,
            obstacles=obstacles[i] if obstacles is not None else None,
        )
        for i, p in enumerate(predicted_paths)
    ]
    succeeded = np.array([s[0] for s in statuses], dtype=bool)
    violated = np.array([s[1] for s in statuses], dtype=bool)
    valid = succeeded & ~violated
    costs = apply_censored_penalty(
        predicted_costs, oracle_costs, decoder_succeeded=succeeded, obstacle_violated=violated
    )
    # Geometry is defined on the successful, obstacle-free subset (App. A.3).
    geometry_paths = [p if valid[i] else [] for i, p in enumerate(predicted_paths)]
    return {
        "pred_costs": costs,
        "oracle_costs": np.asarray(oracle_costs, dtype=np.float64),
        "regret": cost_regret(costs, oracle_costs),
        "assd": path_assd(geometry_paths, oracle_paths),
        "h95": path_hausdorff_p95(geometry_paths, oracle_paths),
        "success": succeeded,
        "obstacle_violation": violated,
        "valid_path": valid,
        "obstacle_cells": np.array([s[2] for s in statuses], dtype=np.int64),
    }


def aggregate_primary_metrics(arrays: dict[str, np.ndarray], *, obstacles_available: bool = True) -> dict[str, float]:
    """Aggregate validated per-sample arrays without rerunning inference."""
    stats = cost_regret_summary(arrays["regret"]).as_dict()
    n = len(arrays["regret"])

    def mean_finite(values):
        return float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")

    counts = arrays["obstacle_cells"]
    return {
        "cost_regret_median": stats["cost_regret_median"],
        "cost_regret_p90": stats["cost_regret_p90"],
        "path_assd": mean_finite(arrays["assd"]),
        "path_hausdorff_p95": mean_finite(arrays["h95"]),
        "success_rate": float(np.mean(arrays["success"])) if n else float("nan"),
        "obstacle_violation_rate": float(np.mean(arrays["obstacle_violation"]))
        if n and obstacles_available
        else float("nan"),
        "cost_regret_p95": stats["cost_regret_p95"],
        "cost_regret_mean": stats["cost_regret_mean"],
        "cost_regret_geomean": stats["cost_regret_geomean"],
        "obstacle_cells_traversed_per_path": float(np.mean(counts[counts > 0]))
        if obstacles_available and np.any(counts > 0)
        else float("nan"),
        "n_samples": n,
        "n_regret_valid": stats["cost_regret_n_valid"],
        "n_regret_nan": stats["cost_regret_n_nan"],
        "n_path_valid": int(np.count_nonzero(arrays["valid_path"])),
    }


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
    """Six metrics; costs include censored failures, geometry uses valid paths.

    Costs passed to this lower-level API must already use the analytic scene
    model. Use score_planner to recompute them directly from candidate paths.
    """
    arrays = per_sample_metrics(
        predicted_paths=predicted_paths,
        oracle_paths=oracle_paths,
        predicted_costs=predicted_costs,
        oracle_costs=oracle_costs,
        obstacles=obstacles,
        starts=starts,
        goals=goals,
    )
    return aggregate_primary_metrics(arrays, obstacles_available=obstacles is not None)


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
