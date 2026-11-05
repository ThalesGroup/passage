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

"""Load PASSAGE per-sample prediction arrays + score them.

The vision and graph baselines emit one NPZ per (config, seed) with the
standard schema below. This file gives a single-call path from that
NPZ to the headline metric dict, so notebooks and pairwise-comparison
scripts do not have to re-implement the bookkeeping.

Schema (per-sample arrays, length = number of val samples = 1024 in the
paper):

    regret          : float (cost_regret per sample)
    assd            : float (path_assd per sample, normalised)
    h95             : float (path_hausdorff_p95 per sample, raw pixels)
    success         : bool   (1 if predicted path connects start to goal)
    pred_costs      : float (predicted-path cost under the scene's cost model)
    oracle_costs    : float (A* oracle path cost)
    obs_density     : float (per-sample obstacle coverage, used as ODD stratifier)
    elev_std        : float (per-sample elevation σ, used as ODD stratifier)
    oracle_len      : int   (oracle path length in cells)

``obstacle_violation`` and ``obstacle_cells_traversed_per_path`` are
derived from ``success`` + the per-sample obstacle field if the npz is
extended; they fall back to NaN if the columns are absent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from passage_eval.metrics.regret import cost_regret
from passage_eval.metrics.summary import aggregate_primary_metrics

__all__ = [
    "PerSampleArrays",
    "load_per_sample_npz",
    "primary_metrics_from_arrays",
]


@dataclass(frozen=True)
class PerSampleArrays:
    """Per-sample array bundle for one (config, seed) pair.

    Length equals the number of evaluated samples (1024 in §5.2 of the
    paper). All fields except ``success`` and ``oracle_len`` are
    ``np.float64``; failed-decode geometry is ``NaN`` while regret remains finite
    under the censored-penalty convention. Legacy artifacts require explicit
    ``allow_legacy=True`` when aggregating.
    """

    regret: np.ndarray
    assd: np.ndarray
    h95: np.ndarray
    success: np.ndarray
    pred_costs: np.ndarray
    oracle_costs: np.ndarray
    obs_density: np.ndarray
    elev_std: np.ndarray
    oracle_len: np.ndarray
    extras: dict[str, np.ndarray]

    def __len__(self) -> int:
        return int(self.regret.shape[0])


def _to_float(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def _to_int(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.int64)


def _to_bool(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=bool)


def load_per_sample_npz(path: str | Path) -> PerSampleArrays:
    """Load a per-sample NPZ produced by the released baselines.

    Missing columns get ``NaN`` arrays of the right length to keep the
    downstream API uniform across vision / graph emissions.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as store:
        keys = set(store.files)
        if "regret" not in keys:
            raise ValueError(f"{path}: missing required column 'regret'")
        regret = _to_float(store["regret"])
        if regret.ndim != 1:
            raise ValueError(f"{path}: regret must be a one-dimensional column")
        n = regret.shape[0]
        contract = store["metric_contract"].item() if "metric_contract" in keys else None
        qualified_complete = contract == "passage-qualified-complete-v1"
        if contract in {"neurips-2026-submission-v1", "passage-qualified-complete-v1"}:
            required = {"regret", "assd", "h95", "success", "pred_costs", "oracle_costs", "obstacle_violation"}
            if qualified_complete:
                required |= {
                    "primary_quality",
                    "quality_repeat_index",
                    "repeat_index",
                    "status",
                    "reference_qualified",
                    "reference_id",
                }
            if not required <= keys:
                raise ValueError(f"{path}: missing required current-contract columns {sorted(required - keys)}")
        for key in keys & {"metric_contract", "quality_repeat_index"}:
            if store[key].ndim != 0:
                raise ValueError(f"{path}: {key} must be scalar metadata")
        for key in keys - {"metric_contract", "quality_repeat_index"}:
            if store[key].ndim != 1 or len(store[key]) != n:
                raise ValueError(f"{path}: column {key!r} must have length {n}")

        def _get_float(key: str) -> np.ndarray:
            return _to_float(store[key]) if key in keys else np.full(n, np.nan, dtype=np.float64)

        def _get_int(key: str) -> np.ndarray:
            return _to_int(store[key]) if key in keys else np.zeros(n, dtype=np.int64)

        def _get_bool(key: str) -> np.ndarray:
            return _to_bool(store[key]) if key in keys else np.ones(n, dtype=bool)

        bundle = PerSampleArrays(
            regret=regret,
            assd=_get_float("assd"),
            h95=_get_float("h95"),
            success=_get_bool("success"),
            pred_costs=_get_float("pred_costs"),
            oracle_costs=_get_float("oracle_costs"),
            obs_density=_get_float("obs_density"),
            elev_std=_get_float("elev_std"),
            oracle_len=_get_int("oracle_len"),
            extras={
                k: np.asarray(store[k])
                for k in keys
                if k
                not in {
                    "regret",
                    "assd",
                    "h95",
                    "success",
                    "pred_costs",
                    "oracle_costs",
                    "obs_density",
                    "elev_std",
                    "oracle_len",
                }
            },
        )
    return bundle


def primary_metrics_from_arrays(
    arrays: PerSampleArrays,
    *,
    obstacle_violation_per_sample: np.ndarray | None = None,
    allow_legacy: bool = False,
) -> dict[str, float]:
    """Compute the §5.2 primary-metric bundle directly from per-sample arrays.

    This is the dual of
    :func:`passage_eval.metrics.summary.compute_primary_metrics`: that
    function ingests paths and returns metrics; this one ingests the
    already-computed per-sample metric arrays (the released NPZ
    schema) and emits the same dict, with the same keys, so notebooks
    can score artefacts without re-running the decoder.
    """
    contract = arrays.extras.get("metric_contract")
    contract_name = np.asarray(contract).item() if contract is not None else None
    current = contract_name in {"neurips-2026-submission-v1", "passage-qualified-complete-v1"}
    if not current:
        if not allow_legacy:
            raise ValueError(
                "Unknown metric contract; re-evaluate paths or explicitly set allow_legacy=True for historical analysis"
            )
    n = len(arrays)
    selection = np.ones(n, dtype=bool)
    succeeded = arrays.success.copy()
    if contract_name == "passage-qualified-complete-v1":
        required = {
            "primary_quality",
            "quality_repeat_index",
            "repeat_index",
            "status",
            "reference_qualified",
            "reference_id",
        }
        if not required <= arrays.extras.keys():
            raise ValueError("Qualified complete arrays require frozen primary-repeat evidence")
        repeat = np.asarray(arrays.extras["quality_repeat_index"])
        indices = np.asarray(arrays.extras["repeat_index"])
        selection = np.asarray(arrays.extras["primary_quality"])
        if (
            repeat.ndim != 0
            or repeat.dtype.kind not in "iu"
            or repeat.item() < 0
            or selection.dtype.kind != "b"
            or selection.shape != (n,)
            or indices.shape != (n,)
            or indices.dtype.kind not in "iu"
            or not np.array_equal(selection, indices == repeat.item())
            or not np.any(selection)
        ):
            raise ValueError("Qualified primary selection must match the frozen repeat index")
        if not np.asarray(arrays.extras["reference_qualified"], dtype=bool).all():
            raise ValueError("Qualified complete arrays require qualified references")
        from passage_eval.protocol.contracts import STATUSES

        statuses = np.asarray(arrays.extras["status"])
        if statuses.shape != (n,) or not np.isin(statuses, list(STATUSES)).all():
            raise ValueError("Qualified complete arrays require recognized terminal statuses")
        succeeded &= statuses == "ok"
    if current and (not np.isfinite(arrays.oracle_costs).all() or np.any(arrays.oracle_costs <= 0)):
        raise ValueError("Current-contract oracle costs must be positive and finite for every row")
    violations = obstacle_violation_per_sample
    if violations is None:
        violations = arrays.extras.get("obstacle_violation")
    known_obstacles = violations is not None
    violated = np.zeros(n, dtype=bool) if violations is None else np.asarray(violations, dtype=bool)
    if violated.shape != (n,):
        raise ValueError("obstacle violations must align with per-sample arrays")
    valid = succeeded & ~violated
    if current and np.any(valid & (~np.isfinite(arrays.pred_costs) | (arrays.pred_costs <= 0))):
        raise ValueError("Current-contract predicted costs must be positive and finite for valid paths")
    regret = cost_regret(arrays.pred_costs, arrays.oracle_costs) if current else arrays.regret.copy()
    failed = ~valid
    # Historical arrays can omit costs. Never turn an unscorable failure into
    # a successful decode or silently discard its recorded regret.
    censorable = failed & np.isfinite(arrays.oracle_costs) & (arrays.oracle_costs > 0)
    regret[censorable] = cost_regret(1e6 * arrays.oracle_costs[censorable], arrays.oracle_costs[censorable])
    columns = {
        "regret": regret,
        "assd": np.where(valid, arrays.assd, np.nan),
        "h95": np.where(valid, arrays.h95, np.nan),
        "success": succeeded,
        "obstacle_violation": violated,
        "valid_path": valid,
        "obstacle_cells": arrays.extras.get("obstacle_cells", np.zeros(n, dtype=int)),
    }
    return aggregate_primary_metrics(
        {key: values[selection] for key, values in columns.items()},
        obstacles_available=known_obstacles,
    )
