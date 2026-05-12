# MIT License
#
# Copyright (c) 2026 THALES

"""Load PASSAGE per-sample prediction arrays + score them.

The vision and graph baselines emit one NPZ per (config, seed) with the
standard schema below. This file gives a single-call path from that
NPZ to the headline metric dict, so notebooks and pairwise-comparison
scripts do not have to re-implement the bookkeeping.

Schema (per-sample arrays, length = number of val samples = 1024 in the
paper):

    regret          : float (cost_regret per sample)
    assd            : float (path_assd per sample, normalised)
    h95             : float (path_hausdorff_p95 per sample, normalised)
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

from passage_eval.metrics.regret import cost_regret_summary

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
    ``np.float64``; failed-decode rows carry ``NaN`` in
    ``regret/assd/h95`` per the censored-penalty convention.
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
        n = regret.shape[0]

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
                if k not in {
                    "regret", "assd", "h95", "success",
                    "pred_costs", "oracle_costs",
                    "obs_density", "elev_std", "oracle_len",
                }
            },
        )
    return bundle


def primary_metrics_from_arrays(
    arrays: PerSampleArrays,
    *,
    obstacle_violation_per_sample: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute the §5.2 primary-metric bundle directly from per-sample arrays.

    This is the dual of
    :func:`passage_eval.metrics.summary.compute_primary_metrics`: that
    function ingests paths and returns metrics; this one ingests the
    already-computed per-sample metric arrays (the released NPZ
    schema) and emits the same dict, with the same keys, so notebooks
    can score artefacts without re-running the decoder.
    """
    regret_summary = cost_regret_summary(arrays.regret).as_dict()
    assd = arrays.assd
    h95 = arrays.h95
    success_rate = float(np.mean(arrays.success.astype(np.float64)))
    if obstacle_violation_per_sample is None:
        oviol = float("nan")
        obstacle_cells = float("nan")
    else:
        ov = np.asarray(obstacle_violation_per_sample).astype(np.float64)
        oviol = float(np.mean(ov))
        obstacle_cells = float(np.nanmean(ov))
    return {
        "cost_regret_median": regret_summary["cost_regret_median"],
        "cost_regret_p90": regret_summary["cost_regret_p90"],
        "path_assd": float(np.nanmean(assd)) if np.any(np.isfinite(assd)) else float("nan"),
        "path_hausdorff_p95": float(np.nanmean(h95)) if np.any(np.isfinite(h95)) else float("nan"),
        "success_rate": success_rate,
        "obstacle_violation_rate": oviol,
        "cost_regret_p95": regret_summary["cost_regret_p95"],
        "cost_regret_mean": regret_summary["cost_regret_mean"],
        "cost_regret_geomean": regret_summary["cost_regret_geomean"],
        "obstacle_cells_traversed_per_path": obstacle_cells,
        "n_samples": int(len(arrays)),
        "n_regret_valid": int(regret_summary["cost_regret_n_valid"]),
        "n_regret_nan": int(regret_summary["cost_regret_n_nan"]),
    }
