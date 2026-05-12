# MIT License
#
# Copyright (c) 2026 THALES

"""Tests for the per-sample NPZ loader + primary_metrics_from_arrays."""
from __future__ import annotations

import numpy as np
import pytest

from passage_eval import (
    PerSampleArrays,
    load_per_sample_npz,
    primary_metrics_from_arrays,
)


def _write_npz(path, *, n=32, regret_offset=0.05, success_rate=1.0, seed=0):
    rng = np.random.default_rng(seed)
    regret = rng.uniform(regret_offset, regret_offset + 0.1, size=n).astype(np.float64)
    assd = rng.uniform(0.05, 0.15, size=n).astype(np.float64)
    h95 = rng.uniform(0.10, 0.25, size=n).astype(np.float64)
    succ = rng.uniform(0, 1, size=n) < success_rate
    pred_costs = rng.uniform(1.0, 2.0, size=n).astype(np.float64)
    oracle_costs = pred_costs * (1.0 + regret)
    obs_density = rng.uniform(0.0, 0.3, size=n).astype(np.float64)
    elev_std = rng.uniform(50, 500, size=n).astype(np.float64)
    oracle_len = rng.integers(50, 200, size=n).astype(np.int64)
    np.savez(
        path,
        regret=regret,
        assd=assd,
        h95=h95,
        success=succ,
        pred_costs=pred_costs,
        oracle_costs=oracle_costs,
        obs_density=obs_density,
        elev_std=elev_std,
        oracle_len=oracle_len,
    )


def test_load_and_compute_metrics_round_trip(tmp_path):
    npz = tmp_path / "preds.npz"
    _write_npz(npz, n=64, regret_offset=0.01, success_rate=1.0)
    arrays = load_per_sample_npz(npz)
    assert isinstance(arrays, PerSampleArrays)
    assert len(arrays) == 64

    metrics = primary_metrics_from_arrays(arrays)
    for k in (
        "cost_regret_median", "cost_regret_p90",
        "path_assd", "path_hausdorff_p95",
        "success_rate", "n_samples",
    ):
        assert k in metrics
    assert metrics["n_samples"] == 64
    assert metrics["success_rate"] == pytest.approx(1.0, abs=1e-9)


def test_load_missing_columns_fallback_to_nan(tmp_path):
    npz = tmp_path / "regret_only.npz"
    np.savez(npz, regret=np.array([0.01, 0.02, 0.03], dtype=np.float64))
    arrays = load_per_sample_npz(npz)
    assert len(arrays) == 3
    assert np.all(np.isnan(arrays.assd))
    assert np.all(np.isnan(arrays.h95))


def test_load_rejects_missing_regret(tmp_path):
    npz = tmp_path / "broken.npz"
    np.savez(npz, foo=np.array([1, 2, 3]))
    with pytest.raises(ValueError):
        load_per_sample_npz(npz)


def test_obstacle_violation_passthrough(tmp_path):
    npz = tmp_path / "with_oviol.npz"
    _write_npz(npz, n=10, regret_offset=0.0, success_rate=1.0)
    arrays = load_per_sample_npz(npz)
    oviol = np.zeros(10, dtype=np.float64)
    oviol[3] = 1.0
    metrics = primary_metrics_from_arrays(arrays, obstacle_violation_per_sample=oviol)
    assert metrics["obstacle_violation_rate"] == pytest.approx(0.1, abs=1e-12)
