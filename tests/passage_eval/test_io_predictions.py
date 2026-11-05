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

    metrics = primary_metrics_from_arrays(arrays, allow_legacy=True)
    for k in (
        "cost_regret_median",
        "cost_regret_p90",
        "path_assd",
        "path_hausdorff_p95",
        "success_rate",
        "n_samples",
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
    metrics = primary_metrics_from_arrays(arrays, obstacle_violation_per_sample=oviol, allow_legacy=True)
    assert metrics["obstacle_violation_rate"] == pytest.approx(0.1, abs=1e-12)


def test_npz_roundtrip_preserves_failed_decode_penalty(tmp_path):
    path = tmp_path / "failed.npz"
    np.savez(
        path,
        regret=[0.0, 0.0],
        assd=[0.0, 8.0],
        h95=[0.0, 9.0],
        success=[True, False],
        pred_costs=[100.0, 0.0],
        oracle_costs=[100.0, 100.0],
        obstacle_violation=[False, False],
        metric_contract=np.array("neurips-2026-submission-v1"),
    )
    metrics = primary_metrics_from_arrays(load_per_sample_npz(path))
    assert metrics["cost_regret_p90"] == pytest.approx(0.9 * (1e6 - 1))
    assert metrics["path_assd"] == 0
    assert metrics["path_hausdorff_p95"] == 0
    assert metrics["obstacle_violation_rate"] == 0


def test_npz_rejects_unpaired_column_lengths(tmp_path):
    path = tmp_path / "unpaired.npz"
    np.savez(path, regret=[0.0, 0.0], assd=[0.0])
    with pytest.raises(ValueError, match="length"):
        load_per_sample_npz(path)


def test_current_scoring_requires_explicit_legacy_geometry_opt_in(tmp_path):
    path = tmp_path / "legacy.npz"
    _write_npz(path)
    with pytest.raises(ValueError, match="contract"):
        primary_metrics_from_arrays(load_per_sample_npz(path))


def test_current_contract_rejects_missing_status_evidence(tmp_path):
    path = tmp_path / "incomplete.npz"
    np.savez(path, regret=[0.0, np.nan], metric_contract=np.array("neurips-2026-submission-v1"))
    with pytest.raises(ValueError, match="required"):
        primary_metrics_from_arrays(load_per_sample_npz(path))


def test_current_contract_cannot_drop_failure_with_missing_oracle(tmp_path):
    path = tmp_path / "bad_oracle.npz"
    np.savez(
        path,
        regret=[0.0, np.nan],
        assd=[0.0, np.nan],
        h95=[0.0, np.nan],
        success=[True, False],
        pred_costs=[100.0, 0.0],
        oracle_costs=[100.0, np.nan],
        obstacle_violation=[False, False],
        metric_contract=np.array("neurips-2026-submission-v1"),
    )
    with pytest.raises(ValueError, match="oracle"):
        primary_metrics_from_arrays(load_per_sample_npz(path))


def test_current_contract_rejects_missing_cost_for_claimed_success(tmp_path):
    path = tmp_path / "unknown_cost.npz"
    np.savez(
        path,
        regret=[0.0],
        assd=[0.0],
        h95=[0.0],
        success=[True],
        pred_costs=[np.nan],
        oracle_costs=[100.0],
        obstacle_violation=[False],
        metric_contract=np.array("neurips-2026-submission-v1"),
    )
    with pytest.raises(ValueError, match="predicted"):
        primary_metrics_from_arrays(load_per_sample_npz(path))


def test_qualified_complete_arrays_use_frozen_primary_repeat(tmp_path):
    path = tmp_path / "qualified.npz"
    np.savez(
        path,
        regret=[0.0, 999999.0],
        assd=[0.0, np.nan],
        h95=[0.0, np.nan],
        success=[True, False],
        pred_costs=[100.0, 1e8],
        oracle_costs=[100.0, 100.0],
        obstacle_violation=[False, False],
        obstacle_cells=[0, 0],
        status=["ok", "no_path"],
        repeat_index=[0, 1],
        primary_quality=[True, False],
        quality_repeat_index=np.array(0),
        reference_qualified=[True, True],
        reference_id=["r", "r"],
        metric_contract=np.array("passage-qualified-complete-v1"),
    )
    metrics = primary_metrics_from_arrays(load_per_sample_npz(path))
    assert metrics["n_samples"] == 1
    assert metrics["cost_regret_median"] == 0
    assert metrics["success_rate"] == 1


def test_qualified_complete_arrays_require_repeat_selection(tmp_path):
    path = tmp_path / "unselected.npz"
    np.savez(
        path,
        regret=[0.0],
        assd=[0.0],
        h95=[0.0],
        success=[True],
        pred_costs=[100.0],
        oracle_costs=[100.0],
        obstacle_violation=[False],
        metric_contract=np.array("passage-qualified-complete-v1"),
    )
    with pytest.raises(ValueError, match="required"):
        load_per_sample_npz(path)


def test_qualified_timeout_cannot_be_rescued_by_connected_success(tmp_path):
    path = tmp_path / "timeout.npz"
    np.savez(
        path,
        regret=[0.0],
        assd=[0.0],
        h95=[0.0],
        success=[True],
        pred_costs=[100.0],
        oracle_costs=[100.0],
        obstacle_violation=[False],
        status=["timeout"],
        repeat_index=[0],
        primary_quality=[True],
        quality_repeat_index=np.array(0),
        reference_qualified=[True],
        reference_id=["r"],
        metric_contract=np.array("passage-qualified-complete-v1"),
    )
    metrics = primary_metrics_from_arrays(load_per_sample_npz(path))
    assert metrics["success_rate"] == 0
    assert metrics["cost_regret_median"] == 999999
    assert metrics["n_path_valid"] == 0
    assert np.isnan(metrics["path_assd"])
