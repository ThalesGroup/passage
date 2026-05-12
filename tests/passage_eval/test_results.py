# MIT License
#
# Copyright (c) 2026 THALES

"""Tests for the leaderboard storage + visualisation module."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from passage_eval import (
    Leaderboard,
    Submission,
    load_leaderboard,
    rebuild_leaderboard,
    submit,
)
from passage_eval import (
    plot_pairwise_heatmap,
    plot_pareto_accuracy_latency,
    plot_primary_bars,
    plot_trustworthiness_radar,
)


def _make_submission(method: str, regret_med: float = 0.05, seed: int = 1234) -> Submission:
    return Submission(
        method_name=method,
        family="dense_regression",
        architecture="repvit_m1",
        decoder_variant="V_bridge",
        resolution=256,
        cost_model="elevation",
        seed=seed,
        n_samples=1024,
        metrics={
            "cost_regret_median": regret_med,
            "cost_regret_p90": regret_med * 4,
            "path_assd": 0.04,
            "path_hausdorff_p95": 0.10,
            "success_rate": 0.997,
            "obstacle_violation_rate": 0.003,
        },
        latency_e2e_ms_p50=11.0,
        package_version="0.3.0",
        dataset_version="passage:v0.4-test",
    )


def test_submit_writes_run_dir_and_appends_csv(tmp_path):
    sub = _make_submission("reg-vbridge", regret_med=0.001)
    run_dir = submit(sub, root=tmp_path)
    assert run_dir.exists()
    assert (run_dir / "metrics.json").exists()
    csv_path = tmp_path / "leaderboard.csv"
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    assert df.shape[0] == 1
    assert df.iloc[0]["method_name"] == "reg-vbridge"
    assert df.iloc[0]["cost_regret_median"] == pytest.approx(0.001, abs=1e-9)


def test_submit_with_per_sample_and_sidecar(tmp_path):
    sub = _make_submission("seg-bce", regret_med=0.04)
    per_sample = {
        "regret": np.linspace(0, 0.5, 10),
        "assd": np.linspace(0, 0.2, 10),
        "h95": np.linspace(0, 0.4, 10),
    }
    sidecar = {"method_name": "seg-bce", "seeds": [1234], "extras": {"x": 1}}
    run_dir = submit(sub, root=tmp_path, per_sample=per_sample, sidecar=sidecar)
    assert (run_dir / "per_sample.npz").exists()
    assert (run_dir / "sidecar.json").exists()
    payload = json.loads((run_dir / "sidecar.json").read_text())
    assert payload["method_name"] == "seg-bce"


def test_load_leaderboard_returns_dataframe_wrapper(tmp_path):
    submit(_make_submission("a", 0.10), root=tmp_path)
    submit(_make_submission("b", 0.02), root=tmp_path)
    submit(_make_submission("c", 0.15), root=tmp_path)
    lb = load_leaderboard(tmp_path)
    assert isinstance(lb, Leaderboard)
    assert len(lb) == 3
    best = lb.best_per_method(metric="cost_regret_median")
    assert set(best["method_name"]) == {"a", "b", "c"}
    a_only = lb.filter(method_name="a")
    assert len(a_only) == 1


def test_rebuild_leaderboard_round_trip(tmp_path):
    submit(_make_submission("a", 0.10), root=tmp_path)
    submit(_make_submission("b", 0.02), root=tmp_path)
    csv_path = tmp_path / "leaderboard.csv"
    csv_path.unlink()
    rebuilt = rebuild_leaderboard(tmp_path)
    df = pd.read_csv(rebuilt)
    assert set(df["method_name"]) == {"a", "b"}


def test_plot_primary_bars_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    submit(_make_submission("a", 0.10), root=tmp_path)
    submit(_make_submission("b", 0.02), root=tmp_path)
    df = load_leaderboard(tmp_path).df
    ax = plot_primary_bars(df, metric="cost_regret_median")
    assert ax is not None


def test_plot_pareto_runs(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    submit(_make_submission("a", 0.10), root=tmp_path)
    submit(_make_submission("b", 0.02), root=tmp_path)
    df = load_leaderboard(tmp_path).df
    ax = plot_pareto_accuracy_latency(df)
    assert ax is not None


def test_plot_trustworthiness_radar_runs():
    import matplotlib
    matplotlib.use("Agg")
    rows = [
        {"method_name": "a", "Stab": 0.01, "Geo": -0.01, "ODDe": 0.05, "Eff": 4.4},
        {"method_name": "b", "Stab": 0.05, "Geo":  0.02, "ODDe": 0.10, "Eff": 1.2},
    ]
    ax = plot_trustworthiness_radar(rows)
    assert ax is not None


def test_plot_pairwise_heatmap_runs():
    import matplotlib
    matplotlib.use("Agg")
    decisions = {("a", "b"): "PARETO_DOMINANT_A", ("a", "c"): "TRADE_OFF"}
    ax = plot_pairwise_heatmap(decisions)
    assert ax is not None
