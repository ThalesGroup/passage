# MIT License
#
# Copyright (c) 2026 THALES

"""Tests for the leaderboard storage + visualisation module."""

from __future__ import annotations

import json

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
        package_version="1.0.0",
        dataset_version="passage:v1.0-test",
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
        {"method_name": "b", "Stab": 0.05, "Geo": 0.02, "ODDe": 0.10, "Eff": 1.2},
    ]
    ax = plot_trustworthiness_radar(rows)
    assert ax is not None


def test_plot_pairwise_heatmap_runs():
    import matplotlib

    matplotlib.use("Agg")
    decisions = {("a", "b"): "PARETO_DOMINANT_A", ("a", "c"): "TRADE_OFF"}
    ax = plot_pairwise_heatmap(decisions)
    assert ax is not None


def test_submission_refuses_mixed_metric_contracts(tmp_path):
    submit(_make_submission("historical"), root=tmp_path)
    current = _make_submission("aligned")
    current.extra["metric_contract"] = "neurips-2026-submission-v1"
    with pytest.raises(ValueError, match="contract"):
        submit(current, root=tmp_path)
    assert len(load_leaderboard(tmp_path)) == 1


def test_rebuild_retains_contract_and_submission_metadata(tmp_path):
    first = _make_submission("current")
    first.extra["metric_contract"] = "neurips-2026-submission-v1"
    first.edge_accelerator = "Thor-GPU"
    first.extra["task_variant"] = "obstacles"
    submit(first, root=tmp_path)
    rebuild_leaderboard(tmp_path)
    second = _make_submission("next")
    second.extra["metric_contract"] = "neurips-2026-submission-v1"
    submit(second, root=tmp_path)
    rebuilt = load_leaderboard(tmp_path).df
    row = rebuilt[rebuilt.method_name == "current"].iloc[0]
    assert row.edge_accelerator == "Thor-GPU"
    assert row.latency_e2e_ms_p50 == 11.0
    assert json.loads(row.extra_json)["task_variant"] == "obstacles"


def test_seed_and_accelerator_exports_preserve_measured_values(tmp_path):
    from passage_eval import (
        summarise_leaderboard,
        scoreboard_markdown,
        plot_grouped_bars_by_accelerator,
        plot_latency_heatmap,
        plot_metric_vs_accelerator,
    )
    import matplotlib.pyplot as plt

    for accel, latency in [("H100", 2.0), ("Thor-GPU", 20.0)]:
        for seed, regret in [(1234, 0.01), (2026, 0.02), (3407, 0.03)]:
            sub = _make_submission("reg", regret, seed)
            sub.edge_accelerator = accel
            sub.latency_e2e_ms_p50 = latency
            submit(sub, root=tmp_path)
    df = load_leaderboard(tmp_path).df
    summary = summarise_leaderboard(df)
    assert summary.n_seeds.tolist() == [3, 3]
    assert summary["mean"].tolist() == pytest.approx([0.02, 0.02])
    assert summary["std"].tolist() == pytest.approx([0.01, 0.01])
    markdown = scoreboard_markdown(df)
    assert "`h100` | 3 | 0.0200 ± 0.0100 | 2.00" in markdown
    for plot in [plot_grouped_bars_by_accelerator, plot_latency_heatmap, plot_metric_vs_accelerator]:
        ax = plot(df)
        output = tmp_path / f"{plot.__name__}.png"
        ax.figure.savefig(output)
        assert output.stat().st_size > 1000
        if plot is plot_latency_heatmap:
            assert ax.images[0].get_array().tolist() == [[2.0, 20.0]]
        elif plot is plot_metric_vs_accelerator:
            assert ax.lines[0].get_ydata().tolist() == pytest.approx([0.02, 0.02])
        else:
            assert [p.get_height() for p in ax.patches] == pytest.approx([0.02, 0.02])
        plt.close(ax.figure)


def test_exports_show_empty_and_missing_latency_honestly(tmp_path):
    from passage_eval import (
        summarise_leaderboard,
        scoreboard_markdown,
        plot_grouped_bars_by_accelerator,
        plot_latency_heatmap,
        plot_metric_vs_accelerator,
    )
    import matplotlib.pyplot as plt

    empty = load_leaderboard(tmp_path).df
    assert summarise_leaderboard(empty).empty
    assert "no leaderboard rows" in scoreboard_markdown(empty)
    for plot in [plot_grouped_bars_by_accelerator, plot_latency_heatmap, plot_metric_vs_accelerator]:
        ax = plot(empty)
        assert "no leaderboard rows" in ax.texts[0].get_text()
        plt.close(ax.figure)
    sub = _make_submission("untimed")
    sub.latency_e2e_ms_p50 = None
    submit(sub, root=tmp_path)
    df = load_leaderboard(tmp_path).df
    ax = plot_latency_heatmap(df)
    assert "no finite" in ax.texts[0].get_text()
    plt.close(ax.figure)
    assert np.isnan(summarise_leaderboard(df).latency_e2e_ms_p50.iloc[0])


def test_rebuild_recovers_old_bundle_contract_from_sidecar(tmp_path):
    first = _make_submission("old-storage")
    first.extra["metric_contract"] = "neurips-2026-submission-v1"
    run = submit(first, root=tmp_path, sidecar={"extra": first.extra})
    path = run / "metrics.json"
    payload = json.loads(path.read_text())
    del payload["submission"]
    path.write_text(json.dumps(payload))
    rebuild_leaderboard(tmp_path)
    second = _make_submission("next")
    second.extra["metric_contract"] = "neurips-2026-submission-v1"
    submit(second, root=tmp_path)
    assert len(load_leaderboard(tmp_path)) == 2
