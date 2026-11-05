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

"""Tests for the leaderboard storage + visualisation module."""

from __future__ import annotations

import json
import csv
import io
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

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
    assert "`h100` | 3/3 | 0.0200 ± 0.0100 | 2.00" in markdown
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


def test_csv_round_trip_restores_numeric_types_and_optional_values():
    original = _make_submission("typed")
    original.cost_weight = 62.4
    original.latency_fwd_ms_p50 = 1.25
    original.extra = {"metric_contract": "contract-v1", "task_variant": "free"}
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=original.to_row())
    writer.writeheader()
    writer.writerow(original.to_row())
    buffer.seek(0)
    restored = Submission.from_row(next(csv.DictReader(buffer)))
    assert (restored.resolution, restored.seed, restored.batch_size, restored.n_samples) == (256, 1234, 1, 1024)
    assert all(
        type(value) is int for value in (restored.resolution, restored.seed, restored.batch_size, restored.n_samples)
    )
    assert type(restored.cost_weight) is float
    assert restored.cost_weight == 62.4
    assert restored.latency_fwd_ms_p50 == 1.25
    assert restored.latency_decoder_ms_p50 is None
    assert restored.extra == original.extra


def test_csv_round_trip_rejects_invalid_integer_metadata():
    row = _make_submission("invalid").to_row()
    row["seed"] = "1234.5"
    with pytest.raises(ValueError, match="seed"):
        Submission.from_row(row)


def test_summary_keeps_benchmark_conditions_and_seed_identity():
    from passage_eval import summarise_leaderboard

    first = _make_submission("same", 0.01)
    first.resolution, first.cost_model, first.split = 64, "elevation", "validation"
    second = _make_submission("same", 0.99)
    second.resolution, second.cost_model, second.split = 512, "energy", "test"
    summary = summarise_leaderboard(pd.DataFrame([first.to_row(), second.to_row()]))
    assert len(summary) == 2
    assert summary["mean"].tolist() == pytest.approx([0.01, 0.99])
    assert summary.n_seeds.tolist() == [1, 1]
    assert set(summary.split) == {"validation", "test"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("cost_weight", 62.4),
        ("dataset_version", "different-revision"),
        ("dtype", "fp16"),
        ("batch_size", 2),
        ("decoder_variant", "other"),
        ("n_samples", 512),
        ("host", "different-machine"),
    ],
)
def test_summary_does_not_pool_different_conditions(field, value):
    from passage_eval import summarise_leaderboard

    first = _make_submission("same", 0.01)
    second = _make_submission("same", 0.99, seed=2026)
    setattr(second, field, value)
    summary = summarise_leaderboard(pd.DataFrame([first.to_row(), second.to_row()]))
    assert len(summary) == 2
    assert sorted(summary["mean"]) == pytest.approx([0.01, 0.99])


@pytest.mark.parametrize("key", ["metric_contract", "task_variant", "dataset_revision", "temperature"])
def test_summary_keeps_extra_conditions_and_source_provenance(key):
    from passage_eval import summarise_leaderboard

    first = _make_submission("same", 0.01)
    second = _make_submission("same", 0.99, seed=2026)
    first.extra[key], second.extra[key] = "first", "second"
    first.per_sample_path = "runs/first/per_sample.npz"
    summary = summarise_leaderboard(pd.DataFrame([first.to_row(), second.to_row()]))
    assert len(summary) == 2
    provenance = [record for value in summary.provenance_json for record in json.loads(value)]
    assert {record["extra"][key] for record in provenance} == {"first", "second"}
    assert any(record["per_sample_path"] == "runs/first/per_sample.npz" for record in provenance)


def test_summary_rejects_repeated_seeds_without_explicit_replicate_policy():
    from passage_eval import summarise_leaderboard

    rows = [_make_submission("same", regret).to_row() for regret in [0.01, 0.99]]
    with pytest.raises(ValueError, match="[Rr]epeated seed"):
        summarise_leaderboard(pd.DataFrame(rows))


def test_summary_weights_distinct_seeds_equally_after_explicit_replicate_mean():
    from passage_eval import summarise_leaderboard

    rows = [_make_submission("same", value, seed).to_row() for seed, value in [(1234, 0), (1234, 0.2), (2026, 0.9)]]
    summary = summarise_leaderboard(pd.DataFrame(rows), replicate_policy="mean")
    assert summary["mean"].tolist() == pytest.approx([0.5])
    assert summary["std"].tolist() == pytest.approx([0.565685424949238])
    assert summary.n_seeds.tolist() == [2]
    assert summary.n_runs.tolist() == [3]


def test_submission_collision_refuses_overwrite_and_survives_rebuild(tmp_path):
    first, second = _make_submission("same", 0.01), _make_submission("same", 0.99)
    first.timestamp_utc = second.timestamp_utc = "2026-10-24T00-00-00Z"
    run = submit(first, root=tmp_path, per_sample={"regret": np.array([0.01])})
    with pytest.raises(FileExistsError):
        submit(second, root=tmp_path, per_sample={"regret": np.array([0.99])})
    assert np.load(run / "per_sample.npz")["regret"].tolist() == [0.01]
    assert len(load_leaderboard(tmp_path)) == 1
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.cost_regret_median.tolist() == [0.01]


def test_failed_submission_leaves_no_rebuildable_run_or_mutated_metadata(tmp_path):
    sub = _make_submission("broken")
    with pytest.raises(TypeError):
        submit(sub, root=tmp_path, per_sample={"regret": np.array([0.1])}, sidecar={"bad": object()})
    assert sub.per_sample_path is None
    assert sub.sidecar_path is None
    assert not list(tmp_path.rglob("metrics.json"))
    assert not list(tmp_path.rglob("per_sample.npz"))
    rebuild_leaderboard(tmp_path)
    assert len(load_leaderboard(tmp_path)) == 0


def test_index_write_failure_rolls_back_run_and_preserves_existing_index(tmp_path, monkeypatch):
    import passage_eval.results as results

    submit(_make_submission("good"), root=tmp_path)
    before = (tmp_path / "leaderboard.csv").read_bytes()
    broken = _make_submission("broken")
    real_replace = results.os.replace

    def fail_csv_replace(source, destination):
        if str(destination).endswith("leaderboard.csv"):
            raise OSError("simulated index failure")
        return real_replace(source, destination)

    monkeypatch.setattr(results.os, "replace", fail_csv_replace)
    with pytest.raises(OSError, match="simulated index failure"):
        submit(broken, root=tmp_path, per_sample={"regret": np.array([0.1])})
    assert (tmp_path / "leaderboard.csv").read_bytes() == before
    assert broken.per_sample_path is None
    assert len(list(tmp_path.rglob("metrics.json"))) == 1
    monkeypatch.undo()
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.method_name.tolist() == ["good"]


def test_concurrent_submissions_keep_one_header_and_all_runs(tmp_path):
    def write(index):
        return submit(_make_submission(f"parallel-{index}"), root=tmp_path)

    with ThreadPoolExecutor(max_workers=8) as pool:
        runs = list(pool.map(write, range(24)))
    assert len(set(runs)) == 24
    assert len(load_leaderboard(tmp_path)) == 24
    assert len(list(tmp_path.rglob("metrics.json"))) == 24
    rebuild_leaderboard(tmp_path)
    assert len(load_leaderboard(tmp_path)) == 24


def test_submitted_artifact_paths_describe_only_written_artifacts(tmp_path):
    sub = _make_submission("no-artifact")
    sub.per_sample_path = "unrelated/stale.npz"
    sub.sidecar_path = "unrelated/stale.json"
    run = submit(sub, root=tmp_path)
    stored = json.loads((run / "metrics.json").read_text())["submission"]
    assert stored["per_sample_path"] is None
    assert stored["sidecar_path"] is None
    assert sub.per_sample_path is None


def test_submission_rejects_per_sample_contract_mismatch_before_writing(tmp_path):
    sub = _make_submission("mismatched")
    sub.extra["metric_contract"] = "current-v1"
    with pytest.raises(ValueError, match="contract"):
        submit(sub, root=tmp_path, per_sample={"metric_contract": np.asarray("other-v1"), "regret": np.array([0.1])})
    assert not list(tmp_path.rglob("metrics.json"))


def test_submission_checks_authoritative_contract_when_csv_is_missing(tmp_path):
    submit(_make_submission("historical"), root=tmp_path)
    (tmp_path / "leaderboard.csv").unlink()
    current = _make_submission("current")
    current.extra["metric_contract"] = "current-v1"
    with pytest.raises(ValueError, match="contract"):
        submit(current, root=tmp_path)
    assert len(list(tmp_path.rglob("metrics.json"))) == 1


@pytest.mark.parametrize(
    "plot_name",
    [
        "plot_grouped_bars_by_accelerator",
        "plot_latency_heatmap",
        "plot_metric_vs_accelerator",
    ],
)
def test_matrix_plots_require_filtering_ambiguous_benchmark_conditions(plot_name):
    import passage_eval
    import matplotlib.pyplot as plt

    first = _make_submission("same", 0.01)
    second = _make_submission("same", 0.99, seed=2026)
    second.split = "test"
    frame = pd.DataFrame([first.to_row(), second.to_row()])
    try:
        with pytest.raises(ValueError, match="[Ff]ilter.*conditions"):
            getattr(passage_eval, plot_name)(frame)
    finally:
        plt.close("all")


def test_scoreboard_exposes_conditions_and_source_artifact():
    from passage_eval import scoreboard_markdown

    first = _make_submission("same", 0.01)
    first.extra = {"metric_contract": "contract-v1", "task_variant": "free"}
    first.per_sample_path = "runs/source/per_sample.npz"
    second = _make_submission("same", 0.99, seed=2026)
    second.split = "test"
    second.extra = first.extra.copy()
    markdown = scoreboard_markdown(pd.DataFrame([first.to_row(), second.to_row()]))
    assert '"split": "val"' in markdown
    assert '"split": "test"' in markdown
    assert "contract-v1" in markdown
    assert "runs/source/per_sample.npz" in markdown


@pytest.mark.parametrize("action", ["submit", "rebuild"])
def test_nested_execution_metrics_are_not_leaderboard_submissions(tmp_path, action):
    first = _make_submission("first")
    first.extra["metric_contract"] = "scoring-v1"
    run = submit(first, root=tmp_path)
    nested = run / "complete_requests" / "warm"
    nested.mkdir(parents=True)
    (nested / "metrics.json").write_text(json.dumps({"cost_regret_median": 0.01, "success_rate": 1.0}))
    if action == "submit":
        second = _make_submission("second")
        second.extra["metric_contract"] = "scoring-v1"
        submit(second, root=tmp_path)
        expected = ["first", "second"]
    else:
        expected = ["first"]
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.method_name.tolist() == expected


def test_submission_refuses_conflicting_sidecar_contract(tmp_path):
    sub = _make_submission("contradictory")
    sub.extra["metric_contract"] = "current-v1"
    with pytest.raises(ValueError, match="contract"):
        submit(sub, root=tmp_path, sidecar={"extra": {"metric_contract": "historical-v0"}})
    assert sub.sidecar_path is None
    assert not list(tmp_path.rglob("metrics.json"))


def test_from_row_accepts_missing_values_from_loaded_dataframe(tmp_path):
    submit(_make_submission("pandas-roundtrip"), root=tmp_path)
    restored = Submission.from_row(load_leaderboard(tmp_path).df.iloc[0])
    assert restored.edge_accelerator is None
    assert restored.commit_sha is None
    assert restored.latency_fwd_ms_p50 is None
    assert restored.per_sample_path is None
    assert restored.extra == {}
    assert type(restored.resolution) is int


@pytest.mark.parametrize("different_method", [False, True])
@pytest.mark.parametrize(
    "plot_name",
    [
        "plot_grouped_bars_by_accelerator",
        "plot_latency_heatmap",
        "plot_metric_vs_accelerator",
    ],
)
def test_matrix_plots_do_not_compare_different_splits_across_accelerators(plot_name, different_method):
    import passage_eval
    import matplotlib.pyplot as plt

    first = _make_submission("same", 0.01)
    first.edge_accelerator = "H100"
    second = _make_submission("other" if different_method else "same", 0.99, seed=2026)
    second.edge_accelerator, second.split = "Thor-GPU", "test"
    try:
        with pytest.raises(ValueError, match="[Ff]ilter.*conditions"):
            getattr(passage_eval, plot_name)(pd.DataFrame([first.to_row(), second.to_row()]))
    finally:
        plt.close("all")


def test_summary_retains_per_seed_checkpoint_controls_as_provenance():
    from passage_eval import summarise_leaderboard

    first = _make_submission("learned", 0.01)
    second = _make_submission("learned", 0.03, seed=2026)
    for sub, checkpoint in [(first, "weights-seed1234"), (second, "weights-seed2026")]:
        sub.extra["component_control"] = {"variant": "direct", "guide": "learned", "checkpoint_id": checkpoint}
    summary = summarise_leaderboard(pd.DataFrame([first.to_row(), second.to_row()]))
    assert summary.n_seeds.tolist() == [2]
    assert summary["mean"].tolist() == pytest.approx([0.02])
    records = json.loads(summary.provenance_json.iloc[0])
    assert {record["extra"]["component_control"]["checkpoint_id"] for record in records} == {
        "weights-seed1234",
        "weights-seed2026",
    }


def test_interrupted_submission_rebuilds_completed_run_and_releases_writer_lock(tmp_path):
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import os
import sys
import passage_eval.results as results
from passage_eval import Submission, submit
def interrupt_before_index(*args, **kwargs):
    os._exit(23)
results._atomic_append_csv = interrupt_before_index
submit(Submission(method_name='interrupted', family='classical', architecture='analytic',
                  metrics={'cost_regret_median': .125}), root=sys.argv[1])
""",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert child.returncode == 23, child.stderr
    assert len(list((tmp_path / "runs").rglob("metrics.json"))) == 1
    assert not (tmp_path / "leaderboard.csv").exists()
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.cost_regret_median.tolist() == [0.125]
    submit(_make_submission("after-interruption"), root=tmp_path)
    assert set(load_leaderboard(tmp_path).df.method_name) == {"interrupted", "after-interruption"}


def test_rebuild_ignores_unfinished_staging_artifacts(tmp_path):
    staging = tmp_path / ".submission-interrupted" / "run"
    staging.mkdir(parents=True)
    (staging / "metrics.json").write_text(json.dumps({"method_name": "unfinished", "metrics": {}}))
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.empty
    assert (staging / "metrics.json").exists()


@pytest.mark.parametrize(
    "values,coverage,estimate",
    [
        ([0.0, 1.0, float("nan")], "2/3", "0.5000 ± 0.7071"),
        ([float("nan")] * 3, "0/3", "unavailable"),
    ],
)
def test_scoreboard_reports_metric_seed_coverage_and_unavailable_estimates(values, coverage, estimate):
    from passage_eval import scoreboard_markdown, summarise_leaderboard

    rows = [_make_submission("coverage", value, seed).to_row() for value, seed in zip(values, [1234, 2026, 3407])]
    summary = summarise_leaderboard(pd.DataFrame(rows))
    assert summary.n_seeds.tolist() == [3]
    assert summary.n_metric_seeds.tolist() == [int(coverage[0])]
    markdown = scoreboard_markdown(pd.DataFrame(rows))
    assert "metric seeds (available / total)" in markdown
    record = next(line for line in markdown.splitlines() if line.startswith("| `coverage`"))
    assert f"| {coverage} | {estimate} |" in record


def test_interrupt_after_index_publication_preserves_committed_run(tmp_path, monkeypatch):
    import passage_eval.results as results

    real_replace = results.os.replace

    def interrupt_after_csv_publication(source, destination):
        real_replace(source, destination)
        if str(destination).endswith("leaderboard.csv"):
            raise KeyboardInterrupt("signal after CSV publication")

    monkeypatch.setattr(results.os, "replace", interrupt_after_csv_publication)
    with pytest.raises(KeyboardInterrupt):
        submit(_make_submission("committed", 0.125), root=tmp_path, per_sample={"regret": np.array([0.125])})
    assert load_leaderboard(tmp_path).df.cost_regret_median.tolist() == [0.125]
    assert len(list((tmp_path / "runs").rglob("metrics.json"))) == 1
    assert len(list((tmp_path / "runs").rglob("per_sample.npz"))) == 1
    monkeypatch.undo()
    rebuild_leaderboard(tmp_path)
    assert load_leaderboard(tmp_path).df.cost_regret_median.tolist() == [0.125]
