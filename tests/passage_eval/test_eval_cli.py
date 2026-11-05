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

"""End-to-end evaluation using local released-format shards and real CLI files."""

import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage_eval.cli import BenchSpec, _build_parser, _spec_from_args, main


def test_cli_scores_released_labels_and_persists_paired_evidence(tmp_path, release_row):
    split = tmp_path / "data" / "3x3" / "validation"
    split.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([release_row]), split / "data-00000.parquet")
    root = tmp_path / "results"
    assert (
        main(
            [
                "bench",
                "--method-name",
                "oracle",
                "--family",
                "classical",
                "--architecture",
                "analytic",
                "--inference",
                "passage_eval:AStarPlanner",
                "--resolution",
                "3",
                "--dataset-path",
                str(tmp_path / "data"),
                "--results-root",
                str(root),
                "--n-samples",
                "1",
                "--seeds",
                "1234",
                "--no-latency",
            ]
        )
        == 0
    )
    run = next(root.glob("runs/**/metrics.json")).parent
    metrics = json.loads((run / "metrics.json").read_text())["metrics"]
    assert metrics["cost_regret_median"] == pytest.approx(0)
    assert metrics["success_rate"] == 1
    with np.load(run / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["sample_id"].tolist() == [release_row["id"]]
        assert arrays["regret"].tolist() == pytest.approx([0])
        assert arrays["metric_contract"].item() == "passage-qualified-complete-v1"
    sidecar = json.loads((run / "sidecar.json").read_text())
    stored = json.loads((run / "metrics.json").read_text())["submission"]
    assert json.loads(stored["extra_json"])["metric_contract"] == "passage-qualified-complete-v1"
    assert sidecar["extra"]["metric_contract"] == "passage-qualified-complete-v1"
    assert sidecar["dataset_content_hash"]
    assert sidecar["extra"]["task_variant"] == "obstacles"
    assert sidecar["extra"]["sample_ids"] == [release_row["id"]]
    assert main(["leaderboard", "--root", str(root)]) == 0
    assert main(["rebuild", "--root", str(root)]) == 0
    from passage_eval import load_leaderboard

    assert load_leaderboard(root).df.method_name.tolist() == ["oracle"]


def test_spec_paths_are_respected_unless_explicitly_overridden(tmp_path):
    path = tmp_path / "method.json"
    spec = BenchSpec("method", "classical", "analytic", results_root=str(tmp_path / "chosen"))
    path.write_text(json.dumps(spec.to_dict()))
    args = _build_parser().parse_args(["bench", "--spec", str(path)])
    assert _spec_from_args(args).results_root == spec.results_root


def test_latency_reports_tail_quantiles_over_multiple_scenes(metre_scene):
    from passage_eval import AStarPlanner
    from passage_eval.cli import _measure_e2e_latencies

    timing = _measure_e2e_latencies(AStarPlanner(), [metre_scene], warmup=1, repeats=5)
    assert 0 <= timing["latency_e2e_ms_p50"] <= timing["latency_e2e_ms_p95"] <= timing["latency_e2e_ms_p99"]
    assert timing["throughput_samples_per_s"] > 0


def test_explicit_flags_override_spec_without_resetting_other_values(tmp_path):
    path = tmp_path / "method.json"
    spec = BenchSpec("method", "classical", "analytic", resolution=64, n_samples=8)
    path.write_text(json.dumps(spec.to_dict()))
    args = _build_parser().parse_args(["bench", "--spec", str(path), "--split", "calibration", "--n-samples", "2"])
    actual = _spec_from_args(args)
    assert actual.split == "calibration"
    assert actual.n_samples == 2
    assert actual.resolution == 64


def test_candidate_cannot_mutate_scoring_inputs(metre_scene):
    from passage_eval import Planner, PlannerOutput, score_planner

    class MutatingPlanner(Planner):
        def predict_path(self, scene):
            scene.elevation[:] = 100
            scene.obstacles[:] = True
            return PlannerOutput([(0, 0), (1, 1), (2, 2)], 0)

    score_planner(MutatingPlanner(), [metre_scene])
    assert not metre_scene.elevation.any()
    assert not metre_scene.obstacles.any()


def test_seed_checkpoint_factories_and_timing_provenance(tmp_path, metre_scene, monkeypatch):
    from passage_eval import Planner, PlannerOutput
    from passage_eval.cli import cmd_bench
    import passage_eval.cli as cli
    from types import SimpleNamespace

    constructors = []
    seeded = []

    class SeededPlanner(Planner):
        def __init__(self, checkpoint):
            constructors.append(checkpoint)

        def predict_path(self, scene):
            return PlannerOutput([(0, 0), (1, 1), (2, 2)], float("nan"))

    def load_entry(entry):
        # Candidate runtime appears during import; constructor seeding follows.
        monkeypatch.setitem(
            cli.sys.modules,
            "torch",
            SimpleNamespace(
                manual_seed=lambda seed: seeded.append(seed),
                cuda=SimpleNamespace(is_initialized=lambda: False),
            ),
        )
        return SeededPlanner

    monkeypatch.setattr(cli, "_import_entry", load_entry)
    monkeypatch.setattr(cli, "_load_scenes", lambda spec: [metre_scene])
    checkpoints = {}
    for seed in (1234, 2026, 3407):
        checkpoint = tmp_path / f"{seed}.pt"
        checkpoint.write_bytes(f"fixture checkpoint {seed}".encode())
        checkpoints[seed] = str(checkpoint)
    spec = BenchSpec(
        "replica",
        "classical",
        "analytic",
        inference="test:Planner",
        results_root=str(tmp_path),
        latency_warmup=0,
        latency_repeats=2,
        inference_kwargs_by_seed={str(s): {"checkpoint": checkpoints[s]} for s in (1234, 2026, 3407)},
    )
    cmd_bench(spec)
    assert constructors == [checkpoints[s] for s in (1234, 2026, 3407)]
    assert seeded == [1234, 2026, 3407]
    for path in tmp_path.glob("runs/**/sidecar.json"):
        sidecar = json.loads(path.read_text())
        assert sidecar["determinism_flag"] == "non_deterministic"
        assert sidecar["extra"]["latency_repeats"] == 2
        assert len(sidecar["extra"]["evaluator_source_hash"]) == 64
        censored = sidecar["extra"]["latency_e2e_censored"]  # proposed rule, beside the legacy keys
        assert censored["status"] == "proposed_not_sealed" and censored["n_requests"] == 2
        assert censored["n_repeats"] == 2 and censored["n_distinct_scenes"] == 1
        # latency_warmup=0: the warm series follows no declared warm-up, whatever its name
        assert censored["warmup"]["source"] == "bench_spec_in_process_runner"
        assert censored["meets_min_warm_requests"] is False
        assert censored["meets_min_warm_requests_reason"] == "no_declared_warmup"
        assert {"latency_e2e_ms_p50", "latency_e2e_ms_p95", "latency_e2e_ms_p99"} <= set(sidecar["extra"])
        # The in-process runner times validation and recost inside the request and records no separate value.
        recost = sidecar["extra"]["validation_recost_ms"]
        assert recost["reported_separately"] is True and recost["inside_scored_latency"] is True
        assert recost["status"] == "not_recorded" and recost["n_requests"] == 2 and recost["n_recorded"] == 0
    from passage_eval import load_leaderboard, rebuild_leaderboard, summarise_leaderboard

    rebuild_leaderboard(tmp_path)
    summary = summarise_leaderboard(load_leaderboard(tmp_path).df)
    assert summary.n_seeds.tolist() == [3]
    assert summary.n_runs.tolist() == [3]


@pytest.mark.parametrize(
    "setting,values",
    [
        ("budget", [{"max_expansions": 1}, {"max_expansions": 100}]),
        (
            "component_control",
            [
                {"variant": "exact_classical", "guide": "none"},
                {"variant": "budgeted_classical", "guide": "none"},
            ],
        ),
        ("hardware_stack", ["runtime-A", "runtime-B"]),
    ],
)
def test_cli_retains_condition_bearing_provenance_for_seed_summary(tmp_path, release_row, setting, values):
    from passage_eval import load_leaderboard, summarise_leaderboard
    from passage_eval.cli import cmd_bench

    split = tmp_path / "data" / "3x3" / "validation"
    split.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([release_row]), split / "data-00000.parquet")
    for seed, value in zip([1234, 2026], values):
        spec = BenchSpec(
            "same-procedure-name",
            "classical",
            "analytic",
            inference="passage_eval.baselines.classical:BoundedExactPlanner",
            resolution=3,
            seeds=[seed],
            dataset_path=str(tmp_path / "data"),
            results_root=str(tmp_path / "results"),
            n_samples=1,
            measure_latency=False,
            **{setting: value},
        )
        cmd_bench(spec)
    summary = summarise_leaderboard(load_leaderboard(tmp_path / "results").df)
    assert len(summary) == 2
    assert summary.n_seeds.tolist() == [1, 1]
    condition_values = [json.loads(value)[setting] for value in summary.comparison_extra_json]
    if setting == "component_control":
        assert {value["variant"] for value in condition_values} == {"exact_classical", "budgeted_classical"}
    else:
        assert condition_values == values
    if setting == "budget":
        assert summary["mean"].tolist() == [999999.0, 0.0]


def test_cli_init_rebuild_and_leaderboard(tmp_path, capsys):
    spec_path = tmp_path / "starter.yaml"
    assert main(["init-spec", "--out", str(spec_path)]) == 0
    spec = BenchSpec.from_file(spec_path)
    assert spec.cost_weight is None
    assert main(["leaderboard", "--root", str(tmp_path / "results")]) == 0
    assert "empty leaderboard" in capsys.readouterr().out
    assert main(["rebuild", "--root", str(tmp_path / "results")]) == 0


@pytest.mark.parametrize(
    "entry,error",
    [
        ("invalid", ValueError),
        ("passage_eval:no_such_entry", AttributeError),
        ("passage_eval:DEFAULT_SEEDS", TypeError),
    ],
)
def test_cli_rejects_invalid_candidates(entry, error):
    from passage_eval.cli import _materialise_planner

    with pytest.raises(error):
        _materialise_planner(BenchSpec("invalid", "classical", "analytic", inference=entry))


@pytest.mark.parametrize("seeds,batch", [([], 1), ([1234, 1234], 1), ([1234], 0)])
def test_invalid_campaign_fails_before_data_loading(seeds, batch, monkeypatch):
    import passage_eval.cli as cli

    monkeypatch.setattr(cli, "_load_scenes", lambda spec: pytest.fail("must validate before loading"))
    with pytest.raises(ValueError):
        cli.cmd_bench(BenchSpec("invalid", "classical", "analytic", seeds=seeds, batch_size=batch))


def test_latency_starts_each_call_with_the_same_released_input(metre_scene):
    from passage_eval import Planner, PlannerOutput
    from passage_eval.cli import _measure_e2e_latencies

    class PreprocessingPlanner(Planner):
        def predict_path(self, scene):
            assert not scene.elevation.any()
            scene.elevation[:] = 1
            return PlannerOutput([(0, 0), (1, 1), (2, 2)], float("nan"))

    _measure_e2e_latencies(PreprocessingPlanner(), [metre_scene], warmup=1, repeats=2)
    assert not metre_scene.elevation.any()
