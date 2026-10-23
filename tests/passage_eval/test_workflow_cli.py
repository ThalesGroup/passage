"""Bounded offline workflow receipts and admission regressions."""

import json
from dataclasses import replace
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage_eval.baselines.classical import BoundedExactPlanner
from passage_eval.cli import BenchSpec, cmd_bench, cmd_run, main
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult


def local_spec(tmp_path, release_row, **changes):
    split = tmp_path / "data" / "3x3" / "validation"
    split.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([release_row]), split / "data-00000.parquet")
    spec = BenchSpec(
        "prototype",
        "classical",
        "bounded_dijkstra",
        inference="passage_eval.baselines.classical:BoundedExactPlanner",
        resolution=3,
        n_samples=1,
        seeds=[1234],
        dataset_path=str(tmp_path / "data"),
        latency_repeats=2,
        latency_warmup=1,
        results_root=str(tmp_path / "results"),
    )
    spec = replace(spec, **changes)
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return spec, path


def test_conform_run_score_local_roundtrip(tmp_path, release_row):
    _, spec_path = local_spec(tmp_path, release_row)
    assert main(["conform", "--spec", str(spec_path), "--output", str(tmp_path / "conform")]) == 0
    receipt = json.loads((tmp_path / "conform" / "conformance.json").read_text())
    assert receipt["status"] == "passed"
    assert receipt["harness_checks"]["absolute_deadline"] == "passed"
    assert receipt["harness_checks"]["ordinary_input_no_reference_fields"] == "passed"
    assert receipt["harness_checks"]["terminal_failure_accounting"] == "passed"
    assert receipt["claim_scope"] == "cpu_contract_checks_not_device_or_population_certification"
    run_root = tmp_path / "run"
    assert main(["run", "--spec", str(spec_path), "--output", str(run_root)]) == 0
    seed_root = run_root / "seed-1234"
    assert not (seed_root / "score_receipt.json").exists()
    assert main(["score", "--run", str(run_root)]) == 0
    expected = [(seed_root, 2), (seed_root / "cold", 1), (seed_root / "warmup", 1)]
    for directory, repeats in expected:
        manifest = json.loads((directory / "run_manifest.json").read_text())
        score = json.loads((directory / "score_receipt.json").read_text())
        assert manifest["admission"]["qualified_reference"]
        assert manifest["reference_identity"]
        assert score["qualified_reference"]
        assert score["reference_policy"] == "qualified_exact"
        assert score["scored_repeats"] == repeats
        with np.load(directory / "per_sample.npz", allow_pickle=False) as arrays:
            assert arrays["regret"].tolist() == pytest.approx([0] * repeats)
    with pytest.raises(FileExistsError):
        main(["run", "--spec", str(spec_path), "--output", str(run_root)])


def test_compare_report_cli_actual_scored_artifacts(tmp_path, release_row):
    first_spec, first_path = local_spec(tmp_path, release_row)
    second = replace(first_spec, method_name="second_prototype", budget={"max_expansions": 100})
    second_path = tmp_path / "second.json"
    second_path.write_text(json.dumps(second.to_dict()))
    roots = [tmp_path / "first", tmp_path / "second"]
    for spec_path, root in zip([first_path, second_path], roots):
        assert main(["run", "--spec", str(spec_path), "--output", str(root)]) == 0
        assert main(["score", "--run", str(root)]) == 0
    comparison = tmp_path / "comparison.json"
    assert (
        main(
            [
                "compare",
                "--score",
                str(roots[0] / "seed-1234"),
                "--score",
                str(roots[1] / "seed-1234"),
                "--output",
                str(comparison),
                "--bootstrap-resamples",
                "10",
            ]
        )
        == 0
    )
    data = json.loads(comparison.read_text())
    assert data["n_tasks"] == 1
    assert data["primary_repeat_index"] == 0
    assert data["methods"][0]["all_repeat_latency_ms"]["n"] == 2
    assert data["uncertainty"]["status"] == "unknown_geographic_blocks"
    report = tmp_path / "report.md"
    assert main(["report", "--compare", str(comparison), "--output", str(report)]) == 0
    assert "1 identical frozen tasks" in report.read_text()
    json_report = tmp_path / "report.json"
    assert main(["report", "--compare", str(comparison), "--output", str(json_report), "--format", "json"]) == 0
    assert json.loads(json_report.read_text()) == data
    odd_spec = {
        "axes": ["elev_std", "endpoint_separation_normalized", "obstacle_fraction"],
        "edges": {"elev_std": [0, 1], "endpoint_separation_normalized": [0, 1], "obstacle_fraction": [0, 0.5, 1]},
    }
    odd_path = tmp_path / "odd.json"
    odd_path.write_text(json.dumps(odd_spec))
    assert (
        main(
            [
                "compare",
                "--score",
                str(roots[0] / "seed-1234"),
                "--score",
                str(roots[1] / "seed-1234"),
                "--output",
                str(comparison),
                "--odd-spec",
                str(odd_path),
                "--bootstrap-resamples",
                "10",
            ]
        )
        == 0
    )
    coverage = json.loads(comparison.read_text())["joint_odd_coverage"]
    assert coverage["spec"] == odd_spec
    assert len(coverage["spec_sha256"]) == 64
    assert coverage["n_common_cells"] == 1


def test_timed_stateful_outputs_and_failures_are_scored(tmp_path, release_row, monkeypatch):
    module = ModuleType("workflow_candidate")

    class Stateful:
        def __init__(self):
            self.calls = 0

        def plan(self, request):
            assert isinstance(request, PlanRequest)
            assert not hasattr(request, "oracle_path")
            assert "solvers" not in request.metadata
            self.calls += 1
            if self.calls % 2:
                return PlanResult([(0, 0), (1, 1), (2, 2)], precision="float32")
            return PlanResult([], status="no_path", precision="float32")

    module.Stateful = Stateful
    monkeypatch.setitem(__import__("sys").modules, module.__name__, module)
    spec, path = local_spec(tmp_path, release_row, inference="workflow_candidate:Stateful", latency_warmup=0)
    run = tmp_path / "requests"
    main(["run", "--spec", str(path), "--output", str(run)])
    main(["score", "--run", str(run)])
    with np.load(run / "seed-1234" / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["status"].tolist() == ["no_path", "ok"]
        assert arrays["regret"].tolist() == pytest.approx([999999, 0])
        assert arrays["route_hash"][0] != arrays["route_hash"][1]
        assert len(arrays["latency_ms"]) == 2
    result = cmd_bench(spec)
    metrics = json.loads((result / "metrics.json").read_text())["metrics"]
    assert metrics["success_rate"] == 0
    assert (result / "complete_requests" / "score_receipt.json").exists()
    assert (result / "complete_requests" / "cold" / "score_receipt.json").exists()


def test_unqualified_reference_rejected_before_candidate_construction(tmp_path, release_row, monkeypatch):
    import passage_eval.cli as cli

    release_row["path_astar_elevation_obstacles"] = [[0, 0], [1, 0], [2, 1], [2, 2]]
    release_row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"] = 60 + 30 * np.sqrt(2)
    spec, _ = local_spec(tmp_path, release_row)
    monkeypatch.setattr(
        cli, "_materialise_planner", lambda spec: pytest.fail("candidate admission must follow qualification")
    )
    with pytest.raises(ValueError, match="stored_path_suboptimal"):
        cmd_run(spec, tmp_path / "run")


def test_fresh_named_cohort_preserves_source_label_defect(tmp_path, release_row):
    release_row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"] += 1
    _, spec_path = local_spec(
        tmp_path,
        release_row,
        reference_policy="independent_analytic_reference",
        reference_cohort_id="tiny-fresh-reference-v1",
    )
    run = tmp_path / "fresh"
    assert main(["run", "--spec", str(spec_path), "--output", str(run)]) == 0
    assert main(["score", "--run", str(run)]) == 0
    manifest = json.loads((run / "seed-1234" / "run_manifest.json").read_text())
    reference = manifest["reference_qualification"][release_row["id"]]
    assert reference["qualified"]
    assert not reference["source_label_qualified"]
    assert "stored_cost_disagreement" in reference["source_label_qualification_reasons"]
    assert manifest["reference_policy"] == "independent_analytic_reference"
    assert manifest["reference_cohort_id"] == "tiny-fresh-reference-v1"
    with np.load(run / "seed-1234" / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["regret"].tolist() == pytest.approx([0, 0])


def test_independent_cohort_requires_name_before_data_loading(tmp_path, monkeypatch):
    import passage_eval.cli as cli

    monkeypatch.setattr(cli, "_load_scenes", lambda spec: pytest.fail("validate policy before loading"))
    with pytest.raises(ValueError, match="reference_cohort_id"):
        cmd_run(BenchSpec("m", "classical", "bounded", reference_policy="independent_analytic_reference"), tmp_path)


def test_procedure_identity_binds_external_entrypoint_source(tmp_path, metre_scene, monkeypatch):
    import sys
    from passage_eval.cli import _manifest

    path = tmp_path / "external_candidate.py"
    module = ModuleType("external_candidate")
    module.__file__ = str(path)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = BenchSpec("external", "classical", "custom", inference="external_candidate:Candidate")
    source = "class Candidate:\n    def plan(self, request):\n        return []\n"
    path.write_text(source)
    exec(compile(source, str(path), "exec"), module.__dict__)
    before = _manifest(spec, [metre_scene], {})
    changed = source.replace("return []", "return [(0, 0)]")
    path.write_text(changed)
    exec(compile(changed, str(path), "exec"), module.__dict__)
    after = _manifest(spec, [metre_scene], {})
    assert before["procedure_id"] != after["procedure_id"]
    files = before["candidate_identity"]["module_files"]
    assert (
        files["external_candidate"]["sha256"]
        != after["candidate_identity"]["module_files"]["external_candidate"]["sha256"]
    )
    assert before["candidate_identity"]["loaded_code_sha256"] != after["candidate_identity"]["loaded_code_sha256"]


def test_sourceless_candidate_requires_explicit_identity(monkeypatch):
    import sys
    from passage_eval.cli import _candidate_identity

    module = ModuleType("sourceless_candidate")
    exec("def candidate(request):\n    return []\n", module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = BenchSpec("external", "classical", "custom", inference="sourceless_candidate:candidate")
    with pytest.raises(ValueError, match="no readable file"):
        _candidate_identity(spec)
    identity = _candidate_identity(replace(spec, candidate_source_sha256="a" * 64))
    assert identity["source_verification"] == "explicit_declared_identity"
    assert identity["module_files"] == {}


def test_runtime_bundle_dependencies_are_verified_and_change_identity(tmp_path, metre_scene):
    import hashlib
    from passage_eval.cli import _manifest

    helper = tmp_path / "helper.bin"
    helper.write_bytes(b"first runtime")
    bundle = tmp_path / "bundle.json"

    def write_bundle():
        bundle.write_text(
            json.dumps(
                {
                    "schema": "passage-runtime-bundle-v1",
                    "files": [{"path": helper.name, "sha256": hashlib.sha256(helper.read_bytes()).hexdigest()}],
                }
            )
        )

    write_bundle()
    spec = BenchSpec(
        "bounded",
        "classical",
        "prototype",
        inference="passage_eval.baselines.classical:BoundedExactPlanner",
        runtime_bundle_manifest=str(bundle),
    )
    before = _manifest(spec, [metre_scene], {})
    helper.write_bytes(b"second runtime")
    with pytest.raises(ValueError, match="runtime bundle file identity mismatch"):
        _manifest(spec, [metre_scene], {})
    write_bundle()
    after = _manifest(spec, [metre_scene], {})
    assert before["procedure_id"] != after["procedure_id"]


def test_rescore_uses_frozen_candidate_identity_without_import(tmp_path, release_row, monkeypatch):
    import passage_eval.cli as cli

    spec, _ = local_spec(tmp_path, release_row)
    root = tmp_path / "run"
    cmd_run(spec, root)
    monkeypatch.setattr(cli, "_import_entry", lambda entry: pytest.fail("offline scoring cannot import a candidate"))
    assert main(["score", "--run", str(root)]) == 0
    manifest_path = root / "seed-1234" / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["procedure"]["inference"] = "tampered:entry"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="persisted candidate/procedure identity"):
        main(["score", "--run", str(root)])


@pytest.mark.parametrize(
    "changes,message",
    [
        ({"batch_size": 2}, "batch_size=1"),
        ({"quality_repeat_index": 2}, "quality_repeat_index"),
        ({"budget": {"max_expansions": 0}}, "positive integer"),
    ],
)
def test_invalid_procedure_rejected_before_loading(tmp_path, changes, message, monkeypatch):
    import passage_eval.cli as cli

    monkeypatch.setattr(cli, "_load_scenes", lambda spec: pytest.fail("validate first"))
    with pytest.raises(ValueError, match=message):
        cmd_run(replace(BenchSpec("m", "classical", "bounded", latency_repeats=2), **changes), tmp_path)


def test_remote_revision_and_missing_local_root_fail_explicitly(tmp_path):
    from passage_eval.cli import _load_scenes

    spec = BenchSpec("m", "classical", "bounded")
    assert spec.dataset_repo == "thalesgroup/passage"
    with pytest.raises(ValueError, match="explicit dataset_revision"):
        _load_scenes(spec)
    with pytest.raises(FileNotFoundError, match="local dataset root"):
        _load_scenes(replace(spec, dataset_path=str(tmp_path / "missing")))


def test_console_cli_imports_candidate_module_from_working_directory(tmp_path, monkeypatch):
    import sys
    from passage_eval.cli import _import_entry

    (tmp_path / "cwd_algorithm_fixture.py").write_text("class Algorithm:\n    pass\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [item for item in sys.path if item not in ("", str(tmp_path))])
    assert _import_entry("cwd_algorithm_fixture:Algorithm").__name__ == "Algorithm"


def test_callable_route_snapshot_waits_for_completion(metre_scene):
    from passage_eval.cli import _CallablePlanner
    from passage_eval.execution import run_requests

    pending = np.zeros((3, 2), dtype=np.int64)
    planner = _CallablePlanner(lambda **kwargs: pending)
    calls = [0]

    def sync():
        calls[0] += 1
        if calls[0] == 2:
            pending[:] = [[0, 0], [1, 1], [2, 2]]

    artifact = run_requests(planner, [PlanRequest.from_scene(metre_scene)], synchronize=sync)
    assert artifact.records[0]["status"] == "ok"
    np.testing.assert_array_equal(artifact.paths[0], pending)


def test_checkpoint_identity_is_rechecked_after_construction(tmp_path, metre_scene, monkeypatch):
    import sys
    from passage_eval.cli import _execute_series

    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"frozen checkpoint")
    module = ModuleType("mutating_checkpoint_fixture")
    source = "from pathlib import Path\nclass Candidate:\n    def __init__(self, checkpoint):\n        Path(checkpoint).write_bytes(b'changed checkpoint')\n    def plan(self, request):\n        return []\n"
    source_path = tmp_path / "mutating_checkpoint_fixture.py"
    source_path.write_text(source)
    module.__file__ = str(source_path)
    exec(compile(source, str(source_path), "exec"), module.__dict__)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    spec = BenchSpec(
        "mutable",
        "classical",
        "fixture",
        inference="mutating_checkpoint_fixture:Candidate",
        inference_kwargs={"checkpoint": str(checkpoint)},
        seeds=[1234],
        measure_latency=False,
    )
    with pytest.raises(ValueError, match="checkpoint.*construction"):
        _execute_series(spec, [metre_scene], {}, tmp_path / "run")


@pytest.mark.parametrize(
    "budget", [BudgetSpec(max_expansions=1), BudgetSpec(max_queue_operations=1), BudgetSpec(max_live_states=1)]
)
def test_classical_work_limits_are_observed(metre_scene, budget):
    result = BoundedExactPlanner().plan(PlanRequest.from_scene(metre_scene, budget=budget))
    assert result.status == "budget_exhausted"
    assert result.path == []
    for key, limit in (
        ("expansions", budget.max_expansions),
        ("queue_operations", budget.max_queue_operations),
        ("live_states", budget.max_live_states),
    ):
        if limit is not None:
            assert result.work[key] <= limit


@pytest.mark.parametrize(
    "model,terrain",
    [
        ("elevation", [[0.0, 1.0]]),
        ("energy", [[0.0, 1.0, 2.0]]),
    ],
)
def test_classical_prototype_cannot_label_arithmetic_overflow_unreachable(model, terrain, metre_scene):
    from passage_eval.execution import run_requests

    terrain = np.asarray(terrain, dtype=float)
    scene = replace(
        metre_scene,
        elevation=terrain,
        obstacles=np.zeros_like(terrain, dtype=bool),
        start=(0, 0),
        goal=(0, terrain.shape[1] - 1),
        cost_model=model,
        cost_weight=1e308,
    )
    artifact = run_requests(BoundedExactPlanner(), [PlanRequest.from_scene(scene)])
    assert artifact.records[0]["status"] == "runtime_error"
    assert artifact.records[0]["legal_path"] is False


@pytest.mark.parametrize("model", ["distance", "elevation", "absolute_elevation", "slope_uphill", "energy"])
def test_classical_prototype_matches_bounded_exact_reference(model, metre_scene):
    from passage.audit import exact_reference
    from passage_eval.metrics.cost import path_cost

    scene = replace(metre_scene, cost_model=model, elevation=np.arange(9, dtype=float).reshape(3, 3))
    request = PlanRequest.from_scene(scene)
    output = BoundedExactPlanner().plan(request)
    reference = exact_reference(
        scene.elevation, scene.obstacles, scene.start, scene.goal, cost_model=model, weight=scene.cost_weight
    )
    assert output.status == "ok"
    assert path_cost(scene.elevation, output.path, cost_model=model, weight=scene.cost_weight) == pytest.approx(
        reference["cost"]
    )


def test_canonical_astar_uses_ordinary_inputs_and_retains_censored_work(metre_scene):
    from passage_eval.baselines.classical import CanonicalAStarPlanner
    from passage_eval.execution import run_requests

    request = PlanRequest.from_scene(metre_scene)
    exact = run_requests(CanonicalAStarPlanner(), [request]).records[0]
    assert exact["status"] == "ok"
    assert exact["recost"] == pytest.approx(metre_scene.oracle_cost)
    assert exact["work"]["expansions"] > 0
    bounded = run_requests(CanonicalAStarPlanner(), [replace(request, budget=BudgetSpec(max_expansions=1))]).records[0]
    assert bounded["status"] == "budget_exhausted"
    assert bounded["work"]["expansions"] == 1
