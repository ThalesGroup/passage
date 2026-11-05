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

"""Frozen diagnostic route campaigns retain defects and reject stale evidence."""

import copy
import fcntl
import importlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage.dataset import DatasetSource
from passage.qualification import _hash, run_metadata_census
from passage.utils import encode_tensor_blob


@pytest.fixture
def campaign(tmp_path):
    root, metadata, output = (tmp_path / name for name in ("data", "metadata", "routes"))
    tensor = np.zeros((64, 64, 3), dtype=np.float32)
    tensor[:, :, 0] = 0.5
    tensor[:, :, 1] = -1
    tensor[0, 0, 1], tensor[2, 2, 1] = 0, 1
    path = [[0, 0], [1, 1], [2, 2]]
    solvers = {
        f"astar_{model}": {"cost_model": model, "weight": 62.4, "diagonal_movement": "always"}
        for model in ("elevation", "slope", "energy")
    }
    row = {
        "id": "../../unsafe/name",
        "tensor": encode_tensor_blob(tensor),
        "metadata": {
            "resolution": 64,
            "split": "validation",
            "calibration": {"min": 0.0, "max": 100.0},
            "crop": {
                "min_elevation": 0.0,
                "max_elevation": 250.0,
                "top_left": {"lat": 1.0, "lon": 0.0},
                "bottom_right": {"lat": 0.0, "lon": 1.0},
            },
            "tile": {"source": {"name": "tile"}, "used": [{"name": "tile"}], "overlapping": [{"name": "tile"}]},
            "obstacles": {"ratio_actual": 0.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": solvers,
            "paths": {
                solver: {variant: {"cost": 60 * np.sqrt(2)} for variant in ("free", "obstacles")} for solver in solvers
            },
        },
    }
    row.update({f"path_{solver}_{variant}": path for solver in solvers for variant in ("free", "obstacles")})
    shard = root / "64x64" / "validation" / "part.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), shard)
    source = DatasetSource(root, repo="fixture", revision="a" * 40)
    run_metadata_census(source, metadata, resolutions=(64,), spatial=False)
    return source, metadata, output, row, shard


def run(campaign, **kwargs):
    from passage.qualification_runner import run_route_audit

    source, metadata, output, _, _ = campaign
    return run_route_audit(source, metadata, output, workers=kwargs.pop("workers", 1), **kwargs)


def test_all_six_routes_are_read_and_frozen_before_outcomes(campaign):
    result = run(campaign)
    assert result["status"] == "complete"
    assert result["coverage"] == {
        "selected_rows": 1,
        "completed_rows": 1,
        "expected_task_records": 6,
        "completed_task_records": 6,
    }
    assert result["counts"]["qualified"] == 6
    assert result["counts"]["reference_exact"] == 6
    assert result["by_resolution"]["64"]["task_records"] == 6
    assert result["population_estimation"] is False and result["content_verified"] is False
    output = campaign[2]
    protocol = json.loads((output / "protocol.json").read_text())
    assert protocol["max_expansions"] == 100000
    assert protocol["cost_models"] == ["elevation", "slope_uphill", "energy"]
    assert protocol["new_reference_cohort_admitted"] is False
    from passage.qualification_runner import IMPLEMENTATION_MODULES

    assert set(protocol["implementation_sha256"]) == set(IMPLEMENTATION_MODULES)
    files = list((output / "findings").glob("*.json"))
    assert len(files) == 1 and len(files[0].stem) == 64
    finding = json.loads(files[0].read_text())
    assert finding["finding_sha256"] == _hash({k: v for k, v in finding.items() if k != "finding_sha256"})
    assert finding["source_stat"]["size_bytes"] == campaign[4].stat().st_size
    assert len(finding["records"]) == 6
    assert all(record["reference"]["max_expansions"] == 100000 for record in finding["records"])
    assert len(result["qualified_reference_tasks"]) == 6
    assert all(task["support_cluster_id"] for task in result["qualified_reference_tasks"])
    assert all(task["source_cell_status"] == "unknown" for task in result["qualified_reference_tasks"])


def test_resume_uses_identical_findings_without_reference_recompute(campaign, monkeypatch):
    first = run(campaign)
    from passage import qualification_runner

    monkeypatch.setattr(qualification_runner, "_audit_entry", lambda *a: pytest.fail("must resume"))
    second = run(campaign)
    assert second["resumed_rows"] == 1
    assert first["findings"] == second["findings"]
    assert first["qualified_reference_tasks"] == second["qualified_reference_tasks"]


def test_source_changed_after_completion_rejects_cached_findings(campaign):
    run(campaign)
    with campaign[4].open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="changed"):
        run(campaign)
    summary = json.loads((campaign[2] / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["roster_sha256"]


def test_missing_path_is_retained_as_unqualified_not_silently_dropped(campaign):
    source, metadata, output, row, shard = campaign
    broken = copy.deepcopy(row)
    del broken["path_astar_slope_obstacles"]
    pq.write_table(pa.Table.from_pylist([broken]), shard)
    new_metadata = metadata.with_name("metadata-broken")
    run_metadata_census(source, new_metadata, resolutions=(64,), spatial=False)
    result = run((source, new_metadata, output, broken, shard))
    assert result["counts"]["task_records"] == 6
    assert result["counts"]["qualified"] == 5
    assert result["counts"]["qualification_reasons"]["stored_path_illegal"] == 1


def test_invalid_tensor_keeps_six_terminal_task_records(campaign):
    source, metadata, output, row, shard = campaign
    broken = copy.deepcopy(row)
    broken["tensor"] = b"invalid"
    pq.write_table(pa.Table.from_pylist([broken]), shard)
    new_metadata = metadata.with_name("metadata-broken")
    run_metadata_census(source, new_metadata, resolutions=(64,), spatial=False)
    result = run((source, new_metadata, output, broken, shard))
    assert result["counts"]["qualified"] == 0 and result["counts"]["reference_not_run"] == 6
    assert result["counts"]["row_statuses"]["invalid_metadata"] == 1


@pytest.mark.parametrize("target", ["roster", "finding", "protocol", "summary"])
def test_tampered_frozen_evidence_is_rejected(campaign, target):
    run(campaign)
    if target == "roster":
        path = campaign[1] / "audit_roster.json"
    elif target == "finding":
        path = next((campaign[2] / "findings").glob("*.json"))
    elif target == "summary":
        path = campaign[1] / "summary.json"
    else:
        path = campaign[2] / "protocol.json"
    payload = json.loads(path.read_text())
    payload["tampered"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        run(campaign)


@pytest.mark.parametrize("workers", [0, 9, True, 1.5])
def test_worker_count_is_bounded_before_output_creation(campaign, workers):
    with pytest.raises(ValueError, match="workers"):
        run(campaign, workers=workers)
    assert not campaign[2].exists()


def test_spawn_workers_use_real_frozen_source(campaign):
    result = run(campaign, workers=2)
    assert result["status"] == "complete" and result["counts"]["qualified"] == 6


def test_incomplete_metadata_and_overlap_are_rejected(campaign):
    source, metadata, _, row, shard = campaign
    with pytest.raises(ValueError, match="outside|overlap"):
        run((source, metadata, source.root / "routes", row, shard))
    summary_path = metadata / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["status"] = "in_progress"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="complete"):
        run(campaign)


def test_active_writer_is_rejected(campaign):
    output = campaign[2]
    output.mkdir()
    with (output / ".route-audit.lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="writer"):
            run(campaign)


def test_censoring_remains_in_task_denominators_and_never_qualifies(campaign, monkeypatch):
    from passage import qualification_runner

    original = qualification_runner.audit_row

    def censored(*args, **kwargs):
        assert kwargs["max_expansions"] == 100000
        finding = original(*args, **kwargs)
        for record in finding["records"]:
            record.update(
                reference_status="budget_exhausted",
                qualified=False,
                optimality=None,
                qualification_reasons=["reference_budget_exhausted"],
            )
        return finding

    monkeypatch.setattr(qualification_runner, "audit_row", censored)
    result = run(campaign)
    assert result["counts"]["reference_censored"] == result["counts"]["task_records"] == 6
    assert result["counts"]["qualified"] == 0
    assert result["qualified_reference_tasks"] == []


def test_interruption_publishes_failed_summary_and_can_resume(campaign, monkeypatch):
    from passage import qualification_runner

    original = qualification_runner._audit_entry

    def interrupted(*args):
        raise KeyboardInterrupt("interrupted fixture")

    monkeypatch.setattr(qualification_runner, "_audit_entry", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run(campaign)
    summary = json.loads((campaign[2] / "summary.json").read_text())
    assert summary["status"] == "failed" and summary["coverage"]["completed_rows"] == 0
    assert summary["failure"]["error_type"] == "KeyboardInterrupt"
    monkeypatch.setattr(qualification_runner, "_audit_entry", original)
    assert run(campaign)["counts"]["qualified"] == 6


def test_authentication_expected_manifest_bound_without_claiming_authentication(campaign):
    source, metadata, output, row, shard = campaign
    from passage.authentication import _json_bytes
    import hashlib

    expected = {
        "schema_version": 1,
        "root": str(source.root.resolve()),
        "repo": source.repo,
        "revision": source.revision,
        "repo_type": "dataset",
        "files": [
            {
                "path": str(shard.relative_to(source.root)),
                "size": shard.stat().st_size,
                "algorithm": "sha256",
                "expected_hash": "b" * 64,
            }
        ],
    }
    manifest = metadata.with_name("manifest.json")
    manifest.write_bytes(_json_bytes(expected))
    bound_metadata = metadata.with_name("metadata-bound")
    run_metadata_census(source, bound_metadata, resolutions=(64,), spatial=False, authentication_manifest=manifest)
    result = run((source, bound_metadata, output, row, shard))
    assert result["authentication_manifest"]["manifest_sha256"] == hashlib.sha256(_json_bytes(expected)).hexdigest()
    assert result["authentication_manifest"]["content_verified"] is False
    assert result["content_verified"] is False
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest changed"):
        run((source, bound_metadata, output, row, shard))


def test_support_clusters_connect_all_recorded_tiles_and_preserve_unknown():
    from passage.qualification_runner import _row_id, _support_clusters

    def entry(identifier, names):
        return {
            "sample_id": identifier,
            "resolution": 64,
            "split": "train",
            "metadata": {"tile": {"used": [{"name": name} for name in names]}},
        }

    scenes = [entry("a", ["x", "y"]), entry("b", ["y", "z"]), entry("c", ["separate"]), entry("d", [])]
    assignments, support = _support_clusters(scenes)
    assert assignments[_row_id(scenes[0])] == assignments[_row_id(scenes[1])]
    assert assignments[_row_id(scenes[2])] != assignments[_row_id(scenes[0])]
    assert assignments[_row_id(scenes[3])] is None
    assert len(support["clusters"]) == 2 and support["unknown_rows"] == 1
    assert support["independence_certified"] is False


@pytest.mark.parametrize(
    "module_name",
    [
        "passage.utils",
        "passage.pathfinding_utils",
        "passage.qualification",
        "passage_eval.metrics.cost",
        "passage_eval.metrics.reliability",
        "passage_eval.protocol.contracts",
        "passage_eval._typing",
    ],
)
def test_changed_scientific_helper_refuses_cached_replay_and_restored_helper_resumes(
    campaign, monkeypatch, module_name
):
    first = run(campaign)
    protocol_before = (campaign[2] / "protocol.json").read_bytes()
    findings_before = {path.name: path.read_bytes() for path in (campaign[2] / "findings").glob("*.json")}
    module = importlib.import_module(module_name)
    original_file = module.__file__
    helper_file = campaign[2].parent / f"modified-{module_name}.py"
    helper_file.write_bytes(Path(original_file).read_bytes() + b"\n# changed scientific helper source\n")
    monkeypatch.setattr(module, "__file__", str(helper_file))
    with pytest.raises(ValueError, match="protocol|identity"):
        run(campaign)
    assert (campaign[2] / "protocol.json").read_bytes() == protocol_before
    assert {path.name: path.read_bytes() for path in (campaign[2] / "findings").glob("*.json")} == findings_before
    failed = json.loads((campaign[2] / "summary.json").read_text())
    assert failed["status"] == "failed" and failed["protocol_sha256"] == first["protocol_sha256"]
    monkeypatch.setattr(module, "__file__", original_file)
    from passage import qualification_runner

    monkeypatch.setattr(qualification_runner, "_audit_entry", lambda *args: pytest.fail("must resume cached findings"))
    resumed = run(campaign)
    assert resumed["resumed_rows"] == 1 and resumed["counts"]["qualified"] == 6


def test_scientific_helper_changed_during_audit_prevents_complete_claim(campaign, monkeypatch):
    from passage import qualification_runner

    module = importlib.import_module("passage_eval.metrics.cost")
    helper_file = campaign[2].parent / "changed-cost.py"
    helper_file.write_bytes(Path(module.__file__).read_bytes() + b"\n# changed during audit\n")
    original = qualification_runner._audit_entry

    def change_after_audit(*args):
        finding = original(*args)
        monkeypatch.setattr(module, "__file__", str(helper_file))
        return finding

    monkeypatch.setattr(qualification_runner, "_audit_entry", change_after_audit)
    with pytest.raises(ValueError, match="implementation.*changed"):
        run(campaign)
    summary = json.loads((campaign[2] / "summary.json").read_text())
    assert summary["status"] == "failed"
    assert summary["coverage"]["completed_rows"] == 1
