"""Scientific report producers use released rows and declared measured records."""

import json

import numpy as np

from passage.notebook_reports import dataset_report, evaluation_report, route_metrics_report
from passage.utils import encode_tensor_blob


def test_portable_qualification_bundle_resolves_from_publication_ancestor(tmp_path, monkeypatch):
    from passage.notebook_reports import resolve_evidence_path

    checkout = tmp_path / "checkout"
    staging = checkout / "outputs" / "publication"
    bundle = staging / "notebook-artifacts" / "qualification-evidence"
    bundle.mkdir(parents=True)
    working = staging / "256x256"
    working.mkdir()
    monkeypatch.chdir(working)
    assert resolve_evidence_path("notebook-artifacts/qualification-evidence", checkout) == bundle
    assert resolve_evidence_path(None, checkout) is None
    assert resolve_evidence_path(bundle, checkout) == bundle


class MetadataSource:
    def __init__(self):
        self.calls = []

    def iter_rows(self, resolution, split, **kwargs):
        self.calls.append((split, kwargs))
        for index in range(2):
            yield {
                "id": f"{split}-{index}",
                "metadata": {
                    "resolution": resolution,
                    "split": split,
                    "tile": {"source": {"name": "N000E000"}},
                    "calibration": {"min": -100, "max": 100},
                    "obstacles": {"ratio_actual": 0.25},
                    "crop": {"min_elevation": 10, "max_elevation": 30},
                },
            }


def test_census_reads_all_core_splits_without_tensor_or_limit():
    source = MetadataSource()
    report = dataset_report(source, 64)
    assert report["population"] == "full_core"
    assert report["total_rows"] == 8
    assert {r["split"] for r in report["splits"]} == {"train", "calibration", "validation", "test"}
    assert all(
        k["columns"] == ["id", "metadata"] and not k["samples"] and k.get("limit") is None for _, k in source.calls
    )
    assert all(r["qualification_status"] == "not_run" for r in report["splits"])


def test_same_scene_cost_metrics_include_zero_divergence_and_reference_identity():
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[..., 1] = -1
    tensor[0, 0, 1], tensor[2, 2, 1] = 0, 1
    row = {
        "id": "released-id",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_elevation_free": [[0, 0], [1, 1], [2, 2]],
        "path_astar_elevation_obstacles": [[0, 0], [1, 1], [2, 2]],
        "metadata": {
            "resolution": 3,
            "split": "validation",
            "calibration": {"min": 0, "max": 100},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 1, "diagonal_movement": "always"}},
            "paths": {"astar_elevation": {v: {"cost": 60 * np.sqrt(2)} for v in ("free", "obstacles")}},
        },
    }
    report = route_metrics_report(row, 3, "validation")
    assert len(report["routes"]) == 2
    assert report["counts"]["routes_requested"] == 2
    assert all(r["legal"] and r["cost_agreement"] for r in report["routes"])
    assert all(r["assd"] == 0 and r["hausdorff_p95_pixels"] == 0 for r in report["comparisons"])
    assert report["reference_status"] == "stored_route_not_exact_revalidated"
    assert report["sample_id"] == "released-id"


def test_missing_learned_records_are_visible_not_run(tmp_path):
    report = evaluation_report(tmp_path, 256)
    assert report["status"] == "not_run"
    assert report["records"] == []
    assert report["blocked_gate"] == "S2-S4"


def test_aggregate_without_receipt_is_never_admitted(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"resolution": 256, "metrics": {"success_rate": 1}}))
    report = evaluation_report(tmp_path, 256)
    assert report["status"] == "not_run"
    assert report["records"] == []
    assert report["historical_aggregates"][0]["support_status"] == "unverified"


def test_explicit_bounded_census_cannot_claim_full_population():
    source = MetadataSource()
    report = dataset_report(source, 64, census_limit=1)
    assert report["population"] == "bounded_core_metadata"
    assert report["census_limit_per_split"] == 1
    assert all(k["limit"] == 1 for _, k in source.calls)


def test_real_runner_receipt_requires_hash_binding_and_qualified_scoring(tmp_path):
    from passage_eval.execution.artifacts import RunArtifact, route_hash, sha256_file, write_run_artifact
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.execution.runner import request_identity

    request = PlanRequest("released-id", np.zeros((2, 2)), np.zeros((2, 2), dtype=bool), (0, 0), (1, 1), "elevation", 1)
    path = np.array([[0, 0], [1, 1]], dtype=np.int32)
    manifest = {
        "schema": "passage-complete-request-v1",
        "roster": ["released-id"],
        "repeats": 1,
        "quality_repeat_index": 0,
        "series": "warm",
        "request_specs": {"released-id": request_identity(request)},
        "request_metadata": {"released-id": {}},
        "dataset": {"resolution": 256, "content_hash": "bound-input"},
        "procedure_id": "bound-procedure",
        "procedure": {"family": "seg", "checkpoint_hashes": {"weights": "sha256"}},
        "admission": {"qualified_reference": True, "ordinary_input_contract": "PlanRequest-v1"},
        "reference_identity": "bound-reference",
    }
    record = {
        "sample_id": "released-id",
        "repeat_index": 0,
        "series": "warm",
        "status": "ok",
        "route_hash": route_hash(path),
        "safe_completion": True,
        "total_wall_ms": 1,
        "failed_attempt_ms": 0,
        "drain_ms": 0,
        "connected_path": True,
        "legal_path": True,
        "obstacle_violation": False,
        "late_completion": False,
        "fallback": False,
        "deadline_ms": None,
        "stage": "primary",
        "stage_work": {},
        "work": {},
        "resource_peaks": {},
        "stage_spans": {},
    }
    write_run_artifact(RunArtifact(manifest, [record], [path]), tmp_path)
    np.savez(tmp_path / "per_sample.npz", sample_id=np.array(["released-id"]))
    (tmp_path / "metrics.json").write_text('{"success_rate": 1}')
    receipt = {
        "schema": "passage-offline-score-v1",
        "qualified_reference": True,
        "reference_policy": "qualified_exact",
        "scored_records": 1,
        "run_manifest_sha256": sha256_file(tmp_path / "run_manifest.json"),
        "metrics_sha256": sha256_file(tmp_path / "metrics.json"),
        "per_sample_sha256": sha256_file(tmp_path / "per_sample.npz"),
    }
    (tmp_path / "score_receipt.json").write_text(json.dumps(receipt))
    report = evaluation_report(tmp_path, 256)
    assert report["status"] == "admitted"
    assert report["task_ids"] == ["released-id"]
    assert report["records"][0]["status"] == "ok"
    assert report["learned_status"] == "not_run"
    assert report["blocked_gate"] == "S2-S4"
    (tmp_path / "metrics.json").write_text('{"success_rate": 0}')
    import pytest

    with pytest.raises(ValueError, match="hash mismatch"):
        evaluation_report(tmp_path, 256)


def test_global_split_inventory_uses_declared_four_split_counts():
    from passage.notebook_reports import split_inventory_report

    class Source:
        def split_counts(self, resolution, samples=False):
            assert samples is False
            return {"train": resolution, "calibration": 2, "validation": 4, "test": 4}

    result = split_inventory_report(Source(), [64, 128])
    assert result["records"][0]["train"] == 64
    assert result["records"][1]["total"] == 138
    assert result["population"] == "full_core_footer_counts"


def test_pinned_remote_source_can_use_installed_package_without_checkout(monkeypatch):
    import passage.notebook_reports as reports

    def missing_checkout():
        raise FileNotFoundError("No checkout")

    monkeypatch.setattr(reports, "find_project_root", missing_checkout)
    source = reports.make_source(None, dataset_repo="declared/release", dataset_revision="a" * 40)
    assert source.root is None
    assert source.revision == "a" * 40


def test_missing_objectives_cannot_admit_six_variant_reference_cohort(monkeypatch):
    from passage.notebook_reports import classical_diagnostics_report
    import passage.audit

    def partial(source, **kwargs):
        assert kwargs["cost_models"] == ("elevation", "slope", "energy")
        return {
            "rows": [{"sample_id": "released-id"}],
            "records": [{"qualified": True}] * 2,
            "coverage": {"audited_rows": 1},
            "counts": {"qualified": 2, "invalid_metadata": 0},
        }

    monkeypatch.setattr(passage.audit, "audit_source", partial)
    report = classical_diagnostics_report(MetadataSource(), 64, budget=1)
    assert report["qualified_reference_status"] == "not_run"
    assert report["expected_task_variants"] == 6


def test_successor_terminal_route_status_and_null_cost_are_retained():
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[..., 1] = -1
    tensor[0, 0, 1], tensor[2, 2, 1] = 0, 1
    row = {
        "id": "release-id",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_energy_free": [],
        "path_astar_energy_obstacles": [],
        "metadata": {
            "resolution": 3,
            "split": "validation",
            "calibration": {"min": 0, "max": 100},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_energy": {"cost_model": "energy", "weight": 1}},
            "paths": {
                "astar_energy": {
                    "free": {"status": "failed", "cost": None},
                    "obstacles": {"status": "infeasible", "cost": None},
                }
            },
        },
    }
    report = route_metrics_report(row, 3, "validation")
    assert [record["stored_status"] for record in report["routes"]] == ["failed", "infeasible"]
    assert all(record["stored_cost"] is None and not record["cost_agreement"] for record in report["routes"])
    assert report["counts"]["routes_requested"] == 2
    assert report["counts"]["stored_status_counts"] == {"failed": 1, "infeasible": 1}


def test_successor_census_includes_rejected_scene_denominators(tmp_path):
    source = MetadataSource()
    source.root = tmp_path
    receipts = tmp_path / "release-receipts"
    receipts.mkdir()
    (receipts / "batch-64-0000000000-0000000002.json").write_text(
        json.dumps(
            {
                "schema_version": "passage-export-batch-v1",
                "resolution": 64,
                "requested_scenes": 3,
                "requested_variants": 18,
                "scene_status_counts": {"train:complete": 1, "train:rejected": 2},
                "variant_status_counts": {"train:ok": 6, "train:failed": 12},
            }
        )
    )
    report = dataset_report(source, 64)
    assert report["generation_outcomes"]["requested_scenes"] == 3
    assert report["generation_outcomes"]["scene_status_counts"]["train:rejected"] == 2
    assert report["generation_outcomes"]["requested_variants"] == 18
    assert report["generation_outcomes"]["content_verified"] is False
