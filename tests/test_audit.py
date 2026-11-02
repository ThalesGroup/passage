"""Independent optima, qualification failures and genuine footprint evidence."""

import copy
import importlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage.utils import encode_tensor_blob


def audit():
    return importlib.import_module("passage.audit")


@pytest.fixture
def row():
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0] = 0.5
    tensor[:, :, 1] = -1
    tensor[0, 0, 1] = 0
    tensor[2, 2, 1] = 1
    cost = 60 * np.sqrt(2)
    return {
        "id": "42",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_elevation_free": [[0, 0], [1, 1], [2, 2]],
        "path_astar_elevation_obstacles": [[0, 0], [1, 1], [2, 2]],
        "metadata": {
            "resolution": 3,
            "split": "validation",
            "calibration": {"min": 0, "max": 100},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
            "paths": {"astar_elevation": {"free": {"cost": cost}, "obstacles": {"cost": cost}}},
        },
    }


def test_exact_solver_flat_directed_models_and_budget_are_separate():
    module = audit()
    flat = np.full((3, 3), 100.0)
    obs = np.zeros((3, 3), dtype=bool)
    for model in ("elevation", "energy", "slope"):
        ref = module.exact_reference(flat, obs, (0, 0), (2, 2), cost_model=model, weight=10, max_expansions=20)
        assert ref["status"] == "exact"
        assert ref["cost"] == pytest.approx(60 * np.sqrt(2))
        assert ref["path"][0] == [0, 0] and ref["path"][-1] == [2, 2]
    censored = module.exact_reference(flat, obs, (0, 0), (2, 2), cost_model="energy", weight=10, max_expansions=0)
    assert censored["status"] == "budget_exhausted"
    assert censored["cost"] is None and censored["path"] == []


def test_solver_preserves_positive_directed_cost_and_destination_only_corners():
    module = audit()
    elevation = np.array([[0.0, 2.0], [0.0, 0.0]])
    obs = np.array([[False, False], [True, True]])
    up = module.exact_reference(elevation, obs, (0, 0), (0, 1), cost_model="energy", weight=5, max_expansions=10)
    down = module.exact_reference(elevation, obs, (0, 1), (0, 0), cost_model="energy", weight=5, max_expansions=10)
    assert up["cost"] == 40 and down["cost"] == 30
    corner_obs = np.array([[False, True], [True, False]])
    ref = module.exact_reference(
        np.zeros((2, 2)), corner_obs, (0, 0), (1, 1), cost_model="elevation", weight=10, max_expansions=10
    )
    assert ref["status"] == "exact"
    assert ref["cost"] == pytest.approx(30 * np.sqrt(2))


def test_solver_no_path_is_not_censored():
    ref = audit().exact_reference(
        np.zeros((3, 3)), np.ones((3, 3), dtype=bool), (0, 0), (2, 2), cost_model="energy", weight=10, max_expansions=20
    )
    assert ref["status"] == "no_path"
    assert ref["cost"] is None


def test_audit_stored_cost_agreement_does_not_prove_optimality(row):
    route = [[0, 0], [0, 1], [1, 2], [2, 2]]
    row["path_astar_elevation_obstacles"] = route
    row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"] = 60 + 30 * np.sqrt(2)
    result = audit().audit_row(row, resolution=3, split="validation", task_variants=("obstacles",))
    record = result["records"][0]
    assert record["legal"] and record["stored_cost_agreement"]
    assert record["reference_status"] == "exact"
    assert record["optimality"] is False and record["qualified"] is False
    assert "stored_path_suboptimal" in record["qualification_reasons"]
    assert record["optimal_cost"] == pytest.approx(60 * np.sqrt(2))


def test_illegal_and_corrupt_stored_paths_remain_findings(row):
    row["path_astar_elevation_obstacles"] = [[0, 0], [0.5, 1], [2, 2]]
    result = audit().audit_row(row, resolution=3, split="validation", task_variants=("obstacles",))
    assert not result["records"][0]["legal"]
    assert not result["records"][0]["qualified"]
    row["tensor"] = b"corrupt"
    corrupt = audit().audit_row(row, resolution=3, split="validation")
    assert corrupt["status"] == "invalid_metadata"
    assert "42" in corrupt["error"]


def test_censored_reference_never_qualifies_and_reports_denominator(row, tmp_path):
    directory = tmp_path / "3x3" / "validation"
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), directory / "data.parquet")
    from passage.dataset import DatasetSource

    receipt = audit().audit_source(
        DatasetSource(tmp_path, revision="pinned"),
        resolution=3,
        split="validation",
        limit=1,
        max_expansions=0,
        task_variants=("obstacles",),
    )
    assert receipt["counts"]["reference_censored"] == 1
    assert receipt["counts"]["qualified"] == 0
    assert receipt["coverage"]["audited_rows"] == 1
    assert receipt["source"]["revision"] == "pinned"
    assert receipt["source"]["content_verified"] is False
    assert audit().reference_store(receipt, "elevation", "obstacles") == {}


def footprint_row(identifier, split, west, east, south=0, north=1, source="same"):
    return {
        "id": identifier,
        "metadata": {
            "split": split,
            "crop": {"top_left": {"lat": north, "lon": west}, "bottom_right": {"lat": south, "lon": east}},
            "tile": {"source": {"name": source}, "overlapping": [{"name": source}]},
        },
    }


def test_geographic_overlap_uses_actual_crop_not_center_tile():
    rows = [footprint_row("a", "train", 0, 1), footprint_row("b", "test", 2, 3)]
    receipt = audit().footprint_overlaps(rows)
    assert receipt["cross_split_geographic_overlaps"] == []
    assert len(receipt["cross_split_shared_source_tiles"]) == 1
    assert receipt["source_cell_coverage"]["unknown"] == 2
    rows.append(footprint_row("c", "test", 0.5, 1.5, source="other"))
    receipt = audit().footprint_overlaps(rows)
    assert receipt["cross_split_geographic_overlaps"][0]["task_ids"] == ["a", "c"]


def test_dateline_and_boundary_touch_are_reported_separately():
    rows = [
        footprint_row("a", "train", 179, -179),
        footprint_row("b", "test", -180, -178),
        footprint_row("c", "validation", -178, -177),
    ]
    receipt = audit().footprint_overlaps(rows)
    assert ["a", "b"] in [pair["task_ids"] for pair in receipt["cross_split_geographic_overlaps"]]
    assert ["b", "c"] in [pair["task_ids"] for pair in receipt["cross_split_boundary_intersections"]]
    assert audit().geographic_footprint({"id": "unknown", "metadata": {}})["status"] == "unknown"


def test_valid_row_reference_store_preserves_source_label_and_exact_label(row):
    result = audit().audit_row(row, resolution=3, split="validation", task_variants=("obstacles",))
    record = result["records"][0]
    assert record["qualified"]
    assert record["stored_cost"] == row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"]
    assert record["optimal_cost"] == pytest.approx(record["stored_cost"])
    store = audit().reference_store({"records": result["records"]}, "elevation", "obstacles")
    assert store["42"]["reference_id"] and store["42"]["graph_id"]


def test_scene_identity_binds_ordinary_inputs_and_matches_scene_adapter(row):
    from passage_eval.io.dataset import scene_from_row

    scene = scene_from_row(row, resolution=3, split="validation", cost_model="elevation")
    identity = audit().scene_identity(scene)
    record = audit().qualified_reference_for_scene(scene, max_expansions=20)
    assert identity == record["scene_id"]
    scene.elevation[0, 1] += 1
    assert audit().scene_identity(scene) != identity


@pytest.mark.parametrize("bad_metadata", [None, "corrupt", 4, []])
def test_corrupt_metadata_remains_explicit_finding(row, bad_metadata):
    row["metadata"] = bad_metadata
    result = audit().audit_row(row, resolution=3, split="validation")
    assert result["status"] == "invalid_metadata"
    assert result["footprint"]["status"] == "unknown"


def test_missing_fields_never_qualify(row):
    del row["path_astar_elevation_obstacles"]
    del row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"]
    record = audit().audit_row(row, resolution=3, split="validation", task_variants=("obstacles",))["records"][0]
    assert not record["legal"] and not record["qualified"]
    assert "stored_cost_missing_or_invalid" in record["qualification_reasons"]


def test_explicit_source_cell_overlap_is_independent_of_geographic_unknown():
    rows = [
        {
            "id": "a",
            "metadata": {"split": "train", "source_cell_windows": [{"source_id": "raster", "bounds": [0, 0, 10, 10]}]},
        },
        {
            "id": "b",
            "metadata": {"split": "test", "source_cell_windows": [{"source_id": "raster", "bounds": [5, 5, 15, 15]}]},
        },
    ]
    receipt = audit().footprint_overlaps(rows)
    assert len(receipt["cross_split_source_cell_overlaps"]) == 1
    assert receipt["cross_split_geographic_overlaps"] == []


def test_bad_ids_remain_findings_without_entering_task_roster(row, tmp_path):
    directory = tmp_path / "3x3" / "validation"
    directory.mkdir(parents=True)
    missing = copy.deepcopy(row)
    missing["id"] = None
    pq.write_table(pa.Table.from_pylist([row, missing]), directory / "data.parquet")
    from passage.dataset import DatasetSource

    receipt = audit().audit_source(
        DatasetSource(tmp_path), resolution=3, split="validation", limit=2, task_variants=("obstacles",)
    )
    assert receipt["task_ids"] == ["42"]
    assert receipt["coverage"]["audited_rows"] == 2
    assert receipt["counts"]["invalid_metadata"] == 1


def test_duplicate_solver_objective_invalidates_all_matching_labels(row):
    row["metadata"]["solvers"]["duplicate"] = copy.deepcopy(row["metadata"]["solvers"]["astar_elevation"])
    result = audit().audit_row(row, resolution=3, split="validation", task_variants=("obstacles",))
    assert all(not record["qualified"] for record in result["records"])
    assert result["status"] == "invalid_metadata"


def test_missing_requested_solver_emits_terminal_findings(row):
    result = audit().audit_row(
        row, resolution=3, split="validation", cost_models=["energy"], task_variants=("free", "obstacles")
    )
    assert result["status"] == "invalid_metadata"
    assert len(result["records"]) == 2
    assert all(record["reference_status"] == "not_run" and not record["qualified"] for record in result["records"])


def test_footprint_normalizes_split_aliases_and_seam_only_touch():
    rows = [footprint_row("a", "val", 0, 1), footprint_row("b", "validation", 0, 1)]
    assert audit().footprint_overlaps(rows)["cross_split_geographic_overlaps"] == []
    seam = [footprint_row("a", "train", 179, 180), footprint_row("b", "test", -180, -179)]
    assert len(audit().footprint_overlaps(seam)["cross_split_boundary_intersections"]) == 1


def test_audit_filter_generators_have_same_scope_for_each_row(row, tmp_path):
    directory = tmp_path / "3x3" / "validation"
    directory.mkdir(parents=True)
    other = copy.deepcopy(row)
    other["id"] = "43"
    pq.write_table(pa.Table.from_pylist([row, other]), directory / "data.parquet")
    from passage.dataset import DatasetSource

    receipt = audit().audit_source(
        DatasetSource(tmp_path),
        resolution=3,
        split="validation",
        limit=2,
        cost_models=(value for value in ["elevation"]),
        task_variants=(value for value in ["obstacles"]),
    )
    assert len(receipt["records"]) == 2
    assert receipt["counts"]["qualified"] == 2
    assert receipt["qualified_task_ids"] == ["42", "43"]


def test_fresh_reference_admission_requires_named_cohort_and_retains_label_defect(row):
    from passage_eval.io.dataset import scene_from_row

    row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"] += 1
    scene = scene_from_row(row, resolution=3, split="validation", cost_model="elevation")
    label = audit().qualified_reference_for_scene(scene, max_expansions=20)
    assert not label["qualified"]
    fresh = audit().independently_qualified_reference_for_scene(
        scene, cohort_id="diagnostic-cohort-v1", max_expansions=20
    )
    assert fresh["qualified"]
    assert fresh["qualification_policy"] == "independent_analytic_reference_v1"
    assert fresh["cohort_id"] == "diagnostic-cohort-v1"
    assert fresh["source_label_qualified"] is False
    assert "stored_cost_disagreement" in fresh["source_label_qualification_reasons"]
    assert fresh["stored_cost"] == label["stored_cost"]
    censored = audit().independently_qualified_reference_for_scene(
        scene, cohort_id="diagnostic-cohort-v1", max_expansions=0
    )
    assert not censored["qualified"]
    with pytest.raises(ValueError, match="cohort"):
        audit().independently_qualified_reference_for_scene(scene, cohort_id="")


def test_invalid_source_cell_identity_is_explicit():
    row = {
        "id": "a",
        "metadata": {
            "split": "test",
            "source_cell_windows": [{"source_id": {"invalid": "container"}, "bounds": [0, 0, 2, 2]}],
        },
    }
    assert audit().geographic_footprint(row)["source_cell_status"] == "invalid"


def test_audit_receipt_includes_four_split_header_census(row, tmp_path):
    directory = tmp_path / "3x3" / "validation"
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([row]), directory / "data.parquet")
    from passage.dataset import DatasetSource

    receipt = audit().audit_source(
        DatasetSource(tmp_path), resolution=3, split="validation", limit=1, task_variants=("obstacles",)
    )
    assert set(receipt["header_census"]) == {"train", "validation", "calibration", "test"}
    assert receipt["header_census"]["validation"]["count"] == 1
    assert receipt["header_census"]["train"]["status"] == "missing"
    assert receipt["coverage"]["population_total"] == 1
    assert receipt["coverage"]["complete_population"] is False


def test_stored_label_identity_binds_route_cost_and_variant(row):
    module = audit()
    result = module.audit_row(row, resolution=3, split="validation")
    records = {record["task_variant"]: record for record in result["records"]}
    free = records["free"]
    assert free["stored_label_id"] == module.stored_label_identity(
        free["stored_path"], free["stored_cost"], task_variant="free"
    )
    assert free["stored_label_id"] != records["obstacles"]["stored_label_id"]
    assert free["stored_label_id"] != module.stored_label_identity([], free["stored_cost"], task_variant="free")
    assert free["stored_label_id"] != module.stored_label_identity(
        free["stored_path"], free["stored_cost"] + 1, task_variant="free"
    )
