"""Ordinary-input and raw-coordinate contracts for complete requests."""

from dataclasses import replace

import numpy as np
import pytest

from passage_eval.protocol import contracts
from passage_eval.protocol.planner import candidate_scene


def test_request_never_exposes_oracle_metadata_and_copies_inputs(metre_scene):
    scene = replace(
        metre_scene,
        metadata={
            "split": "validation",
            "resolution": 3,
            "calibration": {"min": -1, "max": 1, "oracle_cost": 77},
            "reachable": True,
            "reference_cost": 77,
            "cost_to_go_max": 77,
            "nested": {"oracle_path": [[0, 0]]},
            "paths": {"cost": 77},
        },
    )
    request = contracts.PlanRequest.from_scene(scene)
    assert request.metadata == {"split": "validation", "resolution": 3, "calibration": {"min": -1, "max": 1}}
    assert not hasattr(request, "oracle_path")
    assert not hasattr(request, "oracle_cost")
    request.elevation[0, 0] = 123
    assert scene.elevation[0, 0] == 0
    assert candidate_scene(scene).metadata == request.metadata


def test_contract_defaults_match_released_graph_and_validate_budget():
    graph = contracts.GraphSpec()
    assert graph.connectivity == 8
    assert graph.diagonal_policy == "destination_only"
    assert graph.pixel_size_m == 30
    assert graph.to_dict()["coordinate_order"] == "row_col"
    for kwargs in ({"wall_time_ms": -1}, {"wall_time_ms": True}, {"max_expansions": 1.5}, {"max_live_states": 0}):
        with pytest.raises(ValueError):
            contracts.BudgetSpec(**kwargs)


def test_result_preserves_fractional_and_nonfinite_coordinates():
    raw = np.array([[0.0, 0.0], [1.25, np.nan], [2.0, 2.0]])
    result = contracts.PlanResult(raw)
    np.testing.assert_equal(result.path, raw)
    assert result.status == "ok"
    with pytest.raises(ValueError, match="status"):
        contracts.PlanResult([], status="magic")


def test_ordinary_source_support_identifiers_survive_without_oracle_fields(metre_scene):
    scene = replace(
        metre_scene,
        metadata={
            "source_footprint_id": "joined-footprints",
            "source_tile_id": "tile",
            "reachable": True,
            "oracle_cost": 12,
            "geographic": {"cluster_id": "wide", "reference_cost": 12},
        },
    )
    assert contracts.PlanRequest.from_scene(scene).metadata == {
        "source_footprint_id": "joined-footprints",
        "source_tile_id": "tile",
        "geographic": {"cluster_id": "wide"},
    }


@pytest.mark.parametrize(
    "calibration", [{"min": 0, "max": '{"oracle_cost": 123}'}, {"min": 0, "max": float("nan")}, {"min": True, "max": 1}]
)
def test_calibration_cannot_smuggle_targets_as_strings_or_nonfinite_numbers(metre_scene, calibration):
    with pytest.raises(ValueError, match="calibration"):
        contracts.PlanRequest.from_scene(replace(metre_scene, metadata={"calibration": calibration}))


@pytest.mark.parametrize("corruption", ["path_markers", "obstacles", "terrain"])
def test_public_tensor_boundary_rejects_hidden_route_or_inconsistent_inputs(metre_scene, corruption):
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 1] = -1
    tensor[0, 0, 1], tensor[2, 2, 1] = 0, 1
    if corruption == "path_markers":
        tensor[1, 1, 1] = 1
    elif corruption == "obstacles":
        tensor[1, 1, 2] = 1
    else:
        tensor[1, 1, 0] = 0.25
    scene = replace(metre_scene, input_tensor=tensor, metadata={"calibration": {"min": 0, "max": 1}})
    with pytest.raises(ValueError, match="tensor|channel"):
        contracts.PlanRequest.from_scene(scene)
    with pytest.raises(ValueError, match="tensor|channel"):
        candidate_scene(scene)
