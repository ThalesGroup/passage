"""What a candidate may see: allow-lists, opaque tokens and the decoder-only structure."""

import inspect
import json

import numpy as np
import pytest

from passage_eval.protocol.contracts import BudgetSpec, PlanRequest
from passage_eval.supervision import POLICY_DECODER_ONLY, POLICY_ORDINARY_REQUEST
from passage_eval.supervision.redaction import (
    DECODER_BODY_KEYS,
    FIELD_CONTRACT,
    FieldContractError,
    ORDINARY_ARRAYS,
    ORDINARY_BODY_KEYS,
    WireMessage,
    assert_policy,
    normalize_allow_list,
    redact_decoder,
    redact_ordinary,
    request_token,
)

SECRET = b"0123456789abcdef0123456789abcdef"
ORDINARY_METADATA = {
    "split": "validation",
    "index": 42,
    "source_id": "tile-17",
    "cluster_id": "c9",
    "calibration": {"min": -100.0, "max": 100.0, "elevation_min": -5.0, "elevation_max": 90.0},
    "geographic": {"latitude": 1.5, "longitude": 2.5, "tile_id": "t", "cluster_id": "c"},
}


def request(sample_id="sample-secret-0042", tensor=False, metadata=None):
    elevation = np.full((4, 4), 12.0)
    obstacles = np.zeros((4, 4), dtype=bool)
    obstacles[1, 1] = True
    metadata = dict(ORDINARY_METADATA if metadata is None else metadata)
    input_tensor = None
    if tensor:
        input_tensor = np.zeros((4, 4, 3), dtype=np.float32)
        input_tensor[:, :, 1] = -1
        input_tensor[0, 0, 1], input_tensor[3, 3, 1] = 0, 1
        input_tensor[:, :, 2] = obstacles
        metadata["calibration"] = {"min": 0.0, "max": 24.0}
        input_tensor[:, :, 0] = 0.5
    return PlanRequest(
        sample_id,
        elevation,
        obstacles,
        (0, 0),
        (3, 3),
        "elevation",
        62.4,
        budget=BudgetSpec(wall_time_ms=250.0, max_expansions=100),
        metadata=metadata,
        input_tensor=input_tensor,
    )


def test_ordinary_request_carries_exactly_the_policy_fields():
    message = redact_ordinary(request(), token="req-abc", remaining_wall_ms=200.0)
    assert message.policy == POLICY_ORDINARY_REQUEST
    assert set(message.body) == ORDINARY_BODY_KEYS and set(message.arrays) == {"elevation", "obstacles"}
    assert message.body["start"] == [0, 0] and message.body["goal"] == [3, 3]
    assert message.body["cost_model"] == "elevation" and message.body["cost_weight"] == 62.4
    assert message.body["graph"]["connectivity"] == 8 and message.body["budget"]["max_expansions"] == 100
    assert message.body["remaining_wall_ms"] == 200.0
    assert message.payload_bytes == 16 * 8 + 16


def test_sample_id_never_reaches_the_wire_and_metadata_is_empty_by_default():
    message = redact_ordinary(request(), token=request_token(SECRET, 3), remaining_wall_ms=None)
    wire = json.dumps(message.body)
    assert "sample-secret-0042" not in wire and "0042" not in wire
    assert message.body["metadata"] == {} and message.metadata_fields == ()
    assert message.body["token"].startswith("req-") and message.body["remaining_wall_ms"] is None


def test_token_is_opaque_stable_per_run_and_changes_between_runs():
    first = [request_token(SECRET, index) for index in range(50)]
    assert len(set(first)) == 50 and first == [request_token(SECRET, index) for index in range(50)]
    other = [request_token(b"f" * 32, index) for index in range(50)]
    assert not set(first) & set(other)
    with pytest.raises(ValueError):
        request_token(b"short", 0)


def test_metadata_allow_list_selects_only_named_ordinary_fields():
    message = redact_ordinary(
        request(),
        token="req-x",
        remaining_wall_ms=1.0,
        metadata_allow=["split", "calibration.min", "geographic.latitude"],
    )
    assert message.body["metadata"] == {
        "split": "validation",
        "calibration": {"min": -100.0},
        "geographic": {"latitude": 1.5},
    }
    assert message.metadata_fields == ("calibration.min", "geographic.latitude", "split")


@pytest.mark.parametrize(
    "entry",
    ["oracle_cost", "calibration.cost_to_go_max", "solvers", "paths.astar", "geographic", "", "split.x", "index "],
)
def test_unknown_allow_list_entries_fail_closed(entry):
    with pytest.raises(ValueError, match="allow-list"):
        normalize_allow_list([entry])


def test_input_tensor_implies_exactly_the_calibration_range():
    message = redact_ordinary(request(tensor=True), token="req-x", remaining_wall_ms=1.0)
    assert set(message.arrays) == ORDINARY_ARRAYS and message.arrays["input_tensor"].dtype == np.float32
    assert message.body["metadata"] == {"calibration": {"max": 24.0, "min": 0.0}}
    assert message.metadata_fields == ("calibration.max", "calibration.min")


def test_a_metadata_value_equal_to_the_sample_id_is_refused():
    with pytest.raises(ValueError, match="sample id"):
        redact_ordinary(
            request(sample_id="tile-17"), token="req-x", remaining_wall_ms=1.0, metadata_allow=["source_id"]
        )


def test_negative_or_nonfinite_remaining_time_is_refused():
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            redact_ordinary(request(), token="req-x", remaining_wall_ms=bad)


def test_decoder_builder_signature_cannot_receive_a_request_or_elevation():
    names = list(inspect.signature(redact_decoder).parameters)
    assert names == ["fields", "field_kind", "obstacles", "start", "goal", "budget", "grid_shape"]


def test_decoder_only_message_has_field_obstacles_start_goal_budget_and_nothing_else():
    field = np.ones((4, 4), dtype=np.float32)
    obstacles = np.zeros((4, 4), dtype=bool)
    message = redact_decoder(field, "array", obstacles, (0, 0), (3, 3), BudgetSpec(wall_time_ms=50.0))
    assert message.policy == POLICY_DECODER_ONLY and set(message.body) == DECODER_BODY_KEYS
    assert set(message.arrays) == {"field_0", "obstacles"}
    assert message.body == {
        "policy": POLICY_DECODER_ONLY,
        "start": [0, 0],
        "goal": [3, 3],
        "budget": BudgetSpec(wall_time_ms=50.0).to_dict(),
        "field_kind": "array",
        "field_count": 1,
    }
    json_text = json.dumps(message.body)
    for forbidden in ("token", "sample", "metadata", "cost", "graph", "elevation"):
        assert forbidden not in json_text


def test_decoder_field_contract_admits_exactly_one_finite_grid_shaped_float32_or_float16_plane():
    obstacles = np.zeros((4, 4), dtype=bool)
    for dtype in (np.float32, np.float16):
        message = redact_decoder(np.ones((4, 4), dtype=dtype), "array", obstacles, (0, 0), (3, 3), BudgetSpec())
        assert message.arrays["field_0"].dtype == dtype and set(message.arrays) == {"field_0", "obstacles"}
    one_in_a_sequence = redact_decoder(
        [np.ones((4, 4), dtype=np.float32)], "sequence", obstacles, (0, 0), (3, 3), BudgetSpec()
    )
    assert one_in_a_sequence.body["field_kind"] == "array" and one_in_a_sequence.body["field_count"] == 1
    assert FIELD_CONTRACT == "decoder-field/v1"


@pytest.mark.parametrize(
    "fields, match",
    [
        ([np.ones((4, 4), np.float32), np.ones((4, 4), np.float64)], "exactly one field"),  # elevation as a 2nd channel
        ([np.ones((4, 4), np.float32), np.zeros((4, 4), np.float16)], "exactly one field"),
        ([np.ones((4, 4), np.float32)] * 9, "exactly one field"),
        ([np.ones((4, 4), np.float64)], "dtype float64"),  # float64 can carry elevation
        ([np.ones((4, 4), np.int32)], "dtype int32"),
        ([np.ones((4, 4), np.bool_)], "dtype bool"),
        ([np.ones((3, 4), np.float32)], "shape"),
        ([np.ones((4, 4, 1), np.float32)], "shape"),
        ([np.ones((1, 2, 4, 4), np.float32)], "shape"),  # two channels
        ([np.ones((2, 1, 4, 4), np.float32)], "shape"),  # batch of two
        ([np.ones((2, 4, 4), np.float32)], "shape"),
        ([np.ones(16, np.float32)], "shape"),
        ([np.full((4, 4), np.nan, np.float32)], "non-finite"),
        ([np.full((4, 4), np.inf, np.float16)], "non-finite"),
        ([np.full((4, 4), -np.inf, np.float32)], "non-finite"),
    ],
)
def test_decoder_field_contract_violations_are_refused(fields, match):
    obstacles = np.zeros((4, 4), dtype=bool)
    with pytest.raises(FieldContractError, match=match):
        redact_decoder(fields, "sequence", obstacles, (0, 0), (3, 3), BudgetSpec())
    assert issubclass(FieldContractError, ValueError)


@pytest.mark.parametrize("shape", [(1, 1, 4, 4), (1, 4, 4)])
def test_a_batch_one_channel_one_network_output_is_the_same_plane_and_reaches_the_decoder_as_h_by_w(shape):
    """The Reg pilot predictor returns (1, 1, H, W): one plane, so it is admitted and flattened to (H, W)."""
    plane = np.arange(16, dtype=np.float32).reshape(shape)
    message = redact_decoder(plane, "array", np.zeros((4, 4), bool), (0, 0), (3, 3), BudgetSpec())
    assert message.arrays["field_0"].shape == (4, 4) and message.arrays["field_0"].dtype == np.float32
    assert np.array_equal(message.arrays["field_0"], np.arange(16, dtype=np.float32).reshape(4, 4))


def test_decoder_field_must_have_the_grid_shape_given_by_the_request_not_the_obstacles_it_sends():
    with pytest.raises(FieldContractError, match="shape"):
        redact_decoder(
            np.ones((4, 4), np.float32),
            "array",
            np.zeros((4, 4), bool),
            (0, 0),
            (3, 3),
            BudgetSpec(),
            grid_shape=(5, 5),
        )


def test_decoder_builder_rejects_unknown_kinds_and_non_budgets():
    obstacles = np.zeros((4, 4), dtype=bool)
    field = np.ones((4, 4), np.float32)
    with pytest.raises(ValueError, match="field_kind"):
        redact_decoder([field], "dict", obstacles, (0, 0), (3, 3), BudgetSpec())
    with pytest.raises(ValueError, match="BudgetSpec"):
        redact_decoder(field, "array", obstacles, (0, 0), (3, 3), {"wall_time_ms": 5})


def test_tokens_differ_between_passes_for_the_same_sample():
    secret = b"s" * 32
    first, second = request_token(secret, 3, scope="warm/0"), request_token(secret, 3, scope="warm/1")
    assert first != second and first != request_token(secret, 3) and first == request_token(secret, 3, scope="warm/0")
    assert len({request_token(secret, i, scope="cold/0") for i in range(100)}) == 100


def test_assert_policy_catches_tampered_messages():
    good = redact_ordinary(request(), token="req-x", remaining_wall_ms=1.0)
    extra_array = WireMessage(good.policy, good.body, dict(good.arrays, oracle_path=np.zeros(3)))
    extra_key = WireMessage(good.policy, dict(good.body, sample_id="leak"), good.arrays)
    for tampered in (extra_array, extra_key):
        with pytest.raises(ValueError):
            assert_policy(tampered)
    decoder = redact_decoder(
        np.ones((4, 4), dtype=np.float32), "array", np.zeros((4, 4), dtype=bool), (0, 0), (3, 3), BudgetSpec()
    )
    with_elevation = WireMessage(decoder.policy, decoder.body, dict(decoder.arrays, elevation=np.zeros((4, 4))))
    with_token = WireMessage(decoder.policy, dict(decoder.body, token="req-x"), decoder.arrays)
    mismatched = WireMessage(decoder.policy, dict(decoder.body, field_count=2), decoder.arrays)
    second_field = WireMessage(decoder.policy, decoder.body, dict(decoder.arrays, field_1=np.zeros((4, 4), np.float32)))
    for tampered in (with_elevation, with_token, mismatched, second_field):
        with pytest.raises(ValueError):
            assert_policy(tampered)
    with pytest.raises(ValueError, match="unknown redaction policy"):
        assert_policy(WireMessage("other/v1", {}, {}))
