# MIT License
#
# Copyright (c) 2026 THALES

"""What a candidate process may see.

Two policies, both recorded in the run manifest:

``ordinary-request/v1``
    elevation, obstacles, start, goal, cost model/weight, graph spec, budget,
    the released input tensor and an *allow-listed* metadata subset (empty by
    default). The sample identifier is replaced by an opaque per-run token so a
    candidate cannot look up a reference by id.

``decoder-only/v1``
    exactly one predicted field, obstacles, start, goal and budget. Nothing
    else: the builder function does not accept the request, so elevation cannot
    be passed by mistake. The field must satisfy ``decoder-field/v1``: one array,
    shape equal to the grid ``(H, W)``, dtype float32 or float16, all values finite.
    A network output with leading singleton dimensions, such as the ``(1, 1, H, W)``
    batch-1, channel-1 layout of an ONNX model, is the same plane: the supervisor removes
    the leading ones and the decoder always receives ``(H, W)``. Any other layout, such as
    ``(H, W, 1)``, ``(1, 2, H, W)`` or ``(2, H, W)``, is refused.
    Anything else is refused for that request (:class:`FieldContractError`), because a
    predictor could otherwise route several float64 channels, e.g. the elevation it
    received, to the decoder.

    Limit: the contract bounds what the decoder can receive, not what a predictor
    puts in its single field. A predictor can still emit elevation-shaped data as that
    one plane. The guarantee is that the operator's fixed decoder only ever sees one
    network-output plane plus obstacles, start, goal and budget.

Every builder constructs the message field by field; none serialises a request
object wholesale. :func:`assert_policy` re-checks the finished message.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, _METADATA_CONTAINERS, _METADATA_SCALARS
from passage_eval.supervision import POLICY_DECODER_ONLY, POLICY_ORDINARY_REQUEST

ORDINARY_BODY_KEYS = frozenset(
    {
        "policy",
        "token",
        "start",
        "goal",
        "cost_model",
        "cost_weight",
        "graph",
        "budget",
        "metadata",
        "remaining_wall_ms",
    }
)
ORDINARY_ARRAYS = frozenset({"elevation", "obstacles", "input_tensor"})
DECODER_BODY_KEYS = frozenset({"policy", "start", "goal", "budget", "field_kind", "field_count"})
MAX_FIELDS = 8  # what a predictor may put on the wire; the decoder gets one
#: Field contract enforced by the supervisor before anything reaches a decoder.
FIELD_CONTRACT = "decoder-field/v1"
FIELD_DTYPES = ("float16", "float32")
#: Implied by the released input tensor contract (PlanRequest refuses a tensor without it).
TENSOR_IMPLIED_METADATA = ("calibration.max", "calibration.min")


@dataclass(frozen=True)
class WireMessage:
    """A finished, policy-checked candidate-facing message (body + arrays)."""

    policy: str
    body: dict[str, Any]
    arrays: dict[str, np.ndarray]
    metadata_fields: tuple[str, ...] = ()

    @property
    def payload_bytes(self) -> int:
        return int(sum(array.nbytes for array in self.arrays.values()))


def request_token(run_secret: bytes, index: int, *, scope: str = "") -> str:
    """Opaque, per-run, per-pass, per-sample token. Not reversible without ``run_secret``.

    ``scope`` names the pass (series and repeat) so that the same sample gets a different token in every
    timed pass: a candidate that memoises on the token cannot recognise a sample it has seen before.
    """
    if not isinstance(run_secret, bytes) or len(run_secret) < 16:
        raise ValueError("run_secret must be at least 16 random bytes")
    label = f"sample:{int(index)}" if not scope else f"pass:{scope}|sample:{int(index)}"
    digest = hmac.new(run_secret, label.encode(), hashlib.sha256).hexdigest()
    return "req-" + digest[:24]


def _valid_allow_entry(entry: str) -> bool:
    if entry in _METADATA_SCALARS:
        return True
    head, _, tail = entry.partition(".")
    return head in _METADATA_CONTAINERS and tail in _METADATA_CONTAINERS[head]


def normalize_allow_list(entries: Iterable[str]) -> tuple[str, ...]:
    """Validate dotted metadata keys against the ordinary-metadata schema (fail closed)."""
    result = []
    for entry in entries:
        if not isinstance(entry, str) or not _valid_allow_entry(entry):
            raise ValueError(f"metadata allow-list entry {entry!r} is not an ordinary metadata field")
        result.append(entry)
    return tuple(sorted(set(result)))


def _select_metadata(metadata: Mapping[str, Any], entries: Sequence[str]) -> dict[str, Any]:
    selected: dict[str, Any] = {}
    for entry in entries:
        head, dot, tail = entry.partition(".")
        if not dot:
            if head in metadata:
                selected[head] = metadata[head]
        elif isinstance(metadata.get(head), Mapping) and tail in metadata[head]:
            selected.setdefault(head, {})[tail] = metadata[head][tail]
    return selected


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def redact_ordinary(
    request: PlanRequest, *, token: str, remaining_wall_ms: float | None, metadata_allow: Iterable[str] = ()
) -> WireMessage:
    """Build ``ordinary-request/v1`` for one request."""
    entries = set(normalize_allow_list(metadata_allow))
    if request.input_tensor is not None:
        entries |= set(TENSOR_IMPLIED_METADATA)
    entries_t = tuple(sorted(entries))
    metadata = _select_metadata(request.metadata, entries_t)
    for value in _flatten(metadata):
        if isinstance(value, str) and value == request.sample_id:
            raise ValueError("an allow-listed metadata value equals the sample id; refusing to leak it")
    if remaining_wall_ms is not None and (not np.isfinite(remaining_wall_ms) or remaining_wall_ms < 0):
        raise ValueError("remaining_wall_ms must be finite and nonnegative")
    body = {
        "policy": POLICY_ORDINARY_REQUEST,
        "token": token,
        "start": [int(request.start[0]), int(request.start[1])],
        "goal": [int(request.goal[0]), int(request.goal[1])],
        "cost_model": request.cost_model,
        "cost_weight": float(request.cost_weight),
        "graph": {key: _plain(value) for key, value in request.graph.to_dict().items()},
        "budget": {key: _plain(value) for key, value in request.budget.to_dict().items()},
        "metadata": metadata,
        "remaining_wall_ms": None if remaining_wall_ms is None else float(remaining_wall_ms),
    }
    arrays = {"elevation": request.elevation, "obstacles": request.obstacles}
    if request.input_tensor is not None:
        arrays["input_tensor"] = request.input_tensor
    message = WireMessage(POLICY_ORDINARY_REQUEST, body, arrays, entries_t)
    assert_policy(message)
    return message


def _flatten(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _flatten(item)
    else:
        yield value


class FieldContractError(ValueError):
    """The predicted field violates ``decoder-field/v1``; that request is a ``runtime_error``."""


def check_field_contract(fields: Sequence[np.ndarray], grid_shape: Sequence[int]) -> np.ndarray:
    """Return the single admissible field as ``(H, W)`` or raise :class:`FieldContractError` naming the first violation."""
    if len(fields) != 1:
        raise FieldContractError(
            f"{FIELD_CONTRACT}: exactly one field is required, the predictor returned {len(fields)}"
        )
    field_array = np.asarray(fields[0])
    expected = tuple(int(x) for x in grid_shape)
    if len(expected) != 2:
        raise FieldContractError(f"{FIELD_CONTRACT}: the grid must be two-dimensional, got {expected}")
    given = tuple(field_array.shape)
    while field_array.ndim > 2 and field_array.shape[0] == 1:  # (1, 1, H, W) and (1, H, W) are one plane
        field_array = field_array[0]
    if tuple(field_array.shape) != expected:
        raise FieldContractError(f"{FIELD_CONTRACT}: shape {given} differs from the grid {expected}")
    if field_array.dtype.name not in FIELD_DTYPES:
        raise FieldContractError(f"{FIELD_CONTRACT}: dtype {field_array.dtype.name} is not one of {list(FIELD_DTYPES)}")
    if not np.isfinite(field_array).all():
        raise FieldContractError(f"{FIELD_CONTRACT}: the field contains non-finite values")
    return field_array


def redact_decoder(
    fields: Sequence[np.ndarray] | np.ndarray,
    field_kind: str,
    obstacles: np.ndarray,
    start: Sequence[int],
    goal: Sequence[int],
    budget: BudgetSpec,
    *,
    grid_shape: Sequence[int] | None = None,
) -> WireMessage:
    """Build ``decoder-only/v1`` after checking ``decoder-field/v1``.

    The signature is the policy: one field array, obstacles, start, goal, budget. ``grid_shape`` is the
    shape of the request's grid (the contract requires the field to have exactly that shape); it defaults
    to the shape of ``obstacles``, which is the grid by construction.
    ``budget.wall_time_ms`` is the *remaining* wall time at dispatch (one admission
    deadline is shared by both hops); the other limits are unchanged.
    """
    if field_kind not in {"array", "sequence"}:
        raise ValueError("field_kind must be 'array' or 'sequence'")
    field_list = [fields] if field_kind == "array" else list(fields)
    obstacles = np.asarray(obstacles)
    field_array = check_field_contract(field_list, obstacles.shape if grid_shape is None else grid_shape)
    arrays: dict[str, np.ndarray] = {"field_0": field_array, "obstacles": obstacles}
    if not isinstance(budget, BudgetSpec):
        raise ValueError("budget must be a BudgetSpec")
    body = {
        "policy": POLICY_DECODER_ONLY,
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
        "budget": {key: _plain(value) for key, value in budget.to_dict().items()},
        "field_kind": "array",
        "field_count": 1,
    }
    message = WireMessage(POLICY_DECODER_ONLY, body, arrays)
    assert_policy(message)
    return message


def assert_policy(message: WireMessage) -> None:
    """Defence in depth: the finished message contains only policy-listed names."""
    if message.policy == POLICY_ORDINARY_REQUEST:
        if (
            set(message.body) != ORDINARY_BODY_KEYS
            or not set(message.arrays) <= ORDINARY_ARRAYS
            or not {"elevation", "obstacles"} <= set(message.arrays)
        ):
            raise ValueError("ordinary-request/v1 field set violated")
        if message.body["policy"] != POLICY_ORDINARY_REQUEST:
            raise ValueError("policy tag mismatch")
        return
    if message.policy == POLICY_DECODER_ONLY:
        names = set(message.arrays)
        if (
            set(message.body) != DECODER_BODY_KEYS
            or message.body["policy"] != POLICY_DECODER_ONLY
            or message.body["field_kind"] != "array"
            or message.body["field_count"] != 1
            or names != {"obstacles", "field_0"}
        ):
            raise ValueError("decoder-only/v1 field set violated")
        return
    raise ValueError(f"unknown redaction policy {message.policy!r}")
