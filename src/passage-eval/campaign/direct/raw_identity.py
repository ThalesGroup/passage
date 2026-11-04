"""Label-free identity of the actual arrays, endpoints and fixed graph used by direct execution."""

from passage_eval.execution.runner import request_identity
from passage_eval.io import pinned

FIELDS = ("graph_id", "cost_model", "cost_weight", "start", "goal", "shape", "elevation_hash", "obstacles_hash")


def raw_identity_from_full(identity):
    if any(field not in identity for field in FIELDS):
        raise ValueError("raw identity fields missing")
    return {field: identity[field] for field in FIELDS}


def raw_identity(request):
    return raw_identity_from_full(request_identity(request))


def raw_digest(request):
    return pinned.canonical_sha256(raw_identity(request))


def task_from_full(sample_id, identity):
    return {"sample_id": sample_id, "raw_input_sha256": pinned.canonical_sha256(raw_identity_from_full(identity))}
