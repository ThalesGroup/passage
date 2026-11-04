"""Native-adapter identity records for the supported edge targets."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from passage_eval.campaign.edge import EDGE_TARGETS, FINAL_DECODERS_SHA256, NATIVE_ADAPTER_SCHEMA, EdgeRelayError
from passage_eval.io import pinned

PROFILES = {"thor": "TensorRT", "orin": "TensorRT", "tiger": "OpenVINO"}
_KEYS = {
    "schema",
    "target",
    "backend",
    "precision",
    "artifact_kind",
    "engine_or_model_sha256",
    "checkpoint_sha256",
    "procedure",
    "package_sha256",
    "plan_sha256",
    "final_decoders_sha256",
    "seed",
}


def frozen_candidate_identity(procedure: str, *, seed: int = 2026, replica_catalog=None, replica_catalog_sha256=None) -> dict[str, Any]:
    """Read the exact frozen 256 candidate from the committed local harness."""
    if procedure not in ("seg", "reg"):
        raise EdgeRelayError("only the frozen Seg and Reg candidates have an edge identity")
    if seed != 2026:
        from passage_eval.protocol.replicas import load_replica_catalog, replica_identity
        if replica_catalog is None or not pinned.is_hex64(replica_catalog_sha256):
            raise EdgeRelayError("replica identity requires its pinned catalogue")
        return replica_identity(load_replica_catalog(replica_catalog,replica_catalog_sha256),procedure,seed)
    from passage_eval import cli
    from passage_eval.protocol.admission import committed_final_candidates

    try:
        frozen, _ = committed_final_candidates(cli._checkout_commit())
        checkpoint = frozen["candidates"][procedure]["checkpoint_256"]
    except (KeyError, TypeError, ValueError) as exc:
        raise EdgeRelayError(f"frozen {procedure} candidate is unavailable: {exc}") from exc
    if not pinned.is_hex64(checkpoint.get("sha256")) or checkpoint.get("seed") != 2026:
        raise EdgeRelayError(f"frozen {procedure} candidate identity differs")
    return {"procedure": procedure, "checkpoint_sha256": checkpoint["sha256"], "seed": 2026}


def validate_native_adapter(receipt: Any, *, expected_seed: int = 2026) -> dict[str, Any]:
    """Validate the receipt that a target native adapter must return with a run."""
    if not isinstance(receipt, Mapping) or set(receipt) != _KEYS:
        raise EdgeRelayError("native adapter receipt keys differ")
    checked = dict(receipt)
    if checked["schema"] != NATIVE_ADAPTER_SCHEMA:
        raise EdgeRelayError("native adapter receipt schema differs")
    if checked["target"] not in EDGE_TARGETS or checked["backend"] != PROFILES[checked["target"]]:
        raise EdgeRelayError("native adapter backend does not match target")
    if checked["precision"] != "fp16":
        raise EdgeRelayError("native adapter precision must be fp16")
    expected_kind = "engine" if checked["backend"] == "TensorRT" else "model"
    if checked["artifact_kind"] != expected_kind:
        raise EdgeRelayError("native adapter engine/model kind does not match backend")
    if not isinstance(checked["procedure"], str) or not checked["procedure"]:
        raise EdgeRelayError("native adapter procedure is required")
    for field in ("engine_or_model_sha256", "checkpoint_sha256", "package_sha256", "plan_sha256"):
        if not pinned.is_hex64(checked[field]):
            raise EdgeRelayError(f"native adapter {field} must be a SHA-256")
    if checked["final_decoders_sha256"] != FINAL_DECODERS_SHA256:
        raise EdgeRelayError("native adapter decoder SHA-256 differs from frozen final_decoders.py")
    if type(expected_seed) is not int or expected_seed not in (1234,2026,3407) or type(checked["seed"]) is not int or checked["seed"] != expected_seed:
        raise EdgeRelayError("native adapter seed must equal its pinned expected seed")
    return checked


__all__ = ["PROFILES", "frozen_candidate_identity", "validate_native_adapter"]
