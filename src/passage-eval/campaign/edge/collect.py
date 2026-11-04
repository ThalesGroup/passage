"""Host-side collection: verify target provenance before invoking a scorer."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from passage_eval.campaign.edge import TARGET_RESULT_SCHEMA, EdgeRelayError
from passage_eval.campaign.edge.package import InputPackage, PackageView, verify_package
from passage_eval.campaign.edge.target import validate_adapter_output
from passage_eval.campaign.edge.targets import validate_native_adapter
from passage_eval.io import pinned


def _package_view(package: InputPackage | PackageView) -> PackageView:
    return verify_package(package.root, package.sha256)


def verify_collected_run(root: str | Path, *, package: InputPackage | PackageView) -> dict[str, Any]:
    """Return an unscored native payload only if its complete identity chain is intact."""
    view = _package_view(package)
    root = Path(root)
    try:
        raw = pinned.read_bounded(root / "target-result.json")
        expected = pinned.read_bounded(root / "target-result.sha256").decode("ascii").strip()
    except (OSError, UnicodeDecodeError, pinned.PinnedFileError) as exc:
        raise EdgeRelayError(f"target result digest is unavailable: {exc}") from exc
    if not pinned.is_hex64(expected) or pinned.sha256_bytes(raw) != expected:
        raise EdgeRelayError("target result digest differs")
    try:
        result = pinned.strict_json_loads(raw, name="target-result.json")
    except pinned.PinnedFileError as exc:
        raise EdgeRelayError(f"target result is invalid: {exc}") from exc
    required = {"schema", "package_sha256", "plan_sha256", "procedure", "receipt", "adapter", "outputs"}
    if not isinstance(result, Mapping) or set(result) != required or result["schema"] != TARGET_RESULT_SCHEMA:
        raise EdgeRelayError("target result schema differs")
    receipt = validate_native_adapter(result["receipt"],expected_seed=(view.manifest.get("candidate") or {}).get("seed",2026))
    adapter = result["adapter"]
    if not isinstance(adapter, Mapping) or set(adapter) != {
        "command",
        "artifact_sha256",
        "receipt_sha256",
    }:
        raise EdgeRelayError("target adapter identity differs")
    if (
        not isinstance(adapter["command"], list)
        or not adapter["command"]
        or not all(isinstance(part, str) and part for part in adapter["command"])
        or adapter["artifact_sha256"]
        != {
            "engine_or_model_sha256": receipt["engine_or_model_sha256"],
            "checkpoint_sha256": receipt["checkpoint_sha256"],
            "final_decoders_sha256": receipt["final_decoders_sha256"],
        }
        or adapter["receipt_sha256"] != pinned.sha256_bytes(pinned.canonical_json(receipt))
    ):
        raise EdgeRelayError("target adapter identity differs")
    outputs = validate_adapter_output(result["outputs"], n_inputs=len(view.inputs["sample_ids"]))
    expected_identity = {
        "package_sha256": view.sha256,
        "plan_sha256": view.manifest["plan_sha256"],
        "procedure": view.manifest["procedure"],
    }
    if any(result[key] != value or receipt[key] != value for key, value in expected_identity.items()):
        raise EdgeRelayError("target result identity does not match sealed package")
    if receipt["target"] != view.manifest["target"]:
        raise EdgeRelayError("native adapter target does not match sealed package")
    candidate = view.manifest.get("candidate")
    if candidate is not None and (
        receipt["checkpoint_sha256"] != candidate["checkpoint_sha256"] or receipt["seed"] != candidate["seed"]
    ):
        raise EdgeRelayError("native adapter candidate identity does not match sealed package")
    return outputs


def score_collected(
    root: str | Path, *, package: InputPackage | PackageView, scorer: Callable[[dict[str, Any]], Any]
) -> Any:
    """Invoke the host-only scorer after, and only after, collection verification."""
    return scorer(verify_collected_run(root, package=package))


__all__ = ["score_collected", "verify_collected_run"]
