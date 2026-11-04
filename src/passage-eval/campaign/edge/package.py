"""Build and verify small, sealed, input-only target packages."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passage_eval.campaign.edge import EDGE_TARGETS, INPUT_MEMBERS, PACKAGE_SCHEMA, PackageError
from passage_eval.io import pinned
from passage_eval.io import roster as roster_io

MANIFEST_NAME = "package.json"
INPUTS_NAME = "inputs.json"


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise PackageError(message)


def _require_hex(value: Any, name: str) -> str:
    _need(pinned.is_hex64(value), f"{name} must be a SHA-256")
    return str(value)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _numeric_tree(value: Any) -> bool:
    return _number(value) or (isinstance(value, list) and bool(value) and all(_numeric_tree(item) for item in value))


def _bool_tree(value: Any) -> bool:
    return isinstance(value, bool) or (
        isinstance(value, list) and bool(value) and all(_bool_tree(item) for item in value)
    )


def _point(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and all(isinstance(item, int) and not isinstance(item, bool) for item in value)
    )


def _inputs(value: Any) -> dict[str, Any]:
    _need(isinstance(value, Mapping), "inputs must be an object")
    unknown = sorted(set(value) - set(INPUT_MEMBERS))
    missing = sorted(set(INPUT_MEMBERS) - set(value))
    _need(
        not unknown and not missing, f"inputs differ from the input-only members: unknown {unknown}, missing {missing}"
    )
    result = dict(value)
    _need(
        isinstance(result["sample_ids"], list) and all(isinstance(item, str) and item for item in result["sample_ids"]),
        "sample_ids",
    )
    _need(len(set(result["sample_ids"])) == len(result["sample_ids"]), "sample_ids must be distinct")
    lengths = {len(item) for item in result.values()}
    _need(len(lengths) == 1 and next(iter(lengths), 0) > 0, "every input member needs one entry per request")
    _need(all(_numeric_tree(item) for item in result["terrain"]), "terrain must contain finite numbers only")
    _need(all(_bool_tree(item) for item in result["obstacles"]), "obstacles must contain booleans only")
    _need(all(_point(item) for item in result["starts"]), "starts must contain row/column points")
    _need(all(_point(item) for item in result["goals"]), "goals must contain row/column points")
    _need(all(_numeric_tree(item) for item in result["input_tensor"]), "input_tensor must contain finite numbers only")
    return result


def _validation_binding(roster: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, str]:
    """Bind package sample IDs to a validated, non-test roster before target input is written."""
    try:
        checked = roster_io.validate_roster(roster)
    except ValueError as exc:
        raise PackageError(f"validation roster is invalid: {exc}") from exc
    _need(checked["dataset"]["split"] == "validation", "validation roster split differs")
    known = {task["sample_id"] for task in checked["tasks"]}
    _need(set(inputs["sample_ids"]) <= known, "package inputs are not bound to the validation roster")
    return {
        "roster_sha256": checked["roster_sha256"],
        "sample_ids_sha256": pinned.canonical_sha256(inputs["sample_ids"]),
    }


def _candidate(value: Mapping[str, Any]) -> dict[str, Any]:
    _need(isinstance(value, Mapping) and set(value) == {"checkpoint_sha256", "seed"}, "candidate binding differs")
    checked = dict(value)
    _require_hex(checked["checkpoint_sha256"], "candidate checkpoint_sha256")
    _need(type(checked["seed"]) is int and checked["seed"] in (1234,2026,3407), "candidate seed must be 1234,2026 or3407")
    return checked


@dataclass(frozen=True)
class InputPackage:
    """A local package root and its SHA-256 pinned manifest."""

    root: Path
    sha256: str
    manifest: dict[str, Any]


@dataclass(frozen=True)
class PackageView:
    root: Path
    sha256: str
    manifest: dict[str, Any]
    inputs: dict[str, Any]


def build_input_package(
    root: str | Path,
    *,
    target: str,
    plan_sha256: str,
    procedure: str,
    inputs: Mapping[str, Any],
    validation_roster: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    replica_catalog: str | Path | None = None,
    replica_catalog_sha256: str | None = None,
) -> InputPackage:
    """Create a new package containing ordinary inputs and no score-bearing data."""
    _need(target in EDGE_TARGETS, f"unknown edge target {target!r}")
    _require_hex(plan_sha256, "plan_sha256")
    _need(isinstance(procedure, str) and procedure.strip(), "procedure is required")
    safe_inputs = _inputs(inputs)
    _need(
        (validation_roster is None) == (candidate is None),
        "validation roster and candidate bindings are required together",
    )
    binding = _validation_binding(validation_roster, safe_inputs) if validation_roster is not None else None
    candidate_binding = _candidate(candidate) if candidate is not None else None
    if candidate_binding is not None and candidate_binding["seed"] != 2026:
        from passage_eval.protocol.replicas import load_replica_catalog, replica_identity
        _need(replica_catalog is not None and pinned.is_hex64(replica_catalog_sha256), "replica package requires its pinned catalogue")
        catalog = load_replica_catalog(replica_catalog,replica_catalog_sha256)
        identity = replica_identity(catalog,procedure,candidate_binding["seed"])
        _need(candidate_binding["checkpoint_sha256"] == identity["checkpoint_sha256"], "replica checkpoint differs from catalogue")
    root = Path(root)
    _need(not root.exists(), f"package directory already exists: {root}")
    root.mkdir(mode=0o700, parents=True)
    try:
        input_bytes = pinned.canonical_json(safe_inputs)
        (root / INPUTS_NAME).write_bytes(input_bytes)
        manifest = {
            "schema": PACKAGE_SCHEMA,
            "target": target,
            "plan_sha256": plan_sha256,
            "procedure": procedure,
            "inputs": {"path": INPUTS_NAME, "sha256": pinned.sha256_bytes(input_bytes)},
        }
        if binding is not None:
            manifest["validation"] = binding
            manifest["candidate"] = candidate_binding
        manifest_bytes = pinned.canonical_json(manifest)
        (root / MANIFEST_NAME).write_bytes(manifest_bytes)
    except Exception:
        for path in root.iterdir():
            path.unlink()
        root.rmdir()
        raise
    return InputPackage(root=root, sha256=pinned.sha256_bytes(manifest_bytes), manifest=manifest)


def verify_package(root: str | Path, manifest_sha256: str) -> PackageView:
    """Verify a package pin, its exact two files, and its label-free input payload."""
    root = Path(root)
    _require_hex(manifest_sha256, "package SHA-256")
    try:
        names = {path.name for path in root.iterdir()}
    except OSError as exc:
        raise PackageError(f"cannot read package: {exc}") from exc
    _need(names == {MANIFEST_NAME, INPUTS_NAME}, "package contains unlisted files")
    _need(all(not path.is_symlink() and path.is_file() for path in root.iterdir()), "package contains an unsafe file")
    try:
        manifest_bytes = pinned.read_bounded(root / MANIFEST_NAME)
        _need(pinned.sha256_bytes(manifest_bytes) == manifest_sha256, "package manifest SHA-256 mismatch")
        manifest = pinned.strict_json_loads(manifest_bytes, name=MANIFEST_NAME)
    except pinned.PinnedFileError as exc:
        raise PackageError(f"invalid package manifest: {exc}") from exc
    _need(isinstance(manifest, dict), "package manifest must be an object")
    base_keys = {"schema", "target", "plan_sha256", "procedure", "inputs"}
    bound_keys = base_keys | {"validation", "candidate"}
    _need(set(manifest) in (base_keys, bound_keys), "package manifest keys differ")
    _need(manifest["schema"] == PACKAGE_SCHEMA, "package schema differs")
    _need(manifest["target"] in EDGE_TARGETS, "package target differs")
    _require_hex(manifest["plan_sha256"], "package plan_sha256")
    _need(isinstance(manifest["procedure"], str) and manifest["procedure"], "package procedure differs")
    if set(manifest) == bound_keys:
        validation = manifest["validation"]
        _need(
            isinstance(validation, Mapping) and set(validation) == {"roster_sha256", "sample_ids_sha256"},
            "package validation binding differs",
        )
        _require_hex(validation["roster_sha256"], "package validation roster_sha256")
        _require_hex(validation["sample_ids_sha256"], "package validation sample_ids_sha256")
        manifest["validation"] = dict(validation)
        manifest["candidate"] = _candidate(manifest["candidate"])
    source = manifest["inputs"]
    _need(isinstance(source, Mapping) and set(source) == {"path", "sha256"}, "package inputs binding differs")
    _need(source["path"] == INPUTS_NAME, "package inputs path differs")
    _require_hex(source["sha256"], "package inputs SHA-256")
    try:
        input_bytes = pinned.read_bounded(root / INPUTS_NAME)
        _need(pinned.sha256_bytes(input_bytes) == source["sha256"], "package input SHA-256 mismatch")
        inputs = _inputs(pinned.strict_json_loads(input_bytes, name=INPUTS_NAME))
    except pinned.PinnedFileError as exc:
        raise PackageError(f"invalid package inputs: {exc}") from exc
    return PackageView(root=root, sha256=manifest_sha256, manifest=manifest, inputs=inputs)


open_package = verify_package

__all__ = ["InputPackage", "PackageView", "build_input_package", "open_package", "verify_package"]
