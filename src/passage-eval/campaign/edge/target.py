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

"""Target entry: prove local native artifacts, then run one blocking adapter command."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passage_eval.campaign.edge import TARGET_RESULT_SCHEMA, EdgeRelayError
from passage_eval.campaign.edge.package import verify_package
from passage_eval.campaign.edge.targets import validate_native_adapter
from passage_eval.io import pinned

RESULT_NAME = "target-result.json"
DIGEST_NAME = "target-result.sha256"
MAX_NATIVE_ARTIFACT_BYTES = 4 * 1024**3


def _need(condition: bool, message: str) -> None:
    if not condition:
        raise EdgeRelayError(message)


def _sha256_regular_file(path: str | Path) -> str:
    """Hash one bounded native artifact without following a symlink or trusting a changing file."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise EdgeRelayError(f"cannot open native artifact {path.name}: {exc.strerror or exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_NATIVE_ARTIFACT_BYTES:
            raise EdgeRelayError(f"native artifact {path.name} is not a bounded regular file")
        digest = hashlib.sha256()
        while chunk := os.read(fd, 1024 * 1024):
            digest.update(chunk)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise EdgeRelayError(f"native artifact {path.name} changed while it was hashed")
    return digest.hexdigest()


def _copy_verified_artifact(source: str | Path, destination: Path, expected_sha256: str) -> str:
    """Copy one verified artifact into the private execution directory without following links."""
    source = Path(source)
    try:
        source_fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise EdgeRelayError(f"cannot open native artifact {source.name}: {exc.strerror or exc}") from exc
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_NATIVE_ARTIFACT_BYTES:
            raise EdgeRelayError(f"native artifact {source.name} is not a bounded regular file")
        destination_fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
        try:
            digest = hashlib.sha256()
            while chunk := os.read(source_fd, 1024 * 1024):
                digest.update(chunk)
                os.write(destination_fd, chunk)
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
        after = os.fstat(source_fd)
    finally:
        os.close(source_fd)
    copied_sha256 = digest.hexdigest()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ) or copied_sha256 != expected_sha256:
        raise EdgeRelayError(f"native artifact {source.name} changed before its private snapshot was sealed")
    os.chmod(destination, 0o400)
    return copied_sha256


def validate_adapter_output(value: Any, *, n_inputs: int) -> dict[str, Any]:
    """Accept only per-input paths and per-input timing from a target adapter."""
    _need(isinstance(value, Mapping) and set(value) == {"paths", "timing_ms"}, "target output schema differs")
    paths, timing = value["paths"], value["timing_ms"]
    _need(isinstance(paths, list) and len(paths) == n_inputs, "target output paths differ from inputs")
    _need(isinstance(timing, list) and len(timing) == n_inputs, "target output timing differs from inputs")
    for path in paths:
        _need(
            isinstance(path, list)
            and path
            and all(
                isinstance(point, list)
                and len(point) == 2
                and all(isinstance(value, int) and not isinstance(value, bool) for value in point)
                for point in path
            ),
            "target output paths must contain row/column points",
        )
    _need(
        all(isinstance(item, (int, float)) and not isinstance(item, bool) and item >= 0 for item in timing),
        "timing_ms",
    )
    return {"paths": paths, "timing_ms": timing}


@dataclass(frozen=True)
class TargetRun:
    root: Path
    payload: dict[str, Any]
    sha256: str


def run_target(
    package_root: str | Path,
    package_sha256: str,
    *,
    receipt: Mapping[str, Any],
    output: str | Path,
    engine_or_model: str | Path,
    checkpoint: str | Path,
    final_decoders: str | Path,
    adapter_command: Sequence[str],
) -> TargetRun:
    """Run a blocking native adapter only after its declared local artifacts verify."""
    package = verify_package(package_root, package_sha256)
    checked = validate_native_adapter(
        receipt, expected_seed=(package.manifest.get("candidate") or {}).get("seed", 2026)
    )
    if (
        checked["target"] != package.manifest["target"]
        or checked["package_sha256"] != package.sha256
        or checked["plan_sha256"] != package.manifest["plan_sha256"]
        or checked["procedure"] != package.manifest["procedure"]
    ):
        raise EdgeRelayError("native adapter identity does not match sealed package")
    candidate = package.manifest.get("candidate")
    if candidate is not None and (
        checked["checkpoint_sha256"] != candidate["checkpoint_sha256"] or checked["seed"] != candidate["seed"]
    ):
        raise EdgeRelayError("native adapter candidate identity does not match sealed package")
    actual = {
        "engine_or_model_sha256": _sha256_regular_file(engine_or_model),
        "checkpoint_sha256": _sha256_regular_file(checkpoint),
        "final_decoders_sha256": _sha256_regular_file(final_decoders),
    }
    if any(checked[field] != digest for field, digest in actual.items()):
        raise EdgeRelayError("native adapter artifact SHA-256 differs from declared receipt")
    _need(
        bool(adapter_command) and all(isinstance(part, str) and part for part in adapter_command),
        "adapter command is required",
    )
    root = Path(output)
    if root.exists():
        raise EdgeRelayError(f"target result directory already exists: {root}")
    root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".edge-adapter-", dir=root.parent) as temporary:
        temporary_root = Path(temporary)
        snapshot_inputs = temporary_root / "sealed-inputs.json"
        snapshot_receipt = temporary_root / "sealed-receipt.json"
        snapshot_artifacts = {
            "engine_or_model_sha256": temporary_root / "engine-or-model.snapshot",
            "checkpoint_sha256": temporary_root / "checkpoint.snapshot",
            "final_decoders_sha256": temporary_root / "final-decoders.snapshot",
        }
        adapter_output = Path(temporary) / "adapter-output.json"
        try:
            input_bytes = pinned.canonical_json(package.inputs)
            receipt_bytes = pinned.canonical_json(checked)
            snapshot_inputs.write_bytes(input_bytes)
            snapshot_receipt.write_bytes(receipt_bytes)
            os.chmod(snapshot_inputs, 0o400)
            os.chmod(snapshot_receipt, 0o400)
            snapshot_digests = {
                "engine_or_model_sha256": _copy_verified_artifact(
                    engine_or_model, snapshot_artifacts["engine_or_model_sha256"], actual["engine_or_model_sha256"]
                ),
                "checkpoint_sha256": _copy_verified_artifact(
                    checkpoint, snapshot_artifacts["checkpoint_sha256"], actual["checkpoint_sha256"]
                ),
                "final_decoders_sha256": _copy_verified_artifact(
                    final_decoders, snapshot_artifacts["final_decoders_sha256"], actual["final_decoders_sha256"]
                ),
            }
            subprocess.run(
                [
                    *adapter_command,
                    "--inputs",
                    str(snapshot_inputs),
                    "--output",
                    str(adapter_output),
                    "--engine-or-model",
                    str(snapshot_artifacts["engine_or_model_sha256"]),
                    "--checkpoint",
                    str(snapshot_artifacts["checkpoint_sha256"]),
                    "--final-decoders",
                    str(snapshot_artifacts["final_decoders_sha256"]),
                    "--receipt",
                    str(snapshot_receipt),
                ],
                check=True,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            raw_output = pinned.read_bounded(adapter_output)
            outputs = validate_adapter_output(
                pinned.strict_json_loads(raw_output, name="adapter output"), n_inputs=len(package.inputs["sample_ids"])
            )
        except (OSError, subprocess.SubprocessError, pinned.PinnedFileError) as exc:
            raise EdgeRelayError(f"native adapter did not produce a valid output: {exc}") from exc
        if pinned.sha256_bytes(pinned.read_bounded(snapshot_inputs)) != pinned.sha256_bytes(
            input_bytes
        ) or pinned.sha256_bytes(pinned.read_bounded(snapshot_receipt)) != pinned.sha256_bytes(receipt_bytes):
            raise EdgeRelayError("sealed adapter snapshot changed during execution")
        if {name: _sha256_regular_file(path) for name, path in snapshot_artifacts.items()} != snapshot_digests:
            raise EdgeRelayError("sealed native artifact snapshot changed during adapter execution")
        adapter_identity = {
            "command": list(adapter_command),
            "artifact_sha256": snapshot_digests,
            "receipt_sha256": pinned.sha256_bytes(pinned.canonical_json(checked)),
        }
    payload = {
        "schema": TARGET_RESULT_SCHEMA,
        "package_sha256": package.sha256,
        "plan_sha256": package.manifest["plan_sha256"],
        "procedure": package.manifest["procedure"],
        "receipt": checked,
        "adapter": adapter_identity,
        "outputs": outputs,
    }
    raw = pinned.canonical_json(payload)
    root.mkdir(mode=0o700)
    (root / RESULT_NAME).write_bytes(raw)
    digest = pinned.sha256_bytes(raw)
    (root / DIGEST_NAME).write_text(digest + "\n")
    return TargetRun(root=root, payload=payload, sha256=digest)


__all__ = ["TargetRun", "run_target", "validate_adapter_output"]
