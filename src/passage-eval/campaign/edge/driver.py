"""One synchronous, CAVIAR-leased process for a sealed Edge package invocation.

The driver has no SSH, target adapter, service, telemetry, or retry machinery.
It can run only the fixed local package-verification entrypoint, so a caller
cannot turn a lease into an arbitrary subprocess launcher.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passage_eval.campaign.edge.lease import (
    EdgeLeaseError,
    holder_host_for,
    new_effect_key,
    parse_job_id,
    resource_for,
    validate_target,
)
from passage_eval.campaign.edge.package import InputPackage, PackageView, verify_package
from passage_eval.io import pinned

CAVIAR_ROOT = Path.cwd()
PACKAGE_TIMEOUT_SECONDS = 1200
_OUTER_TIMEOUT_SECONDS = float(PACKAGE_TIMEOUT_SECONDS + 30)
_TASK = "passage-edge-qualification"
_CLAIM = "passage-edge-qualification"
_ROOT_SCHEMA = "passage-edge-sealed-root/v1"
_ROOT_FILE = "sealed-root.json"
_PACKAGE_ENTRYPOINT = (sys.executable, "-I", "-B", "-m", "passage_eval.campaign.edge", "sealed-package")


class LeaseReleaseError(ValueError):
    """A retained lease cannot be released without specific idle evidence."""


@dataclass(frozen=True)
class SealedRoot:
    """A root containing exactly one hash-pinned sealed-root control artifact."""

    root: Path
    sha256: str


@dataclass(frozen=True)
class SealedPackageInvocation:
    """The only package invocation form accepted by the leased driver."""

    package: InputPackage | PackageView
    adapter_root: SealedRoot
    result_root: SealedRoot


@dataclass(frozen=True)
class LeaseAttempt:
    """The local record for one CAVIAR attempt, including its sealed scope for later drain work."""

    target: str
    resource: str
    effect_key: str
    effect_fingerprint: str
    invocation: SealedPackageInvocation
    job_id: str | None
    succeeded: bool
    lease_retained: bool
    returncode: int
    stderr: str


@dataclass(frozen=True)
class IdleCheckEvidence:
    """Explicit operator proof that exactly one retained attempt's target is idle."""

    target: str
    resource: str
    job_id: str | None
    effect_key: str
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class LeaseRelease:
    """The explicit CAVIAR release associated with its prior idle-check evidence."""

    resource: str
    job_id: str | None
    idle_evidence: dict[str, Any]
    stdout: str


RunSubprocess = Callable[..., subprocess.CompletedProcess[str]]


def _root_payload(root: SealedRoot, *, target: str, package_sha256: str, role: str) -> None:
    if not isinstance(root, SealedRoot) or not isinstance(root.root, Path):
        raise ValueError(f"{role} root must be a sealed root")
    if root.root.is_symlink() or not root.root.is_dir():
        raise ValueError(f"{role} root is not a regular directory")
    try:
        names = {path.name for path in root.root.iterdir()}
    except OSError as exc:
        raise ValueError(f"{role} root cannot be read: {exc}") from exc
    if names != {_ROOT_FILE}:
        raise ValueError(f"{role} root contents differ from its sealed artifact")
    try:
        payload = pinned.strict_json_loads(pinned.read_pinned(root.root / _ROOT_FILE, root.sha256), name=_ROOT_FILE)
    except pinned.PinnedFileError as exc:
        raise ValueError(f"{role} root seal differs: {exc}") from exc
    expected = {"schema": _ROOT_SCHEMA, "role": role, "target": target, "package_sha256": package_sha256}
    if payload != expected:
        raise ValueError(f"{role} root seal does not bind this target package")


def _checked_invocation(target: str, invocation: SealedPackageInvocation) -> PackageView:
    if not isinstance(invocation, SealedPackageInvocation):
        raise ValueError("leased driver requires a sealed package invocation")
    if not isinstance(invocation.package, (InputPackage, PackageView)):
        raise ValueError("sealed package invocation has no verified package")
    try:
        package = verify_package(invocation.package.root, invocation.package.sha256)
    except ValueError as exc:
        raise ValueError(f"sealed package invocation differs: {exc}") from exc
    if package.manifest["target"] != target:
        raise ValueError("sealed package invocation target differs")
    _root_payload(invocation.adapter_root, target=target, package_sha256=package.sha256, role="adapter")
    _root_payload(invocation.result_root, target=target, package_sha256=package.sha256, role="result")
    if invocation.adapter_root.root.resolve() == invocation.result_root.root.resolve():
        raise ValueError("adapter and result roots must differ")
    return package


def sealed_package_argv(target: str, invocation: SealedPackageInvocation) -> list[str]:
    """Return the sole allowlisted package entrypoint after validating all sealed roots."""
    package = _checked_invocation(validate_target(target), invocation)
    return [
        *_PACKAGE_ENTRYPOINT,
        f"--package={package.root.resolve()}",
        f"--package-sha256={package.sha256}",
        f"--adapter-root={invocation.adapter_root.root.resolve()}",
        f"--adapter-root-sha256={invocation.adapter_root.sha256}",
        f"--result-root={invocation.result_root.root.resolve()}",
        f"--result-root-sha256={invocation.result_root.sha256}",
    ]


def _effect_fingerprint(target: str, package_argv: Sequence[str]) -> str:
    encoded = json.dumps(
        {"package_argv": list(package_argv), "target": target}, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _job_run_argv(target: str, package_argv: Sequence[str], *, effect_key: str, effect_fingerprint: str, lease_command: str = "cavops") -> list[str]:
    if not isinstance(effect_key, str) or not effect_key.startswith(f"passage-edge-{target}-"):
        raise ValueError("effect key does not belong to this target")
    if not pinned.is_hex64(effect_fingerprint):
        raise ValueError("effect fingerprint must be a SHA-256")
    return [
        lease_command,
        "job-run",
        "--host",
        holder_host_for(target),
        "--resource",
        resource_for(target),
        "--effect-key",
        effect_key,
        "--effect-fingerprint",
        effect_fingerprint,
        "--task",
        _TASK,
        "--claim",
        _CLAIM,
        "--wait",
        "--",
        "/usr/bin/timeout",
        "--signal=TERM",
        "--kill-after=30s",
        f"{PACKAGE_TIMEOUT_SECONDS}s",
        "--",
        *package_argv,
    ]


class EdgeJobDriver:
    """Synchronously dispatch exactly one sealed package invocation under a CAVIAR lease."""

    def __init__(self, *, run: RunSubprocess = subprocess.run, caviar_root: str | Path = CAVIAR_ROOT,
                 lease_command: str | None = None):
        self._run = run
        self._lease_command = lease_command or os.environ.get("PASSAGE_EDGE_LEASE_COMMAND", "cavops")
        self._caviar_root = Path(caviar_root)
        self._attempts: dict[str, LeaseAttempt] = {}

    def _record(self, attempt: LeaseAttempt) -> LeaseAttempt:
        self._attempts[attempt.effect_key] = attempt
        return attempt

    def _ambiguous(
        self,
        *,
        target: str,
        invocation: SealedPackageInvocation,
        effect_key: str,
        effect_fingerprint: str,
        error: BaseException,
    ) -> LeaseAttempt:
        return self._record(
            LeaseAttempt(
                target=target,
                resource=resource_for(target),
                effect_key=effect_key,
                effect_fingerprint=effect_fingerprint,
                invocation=invocation,
                job_id=None,
                succeeded=False,
                lease_retained=True,
                returncode=-1,
                stderr=f"{type(error).__name__}: {error}",
            )
        )

    def run_package(self, target: str, invocation: SealedPackageInvocation) -> LeaseAttempt:
        """Run the one fixed package entrypoint; every runner exception becomes a retained ambiguous attempt."""
        target = validate_target(target)
        package_argv = sealed_package_argv(target, invocation)
        effect_key = new_effect_key(target)
        effect_fingerprint = _effect_fingerprint(target, package_argv)
        try:
            completed = self._run(
                _job_run_argv(target, package_argv, effect_key=effect_key, effect_fingerprint=effect_fingerprint,
                              lease_command=self._lease_command),
                cwd=self._caviar_root,
                timeout=_OUTER_TIMEOUT_SECONDS,
                capture_output=True,
                text=True,
                check=False,
            )
        except BaseException as exc:
            return self._ambiguous(
                target=target,
                invocation=invocation,
                effect_key=effect_key,
                effect_fingerprint=effect_fingerprint,
                error=exc,
            )
        try:
            job_id = parse_job_id(completed.stdout)
        except EdgeLeaseError:
            job_id = None
        succeeded = completed.returncode == 0 and job_id is not None
        return self._record(
            LeaseAttempt(
                target=target,
                resource=resource_for(target),
                effect_key=effect_key,
                effect_fingerprint=effect_fingerprint,
                invocation=invocation,
                job_id=job_id,
                succeeded=succeeded,
                lease_retained=not succeeded,
                returncode=completed.returncode,
                stderr=completed.stderr,
            )
        )

    def release_after_idle(self, attempt: LeaseAttempt, idle_check: IdleCheckEvidence | None) -> LeaseRelease:
        """Force-release only this driver's retained attempt and a nonempty, matching idle proof."""
        if not isinstance(attempt, LeaseAttempt) or self._attempts.get(attempt.effect_key) is not attempt:
            raise LeaseReleaseError("only an attempt issued by this driver may be released")
        if attempt.resource != resource_for(attempt.target):
            raise LeaseReleaseError("attempt resource differs from its target resource")
        if not attempt.lease_retained:
            raise LeaseReleaseError("only an attempt with a retained lease may be released")
        if not isinstance(idle_check, IdleCheckEvidence):
            raise LeaseReleaseError("idle-check evidence is required before release")
        if (
            idle_check.target != attempt.target
            or idle_check.resource != attempt.resource
            or idle_check.job_id != attempt.job_id
            or idle_check.effect_key != attempt.effect_key
        ):
            raise LeaseReleaseError("idle-check evidence belongs to another attempt")
        evidence = idle_check.evidence
        proof = evidence.get("proof") if isinstance(evidence, Mapping) else None
        if (
            not isinstance(evidence, Mapping)
            or evidence.get("idle") is not True
            or not isinstance(proof, Mapping)
            or not proof
        ):
            raise LeaseReleaseError("idle-check evidence does not prove idle with a nonempty proof")
        completed = self._run(
            [self._lease_command, "release", attempt.resource, "--force"],
            cwd=self._caviar_root,
            timeout=30.0,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise LeaseReleaseError(f"cavops release failed with exit {completed.returncode}")
        return LeaseRelease(
            resource=attempt.resource,
            job_id=attempt.job_id,
            idle_evidence=dict(idle_check.evidence),
            stdout=completed.stdout,
        )


__all__ = [
    "CAVIAR_ROOT",
    "PACKAGE_TIMEOUT_SECONDS",
    "EdgeJobDriver",
    "IdleCheckEvidence",
    "LeaseAttempt",
    "LeaseRelease",
    "LeaseReleaseError",
    "SealedPackageInvocation",
    "SealedRoot",
    "sealed_package_argv",
]
