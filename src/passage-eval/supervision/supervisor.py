# MIT License
#
# Copyright (c) 2026 THALES

"""Trusted supervisor: launches, feeds, times, preempts and validates candidate workers.

The supervisor never imports candidate code. It holds the scenes and references;
workers receive only redacted messages (:mod:`passage_eval.supervision.redaction`).

Timing boundary ``passage-timing-boundary/v3-supervised-ipc-fresh-pass``
-----------------------------------------------------------------------
Per request, with the supervisor's monotonic clock:

``service_wall_ms``
    from immediately before the redacted message is built and serialised, to
    immediately after the reply has been fully read and deserialised. It contains
    both IPC directions, worker-side deserialisation, candidate compute and the
    worker-side device synchronisation that precedes the reply. In split mode it
    spans both hops and the supervisor work between them (including the field contract check).
``candidate_reported_ms``
    worker-internal ``call + synchronize`` time. Self-reported, non-authoritative.
``validation_recost_ms``
    supervisor path legality, physical recost and budget accounting, after the reply.
``total_wall_ms = service_wall_ms + validation_recost_ms``
    legacy profile ``supervised_total_v3`` (the default): the latency compared against the wall-time budget
    and fed to the scorer. The profiles ``supervised_service_v1`` (boundary v4) and
    ``supervised_service_rewarm_v1`` (boundary v5) compare and score ``service_wall_ms`` instead; the total
    remains recorded (see :mod:`passage_eval.execution.timing`).

Pass policy (new in v3): every timed pass (one complete roster traversal: a series and a repeat)
runs on freshly started workers, with fresh per-pass tokens. A warm pass is preceded by an untimed
warm-up of ``latency_warmup`` requests drawn from inputs disjoint from the roster, in the same fresh
worker. A candidate therefore cannot answer a timed request from a cache filled by an earlier pass,
by the warm-up, or by the IPC baseline.

Restart policy of ``supervised_service_rewarm_v1`` (boundary v5, proposed, not sealed): when a worker is
restarted inside a warm pass, the pass warm-up is replayed, outside the boundary, before the next measured
request; at most one re-warm precedes a measured request and a re-warm never triggers another. A re-warm
stops at its first request that does not return ``ok``; a worker it loses is restarted outside
``max_restarts``, and the pass is not re-warmed again. Restarts caused by roster tasks and by the initial
warm-up are counted exactly as in the other profiles, so the re-warm cannot exhaust the restart cap. Every
record states whether a re-warm preceded it and whether the workers that served it had completed a warm-up.

Not inside the boundary: interpreter/process start, sandbox setup, candidate import
and model load, the baseline pings (run on a separate worker), the warm-up requests, worker
kill/restart time (recorded as separate events), artifact persistence and offline scoring.
Under the service profiles the validation and physical recost are outside the boundary too, and under
the re-warm profile so are the re-warm requests.
``ipc_echo_baseline_ms`` is measured before the run for payloads of the same size and stored in
the manifest; it is never subtracted from any recorded value.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
import platform
import secrets
import select
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from passage_eval.execution import timing
from passage_eval.execution.artifacts import RunArtifact, json_safe, route_hash, write_run_artifact
from passage_eval.execution.runner import _budget_status, _work, request_identity
from passage_eval.io import roster as roster_io
from passage_eval.metrics.cost import path_cost
from passage_eval.metrics.reliability import path_status
from passage_eval.protocol.attribution import component_control
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult
from passage_eval.protocol.guards import refuse_test_roster, refuse_test_split
from passage_eval.protocol.planner import candidate_scene, validated_raw_path
from passage_eval.supervision import (
    DEFAULT_FORBIDDEN_MODULES,
    IPC_PROTOCOL,
    POLICY_DECODER_ONLY,
    POLICY_ORDINARY_REQUEST,
    PURPOSE_DIAGNOSTIC,
    PURPOSE_VERIFIED,
    PURPOSES,
    SUPERVISION_SCHEMA,
    TIMING_BOUNDARY_ID,
    WORKER_KILL_WAIT_S,
)
from passage_eval.supervision import environment as environment_mod
from passage_eval.supervision import identity as identity_mod
from passage_eval.supervision import sandbox, sanitize, treehash
from passage_eval.supervision import session as session_mod
from passage_eval.supervision.protocol import (
    DEFAULT_LIMITS,
    FrameLimits,
    ProtocolError,
    StreamClosed,
    TruncatedFrame,
    WORKER_TO_SUPERVISOR,
    encode_frame,
    expect_body,
    expect_type,
    read_frame,
)
from passage_eval.supervision.redaction import (
    FIELD_CONTRACT,
    FieldContractError,
    TENSOR_IMPLIED_METADATA,
    normalize_allow_list,
    redact_decoder,
    redact_ordinary,
    request_token,
)

BUDGET_ENFORCEMENT = "preemptive_wall_deadline_cooperative_work_counters"
CANCELLATION_POLICY = "kill_process_group_and_restart_worker"
NOT_IN_BOUNDARY = (
    "interpreter_and_sandbox_start",
    "candidate_import_and_model_load",
    "ipc_baseline_pings",
    "warmup_requests",
    "worker_kill_and_restart",
    "artifact_persistence",
    "offline_scoring",
)
#: v2 adds the worker rlimits and the IPC frame limits to v1; both can turn a request into a failure.
PROCEDURE_LIMITS_SCHEMA = "passage-procedure-limits/v2"
#: The v1 block, kept as the limits of the v2 split-procedure pin so that digests pinned with it stay valid; that
#: descriptor binds the rlimits and frame limits through its own ``isolation`` and ``frame_limits`` entries.
PROCEDURE_LIMITS_SCHEMA_V1 = "passage-procedure-limits/v1"


def _not_in_boundary(profile: timing.TimingProfile) -> list[str]:
    items = list(NOT_IN_BOUNDARY)
    if not profile.validation_in_boundary:
        items.append("validation_and_physical_recost")
    if profile.rewarm_after_restart:
        items.append("rewarm_requests_after_restart")
    return items


_STDERR_TAIL = 1024
#: Record/manifest fields whose values come from the candidate process (see the counters_source note).
SELF_REPORTED_FIELDS = (
    "work",
    "stage",
    "stage_work",
    "fallback",
    "precision",
    "diagnostics",
    "stage_spans",
    "resource_peaks",
    "candidate_reported_ms",
    "hops[*].worker_spans",
    "workers_ready[*].loaded_files",
    "workers_ready[*].sys_modules_top_level",
    "workers_ready[*].forbidden_modules_checked",
)
COUNTERS_OPERATOR = "operator_decoder"
COUNTERS_PLANNER = "operator_planner"
COUNTERS_SELF_REPORTED = "candidate_self_reported"
COUNTERS_SOURCES = (COUNTERS_OPERATOR, COUNTERS_PLANNER, COUNTERS_SELF_REPORTED)


class SupervisionError(RuntimeError):
    """The run cannot continue safely (worker cannot start, identity changed, restart storm)."""


class IntegrityError(SupervisionError):
    """A worker's identity or isolation attestation contradicts what the supervisor set up. Always aborts the run."""


class PurposeError(ValueError):
    """The run does not satisfy the requirements of its declared purpose."""


class DeadlineExceeded(Exception):
    pass


class WorkerGone(Exception):
    pass


@dataclass(frozen=True)
class SupervisorConfig:
    worker_python: str
    isolation: sandbox.IsolationConfig
    candidate_paths: tuple[str, ...] = ()
    metadata_allow: tuple[str, ...] = ()
    limits: FrameLimits = DEFAULT_LIMITS
    deadline_grace_ms: float = 10.0
    hard_timeout_ms: float = 600_000.0
    startup_timeout_s: float = 600.0
    ipc_baseline_pings: int = 50
    ipc_baseline_warmup: int = 5
    max_restarts: int = 100
    purpose: str = PURPOSE_DIAGNOSTIC
    decoder_allow_sha256: tuple[str, ...] = ()
    planner_allow_sha256: tuple[str, ...] = ()
    environment_allow_sha256: tuple[str, ...] = ()
    split_procedure_allow_sha256: tuple[str, ...] = ()
    require_split_procedure_pin: bool = False
    attest_files: tuple[str, ...] = ()
    forbidden_modules: tuple[str, ...] = DEFAULT_FORBIDDEN_MODULES
    tree_hash_cache: str | None = None  # None: default directory (diagnostic) or none (verified); 'off' disables
    staging_dir: str | None = None
    timing_profile: str = timing.LEGACY  # see passage_eval.execution.timing; the legacy default keeps v3

    def __post_init__(self):
        timing.get_profile(self.timing_profile)
        if self.timing_profile == timing.NATIVE_PERSISTENT:
            raise ValueError("the native persistent timing profile cannot label a supervised worker run")
        if self.purpose not in PURPOSES:
            raise ValueError(f"purpose must be one of {list(PURPOSES)}")
        for label, digests in (
            ("decoder", self.decoder_allow_sha256),
            ("planner", self.planner_allow_sha256),
            ("environment", self.environment_allow_sha256),
            ("split procedure", self.split_procedure_allow_sha256),
        ):
            for digest in digests:
                if not treehash.is_hex64(digest):
                    raise ValueError(f"{label} allow-list entries must be lowercase 64-character SHA-256 values")
        for name in self.forbidden_modules:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"forbidden module names must be top-level module names, got {name!r}")
        if not (isinstance(self.deadline_grace_ms, (int, float)) and 0.5 <= self.deadline_grace_ms <= 10_000):
            raise ValueError("deadline_grace_ms must lie in [0.5, 10000]")
        if not (isinstance(self.hard_timeout_ms, (int, float)) and self.hard_timeout_ms > 0):
            raise ValueError("hard_timeout_ms must be positive")
        if not (isinstance(self.startup_timeout_s, (int, float)) and self.startup_timeout_s > 0):
            raise ValueError("startup_timeout_s must be positive")
        for name in ("ipc_baseline_pings", "ipc_baseline_warmup", "max_restarts"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (1 if name == "ipc_baseline_pings" else 0)
            ):
                raise ValueError(f"{name} is out of range")
        object.__setattr__(self, "metadata_allow", normalize_allow_list(self.metadata_allow))
        if not isinstance(self.require_split_procedure_pin, bool):
            raise ValueError("require_split_procedure_pin must be boolean")
        if any(not isinstance(item, str) or not item for item in self.attest_files):
            raise ValueError("attest_files must contain nonempty role=absolute_path strings")
        object.__setattr__(self, "attest_files", tuple(sorted(set(self.attest_files))))
        for name in (
            "decoder_allow_sha256",
            "planner_allow_sha256",
            "environment_allow_sha256",
            "split_procedure_allow_sha256",
        ):
            object.__setattr__(self, name, tuple(sorted(set(getattr(self, name)))))
        object.__setattr__(self, "forbidden_modules", tuple(sorted(set(self.forbidden_modules))))
        if self.purpose == PURPOSE_VERIFIED:
            problems = []
            if self.tree_hash_cache not in (None, "", "off"):
                problems.append(
                    "a tree-hash cache is not allowed (every digest of a verified run is recomputed; a cache "
                    "keyed by stat signature could be filled by an earlier process)"
                )
            missing = sorted(set(DEFAULT_FORBIDDEN_MODULES) - set(self.forbidden_modules))
            if missing:
                problems.append(
                    f"the default forbidden modules {missing} must stay enforced "
                    f"(do not use --no-default-forbidden-modules)"
                )
            if problems:
                raise PurposeError("verified-candidate-run refused: " + "; ".join(problems))

    def cache_setting(self) -> str | None:
        """Where tree digests are cached. A verified run never uses a cache; a diagnostic run may."""
        if self.purpose == PURPOSE_VERIFIED:
            return None
        if self.tree_hash_cache is not None:
            return self.tree_hash_cache
        return os.environ.get("PASSAGE_TREE_HASH_CACHE") or environment_mod.default_cache_directory()


@dataclass(frozen=True)
class WorkerSpec:
    role: str  # planner | predictor | decoder
    entry: str  # module:attr
    kwargs: Mapping[str, Any] = field(default_factory=dict)
    inference_format: str = "request"
    seed: int | None = None


@dataclass(frozen=True)
class SplitSpec:
    predictor: str
    decoder: str
    decoder_kwargs: Mapping[str, Any] = field(default_factory=dict)
    precision: str = "unspecified"


# ---------------------------------------------------------------------------
# one worker process
# ---------------------------------------------------------------------------


class WorkerProcess:
    """A single sandboxed worker with deadline-bound framed IO. Lifecycle owned by the runner."""

    def __init__(
        self,
        spec: WorkerSpec,
        config: SupervisorConfig,
        binds: Sequence[sandbox.Bind],
        import_paths: Sequence[str],
        clock: Callable[[], float],
    ):
        self.spec, self.config, self.binds, self.import_paths, self.clock = (
            spec,
            config,
            list(binds),
            list(import_paths),
            clock,
        )
        self.proc: subprocess.Popen | None = None
        self.private: Path | None = None
        self.generation = -1
        self.seq = 0
        self.alive = False
        self.last_exit: dict[str, Any] | None = None
        self._stderr_path: Path | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> dict[str, Any]:
        iso = self.config.isolation
        self.private = Path(tempfile.mkdtemp(prefix="passage-worker-"))
        os.chmod(self.private, 0o700)
        in_sandbox = iso.level == "bwrap"
        home = sandbox.SANDBOX_HOME if in_sandbox else str(self.private / "home")
        tmpdir = "/tmp" if in_sandbox else str(self.private / "tmp")
        if not in_sandbox:
            os.mkdir(home, 0o700)
            os.mkdir(tmpdir, 0o700)
        env = sandbox.scrubbed_env(
            os.environ, home=home, tmpdir=tmpdir, extras=iso.env_set, passthrough=iso.env_passthrough
        )
        prefix = sandbox.SANDBOX_PYCACHE if in_sandbox else str(self.private / "pycache")
        inner = [
            os.path.abspath(self.config.worker_python),
            "-I",
            *sandbox.pycache_args(prefix),
            "-m",
            "passage_eval.supervision.worker",
            *iso.rlimits.worker_args(),
            "--max-header-bytes",
            str(self.config.limits.max_header_bytes),
            "--max-payload-bytes",
            str(self.config.limits.max_payload_bytes),
            "--max-dim",
            str(self.config.limits.max_dim),
            "--max-elements",
            str(self.config.limits.max_elements),
            *[arg for name in self.config.forbidden_modules for arg in ("--forbid-module", name)],
        ]
        argv = sandbox.bwrap_command(iso, self.binds, inner) if in_sandbox else inner
        self._stderr_path = self.private / "stderr.log"
        try:
            with open(self._stderr_path, "wb") as stderr_file:
                self.proc = subprocess.Popen(
                    argv,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=stderr_file,
                    env=env,
                    cwd=home if not in_sandbox else str(self.private),
                    start_new_session=True,
                    close_fds=True,
                    bufsize=0,
                )
        except OSError as error:
            shutil.rmtree(self.private, ignore_errors=True)
            self.private = None
            raise SupervisionError(f"cannot launch {self.spec.role} worker: {sanitize.cap_text(error, 256)}") from None
        self._rfd, self._wfd = self.proc.stdout.fileno(), self.proc.stdin.fileno()
        os.set_blocking(self._rfd, False)
        os.set_blocking(self._wfd, False)
        self._poll = select.poll()
        self.generation += 1
        self.seq = 0
        self.alive = True
        deadline = self.clock() + self.config.startup_timeout_s
        try:
            hello = {
                "role": self.spec.role,
                "entry": self.spec.entry,
                "kwargs": dict(self.spec.kwargs),
                "inference_format": self.spec.inference_format,
                "seed": self.spec.seed,
                "import_paths": list(self.import_paths),
                "ipc": IPC_PROTOCOL,
            }
            self.send(encode_frame("hello", self._next_seq(), hello, limits=self.config.limits), deadline)
            attestation = self.receive(deadline)  # sent before any candidate code is imported
            loaded = self.receive(deadline) if attestation.type == "ready" else attestation
            if attestation.type == "ready" and loaded.type == "result":
                _validate_startup(attestation.body, loaded.body)
        except (DeadlineExceeded, WorkerGone, ProtocolError, OSError, ValueError, TypeError) as error:
            detail = self.kill()
            raise SupervisionError(
                f"{self.spec.role} worker failed to start ({type(error).__name__}: "
                f"{sanitize.cap_text(error, 256)}); "
                f"stderr tail: {detail.get('stderr_tail', '')[-sanitize.MAX_STDERR_CHARS :]!r}"
            ) from None
        for frame in (attestation, loaded):
            if frame.type == "error":
                detail = self.kill()
                raise SupervisionError(
                    f"{self.spec.role} worker refused to load {self.spec.entry}: "
                    f"{sanitize.cap_text(frame.body.get('error_type'), sanitize.MAX_ERROR_TYPE_CHARS)}: "
                    f"{sanitize.cap_text(frame.body.get('error'), sanitize.MAX_ERROR_CHARS)}; "
                    f"stderr tail: {detail.get('stderr_tail', '')[-sanitize.MAX_STDERR_CHARS :]!r}"
                )
        if attestation.type != "ready" or loaded.type != "result":
            self.kill()
            raise SupervisionError(f"unexpected startup replies: {attestation.type}, {loaded.type}")
        return {**attestation.body, **loaded.body}

    def _next_seq(self) -> int:
        self.seq += 1
        return self.seq

    def next_seq(self) -> int:
        return self._next_seq()

    def kill(self) -> dict[str, Any]:
        """SIGKILL the whole process group, reap it and discard the private directory."""
        started = self.clock()
        detail: dict[str, Any] = {}
        proc = self.proc
        if proc is not None:
            if proc.returncode is None:  # leader not yet reaped: its pgid is still reserved for us
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    os.killpg(proc.pid, signal.SIGKILL)
            try:
                proc.wait(timeout=WORKER_KILL_WAIT_S)
            except subprocess.TimeoutExpired:
                raise SupervisionError("worker did not exit after SIGKILL") from None
            detail["returncode"] = proc.returncode
            detail["stderr_tail"] = self._read_stderr_tail()
            for stream in (proc.stdin, proc.stdout):
                with contextlib.suppress(OSError):
                    stream.close()
        self.alive = False
        self.last_exit = detail
        if self.private is not None:
            shutil.rmtree(self.private, ignore_errors=True)
            self.private = None
        detail["kill_ms"] = 1000.0 * (self.clock() - started)
        return detail

    def _read_stderr_tail(self) -> str:
        try:
            with open(self._stderr_path, "rb") as stream:
                stream.seek(0, os.SEEK_END)
                stream.seek(max(0, stream.tell() - _STDERR_TAIL))
                return stream.read().decode("utf-8", "replace")[-sanitize.MAX_STDERR_CHARS :]
        except OSError:
            return ""

    def shutdown(self) -> None:
        """Ask the worker to exit, wait briefly for its pipe to close, then kill the whole group."""
        if self.alive:
            deadline = self.clock() + 2.0
            with contextlib.suppress(Exception):
                self.send(encode_frame("shutdown", self._next_seq(), {}, limits=self.config.limits), deadline)
            with contextlib.suppress(Exception):
                while True:  # drain until EOF (worker exited) or the deadline
                    self._wait(self._rfd, select.POLLIN, deadline)
                    if not os.read(self._rfd, 1 << 16):
                        break
        if self.proc is not None and (self.alive or self.private is not None):
            self.kill()

    # -- deadline-bound framed IO -------------------------------------------

    def _wait(self, fd: int, event: int, deadline: float | None) -> None:
        self._poll.register(fd, event)
        try:
            timeout = None if deadline is None else (deadline - self.clock()) * 1000.0
            if timeout is not None and timeout <= 0:
                raise DeadlineExceeded
            ready = self._poll.poll(timeout)
        finally:
            self._poll.unregister(fd)
        if not ready:
            raise DeadlineExceeded

    def send(self, data: bytes, deadline: float | None) -> None:
        view = memoryview(data)
        while view:
            self._wait(self._wfd, select.POLLOUT, deadline)
            try:
                written = os.write(self._wfd, view[: 1 << 16])
            except BlockingIOError:
                continue
            except (BrokenPipeError, ConnectionResetError):
                raise WorkerGone("worker closed its stdin") from None
            view = view[written:]

    def _read_exact(self, count: int, deadline: float | None) -> bytes:
        chunks, got = [], 0
        while got < count:
            self._wait(self._rfd, select.POLLIN, deadline)
            try:
                chunk = os.read(self._rfd, min(count - got, 1 << 20))
            except BlockingIOError:
                continue
            if not chunk:
                raise StreamClosed(got)
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def receive(self, deadline: float | None):
        try:
            frame = read_frame(lambda count: self._read_exact(count, deadline), self.config.limits)
        except TruncatedFrame as error:
            raise WorkerGone(str(error)) from None
        return expect_type(frame, WORKER_TO_SUPERVISOR)


# ---------------------------------------------------------------------------
# result validation: mirrors execution/runner.py (same helpers, same ordering)
# ---------------------------------------------------------------------------

_PLAN_KEYS = {
    "kind",
    "status",
    "stage",
    "precision",
    "work",
    "stage_work",
    "stage_spans",
    "resource_peaks",
    "diagnostics",
    "malformed",
    "spans",
    "candidate_reported_ms",
}
_FIELD_KEYS = {"kind", "field_kind", "field_count", "spans", "candidate_reported_ms"}
_ERROR_KEYS = {"error_type", "error", "phase", "fatal"}
_SPAN_KEYS = {"request_build_ms", "call_ms", "sync_ms", "pack_ms"}
_READY_KEYS = {"role", "ipc", "python", "numpy", "rlimits", "pdeathsig", "isolation_probe"}
_LOADED_KEYS = {"kind", "loaded_files", "has_synchronize", "load_ms", "sys_modules_top_level", "forbidden_modules"}
_PROBE_KEYS = {"pid", "ppid", "uid", "root_entries", "net_devices", "cwd"}


def _validate_startup(ready: Mapping[str, Any], loaded: Mapping[str, Any]) -> None:
    """Shape-check the two startup frames; everything recorded later comes from these checked values."""
    expect_body(ready, _READY_KEYS)
    expect_body(loaded, _LOADED_KEYS)
    probe = ready["isolation_probe"]
    if (
        ready["ipc"] != IPC_PROTOCOL
        or not isinstance(probe, dict)
        or set(probe) != _PROBE_KEYS
        or not isinstance(ready["rlimits"], dict)
        or not isinstance(ready["pdeathsig"], bool)
        or not all(isinstance(ready[k], str) for k in ("role", "python", "numpy"))
        or any(isinstance(probe[k], bool) or not isinstance(probe[k], int) for k in ("pid", "ppid", "uid"))
        or not (
            probe["root_entries"] is None
            or (isinstance(probe["root_entries"], list) and all(isinstance(x, str) for x in probe["root_entries"]))
        )
        or not (
            probe["net_devices"] is None
            or (isinstance(probe["net_devices"], list) and all(isinstance(x, str) for x in probe["net_devices"]))
        )
        or "core" not in ready["rlimits"]
    ):
        raise ProtocolError("malformed worker attestation")
    if (
        loaded["kind"] != "loaded"
        or not isinstance(loaded["has_synchronize"], bool)
        or not isinstance(loaded["loaded_files"], list)
        or len(loaded["loaded_files"]) > 64
        or not all(isinstance(x, str) and len(x) <= 1024 for x in loaded["loaded_files"])
        or not isinstance(loaded["sys_modules_top_level"], list)
        or len(loaded["sys_modules_top_level"]) > 4096
        or not all(isinstance(x, str) and len(x) <= 128 for x in loaded["sys_modules_top_level"])
        or not isinstance(loaded["forbidden_modules"], list)
        or not all(isinstance(x, str) and len(x) <= 128 for x in loaded["forbidden_modules"])
    ):
        raise ProtocolError("malformed worker load report")
    _nonneg(loaded["load_ms"])


def _nonneg(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ProtocolError("timing values must be finite nonnegative numbers")
    return float(value)


def _check_result(frame, role: str):
    """Strict validation of a worker result; violations are protocol errors (fatal for the worker).

    Every candidate-controlled string and table is bounded here (see :mod:`passage_eval.supervision.sanitize`).
    """
    if frame.type == "error":
        expect_body(frame.body, _ERROR_KEYS)
        if not all(isinstance(frame.body[k], str) for k in ("error_type", "error", "phase")) or not isinstance(
            frame.body["fatal"], bool
        ):
            raise ProtocolError("malformed error frame")
        frame.body["error_type"] = frame.body["error_type"][: sanitize.MAX_ERROR_TYPE_CHARS]
        frame.body["error"] = frame.body["error"][: sanitize.MAX_ERROR_CHARS]
        frame.body["phase"] = frame.body["phase"][:32]
        return
    if frame.type != "result":
        raise ProtocolError(f"unexpected reply type {frame.type!r}")
    body = frame.body
    if role == "predictor":
        expect_body(body, _FIELD_KEYS)
        count = body["field_count"]
        if (
            body["kind"] != "field"
            or body["field_kind"] not in {"array", "sequence"}
            or isinstance(count, bool)
            or not isinstance(count, int)
            or not 1 <= count <= 8
            or set(frame.arrays) != {f"field_{i}" for i in range(count)}
            or (body["field_kind"] == "array" and count != 1)
        ):
            raise ProtocolError("malformed predictor result")
    else:
        expect_body(body, _PLAN_KEYS)
        if body["kind"] != "plan" or set(frame.arrays) != {"path"}:
            raise ProtocolError("malformed plan result")
        for key in ("status", "stage", "precision"):
            if not isinstance(body[key], str) or len(body[key]) > sanitize.MAX_LABEL_CHARS:
                raise ProtocolError("malformed plan result")
        for key in ("work", "stage_work", "stage_spans", "resource_peaks", "diagnostics"):
            if not isinstance(body[key], dict):
                raise ProtocolError("malformed plan result")
        if body["malformed"] is not None and not isinstance(body["malformed"], str):
            raise ProtocolError("malformed plan result")
        try:
            for key in ("work", "stage_spans", "resource_peaks", "stage_work"):
                sanitize.check_table(key, body[key])
            for counters in body["stage_work"].values():
                sanitize.check_table("stage_work entry", counters)
        except ValueError as error:
            raise ProtocolError(str(error)) from None
        if body["malformed"] is not None:
            body["malformed"] = body["malformed"][: sanitize.MAX_ERROR_CHARS]
    _nonneg(body["candidate_reported_ms"])
    if not isinstance(body["spans"], dict) or set(body["spans"]) != _SPAN_KEYS:
        raise ProtocolError("malformed spans")
    for value in body["spans"].values():
        _nonneg(value)


def validate_plan(
    request: PlanRequest, control: Mapping[str, Any], body: Mapping[str, Any], raw: np.ndarray, outcome: dict[str, Any]
) -> None:
    """Legality, physical recost, work and budget accounting for one candidate reply.

    This follows ``execution/runner.py`` step by step (``validated_raw_path`` -> ``path_status`` ->
    ``path_cost`` -> ``_work`` -> ``_budget_status``) with the same helpers. ``outcome`` is updated
    progressively, in the runner's order, so that an exception leaves exactly the partial state the
    in-process runner would have recorded (see the parity tests).
    """
    diagnostics = dict(body["diagnostics"])
    malformed = body["malformed"]
    result = PlanResult(
        raw,
        status=body["status"],
        stage=body["stage"],
        work=dict(body["work"]),
        stage_work=body["stage_work"],
        precision=body["precision"],
        diagnostics=diagnostics,
        stage_spans=body["stage_spans"],
        resource_peaks=body["resource_peaks"],
    )
    outcome["fallback"] = result.stage == "fallback" or "fallback" in result.stage_work
    path = validated_raw_path(raw)
    connected, violated, obstacle_cells = path_status(
        path, start=request.start, goal=request.goal, obstacles=request.obstacles
    )
    outcome.update(connected=connected, violated=violated, obstacle_cells=obstacle_cells)
    recost = path_cost(request.elevation, path, weight=request.cost_weight, cost_model=request.cost_model)
    outcome.update(recost=float(recost), legal=bool(connected and not violated and np.isfinite(recost)))
    outcome.update(
        stage=result.stage,
        stage_work=result.stage_work,
        stage_spans=result.stage_spans,
        resource_peaks=result.resource_peaks,
        precision=result.precision,
        diagnostics=diagnostics,
    )
    if malformed is not None:
        diagnostics["malformed_raw_output"] = malformed
    work = _work(result)
    outcome["work"] = work
    status = result.status
    if status == "ok" and not outcome["legal"]:
        status = "no_path" if raw.size == 0 and malformed is None else "invalid_path"
    fallback_used = outcome["fallback"]
    if (
        control.get("variant") in {"direct", "bounded_repair", "prediction_free"}
        and fallback_used
        or control.get("variant") == "direct"
        and (result.stage != "primary" or "repair" in result.stage_work)
    ):
        diagnostics["component_control_violation"] = "undeclared_repair_or_fallback"
        status = "runtime_error"
    exceeded, missing = _budget_status(request.budget, work)
    if exceeded:
        diagnostics["exceeded_work_budgets"] = exceeded
        if status == "ok":
            status = "budget_exhausted"
    if missing:
        diagnostics["unobservable_work_budgets"] = missing
        if status == "ok":
            status = "runtime_error"
    outcome["status"] = status


def _error_status(error_type: str) -> str:
    if error_type == "MemoryError":
        return "oom"
    if error_type in {"CancelledError"}:
        return "cancelled"
    return "runtime_error"


@dataclass
class Hop:
    role: str
    kind: str  # ok | worker_error | preempted | crashed | protocol_error | skipped_after_deadline
    # | contract_violation
    service_ms: float
    frame: Any = None
    detail: dict = field(default_factory=dict)
    finished: float = 0.0  # supervisor clock at the end of the hop, taken before any kill/cleanup

    # kind: ok | worker_error | preempted | crashed | protocol_error | skipped_after_deadline | contract_violation


_COMPLETION_MODE = {
    "ok": "worker_replied",
    "worker_error": "worker_replied",
    "preempted": "worker_killed_after_deadline",
    "crashed": "worker_exited",
    "protocol_error": "worker_killed_after_protocol_error",
    "skipped_after_deadline": "worker_replied",
    "contract_violation": "worker_replied_field_contract_violation",
}


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


class SupervisedRunner:
    """Owns the workers of one seed: start, baseline, passes, request loop, preemption, restart."""

    def __init__(
        self,
        config: SupervisorConfig,
        workers: Mapping[str, WorkerSpec],
        *,
        bound: environment_mod.BoundSet,
        identity: dict[str, Any],
        identity_check: Callable[[], dict[str, Any]],
        split: SplitSpec | None = None,
        clock: Callable[[], float] = time.perf_counter,
        counters_source: str = COUNTERS_SELF_REPORTED,
        scrub: Sequence[str] = (),
        views: Mapping[str, environment_mod.RoleBinds] | None = None,
    ):
        self.config, self.clock, self.identity, self.identity_check, self.split = (
            config,
            clock,
            identity,
            identity_check,
            split,
        )
        self.bound = bound
        full = environment_mod.RoleBinds(tuple(bound.binds), tuple(config.candidate_paths), narrowed=False)
        self.views = {role: (views or {}).get(role, full) for role in workers}
        self.workers = {
            role: WorkerProcess(spec, config, self.views[role].binds, self.views[role].import_paths, clock)
            for role, spec in workers.items()
        }
        self.events: list[dict[str, Any]] = []
        self.pass_log: list[dict[str, Any]] = []
        self.ready: dict[str, dict[str, Any]] = {}
        self.baseline: dict[str, list[dict[str, Any]]] = {}
        self.observed_payload_bytes: dict[str, set[int]] = {role: set() for role in workers}
        self.restarts = 0
        self.cap: dict[str, Any] | None = None
        self.tasks_marked_not_executed = 0
        self.secret = secrets.token_bytes(32)
        self.control: dict[str, Any] = {}
        self.isolation_record: dict[str, Any] = {}
        self.counters_source = counters_source
        self.scrub = tuple(scrub)
        self.facts: dict[str, Any] = {}
        self.profile = timing.get_profile(config.timing_profile)
        self.boundary = self.profile.boundary or TIMING_BOUNDARY_ID
        # Warm-up state of the current pass (set by set_warm_plan, cleared by begin_pass / clear_warm_plan).
        self._warm_plan: tuple[Sequence[PlanRequest], int, str] | None = None
        self._warm_attempted: dict[str, int] | None = None  # worker generations of the last warm-up attempt
        self._warm_complete: dict[str, int] | None = None  # worker generations of the last complete warm-up
        self._warming = False  # recursion guard: a warm-up request never triggers a re-warm
        self._rewarm_failed = False  # a re-warm of the current pass failed: the pass is not re-warmed again
        self.rewarms = 0
        #: Restarts of workers lost during a re-warm. They are not charged to ``max_restarts``; at most one re-warm
        #: fails per pass, so at most one restart per role and warm pass is made outside the cap.
        self.rewarm_restarts = 0
        self.pass_rewarms: list[dict[str, Any]] = []
        #: Campaign session mode (:mod:`passage_eval.supervision.session`): dynamic request admission and session
        #: invalidation. ``None`` keeps the historical behaviour of every other caller.
        self.session: session_mod.SessionPolicy | None = None
        self.initial_start_ms: dict[str, float] = {}
        #: The first initial warm-up request of the last warm-up (cold start of a session): latency and status.
        self.first_warmup_request: dict[str, Any] | None = None

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self):
        try:
            for role in self.workers:
                self.initial_start_ms[role] = self._start_worker(role, first=True)
        except BaseException:
            self.close()
            raise
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        for worker in self.workers.values():
            with contextlib.suppress(Exception):
                worker.shutdown()

    def _start_worker(self, role: str, *, first: bool) -> float:
        worker = self.workers[role]
        started = self.clock()
        ready = worker.start()
        try:
            identity_mod.check_loaded_files(role, ready["loaded_files"], self.identity)
            self._check_isolation(role, ready)
        except (identity_mod.IdentityError, IntegrityError):
            worker.kill()
            raise
        self.ready[role] = ready
        return 1000.0 * (self.clock() - started)

    def _check_isolation(self, role: str, ready: Mapping[str, Any]) -> None:
        probe = ready["isolation_probe"]
        if self.config.isolation.level == "bwrap":
            allowed_roots = {
                "bin",
                "sbin",
                "lib",
                "lib32",
                "lib64",
                "libx32",
                "usr",
                "etc",
                "dev",
                "proc",
                "tmp",
                "home",
            } | {p.parts[1] for p in self._bind_paths(role) if len(p.parts) > 1}
            if (
                probe["net_devices"] is None
                or set(probe["net_devices"]) - {"lo"}
                or probe["root_entries"] is None
                or set(probe["root_entries"]) - allowed_roots
                or probe["pid"] > 64
            ):
                raise IntegrityError(f"{role} worker does not look sandboxed although bwrap was requested: {probe}")
        if ready["rlimits"]["core"] != [0, 0]:
            raise IntegrityError(f"{role} worker did not disable core dumps: {ready['rlimits']}")

    def _bind_paths(self, role: str) -> list[Path]:
        return [Path(b.target) for b in self.workers[role].binds]

    def _event(self, **row: Any) -> None:
        self.events.append(row)

    def ensure_ready(self) -> tuple[float, dict[str, Any] | None]:
        """Restart workers that died, charged against the restart cap.

        Returns ``(restart_ms, failure)``. ``failure`` is ``None`` when every worker is alive, else a dict with a
        ``reason``: ``worker_restart_cap`` (sticky: every later request is also not executed) or
        ``worker_restart_failed`` (this request only; the next request tries again and counts again).
        Neither aborts the run. Identity or isolation contradictions still do (:class:`IntegrityError`). In session
        mode both invalidate the session instead (:class:`~passage_eval.supervision.session.SessionInvalid`): the cap
        never decides the status of a task, and a worker that cannot start is never imputed to one.
        """
        total = 0.0
        for role, worker in self.workers.items():
            if worker.alive:
                continue
            if self.cap is not None:
                return total, self.cap
            if self.restarts >= self.config.max_restarts:
                self.cap = {
                    "reason": "worker_restart_cap",
                    "role": role,
                    "detail": f"more than {self.config.max_restarts} worker restarts",
                }
                self._event(event="restart_cap_reached", role=role, max_restarts=self.config.max_restarts)
                if self.session is not None:
                    raise session_mod.SessionInvalid("worker_restart_cap", self.cap["detail"])
                return total, self.cap
            self.restarts += 1
            try:
                elapsed = self._start_worker(role, first=False)
            except IntegrityError:
                raise
            except SupervisionError as error:
                text = sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub)
                self._event(event="restart_failed", role=role, error=text)
                if self.session is not None:
                    raise session_mod.SessionInvalid("worker_restart_failed", text) from None
                return total, {"reason": "worker_restart_failed", "role": role, "detail": text}
            total += elapsed
            self._event(event="restart", role=role, generation=worker.generation, restart_ms=elapsed)
        return total, None

    def begin_pass(self, pass_id: str) -> float:
        """Kill every worker and start fresh ones, outside every request's boundary. Not charged to the cap."""
        self.clear_warm_plan()
        if self.cap is not None:
            return 0.0
        started = self.clock()
        for worker in self.workers.values():
            if worker.alive:
                worker.kill()
        for role, worker in self.workers.items():
            began = self.clock()
            try:
                self._start_worker(role, first=False)
            except IntegrityError:
                raise
            except SupervisionError as error:
                text = sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub)
                self._event(event="pass_start_failed", pass_id=pass_id, role=role, error=text)
                if self.session is not None:
                    raise session_mod.SessionInvalid("worker_start_failed", text) from None
                continue
            self._event(
                event="pass_start",
                pass_id=pass_id,
                role=role,
                generation=worker.generation,
                start_ms=1000.0 * (self.clock() - began),
            )
        return 1000.0 * (self.clock() - started)

    def _generations(self) -> dict[str, int]:
        return {role: worker.generation for role, worker in self.workers.items()}

    def set_warm_plan(self, requests: Sequence[PlanRequest], count: int, scope: str) -> None:
        """Declare the warm-up of the current pass, which a re-warm profile replays after a restart."""
        self.clear_warm_plan()
        self._warm_plan = (tuple(requests), count, scope)

    def clear_warm_plan(self) -> None:
        self._warm_plan = None
        self._warm_attempted = None
        self._warm_complete = None
        self._rewarm_failed = False
        self.pass_rewarms = []

    def run_warmup(
        self, requests: Sequence[PlanRequest], count: int, scope: str, *, kind: str = "warmup"
    ) -> dict[str, Any]:
        """Untimed warm-up in the pass's workers, on inputs disjoint from the roster. Never recorded as a task.

        The summary counts the requests that returned ``ok`` and says whether the warm-up is complete: every
        requested warm-up returned ``ok`` and every worker is still alive. An incomplete warm-up is recorded
        by a ``<kind>_failed`` event. The initial warm-up of a pass runs every request in every profile, so its
        restarts are the same, and counted the same way, whatever the profile. A re-warm (``kind="rewarm"``)
        stops at its first request that does not return ``ok`` or that loses a worker (proposed policy, not
        sealed), so that it never needs a restart of its own. In session mode the initial warm-up stops the same way
        (its failure invalidates the session, so it never consumes the restart cap), and every warm-up request is
        admitted dynamically.
        """
        statuses: Counter = Counter()
        began = self.clock()
        done = succeeded = 0
        token_scope = f"warmup|{scope}" if kind == "warmup" else f"{kind}-{self.rewarms}|{scope}"
        stop_at_failure = kind == "rewarm" or self.session is not None
        point = "before_rewarm_request" if kind == "rewarm" else "before_warmup_request"
        if kind == "warmup":
            self.first_warmup_request = None
        self._warming = True
        try:
            for index in range(count):
                if self.cap is not None:
                    break
                request = requests[index % len(requests)]
                record, _ = self.run_request(
                    request,
                    request_token(self.secret, index, scope=token_scope),
                    0,
                    "warmup-untimed",
                    pass_id=scope,
                    admission_point=point,
                )
                if kind == "warmup" and index == 0:
                    self.first_warmup_request = {
                        "latency_ms": record[self.profile.latency_field],
                        "service_wall_ms": record["service_wall_ms"],
                        "status": record["status"],
                        "input_sha256": _input_digest(request),
                    }
                statuses[record["status"]] += 1
                done += 1
                if record["status"] == "ok":
                    succeeded += 1
                if stop_at_failure and (
                    record["status"] != "ok" or not all(worker.alive for worker in self.workers.values())
                ):
                    break
        finally:
            self._warming = False
        alive = all(worker.alive for worker in self.workers.values())
        complete = done == count and succeeded == count and alive
        generations = self._generations()
        self._warm_attempted = generations
        self._warm_complete = generations if complete else None
        summary = {
            "requests": done,
            "requested": count,
            "statuses": dict(statuses),
            "wall_ms_total": 1000.0 * (self.clock() - began),
            "succeeded": succeeded,
            "complete": complete,
            "workers_alive_after": alive,
        }
        self._event(event=kind, pass_id=scope, **summary)
        if not complete:
            self._event(
                event=f"{kind}_failed", pass_id=scope, requests=done, succeeded=succeeded, statuses=dict(statuses)
            )
        return summary

    def _rewarm_due(self) -> bool:
        return (
            self.profile.rewarm_after_restart
            and self._warm_plan is not None
            and not self._warming
            and not self._rewarm_failed
            and self._generations() != self._warm_attempted
        )

    def _rewarm(self) -> dict[str, Any]:
        """Replay the pass warm-up on restarted workers, outside every request's boundary.

        A failed re-warm ends re-warming for the pass: the same inputs would fail again on the next restarted
        worker, and every retry would cost a restart. The measured requests that follow carry
        ``warmup_complete_before_request = false``.
        """
        requests, count, scope = self._warm_plan
        self.rewarms += 1
        summary = self.run_warmup(requests, count, scope, kind="rewarm")
        self.pass_rewarms.append(summary)
        if not summary["complete"]:
            self._rewarm_failed = True
        return summary

    def _restart_rewarm_losses(self) -> tuple[float, dict[str, Any] | None]:
        """Restart the workers that a failed re-warm lost, outside the restart cap and the boundary.

        The re-warm is the supervisor's own action on inputs outside the roster, so its losses are not charged
        to ``max_restarts``: a warm-up input can then never turn a roster task into a ``worker_restart_cap``
        failure. The bound holds because a re-warm stops at its first loss and at most one re-warm fails per
        pass. A start that fails leaves the measured request unexecuted (``worker_restart_failed``); the next
        request retries through :meth:`ensure_ready`, counted as usual.
        """
        total = 0.0
        for role, worker in self.workers.items():
            if worker.alive:
                continue
            self.rewarm_restarts += 1
            try:
                elapsed = self._start_worker(role, first=False)
            except IntegrityError:
                raise
            except SupervisionError as error:
                text = sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub)
                self._event(event="rewarm_restart_failed", role=role, error=text, counted=False)
                if self.session is not None:
                    raise session_mod.SessionInvalid("worker_restart_failed", text) from None
                return total, {"reason": "worker_restart_failed", "role": role, "detail": text}
            total += elapsed
            self._event(
                event="rewarm_restart", role=role, generation=worker.generation, restart_ms=elapsed, counted=False
            )
        return total, None

    def admission_required_ms(self, request: PlanRequest) -> float:
        """The window that engaging ``request`` needs now: its reserve, plus a start for every dead worker."""
        assert self.session is not None
        dead = sum(not worker.alive for worker in self.workers.values())
        return session_mod.request_reserve_ms(
            wall_time_ms=request.budget.wall_time_ms,
            deadline_grace_ms=self.config.deadline_grace_ms,
            hard_timeout_ms=self.config.hard_timeout_ms,
            persistence_ms=self.session.persistence_ms,
        ) + dead * session_mod.start_reserve_ms(self.config.startup_timeout_s)

    def _admit(self, request: PlanRequest, point: str) -> None:
        """Dynamic request admission (session mode only): close the session when the window cannot hold the request."""
        if self.session is None:
            return
        required = self.admission_required_ms(request)
        remaining = self.session.remaining_ms()
        if remaining < required:
            self._event(event="session_closed", point=point, remaining_ms=remaining, required_ms=required)
            raise session_mod.SessionClosed(point, remaining, required)

    def _warm_state(self, rewarmed: bool, executed: bool) -> dict[str, Any]:
        """Per-record warm-up state of a re-warm profile (nothing for the legacy and v4 profiles)."""
        if not self.profile.rewarm_after_restart:
            return {}
        if self._warm_plan is None:
            complete = None
        else:
            complete = executed and self._warm_complete is not None and self._generations() == self._warm_complete
        return {"rewarmed_before_request": rewarmed, "warmup_complete_before_request": complete}

    # -- IPC baseline ----------------------------------------------------------

    def measure_baseline(self, role: str, payload_bytes: int) -> list[dict[str, Any]]:
        """Echo round trips of a payload of this size: full-size echo and tiny-reply echo.

        Measured exactly like a request (serialise ... deserialise). Recorded, never subtracted.
        """
        worker = self.workers[role]
        payload = np.zeros(max(1, payload_bytes), dtype=np.uint8)
        results = []
        for mode in ("full", "small"):
            samples = []
            for index in range(self.config.ipc_baseline_warmup + self.config.ipc_baseline_pings):
                deadline = self.clock() + self.config.startup_timeout_s
                started = self.clock()
                data = encode_frame(
                    "ping", worker.next_seq(), {"echo": mode}, {"payload": payload}, limits=self.config.limits
                )
                worker.send(data, deadline)
                frame = worker.receive(deadline)
                finished = self.clock()
                if frame.type != "echo" or (
                    mode == "full" and frame.arrays.get("payload", payload[:0]).nbytes != payload.nbytes
                ):
                    raise SupervisionError("IPC echo baseline returned an unexpected reply")
                if index >= self.config.ipc_baseline_warmup:
                    samples.append(1000.0 * (finished - started))
            q = np.percentile(samples, [5, 25, 50, 75, 95])
            results.append(
                {
                    "mode": "round_trip_same_size" if mode == "full" else "request_size_out_small_back",
                    "payload_bytes": int(payload.nbytes),
                    "n": len(samples),
                    "warmup_discarded": self.config.ipc_baseline_warmup,
                    "median_ms": float(q[2]),
                    "p05_ms": float(q[0]),
                    "p25_ms": float(q[1]),
                    "p75_ms": float(q[3]),
                    "p95_ms": float(q[4]),
                    "min_ms": float(min(samples)),
                    "max_ms": float(max(samples)),
                    "mean_ms": float(np.mean(samples)),
                    "std_ms": float(np.std(samples)),
                    "iqr_ms": float(q[3] - q[1]),
                }
            )
        return results

    # -- one hop -----------------------------------------------------------------

    def _hop(
        self,
        role: str,
        msg_type: str,
        body: Callable[[], dict],
        arrays: Mapping[str, np.ndarray],
        io_deadline: float,
        clock_start: float | None = None,
    ) -> Hop:
        worker = self.workers[role]
        started = self.clock() if clock_start is None else clock_start
        try:
            seq = worker.next_seq()
            try:
                data = encode_frame(msg_type, seq, body, arrays, limits=self.config.limits)
            except ValueError as error:  # our own request does not fit the configured limits
                raise SupervisionError(f"cannot encode the {role} request: {error}") from None
            try:
                worker.send(data, io_deadline)
                frame = worker.receive(io_deadline)
                finished = self.clock()
                if frame.seq != seq:
                    raise ProtocolError("reply sequence number does not match the request")
                _check_result(frame, role)
            except (DeadlineExceeded, WorkerGone, ProtocolError):
                raise
            except Exception as error:
                # Nothing a peer sends may abort the run: whatever went wrong while decoding or validating
                # its frame is that worker's protocol failure.
                raise ProtocolError(f"{type(error).__name__} while handling the reply") from None
        except DeadlineExceeded:
            finished = self.clock()
            detail = worker.kill()
            self._event(event="preempt_kill", role=role, kill_ms=detail["kill_ms"], returncode=detail.get("returncode"))
            return Hop(role, "preempted", 1000.0 * (finished - started), detail=detail, finished=finished)
        except WorkerGone as error:
            finished = self.clock()
            detail = worker.kill()
            detail["error"] = sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub)
            self._event(event="worker_gone", role=role, returncode=detail.get("returncode"))
            return Hop(role, "crashed", 1000.0 * (finished - started), detail=detail, finished=finished)
        except ProtocolError as error:
            finished = self.clock()
            detail = worker.kill()
            detail["error"] = sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub)
            self._event(event="protocol_error", role=role, error=detail["error"])
            return Hop(role, "protocol_error", 1000.0 * (finished - started), detail=detail, finished=finished)
        service_ms = 1000.0 * (finished - started)
        if frame.type == "error":
            if frame.body["fatal"]:
                detail = worker.kill()
                self._event(event="worker_fatal_error", role=role, phase=frame.body["phase"][:32])
                return Hop(role, "worker_error", service_ms, frame, detail, finished)
            return Hop(role, "worker_error", service_ms, frame, finished=finished)
        return Hop(role, "ok", service_ms, frame, finished=finished)

    # -- one request ---------------------------------------------------------------

    def _unexecuted(
        self,
        request: PlanRequest,
        repeat: int,
        series: str,
        failure: Mapping[str, Any],
        restart_ms: float,
        pass_id: str | None,
        pass_start_ms: float,
        warm_state: Mapping[str, Any] | None = None,
    ) -> tuple[dict, np.ndarray]:
        """A terminal record for a task that was never sent to a worker (restart cap or failed restart).

        The task is a ``runtime_error``. Its latency is *charged*, not measured: the wall budget, or the hard
        timeout when there is none, so that a failure can never improve a latency statistic.
        """
        raw = np.empty((0, 2), dtype=np.float64)
        budget_ms = request.budget.wall_time_ms
        charged = float(budget_ms if budget_ms is not None else self.config.hard_timeout_ms)
        reason = failure["reason"]
        self.tasks_marked_not_executed += 1
        diagnostics = sanitize.cap_json(
            {
                "error_type": "WorkerRestartCap" if reason == "worker_restart_cap" else "WorkerRestartFailed",
                "reason": reason,
                "error": failure.get("detail", reason),
                "failure": "not_executed",
                "max_restarts": self.config.max_restarts,
                "timing_charged": "wall_budget" if budget_ms is not None else "hard_timeout",
            },
            self.scrub,
        )
        record = dict(
            sample_id=request.sample_id,
            repeat_index=repeat,
            series=series,
            status="runtime_error",
            route_hash=route_hash(raw),
            recost=float("nan"),
            connected_path=False,
            legal_path=False,
            obstacle_violation=False,
            obstacle_cells=0,
            total_wall_ms=charged,
            candidate_wall_ms=0.0,
            failed_attempt_ms=charged,
            drain_ms=0.0,
            deadline_ms=budget_ms,
            late_completion=False,
            safe_completion=True,
            stage="primary",
            stage_work={},
            work={},
            fallback=False,
            precision="unspecified",
            stage_spans={},
            resource_peaks={},
            diagnostics=diagnostics,
            timing_boundary=self.boundary,
            service_wall_ms=charged,
            candidate_reported_ms=0.0,
            validation_recost_ms=0.0,
            inter_hop_ms=0.0,
            hops=[],
            preempted=False,
            completion_mode=f"not_executed_{reason}",
            wire_policies=[],
            worker_restart_ms_before_request=restart_ms,
            pass_id=pass_id,
            pass_worker_start_ms=pass_start_ms,
            latency_charged_not_measured=True,
            counters_source=self.counters_source,
            **(warm_state or {}),
        )
        return json_safe(record), raw

    def run_request(
        self,
        request: PlanRequest,
        token: str,
        repeat: int,
        series: str,
        *,
        pass_id: str | None = None,
        pass_start_ms: float = 0.0,
        admission_point: str = "before_measured_request",
    ) -> tuple[dict, np.ndarray]:
        self._admit(request, admission_point)  # session mode: the window must hold the restarts and the request
        restart_ms, failure = self.ensure_ready()  # outside the boundary
        rewarmed = False
        if failure is None and self._rewarm_due():
            # Re-warm profile: the restarted workers replay the pass warm-up, outside the boundary. A failed
            # re-warm may have lost a worker: restart it outside the cap, and do not re-warm this pass again.
            rewarmed = True
            self._rewarm()
            more_ms, failure = self._restart_rewarm_losses()
            restart_ms += more_ms
            if failure is None:
                self._admit(request, admission_point)  # the re-warm used part of the window
        if failure is not None:
            return self._unexecuted(
                request,
                repeat,
                series,
                failure,
                restart_ms,
                pass_id,
                pass_start_ms,
                self._warm_state(rewarmed, executed=False),
            )
        warm_state = self._warm_state(rewarmed, executed=True)
        budget_ms = request.budget.wall_time_ms
        clock = self.clock
        t0 = clock()
        deadline_abs = None if budget_ms is None else t0 + budget_ms / 1000.0
        io_deadline = (
            t0 + self.config.hard_timeout_ms / 1000.0
            if deadline_abs is None
            else deadline_abs + self.config.deadline_grace_ms / 1000.0
        )

        def remaining_ms() -> float | None:
            return None if deadline_abs is None else max(0.0, 1000.0 * (deadline_abs - clock()))

        hops: list[Hop] = []
        raw = np.empty((0, 2), dtype=np.float64)
        wire_policy = [POLICY_ORDINARY_REQUEST]
        if self.split is None:
            message = redact_ordinary(
                request, token=token, remaining_wall_ms=None, metadata_allow=self.config.metadata_allow
            )
            self.observed_payload_bytes["planner"].add(message.payload_bytes)
            hop = self._hop(
                "planner",
                "plan",
                lambda: {**message.body, "remaining_wall_ms": remaining_ms()},
                message.arrays,
                io_deadline,
                clock_start=t0,
            )
            hops.append(hop)
            terminal = hop
        else:
            message = redact_ordinary(
                request, token=token, remaining_wall_ms=None, metadata_allow=self.config.metadata_allow
            )
            self.observed_payload_bytes["predictor"].add(message.payload_bytes)
            first = self._hop(
                "predictor",
                "predict",
                lambda: {**message.body, "remaining_wall_ms": remaining_ms()},
                message.arrays,
                io_deadline,
                clock_start=t0,
            )
            hops.append(first)
            terminal = first
            if first.kind == "ok":
                budget_left = remaining_ms()
                if budget_left is not None and budget_left <= 0:
                    terminal = Hop("decoder", "skipped_after_deadline", 0.0, finished=clock())
                else:
                    decoder_budget = (
                        request.budget if budget_left is None else replace(request.budget, wall_time_ms=budget_left)
                    )
                    fields = [first.frame.arrays[f"field_{i}"] for i in range(first.frame.body["field_count"])]
                    try:
                        wire = redact_decoder(
                            fields[0] if first.frame.body["field_kind"] == "array" else fields,
                            first.frame.body["field_kind"],
                            request.obstacles,
                            request.start,
                            request.goal,
                            decoder_budget,
                            grid_shape=request.elevation.shape,
                        )
                    except FieldContractError as error:
                        terminal = Hop(
                            "predictor",
                            "contract_violation",
                            first.service_ms,
                            first.frame,
                            {"error": str(error)},
                            finished=clock(),
                        )
                    else:
                        wire_policy.append(POLICY_DECODER_ONLY)
                        self.observed_payload_bytes["decoder"].add(wire.payload_bytes)
                        second = self._hop("decoder", "decode", lambda: dict(wire.body), wire.arrays, io_deadline)
                        hops.append(second)
                        terminal = second
        t1 = terminal.finished  # end of the last hop: kill/restart housekeeping is outside the boundary
        service_ms = 1000.0 * (t1 - t0)

        diagnostics: dict[str, Any] = {}
        work: dict = {}
        outcome: dict[str, Any] = dict(
            status="runtime_error",
            stage="primary",
            stage_work={},
            stage_spans={},
            resource_peaks={},
            precision="unspecified",
            fallback=False,
            connected=False,
            violated=False,
            obstacle_cells=0,
            legal=False,
            recost=float("nan"),
        )
        preempted = terminal.kind == "preempted"
        try:
            if terminal.kind == "ok":
                raw = terminal.frame.arrays["path"]
                validate_plan(request, self.control, terminal.frame.body, raw, outcome)
                diagnostics, work = outcome.pop("diagnostics"), outcome.pop("work")
            elif terminal.kind == "worker_error":
                body = terminal.frame.body
                outcome["status"] = _error_status(body["error_type"])
                diagnostics = {
                    "error_type": body["error_type"],
                    "error": body["error"],
                    "phase": body["phase"],
                    "worker_fatal": body["fatal"],
                }
            elif terminal.kind == "contract_violation":
                outcome["status"] = "runtime_error"
                diagnostics = {
                    "error_type": "FieldContractViolation",
                    "error": terminal.detail["error"],
                    "contract": FIELD_CONTRACT,
                    "phase": "predictor_field",
                }
            elif terminal.kind == "preempted":
                outcome["status"] = "timeout"
                diagnostics = {"preempted": True, "preempted_role": terminal.role}
            elif terminal.kind == "skipped_after_deadline":
                outcome["status"] = "timeout"
                diagnostics = {"decoder_skipped_after_deadline": True}
            else:
                outcome["status"] = "runtime_error"
                diagnostics = {
                    "error_type": "WorkerFailure",
                    "error": terminal.detail.get("error", terminal.kind),
                    "failure": terminal.kind,
                    "worker_returncode": terminal.detail.get("returncode"),
                    "stderr_tail": terminal.detail.get("stderr_tail", "")[-sanitize.MAX_STDERR_CHARS :],
                }
        except Exception as error:
            # Same as the runner: keep what was already computed (legality, recost, fallback flag),
            # drop the candidate's stage/work attribution.
            outcome.update(
                status="runtime_error",
                stage="primary",
                stage_work={},
                stage_spans={},
                resource_peaks={},
                precision="unspecified",
            )
            outcome.pop("diagnostics", None)
            outcome.pop("work", None)
            diagnostics = {
                "error_type": type(error).__name__,
                "error": sanitize.cap_text(error, sanitize.MAX_ERROR_CHARS, self.scrub),
            }
            work = {}
        diagnostics = sanitize.cap_json(diagnostics, self.scrub)
        t2 = clock()
        validation_ms = 1000.0 * (t2 - t1) if terminal.kind == "ok" else 0.0
        total_ms = service_ms + validation_ms

        status = outcome["status"]
        # The legacy profile charges the supervisor's validation and recost to the request; the service
        # profiles compare the service time alone and keep the total as a separate record.
        elapsed_ms = total_ms if self.profile.validation_in_boundary else service_ms
        late = budget_ms is not None and elapsed_ms > budget_ms
        if late and status != "timeout":
            diagnostics["status_before_deadline_policy"] = status
            status = "timeout"
        failed_attempt_ms = min(elapsed_ms, budget_ms) if late else elapsed_ms if status != "ok" else 0.0
        drain_ms = max(0.0, elapsed_ms - budget_ms) if late else 0.0
        reported = sum(
            float(h.frame.body["candidate_reported_ms"])
            for h in hops
            if h.frame is not None and h.frame.type == "result"
        )
        hop_rows = [
            {
                "role": h.role,
                "kind": h.kind,
                "service_wall_ms": h.service_ms,
                "candidate_reported_ms": (
                    float(h.frame.body["candidate_reported_ms"])
                    if h.frame is not None and h.frame.type == "result"
                    else None
                ),
                "worker_spans": (h.frame.body["spans"] if h.frame is not None and h.frame.type == "result" else None),
                "worker_generation": self.workers[h.role].generation if h.role in self.workers else None,
            }
            for h in hops
        ]
        precision = (
            self.split.precision
            if self.split is not None and outcome["precision"] == "unspecified"
            else outcome["precision"]
        )
        record = dict(
            sample_id=request.sample_id,
            repeat_index=repeat,
            series=series,
            status=status,
            route_hash=route_hash(raw),
            recost=outcome["recost"],
            connected_path=outcome["connected"],
            legal_path=outcome["legal"],
            obstacle_violation=outcome["violated"],
            obstacle_cells=outcome["obstacle_cells"],
            total_wall_ms=total_ms,
            candidate_wall_ms=reported,
            failed_attempt_ms=failed_attempt_ms,
            drain_ms=drain_ms,
            deadline_ms=budget_ms,
            late_completion=late,
            safe_completion=True,
            stage=outcome["stage"],
            stage_work=outcome["stage_work"],
            work=work,
            fallback=outcome["fallback"],
            precision=sanitize.cap_text(precision, sanitize.MAX_LABEL_CHARS, self.scrub),
            stage_spans=outcome["stage_spans"],
            resource_peaks=outcome["resource_peaks"],
            diagnostics=diagnostics,
            timing_boundary=self.boundary,
            service_wall_ms=service_ms,
            candidate_reported_ms=reported,
            validation_recost_ms=validation_ms,
            inter_hop_ms=max(0.0, service_ms - sum(h.service_ms for h in hops)),
            hops=hop_rows,
            preempted=preempted,
            completion_mode=_COMPLETION_MODE[terminal.kind],
            wire_policies=wire_policy,
            worker_restart_ms_before_request=restart_ms,
            pass_id=pass_id,
            pass_worker_start_ms=pass_start_ms,
            counters_source=self.counters_source,
            **warm_state,
        )
        return json_safe(record), raw


# ---------------------------------------------------------------------------
# run orchestration
# ---------------------------------------------------------------------------


def _stats_manifest(runner: SupervisedRunner) -> dict[str, Any]:
    return {role: sorted(sizes) for role, sizes in runner.observed_payload_bytes.items()}


def _worker_specs(spec, split: SplitSpec | None, seed: int | None) -> dict[str, WorkerSpec]:
    if split is None:
        return {
            "planner": WorkerSpec("planner", spec.inference, dict(spec.inference_kwargs), spec.inference_format, seed)
        }
    return {
        "predictor": WorkerSpec("predictor", split.predictor, dict(spec.inference_kwargs), "request", seed),
        "decoder": WorkerSpec("decoder", split.decoder, dict(split.decoder_kwargs), "request", seed),
    }


def _public_bench_spec(spec) -> dict[str, Any]:
    payload = json.loads(json.dumps(spec.to_dict()))
    payload["dataset_path"] = None  # the dataset directory holds the reference labels
    return payload


@dataclass
class RuntimeEnvironment:
    """Facts about the worker interpreter and everything the sandbox binds, established without running candidate code.

    The interpreter is probed inside the sandbox the worker will use (``bwrap``), or at the worker's own
    (unsandboxed) level under ``process`` isolation. ``bound`` holds the staged candidate trees and the hashed
    interpreter environment; ``close()`` removes the staging area.
    """

    probe: Mapping[str, Any]
    layout: sandbox.RuntimeLayout
    bound: environment_mod.BoundSet
    protected: tuple[str, ...]
    bwrap_version: str | None
    uncovered_paths: tuple[str, ...] = ()
    scrub: tuple[str, ...] = ()

    @property
    def runtime_binds(self) -> tuple[Path, ...]:
        return tuple(Path(b.target) for b in self.bound.binds if b.role == "runtime")

    def close(self) -> None:
        self.bound.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def inspect_environment(
    config: SupervisorConfig, entries: Sequence[str], extra_protected: Sequence[str] = ()
) -> RuntimeEnvironment:
    sys.dont_write_bytecode = True
    level = config.isolation.level
    protected = sandbox.default_protected_paths([*config.isolation.protected_paths, *extra_protected])
    version = sandbox.bwrap_version(config.isolation.bwrap_path) if level == "bwrap" else None
    layout = sandbox.static_runtime_layout(
        config.worker_python, config.isolation.runtime_binds, trusted_roots=sandbox.evaluator_roots()
    )
    cache = environment_mod.tree_cache_from(config.cache_setting())
    bound = environment_mod.build_bound_set(
        layout,
        isolation=level,
        declared=[*config.candidate_paths, *config.isolation.ro_binds],
        protected=protected,
        cache=cache,
        staging_base=config.staging_dir,
    )
    try:
        probe = sandbox.probe_interpreter(
            config.worker_python,
            isolation=config.isolation,
            binds=bound.binds,
            candidate_paths=[str(Path(p).resolve()) for p in config.candidate_paths],
            top_levels=identity_mod.top_level_names(entries),
        )
        covered = bound.covered_roots() + ([] if level == "bwrap" else [Path("/usr")])
        uncovered = sandbox.uncovered_probe_paths(probe, covered)
        if uncovered and level == "bwrap":
            raise sandbox.SandboxError(
                "the worker interpreter reports locations that no bind covers: "
                f"{[str(p) for p in uncovered[:6]]}; declare them with --runtime-bind (environment) or "
                f"--ro-bind (candidate files)"
            )
    except BaseException:
        bound.close()
        raise
    return RuntimeEnvironment(
        probe=probe,
        layout=layout,
        bound=bound,
        protected=tuple(protected),
        bwrap_version=version,
        uncovered_paths=tuple(str(p) for p in uncovered),
    )


def _roles(seeded, split: SplitSpec | None) -> list[tuple[str, str, Mapping[str, Any]]]:
    if split is None:
        return [("planner", seeded.inference, seeded.inference_kwargs)]
    return [("predictor", split.predictor, seeded.inference_kwargs), ("decoder", split.decoder, split.decoder_kwargs)]


def _identity_function(seeded, split: SplitSpec | None, probe: Mapping[str, Any], bound: environment_mod.BoundSet):
    def measure():
        components = {
            role: identity_mod.measure_component(entry, kwargs, probe, bound.view)
            for role, entry, kwargs in _roles(seeded, split)
        }
        label = seeded.inference if split is None else f"split:{split.predictor}+{split.decoder}"
        bundle = identity_mod.runtime_bundle(seeded.runtime_bundle_manifest, bound.view)
        identity = identity_mod.build_identity(
            components,
            entry_label=label,
            bundle=bundle,
            declared_source_sha256=seeded.candidate_source_sha256,
            bound_trees=bound.manifest(),
            binding=bound.binding,
        )
        return identity, components, bundle

    return measure


def _evaluator_source_hash(probe: Mapping[str, Any], view: Callable[[Path], Path]) -> str:
    """The evaluator source fingerprint (same algorithm as ``cli._source_hash``) of the worker's own packages."""
    digest = hashlib.sha256()
    for name in ("passage", "passage_eval"):
        origin = probe["modules"][name]["origin"]
        root = view(Path(origin).resolve().parent)
        for path in sorted(root.rglob("*.py")):
            digest.update(f"{name}/{path.relative_to(root)}".encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _evaluator_extra_files(probe: Mapping[str, Any], view: Callable[[Path], Path]) -> dict[str, Any]:
    """Non-source files of the worker's evaluator packages that the supervisor's own packages do not hold identically.

    The source fingerprint covers ``*.py`` only; an extension module, a sourceless ``.pyc`` or a ``.pth`` file in the
    worker's copy would be imported without being part of it (:func:`treehash.unmatched_non_source_files`).
    """
    import passage
    import passage_eval

    own = {
        "passage": Path(passage.__file__).resolve().parent,
        "passage_eval": Path(passage_eval.__file__).resolve().parent,
    }
    report: dict[str, Any] = {}
    for name in ("passage", "passage_eval"):
        worker_root = view(Path(probe["modules"][name]["origin"]).resolve().parent)
        if Path(os.path.realpath(worker_root)) == own[name]:
            report[name] = {
                "same_directory_as_the_supervisor": True,
                "checked": 0,
                "unmatched_count": 0,
                "unmatched": [],
            }
        else:
            report[name] = dict(
                treehash.unmatched_non_source_files(worker_root, own[name]), same_directory_as_the_supervisor=False
            )
    return report


def check_purpose_static(
    config: SupervisorConfig, spec, split: SplitSpec | None, roster: "RosterInput | None"
) -> list[dict]:
    """Requirements of ``verified-candidate-run`` that can be checked before anything is loaded or started.

    Nothing here reads the spec's family or any other spec text: what makes the counters trustworthy is a digest the
    operator pinned, either of the decoder behind a predictor (split mode) or of the planner itself (planner mode).
    """
    checks = [{"check": "purpose", "value": config.purpose}]
    if config.require_split_procedure_pin and (split is None or not config.split_procedure_allow_sha256):
        raise PurposeError("split procedure pin required: use split mode and supply --split-procedure-allow-sha256")
    if config.purpose != PURPOSE_VERIFIED:
        return checks
    problems = []
    if config.isolation.level != "bwrap":
        problems.append("process isolation runs candidate code unsandboxed (use --isolation bwrap)")
    if roster is None:
        problems.append("a pinned roster is required (--roster, --roster-sha256, --dataset-root)")
    if split is not None and not config.decoder_allow_sha256:
        problems.append("split mode needs an operator decoder allow-list (--decoder-allow-sha256)")
    if split is None and not config.planner_allow_sha256:
        problems.append(
            "planner mode needs an operator planner allow-list (--planner-allow-sha256); otherwise use split "
            "mode with --decoder-allow-sha256 (the family named in the spec is not consulted)"
        )
    if not config.environment_allow_sha256:
        problems.append("an operator interpreter-environment allow-list is required (--environment-allow-sha256)")
    if problems:
        raise PurposeError("verified-candidate-run refused: " + "; ".join(problems))
    checks += [
        {"check": "isolation_is_bwrap", "passed": True},
        {"check": "roster_pinned", "passed": True},
        {"check": "decoder_allow_list_present", "passed": split is not None, "applies": split is not None},
        {"check": "planner_allow_list_present", "passed": split is None, "applies": split is None},
        {"check": "environment_allow_list_present", "passed": True},
        {"check": "no_tree_hash_cache", "passed": True},
        {"check": "default_forbidden_modules_enforced", "passed": True},
    ]
    return checks


def _pin_decision(
    config: SupervisorConfig, spec, split: SplitSpec | None, probe: Mapping[str, Any], view: Callable[[Path], Path]
):
    """``(role, record, counters_source)`` for the one entry the operator may pin; ``role`` is None when it is not pinned.

    Split mode pins the decoder (``--decoder-allow-sha256``). Planner mode pins the planner (``--planner-allow-sha256``).
    A namespace-package decoder is always refused; a namespace-package planner is refused whenever a pin applies.
    """
    if split is not None:
        record = identity_mod.entry_tree(split.decoder, probe, view, role="decoder_entry")
        if record.sha256 in config.decoder_allow_sha256:
            return "decoder", record, COUNTERS_OPERATOR
        return None, record, COUNTERS_SELF_REPORTED
    if config.planner_allow_sha256 or config.purpose == PURPOSE_VERIFIED:
        record = identity_mod.entry_tree(spec.inference, probe, view, role="planner_entry")
        if record.sha256 in config.planner_allow_sha256:
            return "planner", record, COUNTERS_PLANNER
        return None, record, COUNTERS_SELF_REPORTED
    return None, None, COUNTERS_SELF_REPORTED


def _environment_pin(bound: environment_mod.BoundSet, allow: Sequence[str]) -> dict[str, Any]:
    rows = [
        {
            "path": record.path,
            "sha256": record.sha256,
            "hash_source": record.hash_source,
            "in_allow_list": record.sha256 in allow,
        }
        for record in sorted(bound.environment, key=lambda r: r.path)
    ]
    return {
        "allow_list": list(allow),
        "trees": rows,
        "all_trees_allow_listed": bool(rows) and all(r["in_allow_list"] for r in rows),
        "rule": "every interpreter-environment tree the sandbox binds, /usr included, must have its digest in the list",
    }


def procedure_limits(
    budget: BudgetSpec, config: SupervisorConfig, *, schema: str = PROCEDURE_LIMITS_SCHEMA
) -> dict[str, Any]:
    """Every bound that applies to a supervised request, as hashed into the procedure identity.

    The complete budget (every field, ``None`` where unlimited), the timing profile and its boundary, the
    deadline grace, the hard timeout, the worker start-up timeout, the restart cap and the kill wait. Schema v2
    (the default, written by new runs) adds the rlimits each worker applies to itself (a CPU-time or address-space
    limit kills the worker, a file-size limit fails a write) and the IPC frame limits (an oversized request or
    reply is a protocol failure). ``schema=PROCEDURE_LIMITS_SCHEMA_V1`` returns the earlier block, used only by
    the v2 split-procedure pin.
    """
    if not isinstance(budget, BudgetSpec):
        raise TypeError("budget must be a BudgetSpec")
    if schema not in (PROCEDURE_LIMITS_SCHEMA, PROCEDURE_LIMITS_SCHEMA_V1):
        raise ValueError(f"unknown procedure limits schema {schema!r}")
    profile = timing.get_profile(config.timing_profile)
    limits = {
        "schema": schema,
        "budget": budget.to_dict(),
        "timing_profile": profile.name,
        "timing_boundary": profile.boundary or TIMING_BOUNDARY_ID,
        "latency_field": profile.latency_field,
        "rewarm_after_restart": profile.rewarm_after_restart,
        "deadline_grace_ms": config.deadline_grace_ms,
        "hard_timeout_ms": config.hard_timeout_ms,
        "startup_timeout_s": config.startup_timeout_s,
        "max_restarts": config.max_restarts,
        "worker_kill_wait_s": WORKER_KILL_WAIT_S,
    }
    if schema == PROCEDURE_LIMITS_SCHEMA:
        limits["worker_rlimits"] = asdict(config.isolation.rlimits)
        limits["ipc_frame_limits"] = asdict(config.limits)
    return limits


def _split_procedure_digest(
    seeded,
    split: SplitSpec,
    config: SupervisorConfig,
    bound: environment_mod.BoundSet,
    *,
    seed: int | None = None,
    attested_files: Sequence[Mapping[str, str]] = (),
) -> str:
    """Hash the reviewed combination of split entries, configuration and bound bytes.

    Values (including worker environment) are hashed, not printed. Candidate
    trees include siblings, model files and other binds visible to the predictor;
    environment trees cover installed runtime and evaluator code. A match is an
    identity check, not proof that a predictor actually used the model.

    A timing profile other than the legacy one hashes a ``v2`` descriptor that adds
    the ``v1`` block of :func:`procedure_limits` (budget, profile, boundary, start-up
    timeout, restart cap); the descriptor's own ``isolation.rlimits`` and
    ``frame_limits`` already bind the rlimits and frame limits that the ``v2`` block
    adds, so digests pinned with this descriptor stay valid. The legacy descriptor
    is unchanged, so digests already pinned for it stay valid; it binds the budget,
    the grace, the hard timeout, the rlimits and the frame limits, but not the
    start-up timeout or the restart cap.
    """

    def trees(records):
        return sorted(
            ({"path": r.path, "kind": r.kind, "sha256": r.sha256} for r in records),
            key=lambda row: (row["path"], row["kind"], row["sha256"]),
        )

    in_bwrap = config.isolation.level == "bwrap"
    worker_env = sandbox.scrubbed_env(
        os.environ,
        home=sandbox.SANDBOX_HOME if in_bwrap else "<private>/home",
        tmpdir="/tmp" if in_bwrap else "<private>/tmp",
        extras=config.isolation.env_set,
        passthrough=config.isolation.env_passthrough,
    )
    descriptor = {
        "schema": "passage-split-procedure-pin/v1",
        "predictor": split.predictor,
        "predictor_kwargs": seeded.inference_kwargs,
        "decoder": split.decoder,
        "decoder_kwargs": dict(split.decoder_kwargs),
        "precision": split.precision,
        "seed": seed,
        "request": {
            "budget": seeded.budget,
            "resolution": seeded.resolution,
            "cost_model": seeded.cost_model,
            "cost_weight": seeded.cost_weight,
            "task_variant": seeded.task_variant,
        },
        "target": {
            "accelerator": seeded.edge_accelerator,
            "hardware_stack": seeded.hardware_stack,
            "dtype": seeded.dtype,
        },
        "candidate_paths": list(config.candidate_paths),
        "candidate_trees": trees(bound.candidate),
        "environment_trees": trees(bound.environment),
        "worker_python": config.worker_python,
        "attested_files": list(attested_files),
        "isolation": {
            "level": config.isolation.level,
            "device_binds": list(config.isolation.dev_binds),
            "rlimits": asdict(config.isolation.rlimits),
            "worker_environment": worker_env,
        },
        "metadata_allow": list(config.metadata_allow),
        "frame_limits": asdict(config.limits),
        "forbidden_modules": list(config.forbidden_modules),
        "hard_timeout_ms": config.hard_timeout_ms,
        "deadline_grace_ms": config.deadline_grace_ms,
        "decoder_allow_sha256": list(config.decoder_allow_sha256),
    }
    if config.timing_profile != timing.LEGACY:
        descriptor["schema"] = "passage-split-procedure-pin/v2"
        descriptor["limits"] = procedure_limits(BudgetSpec(**seeded.budget), config, schema=PROCEDURE_LIMITS_SCHEMA_V1)
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _attest_split_files(
    config: SupervisorConfig, bound: environment_mod.BoundSet, decoder_record: treehash.TreeRecord | None
) -> list[dict[str, str]]:
    """Hash named files from the staged view after proving visibility to their role."""
    rows = []
    for item in config.attest_files:
        role, sep, filename = item.partition("=")
        if not sep or role not in {"predictor", "decoder"} or not os.path.isabs(filename):
            raise PurposeError("attested file must have the form predictor=/absolute/path or decoder=/absolute/path")
        path = Path(os.path.abspath(filename))
        if os.path.realpath(path) != str(path):
            raise PurposeError("attested file path may not contain a symlink")
        roots = [Path(r.path) for r in bound.environment]
        if role == "predictor" or decoder_record is None:
            roots += [Path(r.path) for r in bound.candidate]
        else:
            decoder_root = Path(decoder_record.path)
            overlaps_candidate = any(path == Path(r.path) or Path(r.path) in path.parents for r in bound.candidate)
            if overlaps_candidate and not (path == decoder_root or decoder_root in path.parents):
                raise PurposeError("decoder attestation cannot use another candidate tree through an environment mount")
            roots.append(Path(decoder_record.path))
        if not any(path == root or root in path.parents for root in roots):
            raise PurposeError(f"attested {role} file is not mounted to that worker")
        staged_path = bound.view(path)
        try:
            # Walk every path component through directory descriptors without
            # following symlinks, including a final nonblocking open to reject
            # FIFOs and other special files without hanging the supervisor.
            parts = Path(staged_path).parts
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
            try:
                for part in parts[1:-1]:
                    next_dir = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
                    os.close(directory)
                    directory = next_dir
                fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
            finally:
                os.close(directory)
            with os.fdopen(fd, "rb") as stream:
                before = os.fstat(stream.fileno())
                if not stat.S_ISREG(before.st_mode):
                    raise PurposeError("attested file must be a regular file")
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
                after = os.fstat(stream.fileno())
            keys = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if any(getattr(before, k) != getattr(after, k) for k in keys):
                raise PurposeError("attested file changed during hashing")
        except OSError as error:
            raise PurposeError(f"cannot hash attested {role} file: {type(error).__name__}") from None
        rows.append({"role": role, "path": str(path), "sha256": digest})
    return rows


def _check_split_procedure_pin(config: SupervisorConfig, digest: str) -> dict[str, Any]:
    matched = digest in config.split_procedure_allow_sha256
    if (config.require_split_procedure_pin or config.split_procedure_allow_sha256) and not matched:
        raise PurposeError("split procedure SHA-256 is not in the operator allow-list")
    return {
        "sha256": digest,
        "required": config.require_split_procedure_pin,
        "matched": matched,
        "scope": "identity of reviewed split configuration and bound bytes; not proof of model execution",
    }


def build_runner(
    seeded,
    split: SplitSpec | None,
    config: SupervisorConfig,
    environment: RuntimeEnvironment,
    *,
    seed: int | None = None,
    clock: Callable[[], float] = time.perf_counter,
) -> SupervisedRunner:
    """Stage the declared files, measure the identity of everything bound, apply the purpose rules, construct a runner."""
    roles = _roles(seeded, split)
    declared: list[Path] = []
    for _, _, kwargs in roles:
        declared += list(identity_mod.checkpoint_files(kwargs).values())
    declared += identity_mod.declared_bundle_members(seeded.runtime_bundle_manifest)
    bound = environment.bound.with_declared(declared, environment.protected)
    if split is None and (config.require_split_procedure_pin or config.split_procedure_allow_sha256):
        raise PurposeError("split procedure pin requires split mode")
    if split is None and config.attest_files:
        raise PurposeError("attested split files require split mode")
    pinned_role, pinned_record, counters = _pin_decision(config, seeded, split, environment.probe, bound.view)
    attested_files = (
        [] if split is None else _attest_split_files(config, bound, pinned_record if pinned_role == "decoder" else None)
    )
    split_pin = (
        None
        if split is None
        else _check_split_procedure_pin(
            config, _split_procedure_digest(seeded, split, config, bound, seed=seed, attested_files=attested_files)
        )
    )
    if split_pin is not None:
        split_pin["attested_files"] = attested_files
    dev_nodes = [Path(d) for d in config.isolation.dev_binds]
    environment_roots = [Path(record.path) for record in bound.environment]
    kwargs_inside = {}
    for role, _, kwargs in roles:
        if role == pinned_role:
            # A pinned worker is given its own package tree and the environment, nothing else (BoundSet.pinned_view).
            if identity_mod.checkpoint_files(kwargs):
                raise identity_mod.IdentityError(
                    f"the pinned {role}'s kwargs declare files ({sorted(identity_mod.checkpoint_files(kwargs))}) that its "
                    f"sandbox would not hold: a pinned {role} is self-contained, so put such files inside its pinned tree"
                )
            covered = [Path(pinned_record.path), *environment_roots, *dev_nodes]
        else:
            covered = bound.covered_roots() + dev_nodes
        kwargs_inside[role] = identity_mod.check_kwargs_paths(role, kwargs, covered)
    measure = _identity_function(seeded, split, environment.probe, bound)
    identity, components, bundle = measure()

    evaluator_hash = _evaluator_source_hash(environment.probe, bound.view)
    from passage_eval import cli

    evaluator_matches = evaluator_hash == cli._source_hash()
    extra_files = _evaluator_extra_files(environment.probe, bound.view)
    extras_clean = all(row["unmatched_count"] == 0 for row in extra_files.values())
    env_pin = _environment_pin(bound, config.environment_allow_sha256)
    scan = bound.forbidden_scan()
    if config.purpose == PURPOSE_VERIFIED:
        problems = []
        if pinned_role is None:
            what = (
                f"the decoder entry tree sha256 {pinned_record.sha256}"
                if split is not None
                else f"the planner entry tree sha256 {pinned_record.sha256}"
            )
            problems.append(f"{what} is not in the operator allow-list")
        if not evaluator_matches:
            problems.append("the evaluator packages in the worker environment differ from the supervisor's own")
        if not extras_clean:
            problems.append(
                "the worker's evaluator packages hold non-source files (extension modules, bytecode, .pth) "
                "that the supervisor's own packages do not hold identically: "
                f"{ {n: r['unmatched'][:3] for n, r in extra_files.items() if r['unmatched_count']} }"
            )
        if not env_pin["all_trees_allow_listed"]:
            missing = [f"{row['path']} {row['sha256']}" for row in env_pin["trees"] if not row["in_allow_list"]]
            problems.append(
                "interpreter-environment tree digest(s) not in the operator allow-list "
                f"(--environment-allow-sha256): {'; '.join(missing[:6])}"
            )
        if any(row["hash_source"] != "computed" for row in env_pin["trees"]):
            problems.append("an interpreter-environment digest was not computed in this run")
        if not scan["clean"]:
            hits = [f"{row['tree']}: {row['entries'][:3]}" for row in scan["found"]]
            problems.append(
                "the supervisor found onnxruntime distributions or libraries in a bound tree "
                f"({'; '.join(hits[:4])}); ONNX Runtime is not allowed in a verified run"
            )
        if environment.probe["channel"]["trailing_bytes_ignored"]:
            problems.append("the interpreter probe wrote more than one message to its result channel (tampering)")
        if problems:
            raise PurposeError("verified-candidate-run refused: " + "; ".join(problems))

    views = {}
    for role, _, _ in roles:
        if role == pinned_role:
            views[role] = bound.pinned_view(pinned_record.path)
        else:
            views[role] = environment_mod.RoleBinds(tuple(bound.binds), tuple(config.candidate_paths), narrowed=False)

    def check_identity():
        bound.recheck()
        if split is not None and (config.require_split_procedure_pin or config.split_procedure_allow_sha256):
            current_files = _attest_split_files(config, bound, pinned_record if pinned_role == "decoder" else None)
            current_digest = _split_procedure_digest(
                seeded, split, config, bound, seed=seed, attested_files=current_files
            )
            if current_digest != split_pin["sha256"]:
                raise IntegrityError("split procedure changed during the run")
        return measure()[0]

    runner = SupervisedRunner(
        config,
        _worker_specs(seeded, split, seed),
        bound=bound,
        identity=identity,
        identity_check=check_identity,
        split=split,
        clock=clock,
        counters_source=counters,
        scrub=environment.scrub,
        views=views,
    )
    runner.isolation_record = sandbox.isolation_record(
        config.isolation, bound.binds, bwrap_version=environment.bwrap_version
    )
    runner.control = component_control(seeded.component_control)
    runner.facts = {
        "components": components,
        "bundle": bundle,
        "kwargs_paths_inside": kwargs_inside,
        "split_procedure_pin": split_pin,
        "pinned_role": pinned_role,
        "pinned_entry_tree": None if pinned_role is None else pinned_record.to_manifest(),
        "decoder_entry_tree": None if split is None or pinned_record is None else pinned_record.to_manifest(),
        "planner_entry_tree": None if split is not None or pinned_record is None else pinned_record.to_manifest(),
        "environment_pin": env_pin,
        "forbidden_scan": scan,
        "evaluator_extra_files": extra_files,
        "evaluator_source_hash": evaluator_hash,
        "evaluator_matches_supervisor": evaluator_matches,
        "worker_views": {
            role: _view_manifest(view, config.isolation.level, pinned=role == pinned_role)
            for role, view in views.items()
        },
    }
    return runner


def _view_manifest(view: environment_mod.RoleBinds, level: str, *, pinned: bool) -> dict[str, Any]:
    return {
        "pinned": pinned,
        "narrowed_to_pinned_tree": view.narrowed and pinned,
        "pinned_tree": view.tree,
        "read_only_binds": [bind.target for bind in view.binds],
        "import_paths": list(view.import_paths),
        "enforced_by": (
            "bwrap_mount_namespace" if level == "bwrap" else "not_enforced_process_isolation_shares_the_file_system"
        ),
        "note": (
            "only the pinned package tree (from its staged copy) and the interpreter environment are mounted"
            if pinned and view.narrowed and view.import_paths
            else "only the interpreter environment is mounted; the pinned package lies inside it"
            if pinned and view.narrowed
            else "all declared candidate trees and the interpreter environment are mounted"
            if level == "bwrap"
            else "no mount namespace: sys.path is narrowed but the file system is shared"
        ),
    }


@dataclass(frozen=True)
class RosterInput:
    """A pinned roster file (``passage-roster-v2``, or a readable v1), its expected digest and its dataset root."""

    path: str
    sha256: str
    dataset_root: str
    roster_id: str | None = None
    verify_file_hashes: bool = True

    def __post_init__(self):
        if not treehash.is_hex64(self.sha256):
            raise ValueError("the roster pin must be a lowercase 64-character SHA-256")


@dataclass(frozen=True)
class WarmupInput:
    """Where the untimed warm-up inputs come from: another split of the dataset, or a second pinned roster.

    The test split is refused as a warm-up source (``protocol.guards``) before any of its rows is read, unless
    ``allow_test`` is true and ``admission`` is an all-GREEN test admission grant (``guards.test_split_allowed``).
    """

    split: str | None = None
    roster: RosterInput | None = None
    allow_test: bool = False
    admission: Any = None

    def __post_init__(self):
        if (self.split is None) == (self.roster is None):
            raise ValueError("give exactly one warm-up source: a split name or a roster")
        if self.split is not None:
            refuse_test_split(
                self.split, allow_test=self.allow_test, admission=self.admission, context="the supervisor's warm-up"
            )


def dataset_location_fragments(root: str | Path | None) -> list[str]:
    """Strings that identify the dataset location: the root as given, resolved, and the long file names below it."""
    if root is None:
        return []
    given = str(root)
    fragments = {os.path.abspath(given).rstrip("/"), os.path.realpath(given).rstrip("/")}
    if "/" in given.strip("/") and len(given) >= 8:  # a short relative spelling such as "data" would match everything
        fragments.add(given.rstrip("/"))
    fragments = {f for f in fragments if len(f) >= 4}
    base = Path(os.path.realpath(given))
    if base.is_dir():
        for count, item in enumerate(base.rglob("*")):
            if count >= 50_000:
                break
            if item.is_file() and item.suffix in {".parquet", ".json"}:
                relative = item.relative_to(base).as_posix()
                if len(relative) >= 16:
                    fragments.add(relative)
                if len(item.name) >= 24:
                    fragments.add(item.name)
    return sorted(fragments, key=lambda text: (-len(text), text))


def _walk_strings(value: Any, path: str = ""):
    if isinstance(value, str):
        yield path, value
    elif isinstance(value, Mapping):
        for key, item in value.items():
            yield from _walk_strings(str(key), f"{path}.<key>")
            yield from _walk_strings(item, f"{path}.{key}" if path else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk_strings(item, f"{path}[{index}]")


def assert_spec_has_no_dataset_location(spec, fragments: Sequence[str]) -> None:
    """Refuse to start when any spec string other than ``dataset_path`` names the dataset.

    The public manifest copies every spec field except ``dataset_path``; free text such as ``dataset_version``
    would carry the location into a file that leaves the private zone.
    """
    payload = spec.to_dict()
    payload.pop("dataset_path", None)
    for path, text in _walk_strings(payload):
        for fragment in fragments:
            if fragment in text:
                raise ValueError(
                    f"spec field {path!r} contains the dataset location; the public run manifest copies every spec "
                    f"field except dataset_path, so remove the path (or file name) from that field before running"
                )


def _roster_spec_mismatches(payload: Mapping[str, Any], spec) -> list[str]:
    from passage_eval.metrics.cost import canonical_cost_model

    dataset, cell = payload["dataset"], payload["cell"]
    problems = []
    if spec.resolution != dataset["resolution"]:
        problems.append(f"resolution {spec.resolution} != roster {dataset['resolution']}")
    if canonical_cost_model(spec.cost_model) != canonical_cost_model(cell["cost_model"]):
        problems.append(f"cost_model {spec.cost_model!r} != roster {cell['cost_model']!r}")
    if spec.task_variant != cell["task_variant"]:
        problems.append(f"task_variant {spec.task_variant!r} != roster {cell['task_variant']!r}")
    if spec.split != dataset["split"]:
        problems.append(f"split {spec.split!r} != roster {dataset['split']!r}")
    if spec.cost_weight is not None and spec.cost_weight != cell["cost_weight"]:
        problems.append(f"cost_weight {spec.cost_weight} != roster {cell['cost_weight']}")
    if spec.dataset_revision is not None and spec.dataset_revision != dataset["commit"]:
        problems.append("dataset_revision differs from the roster's pinned commit")
    return problems


def _load_roster_scenes(roster: RosterInput, spec):
    payload = roster_io.load_roster(roster.path, expected_sha256=roster.sha256)
    problems = _roster_spec_mismatches(payload, spec)
    if problems:
        raise ValueError("the spec does not match the roster's cell: " + "; ".join(problems))
    scenes = roster_io.load_roster_scenes(
        payload,
        roster.dataset_root,
        verify_file_hashes=roster.verify_file_hashes,
        # Under the independent analytic policy a published label is a finding, never the reference (judge R1).
        tolerant_labels=spec.reference_policy == "independent_analytic_reference",
    )
    expected = [task["sample_id"] for task in payload["tasks"]]
    if [scene.sample_id for scene in scenes] != expected or len(scenes) != payload["selection"]["n_expected"]:
        raise ValueError("the loaded scenes do not cover every roster task in order")
    return payload, scenes


def _roster_block(roster: RosterInput, payload: Mapping[str, Any], n_loaded: int) -> dict[str, Any]:
    return {
        "roster_id": roster.roster_id or payload["benchmark_id"],
        "benchmark_id": payload["benchmark_id"],
        "roster_sha256": payload["roster_sha256"],
        "n_expected": payload["selection"]["n_expected"],
        "n_loaded": n_loaded,
        "records_required_per_pass": payload["selection"]["n_expected"],
        "dataset": {key: payload["dataset"][key] for key in ("repo", "commit", "split", "resolution", "config")},
        "cell": dict(payload["cell"]),
        "dataset_root_identity": dict(payload["dataset_root_identity"]),
        "dataset_root_path_recorded": False,
        "source_file_hashes_verified": roster.verify_file_hashes,
        "labels": "supervisor_side_only_dropped_before_workers",
    }


def _input_digest(request: PlanRequest) -> str:
    return roster_io.identity_digest(roster_io.request_input_identity(request))


def _content_digest(request: PlanRequest) -> str:
    """The input identity without the ordinary-metadata digest, which differs between splits for the same content."""
    return roster_io.content_identity_digest(roster_io.request_input_identity(request))


def _load_warmup_requests(
    spec, warmup: WarmupInput, count: int, dataset_location: str | None, roster_requests: Sequence[PlanRequest]
) -> tuple[list[PlanRequest], dict[str, Any]]:
    """Warm-up inputs: ordinary requests that are disjoint from the roster by id and by input content identity."""
    from passage_eval import cli

    if warmup.split is not None:
        from passage.dataset import SPLITS

        refuse_test_split(
            warmup.split, allow_test=warmup.allow_test, admission=warmup.admission, context="the supervisor's warm-up"
        )
        if SPLITS.get(warmup.split, warmup.split) == SPLITS.get(spec.split, spec.split):
            raise ValueError("the warm-up split must differ from the evaluated split")
        scenes = cli._load_scenes(replace(spec, split=warmup.split, n_samples=count, dataset_path=dataset_location))
        source = f"split:{warmup.split}"
    else:
        payload = roster_io.load_roster(warmup.roster.path, expected_sha256=warmup.roster.sha256)
        refuse_test_roster(
            payload, allow_test=warmup.allow_test, admission=warmup.admission, context="the supervisor's warm-up"
        )
        scenes = roster_io.load_roster_scenes(
            payload, warmup.roster.dataset_root, verify_file_hashes=warmup.roster.verify_file_hashes
        )[:count]
        source = f"roster:{payload['roster_sha256']}"
    if not scenes:
        raise ValueError("the warm-up source provided no inputs")
    requests = [PlanRequest.from_scene(candidate_scene(scene), budget=BudgetSpec(**spec.budget)) for scene in scenes]
    del scenes
    roster_ids = {request.sample_id for request in roster_requests}
    roster_inputs = {_content_digest(request) for request in roster_requests}
    shared_ids = sorted(request.sample_id for request in requests if request.sample_id in roster_ids)
    shared_inputs = [request.sample_id for request in requests if _content_digest(request) in roster_inputs]
    if shared_ids or shared_inputs:
        raise ValueError(
            "warm-up inputs are not disjoint from the roster "
            f"({len(set(shared_ids))} shared ids, {len(set(shared_inputs))} shared inputs)"
        )
    identity = hashlib.sha256(json.dumps(sorted(_input_digest(request) for request in requests)).encode()).hexdigest()
    return requests, {
        "policy": "disjoint_inputs",
        "source": source,
        "n_inputs": len(requests),
        "input_set_sha256": identity,
        "disjoint_from_roster": True,
        "disjointness_checked_by": "sample_id_and_input_content_identity",
    }


def _measure_baselines(runner: SupervisedRunner, requests: Sequence[PlanRequest], split: SplitSpec | None) -> None:
    sizes = sorted(
        {
            redact_ordinary(
                r, token="req-baseline", remaining_wall_ms=None, metadata_allow=runner.config.metadata_allow
            ).payload_bytes
            for r in requests
        }
    )
    if len(sizes) > 8:
        sizes = [sizes[0], sizes[len(sizes) // 2], sizes[-1]]
    first_role = "planner" if split is None else "predictor"
    runner.baseline[first_role] = [row for size in sizes for row in runner.measure_baseline(first_role, size)]
    if split is not None:
        # The field size is unknown before the first prediction: the contract fixes one float32/float16 plane of
        # the grid shape, so assume float32 (the larger). The manifest records the sizes actually observed.
        decoder_sizes = sorted({int(r.obstacles.nbytes + 4 * r.obstacles.size) for r in requests})[:8]
        runner.baseline["decoder"] = [
            dict(row, payload_size_basis="assumed_one_float32_field_plus_obstacles")
            for size in decoder_sizes
            for row in runner.measure_baseline("decoder", size)
        ]


REWARM_POLICY = {
    "status": "proposed_not_sealed",
    "trigger": "a worker generation changed (restart) inside a warm pass since the last warm-up attempt",
    "action": "replay the pass warm-up on the restarted workers, outside the boundary, before the next measured request",
    "recursion_guard": "at most one re-warm before a measured request; a re-warm never triggers another",
    "initial_warmup": "runs every request, as in the other profiles; its restarts count against max_restarts",
    "failure": "a re-warm stops at its first request whose status is not ok or that loses a worker; the failure "
    "is recorded, the pass is not re-warmed again, and the measured requests that follow carry "
    "warmup_complete_before_request = false",
    "restarts": "a worker lost during a re-warm is restarted outside max_restarts (at most one failed re-warm per "
    "pass); restarts caused by roster tasks or by the initial warm-up count against max_restarts as in the other "
    "profiles, so a warm-up input cannot turn a roster task into a worker_restart_cap failure",
    "record_fields": list(timing.REWARM_RECORD_FIELDS),
}


def _timing_policy(
    seeded, warmup_block: Mapping[str, Any], config: SupervisorConfig, *, session: bool = False
) -> dict[str, Any]:
    profile = timing.get_profile(config.timing_profile)
    policy = {
        "boundary": profile.boundary or TIMING_BOUNDARY_ID,
        "worker_per_timed_pass": "fresh_workers_started_outside_the_boundary_and_recorded",
        "pass": "one complete roster traversal (series and repeat)",
        "tokens": "fresh_per_pass_hmac_of_series_repeat_and_position",
        "ipc_baseline": "measured_once_on_a_separate_worker_that_serves_no_timed_pass",
        "cold_series": (
            "none: a campaign session measures warm requests only; the cold start is reported per session"
            if session
            else "first pass on fresh workers, no warm-up"
        ),
        "warm_series": "fresh workers, then an untimed warm-up on disjoint inputs, then the timed pass",
        "unwarmed_series": "fresh workers and no warm-up (latency not measured or no warm-up): never called warm",
        "warmup_series_artifact": "not_produced",
        "latency_warmup_unit": "requests_per_warm_pass_from_disjoint_inputs",
        "latency_warmup": seeded.latency_warmup if seeded.measure_latency else 0,
        "warmup": dict(warmup_block),
    }
    if profile.name != timing.LEGACY:
        policy.update(
            latency_profile=profile.name,
            latency_field=profile.latency_field,
            validation_recost="recorded_separately_outside_the_boundary",
        )
    if profile.rewarm_after_restart:
        policy["restart_rewarm"] = dict(REWARM_POLICY)
        if session:
            policy["restart_rewarm"]["initial_warmup"] = (
                "session mode: stops at its first request that does not return ok or that loses a worker; its "
                "failure invalidates the session (no task is charged and max_restarts is not consumed)"
            )
    return policy


def _base_manifest(
    seeded,
    split,
    config: SupervisorConfig,
    runner: SupervisedRunner,
    environment: RuntimeEnvironment,
    content_hash: str,
    reference_identity: str,
    hardware: str,
    *,
    roster_block: Mapping[str, Any] | None,
    warmup_block: Mapping[str, Any],
    purpose_checks: Sequence[Mapping[str, Any]],
    session: bool = False,
) -> dict[str, Any]:
    from passage_eval import cli

    probe = environment.probe
    identity, control, isolation = runner.identity, runner.control, runner.isolation_record
    main_role = "planner" if split is None else "predictor"
    checkpoint_hashes = {
        key: info["sha256"] for key, info in runner.facts["components"][main_role]["checkpoints"].items()
    }
    policies = [POLICY_ORDINARY_REQUEST] + ([POLICY_DECODER_ONLY] if split is not None else [])
    verified_eligible = config.purpose == PURPOSE_VERIFIED
    supervision_procedure = {
        "timing_boundary": runner.boundary,
        "ipc_protocol": IPC_PROTOCOL,
        "purpose": config.purpose,
        "mode": "split" if split is not None else "planner",
        "redaction_policies": policies,
        "decoder_field_contract": FIELD_CONTRACT if split is not None else None,
        "metadata_allow_list": list(config.metadata_allow),
        "isolation_level": config.isolation.level,
        "deadline_grace_ms": config.deadline_grace_ms,
        "hard_timeout_ms": config.hard_timeout_ms,
        "pass_policy": "fresh_worker_and_tokens_per_timed_pass",
        "counters_source": runner.counters_source,
        "pins": {
            "pinned_role": runner.facts["pinned_role"],
            "split_procedure": runner.facts["split_procedure_pin"],
            "pinned_entry_tree_sha256": (runner.facts["pinned_entry_tree"] or {}).get("sha256"),
            "decoder_allow_list": list(config.decoder_allow_sha256),
            "planner_allow_list": list(config.planner_allow_sha256),
            "environment_allow_list": list(config.environment_allow_sha256),
            "environment_trees": [
                {"path": row["path"], "sha256": row["sha256"]} for row in runner.facts["environment_pin"]["trees"]
            ],
        },
        "worker_bytecode": "fresh_private_pycache_prefix",
        "forbidden_modules": list(config.forbidden_modules),
        "roster": None
        if roster_block is None
        else {
            "roster_id": roster_block["roster_id"],
            "roster_sha256": roster_block["roster_sha256"],
            "n_expected": roster_block["n_expected"],
        },
        # Every bound on a request and every timing setting, so that procedure_id changes with any of them.
        "limits": procedure_limits(BudgetSpec(**seeded.budget), config),
        "latency_protocol": {
            "measure_latency": bool(seeded.measure_latency),
            "latency_repeats": seeded.latency_repeats,
            "latency_warmup_requests_per_warm_pass": seeded.latency_warmup if seeded.measure_latency else 0,
            "warmup_source": warmup_block.get("source"),
            "warmup_input_set_sha256": warmup_block.get("input_set_sha256"),
        },
    }
    if config.timing_profile != timing.LEGACY:
        supervision_procedure["timing_profile"] = config.timing_profile
    if session:
        # Session mode changes what a worker that cannot start does to the tasks (an invalid session, never a task
        # failure): it belongs to the procedure. The window deadline and persistence reserve do not.
        supervision_procedure["session_policy"] = session_mod.SESSION_POLICY_ID
    procedure = {
        "method_name": seeded.method_name,
        "family": seeded.family,
        "architecture": seeded.architecture,
        "decoder_variant": seeded.decoder_variant,
        "inference": seeded.inference,
        "inference_format": seeded.inference_format,
        "inference_kwargs": seeded.inference_kwargs,
        "budget": seeded.budget,
        "checkpoint_hashes": checkpoint_hashes,
        "candidate_identity": identity,
        "component_control": control,
        "supervision": supervision_procedure,
    }
    if split is not None:
        procedure["split"] = {
            "predictor": split.predictor,
            "decoder": split.decoder,
            "decoder_kwargs": dict(split.decoder_kwargs),
            "precision": split.precision,
        }
    ready = {
        role: {
            key: value[key]
            for key in (
                "python",
                "numpy",
                "rlimits",
                "pdeathsig",
                "has_synchronize",
                "loaded_files",
                "sys_modules_top_level",
                "forbidden_modules",
            )
        }
        for role, value in runner.ready.items()
    }
    for role, value in runner.ready.items():
        probe_seen = value["isolation_probe"]
        devices = probe_seen["net_devices"]
        ready[role]["isolation_probe_summary"] = {  # counts only: no host interface names or paths
            "pid_seen_by_worker": probe_seen["pid"],
            "ppid_seen_by_worker": probe_seen["ppid"],
            "non_loopback_interfaces": None if devices is None else len([d for d in devices if d != "lo"]),
            "root_entries": probe_seen["root_entries"] if config.isolation.level == "bwrap" else None,
        }
        ready[role]["self_reported"] = ["loaded_files", "sys_modules_top_level", "forbidden_modules"]
        ready[role]["forbidden_modules_checked"] = list(value["forbidden_modules"])
    layout = environment.layout
    isolation = dict(
        isolation,
        attestation={
            "effective_level_checked_at_every_worker_start": config.isolation.level == "bwrap",
            "checks": (
                ["root_entries_within_binds", "only_loopback_interface", "pid_namespace", "core_dumps_disabled"]
                if config.isolation.level == "bwrap"
                else ["core_dumps_disabled"]
            ),
            "on_contradiction": "run_aborted",
            "source": "worker_attestation_sent_before_candidate_import",
            "note": "cross-check of the supervisor's own launch arguments; not an independent proof against a hostile candidate",
        },
    )
    supervision = {
        "schema": SUPERVISION_SCHEMA,
        "timing_boundary": runner.boundary,
        "ipc_protocol": IPC_PROTOCOL,
        "mode": supervision_procedure["mode"],
        "purpose": {
            "name": config.purpose,
            "verified_candidate_run": verified_eligible,
            "checks": list(purpose_checks),
            "note": (
                "diagnostic runs are never verified results"
                if not verified_eligible
                else "all verified-candidate-run requirements were enforced before any worker started"
            ),
        },
        "redaction": {
            "policies": policies,
            "sample_id": "opaque_per_run_token",
            "metadata_allow_list": list(config.metadata_allow),
            "metadata_implied_by_input_tensor": list(TENSOR_IMPLIED_METADATA),
            "decoder_field_contract": FIELD_CONTRACT if split is not None else None,
            "decoder_field_contract_rules": (
                "exactly one field; shape equal to the grid (H, W) after removing leading singleton batch/channel "
                "dimensions (the decoder receives (H, W)); dtype float32 or float16; all values finite; "
                "anything else is a runtime_error for that request"
                if split is not None
                else None
            ),
        },
        "isolation": isolation,
        "interpreter_probe": {
            "executed_in": probe["executed_in"],
            "flags": ["-I", "-B", "-X pycache_prefix=<private>"],
            "static_layout_read_without_execution": True,
            "uncovered_paths": list(environment.uncovered_paths),
            "channel": dict(probe["channel"]),
            "startup_output_chars": len(probe.get("startup_output", "")),
        },
        "worker_bytecode": {
            "policy": "fresh_private_pycache_prefix",
            "prefix": (
                "tmpfs_inside_the_sandbox" if config.isolation.level == "bwrap" else "private_directory_of_the_worker"
            ),
            "effect": "no __pycache__ beside any source is read; every import compiles from source",
            "applies_to": ["worker", "interpreter_probe"],
        },
        "worker_views": runner.facts["worker_views"],
        "environment_pin": runner.facts["environment_pin"],
        "split_procedure_pin": runner.facts["split_procedure_pin"],
        "runtime_layout": {
            "venv_root": layout.venv_root,
            "base_prefix": layout.base_prefix,
            "roots": list(layout.roots),
            "editable_roots": dict(layout.editable_roots),
            "pth_roots": list(layout.pth_roots),
            "named_by_environment_not_bound_on_its_say_so": [list(row) for row in layout.needs_declaration],
            "executable_pth_lines_run_inside_the_worker_sandbox": layout.pth_import_lines,
        },
        "bound_trees": runner.bound.manifest(),
        "kwargs_paths_inside_bound_trees": runner.facts["kwargs_paths_inside"],
        "evaluator": {
            "source_hash_in_worker_environment": runner.facts["evaluator_source_hash"],
            "matches_supervisor": runner.facts["evaluator_matches_supervisor"],
            "non_source_files": runner.facts["evaluator_extra_files"],
            "non_source_files_rule": (
                "every non-.py file of the worker's passage and passage_eval packages "
                "exists identically in the supervisor's own packages"
            ),
        },
        "counters": {
            "counters_source": runner.counters_source,
            "pinned_role": runner.facts["pinned_role"],
            "decoder_allow_list": list(config.decoder_allow_sha256),
            "decoder_entry_tree": runner.facts["decoder_entry_tree"],
            "planner_allow_list": list(config.planner_allow_sha256),
            "planner_entry_tree": runner.facts["planner_entry_tree"],
            "environment_allow_list": list(config.environment_allow_sha256),
            "self_reported_fields": list(SELF_REPORTED_FIELDS),
            "note": (
                "work counters, stage and fallback flags, precision and diagnostics are produced by "
                "the process that runs candidate-influenced code; they are self-reported. With "
                "operator_decoder (split mode) or operator_planner (planner mode) the entry's own "
                "package tree matched an operator allow-list and that worker was given only that tree, the "
                "interpreter environment and the evaluator sources, so the counters come from operator-pinned "
                "code. In a verified run every interpreter-environment tree is pinned too; in a diagnostic "
                "run the environment is hashed (bound_trees) but not pinned."
            ),
        },
        "forbidden_modules": {
            "names": list(config.forbidden_modules),
            "checked_in": "worker_sys_modules_after_import_and_after_each_request",
            "evidence": "self_reported_by_the_worker",
            "action": "runtime_error_and_worker_killed",
            "supervisor_scan": runner.facts["forbidden_scan"],
        },
        "text_limits": dict(sanitize.LIMITS),
        "roster": None if roster_block is None else dict(roster_block),
        "timing_policy": _timing_policy(seeded, warmup_block, config, session=session),
        "environment": {
            "scrubbed": True,
            "allow_list_exact": sorted(sandbox.ENV_ALLOW_EXACT),
            "allow_list_prefix": list(sandbox.ENV_ALLOW_PREFIX),
            "explicit_set_names": sorted(config.isolation.env_set),
            "explicit_passthrough_names": sorted(config.isolation.env_passthrough),
        },
        "worker_python": os.path.abspath(config.worker_python),
        "worker_python_version": probe["version"],
        "workers_ready": ready,
        "workers_ready_source": (
            "worker attestation sent before candidate import; it cross-checks the supervisor's "
            "own launch arguments and is not an independent proof against a hostile candidate"
        ),
        "deadline": {
            "grace_ms": config.deadline_grace_ms,
            "preemption": "SIGKILL_process_group",
            "hard_timeout_ms_without_wall_budget": config.hard_timeout_ms,
        },
        "ipc_echo_baseline_ms": runner.baseline,
        "ipc_baseline_subtracted": False,
        "not_in_boundary": _not_in_boundary(runner.profile),
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "supervisor_python": platform.python_version(),
        },
        "artifacts_private": (
            "run artifacts contain candidate-controlled text (diagnostics, error messages, stderr "
            "tails); they are bounded and scrubbed of the dataset location but must be treated as "
            "private until reviewed"
        ),
    }
    return {
        "bench_spec": _public_bench_spec(seeded),
        "method_name": seeded.method_name,
        "seed": int(seeded.seeds[0]),
        "procedure": procedure,
        "procedure_id": hashlib.sha256(json.dumps(procedure, sort_keys=True).encode()).hexdigest(),
        "checkpoint_hashes": checkpoint_hashes,
        "runtime_source_hash": cli._source_hash(),
        "candidate_identity": identity,
        "component_control": control,
        "reference_qualification_withheld": True,
        "reference_identity": reference_identity,
        "admission": {"qualified_reference": True, "ordinary_input_contract": "PlanRequest-v1"},
        "reference_policy": seeded.reference_policy,
        "reference_cohort_id": seeded.reference_cohort_id,
        "dataset": {
            "repo": seeded.dataset_repo if roster_block is None else roster_block["dataset"]["repo"],
            "revision": seeded.dataset_revision if roster_block is None else roster_block["dataset"]["commit"],
            "path": None,
            "path_redacted": True,
            "split": seeded.split,
            "resolution": seeded.resolution,
            "task_variant": seeded.task_variant,
            "content_hash": content_hash,
        },
        "split_role": "evaluation",
        "hardware_stack": hardware,
        "cold_series_scope": (
            "none: campaign session, warm requests only; the cold start is in supervision.session.cold_start"
            if session
            else "first complete roster pass on freshly started workers, no warm-up; process start and import excluded"
        ),
        "extra": seeded.extra,
        "supervision": supervision,
    }


def _run_series(
    runner: SupervisedRunner,
    requests: Sequence[PlanRequest],
    base: Mapping[str, Any],
    series: str,
    repeats: int,
    out_dir: Path,
    quality_repeat_index: int,
    *,
    warmup_requests: Sequence[PlanRequest] = (),
    warmup_count: int = 0,
) -> RunArtifact:
    roster = [request.sample_id for request in requests]
    if len(set(roster)) != len(roster):
        raise ValueError("request roster must be unique")
    if series == "warm" and not (warmup_count and warmup_requests):
        raise ValueError("a series without a warm-up is not warm: name it 'unwarmed'")
    start_event, start_pass = len(runner.events), len(runner.pass_log)
    marked_before = runner.tasks_marked_not_executed
    records, paths = [], []
    for repeat in range(repeats):
        pass_id = f"{series}/{repeat}"
        tokens = [request_token(runner.secret, index, scope=pass_id) for index in range(len(requests))]
        began = runner.clock()
        start_ms = runner.begin_pass(pass_id)  # fresh workers, outside every request's boundary
        generations = {role: worker.generation for role, worker in runner.workers.items()}
        warmup_summary = None
        if warmup_count and warmup_requests and series == "warm":
            runner.set_warm_plan(warmup_requests, warmup_count, pass_id)
            warmup_summary = runner.run_warmup(warmup_requests, warmup_count, pass_id)
        for request, token in zip(requests, tokens):
            record, raw = runner.run_request(request, token, repeat, series, pass_id=pass_id, pass_start_ms=start_ms)
            records.append(record)
            paths.append(raw)
        _log_pass(runner, pass_id, series, repeat, start_ms, generations, warmup_summary, len(requests), began)
    return _write_series_artifact(
        runner,
        requests,
        base,
        series,
        repeats,
        out_dir,
        quality_repeat_index,
        records,
        paths,
        start_event=start_event,
        start_pass=start_pass,
        marked_before=marked_before,
    )


def _log_pass(runner, pass_id, series, repeat, start_ms, generations, warmup_summary, n_records, began) -> None:
    entry = {
        "pass_id": pass_id,
        "series": series,
        "repeat": repeat,
        "worker_start_ms": start_ms,
        "worker_generations_at_start": generations,
        "warmup": warmup_summary,
        "records": n_records,
        "pass_wall_ms_including_start_and_warmup": 1000.0 * (runner.clock() - began),
    }
    if runner.profile.rewarm_after_restart:
        entry["rewarms"] = list(runner.pass_rewarms)
    runner.pass_log.append(entry)
    runner.clear_warm_plan()


def _write_series_artifact(
    runner: SupervisedRunner,
    requests: Sequence[PlanRequest],
    base: Mapping[str, Any],
    series: str,
    repeats: int,
    out_dir: Path,
    quality_repeat_index: int,
    records: list,
    paths: list,
    *,
    start_event: int,
    start_pass: int,
    marked_before: int,
    session_block: Mapping[str, Any] | None = None,
) -> RunArtifact:
    """Re-check the candidate identity, then write the series as a run artifact (``requests`` are its roster)."""
    roster = [request.sample_id for request in requests]
    try:
        current_identity = runner.identity_check()
    except identity_mod.IdentityError as error:
        raise SupervisionError(f"candidate source/runtime identity changed during the run: {error}") from None
    if current_identity != runner.identity:
        raise SupervisionError("candidate source/runtime identity changed during the run")
    manifest = dict(json_safe(dict(base)))
    for reserved in (
        "schema",
        "roster",
        "repeats",
        "quality_repeat_index",
        "series",
        "request_specs",
        "request_metadata",
        "timing_boundary",
        "timing_profile",
        "budget_enforcement",
        "cancellation_policy",
    ):
        if reserved in manifest:
            raise ValueError(f"manifest cannot override supervisor field {reserved}")
    supervision = dict(manifest["supervision"])
    supervision["worker_events"] = json_safe(runner.events[start_event:])
    supervision["passes"] = json_safe(runner.pass_log[start_pass:])
    supervision["observed_request_payload_bytes"] = _stats_manifest(runner)
    supervision["restart_cap"] = {
        "max_restarts": runner.config.max_restarts,
        "unplanned_restarts_used": runner.restarts,
        "cap_reached": runner.cap is not None,
        "reason": None if runner.cap is None else runner.cap["reason"],
        "tasks_marked_not_executed_in_this_series": runner.tasks_marked_not_executed - marked_before,
        "rule": "after the cap every remaining roster task is a runtime_error with reason worker_restart_cap",
    }
    if runner.profile.rewarm_after_restart:
        supervision["restart_cap"]["rewarm_restarts_outside_cap"] = runner.rewarm_restarts
    if runner.profile.name != timing.LEGACY:
        manifest["timing_profile"] = runner.profile.name
        supervision["timing_profile"] = runner.profile.name
    if session_block is not None:
        supervision["session"] = json_safe(dict(session_block))
    manifest["supervision"] = supervision
    manifest.update(
        schema=runner.profile.schema,
        roster=roster,
        repeats=repeats,
        quality_repeat_index=quality_repeat_index,
        series=series,
        request_specs={r.sample_id: request_identity(r) for r in requests},
        request_metadata={r.sample_id: r.metadata for r in requests},
        timing_boundary=runner.boundary,
        budget_enforcement=BUDGET_ENFORCEMENT,
        cancellation_policy=CANCELLATION_POLICY,
    )
    blob = json.dumps(json_safe(manifest), sort_keys=True)
    for fragment in runner.scrub:
        if fragment in blob:
            raise SupervisionError("a run manifest would contain the dataset location; nothing was written")
    artifact = RunArtifact(manifest, records, paths)
    artifact.validate()
    return write_run_artifact(artifact, out_dir)


def _cold_start(runner: SupervisedRunner, pass_id: str, start_event: int) -> dict[str, Any]:
    """The cold start of one session: the start of each role's worker for the pass, and the first warm-up request."""
    starts = {
        row["role"]: row["start_ms"]
        for row in runner.events[start_event:]
        if row.get("event") == "pass_start" and row.get("pass_id") == pass_id
    }
    return {
        "scope": "per session: worker load time of each role for the measured pass and the first warm-up request "
        "(one warm-up input, once per session); never mixed with the warm latency row",
        "worker_load_ms": dict(sorted(starts.items())),
        "initial_worker_start_ms": dict(sorted(runner.initial_start_ms.items())),
        "first_warmup_request": None if runner.first_warmup_request is None else dict(runner.first_warmup_request),
    }


def _run_session(
    runner: SupervisedRunner,
    requests: Sequence[PlanRequest],
    base: Mapping[str, Any] | Callable[[], Mapping[str, Any]],
    out_dir: Path,
    *,
    warmup_requests: Sequence[PlanRequest],
    warmup_count: int,
    admission: Mapping[str, Any],
) -> dict[str, Any]:
    """One campaign session: fresh workers, warm-up, then one warm measured request per task while admitted.

    Returns the session outcome. A closed session writes the measured prefix of the roster as its run artifact and
    defers the rest, in order; an invalid session writes no run artifact.
    """
    roster = [request.sample_id for request in requests]
    if len(set(roster)) != len(roster):
        raise ValueError("request roster must be unique")
    if not (warmup_count and warmup_requests):
        raise ValueError("a campaign session measures warm requests only and needs a declared warm-up")
    series, repeat, pass_id = "warm", 0, "warm/0"
    start_event, start_pass = len(runner.events), len(runner.pass_log)
    marked_before = runner.tasks_marked_not_executed
    records, paths = [], []
    status, closure, invalid = session_mod.COMPLETE, None, None
    tokens = [request_token(runner.secret, index, scope=pass_id) for index in range(len(requests))]
    began = runner.clock()
    start_ms, generations, warmup_summary = 0.0, None, None
    try:
        start_ms = runner.begin_pass(pass_id)  # fresh workers; a worker that cannot start invalidates the session
        base = base() if callable(base) else base
        generations = {role: worker.generation for role, worker in runner.workers.items()}
        runner.set_warm_plan(warmup_requests, warmup_count, pass_id)
        warmup_summary = runner.run_warmup(warmup_requests, warmup_count, pass_id)
        if not warmup_summary["complete"]:
            raise session_mod.SessionInvalid(
                "initial_warmup_failed",
                f"{warmup_summary['succeeded']} of {warmup_count} warm-up requests returned ok "
                f"({warmup_summary['statuses']}); workers alive: {warmup_summary['workers_alive_after']}",
            )
        for request, token in zip(requests, tokens):
            record, raw = runner.run_request(request, token, repeat, series, pass_id=pass_id, pass_start_ms=start_ms)
            records.append(record)
            paths.append(raw)
    except session_mod.SessionClosed as error:
        status = session_mod.CLOSED
        closure = {"point": error.point, "remaining_ms": error.remaining_ms, "required_ms": error.required_ms}
    except session_mod.SessionInvalid as error:
        status = session_mod.INVALID
        invalid = {"reason": error.reason, "detail": sanitize.cap_text(error.detail, sanitize.MAX_ERROR_CHARS)}
        records, paths = [], []  # an invalid session is never scored and never charged to a task
    _log_pass(runner, pass_id, series, repeat, start_ms, generations, warmup_summary, len(records), began)
    executed = len(records)
    deferred = roster[executed:] if status == session_mod.CLOSED else []
    block = {
        "policy": session_mod.SESSION_POLICY_ID,
        "status": status,
        "n_tasks": len(roster),
        "n_executed": executed,
        "executed_sample_ids": roster[:executed],
        "deferred_sample_ids": deferred,
        "closure": closure,
        "invalid": invalid,
        "admission": dict(admission),
        "cold_start": _cold_start(runner, pass_id, start_event),
        "restart_cap_rule": "max_restarts is at least the batch size; reaching it invalidates the session",
        "events": [row.get("event") for row in runner.events[start_event:]],
    }
    if executed:
        _write_series_artifact(
            runner,
            requests[:executed],
            base,
            series,
            1,
            out_dir,
            0,
            records,
            paths,
            start_event=start_event,
            start_pass=start_pass,
            marked_before=marked_before,
            session_block=block,
        )
    block["run_artifact"] = out_dir.name if executed else None
    return block


def _write_session_receipt(output: Path, seed: int, block: Mapping[str, Any], scrub: Sequence[str]) -> Path:
    receipt = json_safe({"schema": session_mod.SESSION_RECEIPT_SCHEMA, "seed": int(seed), **dict(block)})
    blob = json.dumps(receipt, sort_keys=True, allow_nan=False)
    for fragment in scrub:
        if fragment in blob:
            raise SupervisionError("a session receipt would contain the dataset location; nothing was written")
    output.mkdir(parents=True, exist_ok=True)
    path = output / session_mod.SESSION_RECEIPT
    with path.open("x", encoding="utf-8") as stream:
        stream.write(blob + "\n")
    return path


def _check_session_spec(spec, warmup, config: SupervisorConfig) -> None:
    problems = []
    if not spec.measure_latency or not spec.latency_warmup or warmup is None:
        problems.append("a measured warm series with a declared warm-up on disjoint inputs")
    if spec.latency_repeats not in (None, 1):
        problems.append("one measured request per task (latency_repeats 1)")
    if len(spec.seeds) != 1:
        problems.append("one seed per session")
    if not timing.get_profile(config.timing_profile).rewarm_after_restart:
        problems.append("a re-warm timing profile")
    if problems:
        raise ValueError("a campaign session needs " + "; ".join(problems))


def run_supervised(
    spec,
    output: str | Path,
    config: SupervisorConfig,
    *,
    split: SplitSpec | None = None,
    roster: RosterInput | None = None,
    warmup: WarmupInput | None = None,
    clock: Callable[[], float] = time.perf_counter,
    allow_test: bool = False,
    admission: Any = None,
    session: session_mod.SessionPolicy | None = None,
) -> list[Path]:
    """Supervised counterpart of ``cli.cmd_run``: qualify, then run each seed under supervision.

    The test split is refused (``protocol.guards``) before the dataset root is listed or any scene is read: the
    spec's split, and a roster's declared split and source paths. It is accepted only for a roster run with
    ``allow_test=True`` and an all-GREEN test admission grant (``guards.test_split_allowed``) that admits the roster
    (the sealed admissible roster or a child derived from it); a spec run without a roster is refused on test.

    ``session`` selects the campaign session mode (:mod:`passage_eval.supervision.session`): one seed, one warm
    series, dynamic request admission within the window and a ``session.json`` receipt; the returned list then holds
    the seed directory only when at least one task was measured.
    """
    from passage_eval import cli

    previous_bytecode, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        return _run_supervised(cli, spec, output, config, split, roster, warmup, clock, allow_test, admission, session)
    finally:
        sys.dont_write_bytecode = previous_bytecode


def _guard_evaluated_split(spec, roster: RosterInput | None, allow_test: bool, admission: Any) -> dict | None:
    """Refuse the test split for the evaluated tasks; return the roster payload (metadata only) when given."""
    context = "the supervisor"
    if roster is None:
        refuse_test_split(spec.split, context=context)  # a spec run is never admitted: the admission seals a roster
        return None
    payload = roster_io.load_roster(roster.path, expected_sha256=roster.sha256)
    refuse_test_roster(
        payload, allow_test=allow_test, admission=admission, context=context, require_admitted_roster=True
    )
    refuse_test_split(spec.split, allow_test=allow_test, admission=admission, context=context)
    return payload


def _run_supervised(
    cli, spec, output, config, split, roster, warmup, clock, allow_test, admission, session=None
) -> list[Path]:
    from passage_eval.protocol.admission import AdmissionGrant, NATIVE_PROFILES
    from passage_eval.protocol.guards import TestSplitRefused

    if isinstance(admission, AdmissionGrant) and admission.report.admitted.get("profile") in NATIVE_PROFILES:
        # This API has no attested edge target identity. Only the edge campaign planner binds the
        # native profile's target, seed, procedures, budgets and artifacts before any test data is read.
        raise TestSplitRefused("the native grant is accepted only by the admitted edge campaign")
    cli._validate_spec(spec)
    _guard_evaluated_split(spec, roster, allow_test, admission)  # before any listing of the dataset root
    purpose_checks = check_purpose_static(config, spec, split, roster)
    if session is not None:
        _check_session_spec(spec, warmup, config)  # before anything is loaded or started
    if spec.measure_latency and spec.latency_warmup > 0 and warmup is None:
        raise ValueError(
            f"latency_warmup={spec.latency_warmup} needs warm-up inputs that are disjoint from the roster "
            f"(--warmup-split or --warmup-roster); the supervisor does not warm up on the evaluated inputs, because a "
            f"candidate could then answer the timed pass from a cache. Set latency_warmup to 0 to run without warm-up."
        )
    dataset_location = roster.dataset_root if roster is not None else spec.dataset_path
    fragments = dataset_location_fragments(dataset_location)
    for extra in (spec.dataset_path,):
        fragments = sorted(set(fragments) | set(dataset_location_fragments(extra)), key=lambda t: (-len(t), t))
    assert_spec_has_no_dataset_location(spec, fragments)  # before anything is loaded or started

    roster_payload, roster_block = None, None
    if roster is not None:
        roster_payload, scenes = _load_roster_scenes(roster, spec)
    else:
        scenes = cli._load_scenes(spec)
    content_hash = cli._dataset_content_hash(scenes)
    qualification = cli._qualification(scenes, spec)  # before any candidate is started
    if admission is not None and roster_payload is not None and roster_payload["dataset"]["split"] == "test":
        from passage_eval.protocol.reference_admission import check_reference_ids

        # An admitted test run uses exactly the references of the sealed inventory (one limit, one policy).
        check_reference_ids(qualification, admission.reference_ids(), context="the supervisor")
    reference_identity = hashlib.sha256(json.dumps(qualification, sort_keys=True).encode()).hexdigest()
    hardware = spec.hardware_stack or f"{socket.gethostname()}-cpu"
    # Keep only label-free scenes in this process while candidates run (best effort: freed memory is not wiped).
    ordinary_scenes = [candidate_scene(scene) for scene in scenes]
    if roster is not None:
        roster_block = _roster_block(roster, roster_payload, len(scenes))
    del scenes, qualification
    gc.collect()

    entries = [spec.inference] if split is None else [split.predictor, split.decoder]
    paths = []
    with contextlib.closing(
        inspect_environment(config, entries, [*([dataset_location] if dataset_location else []), str(output)])
    ) as environment:
        environment.scrub = tuple(fragments)
        for seed in spec.seeds:
            seed_output = Path(output) / f"seed-{seed}"
            if seed_output.exists() and any(seed_output.iterdir()):
                raise FileExistsError(f"run output already exists: {seed_output}")
            seeded = cli._seed_spec(spec, int(seed))
            requests = [PlanRequest.from_scene(scene, budget=BudgetSpec(**seeded.budget)) for scene in ordinary_scenes]
            warmup_requests, warmup_block = (
                [],
                {"policy": "none", "reason": "latency_warmup is 0 or latency is not measured"},
            )
            if seeded.measure_latency and seeded.latency_warmup > 0:
                warmup_requests, warmup_block = _load_warmup_requests(
                    seeded, warmup, seeded.latency_warmup, dataset_location, requests
                )
            if session is not None:
                block = _session_seed(
                    seeded,
                    split,
                    config,
                    environment,
                    session,
                    requests,
                    warmup_requests,
                    seed_output,
                    clock=clock,
                    seed=int(seed),
                    base_args=(content_hash, reference_identity, hardware, roster_block, warmup_block, purpose_checks),
                )
                _write_session_receipt(Path(output), int(seed), block, environment.scrub)
                if block["run_artifact"] is not None:
                    paths.append(seed_output)
                continue
            with build_runner(seeded, split, config, environment, seed=int(seed), clock=clock) as runner:
                _measure_baselines(runner, requests, split)
                base = _base_manifest(
                    seeded,
                    split,
                    config,
                    runner,
                    environment,
                    content_hash,
                    reference_identity,
                    hardware,
                    roster_block=roster_block,
                    warmup_block=warmup_block,
                    purpose_checks=purpose_checks,
                )
                plan = []
                if seeded.measure_latency:
                    plan.append(("cold", 1, seed_output / "cold", 0))
                # A timed series is called warm only after a declared warm-up on disjoint inputs.
                warm = seeded.measure_latency and seeded.latency_warmup > 0
                plan.append(
                    (
                        "warm" if warm else "unwarmed",
                        (seeded.latency_repeats or 1) if seeded.measure_latency else 1,
                        seed_output,
                        seeded.quality_repeat_index,
                    )
                )
                for name, repeats, out_dir, quality_repeat in plan:
                    _run_series(
                        runner,
                        requests,
                        base,
                        name,
                        repeats,
                        out_dir,
                        quality_repeat,
                        warmup_requests=warmup_requests,
                        warmup_count=seeded.latency_warmup,
                    )
            paths.append(seed_output)
    return paths


def _session_seed(
    seeded,
    split,
    config: SupervisorConfig,
    environment: RuntimeEnvironment,
    session: session_mod.SessionPolicy,
    requests: Sequence[PlanRequest],
    warmup_requests: Sequence[PlanRequest],
    seed_output: Path,
    *,
    clock: Callable[[], float],
    seed: int,
    base_args: tuple,
) -> dict[str, Any]:
    """One seed in session mode: admission before the workers start, then :func:`_run_session`."""
    content_hash, reference_identity, hardware, roster_block, warmup_block, purpose_checks = base_args
    budget = BudgetSpec(**seeded.budget)
    admission = session_mod.admission_terms(
        session,
        wall_time_ms=budget.wall_time_ms,
        deadline_grace_ms=config.deadline_grace_ms,
        hard_timeout_ms=config.hard_timeout_ms,
        startup_timeout_s=config.startup_timeout_s,
        roles=1 if split is None else 2,
    )
    roster = [request.sample_id for request in requests]

    def outcome(status, *, closure=None, invalid=None, events=()) -> dict[str, Any]:
        return {
            "policy": session_mod.SESSION_POLICY_ID,
            "status": status,
            "n_tasks": len(roster),
            "n_executed": 0,
            "executed_sample_ids": [],
            "deferred_sample_ids": roster if status == session_mod.CLOSED else [],
            "closure": closure,
            "invalid": invalid,
            "admission": admission,
            "cold_start": None,
            "restart_cap_rule": "max_restarts is at least the batch size; reaching it invalidates the session",
            "events": list(events),
            "run_artifact": None,
        }

    runner = build_runner(seeded, split, config, environment, seed=seed, clock=clock)
    runner.session = session
    # The session has no initial worker lifecycle or IPC baseline. Subsequent warm-ups admit independently.
    required = admission["setup_reserve_ms"]
    remaining = session.remaining_ms()
    if remaining < required:
        closure = {"point": "before_start", "remaining_ms": remaining, "required_ms": required}
        return outcome(session_mod.CLOSED, closure=closure)
    try:

        def base():
            return _base_manifest(
                seeded,
                split,
                config,
                runner,
                environment,
                content_hash,
                reference_identity,
                hardware,
                roster_block=roster_block,
                warmup_block=warmup_block,
                purpose_checks=purpose_checks,
                session=True,
            )

        return _run_session(
            runner,
            requests,
            base,
            seed_output,
            warmup_requests=warmup_requests,
            warmup_count=seeded.latency_warmup,
            admission=admission,
        )
    finally:
        runner.close()
