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

"""Untrusted-side worker: ``python -I -m passage_eval.supervision.worker``.

The worker owns the candidate. It speaks ``passage-ipc/v1`` on private duplicates
of fds 0/1; the real stdin is replaced by /dev/null and stdout by stderr, so
candidate ``print`` calls (or native libraries writing to fd 1) cannot corrupt the
frames. IPC descriptors are non-inheritable, so subprocesses started by the
candidate never hold the pipe open.

Roles
-----
``planner``    ``plan(request) -> PlanResult`` (``predict_path(scene)``/callables are adapted
               with the same ``as_planner`` the in-process runner uses)
``predictor``  ``predict(request) -> numpy.ndarray`` (or a tuple/list of arrays)
``decoder``    ``decode(field, start, goal, obstacles, budget) -> PlanResult``, called with
               exactly those five positional arguments

The worker calls the candidate's ``synchronize()`` (instance method, else module
function) and the runtime completion barrier before every reply, and reports its
own internal spans. Those spans are *self-reported* and non-authoritative; the
supervisor measures the wall clock itself.

Forbidden modules: the supervisor passes ``--forbid-module NAME`` (default ``onnxruntime``). After the
candidate is imported, and again after every request, the worker looks the names up in ``sys.modules``;
a hit is fatal: the worker answers with a ``ForbiddenModule`` error and exits, the supervisor records a
``runtime_error`` and discards the process. This check, and the ``sys_modules_top_level`` list in the load
report, run in the candidate's own process: they are **self-reported** evidence, useful against accidents
and honest mistakes, not against a hostile candidate.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import os
import platform
import random
import resource
import sys
import time
from functools import partial
from typing import Any, Callable

import numpy as np

from passage_eval.execution.artifacts import json_safe
from passage_eval.execution.runner import _raw_array, as_planner
from passage_eval.protocol.contracts import BudgetSpec, GraphSpec, PlanRequest, PlanResult
from passage_eval.supervision import IPC_PROTOCOL, POLICY_DECODER_ONLY, POLICY_ORDINARY_REQUEST
from passage_eval.supervision.protocol import (
    ALLOWED_DTYPES,
    FrameLimits,
    ProtocolError,
    StreamClosed,
    SUPERVISOR_TO_WORKER,
    encode_frame,
    expect_body,
    expect_type,
    read_frame,
)
from passage_eval.supervision.redaction import DECODER_BODY_KEYS, ORDINARY_ARRAYS, ORDINARY_BODY_KEYS

_ROLE_MESSAGE = {"planner": "plan", "predictor": "predict", "decoder": "decode"}
_MAX_TEXT = 512
_MAX_SYS_MODULES = 4096
_FORBIDDEN: tuple[str, ...] = ()


class _FatalWorkerError(Exception):
    """Report an error frame, then stop serving."""


def _ms(start: float, end: float) -> float:
    return max(0.0, 1000.0 * (end - start))


def _forbidden_loaded(*, full_scan: bool = False) -> list[str]:
    """Forbidden module names present in ``sys.modules`` (self-reported evidence, see the module docstring)."""
    if not _FORBIDDEN:
        return []
    if full_scan:
        tops = {name.partition(".")[0] for name in list(sys.modules)}
        return sorted(name for name in _FORBIDDEN if name in tops)
    return sorted(name for name in _FORBIDDEN if name in sys.modules)


def _sys_modules_top_level() -> list[str]:
    tops = sorted({name.partition(".")[0] for name in list(sys.modules)})
    return [name[:128] for name in tops[:_MAX_SYS_MODULES]]


# ---------------------------------------------------------------------------
# process hygiene
# ---------------------------------------------------------------------------


def _claim_ipc_descriptors() -> tuple[int, int]:
    ipc_in, ipc_out = os.dup(0), os.dup(1)  # non-inheritable by default (PEP 446)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)  # candidate stdout -> stderr
    return ipc_in, ipc_out


def _apply_rlimits(args: argparse.Namespace) -> dict[str, list[int | None]]:
    wanted = {"core": (resource.RLIMIT_CORE, 0), "nofile": (resource.RLIMIT_NOFILE, args.rlimit_nofile)}
    for name, flag, kind in (
        ("fsize", args.rlimit_fsize, resource.RLIMIT_FSIZE),
        ("as", args.rlimit_as, resource.RLIMIT_AS),
        ("cpu", args.rlimit_cpu, resource.RLIMIT_CPU),
    ):
        if flag is not None:
            wanted[name] = (kind, flag)
    applied: dict[str, list[int | None]] = {}
    for name, (kind, value) in wanted.items():
        soft, hard = resource.getrlimit(kind)
        if hard != resource.RLIM_INFINITY:
            value = min(value, hard)
        resource.setrlimit(kind, (value, value))  # lowering the hard limit is irreversible
        soft, hard = resource.getrlimit(kind)
        applied[name] = [
            None if soft == resource.RLIM_INFINITY else soft,
            None if hard == resource.RLIM_INFINITY else hard,
        ]
    return applied


def _set_parent_death_signal() -> bool:
    try:
        import ctypes
        import signal

        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(1, int(signal.SIGKILL), 0, 0, 0) == 0  # PR_SET_PDEATHSIG
    except Exception:
        return False


def _isolation_probe() -> dict[str, Any]:
    """Facts observable from inside, gathered before any candidate code is imported."""
    probe: dict[str, Any] = {"pid": os.getpid(), "ppid": os.getppid(), "uid": os.getuid()}
    try:
        probe["root_entries"] = sorted(os.listdir("/"))
    except OSError:
        probe["root_entries"] = None
    try:
        with open("/proc/net/dev", encoding="ascii") as stream:
            probe["net_devices"] = sorted(line.split(":")[0].strip() for line in stream.readlines()[2:] if ":" in line)
    except OSError:
        probe["net_devices"] = None
    probe["cwd"] = os.getcwd()
    return probe


# ---------------------------------------------------------------------------
# framed IO on the private descriptors
# ---------------------------------------------------------------------------


class _Channel:
    def __init__(self, ipc_in: int, ipc_out: int, limits: FrameLimits):
        self.ipc_in, self.ipc_out, self.limits = ipc_in, ipc_out, limits

    def _read_exact(self, count: int) -> bytes:
        chunks, got = [], 0
        while got < count:
            chunk = os.read(self.ipc_in, min(count - got, 1 << 20))
            if not chunk:
                raise StreamClosed(got)
            chunks.append(chunk)
            got += len(chunk)
        return b"".join(chunks)

    def receive(self):
        frame = read_frame(self._read_exact, self.limits, allow_eof=True)
        if frame is not None:
            expect_type(frame, SUPERVISOR_TO_WORKER)
        return frame

    def send(self, msg_type: str, seq: int, body: dict, arrays: dict | None = None) -> None:
        data = memoryview(encode_frame(msg_type, seq, body, arrays, limits=self.limits))
        while data:
            written = os.write(self.ipc_out, data[: 1 << 20])
            data = data[written:]

    def send_error(self, seq: int, error_type: str, text: str, phase: str, fatal: bool) -> None:
        self.send(
            "error", seq, {"error_type": error_type[:200], "error": text[:_MAX_TEXT], "phase": phase, "fatal": fatal}
        )


# ---------------------------------------------------------------------------
# candidate loading
# ---------------------------------------------------------------------------


class _Target:
    def __init__(
        self, role: str, call: Callable, sync_owner: Any, module: Any, entry_module: Any, defining_module: Any
    ):
        self.role, self.call, self.sync_owner = role, call, sync_owner
        self.module, self.entry_module, self.defining_module = module, entry_module, defining_module

    def synchronize(self) -> None:
        """Candidate barrier first, then the same runtime barrier the in-process runner uses."""
        hook = getattr(self.sync_owner, "synchronize", None) if self.sync_owner is not None else None
        if not callable(hook):
            hook = getattr(self.entry_module, "synchronize", None)
        if callable(hook):
            hook()
        from passage_eval.cli import _sync_runtime

        _sync_runtime()

    @property
    def has_synchronize(self) -> bool:
        owner = getattr(self.sync_owner, "synchronize", None) if self.sync_owner is not None else None
        return callable(owner) or callable(getattr(self.entry_module, "synchronize", None))


def _seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed % (2**32))
    if "torch" in sys.modules:
        sys.modules["torch"].manual_seed(seed)


def _method_target(obj: Any, kwargs: dict, method: str) -> tuple[Callable, Any]:
    if isinstance(obj, type):
        instance = obj(**kwargs)
        bound = getattr(instance, method, None)
        if not callable(bound):
            raise TypeError(f"{obj.__name__} must implement {method}()")
        return bound, instance
    if not inspect.isroutine(obj) and callable(getattr(obj, method, None)):
        return getattr(obj, method), obj
    if callable(obj):
        return partial(obj, **kwargs), None
    raise TypeError(f"entry point is neither a class, an object with {method}() nor a callable")


def _load_target(hello: dict[str, Any]) -> _Target:
    role, entry, kwargs = hello["role"], hello["entry"], hello["kwargs"]
    for path in hello["import_paths"]:
        if path not in sys.path:
            sys.path.append(path)  # appended: never shadows the stdlib or installed packages
    _seed(hello["seed"])
    module_name, _, attr = entry.partition(":")
    if not module_name or not attr:
        raise ValueError("entry must be 'module:attr'")
    module = importlib.import_module(module_name)
    obj = getattr(module, attr)
    _seed(hello["seed"])
    defining = inspect.getmodule(obj)
    if role == "planner":
        if isinstance(obj, type):
            instance = obj(**kwargs)
            if not (callable(getattr(instance, "plan", None)) or callable(getattr(instance, "predict_path", None))):
                raise TypeError(f"{entry} must implement plan(request) or predict_path(scene)")
        elif not callable(obj) and (
            callable(getattr(obj, "plan", None)) or callable(getattr(obj, "predict_path", None))
        ):
            instance = obj
        elif callable(obj):
            if hello["inference_format"] == "request":
                instance = partial(obj, **kwargs)
            else:
                from passage_eval.cli import _CallablePlanner

                instance = _CallablePlanner(obj, name="candidate", kwargs=kwargs, input_kind=hello["inference_format"])
        else:
            raise TypeError(f"{entry} is neither a Planner class nor callable")
        planner = as_planner(instance)
        return _Target(
            role, planner.plan, instance if not isinstance(instance, partial) else None, module, module, defining
        )
    method = "predict" if role == "predictor" else "decode"
    call, owner = _method_target(obj, kwargs, method)
    return _Target(role, call, owner, module, module, defining)


def _loaded_files(target: _Target) -> list[str]:
    files = set()
    for module in (target.entry_module, target.defining_module):
        path = getattr(module, "__file__", None)
        if path and os.path.isfile(path):
            files.add(os.path.realpath(path))
    return sorted(files)


# ---------------------------------------------------------------------------
# request handling
# ---------------------------------------------------------------------------


def _build_request(frame, received_at: float) -> PlanRequest:
    body = expect_body(frame.body, ORDINARY_BODY_KEYS)
    if body["policy"] != POLICY_ORDINARY_REQUEST:
        raise ProtocolError("wrong redaction policy for this message")
    arrays = frame.arrays
    if not set(arrays) <= ORDINARY_ARRAYS or not {"elevation", "obstacles"} <= set(arrays):
        raise ProtocolError("unexpected arrays in an ordinary request")
    remaining = body["remaining_wall_ms"]
    if remaining is not None and (isinstance(remaining, bool) or not isinstance(remaining, (int, float))):
        raise ProtocolError("invalid remaining_wall_ms")
    return PlanRequest(
        body["token"],
        arrays["elevation"],
        arrays["obstacles"],
        tuple(body["start"]),
        tuple(body["goal"]),
        body["cost_model"],
        body["cost_weight"],
        graph=GraphSpec(**body["graph"]),
        budget=BudgetSpec(**body["budget"]),
        metadata=body["metadata"],
        input_tensor=arrays.get("input_tensor"),
        deadline_monotonic=None if remaining is None else received_at + remaining / 1000.0,
    )


def _plan_message(result: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not isinstance(result, PlanResult):
        raise TypeError("candidate must return PlanResult")
    raw, malformed = _raw_array(result.path)
    if raw.dtype.newbyteorder("<").str not in ALLOWED_DTYPES:
        raw, malformed = np.empty((0, 2), dtype=np.float64), f"unsupported path dtype {raw.dtype}"
    body = {
        "status": result.status,
        "stage": result.stage,
        "precision": result.precision,
        "work": json_safe(dict(result.work)),
        "stage_work": json_safe(result.stage_work),
        "stage_spans": json_safe(result.stage_spans),
        "resource_peaks": json_safe(result.resource_peaks),
        "diagnostics": json_safe(dict(result.diagnostics)),
        "malformed": None if malformed is None else str(malformed)[:4096],
    }
    return body, {"path": raw}


def _field_message(result: Any) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if isinstance(result, np.ndarray):
        fields, kind = [result], "array"
    elif isinstance(result, (tuple, list)) and 1 <= len(result) <= 8 and all(isinstance(x, np.ndarray) for x in result):
        fields, kind = list(result), "sequence"
    else:
        raise TypeError("predict must return a numpy.ndarray (or a tuple/list of 1-8 arrays)")
    arrays = {}
    for index, field_array in enumerate(fields):
        if field_array.dtype.kind != "f" or field_array.dtype.itemsize > 8:
            raise TypeError("predicted fields must be real floating-point arrays")
        arrays[f"field_{index}"] = field_array
    return {"field_kind": kind, "field_count": len(fields)}, arrays


def _serve_request(target: _Target, channel: _Channel, frame) -> bool:
    """Handle one plan/predict/decode message. Returns False if the worker must stop."""
    received = time.perf_counter()
    try:
        if target.role == "decoder":
            body = expect_body(frame.body, DECODER_BODY_KEYS)
            if body["policy"] != POLICY_DECODER_ONLY:
                raise ProtocolError("wrong redaction policy for this message")
            if (
                body["field_kind"] != "array"
                or body["field_count"] != 1
                or isinstance(body["field_count"], bool)
                or set(frame.arrays) != {"obstacles", "field_0"}
            ):
                raise ProtocolError("decoder message does not match decoder-only/v1")
            argument = (
                frame.arrays["field_0"],
                tuple(body["start"]),
                tuple(body["goal"]),
                frame.arrays["obstacles"],
                BudgetSpec(**body["budget"]),
            )
        else:
            argument = (_build_request(frame, received),)
    except ProtocolError:
        raise
    except Exception as error:
        channel.send_error(frame.seq, type(error).__name__, str(error), "request", False)
        return True
    built = time.perf_counter()
    outcome, failure = None, None
    try:
        outcome = target.call(*argument)
    except (SystemExit, KeyboardInterrupt, GeneratorExit) as error:
        failure = error
    except Exception as error:
        failure = error
    called = time.perf_counter()
    try:
        target.synchronize()
    except BaseException as error:  # ownership of device work is uncertain: stop serving
        channel.send_error(frame.seq, type(error).__name__, f"synchronize failed: {error}", "synchronize", True)
        return False
    synced = time.perf_counter()
    banned = _forbidden_loaded()
    if banned:
        channel.send_error(
            frame.seq, "ForbiddenModule", f"forbidden module(s) present in sys.modules: {banned}", "call", True
        )
        return False
    if failure is not None:
        fatal = isinstance(failure, (SystemExit, KeyboardInterrupt, GeneratorExit))
        channel.send_error(frame.seq, type(failure).__name__, str(failure), "call", fatal)
        return not fatal
    try:
        reply, arrays = _field_message(outcome) if target.role == "predictor" else _plan_message(outcome)
    except Exception as error:
        channel.send_error(frame.seq, type(error).__name__, str(error), "pack", False)
        return True
    packed = time.perf_counter()
    spans = {
        "request_build_ms": _ms(received, built),
        "call_ms": _ms(built, called),
        "sync_ms": _ms(called, synced),
        "pack_ms": _ms(synced, packed),
    }
    reply.update(
        kind="field" if target.role == "predictor" else "plan",
        spans=spans,
        candidate_reported_ms=spans["call_ms"] + spans["sync_ms"],
    )
    try:
        channel.send("result", frame.seq, reply, arrays)
    except ValueError as error:  # e.g. a result larger than the frame limits
        channel.send_error(frame.seq, type(error).__name__, str(error), "pack", False)
    return True


def _serve_ping(channel: _Channel, frame) -> None:
    expect_body(frame.body, {"echo"})
    if frame.body["echo"] == "full":
        channel.send("echo", frame.seq, {"echo": "full"}, frame.arrays)
    elif frame.body["echo"] == "small":
        channel.send("echo", frame.seq, {"echo": "small"}, {"small": np.zeros(8, dtype=np.uint8)})
    else:
        raise ProtocolError("unknown echo mode")


def _hello(channel: _Channel, frame, applied_limits, pdeathsig: bool, probe) -> _Target:
    """Attest first, import the candidate second.

    ``ready`` is sent before any candidate code runs, so the attestation (rlimits actually in
    force, what the process can see) cannot have been influenced by the candidate's import-time
    code. The files the candidate actually loaded follow in a second frame.
    """
    started = time.perf_counter()
    body = expect_body(frame.body, {"role", "entry", "kwargs", "inference_format", "seed", "import_paths", "ipc"})
    if (
        body["ipc"] != IPC_PROTOCOL
        or body["role"] not in _ROLE_MESSAGE
        or not isinstance(body["entry"], str)
        or not isinstance(body["kwargs"], dict)
        or body["inference_format"] not in {"request", "tensor", "scene"}
        or not isinstance(body["import_paths"], list)
        or any(not isinstance(p, str) for p in body["import_paths"])
        or (body["seed"] is not None and (isinstance(body["seed"], bool) or not isinstance(body["seed"], int)))
    ):
        raise ProtocolError("invalid hello")
    channel.send(
        "ready",
        frame.seq,
        {
            "role": body["role"],
            "ipc": IPC_PROTOCOL,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "rlimits": applied_limits,
            "pdeathsig": pdeathsig,
            "isolation_probe": probe,
        },
    )
    try:
        target = _load_target(body)
        target.synchronize()
        banned = _forbidden_loaded(full_scan=True)
        if banned:
            raise RuntimeError(f"forbidden module(s) present in sys.modules after import: {banned}")
    except Exception as error:
        kind = "ForbiddenModule" if "forbidden module" in str(error) else type(error).__name__
        channel.send_error(frame.seq, kind, str(error), "load", True)
        raise _FatalWorkerError(str(error)) from None
    channel.send(
        "result",
        frame.seq,
        {
            "kind": "loaded",
            "loaded_files": _loaded_files(target),
            "has_synchronize": target.has_synchronize,
            "load_ms": _ms(started, time.perf_counter()),
            "sys_modules_top_level": _sys_modules_top_level(),
            "forbidden_modules": list(_FORBIDDEN),
        },
    )
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="passage_eval.supervision.worker")
    parser.add_argument("--rlimit-core", type=int, default=0)
    parser.add_argument("--rlimit-nofile", type=int, default=1024)
    parser.add_argument("--rlimit-fsize", type=int, default=None)
    parser.add_argument("--rlimit-as", type=int, default=None)
    parser.add_argument("--rlimit-cpu", type=int, default=None)
    parser.add_argument("--max-header-bytes", type=int, default=FrameLimits().max_header_bytes)
    parser.add_argument("--max-payload-bytes", type=int, default=FrameLimits().max_payload_bytes)
    parser.add_argument("--max-dim", type=int, default=FrameLimits().max_dim)
    parser.add_argument("--max-elements", type=int, default=FrameLimits().max_elements)
    parser.add_argument(
        "--forbid-module",
        action="append",
        default=[],
        metavar="NAME",
        help="top-level module name that must never appear in sys.modules (fatal)",
    )
    args = parser.parse_args(argv)
    if args.rlimit_core != 0:
        parser.error("core dumps must stay disabled")
    global _FORBIDDEN
    _FORBIDDEN = tuple(sorted(set(args.forbid_module)))
    ipc_in, ipc_out = _claim_ipc_descriptors()
    pdeathsig = _set_parent_death_signal()
    applied = _apply_rlimits(args)
    probe = _isolation_probe()
    channel = _Channel(
        ipc_in,
        ipc_out,
        FrameLimits(
            max_header_bytes=args.max_header_bytes,
            max_payload_bytes=args.max_payload_bytes,
            max_dim=args.max_dim,
            max_elements=args.max_elements,
        ),
    )
    target: _Target | None = None
    try:
        while True:
            frame = channel.receive()
            if frame is None or frame.type == "shutdown":
                return 0
            if target is None:
                if frame.type != "hello":
                    raise ProtocolError("the first message must be hello")
                target = _hello(channel, frame, applied, pdeathsig, probe)
            elif frame.type == "ping":
                _serve_ping(channel, frame)
            elif frame.type == _ROLE_MESSAGE[target.role]:
                if not _serve_request(target, channel, frame):
                    return 4
            else:
                raise ProtocolError(f"message {frame.type!r} is not valid for role {target.role!r}")
    except _FatalWorkerError:
        return 5
    except ProtocolError as error:
        try:
            channel.send_error(0, "ProtocolError", str(error), "protocol", True)
        except Exception:
            pass
        return 3
    except BrokenPipeError:
        return 6


if __name__ == "__main__":
    sys.exit(main())
