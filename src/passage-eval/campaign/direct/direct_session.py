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

"""Prototype persistent direct native session. No network or references in timed boundary."""

from __future__ import annotations
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from passage_eval.campaign.direct.raw_identity import raw_digest

CALIBRATION = (-430.0, 8767.0)


def _one(runtime, decoders, procedure, request, clock_ns):
    # Request arrays are already materialized. Tensor, transfer, inference, synchronization,
    # decode and path emission all happen between these clock reads.
    begin = clock_ns()
    sync_in_progress = False
    try:
        tensor = decoders.build_input(request.elevation, request.obstacles, *request.start, *request.goal, *CALIBRATION)
        output = np.asarray(runtime.predict(tensor))
        sync_in_progress = True
        runtime.synchronize()
        sync_in_progress = False
        if (
            output.shape != (1, 1, *request.elevation.shape)
            or output.dtype != np.float32
            or not np.isfinite(output).all()
        ):
            raise ValueError("native output contract differs")
        field = np.ascontiguousarray(output[0, 0])
        if procedure == "seg":
            result = decoders.seg_logit_astar(
                field, request.obstacles, request.start, request.goal, max_expansions=request.budget.max_expansions
            )
        else:
            result = decoders.reg_bellman_descent(field, request.obstacles, request.start, request.goal)
        path_array = np.asarray(result["path"])
        if path_array.size == 0:
            path = []
        elif (
            path_array.ndim == 2
            and path_array.shape[1] == 2
            and path_array.dtype.kind in "iuf"
            and np.isfinite(path_array).all()
        ):
            path = path_array.tolist()
        else:
            raise ValueError("decoder path shape or values differ")
        status = result["status"]
        if status not in {"ok", "no_path", "budget_exhausted"} or (status == "ok" and not path):
            raise ValueError("decoder output contract differs")
        work = {k: int(result[k]) for k in ("expansions", "descent_steps", "fallback_expansions") if k in result}
        fallback = bool(result.get("fallback", False))
        error = None
    except Exception as exc:
        if sync_in_progress:
            raise RuntimeError("runtime did not drain; session invalid") from exc
        # Never continue while an asynchronous runtime may still be using buffers.
        try:
            runtime.synchronize()
        except Exception as sync_error:
            raise RuntimeError("runtime did not drain; session invalid") from sync_error
        path, status, work, fallback = [], "runtime_error", {}, False
        error = type(exc).__name__ + ": " + str(exc)[:200]
    end = clock_ns()
    if end < begin:
        raise ValueError("monotonic clock moved backwards")
    elapsed = end - begin
    return {
        "status": status,
        "path": path,
        "fallback": fallback,
        "work": work,
        "error": error,
        "elapsed_ns": elapsed,
        "elapsed_seconds": f"{elapsed // 1_000_000_000}.{elapsed % 1_000_000_000:09d}",
    }


def check_budget(request, procedure):
    budget = request.budget
    if budget.max_queue_operations is not None or budget.max_live_states is not None:
        raise ValueError("unsupported queue or live-state budget")
    if procedure == "reg" and budget.max_expansions is not None and budget.max_expansions < request.elevation.size + 1:
        raise ValueError("unsupported Reg expansion limit below certified bound")


def _append(stream, value, *, durable=False):
    stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()
    if durable:
        os.fsync(stream.fileno())


def run_direct_session(
    runtime,
    decoders,
    procedure,
    warmup_requests,
    blocks,
    ordered_tasks,
    journal_path,
    *,
    clock_ns=time.perf_counter_ns,
    checkpoint_size=32,
    _journal_opened=None,
):
    """One already initialized native runtime; exactly 16 untimed warmups then a streamed pass.

    A deadline is detected after synchronous completion, never preempted. This is an exploratory
    prototype: the caller must verify and pin model/runtime/decoder identities and block contents.
    """
    if type(checkpoint_size) is not int or checkpoint_size < 1:
        raise ValueError("checkpoint size must be positive")
    if procedure not in {"seg", "reg"} or len(warmup_requests) != 1 or not ordered_tasks:
        raise ValueError("procedure, one warmup scene and ordered tasks required")
    if any(not isinstance(t, dict) or set(t) != {"sample_id", "raw_input_sha256"} for t in ordered_tasks):
        raise ValueError("lightweight task index fields differ")
    if any(r.input_tensor is not None or r.metadata for r in warmup_requests):
        raise ValueError("warmup must contain raw arrays only")
    ids = [task["sample_id"] for task in ordered_tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("task IDs must be unique")
    for request in warmup_requests:
        check_budget(request, procedure)
    warm_ids = [r.sample_id for r in warmup_requests]
    warm_hashes = [raw_digest(r) for r in warmup_requests]
    measured_hashes = {t["raw_input_sha256"] for t in ordered_tasks}
    if set(warm_ids) & set(ids) or set(warm_hashes) & measured_hashes:
        raise ValueError("warmups must be distinct and disjoint")
    path = Path(journal_path)
    completed = 0
    request_sum_ns = 0
    session_id = uuid.uuid4().hex
    session_start = time.perf_counter_ns()
    warmup_start = time.perf_counter_ns()
    with path.open("x", encoding="utf-8") as stream:
        if _journal_opened is not None:
            _journal_opened(stream)
        _append(
            stream,
            {
                "kind": "session_start",
                "session_id": session_id,
                "namespace_pid": os.getpid(),
                "marker_monotonic_ns": time.monotonic_ns(),
                "marker_unix_ns": time.time_ns(),
                "schema": "passage-direct-session-prototype/v1",
                "procedure": procedure,
                "n_expected": len(ids),
                "warmup_required": 16,
                "timing_boundary": "raw_in_memory_arrays_to_emitted_path_non_supervised",
                "deadline_policy": "cooperative_after_safe_completion",
            },
            durable=True,
        )
        for _ in range(16):
            row = _one(runtime, decoders, procedure, warmup_requests[0], clock_ns)
            if row["status"] != "ok":
                _append(
                    stream,
                    {"kind": "session_invalid", "reason": "warmup_failed", "status": row["status"]},
                    durable=True,
                )
                raise RuntimeError("warmup failed")
        warmup_ns = time.perf_counter_ns() - warmup_start
        _append(
            stream,
            {"kind": "warmup_complete", "count": 16, "distinct_inputs": 1, "warmup_wall_ns": warmup_ns},
            durable=True,
        )
        for block in blocks:
            for request in block:
                if request.input_tensor is not None or request.metadata:
                    _append(stream, {"kind": "session_invalid", "reason": "nonraw_input"}, durable=True)
                    raise ValueError("request must contain raw arrays only")
                check_budget(request, procedure)
                identity_sha = raw_digest(request)
                if (
                    completed >= len(ids)
                    or request.sample_id != ids[completed]
                    or identity_sha != ordered_tasks[completed]["raw_input_sha256"]
                ):
                    _append(stream, {"kind": "session_invalid", "reason": "input_identity_or_order"}, durable=True)
                    raise ValueError("measured input identity or order differs")
                if completed % checkpoint_size == 0:
                    _append(
                        stream,
                        {"kind": "reservation", "start": completed, "end": min(completed + checkpoint_size, len(ids))},
                        durable=True,
                    )
                _append(stream, {"kind": "request_started", "sample_id": request.sample_id, "sequence": completed})
                row = _one(runtime, decoders, procedure, request, clock_ns)
                budget = request.budget.wall_time_ms
                if budget is not None and row["elapsed_ns"] > budget * 1_000_000:
                    row["status_before_deadline"] = row["status"]
                    row["status"] = "timeout"
                request_sum_ns += row["elapsed_ns"]
                row.update(
                    kind="request", sample_id=request.sample_id, sequence=completed, raw_input_sha256=identity_sha
                )
                _append(stream, row)
                completed += 1
                if completed % checkpoint_size == 0 or completed == len(ids):
                    _append(stream, {"kind": "checkpoint", "n_completed": completed}, durable=True)
        if completed != len(ids):
            _append(stream, {"kind": "session_incomplete", "n_completed": completed}, durable=True)
            raise ValueError("incomplete session")
        session_wall_ns = time.perf_counter_ns() - session_start
        _append(
            stream,
            {
                "kind": "session_complete",
                "session_id": session_id,
                "namespace_pid": os.getpid(),
                "marker_monotonic_ns": time.monotonic_ns(),
                "marker_unix_ns": time.time_ns(),
                "n_completed": completed,
                "request_sum_ns": request_sum_ns,
                "session_wall_ns": session_wall_ns,
            },
            durable=True,
        )
    return {
        "status": "complete",
        "n_completed": completed,
        "warmups_completed": 16,
        "request_sum_ns": request_sum_ns,
        "warmup_wall_ns": warmup_ns,
        "session_wall_ns": session_wall_ns,
    }


def inspect_journal(journal_path):
    """Report a durable prefix and conservative ambiguity; never authorize an automatic replay."""
    lines = Path(journal_path).read_bytes().splitlines()
    events = []
    for i, line in enumerate(lines):
        try:
            events.append(json.loads(line))
        except (ValueError, UnicodeError):
            if i != len(lines) - 1:
                raise ValueError("invalid journal before final record")
    if not events or events[0].get("kind") != "session_start":
        raise ValueError("journal session header absent")
    prefix = 0
    reservation = None
    visible = []
    complete = False
    released = False
    for event in events[1:]:
        kind = event.get("kind")
        if kind == "reservation":
            if (
                reservation is not None
                or event.get("start") != prefix
                or not prefix < event.get("end", -1) <= events[0]["n_expected"]
            ):
                raise ValueError("journal reservation differs")
            reservation = [event["start"], event["end"]]
        elif kind == "request":
            sequence = event.get("sequence")
            if reservation is None or sequence != prefix + len(visible) or sequence >= reservation[1]:
                raise ValueError("journal request order differs")
            visible.append(sequence)
        elif kind == "checkpoint":
            if (
                reservation is None
                or event.get("n_completed") != reservation[1]
                or len(visible) != reservation[1] - prefix
            ):
                raise ValueError("journal checkpoint differs")
            prefix = reservation[1]
            reservation = None
            visible = []
        elif kind == "runtime_released":
            released = True
        elif kind == "runtime_release_failed":
            released = False
        elif kind == "session_complete":
            complete = reservation is None and prefix == events[0]["n_expected"] and event.get("n_completed") == prefix
    return {
        "committed_prefix": prefix,
        "ambiguous_range": reservation,
        "visible_uncommitted": visible,
        "complete": complete,
        "runtime_released": released,
        "automatic_resume_allowed": False,
    }


def execute_direct_candidate(
    runtime_factory,
    decoders,
    procedure,
    warmup_requests,
    blocks,
    ordered_tasks,
    journal_path,
    *,
    clock_ns=time.perf_counter_ns,
    checkpoint_size=32,
):
    """Load a fresh runtime for one candidate, synchronize and close it before returning.

    Process exit and independent GPU-idle checks remain the launcher's responsibility.
    """
    started = time.perf_counter_ns()
    runtime = runtime_factory()
    load_ns = time.perf_counter_ns() - started
    result = None
    release_journal = None

    def retain_journal(stream):
        nonlocal release_journal
        release_journal = os.fdopen(os.dup(stream.fileno()), "a", encoding="utf-8")

    try:
        result = run_direct_session(
            runtime,
            decoders,
            procedure,
            warmup_requests,
            blocks,
            ordered_tasks,
            journal_path,
            clock_ns=clock_ns,
            checkpoint_size=checkpoint_size,
            _journal_opened=retain_journal,
        )
    finally:
        release_started = time.perf_counter_ns()
        release_started_utc = datetime.now(timezone.utc).isoformat()
        try:
            runtime.synchronize()
            runtime.close()
            release = {
                "kind": "runtime_released",
                "load_wall_ns": load_ns,
                "release_wall_ns": time.perf_counter_ns() - release_started,
                "release_started_utc": release_started_utc,
                "release_finished_utc": datetime.now(timezone.utc).isoformat(),
                "gpu_idle_independently_verified": False,
            }
        except BaseException:
            if release_journal is not None:
                with release_journal:
                    _append(release_journal, {"kind": "runtime_release_failed"}, durable=True)
            raise
        if release_journal is not None:
            with release_journal:
                _append(release_journal, release, durable=True)
    return {
        **result,
        "released": True,
        "load_wall_ns": load_ns,
        "release_wall_ns": release["release_wall_ns"],
        "candidate_runtime_envelope_ns": time.perf_counter_ns() - started,
    }
