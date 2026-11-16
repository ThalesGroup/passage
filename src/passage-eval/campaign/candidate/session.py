"""Persistent complete-procedure sessions; timing ends only after synchronized path emission."""

from __future__ import annotations
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from passage_eval.protocol.contracts import BudgetSpec, PlanResult, STATUSES, STAGES
from passage_eval.protocol.candidate_package import CandidatePackage, canonical_bytes
from passage_eval.campaign.candidate.transport import input_digest
from passage_eval.metrics.cost import canonical_cost_model

BOUNDARY = "raw_in_memory_arrays_to_emitted_path_non_supervised"


def _safe_json(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    if isinstance(value, dict) and all(isinstance(k, str) for k in value):
        return {k: _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    raise ValueError("candidate diagnostics must be finite JSON values")


def _counter_map(value):
    value = _safe_json(value)
    if not isinstance(value, dict):
        raise ValueError("work counters must be a mapping")
    for val in value.values():
        if type(val) not in (int, float) or val < 0:
            raise ValueError("work counters must be nonnegative finite numbers")
    return value


def _snapshot(result):
    if not isinstance(result, PlanResult):
        raise ValueError("candidate must return PlanResult")
    path = np.asarray(result.path)
    if path.size == 0:
        path = np.empty((0, 2), dtype=np.int64)
    if (
        path.ndim != 2
        or path.shape[1] != 2
        or len(path) > 1_000_000
        or path.dtype.kind not in "iuf"
        or not np.isfinite(path).all()
    ):
        raise ValueError("candidate path must be materialized finite Nx2 coordinates")
    status = result.status
    if not np.equal(path, np.floor(path)).all():
        status = "invalid_path"
    if status not in STATUSES or result.stage not in STAGES:
        raise ValueError("invalid terminal status or stage")
    if not isinstance(result.stage_work, dict) or not set(result.stage_work) <= STAGES:
        raise ValueError("invalid stage work mapping")
    work = _counter_map(result.work)
    stage_work = {key: _counter_map(value) for key, value in result.stage_work.items()}
    peaks = _counter_map(result.resource_peaks)
    spans = _counter_map(result.stage_spans)
    if not set(spans) <= STAGES or not isinstance(result.precision, str) or not result.precision.strip():
        raise ValueError("invalid stage spans or execution precision")
    diagnostics = _safe_json(result.diagnostics)
    if len(canonical_bytes(diagnostics)) > 65536:
        raise ValueError("candidate diagnostics exceed limit")
    observation = {"stage": result.stage, "precision": result.precision, "resource_peaks": peaks, "stage_spans": spans}
    return path.tolist(), status, work, diagnostics, stage_work, observation


def run_candidate_session(
    candidate,
    requests,
    *,
    manifest: CandidatePackage,
    journal_path: str | Path,
    device="cpu",
    clock_ns=time.perf_counter_ns,
):
    manifest.verify()
    synchronize = getattr(candidate, "synchronize", lambda: None)
    if device == "cuda" and not callable(getattr(candidate, "synchronize", None)):
        raise ValueError("CUDA candidate requires explicit synchronization")
    completed = 0
    seen = set()
    statuses = {}
    with Path(journal_path).open("x") as stream:

        def append(value):
            stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            if completed % 32 == 0:
                os.fsync(stream.fileno())

        append(
            {
                "kind": "session_start",
                "schema": "passage-candidate-session/v1",
                "manifest_sha256": manifest.digest,
                "candidate": manifest.identity,
                "device": device,
                "timing_boundary": BOUNDARY,
                "warmups": 0,
                "deadline_policy": "cooperative_after_safe_completion",
            }
        )
        synchronize()
        for original in requests:
            if original.sample_id in seen:
                raise ValueError("duplicate request identity")
            if (
                canonical_cost_model(original.cost_model)
                != canonical_cost_model(manifest.document["dataset"]["cost_model"])
                or original.cost_weight != manifest.document["dataset"]["cost_weight"]
            ):
                raise ValueError("request objective differs from candidate identity")
            seen.add(original.sample_id)
            raw_hash = input_digest(original)
            begin = clock_ns()
            error = None
            sync_in_progress = False
            try:
                limits = {}
                for name, value in original.budget.to_dict().items():
                    choices = [v for v in (value, manifest.document["budget"].get(name)) if v is not None]
                    limits[name] = min(choices) if choices else None
                budget = BudgetSpec(**limits)
                limit = budget.wall_time_ms
                deadline = time.perf_counter() + limit / 1000 if limit else None
                request = replace(original.candidate_copy(deadline_monotonic=deadline), budget=budget)
                result = candidate.plan(request)
                sync_in_progress = True
                synchronize()
                sync_in_progress = False
                route, status, work, diagnostics, stage_work, observation = _snapshot(result)
                # Configured counters must be reported and include all stages.
                for key, cap in [
                    ("expansions", budget.max_expansions),
                    ("queue_operations", budget.max_queue_operations),
                    ("live_states", budget.max_live_states),
                ]:
                    if cap is not None:
                        subtotal = [v.get(key, 0) for v in stage_work.values()]
                        actual = max(
                            [
                                work.get(key, 0),
                                observation["resource_peaks"].get(key, 0) if key == "live_states" else 0,
                                max(subtotal, default=0) if key == "live_states" else sum(subtotal),
                            ]
                        )
                        if key not in work:
                            raise ValueError(f"missing configured {key} counter")
                        work[key] = actual
                        if actual > cap:
                            status = "budget_exhausted"
            except Exception as exc:
                # A failed drain invalidates the session, rather than starting another request.
                if sync_in_progress:
                    raise RuntimeError("device drain failed; session invalid") from exc
                synchronize()
                route, status, work, diagnostics, stage_work, observation = (
                    [],
                    (
                        "oom"
                        if isinstance(exc, MemoryError) or type(exc).__name__ == "OutOfMemoryError"
                        else "runtime_error"
                    ),
                    {},
                    {},
                    {},
                    {"stage": None, "precision": "unspecified", "resource_peaks": {}, "stage_spans": {}},
                )
                error = type(exc).__name__ + ": " + str(exc)[:500]
            elapsed = clock_ns() - begin
            if elapsed < 0:
                raise ValueError("monotonic clock moved backwards")
            if limit and elapsed > limit * 1_000_000:
                status = "timeout"
            append(
                {
                    "kind": "result",
                    "sample_id": original.sample_id,
                    "raw_input_sha256": raw_hash,
                    "path": route,
                    "status": status,
                    "elapsed_ns": elapsed,
                    "latency_censored": False,
                    "work": work,
                    "stage_work": stage_work,
                    "diagnostics": diagnostics,
                    "error": error,
                    "budget": budget.to_dict(),
                    **observation,
                }
            )
            completed += 1
            statuses[status] = statuses.get(status, 0) + 1
        append({"kind": "session_complete", "n_completed": completed, "status_counts": statuses})
        os.fsync(stream.fileno())
    return {"n_completed": completed, "status_counts": statuses, "timing_boundary": BOUNDARY}
