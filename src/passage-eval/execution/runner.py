"""Synchronous, safe-completion execution of complete ordinary-input requests.

The timer includes request copying, adapter preprocessing, path materialization,
legality, physical recosting and device completion synchronization. Persistence,
reference comparisons and geometry scoring are outside the request timer.

Budgets are cooperative: uncancellable Python/native calls drain before another
request starts. A late route remains a timeout even if it is otherwise legal.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Any

import numpy as np

from passage_eval.execution.artifacts import RunArtifact, SCHEMA, json_safe, route_hash, write_run_artifact
from passage_eval.metrics.cost import path_cost
from passage_eval.metrics.reliability import path_status
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult
from passage_eval.protocol.planner import Scene, PlannerOutput, validated_raw_path
from passage_eval.protocol.attribution import component_control


class PendingCompletionError(RuntimeError):
    """A device completion failed; candidate buffers remain leased in-process.

    Call drain_pending_requests() after restoring completion capability. A new
    run is rejected until every retained lease has completed safely.
    """


_pending_leases: list[tuple[Callable[[], None], Any, Any, Any, Any, np.ndarray]] = []


def drain_pending_requests() -> int:
    """Synchronize retained device work and release only safely drained leases."""
    drained = 0
    while _pending_leases:
        synchronize, adapter, candidate, candidate_output, result, raw = _pending_leases[0]
        try:
            synchronize()
        except BaseException as error:
            raise PendingCompletionError("completion still pending; request buffers retained") from error
        _pending_leases.pop(0)
        if isinstance(adapter, DecoderPlannerAdapter):
            adapter._retained_fields.clear()
        drained += 1
    return drained


class LegacyPlannerAdapter:
    """Adapt existing predict_path(scene) without labels or oracle metadata."""

    def __init__(self, planner):
        self.planner = planner

    def plan(self, request: PlanRequest) -> PlanResult:
        scene = Scene(
            request.sample_id,
            request.elevation,
            request.obstacles,
            request.start,
            request.goal,
            request.cost_model,
            request.cost_weight,
            [],
            float("nan"),
            request.metadata,
            request.input_tensor,
        )
        output = self.planner.predict_path(scene)
        if isinstance(output, PlanResult):
            return output
        if not isinstance(output, PlannerOutput):
            raise TypeError("predict_path must return PlannerOutput or PlanResult")
        return PlanResult(
            output.predicted_path,
            diagnostics={"adapter": "legacy_predict_path", "candidate_reported_cost": output.predicted_cost},
        )


class DecoderPlannerAdapter:
    """Wrap a field predictor and ordinary-input decoder as one timed procedure.

    predictor(request) -> field; decoder(field, request) -> PlanResult/path.
    No reference reachability, labels or target maxima enter either callable.
    """

    def __init__(self, predictor, decoder, *, precision: str = "unspecified"):
        self.predictor, self.decoder, self.precision = predictor, decoder, precision
        self._retained_fields = []

    def plan(self, request: PlanRequest) -> PlanResult:
        field = self.predictor(request)
        self._retained_fields.append(field)
        output = self.decoder(field, request)
        if isinstance(output, PlanResult):
            return replace(output, precision=self.precision) if output.precision == "unspecified" else output
        return PlanResult(output, precision=self.precision, diagnostics={"adapter": "ordinary_input_field_decoder"})


def as_planner(planner):
    if callable(getattr(planner, "plan", None)):
        return planner
    if callable(getattr(planner, "predict_path", None)):
        return LegacyPlannerAdapter(planner)
    if callable(planner):

        class CallablePlanner:
            def plan(self, request):
                result = planner(request)
                return result if isinstance(result, PlanResult) else PlanResult(result)

        return CallablePlanner()
    raise TypeError("planner must implement plan(request) or predict_path(scene)")


def _raw_array(path):
    try:
        points = np.asarray(path)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64), None
        if points.ndim > 2 or points.dtype.kind not in "iuf":
            return np.empty((0, 2), dtype=np.float64), repr(path)
        return np.ascontiguousarray(points).copy(), None
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64), repr(path)


def _work(result: PlanResult):
    # Validate raw reports before aggregation so negative/NaN stage work cannot
    # cancel positive work or be hidden by a reported total/peak.
    reports = [result.work, result.resource_peaks, result.stage_spans, *result.stage_work.values()]
    for counters in reports:
        if not isinstance(counters, dict):
            raise ValueError("work/resource/span reports must be mappings")
        for key, value in counters.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, float, np.number))
                or not np.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"invalid work/resource/span counter {key!r}")
    work = dict(result.work)
    stage_totals = {}
    for stage, counters in result.stage_work.items():
        for key, value in counters.items():
            # Peaks are maxima; cumulative work charges every attempt/stage.
            if key in {"live_states", "peak_live_states", "max_live_states", "memory_bytes"}:
                stage_totals[key] = max(stage_totals.get(key, 0), value)
            else:
                stage_totals[key] = stage_totals.get(key, 0) + value
    for key, value in stage_totals.items():
        work[key] = max(work.get(key, 0), value)
    live_peaks = [
        report[key]
        for report in (work, result.resource_peaks)
        for key in ("live_states", "peak_live_states", "max_live_states")
        if key in report
    ]
    if live_peaks:
        work["peak_live_states"] = max(live_peaks)
    for counters in (work, result.resource_peaks, result.stage_spans):
        for key, value in counters.items():
            if not isinstance(value, (int, float, np.number)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"invalid work/resource/span counter {key!r}")
    return work


def _budget_status(budget: BudgetSpec, work):
    exceeded, missing = [], []
    for counter, limit in (
        ("expansions", budget.max_expansions),
        ("queue_operations", budget.max_queue_operations),
        ("live_states", budget.max_live_states),
    ):
        if limit is not None:
            observed = (
                max(work.get(key, 0) for key in ("live_states", "peak_live_states", "max_live_states"))
                if counter == "live_states"
                and any(key in work for key in ("live_states", "peak_live_states", "max_live_states"))
                else work.get(counter)
            )
            if observed is None:
                missing.append(counter)
            elif observed > limit:
                exceeded.append(counter)
    return exceeded, missing


def run_requests(
    planner,
    requests: Iterable[PlanRequest],
    *,
    repeats: int = 1,
    output_dir: str | Path | None = None,
    manifest: dict | None = None,
    clock: Callable[[], float] = time.perf_counter,
    synchronize: Callable[[], None] | None = None,
    quality_repeat_index: int = 0,
    series: str = "warm",
) -> RunArtifact:
    if _pending_leases:
        raise PendingCompletionError("previous request completion is pending; buffers retained")
    requests = list(requests)
    if not requests or not all(isinstance(request, PlanRequest) for request in requests):
        raise ValueError("requests must be a nonempty sequence of PlanRequest")
    roster = [request.sample_id for request in requests]
    if len(set(roster)) != len(roster):
        raise ValueError("request roster must be unique")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError("repeats must be a positive integer")
    if (
        isinstance(quality_repeat_index, bool)
        or not isinstance(quality_repeat_index, int)
        or not 0 <= quality_repeat_index < repeats
    ):
        raise ValueError("quality_repeat_index must select a declared repeat")
    if not isinstance(series, str) or not series:
        raise ValueError("series must be a nonempty name")
    adapter = as_planner(planner)
    # Identity hashes bind scoring to precisely the ordinary terrain and graph.
    request_specs = {}
    for request in requests:
        request_specs[request.sample_id] = request_identity(request)
    run_manifest = dict(json_safe(manifest or {}))
    control = component_control(run_manifest.get("component_control", {}))
    if control:
        run_manifest["component_control"] = control
    for reserved in (
        "schema",
        "roster",
        "repeats",
        "quality_repeat_index",
        "series",
        "request_specs",
        "request_metadata",
    ):
        if reserved in run_manifest:
            raise ValueError(f"manifest cannot override runner field {reserved}")
    run_manifest.update(
        schema=SCHEMA,
        roster=roster,
        repeats=repeats,
        quality_repeat_index=quality_repeat_index,
        series=series,
        request_specs=request_specs,
        request_metadata={request.sample_id: request.metadata for request in requests},
        timing_boundary="request_copy_through_validation_recost_completion",
        budget_enforcement="cooperative_safe_completion",
        cancellation_policy="drain_before_next_request",
    )
    records, paths = [], []
    for repeat in range(repeats):
        for request in requests:
            # Previous device work is outside this request's service boundary.
            # Failure here blocks admission; no request has been attempted yet.
            if synchronize is not None:
                synchronize()
            started = clock()
            result = PlanResult([], status="runtime_error")
            diagnostics, work, raw = {}, {}, np.empty((0, 2), dtype=np.float64)
            connected, violated, obstacle_cells, legal, recost = False, False, 0, False, float("nan")
            candidate_finished = started
            candidate = None
            candidate_output = None
            device_completed = False
            fallback_used = False
            status = "runtime_error"
            try:
                deadline_monotonic = (
                    started + request.budget.wall_time_ms / 1000 if request.budget.wall_time_ms is not None else None
                )
                candidate = request.candidate_copy(deadline_monotonic=deadline_monotonic)
                try:
                    candidate_output = adapter.plan(candidate)
                    result = candidate_output
                finally:
                    candidate_finished = clock()
                    # Device outputs must be complete before any CPU snapshot,
                    # legality check or recost consumes their bytes.
                    if synchronize is not None:
                        try:
                            synchronize()
                        except BaseException as error:
                            _pending_leases.append((synchronize, adapter, candidate, candidate_output, result, raw))
                            raise PendingCompletionError(
                                "completion failed; request buffers retained until safe drain"
                            ) from error
                    device_completed = True
                if not isinstance(result, PlanResult):
                    raise TypeError("plan(request) must return PlanResult")
                fallback_used = result.stage == "fallback" or "fallback" in result.stage_work
                raw, malformed = _raw_array(result.path)
                path = validated_raw_path(raw)
                connected, violated, obstacle_cells = path_status(
                    path, start=request.start, goal=request.goal, obstacles=request.obstacles
                )
                recost = path_cost(request.elevation, path, weight=request.cost_weight, cost_model=request.cost_model)
                legal = connected and not violated and np.isfinite(recost)
                diagnostics = json_safe(dict(result.diagnostics))
                result = replace(
                    result,
                    diagnostics=diagnostics,
                    stage_work=json_safe(result.stage_work),
                    stage_spans=json_safe(result.stage_spans),
                    resource_peaks=json_safe(result.resource_peaks),
                )
                if malformed is not None:
                    diagnostics["malformed_raw_output"] = malformed
                work = _work(result)
                status = result.status
                if status == "ok" and not legal:
                    status = "no_path" if raw.size == 0 and malformed is None else "invalid_path"
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
            except PendingCompletionError:
                raise
            except MemoryError as error:
                status = "oom"
                diagnostics = dict(error_type=type(error).__name__, error=str(error))
                result = PlanResult(raw, status=status)
            except (asyncio.CancelledError, concurrent.futures.CancelledError) as error:
                status = "cancelled"
                diagnostics = dict(error_type=type(error).__name__, error=str(error))
                result = PlanResult(raw, status=status)
            except Exception as error:
                status = "runtime_error"
                diagnostics = dict(error_type=type(error).__name__, error=str(error))
                result = PlanResult(raw, status=status)
            # Never release buffers or submit the next request while async work
            # remains. A synchronization failure leaves ownership uncertain and
            # aborts the run; fabricating a safe terminal artifact is forbidden.
            if synchronize is not None and not device_completed:
                try:
                    synchronize()
                except BaseException as error:
                    _pending_leases.append((synchronize, adapter, candidate, candidate_output, result, raw))
                    raise PendingCompletionError(
                        "completion failed; request buffers retained until safe drain"
                    ) from error
            completed = clock()
            elapsed_ms = 1000 * (completed - started)
            if not np.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise ValueError("clock must be finite and monotonic")
            deadline = request.budget.wall_time_ms
            late = deadline is not None and elapsed_ms > deadline
            if late:
                diagnostics["status_before_deadline_policy"] = status
                status = "timeout"
            failed_attempt_ms = min(elapsed_ms, deadline) if late else elapsed_ms if status != "ok" else 0.0
            drain_ms = max(0.0, elapsed_ms - deadline) if late else 0.0
            record = dict(
                sample_id=request.sample_id,
                repeat_index=repeat,
                series=series,
                status=status,
                route_hash=route_hash(raw),
                recost=recost,
                connected_path=connected,
                legal_path=legal,
                obstacle_violation=violated,
                obstacle_cells=obstacle_cells,
                total_wall_ms=elapsed_ms,
                candidate_wall_ms=max(0.0, 1000 * (candidate_finished - started)),
                failed_attempt_ms=failed_attempt_ms,
                drain_ms=drain_ms,
                deadline_ms=deadline,
                late_completion=late,
                safe_completion=True,
                stage=result.stage,
                stage_work=result.stage_work,
                work=work,
                fallback=fallback_used,
                precision=result.precision,
                stage_spans=result.stage_spans,
                resource_peaks=result.resource_peaks,
                diagnostics=diagnostics,
            )
            # Nonfinite raw coordinates are retained in NPZ; JSON numeric fields
            # use null instead of nonstandard NaN tokens.
            records.append(json_safe(record))
            paths.append(raw)
            if isinstance(adapter, DecoderPlannerAdapter):
                adapter._retained_fields.clear()
    artifact = RunArtifact(run_manifest, records, paths)
    artifact.validate()
    if output_dir is not None:
        write_run_artifact(artifact, output_dir)
    return artifact


def request_identity(request: PlanRequest) -> dict[str, Any]:
    """Content identity shared with offline scoring and qualification adapters."""
    import hashlib
    import json

    def grid_hash(grid):
        grid = np.ascontiguousarray(grid)
        return hashlib.sha256(grid.dtype.str.encode() + repr(grid.shape).encode() + grid.tobytes()).hexdigest()

    return dict(
        graph=request.graph.to_dict(),
        graph_id=request.graph.identity(),
        cost_model=request.cost_model,
        cost_weight=request.cost_weight,
        start=list(request.start),
        goal=list(request.goal),
        shape=list(request.elevation.shape),
        elevation_hash=grid_hash(request.elevation),
        obstacles_hash=grid_hash(request.obstacles),
        input_tensor_hash=grid_hash(request.input_tensor) if request.input_tensor is not None else None,
        ordinary_metadata_hash=hashlib.sha256(
            json.dumps(request.metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        task_variant=request.metadata.get("task_variant", "obstacles"),
        budget=request.budget.to_dict(),
    )
