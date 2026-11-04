# MIT License
#
# Copyright (c) 2026 THALES

"""Aggregate the scored batches of a campaign into one result per procedure.

The six primary metrics are recomputed by ``metrics.summary.aggregate_primary_metrics`` on the **concatenated
per-task rows** of every batch, in parent roster order: a quantile of the campaign is never an average of batch
quantiles. Rows come from the public loader ``metrics.comparison.load_scored``, which checks every scored row
against its authenticated timed record, so a candidate failure stays in the denominator with the canonical penalty
(regret ``1e6 - 1``) and geometry is computed on valid paths only.

Population. ``n_source``, ``n_excluded`` and ``n_admissible`` come from the admissible roster's derivation (the
reference admission inventory); ``n_scored`` must equal ``n_admissible``. Every task keeps its parent geographic
block; a null block is refused, and the sealed ``unplaced`` block of a roster that declares it is a block like any
other (its task count is reported).

Completeness. Exactly one accepted record per admissible task, procedure, target and seed: a task missing from the
batches, or present twice, stops the aggregation (``completeness``).

Rows for statistics. ``task_rows`` keeps the per-task outcomes in parent order and ``latency_sessions`` the warm
requests of each batch (one supervised session per batch and seed), so that ``campaign.statistics`` resamples blocks
of tasks and sessions of requests without reading the scored batches again.

Latency. ``metrics.latency.censored_latency_quantiles`` is applied to the concatenated requests of the warm series
of every session, with the warm-up evidence of every session (``execution.timing.manifest_warmup``: the series is
warm only if every session declared and performed its warm-up) and the per-request warm-up states of the re-warm
profile. A candidate failure imputed by R7 counts as a request whose latency is charged at the wall budget and
censored. The rule ``passage-censored-latency/v1`` is proposed, not sealed; the block says so. The cold start is
reported apart, per session (``cold_start``: number of sessions, median and maximum of each role's worker load time
and of the first warm-up request); it never enters the warm row.

Sessions and units. A batch may run in several sessions (dynamic admission defers tasks to the next window) and R7
may split it into halves; every session contributes its measured prefix, in parent order. A task imputed as a
candidate failure by R7 (a single task that killed its batch twice) enters every quality statistic with the canonical
penalty and is listed under ``imputed_failures``; it has no scorer output, so it is outside ``canonical_equivalence``.
Restarts, re-warms and relaunches are counted per procedure (``restarts``, ``relaunches``).

Equivalence with the canonical scorer. For every batch, the six metrics recomputed on the batch's rows must equal
the batch's ``metrics.json`` written by the scorer (``canonical_equivalence``).

Seeds. ``aggregate_campaign`` reports every seed under ``per_seed``, with the checkpoint and bundle digests its runs
loaded. One seed's metrics, failures and latency are repeated at the top level; with several seeds those top-level
fields are null, because no cross-seed reduction is defined (``combine_seeds``).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from passage_eval.campaign import AGGREGATE_SCHEMA
from passage_eval.execution.artifacts import json_safe
from passage_eval.execution.timing import manifest_warmup
from passage_eval.metrics.comparison import FAILURE_REGRET, load_scored
from passage_eval.metrics.latency import MIN_WARM_REQUESTS, censored_latency_quantiles
from passage_eval.metrics.summary import aggregate_primary_metrics
from passage_eval.protocol.geographic_blocks import UNPLACED_BLOCK

SIX_METRICS = (
    "cost_regret_median",
    "cost_regret_p90",
    "path_assd",
    "path_hausdorff_p95",
    "success_rate",
    "obstacle_violation_rate",
)
ROW_FIELDS = ("regret", "assd", "h95", "success", "obstacle_violation", "valid_path", "obstacle_cells")
EXTRA_ROW_FIELDS = ("status", "fallback", "reference_id", "route_hash")


def _same(a: Any, b: Any) -> bool:
    """Equality of two metric values, with ``None`` (written JSON) equal to ``nan``."""

    def norm(value):
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else value

    a, b = norm(a), norm(b)
    if a is None or b is None:
        return a is None and b is None
    return a == b


def _difference(a: Any, b: Any) -> float:
    try:
        if a is None or b is None or math.isnan(float(a)) or math.isnan(float(b)):
            return 0.0 if _same(a, b) else math.inf
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return math.inf


def _quality_rows(run: Mapping[str, Any], sample_ids: Sequence[str], series: str) -> dict[str, np.ndarray]:
    arrays, indices, manifest = run["arrays"], run["indices"], run["manifest"]
    quality = manifest["quality_repeat_index"]
    rows = [indices[(sample_id, quality, series)] for sample_id in sample_ids]
    return {name: arrays[name][rows] for name in (*ROW_FIELDS, *EXTRA_ROW_FIELDS)}


def _latency_rows(run: Mapping[str, Any], series: str) -> dict[str, Any]:
    arrays = run["arrays"]
    mask = arrays["series"] == series
    states = run.get("warmup_complete")
    return {
        "latency_ms": arrays["latency_ms"][mask],
        "status": arrays["status"][mask],
        "sample_id": arrays["sample_id"][mask],
        "censored": arrays["latency_censored"][mask],
        "states": None if states is None else [state for state, keep in zip(states, mask.tolist()) if keep],
    }


def imputed_row(failure: Mapping[str, Any]) -> dict[str, Any]:
    """The per-task row of an R7 candidate failure: the canonical penalty and no valid path."""
    return {
        "regret": FAILURE_REGRET,
        "assd": math.nan,
        "h95": math.nan,
        "success": False,
        "obstacle_violation": False,
        "valid_path": False,
        "obstacle_cells": 0,
        "status": "runtime_error",
        "fallback": False,
        "reference_id": None,
        "route_hash": None,
    }


def aggregate_scored_batches(
    batches: Sequence[Any],
    *,
    parent_order: Sequence[str],
    parent_blocks: Mapping[str, str | None],
    population: Mapping[str, int],
    series: str = "warm",
    min_warm_requests: int = MIN_WARM_REQUESTS,
) -> dict[str, Any]:
    """Aggregate loaded scored sessions (``load_scored`` outputs, each with its task ids in parent order).

    An element of ``batches`` is ``(run, sample_ids)`` or an R7 imputed failure record
    (``campaign.IMPUTED_SCHEMA``, one task). ``population`` gives ``n_source``, ``n_excluded`` and ``n_admissible``;
    ``parent_order`` lists the admissible tasks; ``parent_blocks`` their parent geographic blocks (a null block is
    refused). Raises ``ValueError`` on any duplicate, missing or misordered task.
    """
    from passage_eval.campaign import IMPUTED_SCHEMA

    if set(population) != {"n_source", "n_excluded", "n_admissible"}:
        raise ValueError("population needs n_source, n_excluded and n_admissible")
    if population["n_source"] != population["n_excluded"] + population["n_admissible"]:
        raise ValueError("n_source must equal n_excluded + n_admissible")
    if population["n_admissible"] != len(parent_order) or len(set(parent_order)) != len(parent_order):
        raise ValueError("the admissible tasks must be unique and number n_admissible")
    missing_blocks = [sample_id for sample_id in parent_order if not parent_blocks.get(sample_id)]
    if missing_blocks:
        raise ValueError(
            f"{len(missing_blocks)} task(s) have a null parent geographic block, e.g. {missing_blocks[:3]}"
        )
    collected: dict[str, dict[str, Any]] = {}
    latency_parts: list[dict[str, Any]] = []
    sessions: list[dict[str, Any]] = []
    equivalence = []
    declared, performed = True, True
    imputed = []
    for number, element in enumerate(batches):
        if isinstance(element, Mapping) and element.get("schema") == IMPUTED_SCHEMA:
            sample_id = element["sample_id"]
            if sample_id in collected:
                raise ValueError(f"task {sample_id} appears in more than one batch")
            collected[sample_id] = imputed_row(element)
            charged = float(element["charged_latency_ms"])
            latency_parts.append(
                {
                    "latency_ms": np.asarray([charged]),
                    "status": np.asarray(["runtime_error"]),
                    "sample_id": np.asarray([sample_id]),
                    "censored": np.asarray([True]),
                    "states": [None],
                    "imputed": True,
                }
            )
            imputed.append({"sample_id": sample_id, "unit": element["unit"], "sha256": element["sha256"]})
            continue
        run, sample_ids = element
        manifest = run["manifest"]
        if list(manifest["roster"]) != list(sample_ids):
            raise ValueError(f"batch {number}: the scored roster differs from the batch's tasks")
        rows = _quality_rows(run, sample_ids, series)
        for position, sample_id in enumerate(sample_ids):
            if sample_id in collected:
                raise ValueError(f"task {sample_id} appears in more than one batch")
            collected[sample_id] = {name: values[position] for name, values in rows.items()}
        # The scorer aggregates its rows in record order; the same order keeps floating sums bit-identical.
        record_order = np.argsort(
            [run["indices"][(s, manifest["quality_repeat_index"], series)] for s in sample_ids], kind="stable"
        )
        recomputed = aggregate_primary_metrics({name: rows[name][record_order] for name in ROW_FIELDS})
        stored = run.get("metrics")
        if stored is None:
            stored = json_load_metrics(run["path"])
        differences = {name: _difference(recomputed[name], stored.get(name)) for name in SIX_METRICS}
        equivalence.append(
            {
                "batch": number,
                "n_tasks": len(sample_ids),
                "equal": all(_same(recomputed[name], stored.get(name)) for name in SIX_METRICS),
                "max_abs_difference": max(differences.values()),
            }
        )
        part = _latency_rows(run, series)
        latency_parts.append(part)
        sessions.append(
            {
                "session": number,
                "sample_ids": part["sample_id"].tolist(),
                "latency_ms": part["latency_ms"].tolist(),
                "censored": part["censored"].tolist(),
                "status": part["status"].tolist(),
                "warmup_complete": part["states"],
            }
        )
        evidence = manifest_warmup(manifest)
        declared = declared and bool(evidence["declared"])
        performed = performed and bool(evidence["performed"])
    if list(collected) != list(parent_order):
        if set(collected) != set(parent_order):
            raise ValueError("the batches do not cover the admissible roster exactly")
        raise ValueError("the batches are not in parent roster order")
    concatenated = {name: np.asarray([collected[s][name] for s in parent_order]) for name in ROW_FIELDS}
    metrics = aggregate_primary_metrics(concatenated)
    statuses = Counter(str(collected[s]["status"]) for s in parent_order)
    failed = ~np.asarray([bool(collected[s]["valid_path"]) for s in parent_order])
    regrets = concatenated["regret"]
    if failed.any() and not np.allclose(regrets[failed], FAILURE_REGRET, rtol=0, atol=1e-9):
        raise ValueError("a failed task lost its canonical penalty")
    scored_states = [part["states"] for part in latency_parts if not part.get("imputed")]
    has_states = bool(scored_states) and all(states is not None for states in scored_states)
    latency = censored_latency_quantiles(
        np.concatenate([part["latency_ms"] for part in latency_parts]),
        status=np.concatenate([part["status"] for part in latency_parts]),
        sample_id=np.concatenate([part["sample_id"] for part in latency_parts]),
        censored=np.concatenate([part["censored"] for part in latency_parts]),
        series=series,
        warmup_complete=[state for part in latency_parts for state in part["states"]] if has_states else None,
        warmup={"declared": declared, "performed": performed, "batches": len(latency_parts)},
        min_warm_requests=min_warm_requests,
    )
    blocks = [parent_blocks[s] for s in parent_order]
    task_rows = {"task_id": list(parent_order)}
    task_rows.update({name: [collected[s][name] for s in parent_order] for name in (*ROW_FIELDS, "status")})
    result = {
        "schema": AGGREGATE_SCHEMA,
        "series": series,
        "verified": False,
        "metrics": {name: metrics[name] for name in SIX_METRICS},
        "metrics_extra": {name: value for name, value in metrics.items() if name not in SIX_METRICS},
        "population": {**dict(population), "n_scored": len(parent_order)},
        "failures": {
            "n_failed": int(failed.sum()),
            "statuses": dict(sorted(statuses.items())),
            "penalty": "a failed task keeps regret 1e6 - 1 (cost 1e6 x reference) in every regret statistic",
            "n_fallback": int(sum(bool(collected[s]["fallback"]) for s in parent_order)),
        },
        "geographic_blocks": {
            "source": "parent_geographic_block of the admissible roster",
            "number_of_blocks": len(set(blocks)),
            "block_sizes": dict(sorted(Counter(blocks).items())),
            "n_unplaced": sum(block == UNPLACED_BLOCK for block in blocks),
            "tasks": dict(zip(parent_order, blocks)),
        },
        "completeness": {
            "rule": "exactly one accepted record per admissible task, procedure, target and seed",
            "n_admissible": population["n_admissible"],
            "n_records": len(collected),
            "complete": len(collected) == population["n_admissible"],
        },
        "task_rows": task_rows,
        "latency_sessions": sessions,
        "latency": latency,
        "canonical_equivalence": {
            "rule": "per batch, the six metrics recomputed on its rows equal the scorer's metrics.json",
            "equal": all(row["equal"] for row in equivalence),
            "max_abs_difference": max((row["max_abs_difference"] for row in equivalence), default=0.0),
            "batches": equivalence,
        },
        "quantile_rule": "every quantile is computed on the concatenated per-task rows; batch quantiles are never averaged",
        "imputed_failures": {
            "rule": "R7: a single task that killed its batch twice with green infrastructure is a candidate failure "
            "(runtime_error, canonical penalty, latency charged at the wall budget and censored)",
            "n": len(imputed),
            "tasks": imputed,
        },
    }
    return json_safe(result)


def json_load_metrics(directory: str | Path) -> dict[str, Any]:
    from passage_eval.io import pinned

    return pinned.read_json(Path(directory) / "metrics.json")


#: Limits stated in every campaign aggregate.
LIMITATIONS = (
    "hashes check byte correspondence, not the authenticity of the operator, weights, SDK or dataset",
    "comparative inference is not provided here; the latency censoring rule is proposed, not sealed",
)
#: Per-seed fields that only one seed's result can fill at the top level.
SEED_RESULT_FIELDS = ("metrics", "metrics_extra", "failures", "latency", "imputed_failures")


def combine_seeds(per_seed: Mapping[str, Mapping[str, Any]], seeds: Sequence[int]) -> dict[str, Any]:
    """The top level of a campaign aggregate over the per-seed aggregates of ``aggregate_scored_batches``.

    ``per_seed`` is always reported. With one seed, the top-level metrics, failures and latency are that seed's.
    With several seeds no cross-seed reduction is defined: those top-level fields are null and every seed is read in
    ``per_seed``. ``canonical_equivalence.equal`` holds only if it holds for every seed.
    """
    keys = [str(seed) for seed in seeds]
    if not keys or sorted(per_seed) != sorted(keys) or len(set(keys)) != len(keys):
        raise ValueError("per_seed must hold exactly one aggregate per seed")
    first = per_seed[keys[0]]
    rows = {key: per_seed[key]["canonical_equivalence"] for key in keys}
    single = len(keys) == 1
    combined = {
        "schema": first["schema"],
        "series": first["series"],
        "verified": False,
        "seeds": list(seeds),
        "seed": seeds[0] if single else None,
        "seed_reduction": "single seed: the top-level metrics, failures and latency are that seed's"
        if single
        else "none: several seeds are reported in per_seed only; no cross-seed reduction is defined",
        "seed_interpretation": (
            {
                "status": "descriptive_single_seed_test_exposed_non_confirmatory",
                "seed": 2026,
            }
            if single and seeds[0] == 2026
            else {"status": "per_seed_reporting_only"}
        ),
        "population": first["population"],
        "geographic_blocks": first["geographic_blocks"],
        "quantile_rule": first["quantile_rule"],
        "canonical_equivalence": {
            "rule": "for every seed and batch, the six metrics recomputed on its rows equal the scorer's metrics.json",
            "equal": all(row["equal"] is True for row in rows.values()),
            "max_abs_difference": max(row["max_abs_difference"] for row in rows.values()),
            "per_seed": {key: row["equal"] for key, row in rows.items()},
        },
        "per_seed": {key: dict(per_seed[key]) for key in keys},
    }
    for name in SEED_RESULT_FIELDS:
        combined[name] = first[name] if single else None
    return combined


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    values = [float(v) for v in values if v is not None]
    return {
        "n": len(values),
        "median": float(np.median(values)) if values else None,
        "max": float(max(values)) if values else None,
    }


def cold_start_summary(sessions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Cold start of a procedure over its sessions: worker load time per role and first warm-up request."""
    loads: dict[str, list[float]] = {}
    first, inputs = [], set()
    for row in sessions:
        cold = row.get("cold_start") or {}
        for role, value in (cold.get("worker_load_ms") or {}).items():
            loads.setdefault(role, []).append(value)
        request = cold.get("first_warmup_request") or {}
        if request.get("latency_ms") is not None:
            first.append(request["latency_ms"])
            inputs.add(request.get("input_sha256"))
    return {
        "scope": "one row per session: worker load time of each role and the first warm-up request; never part of "
        "the warm latency row",
        "n_sessions": len(sessions),
        "worker_load_ms": {role: _distribution(values) for role, values in sorted(loads.items())},
        "first_warmup_request_ms": _distribution(first),
        "n_distinct_first_warmup_inputs": len(inputs),
        "repetitions_of_the_first_warmup_input": len(first),
    }


def restart_summary(manifests: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Counted restarts, restarts outside the cap, re-warms and failed re-warms of a procedure's sessions."""
    counted = outside = rewarms = failed = 0
    for manifest in manifests:
        supervision = manifest.get("supervision") or {}
        cap = supervision.get("restart_cap") or {}
        counted += int(cap.get("unplanned_restarts_used") or 0)
        outside += int(cap.get("rewarm_restarts_outside_cap") or 0)
        for entry in supervision.get("passes") or ():
            for summary in entry.get("rewarms") or ():
                rewarms += 1
                failed += summary.get("complete") is not True
    return {
        "n_sessions": len(manifests),
        "counted_restarts": counted,
        "restarts_outside_cap": outside,
        "rewarms": rewarms,
        "failed_rewarms": failed,
    }


def aggregate_campaign(
    plan_path: str | Path,
    plan_sha256: str,
    *,
    procedure: str,
    allow_test: bool = False,
) -> dict[str, Any]:
    """Aggregate one procedure of a completed campaign after re-verifying every unit completion.

    Every partition batch is read through its R7 units (the batch, or its recorded halves) and every unit through its
    windows: the measured prefixes of its sessions, in order, and its imputed failures. Archived attempts under
    ``relaunched/`` are never read.
    """
    from passage_eval.campaign.plan import read_plan
    from passage_eval.campaign.runner import _imputed, _verify_completed, relaunch_summary, unit_leaves
    from passage_eval.io import pinned

    campaign = read_plan(plan_path, plan_sha256, allow_test=allow_test)
    plan = campaign.plan
    qualification = {}
    if campaign.admission is not None:
        from passage_eval.protocol.admission import NATIVE_PROFILES

        profile = campaign.admission.report.admitted.get("profile")
        if profile in NATIVE_PROFILES:
            qualification["qualification_profile"] = profile
    if procedure not in plan["procedures"]:
        raise ValueError(f"{procedure!r} is not a procedure of this plan")
    output = Path(plan["output_dir"])
    parts, receipts, sessions, manifests = [], [], [], []
    for item in campaign.partition["batches"]:
        for unit in unit_leaves(campaign, output, item):
            batch_dir = output / unit.directory
            if not (batch_dir / "completion.json").is_file():
                raise ValueError(f"unit {unit.unit_id} of batch {item['index']} is not complete")
            completion = _verify_completed(campaign, output, unit)
            imputed = _imputed(output, unit)
            if procedure in imputed:
                parts.append((imputed[procedure], None, None))
            for window in completion["windows"]:
                for row in window["sessions"]:
                    if row["procedure"] != procedure:
                        continue
                    sessions.append(row)
                    for record in row["scored_receipts"]:
                        directory = batch_dir / f"window-{window['window']:02d}" / procedure / record["path"]
                        run = load_scored(directory.parent)
                        run["metrics"] = json_load_metrics(run["path"])
                        manifests.append(run["manifest"])
                        parts.append(
                            ((run, row["executed_sample_ids"]), record["seed"], record["executed_artifact_sha256"])
                        )
            receipts.append(
                {
                    "batch_index": item["index"],
                    "unit": unit.unit_id,
                    "completion_sha256": pinned.sha256_bytes(pinned.read_bounded(batch_dir / "completion.json")),
                }
            )
    parent_order = [task["sample_id"] for task in campaign.admissible["tasks"]]
    parent_blocks = {
        row["sample_id"]: row["parent_geographic_block"]
        for item in campaign.partition["batches"]
        for row in item["parent_mapping"]
    }
    derivation = campaign.admissible["selection"]["derivation"]
    population = {key: derivation[key] for key in ("n_source", "n_excluded", "n_admissible")}
    per_seed = {}
    for seed in plan["seeds"]:
        # an imputed failure belongs to every seed of the plan (plans run one seed)
        selected = [element for element, run_seed, _ in parts if run_seed in (seed, None)]
        per_seed[str(seed)] = aggregate_scored_batches(
            selected,
            parent_order=parent_order,
            parent_blocks=parent_blocks,
            population=population,
            min_warm_requests=plan["aggregation"]["min_warm_requests"],
        )
        per_seed[str(seed)]["executed_artifact_sha256"] = sorted(
            {digest for _, run_seed, executed in parts if run_seed == seed for digest in executed}
        )
    return json_safe(
        {
            **combine_seeds(per_seed, plan["seeds"]),
            **qualification,
            "procedure": procedure,
            "plan_sha256": plan_sha256,
            "protocol_record_sha256": plan["protocol_record_sha256"],
            "campaign_id": plan["campaign_id"],
            "split": plan["split"],
            "target": plan["target"],
            "harness": {"commit": plan["harness"]["commit"], "source_sha256": plan["harness"]["source_sha256"]},
            "admissible_roster_sha256": campaign.admissible["roster_sha256"],
            "reference_admission_chain_sha256": campaign.chain["chain_sha256"],
            "budget": plan["budget"],
            "timing_profile": plan["timing"]["profile"],
            "receipts": receipts,
            "sessions": {
                "n_sessions": len(sessions),
                "n_closed_by_admission": sum(row["status"] == "closed_by_admission" for row in sessions),
            },
            "cold_start": cold_start_summary(sessions),
            "restarts": restart_summary(manifests),
            "relaunches": relaunch_summary(output, len(campaign.partition["batches"]), plan["r7_policy"]),
            "limitations": list(LIMITATIONS),
        }
    )
