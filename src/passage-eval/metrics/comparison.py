"""Strict matched-task contrasts with paired geographic cluster resampling.

The estimand is T(A)-T(B), recomputed on the paired rows of every resample.
Training seeds and repeat conditions remain descriptive, separate from geographic
sampling uncertainty. Missing block identities never imply independent rows.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from passage_eval.diagnostics.coverage import joint_odd_coverage, stratify_numeric
from passage_eval.execution.timing import REWARM_RECORD_FIELDS, manifest_profile, manifest_warmup
from passage_eval.metrics.latency import (
    censored_latency_quantiles,
    latency_censoring,
    validation_recost_summary,
    validation_recost_values,
)
from passage_eval.protocol.attribution import component_control

SUPPORTED_QUALIFIED_METRIC_CONTRACT = "passage-qualified-complete-v1"
FAILURE_COST_MULTIPLIER = 1_000_000.0
FAILURE_REGRET = FAILURE_COST_MULTIPLIER - 1.0
STATISTICS = {"mean": np.mean, "median": np.median, "p90": lambda values: np.quantile(values, 0.9)}


def paired_cluster_bootstrap(
    a: Sequence[float],
    b: Sequence[float],
    blocks: Sequence[str] | None,
    *,
    statistic: str = "median",
    resamples: int = 2000,
    seed: int = 0,
    confidence: float = 0.95,
) -> dict[str, Any]:
    """Estimate a contrast and percentile interval over paired whole blocks.

    Each draw samples the observed number of blocks with replacement, then
    concatenates every row in each drawn block. Thus varying cluster sizes retain
    the task-weighted statistic. One block cannot estimate geographic uncertainty.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("paired metric arrays must be matching finite one-dimensional arrays")
    if statistic not in STATISTICS:
        raise ValueError(f"unsupported statistic {statistic!r}")
    if isinstance(resamples, bool) or not isinstance(resamples, int) or resamples < 1:
        raise ValueError("resamples must be a positive integer")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    if blocks is not None and len(blocks) != len(a):
        raise ValueError("geographic blocks must match paired rows")
    result = {
        "statistic": statistic,
        "contrast": "T(A)-T(B)",
        "estimate": None,
        "interval": None,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "n_tasks": len(a),
        "n_blocks": None,
        "interval_method": "paired_geographic_cluster_percentile_bootstrap",
    }
    if not len(a):
        return dict(result, status="empty")
    summary = STATISTICS[statistic]
    result["estimate"] = float(summary(a) - summary(b))
    if blocks is None or any(block is None or str(block) == "" for block in blocks):
        return dict(result, status="unknown_geographic_blocks")
    names = list(dict.fromkeys(str(block) for block in blocks))
    groups = [np.flatnonzero(np.asarray(blocks, dtype=str) == name) for name in names]
    result["n_blocks"] = len(groups)
    if len(groups) < 2:
        return dict(result, status="insufficient_blocks")
    rng = np.random.default_rng(seed)
    contrasts = np.empty(resamples)
    for index in range(resamples):
        rows = np.concatenate([groups[group] for group in rng.integers(0, len(groups), size=len(groups))])
        contrasts[index] = summary(a[rows]) - summary(b[rows])
    tail = (1 - confidence) / 2
    result["interval"] = np.quantile(contrasts, [tail, 1 - tail]).tolist()
    return dict(result, status="estimated")


def _summary(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("summary requires finite one-dimensional metric values")
    return {
        "status": "observed" if values.size else "empty",
        "n": int(values.size),
        **{name: float(function(values)) if values.size else None for name, function in STATISTICS.items()},
    }


#: How the legacy latency summaries and paired contrasts treat censored requests.
LATENCY_CENSORING_NOTE = (
    "recorded_values_of_censored_requests_enter_as_finite_lower_bounds; see all_repeat_latency_censored"
)


def _latency_summary(values: np.ndarray, censored: np.ndarray | None) -> dict[str, Any]:
    """The legacy interpolated latency summary, with the number of censored requests it treats as finite.

    Without censoring flags the count is unknown (``None``), never zero.
    """
    if censored is None:
        return dict(_summary(values), n_censored=None, includes_censored=None, censoring="unknown")
    censored = np.asarray(censored, dtype=bool)
    if censored.shape != np.shape(values):
        raise ValueError("latency censoring flags must align with the latencies")
    n_censored = int(censored.sum())
    return dict(
        _summary(values),
        n_censored=n_censored,
        includes_censored=bool(n_censored),
        censoring=LATENCY_CENSORING_NOTE if n_censored else None,
    )


def _component_attribution(runs, primary, labels, roster):
    controls = [component_control(run["manifest"].get("component_control", {})) for run in runs]
    rows = []
    for run, arrays, label, control in zip(runs, primary, labels, controls):
        selected = [row for row in run["artifact"].records if row["repeat_index"] == arrays["repeat_index"][0]]
        counters = sorted({key for row in selected for key in row["work"]})
        work = {}
        for key in counters:
            observed = [row["work"][key] for row in selected if key in row["work"]]
            work[key] = dict(
                _summary(np.asarray(observed)),
                denominator=len(selected),
                completeness="complete" if len(observed) == len(selected) else "partial",
            )
        repairs = arrays.get("repair_used", np.zeros_like(arrays["fallback"], dtype=bool))
        rows.append(
            {
                "label": label,
                "control": control or {"variant": "unspecified"},
                "checkpoint_hashes": run["manifest"].get("checkpoint_hashes", {}),
                "work_evidence": "complete"
                if work.get("expansions", {}).get("completeness") == "complete"
                else "unreported_or_partial",
                "denominator": len(roster),
                "legal_route_rate": float(np.mean(arrays["valid_path"])),
                "all_task_cost_regret": _summary(arrays["regret"]),
                "fallback_rate": float(np.mean(arrays["fallback"])),
                "repair_rate": float(np.mean(repairs)),
                "direct_success_rate": float(np.mean(arrays["valid_path"] & ~repairs & ~arrays["fallback"])),
                "latency_ms": _latency_summary(arrays["latency_ms"], arrays.get("latency_censored")),
                "work": work,
            }
        )
    # Pair a free guide only with exactly the same machinery and per-task
    # budget. Distinct budgets are valid procedures, but not an S2 control.
    pairs = []
    for a, learned in enumerate(controls):
        if learned.get("variant") != "bounded_repair":
            continue
        for b, free in enumerate(controls):
            if free.get("variant") != "prediction_free" or free["machinery_id"] != learned["machinery_id"]:
                continue
            matched = all(
                runs[a]["manifest"]["request_specs"][task]["budget"]
                == runs[b]["manifest"]["request_specs"][task]["budget"]
                for task in roster
            )
            pairs.append(
                {
                    "learned": labels[a],
                    "prediction_free": labels[b],
                    "status": "matched_machinery_and_budget" if matched else "budget_mismatch",
                }
            )
    complete_checkpoints, checkpoint_verification = [], []
    declaration_complete = False
    failure_statuses = []
    for checkpoint in {control.get("checkpoint_id") for control in controls} - {None}:
        learned_indices = [
            index for index, control in enumerate(controls) if control.get("checkpoint_id") == checkpoint
        ]
        variants = {controls[index]["variant"] for index in learned_indices}
        matching_pairs = [
            pair
            for pair in pairs
            if pair["status"] == "matched_machinery_and_budget"
            and controls[labels.index(pair["learned"])].get("checkpoint_id") == checkpoint
        ]
        if not ({"direct", "bounded_repair", "exact_fallback"} <= variants and matching_pairs):
            continue
        declaration_complete = True
        identities = [
            tuple(sorted(runs[index]["manifest"].get("checkpoint_hashes", {}).values())) for index in learned_indices
        ]
        if any(not identity for identity in identities):
            status = "checkpoint_identity_unknown"
        elif len(set(identities)) != 1:
            status = "checkpoint_identity_mismatch"
        else:
            status = "matched_measured_checkpoint_identity"
        checkpoint_verification.append(
            {"checkpoint_id": checkpoint, "status": status, "labels": [labels[index] for index in learned_indices]}
        )
        if status != "matched_measured_checkpoint_identity":
            failure_statuses.append(status)
            continue
        relevant = set(learned_indices) | {labels.index(pair["prediction_free"]) for pair in matching_pairs}
        relevant |= {index for index, control in enumerate(controls) if control.get("variant") == "exact_classical"}
        if any(rows[index]["work_evidence"] != "complete" for index in relevant):
            failure_statuses.append("incomplete_work_evidence")
            continue
        complete_checkpoints.append(checkpoint)
    complete = bool(complete_checkpoints) and any(control.get("variant") == "exact_classical" for control in controls)
    status = (
        "complete_control_matrix"
        if complete
        else next(
            (
                value
                for value in ("checkpoint_identity_mismatch", "checkpoint_identity_unknown", "incomplete_work_evidence")
                if value in failure_statuses
            ),
            "incomplete_control_matrix",
        )
    )
    return {
        "status": status,
        "declaration_matrix_complete": declaration_complete,
        "rows": rows,
        "matched_repair_controls": pairs,
        "complete_checkpoint_controls": sorted(complete_checkpoints),
        "checkpoint_verification": checkpoint_verification,
        "claim_scope": "declared_controls_with_measured_outcomes; declarations_do_not_authenticate_training",
        "required_variants": ["direct", "bounded_repair", "prediction_free", "exact_fallback", "exact_classical"],
    }


def load_scored(path: str | Path) -> dict[str, Any]:
    """Load one qualified scored artifact directory and check it against its authenticated timed records.

    The receipt, ``run_manifest.json``, ``per_sample.npz`` and ``metrics.json`` must match their recorded hashes;
    every score row must bind the timed status, route, latency (the timing profile's field), censoring flag,
    validation and recost time, legality, fallback and physical recost of its record; failures keep the canonical
    million-times-reference penalty; references are qualified exact. Returns a dictionary with ``path``,
    ``artifact``, ``manifest``, ``receipt``, ``arrays`` (``per_sample.npz`` plus the derived flags of older
    scores), ``source``, ``indices`` (``(sample_id, repeat_index, series)`` to row), ``timing_profile``,
    ``warmup_complete`` (re-warm profile only) and ``block_record`` (the stored roster block record, if any).
    Raises ``ValueError`` on any inconsistency. This is the public loader of ``compare`` and of the campaign
    aggregator (``passage_eval.campaign.aggregate``).
    """
    from passage_eval.execution.artifacts import read_run_artifact, record_repair_used, sha256_file

    path = Path(path).resolve()
    directory = path if path.is_dir() else path.parent
    artifact = read_run_artifact(directory)
    manifest = artifact.manifest
    receipt = json.loads((directory / "score_receipt.json").read_text())
    if (
        receipt.get("schema") != "passage-offline-score-v1"
        or receipt.get("qualified_reference") is not True
        or receipt.get("reference_policy") != "qualified_exact"
    ):
        raise ValueError("comparisons require qualified exact reference scores")
    for filename, key in [
        ("run_manifest.json", "run_manifest_sha256"),
        ("per_sample.npz", "per_sample_sha256"),
        ("metrics.json", "metrics_sha256"),
    ]:
        if receipt.get(key) != sha256_file(directory / filename):
            raise ValueError(f"{filename}: score artifact hash mismatch")
    with np.load(directory / "per_sample.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    required = {
        "sample_id",
        "repeat_index",
        "series",
        "status",
        "route_hash",
        "pred_costs",
        "oracle_costs",
        "regret",
        "assd",
        "h95",
        "connected_path",
        "legal_path",
        "valid_path",
        "reference_id",
        "reference_qualified",
        "reference_policy",
        "latency_ms",
        "metric_contract",
        "quality_repeat_index",
        "primary_quality",
        "raw_pred_costs",
        "fallback",
    }
    if not required <= arrays.keys():
        raise ValueError(f"missing score arrays: {sorted(required - arrays.keys())}")
    n = len(artifact.records)
    for name in required - {"metric_contract", "quality_repeat_index"}:
        if arrays[name].shape != (n,) or arrays[name].dtype.hasobject:
            raise ValueError(f"{name}: score arrays must contain one scalar per complete terminal record")
    if arrays["metric_contract"].shape != ():
        raise ValueError("metric contract must be a scalar identity")
    contract = arrays["metric_contract"].item()
    if contract != SUPPORTED_QUALIFIED_METRIC_CONTRACT:
        raise ValueError("comparison requires the supported qualified metric contract passage-qualified-complete-v1")
    if receipt.get("metric_contract") != contract:
        raise ValueError("score receipt metric contract must equal the scored array contract")
    primary_index = arrays["quality_repeat_index"]
    if (
        primary_index.shape != ()
        or primary_index.dtype.kind not in "iu"
        or primary_index.item() != manifest["quality_repeat_index"]
    ):
        raise ValueError("scored primary repeat index must equal the frozen manifest declaration")
    if arrays["primary_quality"].dtype.kind != "b" or not np.array_equal(
        arrays["primary_quality"], arrays["repeat_index"] == primary_index.item()
    ):
        raise ValueError("primary quality flags must match the frozen primary repeat index")
    if (
        receipt.get("scored_records") != n
        or receipt.get("scored_repeats") != manifest["repeats"]
        or receipt.get("quality_repeat_index") != manifest["quality_repeat_index"]
    ):
        raise ValueError("score receipt denominator/repeat declaration mismatch")
    keys = list(zip(arrays["sample_id"].tolist(), arrays["repeat_index"].tolist(), arrays["series"].tolist()))
    expected = {(row["sample_id"], row["repeat_index"], row["series"]) for row in artifact.records}
    if len(set(keys)) != n or set(keys) != expected:
        raise ValueError("score rows must match the complete unique task roster and repeat series")
    indices = {key: index for index, key in enumerate(keys)}
    repairs = np.zeros(n, dtype=bool)
    for row in artifact.records:
        repairs[indices[(row["sample_id"], row["repeat_index"], row["series"])]] = record_repair_used(row)
    if "repair_used" in arrays:
        if (
            arrays["repair_used"].shape != (n,)
            or arrays["repair_used"].dtype.kind != "b"
            or not np.array_equal(arrays["repair_used"], repairs)
        ):
            raise ValueError("scores must retain timed repair attribution")
    else:
        # Legacy v1 scores did not store the derived flag; derive it from the
        # hash-authenticated terminal records rather than assuming no repair.
        arrays["repair_used"] = repairs
    profile = manifest_profile(manifest)
    elapsed_field = profile.latency_field
    ordered = sorted(artifact.records, key=lambda row: indices[(row["sample_id"], row["repeat_index"], row["series"])])
    censored = latency_censoring(
        [row["status"] for row in ordered], [row.get("latency_charged_not_measured", False) is True for row in ordered]
    )
    if "latency_censored" in arrays:
        if (
            arrays["latency_censored"].shape != (n,)
            or arrays["latency_censored"].dtype.kind != "b"
            or not np.array_equal(arrays["latency_censored"], censored)
        ):
            raise ValueError("scores must retain the timed latency censoring flags")
    else:
        # Scores written before the censoring flag existed: derive it from the authenticated records.
        arrays["latency_censored"] = censored
    recost_times = validation_recost_values(ordered)
    if "validation_recost_ms" in arrays:
        stored = arrays["validation_recost_ms"]
        if stored.shape != (n,) or stored.dtype.kind != "f" or not np.array_equal(stored, recost_times, equal_nan=True):
            raise ValueError("scores must retain the timed validation and recost times")
    else:
        # Scores written before the separate report existed: derive it from the authenticated records.
        arrays["validation_recost_ms"] = recost_times
    for row in artifact.records:
        index = indices[(row["sample_id"], row["repeat_index"], row["series"])]
        if arrays["status"][index] != row["status"] or arrays["route_hash"][index] != row["route_hash"]:
            raise ValueError("score rows must bind exactly the timed terminal status and route")
        if not np.isclose(arrays["latency_ms"][index], row[elapsed_field], rtol=0, atol=0):
            raise ValueError("latency must retain complete request time, including terminal failures")
        if (
            bool(arrays["connected_path"][index]) != row["connected_path"]
            or bool(arrays["legal_path"][index]) != row["legal_path"]
            or bool(arrays["fallback"][index]) != row["fallback"]
        ):
            raise ValueError("scores must retain timed route legality and fallback attribution")
        recost = row["recost"]
        raw_cost = arrays["raw_pred_costs"][index]
        if not (
            (recost is None and not np.isfinite(raw_cost))
            or (recost is not None and np.isclose(raw_cost, recost, rtol=1e-12, atol=1e-12))
        ):
            raise ValueError("scores must retain the timed route physical recost")
        if arrays["valid_path"][index] and not np.isclose(
            arrays["pred_costs"][index], raw_cost, rtol=1e-12, atol=1e-12
        ):
            raise ValueError("valid scored cost must equal the timed physical recost")
    for name in ("connected_path", "legal_path", "valid_path", "reference_qualified", "fallback"):
        if arrays[name].dtype.kind != "b":
            raise ValueError(f"{name}: expected explicit boolean flags")
    if (
        not arrays["reference_qualified"].all()
        or not np.all(arrays["reference_policy"] == "qualified_exact")
        or any(not reference for reference in arrays["reference_id"].tolist())
    ):
        raise ValueError("all score rows must bind qualified exact reference identities")
    if (
        not np.isfinite(arrays["oracle_costs"]).all()
        or np.any(arrays["oracle_costs"] <= 0)
        or not np.isfinite(arrays["pred_costs"]).all()
        or not np.isfinite(arrays["regret"]).all()
        or not np.isfinite(arrays["latency_ms"]).all()
        or np.any(arrays["latency_ms"] < 0)
    ):
        raise ValueError("cost, regret and complete-request latency must be finite; reference cost must be positive")
    valid = arrays["legal_path"] & (arrays["status"] == "ok")
    if not np.array_equal(valid, arrays["valid_path"]):
        raise ValueError("valid paths require legal routes and terminal success")
    if not np.allclose(arrays["regret"], arrays["pred_costs"] / arrays["oracle_costs"] - 1, rtol=1e-12, atol=1e-12):
        raise ValueError("regret must match physical/reference cost ratio minus one")
    if not np.allclose(arrays["regret"][~valid], FAILURE_REGRET, rtol=0, atol=1e-9) or not np.allclose(
        arrays["pred_costs"][~valid], FAILURE_COST_MULTIPLIER * arrays["oracle_costs"][~valid]
    ):
        raise ValueError("all task failures must retain the explicit million-times-reference penalty")
    if not np.isfinite(arrays["assd"][valid]).all() or not np.isfinite(arrays["h95"][valid]).all():
        raise ValueError("every valid path must have finite reference geometry")
    if np.any(arrays["assd"][valid] < 0) or np.any(arrays["h95"][valid] < 0):
        raise ValueError("reference geometry distances must be nonnegative")
    from passage_eval.execution.score_integrity import validate_score_references

    validate_score_references(artifact, arrays, receipt, directory, indices)
    source = manifest.get("data_identity", manifest.get("dataset"))
    if not isinstance(source, dict) or not source or not manifest.get("reference_identity"):
        raise ValueError("frozen source identity and qualified reference identity are required")
    for sample_id in manifest["roster"]:
        selected = arrays["sample_id"] == sample_id
        if len(set(arrays["reference_id"][selected].tolist())) != 1:
            raise ValueError("reference identity must remain identical across all repeats of a task")
    block_record = json.loads((directory / "metrics.json").read_text()).get("geographic_blocks")
    return {
        "path": directory,
        "artifact": artifact,
        "manifest": manifest,
        "receipt": receipt,
        "arrays": arrays,
        "source": source,
        "indices": indices,
        "timing_profile": profile,
        "warmup_complete": (
            [row[REWARM_RECORD_FIELDS[1]] for row in ordered] if profile.rewarm_after_restart else None
        ),
        "block_record": block_record,
    }


# Historical private name, kept for external callers written before ``load_scored`` was public.
_load_scored = load_scored

#: Procedure limits that a strict comparison requires to be recorded and identical (besides every task's budget).
LIMIT_FIELDS = (
    "timing_profile",
    "timing_boundary",
    "deadline_grace_ms",
    "hard_timeout_ms",
    "startup_timeout_s",
    "max_restarts",
    "worker_rlimits",
    "ipc_frame_limits",
)
#: Where each limit is recorded: the manifest itself, ``procedure.supervision`` or ``procedure.supervision.limits``.
_LIMIT_SOURCES = {
    "timing_boundary": "manifest",
    "deadline_grace_ms": "supervision",
    "hard_timeout_ms": "supervision",
    "startup_timeout_s": "limits",
    "max_restarts": "limits",
    "worker_rlimits": "limits",
    "ipc_frame_limits": "limits",
}


def _procedure_limits(run: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """What bounds a run's requests, as recorded, and the limits that the run does not record at all.

    A limit whose key is absent is *not recorded* (reported as ``None``); a recorded ``None`` inside a value,
    such as an unset rlimit, means unlimited. Strict comparison refuses unrecorded limits instead of treating two
    absent values as identical.
    """
    manifest = run["manifest"]
    procedure = manifest.get("procedure") if isinstance(manifest.get("procedure"), dict) else {}
    supervision = procedure.get("supervision") if isinstance(procedure.get("supervision"), dict) else {}
    limits = supervision.get("limits") if isinstance(supervision.get("limits"), dict) else {}
    containers = {"manifest": manifest, "supervision": supervision, "limits": limits}
    values: dict[str, Any] = {"timing_profile": run["timing_profile"].name}
    unrecorded = []
    for name, source in _LIMIT_SOURCES.items():
        container = containers[source]
        if name not in container:
            unrecorded.append(name)
        values[name] = container.get(name)
    return values, unrecorded


def _limits_report(runs: list[dict[str, Any]], roster: list[str]) -> dict[str, Any]:
    rows = [_procedure_limits(run) for run in runs]
    recorded = [values for values, _ in rows]
    unrecorded = [missing for _, missing in rows]
    missing_anywhere = sorted({name for missing in unrecorded for name in missing}, key=LIMIT_FIELDS.index)
    differences = sorted(
        name
        for name in LIMIT_FIELDS
        if name not in missing_anywhere and len({json.dumps(row[name], sort_keys=True) for row in recorded}) > 1
    )
    budgets = [
        [json.dumps(run["manifest"]["request_specs"][sample_id].get("budget"), sort_keys=True) for sample_id in roster]
        for run in runs
    ]
    if any(budget != budgets[0] for budget in budgets[1:]):
        differences.insert(0, "budget")
    return {
        "identical": not differences and not missing_anywhere,
        "differences": differences,
        "unrecorded": missing_anywhere,
        "unrecorded_per_run": unrecorded,
        "recorded": recorded,
    }


def _stored_roster_blocks(run: dict[str, Any], roster: list[str]) -> tuple[list[str] | None, dict | None]:
    """Per-task blocks that the roster scorer stored in one score, with their record (``None`` without one).

    The record must name the roster the run records. A record without usable blocks (an unplaced task) gives
    ``None`` blocks with the record.
    """
    from passage_eval.execution.scoring import ROSTER_BLOCK_PROVENANCES
    from passage_eval.io.roster import run_roster_sha256

    record = run["block_record"]
    if record is None:
        return None, None
    if not isinstance(record, dict):
        raise ValueError("the stored geographic block record must be an object")
    if record.get("roster_sha256") != run_roster_sha256(run["manifest"]):
        raise ValueError("stored roster blocks do not belong to the roster the run records")
    provenance = record.get("provenance")
    if provenance not in ROSTER_BLOCK_PROVENANCES:
        return None, record
    arrays = run["arrays"]
    if "cluster_id" not in arrays or "cluster_provenance" not in arrays:
        raise ValueError("scores that declare roster blocks must store cluster_id and cluster_provenance")
    if not np.all(arrays["cluster_provenance"] == provenance):
        raise ValueError("scored cluster provenance disagrees with the declared roster blocks")
    seen: dict[str, set[str]] = {}
    for sample_id, block in zip(arrays["sample_id"].tolist(), arrays["cluster_id"].tolist()):
        seen.setdefault(sample_id, set()).add(block)
    per_task = []
    for sample_id in roster:
        values = seen.get(sample_id, set())
        if len(values) != 1 or not next(iter(values)):
            raise ValueError(f"{sample_id}: scored roster block is missing or varies across repeats")
        per_task.append(next(iter(values)))
    return per_task, record


def _scored_roster_blocks(runs: list[dict[str, Any]], roster: list[str]) -> tuple[list[str] | None, dict | None]:
    """Blocks that the roster scorer stored from one roster block table, identical in every run (or none)."""
    stored = [_stored_roster_blocks(run, roster) for run in runs]
    records = [record for _, record in stored]
    if all(record is None for record in records):
        return None, None
    if any(record is None for record in records):
        raise ValueError(
            "some scores carry roster blocks and others do not; supply the roster the runs were made with (--roster)"
        )
    if len({record.get("block_table_sha256") for record in records}) != 1:
        raise ValueError("scores carry geographic blocks from different roster block tables")
    blocks = stored[0][0]
    if any(other != blocks for other, _ in stored[1:]):
        raise ValueError("matched tasks carry different roster blocks")
    return blocks, records[0]


def _explicit_roster_blocks(
    table: dict[str, Any], runs: list[dict[str, Any]], roster: list[str]
) -> tuple[list[str] | None, dict[str, Any]]:
    """Blocks from the block table of the roster every compared run records; stored blocks must agree with it."""
    from passage_eval.io.roster import block_table_sealed, check_block_table_binds_run, validate_block_table

    table = validate_block_table(table)
    for index, run in enumerate(runs):
        try:
            check_block_table_binds_run(table, run["manifest"])
        except ValueError as exc:
            raise ValueError(f"compared run {index}: {exc}") from None
    sealed = block_table_sealed(table)
    missing = [sample_id for sample_id in roster if sample_id not in table["blocks"]]
    if missing:
        raise ValueError(f"the roster block table does not list {len(missing)} compared task(s)")
    record = {
        "roster_sha256": table["roster_sha256"],
        "roster_schema": table["roster_schema"],
        "sealed": sealed,
        "block_table_sha256": table["block_table_sha256"],
        "assignment": table["assignment"],
        "assignment_roster_sha256": table["assignment_roster_sha256"],
    }
    if any(table["blocks"][sample_id] is None for sample_id in roster):
        return None, dict(record, status="unknown_tasks_without_footprint")
    blocks = [table["blocks"][sample_id] for sample_id in roster]
    for run in runs:  # a score without stored blocks (an earlier score of the same roster) is accepted
        stored, _ = _stored_roster_blocks(run, roster)
        if stored is not None and stored != blocks:
            raise ValueError("scored roster blocks differ from the supplied roster block table")
    return blocks, dict(record, status="sealed_roster_blocks" if sealed else "unsealed_roster_blocks")


def _geographic_blocks(manifest: dict[str, Any], roster: list[str]) -> tuple[list[str] | None, str]:
    """Keep declared support clusters whole before using footprints or tiles."""
    metadata = manifest.get("request_metadata", {})
    # A declared cluster may join multiple source footprints or tiles. Match
    # scoring's top-level/nested cluster precedence, and leave a partially
    # declared broader unit unknown instead of splitting it into finer units.
    # A scorer's sample-ID fallback is never a geographical block.
    choices = [
        (["cluster_id"], "cluster_id"),
        (["geographic", "cluster_id"], "geographic.cluster_id"),
        (["source_footprint_id"], "source_footprint_id"),
        (["source_tile_id"], "source_tile_id"),
        (["tile_id"], "tile_id"),
        (["geographic", "tile_id"], "geographic.tile_id"),
    ]
    for keys, provenance in choices:
        blocks = []
        for sample_id in roster:
            value: Any = metadata.get(sample_id, {})
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            blocks.append(value)
        if any(value is not None and str(value) for value in blocks):
            if all(value is not None and str(value) for value in blocks):
                return [str(value) for value in blocks], provenance
            return None, "unknown"
    return None, "unknown"


def _primary_arrays(run: dict[str, Any], roster: list[str], repeat: int) -> dict[str, np.ndarray]:
    series = run["manifest"].get("series", "warm")
    indices = [run["indices"][(sample_id, repeat, series)] for sample_id in roster]
    return {key: values[indices] for key, values in run["arrays"].items() if values.ndim == 1}


def compare_artifacts(
    paths: Sequence[str | Path],
    *,
    primary_repeat_index: int | None = None,
    bootstrap_resamples: int = 2000,
    bootstrap_seed: int = 0,
    confidence: float = 0.95,
    odd_spec: dict[str, Any] | None = None,
    require_identical_limits: bool = False,
    geographic_blocks: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare qualified scored runs with exactly identical frozen task identities.

    Input paths may be run directories or files inside those directories. Budget,
    checkpoint and procedure identities may differ; dataset, reference, ordinary
    input, endpoints, objective, graph and task variant must match by task ID.
    Quality uses one explicit repeat; latency retains all repeats and failures.
    Runs with different timing profiles (latency fields) are never compared.

    ``require_identical_limits`` (opt-in, off by default) also requires every
    task's budget, the timing boundary and the recorded supervision limits
    (grace, hard timeout, start-up timeout, restart cap) to be identical, as a
    comparison on matched budgets needs. The result always reports which of
    them differ under ``procedure_limits``.

    Blocks come, in this order, from ``geographic_blocks`` (the ``roster_block_table``
    of the roster that every compared run records; a table of another roster, or a
    run without a recorded roster, is refused), from the roster blocks that the
    roster scorer stored in every score, or from the declared request metadata.
    Blocks are labelled sealed only for a ``passage-roster-v2`` table that binds the
    installed block rule.
    """
    if not isinstance(require_identical_limits, bool):
        raise ValueError("require_identical_limits must be a boolean")
    if len(paths) < 2:
        raise ValueError("at least two scored artifacts are required for comparison")
    runs = [load_scored(path) for path in paths]
    if len({run["path"] for run in runs}) != len(runs):
        raise ValueError("duplicate comparison artifact paths")
    base = runs[0]
    roster = sorted(base["manifest"]["roster"])
    for run in runs[1:]:
        if run["timing_profile"] != base["timing_profile"]:
            raise ValueError("matched latency comparison requires identical timing profiles")
        if run["manifest"]["series"] != base["manifest"]["series"]:
            raise ValueError("matched latency comparison requires the same declared request series")
        if set(run["manifest"]["roster"]) != set(roster):
            raise ValueError("matched comparison requires exactly identical frozen task rosters")
        if run["source"] != base["source"]:
            raise ValueError("matched comparison requires the same actual frozen source identity")
        if run["manifest"]["reference_identity"] != base["manifest"]["reference_identity"]:
            raise ValueError("matched comparison requires the same qualified reference identity")
        if run["arrays"]["metric_contract"].item() != base["arrays"]["metric_contract"].item():
            raise ValueError("matched comparison requires identical metric contracts")
        for sample_id in roster:
            left = {k: v for k, v in base["manifest"]["request_specs"][sample_id].items() if k != "budget"}
            right = {k: v for k, v in run["manifest"]["request_specs"][sample_id].items() if k != "budget"}
            if left != right:
                raise ValueError(f"{sample_id}: input/task/objective/graph identity mismatch")
        if _geographic_blocks(run["manifest"], roster) != _geographic_blocks(base["manifest"], roster):
            raise ValueError("matched tasks have inconsistent geographic block metadata")
    limits = _limits_report(runs, roster)
    if require_identical_limits and not limits["identical"]:
        raise ValueError(
            "strict comparison requires recorded, identical budgets, timing profile, timing boundary and "
            f"supervision limits; not recorded: {', '.join(limits['unrecorded']) or 'none'}; "
            f"these differ: {', '.join(limits['differences']) or 'none'}"
        )
    declared = [run["manifest"]["quality_repeat_index"] for run in runs]
    if primary_repeat_index is None:
        if len(set(declared)) != 1:
            raise ValueError("primary repeat declarations differ; explicitly select a common repeat")
        primary_repeat_index = declared[0]
    if (
        isinstance(primary_repeat_index, bool)
        or not isinstance(primary_repeat_index, int)
        or any(not 0 <= primary_repeat_index < run["manifest"]["repeats"] for run in runs)
    ):
        raise ValueError("primary repeat index must select an available repeat in every run")
    primary = [_primary_arrays(run, roster, primary_repeat_index) for run in runs]
    for arrays in primary[1:]:
        if not np.array_equal(arrays["reference_id"], primary[0]["reference_id"]):
            raise ValueError("matched task rows have inconsistent qualified reference identities")
        if not np.array_equal(arrays["oracle_costs"], primary[0]["oracle_costs"]):
            raise ValueError("matched task rows have inconsistent qualified reference costs")
    if geographic_blocks is not None:
        blocks, roster_block_record = _explicit_roster_blocks(geographic_blocks, runs, roster)
        kind = "sealed" if roster_block_record["sealed"] else "unsealed"
        block_provenance = f"{kind}_roster_block_table" if blocks is not None else "unknown"
    else:
        from passage_eval.execution.scoring import ROSTER_BLOCK_PROVENANCE

        blocks, roster_block_record = _scored_roster_blocks(runs, roster)
        kind = "sealed" if (roster_block_record or {}).get("provenance") == ROSTER_BLOCK_PROVENANCE else "unsealed"
        block_provenance = f"scored_{kind}_roster_blocks" if blocks is not None else "unknown"
    if roster_block_record is None:
        blocks, block_provenance = _geographic_blocks(base["manifest"], roster)
    labels = [f"{index}:{run['manifest'].get('method_name', run['path'].name)}" for index, run in enumerate(runs)]
    methods, occupancy = [], {}
    for label, run, arrays in zip(labels, runs, primary):
        valid = arrays["valid_path"]
        all_arrays = run["arrays"]
        divergent = sum(
            len(set(all_arrays["route_hash"][all_arrays["sample_id"] == sample_id].tolist())) > 1
            for sample_id in roster
        )
        method = {
            "label": label,
            "artifact_path": str(run["path"]),
            "procedure_id": run["manifest"].get("procedure_id"),
            "all_task": {
                "denominator": len(roster),
                "cost_regret": _summary(arrays["regret"]),
                "latency_ms": _latency_summary(arrays["latency_ms"], arrays["latency_censored"]),
                "failure_fraction": float(np.mean(~valid)),
                "penalty_determined_quantiles": {
                    name: bool(np.isclose(function(arrays["regret"]), FAILURE_REGRET))
                    for name, function in STATISTICS.items()
                    if name != "mean"
                },
            },
            "own_valid_quality": {"denominator": int(valid.sum()), "cost_regret": _summary(arrays["regret"][valid])},
            "own_valid_geometry": {
                "denominator": int(valid.sum()),
                "status": "observed" if valid.any() else "empty",
                "assd": _summary(arrays["assd"][valid]),
                "h95": _summary(arrays["h95"][valid]),
            },
            "rates": {
                "denominator": len(roster),
                "legal_route_rate": float(np.mean(valid)),
                "fallback_rate": float(np.mean(arrays["fallback"])),
                "repair_rate": float(np.mean(arrays["repair_used"])),
                "direct_success_rate": float(np.mean(valid & ~arrays["repair_used"] & ~arrays["fallback"])),
                "raw_legal_path_rate": float(np.mean(arrays["legal_path"])),
                "historical_connected_path_rate": float(np.mean(arrays["connected_path"])),
            },
            "all_repeat_latency_ms": _latency_summary(all_arrays["latency_ms"], all_arrays["latency_censored"]),
            "all_repeat_latency_censored": dict(
                censored_latency_quantiles(
                    all_arrays["latency_ms"],
                    status=all_arrays["status"],
                    sample_id=all_arrays["sample_id"],
                    censored=all_arrays["latency_censored"],
                    series=run["manifest"].get("series", "warm"),
                    warmup_complete=run["warmup_complete"],
                    warmup=manifest_warmup(run["manifest"]),
                ),
                latency_field=run["timing_profile"].latency_field,
                timing_profile=run["timing_profile"].name,
            ),
            "all_repeat_validation_recost_ms": validation_recost_summary(
                all_arrays["validation_recost_ms"], inside_latency=run["timing_profile"].validation_in_boundary
            ),
            "repeat_series": run["manifest"].get("series", "warm"),
            "declared_primary_repeat_index": run["manifest"]["quality_repeat_index"],
            "counts": {
                "requested": len(run["artifact"].records),
                "attempted": len(run["artifact"].records),
                "safely_completed": len(run["artifact"].records),
                "primary_quality": len(roster),
            },
            "provenance": {
                "candidate_identity": run["manifest"].get("candidate_identity"),
                "checkpoint_hashes": run["manifest"].get("checkpoint_hashes"),
                "runtime_source_hash": run["manifest"].get("runtime_source_hash"),
                "hardware_stack": run["manifest"].get("hardware_stack"),
                "evidence_scope": "persisted_qualified_procedure_outputs",
                "historical_result_authentication": False,
                "exclusive_host_qualification": "not_established_by_this_artifact_schema",
            },
            "terminal_status_counts_all_repeats": {
                status: int(np.sum(all_arrays["status"] == status))
                for status in sorted(set(all_arrays["status"].tolist()))
            },
            "route_divergence": {
                "n_tasks": len(roster),
                "divergent_task_count": divergent,
                "status_divergent_task_count": sum(
                    len(set(all_arrays["status"][all_arrays["sample_id"] == task].tolist())) > 1 for task in roster
                ),
                "all_repeats_retained": True,
            },
            "seed_condition": {
                "seed": run["manifest"].get("seed"),
                "kind": "descriptive_condition",
                "training_seed_uncertainty": "not_estimated",
                "geographic_intervals_condition_on_this_seed": True,
            },
        }
        methods.append(method)
        rows = []
        for index, sample_id in enumerate(roster):
            spec = run["manifest"]["request_specs"][sample_id]
            row = dict(run["manifest"].get("request_metadata", {}).get(sample_id, {}))
            row.update(
                resolution=spec["shape"][0],
                cost_model=spec["cost_model"],
                cost_weight=spec["cost_weight"],
                task_variant=spec["task_variant"],
                legal=bool(valid[index]),
                cost_regret=float(arrays["regret"][index]),
                latency_ms=float(arrays["latency_ms"][index]),
                latency_censored=bool(arrays["latency_censored"][index]),
            )
            if "elev_std" in arrays:
                row["elev_std"] = float(arrays["elev_std"][index])
            if "obs_density" in arrays:
                row["obstacle_fraction"] = float(arrays["obs_density"][index])
            diagonal = float(np.linalg.norm(np.asarray(spec["shape"]) - 1))
            row["endpoint_separation_normalized"] = (
                float(np.linalg.norm(np.asarray(spec["goal"]) - spec["start"])) / diagonal if diagonal else None
            )
            rows.append(row)
        occupancy[label] = rows
    if odd_spec is None:
        coverage = joint_odd_coverage(occupancy)
        coverage.update(
            status="unknown_strata",
            common_support_status="unknown",
            unknown_required_strata=["elev_std", "endpoint_separation_normalized", "obstacle_fraction"],
            spec=None,
            spec_sha256=None,
            limitation="Task categories are descriptive; joint numeric ODD strata require predeclared edges.",
        )
    else:
        if (
            not isinstance(odd_spec, dict)
            or set(odd_spec) != {"axes", "edges"}
            or not isinstance(odd_spec["axes"], list)
            or not odd_spec["axes"]
            or not isinstance(odd_spec["edges"], dict)
            or not set(odd_spec["edges"]) <= set(odd_spec["axes"])
        ):
            raise ValueError("ODD spec must declare nonempty axes and fixed edges for selected numeric axes")
        numeric_axes = {"elev_std", "endpoint_separation_normalized", "obstacle_fraction"}
        if not (numeric_axes & set(odd_spec["axes"])) <= set(odd_spec["edges"]):
            raise ValueError("numeric ODD axes require predeclared bin edges")
        frozen = json.dumps(odd_spec, sort_keys=True, separators=(",", ":"), allow_nan=False)
        bin_unknown = {}
        for label, rows in occupancy.items():
            bin_unknown[label] = {}
            for axis, edges in odd_spec["edges"].items():
                bins = stratify_numeric([row.get(axis) for row in rows], edges)
                bin_unknown[label][axis] = sum(value is None for value in bins)
                for row, value in zip(rows, bins):
                    row[axis] = value
        coverage = joint_odd_coverage(occupancy, axes=odd_spec["axes"])
        coverage.update(
            spec=json.loads(frozen),
            spec_sha256=hashlib.sha256(frozen.encode()).hexdigest(),
            bin_unknown_counts=bin_unknown,
            bins="left_closed_right_open_last_right_closed",
            spec_policy="predeclared_edges_no_outcome_driven_estimation",
        )
    comparisons = []
    options = dict(resamples=bootstrap_resamples, seed=bootstrap_seed, confidence=confidence)

    def contrast(left, right, chosen=None):
        left, right = np.asarray(left), np.asarray(right)
        local_blocks = blocks
        if chosen is not None:
            left, right = left[chosen], right[chosen]
            local_blocks = np.asarray(blocks)[chosen].tolist() if blocks is not None else None
        return {
            name: paired_cluster_bootstrap(left, right, local_blocks, statistic=name, **options) for name in STATISTICS
        }

    def latency_contrast(left, right):
        """Paired latency contrasts, flagged when either side treats a censored request as a finite value."""
        n_left, n_right = int(left["latency_censored"].sum()), int(right["latency_censored"].sum())
        either = int(np.sum(left["latency_censored"] | right["latency_censored"]))
        flags = {
            "n_censored_a": n_left,
            "n_censored_b": n_right,
            "n_tasks_censored_in_either": either,
            "includes_censored": bool(either),
            "censoring": LATENCY_CENSORING_NOTE if either else None,
        }
        return {
            name: dict(estimate, **flags)
            for name, estimate in contrast(left["latency_ms"], right["latency_ms"]).items()
        }

    for a, b in itertools.combinations(range(len(runs)), 2):
        left, right = primary[a], primary[b]
        both = left["valid_path"] & right["valid_path"]
        comparisons.append(
            {
                "a": labels[a],
                "b": labels[b],
                "contrast_direction": "A_minus_B",
                "all_task": {
                    "denominator": len(roster),
                    "cost_regret": contrast(left["regret"], right["regret"]),
                    "latency_ms": latency_contrast(left, right),
                    "legal_route_rate": paired_cluster_bootstrap(
                        left["valid_path"], right["valid_path"], blocks, statistic="mean", **options
                    ),
                },
                "both_valid_geometry": {
                    "denominator": int(both.sum()),
                    "status": "observed" if both.any() else "empty",
                    "assd": contrast(left["assd"], right["assd"], both),
                    "h95": contrast(left["h95"], right["h95"], both),
                },
                "both_valid_quality": {
                    "denominator": int(both.sum()),
                    "status": "observed" if both.any() else "empty",
                    "cost_regret": contrast(left["regret"], right["regret"], both),
                },
            }
        )
    return {
        "schema": "passage-matched-comparison-v1",
        "task_roster": roster,
        "n_tasks": len(roster),
        "source_identity": base["source"],
        "reference_identity": base["manifest"]["reference_identity"],
        "metric_contract": base["arrays"]["metric_contract"].item(),
        "primary_repeat_index": primary_repeat_index,
        "declared_primary_repeat_indices": declared,
        "primary_repeat_override": any(primary_repeat_index != repeat for repeat in declared),
        "timing_profile": base["timing_profile"].name,
        "procedure_limits": dict(limits, strict=require_identical_limits),
        "failure_policy": {
            "cost_multiplier_of_qualified_optimum": FAILURE_COST_MULTIPLIER,
            "failure_regret": FAILURE_REGRET,
            "all_failures_retained": True,
        },
        "uncertainty": {
            "resampling_unit": "geographic_or_source_footprint_block",
            "block_provenance": block_provenance,
            "sealed_roster_blocks": roster_block_record,
            "status": "unknown_geographic_blocks" if blocks is None else "declared_blocks",
            "seed": bootstrap_seed,
            "resamples": bootstrap_resamples,
            "confidence": confidence,
            "seed_variability": "separate_descriptive_conditions_only",
            "limitation": "Intervals condition on declared procedures, seeds and the observed frozen task population.",
        },
        "units": {
            "cost_regret": "dimensionless_ratio_minus_one",
            "latency_ms": "milliseconds",
            "assd": "pixels_per_oracle_waypoint_count",
            "h95": "pixels",
            "legal_route_rate": "fraction",
        },
        "methods": methods,
        "comparisons": comparisons,
        "joint_odd_coverage": coverage,
        "component_attribution": _component_attribution(runs, primary, labels, roster),
    }


def report_artifact(path: str | Path, *, format: str = "markdown") -> str | dict[str, Any]:
    """Read a comparison artifact and render a self-contained human report."""
    result = json.loads(Path(path).read_text())
    if result.get("schema") != "passage-matched-comparison-v1":
        raise ValueError("report requires a matched comparison artifact")
    if format == "json":
        return result
    if format != "markdown":
        raise ValueError("report format must be markdown or json")
    from datetime import datetime
    from zoneinfo import ZoneInfo

    date = datetime.now(ZoneInfo("Europe/Paris")).date().isoformat()

    def number(value):
        return "empty" if value is None else f"{value:.6g}"

    def interval_text(estimate):
        interval = estimate["interval"]
        return "unknown" if interval is None else f"[{interval[0]:.6g}, {interval[1]:.6g}]"

    def cell_text(value):
        return str(value).replace("|", r"\|").replace("\n", " ")

    lines = [
        "# PASSAGE matched comparison",
        "",
        "| Author | Date | Report version |",
        "|---|---|---|",
        f"| Geoffrey Delhomme | {date} | passage-matched-comparison-v1 |",
        "",
        "*Table 1. Report metadata.*",
        "",
        "## 1. Scope and interpretation",
        "",
        f"The comparison covers {result['n_tasks']} identical frozen tasks at primary repeat "
        f"{result['primary_repeat_index']}. Differences use A minus B. Section 2 presents method results; "
        "Section 3 gives paired contrasts and uncertainty; Section 4 describes joint domain coverage; "
        "Section 5 records component attribution.",
        "",
        "Failure costs are 1,000,000 times the qualified optimum, giving regret 999,999. "
        "Every failure remains in the all-task denominator and complete-request latency. "
        "The legal rate requires both a legal route and successful terminal status. The historical "
        "connected rate describes endpoint connectivity and can include an obstacle violation or terminal failure.",
        "",
        f"Source identity: `{json.dumps(result['source_identity'], sort_keys=True)}`. "
        f"Qualified reference identity: `{result['reference_identity']}`. "
        f"Metric contract: `{result['metric_contract']}`.",
        "",
        "## 2. Method results",
        "",
        "| Method | Tasks | Mean regret | Median regret | p90 regret | Legal rate | Historical connected rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in result["methods"]:
        regret, rates = method["all_task"]["cost_regret"], method["rates"]
        lines.append(
            f"| {cell_text(method['label'])} | {method['all_task']['denominator']} | {number(regret['mean'])} | "
            f"{number(regret['median'])} | {number(regret['p90'])} | {number(rates['legal_route_rate'])} | "
            f"{number(rates['historical_connected_path_rate'])} |"
        )
    lines.extend(
        [
            "",
            "*Table 2. All-task cost regret (dimensionless) and reliability rates (fractions).*",
            "",
            "| Method | Own-valid tasks | Own-valid mean regret | Mean ASSD | Mean h95 (pixels) | Primary median latency (ms) | All-repeat latency rows | Censored latency rows | All-repeat p90 latency (ms) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for method in result["methods"]:
        geometry, latency = method["own_valid_geometry"], method["all_repeat_latency_ms"]
        conditional = method.get("own_valid_quality", {}).get("cost_regret", {}).get("mean")
        lines.append(
            f"| {cell_text(method['label'])} | {geometry['denominator']} | {number(conditional)} | {number(geometry['assd']['mean'])} | "
            f"{number(geometry['h95']['mean'])} | {number(method['all_task']['latency_ms']['median'])} | "
            f"{latency['n']} | {latency.get('n_censored', 'unknown')} | {number(latency['p90'])} |"
        )
    lines.extend(
        [
            "",
            "*Table 3. Geometry among each method’s own valid routes and retained complete-request latency. "
            "ASSD is average symmetric surface distance in pixels divided by the qualified oracle waypoint count; "
            "h95 is symmetric 95th-percentile Hausdorff distance. Latency columns interpolate recorded values; "
            "a censored row (a timeout, or a time charged without a measurement) enters as its recorded lower "
            "bound, and the JSON artifact gives the censoring-aware quantiles.*",
            "",
            "| Method | Seed condition | Repeat series | Terminal outcomes across all repeats | Tasks with route divergence |",
            "|---|---|---|---|---:|",
        ]
    )
    for method in result["methods"]:
        lines.append(
            f"| {cell_text(method['label'])} | {cell_text(method['seed_condition']['seed'])} | "
            f"{cell_text(method['repeat_series'])} | {cell_text(json.dumps(method['terminal_status_counts_all_repeats'], sort_keys=True))} | "
            f"{method['route_divergence']['divergent_task_count']} |"
        )
    lines.extend(
        [
            "",
            "*Table 4. Declared seed and repeat conditions, retained failures, and route divergence.*",
            "",
            "## 3. Paired contrasts and geographic uncertainty",
            "",
            "Paired intervals resample whole declared geographic or source-footprint blocks and recompute "
            "each statistic in both methods. Missing geographic metadata leaves uncertainty unknown; "
            "training-seed conditions remain descriptive. One observed block cannot estimate geographic uncertainty. "
            "Intervals condition on the declared procedures, seeds, and observed frozen task population.",
            "",
            f"Block metadata: {result['uncertainty']['block_provenance']}; "
            f"status: {result['uncertainty']['status']}; resamples: {result['uncertainty']['resamples']}; "
            f"bootstrap seed: {result['uncertainty']['seed']}; confidence: {result['uncertainty']['confidence']:.3g}.",
            "",
            "| Contrast | Metric | Statistic | Estimate | Interval | Status | Tasks |",
            "|---|---|---|---:|---|---|---:|",
        ]
    )
    for pair in result["comparisons"]:
        for metric in ("cost_regret", "latency_ms"):
            for statistic, estimate in pair["all_task"][metric].items():
                status = estimate["status"]
                if estimate.get("includes_censored"):
                    status += f"; {estimate['n_tasks_censored_in_either']} censored task(s) as lower bounds"
                lines.append(
                    f"| {cell_text(pair['a'])} − {cell_text(pair['b'])} | {metric} | {statistic} | "
                    f"{number(estimate['estimate'])} | {interval_text(estimate)} | "
                    f"{status} | {pair['all_task']['denominator']} |"
                )
        estimate = pair["all_task"]["legal_route_rate"]
        lines.append(
            f"| {cell_text(pair['a'])} − {cell_text(pair['b'])} | legal_route_rate | mean | "
            f"{number(estimate['estimate'])} | {interval_text(estimate)} | "
            f"{estimate['status']} | {pair['all_task']['denominator']} |"
        )
    lines.extend(
        [
            "",
            "*Table 5. Paired all-task contrasts; regret is dimensionless, latency is in milliseconds, "
            "and legal rate is a fraction.*",
            "",
            "| Contrast | Both-valid geometry tasks | Metric | Mean contrast | Interval | Status |",
            "|---|---:|---|---:|---|---|",
        ]
    )
    for pair in result["comparisons"]:
        geometry = pair["both_valid_geometry"]
        for metric in ("assd", "h95"):
            estimate = geometry[metric]["mean"]
            lines.append(
                f"| {cell_text(pair['a'])} − {cell_text(pair['b'])} | {geometry['denominator']} | {metric} | "
                f"{number(estimate['estimate'])} | {interval_text(estimate)} | {estimate['status']} |"
            )
    lines.extend(
        [
            "",
            "*Table 6. Geometry contrasts restricted to the same tasks valid in both methods. "
            "Zero denominators remain explicitly empty.*",
            "",
            "## 4. Joint domain occupancy and limitations",
            "",
        ]
    )
    coverage = result["joint_odd_coverage"]
    lines.extend(
        [
            f"Joint operational-design-domain (ODD) coverage has {len(coverage['cells'])} observed cells "
            f"and {coverage['n_common_cells']} common cells under the declared axes; common support is "
            f"{coverage['common_support_status']} and coverage status is {coverage['status']}. "
            "These cells are descriptive and establish neither independence nor causal effects. "
            "Shared cells support only the specified axes and cannot prove support on unrecorded dimensions.",
            "",
            f"Axes: `{json.dumps(coverage['axes'])}`. Frozen specification hash: `{coverage.get('spec_sha256')}`.",
        ]
    )
    if coverage.get("spec") is not None:
        lines.extend(
            [
                "",
                "Numeric cells use fixed bin indices with left-closed, right-open boundaries; "
                "the last interval includes its right boundary. Missing and outside-range descriptors remain unknown.",
                "",
                "```json",
                json.dumps(coverage["spec"], sort_keys=True, indent=2),
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "Numeric joint strata are unknown because no predeclared bin specification was supplied. "
                "Task-category occupancy does not establish ODD coverage.",
            ]
        )
    lines.extend(
        [
            "",
            "| Joint cell | Population | Tasks | Legal tasks | Mean regret | Median latency (ms) |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for cell in coverage["cells"]:
        for label, population in cell["populations"].items():
            lines.append(
                f"| {cell_text(json.dumps(cell['coordinates'], sort_keys=True))} | {cell_text(label)} | "
                f"{population['count']} | {population['legal_count']} | "
                f"{number(population['mean_cost_regret'])} | {number(population['median_latency_ms'])} |"
            )
    lines.extend(
        [
            "",
            "*Table 7. Joint descriptive cell occupancy and outcomes. The frozen JSON artifact retains "
            "missing-axis counts, every denominator, all statistics, identities, and complete condition details.*",
            "",
        ]
    )
    attribution = result.get("component_attribution", {"status": "not_declared", "rows": []})
    lines.extend(
        [
            "## 5. Component attribution and evidence scope",
            "",
            f"Control matrix status: {attribution['status']}. A learned-contribution claim requires direct decoding, "
            "bounded repair, a prediction-free guide with the same repair machinery and budgets, an explicitly "
            "named exact-fallback variant, and exact classical search. The learned controls reuse the identified "
            "checkpoint. Missing controls remain visible. Exact fallback is excluded from primary learned rows.",
            "",
            "| Method | Component variant | Tasks | Legal rate | Mean regret | Mean expansions | Work observations | Repair rate | Fallback rate | Median request latency (ms) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in attribution["rows"]:
        expansions = row["work"].get("expansions", {})
        lines.append(
            f"| {cell_text(row['label'])} | {cell_text(row['control']['variant'])} | {row['denominator']} | "
            f"{number(row['legal_route_rate'])} | {number(row['all_task_cost_regret']['mean'])} | "
            f"{number(expansions.get('mean'))} | {expansions.get('n', 0)} | "
            f"{number(row['repair_rate'])} | {number(row['fallback_rate'])} | {number(row['latency_ms']['median'])} |"
        )
    lines.extend(
        [
            "",
            "*Table 8. Component attribution from primary-repeat terminal records, including failures. "
            "Missing work is unobserved rather than zero; the JSON artifact retains all counter summaries "
            "and matched-machinery/budget checks.*",
            "",
            "These artifacts identify current qualified procedure outputs. They do not authenticate historical "
            "numbers or establish an exclusive, qualified measurement host. Sparse CPU fixtures support "
            "interoperability checks; they do not establish precise latency tails or deployment claims.",
            "",
        ]
    )
    return "\n".join(lines)
