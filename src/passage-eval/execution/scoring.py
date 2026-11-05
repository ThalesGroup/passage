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

"""Offline reference/geometry scoring of the exact timed terminal outputs."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterable, Mapping, Any

import numpy as np

from passage_eval.execution.artifacts import RunArtifact, json_safe, record_repair_used, sha256_file
from passage_eval.execution.runner import request_identity
from passage_eval.execution.timing import REWARM_RECORD_FIELDS, manifest_profile, manifest_warmup
from passage_eval.metrics.cost import path_cost
from passage_eval.metrics.latency import (
    censored_latency_quantiles,
    latency_censoring,
    validation_recost_summary,
    validation_recost_values,
)
from passage_eval.metrics.reliability import path_status
from passage_eval.metrics.summary import per_sample_metrics, aggregate_primary_metrics
from passage_eval.protocol.contracts import PlanRequest
from passage_eval.protocol.discipline import apply_censored_penalty
from passage_eval.metrics.regret import cost_regret
from passage_eval.protocol.planner import Scene, validated_raw_path


def _qualified_reference(reference: Mapping[str, Any], scene: Scene, identity: dict, manifest: dict):
    if reference.get("qualified") is not True or reference.get("reference_status", reference.get("status")) != "exact":
        raise ValueError(f"{scene.sample_id}: a qualified exact reference is required")
    if reference.get("sample_id", scene.sample_id) != scene.sample_id:
        raise ValueError("qualified reference sample identity mismatch")
    if reference.get("cost_model") != identity["cost_model"] or reference.get("cost_weight") != identity["cost_weight"]:
        raise ValueError("qualified reference objective mismatch")
    if reference.get("task_variant") != identity["task_variant"]:
        raise ValueError("qualified reference task variant mismatch")
    if reference.get("graph_id") != identity["graph_id"]:
        raise ValueError("qualified reference graph mismatch")
    if not reference.get("reference_id"):
        raise ValueError("qualified reference must carry a reference identity")
    receipt = reference.get("reference")
    if not isinstance(receipt, Mapping) or receipt.get("contract_id") != "passage-analytic-reference-v1":
        raise ValueError("qualified reference requires its canonical exact solver receipt")
    payload = {key: value for key, value in receipt.items() if key not in {"reference_id", "hash"}}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        digest != receipt.get("reference_id")
        or digest != receipt.get("hash")
        or digest != reference["reference_id"]
        or receipt.get("status") != "exact"
        or receipt.get("scene_id") != reference.get("scene_id")
        or receipt.get("graph_id") != identity["graph_id"]
        or receipt.get("graph") != identity["graph"]
        or receipt.get("cost") != reference.get("optimal_cost")
        or not np.array_equal(receipt.get("path"), reference.get("optimal_path"))
    ):
        raise ValueError("qualified reference disagrees with its exact solver receipt")
    qualification_policy = reference.get("qualification_policy", "released_labels")
    if qualification_policy == "independent_analytic_reference_v1":
        cohort = reference.get("cohort_id")
        if (
            not isinstance(cohort, str)
            or not cohort.strip()
            or manifest.get("reference_policy") != "independent_analytic_reference"
            or manifest.get("reference_cohort_id") != cohort
            or "source_label_qualified" not in reference
            or "source_label_qualification_reasons" not in reference
        ):
            raise ValueError("independent reference receipt requires an explicitly matching named cohort")
    elif qualification_policy == "released_labels":
        if manifest.get("reference_policy", "released_labels") != "released_labels":
            raise ValueError("released label receipt admission policy mismatch")
        from passage.audit import stored_label_identity

        current_label_id = stored_label_identity(
            scene.oracle_path, scene.oracle_cost, task_variant=identity["task_variant"]
        )
        recorded_label_id = stored_label_identity(
            reference.get("stored_path"), reference.get("stored_cost"), task_variant=identity["task_variant"]
        )
        if reference.get("stored_label_id") != current_label_id or recorded_label_id != current_label_id:
            raise ValueError("qualified receipt stored label identity mismatch")
        if any(reference.get(key) is not True for key in ("legal", "stored_cost_agreement", "optimality")):
            raise ValueError("qualified reference receipt must establish legality, cost agreement and optimality")
    else:
        raise ValueError("unknown qualified reference admission policy")
    # The audit producer binds reference IDs to the ordinary scene content. Use
    # its same identity function rather than trusting a caller's sample ID alone.
    if reference.get("scene_id") is not None:
        from passage.audit import scene_identity

        if reference["scene_id"] != scene_identity(scene):
            raise ValueError("qualified reference scene content mismatch")
    else:
        for key in ("elevation_hash", "obstacles_hash", "start", "goal"):
            if reference.get(key) != identity[key]:
                raise ValueError("qualified reference must bind terrain, obstacles and endpoints")
    path = validated_raw_path(reference.get("optimal_path", []))
    connected, violated, _ = path_status(path, start=scene.start, goal=scene.goal, obstacles=scene.obstacles)
    cost = path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
    optimum = reference.get("optimal_cost")
    graph = identity["graph"]
    if (
        not connected
        or violated
        or optimum is None
        or not np.isfinite(optimum)
        or optimum <= 0
        or not np.isclose(cost, optimum, rtol=graph["rtol"], atol=graph["atol"])
    ):
        raise ValueError("qualified exact reference route/cost is invalid")
    if qualification_policy == "released_labels":
        stored_path = validated_raw_path(scene.oracle_path)
        stored_connected, stored_violated, _ = path_status(
            stored_path, start=scene.start, goal=scene.goal, obstacles=scene.obstacles
        )
        stored_recost = path_cost(scene.elevation, stored_path, weight=scene.cost_weight, cost_model=scene.cost_model)
        if (
            not stored_connected
            or stored_violated
            or not np.isfinite(scene.oracle_cost)
            or scene.oracle_cost <= 0
            or not np.isclose(stored_recost, scene.oracle_cost, rtol=graph["rtol"], atol=graph["atol"])
            or not np.isclose(stored_recost, optimum, rtol=graph["rtol"], atol=graph["atol"])
        ):
            raise ValueError("current stored label must be legal, agree with its cost and be exactly optimal")
    return path, float(optimum), str(reference["reference_id"])


# Public integration boundary; retain the historical private name for callers.
validate_qualified_reference = _qualified_reference


class NegativeRegretHalt(ValueError):
    """A legal route costs less than the qualified optimum beyond the graph tolerance: the reference is diagnosed.

    The qualified reference ``c*`` is an exact optimum, so no legal route can cost less. Such a route is neither a
    win nor an exclusion: scoring stops for a diagnosis of the reference (judge R6).
    """


ROSTER_BLOCK_PROVENANCE = "sealed_roster_geographic_block"
#: Blocks from a ``passage-roster-v1`` table: the block rule's version is not bound, so they are not sealed.
UNSEALED_ROSTER_BLOCK_PROVENANCE = "unsealed_roster_geographic_block"
ROSTER_BLOCK_PROVENANCES = (ROSTER_BLOCK_PROVENANCE, UNSEALED_ROSTER_BLOCK_PROVENANCE)


def _roster_blocks(
    table: Mapping[str, Any], roster: list[str], manifest: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, Any]]:
    """Block assignments for the run's tasks from its roster's block table, and their provenance record.

    The table must belong to the roster the run records. Only a ``passage-roster-v2`` table that binds the
    installed block rule gives *sealed* blocks; a v1 table gives blocks labelled unsealed.
    """
    from passage_eval.io.roster import block_table_sealed, check_block_table_binds_run, validate_block_table

    table = validate_block_table(table)
    run_roster = check_block_table_binds_run(table, manifest)
    sealed = block_table_sealed(table)
    missing = [sample_id for sample_id in roster if sample_id not in table["blocks"]]
    if missing:
        raise ValueError(f"the roster block table does not list {len(missing)} run task(s), e.g. {missing[:3]}")
    unplaced = [sample_id for sample_id in roster if table["blocks"][sample_id] is None]
    record = {
        "roster_sha256": table["roster_sha256"],
        "run_roster_sha256": run_roster,
        "roster_schema": table["roster_schema"],
        "sealed": sealed,
        "block_table_sha256": table["block_table_sha256"],
        "assignment": table["assignment"],
        "assignment_roster_sha256": table["assignment_roster_sha256"],
        "function": table["function"],
        "n_tasks_without_block": len(unplaced),
    }
    if unplaced:
        # A partial assignment is never completed with invented singleton blocks.
        return {}, dict(record, provenance="unknown", status="unknown_tasks_without_footprint", number_of_blocks=None)
    blocks = {sample_id: table["blocks"][sample_id] for sample_id in roster}
    return blocks, dict(
        record,
        provenance=ROSTER_BLOCK_PROVENANCE if sealed else UNSEALED_ROSTER_BLOCK_PROVENANCE,
        status="sealed_roster_blocks" if sealed else "unsealed_roster_blocks",
        number_of_blocks=len(set(blocks.values())),
    )


def score_run_artifact(
    artifact: RunArtifact,
    scenes: Iterable[Scene],
    *,
    reference_store: Mapping[str, Mapping[str, Any]] | None = None,
    require_qualified_reference: bool = True,
    geographic_blocks: Mapping[str, Any] | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Score every recorded repeat, preserving failures and service deadlines.

    Exact references come from an external qualified store (or evaluator-owned
    ``manifest.reference_qualification``), never from a candidate request. The
    explicit diagnostic opt-out uses stored labels with a visible unqualified
    policy; stored cost alone does not establish optimality.

    With qualified references, a legal recorded route (any status) whose cost is below ``c*`` beyond the graph
    tolerance raises ``NegativeRegretHalt``: scoring stops for a diagnosis of the reference.

    ``geographic_blocks`` is the ``roster_block_table`` of the roster the run records. When given, it is the only
    source of ``cluster_id`` (provenance ``sealed_roster_geographic_block`` for a ``passage-roster-v2`` table
    that binds the installed block rule, ``unsealed_roster_geographic_block`` for a v1 table) and
    ``metrics['geographic_blocks']`` records its digest; a table with an unplaced task leaves every block
    unknown, and a table of another roster, or a run that records no roster, is refused. Without it, blocks come
    from the declared request metadata as before.
    """
    artifact.validate()
    materialized = list(scenes)
    scene_map = {scene.sample_id: scene for scene in materialized}
    if len(scene_map) != len(materialized) or set(scene_map) != set(artifact.manifest["roster"]):
        raise ValueError("scoring scenes must match the complete unique artifact roster")
    block_record = None
    if geographic_blocks is not None:
        roster_blocks, block_record = _roster_blocks(
            geographic_blocks, list(artifact.manifest["roster"]), artifact.manifest
        )
    if reference_store is None:
        reference_store = artifact.manifest.get("reference_qualification")
    if require_qualified_reference and reference_store is None:
        raise ValueError("qualified exact reference store required; stored path costs are unqualified")
    references, identities = {}, {}
    for sample_id in artifact.manifest["roster"]:
        scene = scene_map[sample_id]
        identity = request_identity(PlanRequest.from_scene(scene))
        recorded = artifact.manifest["request_specs"][sample_id]
        # Budget belongs to the procedure, not the task identity for rescoring.
        for key in identity.keys() - {"budget"}:
            if recorded.get(key) != identity[key]:
                raise ValueError(f"{sample_id}: scoring input identity mismatch for {key}")
        identities[sample_id] = identity
        if require_qualified_reference:
            if sample_id not in reference_store:
                raise ValueError(f"{sample_id}: missing qualified reference")
            references[sample_id] = _qualified_reference(reference_store[sample_id], scene, identity, artifact.manifest)
        else:
            cost = path_cost(scene.elevation, scene.oracle_path, weight=scene.cost_weight, cost_model=scene.cost_model)
            connected, violated, _ = path_status(
                scene.oracle_path, start=scene.start, goal=scene.goal, obstacles=scene.obstacles
            )
            if not connected or violated or not np.isfinite(cost) or cost <= 0:
                raise ValueError("stored diagnostic reference route must be legal and have positive physical cost")
            references[sample_id] = (scene.oracle_path, cost, "stored-unqualified")
    artifact.scoring_references = (
        json_safe({sample_id: reference_store[sample_id] for sample_id in artifact.manifest["roster"]})
        if require_qualified_reference
        else None
    )
    row_scenes = [scene_map[record["sample_id"]] for record in artifact.records]
    row_refs = [references[record["sample_id"]] for record in artifact.records]
    paths = [validated_raw_path(path) for path in artifact.paths]
    costs = np.asarray(
        [
            path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
            for scene, path in zip(row_scenes, paths)
        ]
    )
    below_optimum = []
    for row, scene, path, cost, reference in zip(artifact.records, row_scenes, paths, costs, row_refs):
        connected, violated, obstacle_cells = path_status(
            path, start=scene.start, goal=scene.goal, obstacles=scene.obstacles
        )
        graph = identities[scene.sample_id]["graph"]
        if (
            require_qualified_reference
            and connected
            and not violated
            and np.isfinite(cost)
            and cost < reference[1]
            and not np.isclose(cost, reference[1], rtol=graph["rtol"], atol=graph["atol"])
        ):
            below_optimum.append(
                f"{scene.sample_id} (repeat {row.get('repeat_index')}): {cost!r} < c* {reference[1]!r}"
            )
        recorded_cost = row.get("recost")
        agrees = (recorded_cost is None and not np.isfinite(cost)) or (
            recorded_cost is not None
            and np.isclose(
                recorded_cost,
                cost,
                rtol=identities[scene.sample_id]["graph"]["rtol"],
                atol=identities[scene.sample_id]["graph"]["atol"],
            )
        )
        if not agrees:
            raise ValueError("offline physical cost disagrees with timed route recost")
        if (
            row.get("connected_path") != connected
            or row.get("obstacle_violation") != violated
            or row.get("obstacle_cells") != obstacle_cells
            or row.get("legal_path") != (connected and not violated and np.isfinite(cost))
        ):
            raise ValueError("offline legality disagrees with timed route validation")
    if below_optimum:
        raise NegativeRegretHalt(
            f"{len(below_optimum)} legal route(s) cost less than the qualified optimum beyond the graph tolerance, "
            f"e.g. {below_optimum[:3]}; scoring stops for a diagnosis of the reference (no win, no exclusion)"
        )
    status_ok = np.asarray([row["status"] == "ok" for row in artifact.records], dtype=bool)
    arrays = per_sample_metrics(
        predicted_paths=paths,
        oracle_paths=[reference[0] for reference in row_refs],
        predicted_costs=costs,
        oracle_costs=[reference[1] for reference in row_refs],
        obstacles=[scene.obstacles for scene in row_scenes],
        starts=[scene.start for scene in row_scenes],
        goals=[scene.goal for scene in row_scenes],
    )
    arrays["connected_path"] = arrays["success"].copy()
    arrays["legal_path"] = arrays["valid_path"].copy()
    # A late, cancelled, OOM or failed route cannot be rescued by offline scoring.
    arrays["success"] &= status_ok
    arrays["valid_path"] &= status_ok
    arrays["pred_costs"] = apply_censored_penalty(
        costs,
        arrays["oracle_costs"],
        decoder_succeeded=arrays["success"],
        obstacle_violated=arrays["obstacle_violation"],
    )
    arrays["regret"] = cost_regret(arrays["pred_costs"], arrays["oracle_costs"])
    arrays["assd"][~arrays["valid_path"]] = np.nan
    arrays["h95"][~arrays["valid_path"]] = np.nan
    policy = "qualified_exact" if require_qualified_reference else "stored_unqualified"
    n = len(artifact.records)
    # Choose one declared support family over the whole roster. A partial
    # broader annotation must not silently split its tasks into finer tiles.
    metadata = artifact.manifest["request_metadata"]
    choices = [
        (("cluster_id",), "declared_cluster_id"),
        (("geographic", "cluster_id"), "declared_geographic_cluster_id"),
        (("source_footprint_id",), "declared_source_footprint_id"),
        (("source_tile_id",), "declared_source_tile_id"),
        (("tile_id",), "declared_tile_id"),
        (("geographic", "tile_id"), "declared_geographic_tile_id"),
    ]
    if block_record is not None:
        choices = []  # a sealed roster block table replaces every declaration
    block_map, block_source = {}, "unknown"
    for keys, provenance in choices:
        declared = {}
        for sample_id in artifact.manifest["roster"]:
            value = metadata[sample_id]
            for key in keys:
                value = value.get(key) if isinstance(value, dict) else None
            declared[sample_id] = value
        present = [value is not None and bool(str(value)) for value in declared.values()]
        if any(present):
            if all(present):
                block_map = {key: str(value) for key, value in declared.items()}
                block_source = provenance
            break
    if block_record is not None and roster_blocks:
        block_map, block_source = roster_blocks, block_record["provenance"]
    cluster_ids = [block_map.get(scene.sample_id, "") for scene in row_scenes]
    cluster_sources = [block_source] * len(row_scenes)
    profile = manifest_profile(artifact.manifest)
    statuses = [row["status"] for row in artifact.records]
    censored = latency_censoring(
        statuses, [row.get("latency_charged_not_measured", False) is True for row in artifact.records]
    )
    arrays.update(
        sample_id=np.asarray([row["sample_id"] for row in artifact.records]),
        repeat_index=np.asarray([row["repeat_index"] for row in artifact.records], dtype=np.int64),
        quality_repeat_index=np.array(artifact.manifest["quality_repeat_index"], dtype=np.int64),
        primary_quality=np.asarray(
            [row["repeat_index"] == artifact.manifest["quality_repeat_index"] for row in artifact.records], dtype=bool
        ),
        series=np.asarray([row["series"] for row in artifact.records]),
        status=np.asarray([row["status"] for row in artifact.records]),
        route_hash=np.asarray([row["route_hash"] for row in artifact.records]),
        raw_pred_costs=costs,
        reference_id=np.asarray([reference[2] for reference in row_refs]),
        reference_qualified=np.full(n, require_qualified_reference, dtype=bool),
        reference_policy=np.full(n, policy),
        fallback=np.asarray([row["fallback"] for row in artifact.records], dtype=bool),
        repair_used=np.asarray([record_repair_used(row) for row in artifact.records], dtype=bool),
        latency_ms=np.asarray([row[profile.latency_field] for row in artifact.records]),
        latency_censored=censored,
        # Supervisor validation and recost per record (nan where not recorded), reported apart from the latency.
        validation_recost_ms=validation_recost_values(artifact.records),
        cluster_id=np.asarray(cluster_ids),
        cluster_provenance=np.asarray(cluster_sources),
        obs_density=np.asarray([float(np.mean(scene.obstacles)) for scene in row_scenes]),
        elev_std=np.asarray([float(np.std(scene.elevation)) for scene in row_scenes]),
        oracle_len=np.asarray([len(reference[0]) for reference in row_refs], dtype=np.int64),
        metric_contract=np.array(
            "passage-qualified-complete-v1" if require_qualified_reference else "passage-stored-diagnostic-v1"
        ),
    )
    primary_mask = arrays["repeat_index"] == artifact.manifest["quality_repeat_index"]
    primary = {key: values[primary_mask] for key, values in arrays.items() if values.ndim == 1}
    metrics = aggregate_primary_metrics(primary)
    all_repeats = aggregate_primary_metrics(arrays)
    all_repeats.update(
        connected_path_rate=float(np.mean(arrays["connected_path"])),
        legal_route_rate=float(np.mean(arrays["valid_path"])),
        fallback_rate=float(np.mean(arrays["fallback"])),
        repair_rate=float(np.mean(arrays["repair_used"])),
        direct_success_rate=float(np.mean(arrays["valid_path"] & ~arrays["repair_used"] & ~arrays["fallback"])),
    )
    metrics["all_repeats"] = all_repeats
    # Legacy summary: linear interpolation over every recorded value, censored or not.
    metrics["latency_all_repeats_ms"] = {
        name: float(np.quantile(arrays["latency_ms"], quantile))
        for name, quantile in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
    }
    # Proposed (not sealed) censoring-aware order statistics over the same requests.
    metrics["latency_all_repeats_censored"] = censored_latency_quantiles(
        arrays["latency_ms"],
        status=arrays["status"],
        sample_id=arrays["sample_id"],
        censored=censored,
        series=artifact.manifest["series"],
        warmup_complete=(
            [row[REWARM_RECORD_FIELDS[1]] for row in artifact.records] if profile.rewarm_after_restart else None
        ),
        warmup=manifest_warmup(artifact.manifest),  # the declared and performed warm-up, not the series name
    )
    metrics["latency_all_repeats_censored"].update(latency_field=profile.latency_field, timing_profile=profile.name)
    metrics["validation_recost_all_repeats_ms"] = validation_recost_summary(
        arrays["validation_recost_ms"], inside_latency=profile.validation_in_boundary
    )
    metrics.update(
        requested_count=n,
        attempted_count=n,
        completed_count=n,
        connected_path_rate=float(np.mean(primary["connected_path"])),
        legal_route_rate=float(np.mean(primary["valid_path"])),
        fallback_rate=float(np.mean(primary["fallback"])),
        repair_rate=float(np.mean(primary["repair_used"])),
        direct_success_rate=float(np.mean(primary["valid_path"] & ~primary["repair_used"] & ~primary["fallback"])),
        terminal_ok_rate=float(np.mean(status_ok)),
        reference_qualified=bool(require_qualified_reference),
        quality_repeat_index=artifact.manifest["quality_repeat_index"],
        scored_repeat_count=artifact.manifest["repeats"],
    )
    if block_record is not None:
        metrics["geographic_blocks"] = block_record
    return metrics, arrays


def write_scored_artifact(artifact: RunArtifact, metrics: dict, arrays: dict[str, np.ndarray]) -> None:
    if artifact.output_dir is None:
        raise ValueError("scored artifact requires a persisted output directory")
    artifact.validate()
    output_dir = Path(artifact.output_dir)
    reference_flags = np.asarray(arrays.get("reference_qualified"))
    if (
        reference_flags.dtype.kind != "b"
        or reference_flags.shape != (len(artifact.records),)
        or not (reference_flags.all() or (~reference_flags).all())
    ):
        raise ValueError("reference qualification flags must be homogeneous and align with every record")
    qualified = bool(reference_flags.all())
    contract_array = np.asarray(arrays.get("metric_contract"))
    expected_contract = "passage-qualified-complete-v1" if qualified else "passage-stored-diagnostic-v1"
    if contract_array.shape != () or contract_array.item() != expected_contract:
        raise ValueError("metric contract must match the scored reference qualification policy")
    contract = str(contract_array.item())
    for key, values in arrays.items():
        if np.asarray(values).dtype.hasobject:
            raise ValueError(f"object/pickle arrays forbidden: {key}")
    reference_inputs_hash = None
    if qualified:
        if artifact.scoring_references is None:
            raise ValueError("qualified scoring requires its validated reference payload; run the offline scorer first")
        reference_payload = {
            "schema": "passage-qualified-score-inputs-v1",
            "request_specs": artifact.manifest["request_specs"],
            "reference_qualification": artifact.scoring_references,
        }
        reference_file = output_dir / "score_reference_inputs.json"
        reference_file.write_text(json.dumps(reference_payload, sort_keys=True, indent=2, allow_nan=False) + "\n")
        reference_inputs_hash = sha256_file(reference_file)
    np.savez(output_dir / "per_sample.npz", **arrays)
    (output_dir / "metrics.json").write_text(
        json.dumps(json_safe(metrics), sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    receipt = dict(
        schema="passage-offline-score-v1",
        qualified_reference=qualified,
        metric_contract=contract,
        reference_policy="qualified_exact" if qualified else "stored_unqualified",
        run_manifest_sha256=sha256_file(output_dir / "run_manifest.json"),
        per_sample_sha256=sha256_file(output_dir / "per_sample.npz"),
        metrics_sha256=sha256_file(output_dir / "metrics.json"),
        scored_records=len(artifact.records),
        scored_repeats=artifact.manifest["repeats"],
        quality_repeat_index=artifact.manifest["quality_repeat_index"],
        reference_admission_policy=artifact.manifest.get("reference_policy", "released_labels"),
        reference_cohort_id=artifact.manifest.get("reference_cohort_id"),
    )
    if qualified:
        receipt["reference_inputs_sha256"] = reference_inputs_hash
        receipt["reference_payload_identity"] = hashlib.sha256(
            json.dumps(artifact.scoring_references, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
    (output_dir / "score_receipt.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
