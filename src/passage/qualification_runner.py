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

"""Bounded, resumable route qualification of a previously frozen metadata roster.

Only released-label qualification is emitted. Independent exact optima remain
audit evidence; this runner never admits a fresh-reference benchmark cohort.
Stat fingerprints bind reads and resume, while complete content authentication
is a separate operation against the expected object manifest.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import multiprocessing
from pathlib import Path
from typing import Any

from passage.audit import AUDIT_CONTRACT, audit_row, geographic_footprint
from passage.dataset import DEFAULT_COLUMNS, DatasetSource
from passage.qualification import (
    _assert_unchanged,
    _hash,
    _revalidate_source,
    _safe,
    _tile_name,
    _write_json,
    read_roster_row,
)
from passage_eval.metrics.cost import canonical_cost_model
from passage_eval.protocol.contracts import GraphSpec

PROTOCOL_VERSION = "passage-route-qualification-v1"
MAX_EXPANSIONS = 100000
COST_MODELS = ("elevation", "slope_uphill", "energy")
TASK_VARIANTS = ("free", "obstacles")
# Project modules on the route-audit call graph: decode, path normalization,
# directed costs, legality, graph identity, source checks and receipt production.
# Resolve their installed files rather than assuming the passage-eval source
# directory spelling; setuptools installs it as the passage_eval package.
IMPLEMENTATION_MODULES = (
    "passage.audit",
    "passage.dataset",
    "passage.pathfinding_utils",
    "passage.utils",
    "passage.qualification",
    "passage.qualification_runner",
    "passage_eval._typing",
    "passage_eval.metrics.cost",
    "passage_eval.metrics.reliability",
    "passage_eval.protocol.contracts",
)


def _implementation_hashes() -> dict[str, str]:
    hashes = {}
    for name in IMPLEMENTATION_MODULES:
        module = importlib.import_module(name)
        if module.__file__ is None:
            raise ValueError(f"Scientific implementation module has no source file: {name}")
        hashes[name] = hashlib.sha256(Path(module.__file__).resolve().read_bytes()).hexdigest()
    return hashes


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _row_id(entry: dict[str, Any]) -> str:
    return _hash({key: entry[key] for key in ("resolution", "split", "sample_id")})


def _no_symlink(path: Path) -> None:
    if path.is_symlink():
        raise ValueError(f"Audit output must not be a symlink: {path}")


def _load_inputs(source: DatasetSource, metadata: Path) -> tuple[dict, dict, dict]:
    protocol = _read_json(metadata / "protocol.json")
    summary = _read_json(metadata / "summary.json")
    roster = _read_json(metadata / "audit_roster.json")
    protocol_id = _hash(protocol)
    if summary.get("status") != "complete":
        raise ValueError("Route audit requires a complete metadata census")
    if protocol.get("source") != source.identity() or summary.get("source") != source.identity():
        raise ValueError("Metadata source differs from route audit source")
    if (
        summary.get("protocol_sha256") != protocol_id
        or roster.get("protocol_sha256") != protocol_id
        or roster.get("source") != source.identity()
    ):
        raise ValueError("Metadata protocol, summary and roster identity differ")
    roster_id = _hash({key: value for key, value in roster.items() if key != "roster_sha256"})
    if (
        roster.get("roster_sha256") != roster_id
        or summary.get("roster", {}).get("roster_sha256") != roster_id
        or not roster.get("frozen_before_tensor_or_reference_audit")
        or roster.get("population_estimation") is not False
    ):
        raise ValueError("Frozen audit roster hash or selection contract differs")
    reference = protocol.get("reference_audit", {})
    declared = reference.get("variants", [])
    expected = {(model, variant) for model in COST_MODELS for variant in TASK_VARIANTS}
    try:
        actual = {(canonical_cost_model(model), variant) for model, variant in (entry.split("/") for entry in declared)}
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid frozen reference objective declaration") from exc
    if (
        reference.get("max_expansions") != MAX_EXPANSIONS
        or isinstance(reference.get("max_expansions"), bool)
        or actual != expected
        or len(declared) != 6
        or not reference.get("retain_failures_and_censoring")
    ):
        raise ValueError("Frozen reference audit must declare all six tasks and 100000 expansions")
    scenes = roster.get("scenes")
    if not isinstance(scenes, list):
        raise ValueError("Frozen roster scenes must be a list")
    if summary.get("roster", {}).get("selected") != len(scenes):
        raise ValueError("Frozen summary selected-row denominator differs from roster")
    ids = set()
    for entry in scenes:
        if not isinstance(entry, dict) or not isinstance(entry.get("sample_id"), str) or not entry["sample_id"].strip():
            raise ValueError("Frozen roster contains an invalid sample ID")
        identity = _row_id(entry)
        if identity in ids:
            raise ValueError("Frozen roster contains a duplicate compound row ID")
        ids.add(identity)
        _shard_record(protocol, entry)
    return protocol, summary, roster


def _shard_record(protocol: dict, entry: dict) -> dict:
    cell = protocol["manifest"].get(f"{entry['resolution']}/{entry['split']}")
    records = (
        [] if cell is None else [record for record in cell["shards"] if record["path"] == entry["locator"]["shard"]]
    )
    if len(records) != 1:
        raise ValueError("Frozen roster locator is absent from the source manifest")
    record = records[0]
    locator = entry["locator"]
    for key in ("row_index", "row_group", "row_in_group"):
        if isinstance(locator[key], bool) or not isinstance(locator[key], int) or locator[key] < 0:
            raise ValueError("Frozen roster locator has an invalid row index")
    groups = record.get("row_groups", [])
    group = locator["row_group"]
    if (
        group >= len(groups)
        or locator["row_in_group"] >= groups[group]
        or locator["row_index"] != sum(groups[:group]) + locator["row_in_group"]
    ):
        raise ValueError("Frozen roster locator differs from source row-group bounds")
    return record


def _support_clusters(scenes: list[dict]) -> tuple[dict[str, str | None], dict]:
    """Connect recorded source/used/overlapping tiles, without claiming cell overlap."""
    parent: dict[str, str] = {}

    def find(tile: str) -> str:
        parent.setdefault(tile, tile)
        while parent[tile] != tile:
            parent[tile] = parent[parent[tile]]
            tile = parent[tile]
        return tile

    supports = {}
    for entry in scenes:
        metadata = entry.get("metadata")
        tile = metadata.get("tile", {}) if isinstance(metadata, dict) else {}
        tile = tile if isinstance(tile, dict) else {}
        tiles = {name for name in (_tile_name(tile.get("source")), entry.get("source_tile")) if name}
        for kind in ("used", "overlapping"):
            for item in tile.get(kind, []) if isinstance(tile.get(kind), list) else []:
                name = _tile_name(item)
                if name:
                    tiles.add(name)
        supports[_row_id(entry)] = sorted(tiles)
        for name in sorted(tiles):
            parent[find(name)] = find(min(tiles))
    groups = defaultdict(set)
    for name in parent:
        groups[find(name)].add(name)
    clusters = {
        _hash({"support_tiles": sorted(names)}): {"support_tiles": sorted(names), "row_ids": []}
        for names in groups.values()
    }
    by_tile = {name: identity for identity, cluster in clusters.items() for name in cluster["support_tiles"]}
    assignments = {}
    for identity, names in supports.items():
        cluster_id = by_tile[names[0]] if names else None
        assignments[identity] = cluster_id
        if cluster_id:
            clusters[cluster_id]["row_ids"].append(identity)
    return assignments, {
        "method": "connected components of recorded source/used/overlapping tile support",
        "clusters": clusters,
        "unknown_rows": sum(not value for value in assignments.values()),
        "source_cell_windows": "unknown unless explicitly recorded per finding",
        "independence_certified": False,
    }


def _terminal_record(entry: dict, model: str, variant: str, reason: str) -> dict:
    return {
        "sample_id": entry["sample_id"],
        "resolution": entry["resolution"],
        "split": entry["split"],
        "cost_model": model,
        "task_variant": variant,
        "qualified": False,
        "qualification_reasons": [reason],
        "reference_status": "not_run",
        "legal": None,
        "stored_cost_agreement": None,
    }


def _audit_entry(source: DatasetSource, entry: dict, source_stat: dict) -> dict:
    """Worker loads one tensor, audits six tasks sequentially, releases tensor state."""
    _assert_unchanged(source, source_stat)
    try:
        row = read_roster_row(source, entry, columns=list(DEFAULT_COLUMNS))
        finding = audit_row(
            row,
            resolution=entry["resolution"],
            split=entry["split"],
            max_expansions=MAX_EXPANSIONS,
            cost_models=COST_MODELS,
            task_variants=TASK_VARIANTS,
        )
    except Exception as exc:
        # Ordinary malformed/unreadable rows remain denominator-bearing results.
        # Source mutations are checked outside this boundary and fail the run.
        finding = {
            "sample_id": entry["sample_id"],
            "resolution": entry["resolution"],
            "split": entry["split"],
            "status": "audit_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "records": [],
            "footprint": geographic_footprint({"id": entry["sample_id"], "metadata": entry.get("metadata")}),
        }
    finally:
        _assert_unchanged(source, source_stat)
    original = finding.get("records", [])
    records = []
    for model in COST_MODELS:
        for variant in TASK_VARIANTS:
            matches = [
                record
                for record in original
                if record.get("cost_model") == model and record.get("task_variant") == variant
            ]
            if len(matches) == 1:
                records.append(matches[0])
            else:
                reason = "ambiguous_objective_records" if matches else finding.get("status", "missing_task_record")
                records.append(_terminal_record(entry, model, variant, reason))
    if original != records:
        finding["raw_records"] = original
    finding["records"] = records
    return _safe(finding)


def _counts(findings: list[dict]) -> dict:
    records = [record for finding in findings for record in finding["records"]]
    statuses = Counter(record["reference_status"] for record in records)
    return {
        "rows": len(findings),
        "task_records": len(records),
        "legal": sum(record.get("legal") is True for record in records),
        "stored_cost_agreement": sum(record.get("stored_cost_agreement") is True for record in records),
        "reference_exact": statuses["exact"],
        "reference_censored": statuses["budget_exhausted"],
        "reference_no_path": statuses["no_path"],
        "reference_not_run": statuses["not_run"],
        "qualified": sum(record.get("qualified") is True for record in records),
        "qualification_reasons": dict(
            Counter(reason for record in records for reason in record["qualification_reasons"])
        ),
        "row_statuses": dict(Counter(finding["status"] for finding in findings)),
    }


def run_route_audit(
    source: DatasetSource, metadata_root: str | Path, output: str | Path, *, workers: int = 4
) -> dict[str, Any]:
    """Audit a completed, frozen local census roster without source modifications.

    Workers are bounded to 1–8; spawned processes load one projected row each,
    with at most ``workers`` pending entries. Each completed row is atomic and
    resumable only under matching protocol, content hash, locator and source
    stat fingerprints. Failed/censored tasks remain in all denominators. Source
    changes, corrupt caches and interruption publish a failed summary and raise.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("workers must be an integer between 1 and 8")
    if source.root is None:
        raise ValueError("Route qualification requires a local frozen snapshot")
    metadata, output = Path(metadata_root).resolve(), Path(output).resolve()
    root = source.root.resolve()
    for protected in (root, metadata):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("Route audit output must be outside and not overlap source or metadata roots")
    output.mkdir(parents=True, exist_ok=True)
    for name in (".route-audit.lock", "protocol.json", "summary.json", "findings"):
        _no_symlink(output / name)
    with (output / ".route-audit.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Route audit output already has an active writer") from exc
        try:
            return _run(source, metadata, output, workers)
        except BaseException as exc:
            # Validation can fail before _run constructs its reporting closure.
            # Preserve existing identities/counts but revoke a stale complete claim.
            summary_path = output / "summary.json"
            if summary_path.exists():
                previous = _read_json(summary_path)
                if previous.get("status") != "failed":
                    previous.update(
                        status="failed",
                        failure={"error_type": type(exc).__name__, "error": str(exc)},
                        updated_at=datetime.now(timezone.utc).isoformat(),
                    )
                    _write_json(summary_path, previous)
            raise


def _run(source: DatasetSource, metadata: Path, output: Path, workers: int) -> dict:
    metadata_protocol, metadata_summary, roster = _load_inputs(source, metadata)
    assignments, support = _support_clusters(roster["scenes"])
    graph = GraphSpec()
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "audit_contract": AUDIT_CONTRACT,
        "source": source.identity(),
        "metadata_root": str(metadata),
        "metadata_protocol_sha256": _hash(metadata_protocol),
        "metadata_summary_sha256": _hash(metadata_summary),
        "roster_sha256": roster["roster_sha256"],
        "source_manifest_sha256": _hash(metadata_protocol["manifest"]),
        "authentication_manifest": metadata_protocol.get("authentication_manifest"),
        "max_expansions": MAX_EXPANSIONS,
        "cost_models": list(COST_MODELS),
        "task_variants": list(TASK_VARIANTS),
        "rtol": graph.rtol,
        "atol": graph.atol,
        "projected_columns": list(DEFAULT_COLUMNS),
        "retain_failures_and_censoring": True,
        "population_estimation": False,
        "new_reference_cohort_admitted": False,
        "qualification_rule": "legal AND stored_cost_agreement AND independently_exact_optimal",
        "implementation_sha256": _implementation_hashes(),
    }
    protocol_id = _hash(protocol)
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if _read_json(protocol_path) != protocol:
            raise ValueError("Frozen route protocol or metadata identity differs; choose a new output directory")
    else:
        _write_json(protocol_path, protocol)
    findings_dir = output / "findings"
    findings_dir.mkdir(exist_ok=True)
    entries = {_row_id(entry): entry for entry in roster["scenes"]}
    findings: dict[str, dict] = {}
    resumed = 0

    def report(status: str, failure: BaseException | None = None) -> dict:
        ordered = [findings[identity] for identity in sorted(findings)]
        by_resolution, by_cell = defaultdict(list), defaultdict(list)
        qualified = []
        for finding in ordered:
            by_resolution[str(finding["resolution"])].append(finding)
            by_cell[f"{finding['resolution']}/{finding['split']}"].append(finding)
            for record in finding["records"]:
                if record["qualified"]:
                    qualified.append(
                        {
                            key: record[key]
                            for key in (
                                "task_id",
                                "sample_id",
                                "resolution",
                                "split",
                                "cost_model",
                                "task_variant",
                                "reference_id",
                                "scene_id",
                                "graph_id",
                                "support_cluster_id",
                                "source_cell_status",
                            )
                        }
                        | {"row_id": finding["row_id"]}
                    )
        value = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_sha256": protocol_id,
            "metadata_protocol_sha256": protocol["metadata_protocol_sha256"],
            "roster_sha256": roster["roster_sha256"],
            "source": source.identity(),
            "authentication_manifest": protocol["authentication_manifest"],
            "status": status,
            "population_estimation": False,
            "content_verified": False,
            "new_reference_cohort_admitted": False,
            "max_expansions": MAX_EXPANSIONS,
            "cost_models": list(COST_MODELS),
            "task_variants": list(TASK_VARIANTS),
            "coverage": {
                "selected_rows": len(entries),
                "completed_rows": len(ordered),
                "expected_task_records": 6 * len(entries),
                "completed_task_records": sum(len(f["records"]) for f in ordered),
            },
            "counts": _counts(ordered),
            "by_resolution": {key: _counts(value) for key, value in by_resolution.items()},
            "by_cell": {key: _counts(value) for key, value in by_cell.items()},
            "resumed_rows": resumed,
            "workers": workers,
            "support_clusters": support,
            "qualified_reference_tasks": qualified,
            "findings": [
                {"row_id": f["row_id"], "finding_sha256": f["finding_sha256"], "path": f"findings/{f['row_id']}.json"}
                for f in ordered
            ],
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "interpretation": "Unequal-probability diagnostic selection; unweighted rates are not population estimates.",
            "source_verification": "Frozen stat checks before/after reads and at completion; content authentication separate.",
        }
        if failure is not None:
            value["failure"] = {"error_type": type(failure).__name__, "error": str(failure)}
        _write_json(output / "summary.json", value)
        return value

    def record_result(identity: str, result: dict) -> None:
        entry = entries[identity]
        source_stat = _shard_record(metadata_protocol, entry)
        _assert_unchanged(source, source_stat)
        result.update(
            row_id=identity,
            protocol_sha256=protocol_id,
            roster_sha256=roster["roster_sha256"],
            source=source.identity(),
            locator=entry["locator"],
            source_stat=source_stat,
            support_cluster_id=assignments[identity],
        )
        for record in result["records"]:
            record.update(
                task_id=_hash(
                    {"row_id": identity, "cost_model": record["cost_model"], "task_variant": record["task_variant"]}
                ),
                support_cluster_id=assignments[identity],
                source_cell_status=result.get("footprint", {}).get("source_cell_status", "unknown"),
            )
        result["finding_sha256"] = _hash(result)
        destination = findings_dir / f"{identity}.json"
        _no_symlink(destination)
        _write_json(destination, result)
        findings[identity] = result
        report("in_progress")

    try:
        _revalidate_source(source, metadata_protocol)
        for path in sorted(findings_dir.iterdir()):
            _no_symlink(path)
            if path.suffix != ".json":
                continue  # A crash may leave an uncommitted atomic-write temporary.
            identity = path.stem
            if identity not in entries:
                raise ValueError("Cached finding is absent from frozen roster")
            finding = _read_json(path)
            digest = _hash({key: value for key, value in finding.items() if key != "finding_sha256"})
            if (
                finding.get("finding_sha256") != digest
                or finding.get("row_id") != identity
                or finding.get("protocol_sha256") != protocol_id
                or finding.get("roster_sha256") != roster["roster_sha256"]
                or finding.get("source") != source.identity()
                or finding.get("locator") != entries[identity]["locator"]
                or finding.get("source_stat") != _shard_record(metadata_protocol, entries[identity])
                or len(finding.get("records", [])) != 6
            ):
                raise ValueError("Cached finding hash or frozen source identity differs")
            findings[identity] = finding
            resumed += 1
        report("in_progress")
        pending_entries = iter((identity, entry) for identity, entry in entries.items() if identity not in findings)
        if workers == 1:
            for identity, entry in pending_entries:
                record_result(identity, _audit_entry(source, entry, _shard_record(metadata_protocol, entry)))
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
                active = {}

                def fill() -> None:
                    while len(active) < workers:
                        item = next(pending_entries, None)
                        if item is None:
                            return
                        identity, entry = item
                        future = pool.submit(_audit_entry, source, entry, _shard_record(metadata_protocol, entry))
                        active[future] = identity

                fill()
                try:
                    while active:
                        done, _ = wait(active, timeout=10, return_when=FIRST_COMPLETED)
                        if not done:
                            report("in_progress")
                        for future in done:
                            record_result(active.pop(future), future.result())
                        fill()
                except BaseException as exc:
                    for future in active:
                        future.cancel()
                    # Publish cancellation before joining the bounded running workers.
                    # Workers write no artifacts and cannot admit partial outcomes.
                    report("failed", exc)
                    raise
        # Re-open metadata receipts as well as every manifest stat before claiming completion.
        final_protocol, final_summary, final_roster = _load_inputs(source, metadata)
        if final_protocol != metadata_protocol or final_summary != metadata_summary or final_roster != roster:
            raise ValueError("Frozen metadata inputs changed during route audit")
        if _implementation_hashes() != protocol["implementation_sha256"]:
            raise ValueError("Scientific implementation source changed during route audit")
        _revalidate_source(source, metadata_protocol)
        return report("complete")
    except BaseException as exc:
        report("failed", exc)
        raise
