"""Consume bound qualification receipts without scanning data or solving routes.

The full metadata census and the deliberately stratified diagnostic roster have
different denominators. Missing completion evidence remains pending; validated
route counts never estimate population prevalence or model performance.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
from typing import Any

REVISION_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
RESOLUTIONS = (64, 128, 256, 512, 1024, 2048, 4096)
SPLITS = ("train", "validation", "calibration", "test")
RELIEF_LABELS = ("[0,200)", "[200,500)", "[500,1000)", "[1000,3000)", "[3000,inf)")
OCCUPANCY_LABELS = ("[0,0.05)", "[0.05,0.15)", "[0.15,0.3)", "[0.3,0.5)", "[0.5,1]")
COUNT_KEYS = (
    "rows",
    "task_records",
    "legal",
    "stored_cost_agreement",
    "reference_exact",
    "reference_censored",
    "reference_no_path",
    "reference_not_run",
    "qualified",
)


def _bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(value: Any) -> str:
    return hashlib.sha256(_bytes(value)).hexdigest()


def _identity(value: Mapping[str, Any]) -> dict[str, Any]:
    """Return a recorded source identity: a named source pinned to a 40-hex revision."""
    if not isinstance(value, dict):
        raise ValueError("Qualification source identity must be a JSON object")
    repo, revision = value.get("repo"), value.get("revision")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("Qualification source repository identity is missing")
    if not isinstance(revision, str) or REVISION_PATTERN.fullmatch(revision) is None:
        raise ValueError("Qualification source identity must record a pinned 40-hex revision")
    return {k: value.get(k) for k in ("root", "repo", "revision")}


def _requested_identity(value: Mapping[str, Any] | None) -> dict[str, str]:
    """Select the requested ``repo``/``revision``; absent or ``None`` fields stay unconstrained."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("Requested qualification source must be a mapping")
    requested = {k: value[k] for k in ("repo", "revision") if value.get(k) is not None}
    if "repo" in requested and (not isinstance(requested["repo"], str) or not requested["repo"].strip()):
        raise ValueError("Requested qualification source repository must be a nonempty string")
    if "revision" in requested and (
        not isinstance(requested["revision"], str) or REVISION_PATTERN.fullmatch(requested["revision"]) is None
    ):
        raise ValueError("Requested qualification source revision must be a pinned 40-hex commit")
    return requested


def _natural(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a nonnegative integer")
    return value


def _checked_report(report: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(_bytes(report))
    expected = result.get("binding", {}).pop("report_sha256", None)
    if expected is None or _hash(result) != expected:
        raise ValueError("Qualification report content hash differs")
    result["binding"]["report_sha256"] = expected
    return result


def _joint_counts(strata: Mapping[str, Any], rows: int) -> dict[str, Any]:
    if not isinstance(strata, dict) or sum(_natural(n, "Joint stratum count") for n in strata.values()) != rows:
        raise ValueError("Joint fixed-bin stratum counts differ from full core-row denominator")
    counts = [[0] * len(OCCUPANCY_LABELS) for _ in RELIEF_LABELS]
    exclusions = dict.fromkeys(("missing", "invalid", "out_of_range", "unknown"), 0)
    excluded_strata = {}
    for label, n in strata.items():
        match = re.fullmatch(r"relief=(.*);occupancy=(.*)", label)
        if match is None:
            raise ValueError("Invalid joint fixed-bin stratum label")
        relief, occupancy = match.groups()
        if relief in RELIEF_LABELS and occupancy in OCCUPANCY_LABELS:
            counts[RELIEF_LABELS.index(relief)][OCCUPANCY_LABELS.index(occupancy)] += n
        else:
            if relief not in (*RELIEF_LABELS, *exclusions) or occupancy not in (*OCCUPANCY_LABELS, *exclusions):
                raise ValueError("Unsupported joint fixed-bin stratum label")
            # Mutually exclusive categories retain the original compound
            # reasons as well. Priority is declared in the figure legend.
            category = next(k for k in exclusions if k in (relief, occupancy))
            exclusions[category] += n
            excluded_strata[label] = n
    excluded = sum(exclusions.values())
    if sum(sum(row) for row in counts) + excluded != rows:
        raise ValueError("Joint known-bin and excluded counts do not cover all core rows")
    return {
        "core_rows": rows,
        "counts": counts,
        "shares": [[n / rows for n in row] for row in counts] if rows else None,
        "excluded_rows": excluded,
        "exclusion_counts": exclusions,
        "excluded_strata": excluded_strata,
    }


def qualification_report(
    input_root: str | Path | None, resolution: int | None = None, *, source: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Read producer artifacts; reject mixed identities or corrupted bindings.

    ``input_root`` contains source-authentication/, metadata/ and route-audit/.
    A missing root is a supported public-notebook prerequisite state. Every
    artifact must record the same source identity. ``source`` optionally names
    the expected ``repo`` and/or ``revision``: artifacts recording another
    identity are rejected, and a report without artifacts records the requested
    identity (``None`` fields otherwise). The report hash covers its payload
    excluding the report hash itself. Input file hashes retain exact bytes in
    addition to the producers' canonical JSON hashes.
    """
    if resolution is not None and (isinstance(resolution, bool) or resolution not in RESOLUTIONS):
        raise ValueError("Unsupported qualification resolution")
    requested = _requested_identity(source)
    root = Path(input_root) if input_root is not None else None
    hashes, missing, artifacts = {}, [], {}
    names = (
        "source-authentication/manifest.json",
        "source-authentication/receipt.json",
        "metadata/protocol.json",
        "metadata/summary.json",
        "metadata/audit_roster.json",
        "route-audit/protocol.json",
        "route-audit/summary.json",
    )
    for name in names:
        path = root / name if root is not None else None
        if path is None or not path.is_file():
            missing.append(name)
            continue
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError(f"{name} must contain a JSON object")
        artifacts[name] = payload
        hashes[name] = hashlib.sha256(raw).hexdigest()
    sources = []
    for name, value in artifacts.items():
        source = value.get("source") if not name.startswith("source-authentication/") else value
        if source is not None:
            sources.append(_identity(source))
    if sources and any(item != sources[0] for item in sources[1:]):
        raise ValueError("Qualification artifacts have different source identities")
    if sources and any(sources[0][key] != value for key, value in requested.items()):
        raise ValueError("Qualification artifacts differ from the requested source identity")
    source = (
        sources[0] if sources else {"root": None, "repo": requested.get("repo"), "revision": requested.get("revision")}
    )
    manifest = artifacts.get(names[0])
    authentication = artifacts.get(names[1])
    auth_complete = False
    manifest_id = None
    if manifest is not None:
        if manifest.get("schema_version") != 1 or manifest.get("repo_type") != "dataset":
            raise ValueError("Unsupported source authentication manifest schema")
        files = manifest.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError("Authentication manifest files must be nonempty")
        paths = set()
        for entry in files:
            name = entry.get("path") if isinstance(entry, dict) else None
            if (
                not isinstance(name, str)
                or not name
                or PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or "\\" in name
                or str(PurePosixPath(name)) != name
            ):
                raise ValueError("Unsafe source manifest path")
            if name in paths:
                raise ValueError("Duplicate source manifest path")
            paths.add(name)
            _natural(entry.get("size"), "Source object size")
            length = {"sha256": 64, "git-blob-sha1": 40}.get(entry.get("algorithm"))
            if (
                length is None
                or not isinstance(entry.get("expected_hash"), str)
                or not re.fullmatch(rf"[0-9a-f]{{{length}}}", entry["expected_hash"])
            ):
                raise ValueError("Invalid source manifest expected object hash")
        manifest_id = hashlib.sha256(_bytes(manifest) + b"\n").hexdigest()
        if authentication is not None:
            if authentication.get("schema_version") != 1 or authentication.get("manifest_sha256") != manifest_id:
                raise ValueError("Authentication receipt manifest hash/schema differs")
            required = len(files)
            required_bytes = sum(item["size"] for item in files)
            if (
                authentication.get("required_files") != required
                or authentication.get("required_bytes") != required_bytes
            ):
                raise ValueError("Authentication receipt required object coverage differs")
            auth_complete = (
                authentication.get("status") == "complete"
                and authentication.get("content_verified") is True
                and authentication.get("verified_files") == required
                and authentication.get("verified_bytes") == required_bytes
                and all(
                    authentication.get(k) == 0
                    for k in (
                        "mismatched_files",
                        "missing_files",
                        "changed_files",
                        "error_files",
                        "interrupted_files",
                        "pending_files",
                    )
                )
            )
    protocol = artifacts.get(names[2])
    metadata = artifacts.get(names[3])
    roster = artifacts.get(names[4])
    protocol_id, roster_id = None, None
    metadata_complete = False
    progress_name = "source-authentication/progress.jsonl"
    progress_path = root / progress_name if root else None
    observations = {}
    if progress_path is not None and progress_path.is_file():
        raw = progress_path.read_bytes()
        hashes[progress_name] = hashlib.sha256(raw).hexdigest()
        # An active producer can leave an incomplete append visible to a reader.
        complete_lines = raw.splitlines() if raw.endswith(b"\n") else raw.splitlines()[:-1]
        for line in complete_lines:
            value = json.loads(line)
            if value.get("manifest_sha256") != manifest_id:
                raise ValueError("Authentication progress manifest binding differs")
            observations[value["path"]] = value
    else:
        missing.append(progress_name)
    if auth_complete:
        auth_complete = all(
            entry["path"] in observations
            and observations[entry["path"]].get("status") == "verified"
            and observations[entry["path"]].get("observed_hash") == entry["expected_hash"]
            and observations[entry["path"]].get("expected_hash") == entry["expected_hash"]
            and observations[entry["path"]].get("stat", {}).get("size") == entry["size"]
            and observations[entry["path"]].get("stat") == observations[entry["path"]].get("stat_after")
            for entry in manifest["files"]
        )
    if protocol is not None:
        if (
            protocol.get("protocol_version") != "passage-metadata-qualification-v1"
            or protocol.get("population") != "core"
        ):
            raise ValueError("Unsupported metadata protocol schema/population")
        protocol_id = _hash(protocol)
        declared_resolutions = protocol.get("resolutions")
        if (
            not isinstance(declared_resolutions, list)
            or not declared_resolutions
            or len(declared_resolutions) != len(set(declared_resolutions))
            or not set(declared_resolutions) <= set(RESOLUTIONS)
            or set(protocol.get("splits", [])) != set(SPLITS)
        ):
            raise ValueError("Unsupported metadata protocol resolution/split coverage")
        binding = protocol.get("authentication_manifest")
        if binding is not None and manifest is not None:
            if binding.get("manifest_sha256") != manifest_id or binding.get("manifest_file_sha256") != hashes[names[0]]:
                raise ValueError("Metadata protocol authentication manifest binding differs")
        if auth_complete:
            for cell in protocol.get("manifest", {}).values():
                for shard in cell["shards"]:
                    observed = observations.get(shard["path"], {}).get("stat", {})
                    if any(
                        shard.get(k) != observed.get(k) for k in ("device", "inode", "mtime_ns", "ctime_ns")
                    ) or shard.get("size_bytes") != observed.get("size"):
                        raise ValueError("Authenticated source stat differs from frozen metadata source")
        if metadata is not None:
            if (
                metadata.get("protocol_sha256") != protocol_id
                or metadata.get("protocol_version") != protocol["protocol_version"]
            ):
                raise ValueError("Metadata summary protocol binding differs")
            cells = metadata.get("cells")
            if not isinstance(cells, dict) or set(cells) != set(protocol.get("manifest", {})):
                raise ValueError("Metadata cell coverage differs from protocol")
            for key, cell in cells.items():
                for field in ("rows", "expected_footer_rows", "scanned_shards", "total_shards"):
                    _natural(cell.get(field), f"{key}/{field}")
                shards = protocol["manifest"][key].get("shards")
                if not isinstance(shards, list):
                    raise ValueError("Metadata protocol shards must be a list")
                if (
                    cell["total_shards"] != len(shards)
                    or cell["scanned_shards"] > len(shards)
                    or cell["expected_footer_rows"] != sum(s.get("rows", 0) for s in shards)
                ):
                    raise ValueError("Metadata summary footer/shard coverage differs from frozen protocol")
                for field in ("relief", "occupancy"):
                    histogram = cell.get(field)
                    if not isinstance(histogram, dict) or not isinstance(histogram.get("bins"), dict):
                        raise ValueError("Metadata fixed-bin histogram schema differs")
                    entries = list(histogram["bins"].values()) + [
                        histogram.get(k) for k in ("missing", "invalid", "out_of_range")
                    ]
                    if sum(_natural(n, "Metadata bin count") for n in entries) != cell["rows"]:
                        raise ValueError("Metadata histogram denominator differs from scanned rows")
                if "odd_strata" in cell:
                    _joint_counts(cell["odd_strata"], cell["rows"])
            metadata_complete = (
                metadata.get("status") == "complete"
                and metadata.get("population_scope_complete") is True
                and all(
                    c["status"] not in ("missing", "unreadable", "in_progress")
                    and c["rows"] == c["expected_footer_rows"]
                    and c["scanned_shards"] == c["total_shards"]
                    for c in cells.values()
                )
            )
        if roster is not None:
            roster_id = _hash({k: v for k, v in roster.items() if k != "roster_sha256"})
            if roster.get("protocol_sha256") != protocol_id or roster.get("roster_sha256") != roster_id:
                raise ValueError("Frozen roster protocol/content binding differs")
            if (
                roster.get("population_estimation") is not False
                or roster.get("frozen_before_tensor_or_reference_audit") is not True
            ):
                raise ValueError("Frozen roster must declare its diagnostic selection boundary")
    route_protocol = artifacts.get(names[5])
    route_summary = artifacts.get(names[6])
    route_complete, route_counts, finding_bindings = False, {}, {}
    if route_protocol is not None and route_summary is not None:
        route_complete, route_counts, finding_bindings = _route_evidence(
            root, route_protocol, route_summary, protocol_id, roster_id, roster, protocol, metadata
        )
        hashes.update(finding_bindings)
    selected = len(roster.get("scenes", [])) if roster is not None else None
    full_scope = (
        protocol is not None
        and set(protocol.get("resolutions", [])) == set(RESOLUTIONS)
        and set(protocol.get("splits", [])) == set(SPLITS)
        and selected == 196
    )
    if full_scope:
        quotas = {"validation": 16, "train": 4, "calibration": 4, "test": 4}
        actual = Counter((entry["resolution"], entry["split"]) for entry in roster["scenes"])
        full_scope = protocol.get("audit_counts") == quotas and actual == Counter(
            {(r, s): n for r in RESOLUTIONS for s, n in quotas.items()}
        )
    full_complete = (
        auth_complete
        and metadata_complete
        and route_complete
        and full_scope
        and protocol.get("authentication_manifest") is not None
    )
    resolutions = []
    for side in RESOLUTIONS if resolution is None else (resolution,):
        cells = {
            s: metadata["cells"][f"{side}/{s}"]
            for s in SPLITS
            if metadata is not None and f"{side}/{s}" in metadata["cells"]
        }
        scenes = [s for s in roster.get("scenes", []) if s.get("resolution") == side] if roster else []
        routes = route_counts.get(str(side), dict.fromkeys(COUNT_KEYS))
        resolutions.append(
            {
                "resolution": side,
                "metadata": {
                    "status": "complete"
                    if cells and all(c["status"] == "complete" for c in cells.values()) and len(cells) == 4
                    else "pending",
                    "population_rows": sum(c["rows"] for c in cells.values()) if cells else None,
                    "cells": cells,
                },
                "geography": metadata.get("geography", {}).get(
                    str(side), {"status": "not_run", "source_cell_windows": "unknown", "independence_claim": False}
                )
                if metadata
                else {"status": "not_run"},
                "routes": {
                    **routes,
                    "selected_scenes": len(scenes) if roster else None,
                    "requested_variants": len(scenes) * 6 if roster else None,
                    "status": "complete" if route_complete else "pending",
                },
            }
        )
    report = {
        "schema_version": 1,
        "source": source,
        "status": "complete" if full_complete else "pending" if artifacts else "not_run",
        "population_estimation": False,
        "interpretation": "Full metadata census; unequal-probability stratified diagnostic route counts. "
        "No population prevalence, confidence interval, learned-model result or strict geographic independence claim.",
        "completion": {
            "source_authentication": auth_complete,
            "metadata_census": metadata_complete,
            "route_audit": route_complete,
            "full_campaign": full_complete,
        },
        "planned": {
            "resolutions": list(RESOLUTIONS),
            "diagnostic_scenes": 196,
            "diagnostic_variants": 1176,
            "reference_max_expansions": 100000,
            "rtol": 1e-7,
            "atol": 1e-6,
        },
        "counts": {
            "population_rows": sum(c["rows"] for c in metadata.get("cells", {}).values()) if metadata else None,
            "selected_scenes": selected,
            "requested_variants": selected * 6 if selected is not None else None,
            "completed_variants": sum(c["task_records"] for c in route_counts.values()) if route_counts else None,
        },
        "resolutions": resolutions,
        "missing_prerequisites": missing,
        "prerequisite_status": {name: "available" if name in hashes else "not_run" for name in (*names, progress_name)},
        "source_authentication": authentication or {"status": "not_run", "content_verified": False},
        "binding": {
            "source_identity_sha256": _hash(source),
            "metadata_protocol_sha256": protocol_id,
            "roster_sha256": roster_id,
            "artifact_sha256": hashes,
            "consumer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
    }
    report["binding"]["report_sha256"] = _hash(report)
    return report


def _route_evidence(root, protocol, summary, metadata_id, roster_id, roster, metadata_protocol, metadata_summary):
    """Validate the campaign index and recompute counts from bound findings."""
    protocol_id = _hash(protocol)
    if (
        protocol.get("protocol_version") != "passage-route-qualification-v1"
        or summary.get("protocol_version") != protocol.get("protocol_version")
        or summary.get("protocol_sha256") != protocol_id
    ):
        raise ValueError("Route summary protocol binding differs")
    for payload in (protocol, summary):
        if payload.get("metadata_protocol_sha256") != metadata_id or payload.get("roster_sha256") != roster_id:
            raise ValueError("Route campaign metadata/roster binding differs")
        if payload.get("population_estimation") is not False:
            raise ValueError("Route campaign must remain diagnostic")
    # Producer field names and task validation are deliberately explicit.
    if (
        protocol.get("max_expansions") != 100000
        or protocol.get("rtol") != 1e-7
        or protocol.get("atol") != 1e-6
        or set(protocol.get("cost_models", [])) != {"elevation", "slope_uphill", "energy"}
        or set(protocol.get("task_variants", [])) != {"free", "obstacles"}
        or protocol.get("retain_failures_and_censoring") is not True
    ):
        raise ValueError("Route reference expansion budget differs from frozen campaign")
    if metadata_protocol is None or metadata_summary is None or roster is None:
        return False, {}, {}
    if (
        protocol.get("metadata_summary_sha256") != _hash(metadata_summary)
        or protocol.get("source_manifest_sha256") != _hash(metadata_protocol["manifest"])
        or protocol.get("authentication_manifest") != metadata_protocol.get("authentication_manifest")
    ):
        raise ValueError("Route metadata summary/source manifest binding differs")
    entries = {_hash({k: e[k] for k in ("resolution", "split", "sample_id")}): e for e in roster["scenes"]}
    if len(entries) != len(roster["scenes"]):
        raise ValueError("Frozen roster contains duplicate row identities")
    counts, hashes = {}, {}
    reasons, row_statuses, unassessed = defaultdict(Counter), defaultdict(Counter), defaultdict(Counter)
    seen = set()
    for item in summary.get("findings", []):
        name = item["path"]
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "findings":
            raise ValueError("Unsafe route finding path")
        path = root / "route-audit" / name
        if not path.resolve().is_relative_to((root / "route-audit").resolve()):
            raise ValueError("Route finding resolves outside artifact directory")
        raw = path.read_bytes()
        hashes[f"route-audit/{name}"] = hashlib.sha256(raw).hexdigest()
        finding = json.loads(raw)
        finding_id = _hash({k: v for k, v in finding.items() if k != "finding_sha256"})
        if finding.get("finding_sha256") != finding_id or item.get("finding_sha256") != finding_id:
            raise ValueError("Route finding content hash differs")
        row_id = finding.get("row_id")
        if row_id in seen or row_id not in entries or item.get("row_id") != row_id:
            raise ValueError("Route finding row is absent or duplicated in frozen roster")
        seen.add(row_id)
        entry = entries[row_id]
        if (
            finding.get("protocol_sha256") != protocol_id
            or finding.get("roster_sha256") != roster_id
            or _identity(finding.get("source")) != _identity(protocol["source"])
            or any(finding.get(k) != entry[k] for k in ("resolution", "split", "sample_id", "locator"))
        ):
            raise ValueError("Route finding source/roster/protocol identity differs")
        shard = [
            s
            for s in metadata_protocol["manifest"][f"{entry['resolution']}/{entry['split']}"]["shards"]
            if s["path"] == entry["locator"]["shard"]
        ]
        if len(shard) != 1 or finding.get("source_stat") != shard[0]:
            raise ValueError("Route finding frozen source stat differs")
        side = str(finding["resolution"])
        bucket = counts.setdefault(side, Counter(dict.fromkeys(COUNT_KEYS, 0)))
        bucket["rows"] += 1
        row_statuses[side][finding.get("status", "unknown")] += 1
        records = finding.get("records", [])
        if len(records) != 6 or {(r.get("cost_model"), r.get("task_variant")) for r in records} != {
            (model, variant) for model in protocol["cost_models"] for variant in protocol["task_variants"]
        }:
            raise ValueError("Route finding must retain exactly six diagnostic task variants")
        bucket["task_records"] += len(records)
        for record in records:
            task_id = _hash(
                {"row_id": row_id, "cost_model": record["cost_model"], "task_variant": record["task_variant"]}
            )
            if record.get("task_id") != task_id or any(
                record.get(k) != entry[k] for k in ("resolution", "split", "sample_id")
            ):
                raise ValueError("Route finding task identity differs")
            for field in ("legal", "stored_cost_agreement", "qualified"):
                if record.get(field) is not None and not isinstance(record[field], bool):
                    raise ValueError("Route finding outcome must be boolean")
                bucket[field] += int(record.get(field) is True)
                if field != "qualified":
                    unassessed[side][field] += int(record.get(field) is None)
            if not isinstance(record.get("qualified"), bool):
                raise ValueError("Route finding qualification outcome must be boolean")
            record_reasons = record.get("qualification_reasons", [])
            if not isinstance(record_reasons, list) or any(not isinstance(reason, str) for reason in record_reasons):
                raise ValueError("Route finding qualification reasons must be strings")
            reasons[side].update(record_reasons)
            status = record.get("reference_status")
            key = {
                "exact": "reference_exact",
                "budget_exhausted": "reference_censored",
                "no_path": "reference_no_path",
                "not_run": "reference_not_run",
            }.get(status)
            if key is None:
                raise ValueError(f"Unsupported exact reference status {status}")
            bucket[key] += 1
            if record.get("qualified") is True and not (
                record["legal"]
                and record["stored_cost_agreement"]
                and status == "exact"
                and record.get("optimality") is True
            ):
                raise ValueError("Qualified route lacks legality/agreement/exact optimality")
    for side, bucket in counts.items():
        expected = summary.get("by_resolution", {}).get(side, {})
        if any(expected.get(k) != bucket[k] for k in COUNT_KEYS):
            raise ValueError("Route summary counts differ from bound findings")
    if set(summary.get("by_resolution", {})) != set(counts):
        raise ValueError("Route summary resolution coverage differs from bound findings")
    total = Counter()
    for bucket in counts.values():
        total.update(bucket)
    if any(summary.get("counts", {}).get(k) != total[k] for k in COUNT_KEYS):
        raise ValueError("Route global counts differ from bound findings")
    coverage = summary.get("coverage", {})
    n = len(roster["scenes"]) if roster else None
    complete = (
        summary.get("status") == "complete"
        and n is not None
        and coverage.get("selected_rows") == n == coverage.get("completed_rows")
        and coverage.get("expected_task_records") == n * 6 == coverage.get("completed_task_records")
        and sum(c["rows"] for c in counts.values()) == n
        and sum(c["task_records"] for c in counts.values()) == n * 6
    )
    return (
        complete,
        {
            k: {
                **dict(v),
                "qualification_reasons": dict(reasons[k]),
                "row_statuses": dict(row_statuses[k]),
                "legal_unassessed": unassessed[k]["legal"],
                "stored_cost_unassessed": unassessed[k]["stored_cost_agreement"],
            }
            for k, v in counts.items()
        },
        hashes,
    )


def dataset_census_report(source, resolution: int, input_root: str | Path) -> dict[str, Any]:
    """Adapt a completed receipt to the established dataset-report interface.

    This census producer records fixed bins, not means. Means are reported as
    not computed rather than approximated from fixed-bin midpoints.
    """
    report = qualification_report(input_root, resolution)
    identity = _identity(source.identity())
    if any(identity[k] != report["source"][k] for k in ("repo", "revision")):
        raise ValueError("Requested dataset source identity differs from qualification census")
    if not report["completion"]["metadata_census"]:
        raise ValueError("Completed metadata census receipt is required for cached census reporting")
    cells = report["resolutions"][0]["metadata"]["cells"]
    if set(cells) != set(SPLITS):
        raise ValueError("Requested resolution lacks complete core split coverage")
    if source.root is not None:
        if identity["root"] != report["source"]["root"]:
            raise ValueError("Requested local dataset root differs from frozen qualification census")
        from passage.qualification import _assert_unchanged

        protocol = json.loads((Path(input_root) / "metadata/protocol.json").read_bytes())
        for cell in protocol["manifest"].values():
            for shard in cell["shards"]:
                _assert_unchanged(source, shard)
    splits = []
    for split in SPLITS:
        cell = cells[split]
        tiles = cell.get("source_tiles", {})
        splits.append(
            {
                "split": split,
                "rows": cell["rows"],
                "missing_id_or_metadata": cell.get("missing", {}).get("id", 0)
                + cell.get("missing", {}).get("metadata", 0),
                "metadata_mismatch": cell.get("resolution_mismatch", 0) + cell.get("split_mismatch", 0),
                "source_tile_count": len(tiles),
                "tile_counts": tiles,
                "calibrations": cell.get("calibration", {}),
                "mean_relief_m": None,
                "relief_denominator": None,
                "mean_obstacle_fraction": None,
                "obstacle_denominator": None,
                "mean_status": "not_computed; fixed bins are retained in the qualification census",
                "qualification_status": "not_run",
                "qualification_reason": "Metadata alone cannot qualify routes",
                "relief": cell.get("relief"),
                "occupancy": cell.get("occupancy"),
            }
        )
    return {
        "schema_version": 1,
        "resolution": resolution,
        "population": "full_core",
        "census_limit_per_split": None,
        "source": {**source.identity(), "content_verified": report["completion"]["source_authentication"]},
        "splits": splits,
        "total_rows": sum(c["rows"] for c in cells.values()),
        "statistic_scope": "Completed projected metadata census. Only fixed bins are retained for crop relief "
        "and obstacle fraction; their means were not computed and are not approximated from bin midpoints.",
        "qualification_binding": report["binding"],
        "census_status": "complete",
    }


def qualification_table_data(report: Mapping[str, Any]) -> dict[str, Any]:
    """Tabulate bound checks with full-row and valid-rectangle denominators.

    Duplicate-ID excess rows are measured within each split. Tile intersection,
    rectangle intersection and source-cell independence remain distinct checks.
    Missing evidence is ``None`` rather than an inferred zero.
    """
    report = _checked_report(report)
    result = {
        "metadata": [],
        "geography": [],
        "routes": [],
        "metadata_status": "complete" if report["completion"]["metadata_census"] else "pending",
        "route_status": "complete" if report["completion"]["route_audit"] else "pending",
        "source_cell_independence": "unknown",
        "population_estimation": False,
    }

    def total(cells, field):
        return sum(cell[field] for cell in cells.values()) if all(field in c for c in cells.values()) else None

    def valid_rectangles(cell):
        footprints = cell.get("footprints")
        if not isinstance(footprints, dict) or sum(footprints.values()) != cell["rows"]:
            return None
        return footprints.get("valid", 0)

    def tile_count(cell, field):
        return len(cell[field]) if isinstance(cell.get(field), dict) else None

    for record in report["resolutions"]:
        side = record["resolution"]
        cells = record["metadata"]["cells"]
        if report["completion"]["metadata_census"] and cells:
            core_rows = total(cells, "rows")
            valid_counts = [valid_rectangles(c) for c in cells.values()]
            valid_rows = sum(valid_counts) if all(n is not None for n in valid_counts) else None
            exclusions = Counter()
            for cell in cells.values():
                exclusions.update({k: n for k, n in cell.get("footprints", {}).items() if k != "valid"})
            result["metadata"].append(
                {
                    "resolution": side,
                    "core_rows": core_rows,
                    "duplicate_id_excess_rows": total(cells, "duplicate_id_rows"),
                    "resolution_mismatch_rows": total(cells, "resolution_mismatch"),
                    "split_mismatch_rows": total(cells, "split_mismatch"),
                    "valid_rectangle_rows": valid_rows,
                    "excluded_rectangle_rows": core_rows - valid_rows if valid_rows is not None else None,
                    "excluded_rectangle_statuses": dict(exclusions),
                    "schema_issue_cells": sum(
                        c.get("status") != "complete" or c.get("schema_consistent") is not True for c in cells.values()
                    ),
                    "primary_tile_names": len(set().union(*(set(c["source_tiles"]) for c in cells.values())))
                    if all(isinstance(c.get("source_tiles"), dict) for c in cells.values())
                    else None,
                    "support_tile_names": len(set().union(*(set(c["support_tiles"]) for c in cells.values())))
                    if all(isinstance(c.get("support_tiles"), dict) for c in cells.values())
                    else None,
                    "split_rows": {split: cell["rows"] for split, cell in cells.items()},
                }
            )
            geography = record["geography"]
            if geography.get("status") == "complete":
                for pair_name, pair in geography.get("split_pairs", {}).items():
                    left, right = pair_name.split("/")
                    if left not in cells or right not in cells:
                        raise ValueError("Geographic pair is absent from metadata cell coverage")
                    item = {
                        "resolution": side,
                        "left_split": left,
                        "right_split": right,
                        "shared_primary_tiles": pair.get("shared_source_tiles"),
                        "shared_support_tiles": pair.get("shared_support_tiles"),
                        "source_cell_independence": "unknown",
                    }
                    for name, split in (("left", left), ("right", right)):
                        cell = cells[split]
                        valid = valid_rectangles(cell)
                        item.update(
                            {
                                f"{name}_full_rows": cell["rows"],
                                f"{name}_valid_rectangle_rows": valid,
                                f"{name}_primary_tiles": tile_count(cell, "source_tiles"),
                                f"{name}_support_tiles": tile_count(cell, "support_tiles"),
                                f"{name}_positive_overlap_rows": pair.get("positive_overlap_rows", {}).get(split),
                                f"{name}_boundary_contact_rows": pair.get("boundary_contact_rows", {}).get(split),
                            }
                        )
                        for field in ("positive_overlap_rows", "boundary_contact_rows"):
                            n = item[f"{name}_{field}"]
                            if n is not None and (valid is None or _natural(n, field) > valid):
                                raise ValueError("Geographic overlap count exceeds its valid-rectangle denominator")
                    result["geography"].append(item)
        routes = record["routes"]
        result["routes"].append(
            {
                "resolution": side,
                "planned_scenes": 28,
                "planned_variants": 168,
                "selected_scenes": routes.get("selected_scenes"),
                "requested_variants": routes.get("requested_variants"),
                "completed_variants": routes.get("task_records"),
                **{
                    key: routes.get(key)
                    for key in (
                        "legal",
                        "stored_cost_agreement",
                        "reference_exact",
                        "reference_censored",
                        "reference_no_path",
                        "reference_not_run",
                        "qualified",
                    )
                },
                "legal_unassessed": routes.get("legal_unassessed"),
                "stored_cost_unassessed": routes.get("stored_cost_unassessed"),
            }
        )
    return result


def qualification_joint_odd_data(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Descriptive fixed-bin joint shares divided by all core rows per side."""
    report = _checked_report(report)
    if not report["completion"]["metadata_census"]:
        return []
    result = []
    for record in report["resolutions"]:
        cells = record["metadata"]["cells"]
        if not cells or not all(isinstance(cell.get("odd_strata"), dict) for cell in cells.values()):
            continue
        strata = Counter()
        for cell in cells.values():
            _joint_counts(cell["odd_strata"], cell["rows"])
            strata.update(cell["odd_strata"])
        result.append(
            {
                "resolution": record["resolution"],
                "relief_labels": list(RELIEF_LABELS),
                "occupancy_labels": list(OCCUPANCY_LABELS),
                **_joint_counts(dict(strata), sum(cell["rows"] for cell in cells.values())),
            }
        )
    return result


def with_report_figures(
    report: Mapping[str, Any], outputs: Mapping[str, Any], output_dir: str | Path
) -> dict[str, Any]:
    """Return a freshly hash-bound report declaring validated derived files."""
    result = _checked_report(report)
    original_hash = result["binding"].pop("report_sha256")
    directory = Path(output_dir).resolve()
    files = []
    for name, evidence in outputs.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file():
            raise ValueError("Declared qualification figure is missing or outside output directory")
        if hashlib.sha256(path.read_bytes()).hexdigest() != evidence["sha256"]:
            raise ValueError("Declared qualification figure hash differs")
        files.append(str(path))
    result["figures"] = files
    result["figure_sha256"] = dict(outputs)
    result["binding"]["consumer_report_sha256"] = original_hash
    result["binding"]["report_sha256"] = _hash(result)
    return result


def render_qualification_figures(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Export count diagnostics, including a clearly labeled prerequisite panel."""
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import LogNorm, Normalize
    from matplotlib.ticker import PercentFormatter
    from passage.plotting import OBSTACLE_COLOR, ROUTE_COLORS, TERRAIN_COLORS, terrain_colormap

    report = _checked_report(report)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = {}

    def save(fig, stem):
        try:
            for extension in ("pdf", "png"):
                path = directory / f"{stem}.{extension}"
                metadata = {"CreationDate": None, "ModDate": None} if extension == "pdf" else None
                fig.savefig(path, dpi=240, bbox_inches="tight", metadata=metadata)
                outputs[path.name] = {
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "size_bytes": path.stat().st_size,
                }
        finally:
            plt.close(fig)

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linestyle": "-",
        }
    ):
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
        records = report["resolutions"]
        available = any(r["routes"].get("task_records") is not None for r in records)
        if available:
            x = list(range(len(records)))
            for key, label, color, marker in zip(
                ("legal", "stored_cost_agreement", "qualified"),
                ("Legal stored route", "Stored-cost agreement", "Qualified optimum"),
                ROUTE_COLORS,
                ("o", "s", "^"),
            ):
                axes[0].plot(x, [r["routes"].get(key) for r in records], color=color, marker=marker, label=label)
            for key, label, color, marker in zip(
                ("reference_exact", "reference_censored", "reference_no_path", "reference_not_run"),
                ("Exact", "Censored at fixed budget", "No path", "Not run"),
                (TERRAIN_COLORS[0], TERRAIN_COLORS[3], TERRAIN_COLORS[2], OBSTACLE_COLOR),
                ("o", "s", "D", "^"),
            ):
                axes[1].plot(x, [r["routes"].get(key) for r in records], color=color, marker=marker, label=label)
            for ax in axes:
                ax.set_xticks(x, [str(r["resolution"]) for r in records], rotation=35)
                ax.set_xlabel("Raster side (cells)")
                ax.set_ylabel("Diagnostic task variants (count)")
                ax.legend(fontsize=7)
                ax.set_ylim(bottom=0)
            denominator = ", ".join(
                f"{r['resolution']}: {r['routes']['task_records']}/{r['routes']['requested_variants']}" for r in records
            )
            fig.text(
                0.5,
                0.01,
                f"Completed/requested variants by resolution: {denominator}\n"
                "Fixed 100,000-expansion references; rtol=1e−7, atol=1e−6. Stratified diagnostics; no population estimate.",
                ha="center",
                fontsize=7,
            )
        else:
            for ax, label in zip(axes, ("Stored-route checks", "Independent exact references")):
                ax.axis("off")
                ax.text(0.5, 0.55, f"{label}\nnot_run: bound findings are required", ha="center", va="center")
            fig.text(0.5, 0.08, "Qualification prerequisites pending. No route count is inferred as zero.", ha="center")
        axes[0].set_title("Legality, stored-cost agreement and qualification")
        axes[1].set_title("Reference completion and censoring")
        fig.subplots_adjust(bottom=0.30, wspace=0.32)
        save(fig, "fig_qualification")
        if report["completion"]["metadata_census"]:
            fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
            for ax, field, title in zip(
                axes, ("relief", "occupancy"), ("Recorded crop relief (m)", "Recorded obstacle fraction")
            ):
                bins = Counter()
                exclusions = Counter()
                for record in records:
                    for cell in record["metadata"]["cells"].values():
                        hist = cell.get(field, {})
                        bins.update(hist.get("bins", {}))
                        for key in ("missing", "invalid", "out_of_range"):
                            exclusions[key] += hist.get(key, 0)
                labels = sorted(bins, key=lambda label: float(label[1:].split(",")[0]))
                ax.bar(
                    range(len(labels)),
                    [bins[k] for k in labels],
                    color=TERRAIN_COLORS[0] if field == "relief" else OBSTACLE_COLOR,
                )
                ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
                ax.set_ylabel("Full census rows (count)")
                ax.set_title(title)
                ax.text(
                    0.02,
                    0.98,
                    "; ".join(f"{k}: {n}" for k, n in exclusions.items()),
                    transform=ax.transAxes,
                    va="top",
                    fontsize=7,
                )
            fig.subplots_adjust(bottom=0.3, wspace=0.3)
            save(fig, "fig_qualification_metadata")
            joint_records = qualification_joint_odd_data(report)
            joint = {r["resolution"]: r for r in joint_records}
            if any(r["core_rows"] for r in joint_records):
                max_share = max(n for r in joint_records if r["shares"] is not None for row in r["shares"] for n in row)
                norm = Normalize(vmin=0, vmax=max_share if max_share else 1)
                cmap = terrain_colormap()
                fig, axes = plt.subplots(2, 4, figsize=(8.6, 5.25))
                for ax, side in zip(axes.flat, RESOLUTIONS):
                    item = joint.get(side)
                    if item is None or item["shares"] is None:
                        ax.axis("off")
                        ax.text(
                            0.5,
                            0.5,
                            f"{side}²\nnot_run: joint metadata unavailable",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )
                        continue
                    ax.imshow(
                        item["shares"], origin="lower", interpolation="nearest", cmap=cmap, norm=norm, aspect="auto"
                    )
                    ex = item["exclusion_counts"]
                    ax.set_title(
                        f"{side}²; N={item['core_rows']:,}\nExcluded M/I/O/U: "
                        + "/".join(str(ex[k]) for k in ("missing", "invalid", "out_of_range", "unknown")),
                        fontsize=7,
                    )
                    ax.set_xticks(
                        range(5), ("0–.05", ".05–.15", ".15–.30", ".30–.50", ".50–1"), rotation=40, ha="right"
                    )
                    ax.set_yticks(range(5), ("0–200", "200–500", "500–1000", "1000–3000", "≥3000"))
                    ax.tick_params(labelsize=6.5)
                    ax.set_xlabel("Obstacle fraction", fontsize=7)
                    ax.set_ylabel("Crop relief (m)", fontsize=7)
                legend = axes.flat[7]
                legend.axis("off")
                cax = legend.inset_axes([0.10, 0.80, 0.80, 0.08])
                colorbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
                colorbar.ax.xaxis.set_major_formatter(PercentFormatter(1))
                colorbar.ax.tick_params(labelsize=6.5)
                legend.text(
                    0.05,
                    0.68,
                    "Share of all core rows\n(one scale across sizes)\n\n"
                    "N counts all core rows.\n"
                    "Zero: no observed row.\n"
                    "Fixed predeclared bins:\n"
                    "left closed/right open;\n"
                    "final obstacle bin: ≤1.\n\n"
                    "Excluded M/I/O/U counts\n"
                    "use exclusive priority:\n"
                    "missing → invalid →\n"
                    "out of range → unknown.\n\n"
                    "Descriptive metadata only.\n"
                    "No model or causal claim.",
                    va="top",
                    fontsize=6.5,
                    linespacing=1.05,
                )
                fig.subplots_adjust(left=0.09, right=0.985, bottom=0.09, top=0.91, wspace=0.63, hspace=0.88)
                save(fig, "fig_qualification_joint_odd")
            # generate.py records source-tile names as N/SdddE/Wddd. This
            # locates recorded origins, without inferring crop/cell support.
            tiles = Counter()
            for record in records:
                for cell in record["metadata"]["cells"].values():
                    tiles.update(cell.get("source_tiles", {}))
            locations, unlocated = [], 0
            missing_source_rows = sum(
                cell.get("missing", {}).get("tile.source.name", 0)
                for record in records
                for cell in record["metadata"]["cells"].values()
            )
            for name, rows in tiles.items():
                match = re.fullmatch(r"([NS])(\d{3})([EW])(\d{3})", name)
                if match:
                    ns, lat, ew, lon = match.groups()
                    latitude = int(lat) * (-1 if ns == "S" else 1)
                    longitude = int(lon) * (-1 if ew == "W" else 1)
                    if -90 <= latitude < 90 and -180 <= longitude < 180 and rows > 0:
                        locations.append((longitude, latitude, rows))
                        continue
                unlocated += rows
            if locations:
                fig, ax = plt.subplots(figsize=(8.6, 3.8))
                lon, lat, row_counts = zip(*locations)
                artist = ax.scatter(
                    lon,
                    lat,
                    c=row_counts,
                    marker="s",
                    s=20,
                    cmap=terrain_colormap(),
                    norm=LogNorm(vmin=1, vmax=max(2, max(row_counts))),
                    edgecolors=OBSTACLE_COLOR,
                    linewidths=0.25,
                )
                fig.colorbar(artist, ax=ax, label="Recorded source-tile row memberships\n(log color scale)")
                ax.set(
                    xlabel="Encoded tile longitude (degrees)",
                    ylabel="Encoded tile latitude (degrees)",
                    xlim=(-180, 180),
                    ylim=(-90, 90),
                    title="Full-census recorded source-tile support",
                )
                fig.text(
                    0.5,
                    0.01,
                    f"Tile-name origins; {unlocated:,} memberships unlocated; "
                    f"{missing_source_rows:,} rows missing source-tile names. "
                    "Counts pool the displayed resolutions/splits.\n"
                    "Shared source tiles are a support proxy; crop overlap and strict source-cell independence are separate.",
                    ha="center",
                    fontsize=7,
                )
                fig.subplots_adjust(bottom=0.25)
                save(fig, "fig_qualification_tile_support")
    return outputs
