"""Pickle-free complete-request artifacts with exact raw numeric route bytes."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from passage_eval.execution.timing import KNOWN_SCHEMAS, LEGACY_SCHEMA, manifest_profile
from passage_eval.protocol.contracts import STATUSES, STAGES, GraphSpec, BudgetSpec, ordinary_metadata

SCHEMA = LEGACY_SCHEMA


def json_safe(value: Any):
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError(f"artifact metadata is not JSON serializable: {type(value).__name__}")


def route_hash(path: np.ndarray) -> str:
    path = np.ascontiguousarray(path)
    prefix = json.dumps({"dtype": path.dtype.str, "shape": path.shape}, sort_keys=True).encode()
    return hashlib.sha256(prefix + b"\0" + path.tobytes()).hexdigest()


def record_repair_used(record: dict[str, Any]) -> bool:
    """Return whether a timed record reports entering repair, even at zero work.

    Repair and exact fallback are separate stages. Stage-work presence counts
    as an attempted repair even if its budget is exhausted before an expansion.
    Existing v1 records carry these fields, so no artifact migration is needed.
    """
    return record["stage"] == "repair" or "repair" in record["stage_work"]


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@dataclass
class RunArtifact:
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    paths: list[np.ndarray]
    output_dir: Path | None = None
    scoring_references: dict[str, Any] | None = None

    def validate(self):
        roster = self.manifest.get("roster")
        repeats = self.manifest.get("repeats")
        if (
            self.manifest.get("schema") not in KNOWN_SCHEMAS
            or not isinstance(roster, list)
            or not roster
            or any(not isinstance(key, str) or not key for key in roster)
            or len(set(roster)) != len(roster)
        ):
            raise ValueError("artifact must declare a nonempty unique request roster and known schema")
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("artifact repeats must be a positive integer")
        quality_repeat = self.manifest.get("quality_repeat_index")
        if isinstance(quality_repeat, bool) or not isinstance(quality_repeat, int) or not 0 <= quality_repeat < repeats:
            raise ValueError("quality repeat must select a declared repeat")
        series = self.manifest.get("series")
        if not isinstance(series, str) or not series:
            raise ValueError("artifact series must be a nonempty name")
        specs = self.manifest.get("request_specs")
        metadata = self.manifest.get("request_metadata")
        if not isinstance(specs, dict) or set(specs) != set(roster):
            raise ValueError("request specifications must align with the roster")
        if not isinstance(metadata, dict) or set(metadata) != set(roster):
            raise ValueError("ordinary request metadata must align with the roster")
        for key in roster:
            spec = specs[key]
            if not isinstance(spec, dict):
                raise ValueError("request specification must be a mapping")
            try:
                graph = GraphSpec(**spec["graph"])
                BudgetSpec(**spec["budget"])
                if (
                    graph.identity() != spec["graph_id"]
                    or graph.cost_model != spec["cost_model"]
                    or graph.cost_weight != spec["cost_weight"]
                ):
                    raise ValueError("request specification graph/objective mismatch")
                for name in ("elevation_hash", "obstacles_hash", "ordinary_metadata_hash"):
                    digest = spec[name]
                    if (
                        not isinstance(digest, str)
                        or len(digest) != 64
                        or any(c not in "0123456789abcdef" for c in digest)
                    ):
                        raise ValueError("request input hashes must be sha256 identities")
                input_hash = spec["input_tensor_hash"]
                if input_hash is not None and (not isinstance(input_hash, str) or len(input_hash) != 64):
                    raise ValueError("invalid candidate input identity")
                if (
                    hashlib.sha256(
                        json.dumps(metadata[key], sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
                    ).hexdigest()
                    != spec["ordinary_metadata_hash"]
                ):
                    raise ValueError("ordinary metadata identity mismatch")
                if ordinary_metadata(metadata[key]) != metadata[key]:
                    raise ValueError("request metadata contains fields outside ordinary input allowlist")
            except (KeyError, TypeError) as error:
                raise ValueError("invalid request specification") from error
        expected = {(key, repeat, self.manifest.get("series", "warm")) for repeat in range(repeats) for key in roster}
        keys = [(row.get("sample_id"), row.get("repeat_index"), row.get("series")) for row in self.records]
        if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
            raise ValueError("terminal records must be complete and unique over roster, repeat and series")
        if len(self.paths) != len(self.records):
            raise ValueError("paths must align with terminal records")
        profile = manifest_profile(self.manifest)
        elapsed_field = profile.latency_field
        for record, path in zip(self.records, self.paths):
            if path.dtype.kind not in "iuf" or path.ndim > 2:
                raise ValueError("raw paths require real numeric arrays of at most two dimensions")
            if record.get("status") not in STATUSES or record.get("route_hash") != route_hash(path):
                raise ValueError("invalid terminal status or route hash")
            if record.get("safe_completion") is not True:
                raise ValueError("cannot persist records before safe completion")
            for key in ("connected_path", "legal_path", "obstacle_violation", "late_completion", "fallback"):
                if not isinstance(record.get(key), bool):
                    raise ValueError(f"{key} must be an explicit boolean")
            if record["status"] == "ok" and not record["legal_path"]:
                raise ValueError("terminal success requires a legal route")
            if record["legal_path"] and (not record["connected_path"] or record["obstacle_violation"]):
                raise ValueError("legal route requires connected traversable cells")
            for key in dict.fromkeys(("total_wall_ms", "failed_attempt_ms", "drain_ms", elapsed_field)):
                if (
                    isinstance(record.get(key), bool)
                    or not isinstance(record.get(key), (int, float))
                    or not math.isfinite(record[key])
                    or record[key] < 0
                ):
                    raise ValueError(f"{key} must be finite and nonnegative")
            if profile.boundary is not None:
                recost = record.get("validation_recost_ms")
                if (
                    isinstance(recost, bool)
                    or not isinstance(recost, (int, float))
                    or not math.isfinite(recost)
                    or recost < 0
                    or not math.isclose(
                        record["total_wall_ms"], record[elapsed_field] + recost, rel_tol=0, abs_tol=1e-9
                    )
                    or record.get("timing_boundary") != self.manifest.get("timing_boundary")
                ):
                    raise ValueError("service timing must match its boundary and total accounting")
            if profile.rewarm_after_restart:
                complete = record.get("warmup_complete_before_request")
                if not isinstance(record.get("rewarmed_before_request"), bool) or not (
                    complete is None or isinstance(complete, bool)
                ):
                    raise ValueError("re-warm profile records must state their warm-up state")
            deadline = specs[record["sample_id"]]["budget"]["wall_time_ms"]
            elapsed = record[elapsed_field]
            late = deadline is not None and elapsed > deadline
            if (
                record.get("deadline_ms") != deadline
                or record["late_completion"] != late
                or late
                and record["status"] != "timeout"
            ):
                raise ValueError("terminal deadline/status accounting disagrees with the admitted budget")
            expected_failed = min(elapsed, deadline) if late else elapsed if record["status"] != "ok" else 0.0
            expected_drain = max(0.0, elapsed - deadline) if late else 0.0
            if not math.isclose(
                record["failed_attempt_ms"], expected_failed, rel_tol=0, abs_tol=1e-9
            ) or not math.isclose(record["drain_ms"], expected_drain, rel_tol=0, abs_tol=1e-9):
                raise ValueError("failed-attempt and safe-drain durations disagree with terminal status")
            if record.get("stage") not in STAGES or not isinstance(record.get("stage_work"), dict):
                raise ValueError("terminal stage/work attribution is invalid")
            if not set(record["stage_work"]) <= STAGES:
                raise ValueError("unknown terminal work stage")
            for counters in (
                record.get("work"),
                record.get("resource_peaks"),
                record.get("stage_spans"),
                *record["stage_work"].values(),
            ):
                if not isinstance(counters, dict):
                    raise ValueError("work/resource/span counters must be mappings")
                for value in counters.values():
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value < 0
                    ):
                        raise ValueError("work/resource/span counters must be finite and nonnegative")


def write_run_artifact(artifact: RunArtifact, output_dir: str | Path) -> RunArtifact:
    artifact.validate()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "run_manifest.json").exists():
        raise FileExistsError(f"run artifact already exists: {output_dir}")
    blocks = [np.ascontiguousarray(path).view(np.uint8).reshape(-1) for path in artifact.paths]
    offsets = np.concatenate(([0], np.cumsum([block.size for block in blocks], dtype=np.int64)))
    values = np.concatenate(blocks) if blocks else np.empty(0, np.uint8)
    np.savez(output_dir / "paths.npz", values=values, offsets=offsets)
    with (output_dir / "requests.jsonl").open("w", encoding="utf-8") as stream:
        for record, path in zip(artifact.records, artifact.paths):
            row = dict(record, path_shape=list(path.shape), path_dtype=path.dtype.str)
            stream.write(json.dumps(json_safe(row), sort_keys=True, allow_nan=False) + "\n")
    manifest = dict(artifact.manifest)
    manifest["artifact_hashes"] = {name: sha256_file(output_dir / name) for name in ("paths.npz", "requests.jsonl")}
    (output_dir / "run_manifest.json").write_text(
        json.dumps(json_safe(manifest), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    artifact.manifest = manifest
    artifact.output_dir = output_dir
    return artifact


def read_run_artifact(output_dir: str | Path) -> RunArtifact:
    output_dir = Path(output_dir)
    manifest = json.loads((output_dir / "run_manifest.json").read_text())
    hashes = manifest.get("artifact_hashes", {})
    for name in ("paths.npz", "requests.jsonl"):
        if hashes.get(name) != sha256_file(output_dir / name):
            raise ValueError(f"{name} artifact hash mismatch")
    records = [json.loads(line) for line in (output_dir / "requests.jsonl").read_text().splitlines()]
    with np.load(output_dir / "paths.npz", allow_pickle=False) as store:
        if set(store.files) != {"values", "offsets"}:
            raise ValueError("unexpected path storage arrays")
        values, offsets = store["values"], store["offsets"]
    if (
        values.dtype != np.uint8
        or values.ndim != 1
        or offsets.dtype.kind not in "iu"
        or offsets.shape != (len(records) + 1,)
        or offsets[0] != 0
        or offsets[-1] != len(values)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("invalid raw path offsets")
    paths = []
    for index, record in enumerate(records):
        dtype = np.dtype(record["path_dtype"])
        shape = record["path_shape"]
        if (
            dtype.kind not in "iuf"
            or dtype.hasobject
            or not isinstance(shape, list)
            or len(shape) > 2
            or any(isinstance(size, bool) or not isinstance(size, int) or size < 0 for size in shape)
        ):
            raise ValueError("invalid raw path dtype/shape")
        size = math.prod(shape) * dtype.itemsize
        begin, end = int(offsets[index]), int(offsets[index + 1])
        if end - begin != size:
            raise ValueError("path shape does not match offsets")
        paths.append(np.frombuffer(values[begin:end].tobytes(), dtype=dtype).reshape(shape).copy())
    artifact = RunArtifact(manifest, records, paths, output_dir)
    artifact.validate()
    return artifact
