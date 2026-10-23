"""Sequential, bounded and resumable dataset generation and release admission.

The journal owns only named transient scene artifacts. Other dataset roots are
never migration targets. Exported batch receipts retain terminal outcomes
before any raw scene bytes can be removed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import resource
import re
import signal
import stat
import sys
import threading
import zipfile
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

DATASET_VERSION = "passage-successor-v1"
GIB = 1024**3
MAX_RAW_BYTES = 16 * GIB
MIN_FREE_BYTES = 64 * GIB
SPLITS = ("train", "calibration", "validation", "test")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def default_num_cpus() -> int:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count() or 1
    return max(1, math.floor(available * 0.75))


def _split_counts(count: int, ratios: dict) -> dict[str, int]:
    if set(ratios) != set(SPLITS) or any(not math.isfinite(float(v)) or float(v) <= 0 for v in ratios.values()):
        raise ValueError("generation requires four positive split ratios")
    total = sum(float(v) for v in ratios.values())
    targets = {s: count * float(ratios[s]) / total for s in SPLITS}
    counts = {s: math.floor(targets[s]) for s in SPLITS}
    order = sorted(SPLITS, key=lambda s: (-(targets[s] - counts[s]), SPLITS.index(s)))
    for split in order[: count - sum(counts.values())]:
        counts[split] += 1
    return counts


def build_plan(
    config: dict,
    *,
    resolutions=None,
    pilot=None,
    max_raw_bytes=MAX_RAW_BYTES,
    min_free_bytes=MIN_FREE_BYTES,
    batch_scenes=None,
    num_cpus=None,
    export_memory_bytes=8 * GIB,
) -> dict:
    if config.get("generation", {}).get("dataset_version") != DATASET_VERSION:
        raise ValueError("generation.dataset_version must be passage-successor-v1")
    if max_raw_bytes <= 0 or min_free_bytes < 0 or export_memory_bytes <= 0 or (pilot is not None and pilot <= 0):
        raise ValueError("storage bounds and pilot count must be valid")
    if batch_scenes is not None and batch_scenes <= 0:
        raise ValueError("batch scene count must be positive")
    workers = default_num_cpus() if num_cpus is None else int(num_cpus)
    if workers < 0:
        raise ValueError("num_cpus must be nonnegative (0 is serial)")
    selected = sorted(set(config["resolutions"] if resolutions is None else resolutions))
    if not selected or any(r not in config["resolutions"] for r in selected):
        raise ValueError("resolution selection must be configured")
    items = []
    for resolution in selected:
        configured = config["num_samples"]
        count = (
            int(configured.get(resolution, configured.get(str(resolution), 0)))
            if isinstance(configured, dict)
            else int(configured)
        )
        if count <= 0:
            raise ValueError("configured scene counts must be positive")
        # Tensor, six worst-case routes, metadata, images, compression workspace.
        raw_bytes = resolution * resolution * 128 + 1024 * 1024
        capacity = max_raw_bytes // raw_bytes
        if capacity < 1:
            raise ValueError(f"raw byte quota cannot contain one {resolution} scene")
        # Storage stages and physical Parquet shards have independent lifetimes.
        # Long-lived workers can finish a whole admitted low-resolution roster.
        batch = min(int(batch_scenes or count), capacity, count)
        chunk_sizes = config["export"]["chunk_size"]
        shard_rows = int(chunk_sizes.get(resolution, chunk_sizes.get(str(resolution), 64)))
        if shard_rows <= 0:
            raise ValueError("Parquet shard row count must be positive")
        # Feed the persistent pool with long shard jobs. Receipt dictionaries
        # remain in the coordinator; reserve a separate conservative allowance
        # (64 KiB per scene) rather than imposing a fixed small checkpoint cap.
        receipt_capacity = max(1, (2 * export_memory_bytes) // (64 * 1024))
        export_batch = min(batch, max(4, workers) * shard_rows, receipt_capacity)
        items.append(
            {
                "resolution": resolution,
                "total_scenes": count,
                "requested_scenes": min(pilot, count) if pilot else count,
                "split_counts": _split_counts(count, config["splits"]["ratios"]),
                "estimated_raw_scene_bytes": raw_bytes,
                "batch_scenes": batch,
                "export_batch_scenes": export_batch,
                "measurement_scenes": min(64, batch),
                "sample_rows_each_split": min(64, int(config["export"]["sample"]["num_samples"])),
            }
        )
    return {
        "dataset_version": DATASET_VERSION,
        "scope": "pilot" if pilot else "full",
        "core_scenes": sum(i["total_scenes"] for i in items),
        "requested_scenes": sum(i["requested_scenes"] for i in items),
        "route_variants": sum(i["total_scenes"] for i in items) * 6,
        "configs": [f"{r}x{r}{suffix}" for r in selected for suffix in ("", "_samples")],
        "resolutions": items,
        "num_cpus": workers,
        "max_raw_bytes": int(max_raw_bytes),
        "min_free_bytes": int(min_free_bytes),
        "export_memory_bytes": int(export_memory_bytes),
        "export_coordinator_receipt_budget_bytes": int(2 * export_memory_bytes),
        "scheduling": "persistent_workers_per_resolution_staged_generation_then_parallel_export",
        "storage_estimate_status": "conservative_raw_bound_final_compression_unmeasured",
        "notebooks_expected": len(selected) * 4,
        "upload": "not_run",
        "training": "not_run",
    }


def generation_stage_bounds(total: int, maximum: int):
    """Start with a bounded capacity measurement, then fill admitted large stages."""
    if total < 0 or maximum <= 0:
        raise ValueError("invalid generation stage bounds")
    start = 0
    while start < total:
        size = min(64, maximum) if start == 0 else maximum
        end = min(total, start + size)
        yield start, end
        start = end


def _separate(*roots: Path) -> None:
    for i, root in enumerate(roots):
        for other in roots[i + 1 :]:
            if root == other or root in other.parents or other in root.parents:
                raise ValueError("input, final dataset and scratch roots must be separate trees")


def _existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def _check_space(path: Path, required: int, reserve: int) -> None:
    free = shutil.disk_usage(_existing_parent(path)).free
    if free < required + reserve:
        raise OSError(f"insufficient free space at {path}: {free} bytes, require {required + reserve}")


def scratch_usage_bytes(scratch: Path) -> int:
    """Count existing owned staging bytes when replaying an interrupted stage."""
    total = 0
    for directory, _, filenames in os.walk(scratch, followlinks=False):
        for filename in filenames:
            info = (Path(directory) / filename).lstat()
            if stat.S_ISREG(info.st_mode):
                total += info.st_size
    return total


def remaining_storage_reserve(total: int, cursor: dict, minimum: int) -> int:
    """Reserve a measured projection plus 25% margin for this resolution.

    This is a capacity forecast, not a certified compression upper bound. Later
    resolutions receive their own measurements before their second batch.
    """
    remaining = max(0, total - cursor.get("next_offset", 0))
    forecast = math.ceil(remaining * cursor.get("max_observed_core_bytes_per_row", 0) * 1.25)
    return max(minimum, forecast)


def _inventory(tiles: Path, calibration_path: Path) -> list[dict]:
    if not tiles.is_dir():
        raise FileNotFoundError(f"raw terrain directory missing: {tiles}")
    inventory = []
    for path in sorted(tiles.glob("*_DSM.tif")):
        stat = path.stat()
        inventory.append({"name": path.name, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    if not inventory:
        raise ValueError("raw terrain inventory is empty")
    calibration = json.loads(calibration_path.read_text())
    from passage.calibrate import validate_calibration_receipt

    low, high = validate_calibration_receipt(calibration)
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        raise ValueError("calibration range is invalid")
    sources = {row["filename"]: row for row in calibration["source_inventory"]}
    if set(sources) != {item["name"] for item in inventory}:
        raise ValueError("calibration source coverage differs from current complete raw inventory")
    if any(item["size_bytes"] != sources[item["name"]]["size_bytes"] for item in inventory):
        raise ValueError("calibration source inventory size changed")
    return inventory


def code_identity() -> dict:
    root = Path(__file__).resolve().parents[2]
    paths = [*root.glob("src/**/*.py"), *root.glob("notebooks/**/*.ipynb"), root / "pyproject.toml", root / "uv.lock"]
    return {str(path.relative_to(root)): hash_file(path) for path in sorted(paths) if path.is_file()}


def _producer_hashes() -> dict:
    return {
        key.removeprefix("src/"): value
        for key, value in code_identity().items()
        if key.startswith("src/") and key.endswith(".py") or key in {"pyproject.toml", "uv.lock"}
    }


def frozen_controls(config: dict) -> dict:
    holdout = config.get("hold_out", {})
    if not holdout.get("enabled", False):
        return {"hold_out_enabled": False}
    path = Path(holdout.get("mask_path", "assets/hold_out_mask.png")).resolve()
    return {"hold_out_enabled": True, "hold_out_mask_path": str(path), "hold_out_mask_sha256": hash_file(path)}


def _archive_inputs(root: Path, config: Path, calibration: Path) -> None:
    """Retain the exact dirty source snapshot, environment lock and small inputs."""
    source_root = Path(__file__).resolve().parents[2]
    destination = root / "release-receipts" / "producer-source.zip"
    if not destination.exists():
        temporary = destination.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for relative, expected in code_identity().items():
                source = source_root / relative
                if hash_file(source) != expected:
                    raise ValueError("producer source changed while archiving")
                archive.write(source, relative)
        os.replace(temporary, destination)
    for source, name in ((config, "resolved-config.yaml"), (calibration, "calibration.json")):
        target = root / "release-receipts" / name
        if target.exists() and hash_file(target) != hash_file(source):
            raise ValueError("archived configuration or calibration identity changed")
        if not target.exists():
            shutil.copyfile(source, target)


def open_journal(path: Path, identity: dict, plan: dict) -> dict:
    if path.exists():
        journal = json.loads(path.read_text())
        if journal.get("identity") != identity:
            raise ValueError("regeneration resume identity changed; use a fresh release root")
        return journal
    journal = {
        "schema_version": "passage-regeneration-journal-v1",
        "identity": identity,
        "run_id": _identity(identity),
        "plan": plan,
        "status": "planned",
        "batches": {},
        "resolutions": {},
    }
    _write_json(path, journal)
    return journal


def _directory_fd(path: Path) -> int:
    """Traverse absolute ancestry without following a substituted symlink."""
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def acquire_scratch_lock(scratch: Path) -> tuple[int, int]:
    """Share one exclusive scratch lock even when final destinations differ."""
    root_fd = _directory_fd(scratch)
    lock_fd = None
    try:
        lock_fd = os.open(".regeneration.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return root_fd, lock_fd
    except BaseException:
        if lock_fd is not None:
            os.close(lock_fd)
        os.close(root_fd)
        raise


def _hash_fd(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(descriptor, 8 * 1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def _file_identity(value) -> tuple:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def cleanup_owned_artifacts(scratch: Path, records: list[dict], manifest: dict) -> int:
    """Pin ancestry and validated file identities before deleting owned bytes."""
    if not manifest.get("raw_cleanup_safe") or not manifest.get("manifest_sha256"):
        raise ValueError("scratch cleanup requires a verified committed export")
    manifest_path = Path(manifest.get("manifest_path", ""))
    if not manifest_path.is_file() or hash_file(manifest_path) != manifest["manifest_sha256"]:
        raise ValueError("scratch cleanup requires the persisted verified manifest hash")
    scratch = Path(os.path.abspath(scratch))
    root_fd = _directory_fd(scratch)

    def pin(record):
        path = Path(record["path"])
        try:
            relative = path.relative_to(scratch)
        except ValueError as error:
            raise ValueError("cleanup artifact is outside owned scratch") from error
        if not relative.parts or ".." in relative.parts:
            raise ValueError("cleanup artifact is outside owned scratch")
        parent_fd, file_fd = os.dup(root_fd), None
        try:
            for part in relative.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = child
            file_fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent_fd)
            before = os.fstat(file_fd)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("cleanup artifact is not a regular owned file")
            if _hash_fd(file_fd) != record["sha256"] or _file_identity(os.fstat(file_fd)) != _file_identity(before):
                raise ValueError("cleanup artifact hash or identity changed")
            result = (parent_fd, file_fd, relative.name, _file_identity(before))
            parent_fd, file_fd = None, None
            return result
        except FileNotFoundError:
            return None  # Same-identity cleanup replay may find already removed files.
        except OSError as error:
            raise ValueError("cleanup artifact has unsafe or symlink ancestry outside owned scratch") from error
        finally:
            if parent_fd is not None:
                os.close(parent_fd)
            if file_fd is not None:
                os.close(file_fd)

    try:
        # Validate the entire roster before its first unlink, without holding a
        # descriptor for every file in an arbitrarily large export checkpoint.
        identities = []
        for record in records:
            pinned = pin(record)
            if pinned is not None:
                parent_fd, file_fd, _, identity = pinned
                identities.append((record, identity))
                os.close(file_fd)
                os.close(parent_fd)
        soft_limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        group_size = 1024 if soft_limit == resource.RLIM_INFINITY else min(1024, max(1, (soft_limit - 64) // 8))
        removed = 0
        for start in range(0, len(identities), group_size):
            validated = []
            try:
                # Reauthenticate and pin this bounded group's ancestry. Later
                # renames cannot redirect its unlinks outside the owned tree.
                for record, original in identities[start : start + group_size]:
                    pinned = pin(record)
                    if pinned is None:
                        continue
                    validated.append(pinned)
                    if pinned[3] != original:
                        raise ValueError("cleanup artifact identity changed before unlink")
                for parent_fd, file_fd, name, identity in validated:
                    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                    if _file_identity(current) != identity or _file_identity(os.fstat(file_fd)) != identity:
                        raise ValueError("cleanup artifact identity changed before unlink")
                for parent_fd, _, name, identity in validated:
                    if _file_identity(os.stat(name, dir_fd=parent_fd, follow_symlinks=False)) != identity:
                        raise ValueError("cleanup artifact identity changed before unlink")
                    os.unlink(name, dir_fd=parent_fd)
                    removed += 1
            finally:
                for parent_fd, file_fd, _, _ in validated:
                    os.close(file_fd)
                    os.close(parent_fd)
        return removed
    finally:
        os.close(root_fd)


def _owned_records(scratch: Path, manifest: dict) -> list[dict]:
    records = []
    for scene in manifest["scenes"]:
        split_root = scratch / str(manifest["resolution"]) / scene["split"]
        for item in scene["receipt"]["artifacts"].values():
            records.append({"path": str(split_root / item["path"]), "sha256": item["sha256"]})
        for attempt in scene["receipt"].get("attempt_history", []):
            records.append({"path": str(split_root / attempt["receipt"]), "sha256": attempt["sha256"]})
        records.append(
            {
                "path": str(split_root / "receipts" / f"{scene['index']:010d}.json"),
                "sha256": scene["receipt_file_sha256"],
            }
        )
    return records


def _emit(journal: dict, journal_path: Path, event: dict) -> None:
    _write_json(journal_path, journal)
    print(json.dumps({"run_id": journal["run_id"], **event}, sort_keys=True), flush=True)


def _thread_limits() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
    ):
        os.environ[name] = "1"


def _pilot_indices(total: int, requested: int, ratios: dict, seed: int) -> list[int]:
    from passage.generate import _assign_splits

    assignments = _assign_splits(total, ratios, seed)
    # Round-robin across split prefixes gives the pilot explicit split coverage.
    groups = {s: iter([i for i, assigned in assignments.items() if assigned == s]) for s in SPLITS}
    selected = []
    while len(selected) < requested:
        for group in groups.values():
            index = next(group, None)
            if index is not None:
                selected.append(index)
                if len(selected) == requested:
                    break
    return sorted(selected)


def verify_release(root: Path, plan: dict, *, expected_identity=None, expected_provenance_id=None) -> dict:
    """Verify shards with projected batches and at most three bytes per scene."""
    from passage.export import verify_successor_batch
    import pyarrow.parquet as pq

    summaries = {}
    unresolved_total = 0
    all_files = set()
    tiles_to_split = {}
    release_provenance = None
    expected_identity = expected_identity or {}
    configured = set(plan["configs"])
    if any(
        p.is_dir() and re.fullmatch(r"[0-9]+x[0-9]+(?:_samples)?", p.name) and p.name not in configured
        for p in root.iterdir()
    ):
        raise ValueError("release contains an untracked configuration tree")
    for item in plan["resolutions"]:
        resolution = item["resolution"]
        requested = bytearray(item["total_scenes"])
        core_indexes = bytearray(item["total_scenes"])
        sample_indexes = bytearray(item["total_scenes"])
        complete, sample_counts, statuses, variants = Counter(), Counter(), Counter(), Counter()
        reference_counts = Counter()
        successful_variants = proved_infeasible = unresolved_variants = 0
        schema = {}
        cohort_path = root / "release-receipts" / f"cohort-{resolution}.jsonl"
        temporary = cohort_path.with_suffix(".jsonl.tmp")
        supervision_path = root / "release-receipts" / f"supervision-{resolution}.jsonl"
        supervision_tmp = supervision_path.with_suffix(".jsonl.tmp")
        previous_manifest = {}
        with temporary.open("w") as cohort, supervision_tmp.open("w") as supervision:
            for path in sorted((root / "release-receipts").glob(f"batch-{resolution}-*.json")):
                manifest = verify_successor_batch(root, path.stem.removeprefix("batch-"))
                provenance = {
                    key: manifest.get(key)
                    for key in (
                        "config_sha256",
                        "calibration_sha256",
                        "producer_code_sha256",
                        "generation_provenance_id",
                    )
                }
                if (
                    release_provenance is not None
                    and provenance != release_provenance
                    or any(
                        provenance.get(key) != expected_identity[key]
                        for key in ("config_sha256", "calibration_sha256", "producer_code_sha256")
                        if key in expected_identity
                    )
                    or expected_provenance_id is not None
                    and provenance["generation_provenance_id"] != expected_provenance_id
                ):
                    raise ValueError("release batch provenance differs from frozen run identity")
                release_provenance = provenance
                if manifest.get("previous_manifest_sha256", {}) != previous_manifest:
                    raise ValueError("release batch provenance chain differs from committed history")
                previous_manifest = {path.name: manifest["manifest_sha256"]}
                batch_complete = {}
                for scene in manifest["scenes"]:
                    index, split, receipt = scene["index"], scene["split"], scene["receipt"]
                    if not 0 <= index < len(requested) or requested[index]:
                        raise ValueError("duplicate or out of range release requested index")
                    requested[index] = 1
                    statuses[f"{split}:{receipt['scene_status']}"] += 1
                    for value in receipt["variants"].values():
                        variants[f"{split}:{value['status']}"] += 1
                    if receipt["scene_status"] == "complete":
                        complete[split] += 1
                        batch_complete[index] = split
                        windows = receipt.get("source_windows", [])
                        if not windows:
                            raise ValueError("complete scene has no source-cell footprint")
                        for window in windows:
                            tile = (window["lat"], (window["lon"] + 180) % 360 - 180)
                            if tile in tiles_to_split and tiles_to_split[tile] != split:
                                raise ValueError("source-cell split independence violated")
                            tiles_to_split[tile] = split
                    calibration_sha = receipt.get("calibration_sha256", manifest.get("calibration_sha256"))
                    if (
                        not isinstance(calibration_sha, str)
                        or not re.fullmatch("[a-f0-9]{64}", calibration_sha)
                        or calibration_sha != provenance["calibration_sha256"]
                    ):
                        raise ValueError("cohort lacks authoritative calibration SHA256")
                    qualified = receipt["scene_status"] == "complete"
                    for variant_key, value in receipt["variants"].items():
                        reference = value.get("reference_status")
                        reference_counts[f"{split}:{reference}"] += 1
                        successful = value["status"] == "ok" and reference == "exact"
                        infeasible = value["status"] == "infeasible" and reference == "no_path"
                        if successful:
                            successful_variants += 1
                            supervision.write(
                                json.dumps(
                                    {
                                        "index": index,
                                        "split": split,
                                        "variant": variant_key,
                                        "resolution": resolution,
                                        "calibration_sha256": calibration_sha,
                                        "graph_id": value.get("graph_id"),
                                        "receipt_semantic_sha256": scene["receipt_semantic_sha256"],
                                    },
                                    sort_keys=True,
                                )
                                + "\n"
                            )
                        elif infeasible:
                            proved_infeasible += 1
                        else:
                            unresolved_variants += 1
                            qualified = False
                    cohort.write(
                        json.dumps(
                            {
                                "index": index,
                                "split": split,
                                "scene_status": receipt["scene_status"],
                                "qualified": qualified,
                                "calibration_sha256": calibration_sha,
                                "variant_statuses": {k: v["status"] for k, v in receipt["variants"].items()},
                                "receipt_semantic_sha256": scene["receipt_semantic_sha256"],
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                for record in manifest["files"]:
                    relative = record["path"]
                    if relative in all_files:
                        raise ValueError("duplicate committed shard")
                    all_files.add(relative)
                    with pq.ParquetFile(root / relative) as parquet:
                        kind = bool(record["samples"])
                        arrow_schema = parquet.schema_arrow
                        if kind in schema and not arrow_schema.equals(schema[kind]):
                            raise ValueError("release shard schema differs")
                        schema[kind] = arrow_schema
                        if kind:
                            sample_counts[record["split"]] += record["rows"]
                        indexes = sample_indexes if kind else core_indexes
                        for batch in parquet.iter_batches(batch_size=256, columns=["metadata"]):
                            for row in batch.to_pylist():
                                meta = row["metadata"]
                                index = meta["index"]
                                if (
                                    meta["resolution"] != resolution
                                    or meta["split"] != record["split"]
                                    or batch_complete.get(index) != record["split"]
                                    or indexes[index]
                                ):
                                    raise ValueError("parquet row identity or duplicate index mismatch")
                                indexes[index] = 1
            cohort.flush()
            os.fsync(cohort.fileno())
            supervision.flush()
            os.fsync(supervision.fileno())
        os.replace(temporary, cohort_path)
        os.replace(supervision_tmp, supervision_path)
        if sum(requested) != item["requested_scenes"] or sum(core_indexes) != sum(complete.values()):
            raise ValueError("terminal or core scene count differs from frozen plan")
        if plan["scope"] == "full":
            if not all(requested) or dict(complete) != item["split_counts"]:
                raise ValueError("full release core counts incomplete; failures retained, release not admitted")
            if any(sample_counts[s] != item["sample_rows_each_split"] for s in SPLITS):
                raise ValueError("release preview counts incomplete")
        summaries[str(resolution)] = {
            "requested_scenes": sum(requested),
            "core_counts": dict(complete),
            "sample_counts": dict(sample_counts),
            "scene_status_counts": dict(statuses),
            "variant_status_counts": dict(variants),
            "source_tile_partitions": len(tiles_to_split),
            "cohort_sha256": hash_file(cohort_path),
            "supervision_sha256": hash_file(supervision_path),
            "successful_route_variants": successful_variants,
            "proved_infeasible_variants": proved_infeasible,
            "unresolved_variants": unresolved_variants,
            "reference_status_counts": dict(reference_counts),
        }
        unresolved_total += unresolved_variants
    actual = {str(p.relative_to(root)) for p in root.rglob("*.parquet")}
    if actual != all_files:
        raise ValueError("release has uncommitted or missing parquet shards")
    result = {
        "status": "verified",
        "data_status": "verified",
        "dataset_version": DATASET_VERSION,
        "scope": plan["scope"],
        "resolutions": summaries,
        "provenance": release_provenance,
        "admission": {
            "status": "blocked" if unresolved_total else "qualified",
            "unresolved_variants": unresolved_total,
            "policy": "exact_routes_and_proved_no_path_only; successful_routes_in_separate_supervision_roster",
        },
    }
    _write_json(root / "release-receipts" / "release-verification.json", result)
    return result


def run_regenerate(
    *,
    config_filepath="config/config.yaml",
    dataset_dirpath="outputs/release/dataset",
    scratch_dirpath="outputs/release/scratch",
    tiles_dirpath="outputs/download/tiles",
    calibration_filepath="outputs/calibrate/calibrate.json",
    notebooks_dirpath="notebooks",
    resolution=None,
    num_cpus=None,
    pilot=None,
    dry_run=False,
    max_raw_bytes=MAX_RAW_BYTES,
    min_free_bytes=MIN_FREE_BYTES,
    batch_scenes=None,
    export_memory_bytes=8 * GIB,
) -> dict:
    """Run the declared full release by default, or a separately identified bounded pilot."""
    config_path, root, scratch, tiles, calibration = [
        Path(p).resolve()
        for p in (config_filepath, dataset_dirpath, scratch_dirpath, tiles_dirpath, calibration_filepath)
    ]
    _separate(root, scratch, tiles)
    config = yaml.safe_load(config_path.read_text())
    plan = build_plan(
        config,
        resolutions=resolution,
        pilot=pilot,
        max_raw_bytes=max_raw_bytes,
        min_free_bytes=min_free_bytes,
        batch_scenes=batch_scenes,
        num_cpus=num_cpus,
        export_memory_bytes=export_memory_bytes,
    )
    inventory = _inventory(tiles, calibration)
    identity = {
        "dataset_version": DATASET_VERSION,
        "config_sha256": hash_file(config_path),
        "calibration_sha256": hash_file(calibration),
        "raw_inventory_sha256": _identity(inventory),
        "calibration_source_inventory_sha256": json.loads(calibration.read_text())["source_inventory_sha256"],
        "raw_inventory_authentication": "size_mtime_inventory_scene_sources_content_hashed",
        "code_sha256": _identity(code_identity()),
        "producer_code_sha256": _producer_hashes(),
        "scope": plan["scope"],
        "frozen_controls": frozen_controls(config),
        "resolutions": [i["resolution"] for i in plan["resolutions"]],
        "pilot": pilot,
        "dataset_root": str(root),
        "scratch_root": str(scratch),
        "tiles_root": str(tiles),
        "batch_schedule": [{k: v for k, v in i.items()} for i in plan["resolutions"]],
    }
    resuming = (root / "release-receipts" / "regeneration-journal.json").exists()
    initial_raw_budget = (
        0
        if resuming
        else max(
            min(i["requested_scenes"], i["batch_scenes"]) * i["estimated_raw_scene_bytes"] for i in plan["resolutions"]
        )
    )
    _check_space(scratch, initial_raw_budget, min_free_bytes)
    _check_space(root, 0, min_free_bytes)
    if dry_run:
        return {"status": "planned", "plan": plan, "identity": identity, "raw_tiles": len(inventory), "writes": False}
    _thread_limits()
    if root.exists() and any(root.iterdir()) and not (root / "release-receipts" / "regeneration-journal.json").exists():
        raise ValueError("nonempty final root has no owned regeneration journal")
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / ".regeneration.lock").open("a+")
    previous_term = None
    scratch_descriptors = None
    generation_session = None
    if threading.current_thread() is threading.main_thread():
        previous_term = signal.getsignal(signal.SIGTERM)

        def stop(signum, frame):
            raise KeyboardInterrupt("termination requested; committed batches remain resumable")

        signal.signal(signal.SIGTERM, stop)
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        _write_json(
            root / "release-receipts" / "ready-upload.json", {"status": "verification_pending", "upload": "not_run"}
        )
        journal_path = root / "release-receipts" / "regeneration-journal.json"
        journal = open_journal(journal_path, identity, plan)
        _write_json(root / "release-receipts" / "raw-inventory.json", inventory)
        _write_json(root / "release-receipts" / "code-inventory.json", code_identity())
        _archive_inputs(root, config_path, calibration)
        controls = identity["frozen_controls"]
        if controls.get("hold_out_enabled"):
            source = Path(controls["hold_out_mask_path"])
            archived_mask = root / "release-receipts" / "hold-out-mask.png"
            if not archived_mask.exists():
                shutil.copyfile(source, archived_mask)
            if hash_file(archived_mask) != controls["hold_out_mask_sha256"]:
                raise ValueError("archived holdout controls changed")
        scratch.mkdir(parents=True, exist_ok=True)
        scratch_descriptors = acquire_scratch_lock(scratch)
        owner = scratch / ".regeneration-owner.json"
        if owner.exists() and json.loads(owner.read_text()).get("run_id") != journal["run_id"]:
            raise ValueError("scratch ownership differs from current run")
        if not owner.exists() and any(p.name != ".regeneration.lock" for p in scratch.iterdir()):
            raise ValueError("nonempty scratch root has no matching ownership receipt")
        _write_json(owner, {"run_id": journal["run_id"], "identity": identity})
        from passage.successor_generation import SuccessorGenerationSession
        from passage.export import _build_release_context, export_successor_batch, verify_successor_batch

        context = _build_release_context(
            dataset_dirpath=scratch,
            calibration_filepath=calibration,
            config_filepath=config_path,
            export_dirpath=root,
            notebooks_dirpath=notebooks_dirpath,
        )
        journal["status"] = "running"
        _emit(journal, journal_path, {"stage": "started", "plan": plan})
        try:
            for item in plan["resolutions"]:
                res = item["resolution"]
                indices = (
                    _pilot_indices(
                        item["total_scenes"],
                        item["requested_scenes"],
                        config["splits"]["ratios"],
                        int(config["splits"]["seed"]),
                    )
                    if pilot
                    else range(item["total_scenes"])
                )
                cursor = journal["resolutions"].setdefault(str(res), {"next_offset": 0})
                if cursor["next_offset"] >= len(indices):
                    cursor["stage"] = "exported"
                    continue
                if generation_session is None:
                    _emit(journal, journal_path, {"stage": "authenticating_sources"})
                    generation_session = SuccessorGenerationSession(
                        scratch, config_path, tiles, calibration, num_cpus=plan["num_cpus"]
                    )
                generation_session.prepare_resolution(res)
                stages = journal.setdefault("generation_stages", {})
                for stage_start, stage_end in generation_stage_bounds(len(indices), item["batch_scenes"]):
                    if cursor["next_offset"] >= stage_end:
                        continue
                    staged_indices = indices[stage_start:stage_end]
                    stage_id = f"{res}-{staged_indices[0]:010d}-{staged_indices[-1]:010d}"
                    generation = stages.setdefault(
                        stage_id,
                        {"stage": "planned", "resolution": res, "start_offset": stage_start, "end_offset": stage_end},
                    )
                    budget = len(staged_indices) * item["estimated_raw_scene_bytes"]
                    if generation["stage"] in {"planned", "generating"}:
                        existing_bytes = scratch_usage_bytes(scratch) if generation["stage"] == "generating" else 0
                        _check_space(scratch, max(0, budget - existing_bytes), min_free_bytes)
                        _check_space(
                            root, 0, remaining_storage_reserve(item["requested_scenes"], cursor, min_free_bytes)
                        )
                        generation["stage"] = "generating"
                        cursor["stage"] = "generating"
                        _emit(
                            journal,
                            journal_path,
                            {"stage": "generating", "generation_stage": stage_id, "scenes": len(staged_indices)},
                        )
                        before_generation = time.monotonic()
                        last_progress = [before_generation]

                        def progress(completed, total):
                            now = time.monotonic()
                            if now - last_progress[0] >= 5 or completed == total:
                                generation["completed_scenes"] = completed
                                generation["requested_scenes"] = total
                                _emit(
                                    journal,
                                    journal_path,
                                    {
                                        "stage": "generation_progress",
                                        "generation_stage": stage_id,
                                        "completed_scenes": completed,
                                        "requested_scenes": total,
                                    },
                                )
                                last_progress[0] = now

                        generated = generation_session.generate(staged_indices, progress_callback=progress)
                        if (
                            journal.get("generation_provenance_id", generated["provenance_id"])
                            != generated["provenance_id"]
                        ):
                            raise ValueError("generation provenance changed between stages")
                        journal["generation_provenance_id"] = generated["provenance_id"]
                        source_manifest = scratch / "generation_manifest.json"
                        if source_manifest.is_file():
                            archived = root / "release-receipts" / "generation-provenance.json"
                            if archived.exists() and hash_file(archived) != hash_file(source_manifest):
                                raise ValueError("archived generation provenance changed between stages")
                            if not archived.exists():
                                shutil.copyfile(source_manifest, archived)
                        elif not pilot:
                            raise ValueError("full generation lacks its archived provenance manifest")
                        generation.update(
                            stage="generated",
                            generation=generated,
                            generation_seconds=time.monotonic() - before_generation,
                        )
                        cursor["generated_offset"] = stage_end
                        _emit(
                            journal,
                            journal_path,
                            {
                                "stage": "generated",
                                "generation_stage": stage_id,
                                "terminal_counts": generated["terminal_counts"],
                            },
                        )
                    for start in range(max(stage_start, cursor["next_offset"]), stage_end, item["export_batch_scenes"]):
                        selected = list(indices[start : min(stage_end, start + item["export_batch_scenes"])])
                        batch_id = f"{res}-{selected[0]:010d}-{selected[-1]:010d}"
                        state = journal["batches"].setdefault(
                            batch_id, {"stage": "generated", "resolution": res, "generation_stage": stage_id}
                        )
                        if state["stage"] == "cleaned":
                            committed = root / "release-receipts" / f"batch-{batch_id}.json"
                            if hash_file(committed) != state["manifest_sha256"]:
                                raise ValueError("cleaned batch export manifest changed")
                            continue
                        measured_reserve = remaining_storage_reserve(item["requested_scenes"], cursor, min_free_bytes)
                        output_budget = math.ceil(
                            len(selected)
                            * cursor.get("max_observed_core_bytes_per_row", item["estimated_raw_scene_bytes"])
                            * 1.25
                        )
                        _check_space(root, output_budget, measured_reserve)
                        before = time.monotonic()
                        cursor["stage"] = "exporting"
                        if state["stage"] == "generated":
                            _emit(
                                journal,
                                journal_path,
                                {"stage": "exporting", "batch_id": batch_id, "scenes": len(selected)},
                            )
                            exported = export_successor_batch(
                                context,
                                res,
                                selected,
                                batch_id=batch_id,
                                executor=generation_session.ensure_executor(len(selected)),
                                num_cpus=plan["num_cpus"],
                                memory_budget_bytes=plan["export_memory_bytes"],
                            )
                            state.update(
                                stage="exported",
                                manifest_sha256=exported["manifest_sha256"],
                                export_seconds=time.monotonic() - before,
                                output_bytes=sum(f["size_bytes"] for f in exported["files"]),
                            )
                            _emit(
                                journal,
                                journal_path,
                                {"stage": "exported", "batch_id": batch_id, "output_bytes": state["output_bytes"]},
                            )
                        exported = verify_successor_batch(root, batch_id)
                        if exported["manifest_sha256"] != state["manifest_sha256"]:
                            raise ValueError("committed manifest changed before cleanup")
                        observed = [
                            f["size_bytes"] / f["rows"] for f in exported["files"] if not f["samples"] and f["rows"]
                        ]
                        if observed:
                            cursor["max_observed_core_bytes_per_row"] = max(
                                cursor.get("max_observed_core_bytes_per_row", 0), max(observed)
                            )
                        cursor["storage_forecast_policy"] = (
                            "max_observed_shard_bytes_per_row_times_remaining_plus_25_percent"
                        )
                        exported["manifest_path"] = str(root / "release-receipts" / f"batch-{batch_id}.json")
                        records = _owned_records(scratch, exported)
                        raw_bytes = sum(Path(r["path"]).stat().st_size for r in records if Path(r["path"]).exists())
                        if raw_bytes > max_raw_bytes:
                            raise RuntimeError("measured raw export checkpoint exceeds quota; preserved for inspection")
                        state["raw_bytes"] = raw_bytes
                        state["wall_seconds"] = time.monotonic() - before
                        state["requested_scenes_per_second"] = len(selected) / max(state["wall_seconds"], 1e-9)
                        state["parent_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                        state["child_peak_rss_kib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
                        cleanup_owned_artifacts(scratch, records, exported)
                        state["stage"] = "cleaned"
                        _write_json(root / "release-receipts" / f"workflow-{batch_id}.json", state)
                        cursor["next_offset"] = start + len(selected)
                        journal["batches"].pop(batch_id)
                        _emit(journal, journal_path, {"stage": "cleaned", "batch_id": batch_id, "raw_bytes": raw_bytes})
                    generation["stage"] = "cleaned"
                    _write_json(root / "release-receipts" / f"generation-stage-{stage_id}.json", generation)
                    stages.pop(stage_id)
                    _emit(journal, journal_path, {"stage": "generation_stage_cleaned", "generation_stage": stage_id})
                cursor["stage"] = "exported"
                _emit(journal, journal_path, {"stage": "resolution_exported", "resolution": res})
            if (
                _identity(code_identity()) != identity["code_sha256"]
                or frozen_controls(config) != identity["frozen_controls"]
            ):
                raise ValueError("producer code or frozen controls changed during regeneration")
            verified = verify_release(
                root, plan, expected_identity=identity, expected_provenance_id=journal.get("generation_provenance_id")
            )
            journal["verification"] = verified
            journal["status"] = "pilot_complete" if pilot else "notebooks"
            _emit(journal, journal_path, {"stage": journal["status"]})
            if not pilot:
                if verified["admission"]["status"] != "qualified":
                    raise RuntimeError("release admission blocked by censored, numeric or runtime label failures")
                from passage.export import prepare_export

                if not prepare_export(
                    dataset_dirpath=scratch,
                    calibration_filepath=calibration,
                    config_filepath=config_path,
                    export_dirpath=root,
                    notebooks_dirpath=notebooks_dirpath,
                    resolutions=[i["resolution"] for i in plan["resolutions"]],
                    execute_notebooks=True,
                    num_cpus=plan["num_cpus"],
                ):
                    raise RuntimeError("final notebook export failed; release remains unadmitted")
                from passage.notebook_export import MANIFEST_NAME, validate_publication_notebooks

                publication = json.loads((root / MANIFEST_NAME).read_text())
                journal["publication_admission"] = validate_publication_notebooks(
                    publication, root, resolutions=[i["resolution"] for i in plan["resolutions"]]
                )
                from passage.successor_publication import verify_successor_publication_metadata

                journal["publication_metadata_admission"] = verify_successor_publication_metadata(root)
                if (
                    _identity(code_identity()) != identity["code_sha256"]
                    or frozen_controls(config) != identity["frozen_controls"]
                ):
                    raise ValueError("producer code or frozen controls changed during notebook execution")
                journal["status"] = "ready_for_review_and_upload"
                _emit(journal, journal_path, {"stage": journal["status"], "upload": "not_run", "training": "not_run"})
            _write_json(
                root / "release-receipts" / "ready-upload.json",
                {
                    "status": journal["status"],
                    "run_id": journal["run_id"],
                    "identity": identity,
                    "upload": "not_run",
                    "training": "not_run",
                    "verification": verified,
                    "publication_admission": journal.get("publication_admission"),
                    "publication_metadata_admission": journal.get("publication_metadata_admission"),
                },
            )
            return journal
        except BaseException as error:
            journal["status"] = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
            journal["failure"] = {"type": type(error).__name__, "message": str(error)}
            _write_json(
                root / "release-receipts" / "ready-upload.json",
                {
                    "status": journal["status"],
                    "run_id": journal["run_id"],
                    "failure": journal["failure"],
                    "upload": "not_run",
                    "training": "not_run",
                },
            )
            _emit(journal, journal_path, {"stage": journal["status"], "failure": journal["failure"]})
            raise
    finally:
        if generation_session is not None:
            generation_session.close(cancel=bool(sys.exc_info()[0]))
        if previous_term is not None:
            signal.signal(signal.SIGTERM, previous_term)
        if scratch_descriptors is not None:
            for descriptor in scratch_descriptors:
                os.close(descriptor)
        lock.close()
