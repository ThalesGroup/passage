"""Versioned generation from persisted ordinary inputs, with fail-closed receipts.

The ``generate`` command keeps its own sample layout; this receipt contract is opt-in.
The receipt is the commit marker. Interrupted uncommitted files are never admitted.
"""

from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import io
import json
import math
import multiprocessing
import os
import platform
import random
import re
import shutil
import subprocess
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import yaml
from PIL import Image
from rasterio.windows import Window
from tqdm import tqdm

from passage.dataset import decode_scene_inputs
from passage.reference import endpoints_connected
from passage.utils import TENSOR_SUFFIX, save_tensor
from passage_eval.protocol.contracts import GraphSpec

DATASET_VERSION = "passage-successor-v1"
SCENE_RECEIPT_VERSION = "passage-scene-receipt-v1"
SPLITS = ("train", "calibration", "validation", "test")


def semantic_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


_IMPORTED_IMPLEMENTATION_SHA256 = file_hash(Path(__file__))


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("w") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@dataclass
class GeographicPolicy:
    """All source cells of one tile belong to one immutable geographic partition.

    Blocks are indexed globally, including the antimeridian. A crop and its guard
    tiles must belong entirely to its requested split. This stronger tile policy
    implies disjoint source-cell windows, across resolutions and workers.
    """

    ratios: dict
    seed: int
    block_degrees: int = 5
    buffer_tiles: int = 1
    holdout_mask: np.ndarray | None = None

    def __post_init__(self):
        if self.block_degrees <= 2 * self.buffer_tiles or 360 % self.block_degrees or 180 % self.block_degrees:
            raise ValueError("geographic block size must divide 180 and exceed twice the buffer")
        if self.buffer_tiles < 0 or set(self.ratios) - set(SPLITS):
            raise ValueError("invalid geographic buffer or split names")
        self.ratios = {s: float(self.ratios.get(s, 0)) for s in SPLITS}
        if any(not math.isfinite(v) or v < 0 for v in self.ratios.values()) or sum(self.ratios.values()) <= 0:
            raise ValueError("invalid geographic split ratios")

    def split_for_tile(self, lat: int, lon: int) -> str:
        lon = (int(lon) + 180) % 360 - 180
        if not -90 <= lat < 90:
            raise ValueError("latitude outside source support")
        if self.holdout_mask is not None and self.holdout_mask[90 - int(lat), lon + 180] != 0:
            return "test"
        active = [s for s in SPLITS if self.ratios[s] > 0 and (self.holdout_mask is None or s != "test")]
        if not active:
            raise ValueError("no non-test geographic partition configured")
        payload = f"{self.seed}:{(int(lat) + 90) // self.block_degrees}:{(lon + 180) // self.block_degrees}".encode()
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big") / 2**64
        total = sum(self.ratios[s] for s in active)
        cumulative = 0.0
        for split in active:
            cumulative += self.ratios[split] / total
            if value < cumulative:
                return split
        return active[-1]

    def permits(self, tiles: list[tuple[int, int]], split: str) -> bool:
        for lat, lon in tiles:
            for dy in range(-self.buffer_tiles, self.buffer_tiles + 1):
                for dx in range(-self.buffer_tiles, self.buffer_tiles + 1):
                    if not -90 <= lat + dy < 90 or self.split_for_tile(lat + dy, lon + dx) != split:
                        return False
        return bool(tiles)

    def to_dict(self):
        return {
            "version": "source-tile-blocks-v1",
            "ratios": self.ratios,
            "seed": self.seed,
            "block_degrees": self.block_degrees,
            "buffer_tiles": self.buffer_tiles,
            "holdout_mask_hash": None
            if self.holdout_mask is None
            else hashlib.sha256(self.holdout_mask.tobytes()).hexdigest(),
        }


def read_source_window(
    path: Path, row: int, col: int, height: int, width: int, tile_size: int, *, allow_nodata: bool = False
) -> np.ndarray:
    """Read exact native cells; missing, nodata and incomplete support are rejected."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if min(row, col) < 0 or min(height, width) <= 0 or row + height > tile_size or col + width > tile_size:
        raise ValueError("source window is outside tile bounds")
    enforce_no_external_raster_dependencies(path)
    with rasterio.open(path) as source:
        if source.height != tile_size or not 0 < source.width <= tile_size or tile_size % source.width:
            raise ValueError("unsupported native DSM dimensions for named nearest angular normalization")
        # Explicit center-sampled nearest mapping, invariant to crop window boundaries.
        rr = np.floor((np.arange(row, row + height) + 0.5) * source.height / tile_size).astype(int)
        cc = np.floor((np.arange(col, col + width) + 0.5) * source.width / tile_size).astype(int)
        native = source.read(
            1, window=Window(int(cc[0]), int(rr[0]), int(cc[-1] - cc[0] + 1), int(rr[-1] - rr[0] + 1)), masked=True
        )
        if native.shape != (rr[-1] - rr[0] + 1, cc[-1] - cc[0] + 1):
            raise ValueError("incomplete source window")
        values = native[np.ix_(rr - rr[0], cc - cc[0])]
    invalid = np.ma.getmaskarray(values) | ~np.isfinite(values.data)
    if invalid.any() and not allow_nodata:
        raise ValueError("nodata or nonfinite source cells")
    result = np.asarray(values.data, dtype=np.float32).copy()
    result[invalid] = np.nan
    return result


def enforce_no_external_raster_dependencies(path: Path) -> None:
    from passage.calibrate import _external_sidecars, _require_self_contained_raster

    path = Path(path)
    if _external_sidecars(path):
        raise ValueError("external_raster_dependency: sidecar terrain inputs are prohibited")
    with rasterio.open(path) as source:
        _require_self_contained_raster(source, path)


def _check_authenticated_source(source: dict) -> None:
    if "fingerprint" not in source:
        return  # Explicit test/direct contexts may supply only byte identities.
    try:
        enforce_no_external_raster_dependencies(Path(source["path"]))
        if _fingerprint(source["path"]) != source["fingerprint"]:
            raise RuntimeError("authenticated terrain changed during source read")
    except (OSError, ValueError) as exc:
        raise RuntimeError("authenticated source terrain dependency changed") from exc


def _validate_frozen_context(context: dict) -> None:
    # Hashes authenticate controls initially; stat changes trigger rehashing in this live session.
    for path, recorded in context.get("control_files", {}).items():
        if _fingerprint(path) != recorded["fingerprint"]:
            if file_hash(Path(path)) != recorded["sha256"]:
                raise RuntimeError("frozen generation configuration or calibration changed during run")
    for path, expected in context.get("code_fingerprints", {}).items():
        if _fingerprint(path) != expected:
            raise RuntimeError("generating source changed during run; restart required")


def _safe_artifact(split_root: Path, relative: str) -> Path:
    result = (split_root / relative).resolve()
    if not result.is_relative_to(split_root.resolve()):
        raise ValueError("artifact escapes scene root")
    return result


def validate_scene_receipt(split_root: Path, index: int, expected_provenance_id: str | None = None) -> dict:
    """Independently validate commit marker, all paths, decoded costs and provenance."""
    from passage.reference import recost_legal_path

    split_root = Path(split_root)
    receipt_path = split_root / "receipts" / f"{index:010d}.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema_version") != SCENE_RECEIPT_VERSION or receipt.get("dataset_version") != DATASET_VERSION:
        raise ValueError("wrong release receipt version")
    if receipt.get("index") != index:
        raise ValueError("receipt index differs from requested scene")
    if expected_provenance_id is not None and receipt.get("provenance_id") != expected_provenance_id:
        raise ValueError("scene provenance differs from generation run")
    status = receipt.get("scene_status")
    variants = receipt.get("variants", {})
    if len(variants) != 6:
        raise ValueError("scene receipt must retain all six variant outcomes")
    if receipt.get("split") != split_root.name or str(receipt.get("resolution")) != split_root.parent.name:
        raise ValueError("receipt split or resolution differs from scene root")
    for attempt in receipt.get("attempt_history", []):
        history_file = _safe_artifact(split_root, attempt["receipt"])
        if not history_file.is_file() or file_hash(history_file) != attempt["sha256"]:
            raise ValueError("missing or corrupt infrastructure failure history")
    if status in {"rejected", "failed"}:
        if receipt.get("artifacts") or any(v.get("status") != "failed" for v in variants.values()):
            raise ValueError("invalid terminal scene receipt")
        return receipt
    if status != "complete":
        raise ValueError("unknown scene terminal status")
    if (
        not isinstance(receipt.get("sample_seed"), int)
        or not isinstance(receipt.get("attempts"), list)
        or not receipt["attempts"]
        or not isinstance(receipt.get("source_windows"), list)
        or not receipt["source_windows"]
    ):
        raise ValueError("complete receipt lacks required replay source, seed or attempt provenance")
    artifacts = receipt.get("artifacts", {})
    if not {"tensor", "metadata", *(f"path:{key}" for key in variants)} <= artifacts.keys():
        raise ValueError("missing required scene artifact")
    for item in artifacts.values():
        path = _safe_artifact(split_root, item["path"])
        if not path.is_file() or file_hash(path) != item["sha256"]:
            raise ValueError(f"missing or corrupt artifact: {item['path']}")
    metadata = yaml.load(
        _safe_artifact(split_root, artifacts["metadata"]["path"]).read_text(),
        Loader=getattr(yaml, "CSafeLoader", yaml.SafeLoader),
    )
    if (
        metadata.get("index") != index
        or metadata.get("resolution") != receipt["resolution"]
        or metadata.get("split") != receipt["split"]
        or metadata.get("dataset_version") != DATASET_VERSION
        or metadata.get("units") != "metres"
        or receipt["split"] != split_root.name
    ):
        raise ValueError("metadata identity or units differ from receipt")
    from passage.generate import _derive_sample_seed

    provenance = metadata.get("provenance", {})
    if (
        provenance.get("provenance_id") != receipt.get("provenance_id")
        or provenance.get("sample_seed") != receipt["sample_seed"]
        or provenance.get("source_windows") != receipt["source_windows"]
        or not isinstance(provenance.get("generation_seed"), int)
        or _derive_sample_seed(provenance["generation_seed"], receipt["resolution"], index, receipt["split"])
        != receipt["sample_seed"]
    ):
        raise ValueError("receipt replay seed, source windows or provenance differ from metadata")
    calibration_sha = receipt.get("calibration_sha256")
    if (
        not isinstance(calibration_sha, str)
        or not re.fullmatch("[a-f0-9]{64}", calibration_sha)
        or provenance.get("calibration_sha256") != calibration_sha
    ):
        raise ValueError("receipt calibration SHA differs from authoritative metadata")
    tensor = _safe_artifact(split_root, artifacts["tensor"]["path"]).read_bytes()
    inputs = decode_scene_inputs(
        {"id": f"{index:010d}", "tensor": tensor, "metadata": metadata},
        resolution=receipt["resolution"],
        split=receipt["split"],
    )
    for key, variant in variants.items():
        name, kind = key.rsplit(":", 1)
        graph = GraphSpec(**metadata["solvers"][name]["graph_spec"])
        if variant["graph_id"] != graph.identity() or kind not in {"free", "obstacles"}:
            raise ValueError("variant graph or obstacle identity mismatch")
        with _safe_artifact(split_root, artifacts[f"path:{key}"]["path"]).open() as stream:
            path = [tuple(map(int, row)) for row in csv.reader(stream)]
        stored = metadata["paths"][name][kind]
        if stored["status"] != variant["status"] or stored["cost"] != variant["cost"] or len(path) != variant["length"]:
            raise ValueError("path metadata differs from receipt")
        obstacles = inputs["obstacles"] if kind == "obstacles" else np.zeros_like(inputs["obstacles"])
        if variant["status"] == "ok":
            cost = recost_legal_path(
                inputs["elevation"], path, obstacles=obstacles, start=inputs["start"], goal=inputs["goal"], graph=graph
            )
            if not np.isclose(cost, variant["cost"], rtol=graph.rtol, atol=graph.atol):
                raise ValueError("path cost differs from canonical decoded terrain")
        elif variant["status"] in {"infeasible", "failed"}:
            if path or variant["cost"] is not None:
                raise ValueError("empty terminal route must have absent cost")
        else:
            raise ValueError("invalid path status")
    return receipt


_PROVENANCE_CACHE = {}


def _fingerprint(path):
    stat = Path(path).stat()
    return [stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns]


def prepare_provenance(
    config_path: Path, calibration_path: Path, tiles_path: Path, config: dict, policy: GeographicPolicy
):
    """Bind full raw bytes and source snapshots; stat fingerprints never replace hashes."""
    from passage import generate

    if file_hash(Path(__file__)) != _IMPORTED_IMPLEMENTATION_SHA256:
        raise ValueError("generating implementation changed after import; restart required")
    paths = sorted(tiles_path.glob("*_DSM.tif"))
    names = {entry.name for entry in tiles_path.iterdir()}
    for path in paths:
        # Scan sidecar names even on live hash/header-cache reuse. Read paths enforce src.files too.
        if any(path.name + suffix in names for suffix in (".msk", ".aux.xml", ".ovr")):
            raise ValueError("external_raster_dependency: sidecar terrain inputs are prohibited")
    fingerprints = {path.name: _fingerprint(path) for path in paths}
    # Reuse hashes only in this live process after the exact inventory was authenticated.
    # A fresh resume process always rehashes raw bytes; fingerprints guard later reads.
    cache_key = (
        str(tiles_path.resolve()),
        file_hash(config_path),
        file_hash(calibration_path),
        semantic_hash(fingerprints),
    )
    sources = _PROVENANCE_CACHE.get(cache_key)
    if sources is None:
        sources = {}
        for path in paths:
            enforce_no_external_raster_dependencies(path)
            parsed = generate.parse_lat_lon_from_filename(path.name)
            if parsed is None:
                continue
            key = f"{parsed[0]},{(parsed[1] + 180) % 360 - 180}"
            if key in sources:
                raise ValueError(f"ambiguous source tile {key}")
            with rasterio.open(path) as source:
                native = {
                    "native_shape": [source.height, source.width],
                    "native_transform": list(source.transform),
                    "native_crs": str(source.crs),
                    "normalization": "nearest-angular-center-v1",
                    "normalized_shape": [generate.TILE_SIZE_PX, generate.TILE_SIZE_PX],
                }
            sources[key] = {
                "path": str(path.resolve()),
                "name": path.name,
                "sha256": file_hash(path),
                "fingerprint": fingerprints[path.name],
                **native,
            }
            if _fingerprint(path) != fingerprints[path.name]:
                raise ValueError("raw source changed while hashing")
        _PROVENANCE_CACHE.clear()
        _PROVENANCE_CACHE[cache_key] = sources
    if not sources:
        raise ValueError("no source terrain files")
    package_root = Path(__file__).resolve().parents[1]
    snapshot = {}
    for dirname in ("passage", "passage-eval"):
        for path in sorted((package_root / dirname).rglob("*.py")):
            snapshot[str(path.relative_to(package_root))] = file_hash(path)
    versions = {
        name: importlib.metadata.version(name) for name in ("numpy", "numba", "rasterio", "zstandard", "PyYAML")
    }
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=package_root, text=True).strip()
    content = {
        "dataset_version": DATASET_VERSION,
        "configuration": config,
        "config_sha256": file_hash(config_path),
        "calibration_sha256": file_hash(calibration_path),
        "sources": {
            key: {k: v for k, v in item.items() if k not in {"path", "fingerprint"}} for key, item in sources.items()
        },
        "geographic_policy": policy.to_dict(),
        "dependencies": versions,
        "source_snapshot": snapshot,
        "python_version": platform.python_version(),
        "gdal_version": rasterio.__gdal_version__,
        "dependency_lock_sha256": file_hash(package_root.parent / "uv.lock"),
        "package_metadata_sha256": file_hash(package_root.parent / "pyproject.toml"),
        "code_commit": commit,
        "tie_policy": "float64-priority-then-row-major-node-strict-improvement-v1",
        "sample_seeding": "blake2b(seed,resolution,index,split)",
        "nodata_policy": "reject-source-crop",
        "terrain_normalization": "nearest-angular-center-v1",
        "endpoint_policy": "one-distinct-pair-per-accepted-crop-disconnected-endpoints-rejected",
        "scene_attempts": int(config.get("generation", {}).get("max_scene_attempts", 100)),
        "infrastructure_retry_policy": {
            "max_retries": int(config.get("generation", {}).get("max_infrastructure_retries", 2)),
            "same_scene_seed": True,
            "proof_of_infeasibility_retried": False,
        },
    }
    return content, sources


_WORKER_CONTEXT = None
_WORKER_RASTER_ENV = None


def _limit_worker_threads() -> None:
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


def _init_worker(context):
    global _WORKER_CONTEXT, _WORKER_RASTER_ENV
    _limit_worker_threads()
    # GDAL's host-memory-derived default is per process. Retain a bounded
    # environment for this worker's generation and subsequent export jobs.
    _WORKER_RASTER_ENV = rasterio.Env(GDAL_CACHEMAX=256 * 1024**2)
    _WORKER_RASTER_ENV.__enter__()
    source_root = Path(__file__).resolve().parents[1]
    context["_source_mismatch"] = None
    for relative, expected in context.get("source_snapshot", {}).items():
        try:
            actual = file_hash(source_root / relative)
        except OSError:
            actual = None
        if actual != expected:
            context["_source_mismatch"] = relative
            break
    _WORKER_CONTEXT = context


def _task(task):
    if _WORKER_CONTEXT.get("_source_mismatch") is not None:
        raise RuntimeError("generating source changed before worker initialization")
    return generate_successor_scene(_WORKER_CONTEXT, *task)


def _can_retry_infrastructure(receipt: dict, context: dict) -> bool:
    budget = int(
        context.get(
            "max_infrastructure_retries", context["config"].get("generation", {}).get("max_infrastructure_retries", 2)
        )
    )
    return (
        receipt["scene_status"] == "failed"
        and receipt.get("retryable") is True
        and int(receipt.get("infrastructure_attempt", 1)) <= budget
    )


def _archive_failure(split_root: Path, index: int, receipt: dict):
    attempt = int(receipt.get("infrastructure_attempt", 1))
    relative = Path("failed-attempts") / f"{index:010d}" / f"{attempt:04d}.json"
    destination = split_root / relative
    if destination.exists():
        if json.loads(destination.read_text()) != receipt:
            raise ValueError("infrastructure failure history changed during resume")
    else:
        _write_json(destination, receipt)
    history = list(receipt.get("attempt_history", []))
    history.append(
        {
            "attempt": attempt,
            "receipt": str(relative),
            "sha256": file_hash(destination),
            "failure": receipt.get("failure"),
        }
    )
    return attempt + 1, history


def _record_native_worker_failure(context: dict, task, error: BaseException) -> None:
    from passage.generate import _derive_sample_seed

    if not {"output", "resolution", "config", "provenance_id", "calibration"} <= context.keys():
        return
    index, split, _ = task
    split_root = Path(context["output"]) / str(context["resolution"]) / split
    receipt_path = split_root / "receipts" / f"{index:010d}.json"
    infrastructure_attempt, history = 1, []
    if receipt_path.exists():
        previous = validate_scene_receipt(split_root, index, context["provenance_id"])
        if not _can_retry_infrastructure(previous, context):
            return
        infrastructure_attempt, history = _archive_failure(split_root, index, previous)
    variants = {}
    for cfg in context["config"]["solvers"]:
        graph = GraphSpec(cost_model=cfg["cost_model"], cost_weight=cfg["weight"])
        for kind in ("free", "obstacles"):
            variants[f"{cfg['name']}:{kind}"] = {
                "status": "failed",
                "cost": None,
                "length": 0,
                "graph_id": graph.identity(),
            }
    receipt = {
        "schema_version": SCENE_RECEIPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "index": index,
        "resolution": context["resolution"],
        "split": split,
        "scene_status": "failed",
        "sample_seed": _derive_sample_seed(context["seed"], context["resolution"], index, split),
        "provenance_id": context["provenance_id"],
        "artifacts": {},
        "variants": variants,
        "calibration_sha256": context.get("calibration_sha256", semantic_hash(context["calibration"])),
        "attempts": [{"reason": "native_worker_lost"}],
        "retryable": True,
        "infrastructure_attempt": infrastructure_attempt,
        "attempt_history": history,
        "failure": {"type": type(error).__name__, "message": str(error)},
    }
    _write_json(receipt_path, receipt)
    shutil.rmtree(split_root / ".staging" / f"{index:010d}", ignore_errors=True)


def _stop_executor(executor: ProcessPoolExecutor, *, cancel: bool = False) -> None:
    """Join owned processes, including native work which cannot cancel cooperatively.

    Python 3.11–3.13 lack terminate_workers(). Snapshot the executor's owned
    processes before its manager clears them, terminate together, then kill any
    survivor. No scratch owner can outlive a still-running worker after return.
    """
    processes = list((getattr(executor, "_processes", None) or {}).values())
    if cancel:
        for process in processes:
            if process.is_alive():
                process.terminate()
        deadline = time.monotonic() + 5
        for process in processes:
            process.join(timeout=max(0, deadline - time.monotonic()))
        for process in processes:
            if process.is_alive():
                process.kill()
        for process in processes:
            process.join()
    executor.shutdown(wait=True, cancel_futures=True)
    for process in processes:
        process.join()


def _execute_parallel(context: dict, tasks, workers: int, *, executor=None):
    """Bound queued scenes and propagate native worker death without losing futures."""
    pending = {}
    broken = None
    affected = []
    submitting = None
    owns_executor = executor is None
    if owns_executor:
        _limit_worker_threads()
        executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init_worker,
            initargs=(context,),
        )
    failure = None
    try:
        exhausted = False
        while pending or not exhausted:
            while not exhausted and len(pending) < 2 * workers:
                try:
                    task = next(tasks)
                except StopIteration:
                    exhausted = True
                    break
                submitting = task
                pending[executor.submit(_task, task)] = task
                submitting = None
            if not pending:
                break
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                result = future.result()
                pending.pop(future)
                yield result
    except BaseException as exc:
        failure = exc
        if isinstance(exc, (BrokenProcessPool, KeyboardInterrupt, SystemExit, GeneratorExit)):
            broken = exc
            affected = list(pending.values())
            if submitting is not None:
                affected.append(submitting)
        raise
    finally:
        for future in pending:
            future.cancel()
        if failure is not None or owns_executor:
            _stop_executor(executor, cancel=failure is not None)
        if broken is not None:
            # Workers have stopped before recording terminal outcomes or clearing scratch.
            for task in affected:
                _record_native_worker_failure(context, task, broken)


def generate_successor_scene(context: dict, index: int, split: str, preview: bool = False):
    from passage import generate as legacy
    from passage.reference import solve_reference

    started = time.perf_counter()
    _validate_frozen_context(context)
    config, resolution = context["config"], context["resolution"]
    split_root = Path(context["output"]) / str(resolution) / split
    receipt_path = split_root / "receipts" / f"{index:010d}.json"
    infrastructure_attempt, history = 1, []
    if receipt_path.exists():
        receipt = validate_scene_receipt(split_root, index, context["provenance_id"])
        if not _can_retry_infrastructure(receipt, context):
            return index, split, receipt["scene_status"], time.perf_counter() - started
        infrastructure_attempt, history = _archive_failure(split_root, index, receipt)
    seed = legacy._derive_sample_seed(context["seed"], resolution, index, split)
    rng, np_rng = random.Random(seed), np.random.default_rng(seed)
    policy = context["policy"]
    candidates = context["candidates"][split]
    tile_size = context["tile_size"]
    low, high = context["calibration"]["global_min"], context["calibration"]["global_max"]
    solver_cfgs = config["solvers"]
    variants = {
        f"{cfg['name']}:{kind}": {
            "status": "failed",
            "cost": None,
            "length": 0,
            "graph_id": GraphSpec(cost_model=cfg["cost_model"], cost_weight=cfg["weight"]).identity(),
        }
        for cfg in solver_cfgs
        for kind in ("free", "obstacles")
    }
    receipt = {
        "schema_version": SCENE_RECEIPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "index": index,
        "resolution": resolution,
        "split": split,
        "sample_seed": seed,
        "provenance_id": context["provenance_id"],
        "calibration_sha256": context.get("calibration_sha256", semantic_hash(context["calibration"])),
        "artifacts": {},
        "variants": variants,
        "attempts": [],
        "scene_status": "rejected",
        "infrastructure_attempt": infrastructure_attempt,
        "attempt_history": history,
    }
    stage = split_root / ".staging" / f"{index:010d}"
    if stage.exists():
        shutil.rmtree(stage)  # Only this uncommitted deterministic transaction's scratch.
    stage.mkdir(parents=True)
    try:
        if not candidates:
            receipt["attempts"].append({"reason": "no_geographic_source_support"})
        for attempt in range(context["max_attempts"]) if candidates else ():
            center_lat, center_lon = candidates[rng.randrange(len(candidates))]
            center = context["sources"][f"{center_lat},{center_lon}"]
            attempted_windows = []
            authoritative_started = False
            try:
                _check_authenticated_source(center)
                terrain = read_source_window(
                    Path(center["path"]), 0, 0, tile_size, tile_size, tile_size, allow_nodata=True
                )
                target = rng.uniform(float(low), float(high))
                if not np.isfinite(terrain).any():
                    raise ValueError("no valid source terrain cells")
                row, col = divmod(int(np.nanargmin(np.abs(terrain - target))), tile_size)
                # Tile-relative global crop origin. Negative offsets are legitimate neighboring tiles.
                y, x = row - rng.randrange(resolution), col - rng.randrange(resolution)
                windows = []
                for tile_row in range(y // tile_size, (y + resolution - 1) // tile_size + 1):
                    for tile_col in range(x // tile_size, (x + resolution - 1) // tile_size + 1):
                        lat, lon = center_lat - tile_row, (center_lon + tile_col + 180) % 360 - 180
                        top, left = max(y, tile_row * tile_size), max(x, tile_col * tile_size)
                        bottom, right = (
                            min(y + resolution, (tile_row + 1) * tile_size),
                            min(x + resolution, (tile_col + 1) * tile_size),
                        )
                        windows.append(
                            {
                                "lat": lat,
                                "lon": lon,
                                "row": top - tile_row * tile_size,
                                "col": left - tile_col * tile_size,
                                "height": bottom - top,
                                "width": right - left,
                                "dest_row": top - y,
                                "dest_col": left - x,
                            }
                        )
                attempted_windows = windows
                if not policy.permits([(w["lat"], w["lon"]) for w in windows], split):
                    raise ValueError("geographic_footprint_or_guard_crosses_partition")
                crop = np.empty((resolution, resolution), dtype=np.float32)
                for window in windows:
                    source = context["sources"].get(f"{window['lat']},{window['lon']}")
                    if source is None:
                        raise FileNotFoundError(f"missing terrain {window['lat']},{window['lon']}")
                    _check_authenticated_source(source)
                    block = read_source_window(
                        Path(source["path"]), window["row"], window["col"], window["height"], window["width"], tile_size
                    )
                    dy, dx = window["dest_row"], window["dest_col"]
                    crop[dy : dy + window["height"], dx : dx + window["width"]] = block
                    _check_authenticated_source(source)
                    native_shape = source.get("native_shape", [tile_size, tile_size])
                    nr, nc = native_shape
                    r0 = int(math.floor((window["row"] + 0.5) * nr / tile_size))
                    c0 = int(math.floor((window["col"] + 0.5) * nc / tile_size))
                    r1 = int(math.floor((window["row"] + window["height"] - 0.5) * nr / tile_size)) + 1
                    c1 = int(math.floor((window["col"] + window["width"] - 0.5) * nc / tile_size)) + 1
                    window.update(
                        {
                            "name": source["name"],
                            "sha256": source["sha256"],
                            "native_shape": native_shape,
                            "native_transform": source.get("native_transform"),
                            "normalization": "nearest-angular-center-v1",
                            "normalized_shape": [tile_size, tile_size],
                            "native_window": {"row": r0, "col": c0, "height": r1 - r0, "width": c1 - c0},
                        }
                    )
                if crop.min() < low or crop.max() > high:
                    raise ValueError("source elevations outside frozen calibration; clipping prohibited")
                start = (row - y, col - x)
                flat = rng.randrange(resolution**2 - 1)
                if flat >= start[0] * resolution + start[1]:
                    flat += 1
                goal = divmod(flat, resolution)
                obstacle_settings = legacy._resolve_obstacle_config(config)
                if obstacle_settings["enabled"]:
                    obstacles, target_ratio = legacy._generate_obstacles_mask(
                        crop.shape,
                        obstacle_settings["ratio_max"],
                        obstacle_settings["size_min_ratio"],
                        obstacle_settings["size_max_ratio"],
                        obstacle_settings["config"],
                        marker_positions=[start, goal],
                        rng=np_rng,
                        generation_cfg=obstacle_settings["generation_cfg"],
                    )
                else:
                    obstacles, target_ratio = np.zeros_like(crop, dtype=np.uint8), 0.0
                if not endpoints_connected(obstacles, start, goal):
                    # The reference would prove no path: reject this input like any unusable crop.
                    raise ValueError("endpoints_disconnected_by_obstacles")
                normalized = ((crop.astype(np.float64) - low) / (high - low)).astype(np.float32)
                markers = np.full(crop.shape, -1, dtype=np.float32)
                markers[start], markers[goal] = 0, 1
                tensor = np.stack([normalized, markers, obstacles.astype(np.float32)], axis=-1)

                def geo(yy, xx):
                    return {
                        "lat": center_lat + 1 - yy / tile_size,
                        "lon": (center_lon + xx / tile_size + 180) % 360 - 180,
                    }

                metadata = {
                    "index": index,
                    "resolution": resolution,
                    "split": split,
                    "dataset_version": DATASET_VERSION,
                    "units": "metres",
                    "calibration": {"min": low, "max": high},
                    "markers": {
                        "start_point": {"i": start[0], "j": start[1]},
                        "end_point": {"i": goal[0], "j": goal[1]},
                    },
                    "tile": {"source": {"lat": center_lat, "lon": center_lon}, "used": windows, "overlapping": windows},
                    "crop": {
                        "top_left": geo(y, x),
                        "bottom_right": geo(y + resolution, x + resolution),
                        "center": geo(y + resolution / 2, x + resolution / 2),
                    },
                    "obstacles": {
                        "enabled": obstacle_settings["enabled"],
                        "ratio_actual": float(obstacles.mean()),
                        "ratio_target": float(target_ratio),
                    },
                    "provenance": {
                        "generation_seed": context["seed"],
                        "sample_seed": seed,
                        "provenance_id": context["provenance_id"],
                        "source_windows": windows,
                        "calibration_sha256": receipt["calibration_sha256"],
                    },
                    "solvers": {},
                    "paths": {},
                }
                authoritative_started = True
                # Persist calibration, resolved graph configuration and tensor BEFORE authoritative solves.
                for cfg in solver_cfgs:
                    graph = GraphSpec(cost_model=cfg["cost_model"], cost_weight=cfg["weight"])
                    metadata["solvers"][cfg["name"]] = dict(
                        cfg,
                        graph_spec=graph.to_dict(),
                        reference_engine="passage.reference.solve_reference",
                        arithmetic_precision="float64",
                    )
                tensor_relative = (
                    Path("tensors")
                    / legacy.lat_to_str(center_lat)
                    / legacy.lon_to_str(center_lon)
                    / f"{index:010d}{TENSOR_SUFFIX}"
                )
                metadata_relative = (
                    Path("metadata")
                    / legacy.lat_to_str(center_lat)
                    / legacy.lon_to_str(center_lon)
                    / f"{index:010d}.yaml"
                )
                (stage / tensor_relative).parent.mkdir(parents=True, exist_ok=True)
                save_tensor(tensor, stage / tensor_relative)
                (stage / metadata_relative).parent.mkdir(parents=True, exist_ok=True)
                (stage / metadata_relative).write_text(yaml.safe_dump(metadata, sort_keys=False))
                persisted_metadata = yaml.safe_load((stage / metadata_relative).read_text())
                inputs = decode_scene_inputs(
                    {
                        "id": f"{index:010d}",
                        "tensor": (stage / tensor_relative).read_bytes(),
                        "metadata": persisted_metadata,
                    },
                    resolution=resolution,
                    split=split,
                )
                results = {}
                artifact_paths = {"tensor": tensor_relative, "metadata": metadata_relative}
                for cfg in solver_cfgs:
                    name = cfg["name"]
                    graph = GraphSpec(**metadata["solvers"][name]["graph_spec"])
                    metadata["paths"][name] = {}
                    for kind in ("free", "obstacles"):
                        mask = inputs["obstacles"] if kind == "obstacles" else np.zeros_like(inputs["obstacles"])
                        result = solve_reference(
                            inputs["elevation"],
                            mask,
                            inputs["start"],
                            inputs["goal"],
                            graph=graph,
                            max_expansions=context["max_expansions"],
                        )
                        if result.status == "no_path":
                            raise RuntimeError("reference proved no path for endpoints accepted as connected")
                        status = {"exact": "ok", "no_path": "infeasible", "censored": "failed"}[result.status]
                        variant = {
                            "status": status,
                            "cost": result.cost,
                            "length": len(result.path),
                            "graph_id": graph.identity(),
                            "reference_status": result.status,
                            "expansions": result.expansions,
                        }
                        variants[f"{name}:{kind}"] = variant
                        metadata["paths"][name][kind] = dict(
                            variant, considered_area=result.expansions, crosses_obstacles=False
                        )
                        relative = (
                            Path("paths")
                            / name
                            / kind
                            / legacy.lat_to_str(center_lat)
                            / legacy.lon_to_str(center_lon)
                            / f"{index:010d}.csv"
                        )
                        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
                        with (stage / relative).open("w", newline="") as stream:
                            csv.writer(stream).writerows(result.path)
                        artifact_paths[f"path:{name}:{kind}"] = relative
                        results[f"{name}:{kind}"] = result.path
                (stage / metadata_relative).write_text(yaml.safe_dump(metadata, sort_keys=False))
                if preview:
                    for cfg in solver_cfgs:
                        name = cfg["name"]
                        relative = (
                            Path("images")
                            / name
                            / legacy.lat_to_str(center_lat)
                            / legacy.lon_to_str(center_lon)
                            / f"{index:010d}.png"
                        )
                        (stage / relative).parent.mkdir(parents=True, exist_ok=True)
                        image = legacy.tensor_to_image(
                            inputs["input_tensor"],
                            path_pixels=list(results[f"{name}:free"]),
                            alt_path_pixels=list(results[f"{name}:obstacles"]),
                        )
                        legacy._save_png(stage / relative, image)
                        artifact_paths[f"image:{name}"] = relative
                for key, relative in artifact_paths.items():
                    with (stage / relative).open("rb") as stream:
                        os.fsync(stream.fileno())
                    receipt["artifacts"][key] = {"path": str(relative), "sha256": file_hash(stage / relative)}
                receipt["source_windows"] = windows
                receipt["attempts"].append({"attempt": attempt, "reason": "accepted_input"})
                receipt["scene_status"] = "complete"
                # Install files first; commit marker last. A crash before receipt is replayed.
                for relative in artifact_paths.values():
                    destination = split_root / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(stage / relative, destination)
                _validate_frozen_context(context)
                for window in windows:
                    _check_authenticated_source(context["sources"][f"{window['lat']},{window['lon']}"])
                _write_json(receipt_path, receipt)
                validate_scene_receipt(split_root, index, context["provenance_id"])
                return index, split, "complete", time.perf_counter() - started
            except (FileNotFoundError, ValueError) as exc:
                # Source-input rejection only. Once solving begins, failure must never resample labels.
                if authoritative_started or "external_raster_dependency" in str(exc):
                    raise
                receipt["attempts"].append(
                    {
                        "attempt": attempt,
                        "reason": str(exc),
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "source_windows": attempted_windows,
                    }
                )
        _write_json(receipt_path, receipt)
        return index, split, "rejected", time.perf_counter() - started
    except Exception as exc:
        receipt.update(
            scene_status="failed",
            artifacts={},
            variants={key: dict(value, status="failed", cost=None, length=0) for key, value in variants.items()},
            failure={"type": type(exc).__name__, "message": str(exc)},
            retryable=isinstance(exc, (OSError, MemoryError, BrokenProcessPool)),
        )
        _write_json(receipt_path, receipt)
        raise  # Infrastructure/solver errors remain visible and stop the run.
    finally:
        shutil.rmtree(stage, ignore_errors=True)


class SuccessorGenerationSession:
    """One authenticated coordinator and one owned worker pool per resolution.

    The exposed executor may run sequential export stages between generate()
    calls. Callers must finish those stages before changing resolution. A context
    manager aborts and joins owned workers on any exception, before scratch locks
    can be released by the surrounding owner.
    """

    def __init__(
        self,
        outputs_dirpath,
        config_filepath,
        tiles_dirpath,
        calibration_filepath,
        *,
        num_cpus: int | None = None,
        force: bool = False,
    ):
        from passage import generate as legacy

        config_path, calibration_path, tiles_path = map(Path, (config_filepath, calibration_filepath, tiles_dirpath))
        configuration_bytes = config_path.read_bytes()
        config = yaml.safe_load(configuration_bytes)
        if config.get("generation", {}).get("dataset_version") != DATASET_VERSION:
            raise ValueError("release generation requires explicit dataset version")
        from passage.calibrate import validate_calibration_receipt

        calibration_bytes = calibration_path.read_bytes()
        calibration = json.loads(calibration_bytes)
        low, high = validate_calibration_receipt(calibration)
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("invalid calibration")
        config.setdefault("generation", {})
        settings = config["generation"]
        configs = config.get("solvers", [])
        if (
            len(configs) != 3
            or len({cfg["name"] for cfg in configs}) != 3
            or {cfg["cost_model"] for cfg in configs} != {"elevation", "energy", "slope_uphill"}
        ):
            raise ValueError("release generation requires exactly three uniquely named canonical objectives")
        for cfg in configs:
            if cfg.get("diagonal_movement", "always") != "always":
                raise ValueError("release GraphSpec requires eight connected destination-only graph")
            GraphSpec(cost_model=cfg["cost_model"], cost_weight=cfg["weight"])
        obstacle_settings = legacy._resolve_obstacle_config(config)
        if not obstacle_settings["impassable"]:
            raise ValueError("release generation requires blocked obstacle policy")
        ratios = config.get("splits", {}).get("ratios", {"train": 1.0})
        split_seed = int(config.get("splits", {}).get("seed", 42))
        geo = settings.get("geographic_policy", {})
        mask = None
        mask_path = None
        control_paths = [config_path, calibration_path]
        loaded_hashes = {
            str(config_path.resolve()): hashlib.sha256(configuration_bytes).hexdigest(),
            str(calibration_path.resolve()): hashlib.sha256(calibration_bytes).hexdigest(),
        }
        holdout = config.get("hold_out", {})
        if holdout.get("enabled", False):
            mask_path = Path(holdout.get("mask_path", legacy.DEFAULT_HOLD_OUT_MASK_PATH))
            mask_bytes = mask_path.read_bytes()
            with Image.open(io.BytesIO(mask_bytes)) as image:
                pixels = np.asarray(image.convert("RGB"))
            if pixels.shape[:2] != (181, 361):
                raise ValueError("geographic holdout mask must have 181x361 angular cells")
            mask = np.any(pixels > 0, axis=2).astype(np.uint8)
            control_paths.append(mask_path)
            loaded_hashes[str(mask_path.resolve())] = hashlib.sha256(mask_bytes).hexdigest()
        policy = GeographicPolicy(
            ratios, split_seed, int(geo.get("block_degrees", 5)), int(geo.get("buffer_tiles", 1)), mask
        )
        control_files = {
            str(path.resolve()): {"sha256": file_hash(path), "fingerprint": _fingerprint(path)}
            for path in control_paths
        }
        if any(control_files[path]["sha256"] != expected for path, expected in loaded_hashes.items()):
            raise ValueError("generation controls changed while loading inputs")
        provenance, sources = prepare_provenance(config_path, calibration_path, tiles_path, config, policy)
        for path, recorded in control_files.items():
            if file_hash(Path(path)) != recorded["sha256"]:
                raise ValueError("configuration or calibration changed during provenance preparation")
        calibrated_sources = {entry["filename"]: entry["sha256"] for entry in calibration["source_inventory"]}
        current_sources = {entry["name"]: entry["sha256"] for entry in sources.values()}
        if calibrated_sources != current_sources:
            raise ValueError("raw terrain inventory differs from authenticated calibration receipt")
        provenance["holdout_mask_file_sha256"] = None if mask_path is None else loaded_hashes[str(mask_path.resolve())]
        provenance_id = semantic_hash(provenance)
        outputs = Path(outputs_dirpath)
        if force and outputs.exists():
            raise ValueError("release force overwrite prohibited; use a fresh output root")
        outputs.mkdir(parents=True, exist_ok=True)
        run_path = outputs / "generation_manifest.json"
        if run_path.exists() and json.loads(run_path.read_text()).get("provenance_id") != provenance_id:
            raise ValueError("generation provenance mismatch; resume refused")
        # Archive the exact dirty source bytes, rather than relying on HEAD or hashes alone.
        source_root = Path(__file__).resolve().parents[1]
        for relative, expected_hash in provenance["source_snapshot"].items():
            destination = outputs / "generation-source-snapshot" / relative
            if destination.exists():
                if file_hash(destination) != expected_hash:
                    raise ValueError("archived generating source differs from run manifest")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_root / relative, destination)
                if file_hash(destination) != expected_hash:
                    raise ValueError("generating source changed during snapshot archival")
        for path, recorded in control_files.items():
            label = (
                "configuration"
                if Path(path) == config_path.resolve()
                else "calibration"
                if Path(path) == calibration_path.resolve()
                else "holdout-mask"
            )
            destination = outputs / "generation-input-snapshot" / label / Path(path).name
            if destination.exists() and file_hash(destination) != recorded["sha256"]:
                raise ValueError("archived generation configuration or calibration differs")
            if not destination.exists():
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, destination)
                if file_hash(destination) != recorded["sha256"]:
                    raise ValueError("generation control changed during archival")
        _write_json(run_path, dict(provenance, provenance_id=provenance_id))
        candidates = {split: [] for split in SPLITS}
        usable_names = {item["filename"] for item in calibration["tile_stats"]}
        for key in sorted(sources):
            lat, lon = map(int, key.split(","))
            split = policy.split_for_tile(lat, lon)
            # Source validity and geographic guards are fixed before any label outcome.
            if sources[key]["name"] in usable_names and policy.permits([(lat, lon)], split):
                candidates[split].append((lat, lon))
        self.config, self.settings = config, settings
        self.outputs, self.tiles_path = outputs, tiles_path
        self.sources, self.calibration = sources, calibration
        self.policy, self.candidates = policy, candidates
        self.provenance, self.provenance_id = provenance, provenance_id
        self.control_files, self.source_root = control_files, source_root
        producer_hashes = {
            str(source_root / relative): expected for relative, expected in provenance["source_snapshot"].items()
        }
        producer_hashes[str(source_root.parent / "uv.lock")] = provenance["dependency_lock_sha256"]
        producer_hashes[str(source_root.parent / "pyproject.toml")] = provenance["package_metadata_sha256"]
        self.code_fingerprints = {path: _fingerprint(path) for path in producer_hashes}
        if any(
            file_hash(Path(path)) != expected or _fingerprint(path) != self.code_fingerprints[path]
            for path, expected in producer_hashes.items()
        ):
            raise RuntimeError("generating source changed during authentication; restart required")
        self.tile_size = legacy.TILE_SIZE_PX
        self.num_cpus = num_cpus
        self.resolution = None
        self.context = None
        self.executor = None
        self.workers = 0
        self._closed = False
        self._resolution_states = {}
        _limit_worker_threads()

    def __enter__(self):
        if self._closed:
            raise RuntimeError("generation session is closed")
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close(cancel=exc_type is not None)
        return False

    def _close_pool(self, *, cancel: bool = False) -> None:
        executor, self.executor = self.executor, None
        if executor is not None:
            _stop_executor(executor, cancel=cancel)

    def close(self, *, cancel: bool = False) -> None:
        """Close and join the pool; cancel=True also interrupts running native jobs."""
        self._closed = True
        self._close_pool(cancel=cancel)

    def _validate_inventory(self) -> None:
        # Authentication is reused only in this live process. Still detect new
        # inputs, dependencies and changed identities at each storage boundary.
        paths = sorted(self.tiles_path.glob("*_DSM.tif"))
        if {path.name for path in paths} != {item["name"] for item in self.sources.values()}:
            raise RuntimeError("authenticated terrain inventory changed during run")
        names = {entry.name for entry in self.tiles_path.iterdir()}
        source_names = {
            str(path.relative_to(self.source_root))
            for dirname in ("passage", "passage-eval")
            for path in (self.source_root / dirname).rglob("*.py")
        }
        if source_names != set(self.provenance["source_snapshot"]):
            raise RuntimeError("generating source inventory changed during run; restart required")
        for source in self.sources.values():
            if any(source["name"] + suffix in names for suffix in (".msk", ".aux.xml", ".ovr")):
                raise ValueError("external_raster_dependency: sidecar terrain inputs are prohibited")
            if _fingerprint(source["path"]) != source["fingerprint"]:
                raise RuntimeError("authenticated terrain changed during run")

    def prepare_resolution(self, resolution: int, requested_samples=None) -> dict:
        """Freeze the full roster once and keep its pool for every storage stage."""
        from passage import generate as legacy

        if self._closed:
            raise RuntimeError("generation session is closed")
        if isinstance(resolution, bool) or not isinstance(resolution, int) or resolution <= 1:
            raise ValueError("require positive count and valid resolution")
        res = resolution
        config, settings = self.config, self.settings
        counts = config["num_samples"] if requested_samples is None else requested_samples
        outputs, sources, calibration = self.outputs, self.sources, self.calibration
        policy, candidates = self.policy, self.candidates
        provenance, provenance_id = self.provenance, self.provenance_id
        control_files = self.control_files
        ratios = config.get("splits", {}).get("ratios", {"train": 1.0})
        split_seed = int(config.get("splits", {}).get("seed", 42))
        _validate_frozen_context({"control_files": self.control_files, "code_fingerprints": self.code_fingerprints})
        count = (
            int(counts.get(res, counts.get(str(res), counts.get("default", 0))))
            if isinstance(counts, dict)
            else int(counts)
        )
        if count <= 0 or int(res) <= 1:
            raise ValueError("require positive count and valid resolution")
        existing = self._resolution_states.get(res)
        if existing is not None and existing["count"] != count:
            raise ValueError("requested roster differs from existing resolution; resume refused")
        if self.resolution == res:
            return self.context
        self._close_pool()
        if existing is not None:
            self.resolution, self.context = res, existing["context"]
            self._assignments, self._count = existing["assignments"], count
            self._roster_identity = existing["roster_identity"]
            self._start_pool()
            return self.context
        assignments = legacy._assign_splits(count, ratios, split_seed)
        roster_identity = semantic_hash({"resolution": res, "count": count, "assignments": assignments})
        roster_path = outputs / str(res) / "roster.json"
        if roster_path.exists() and json.loads(roster_path.read_text()).get("identity") != roster_identity:
            raise ValueError("requested roster differs from existing resolution; resume refused")
        _write_json(
            roster_path,
            {
                "identity": roster_identity,
                "resolution": res,
                "requested": count,
                "calibration_sha256": provenance["calibration_sha256"],
                "split_counts": dict(Counter(assignments.values())),
                "split_seed": split_seed,
            },
        )
        context = {
            "config": config,
            "resolution": int(res),
            "output": str(outputs),
            "sources": sources,
            "calibration": calibration,
            "calibration_sha256": provenance["calibration_sha256"],
            "seed": int(settings.get("seed", split_seed)),
            "policy": policy,
            "candidates": candidates,
            "tile_size": self.tile_size,
            "source_snapshot": provenance["source_snapshot"],
            "control_files": control_files,
            "code_fingerprints": self.code_fingerprints,
            "max_attempts": int(settings.get("max_scene_attempts", 100)),
            "max_infrastructure_retries": int(settings.get("max_infrastructure_retries", 2)),
            "max_expansions": settings.get("max_reference_expansions"),
            "provenance_id": provenance_id,
        }
        if context["max_infrastructure_retries"] < 0:
            raise ValueError("infrastructure retry count must be nonnegative")
        if context["max_attempts"] <= 0:
            raise ValueError("scene retry count must be positive")
        self.resolution, self.context = res, context
        self._assignments, self._count = assignments, count
        self._roster_identity = roster_identity
        self._resolution_states[res] = {
            "count": count,
            "assignments": assignments,
            "roster_identity": roster_identity,
            "context": context,
        }
        self._start_pool()
        return context

    def _start_pool(self) -> None:
        from passage import generate as legacy

        self.workers = legacy._resolve_generation_pool_size(self.num_cpus, self.resolution, self._count)
        if self.workers:
            self.executor = ProcessPoolExecutor(
                max_workers=self.workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_worker,
                initargs=(self.context,),
            )

    def ensure_executor(self, task_count: int):
        """Return the resolution's shared pool, including export-only resume.

        The full roster determines the cap; a small checkpoint never shrinks an
        existing pool. Worker processes themselves start lazily on submission.
        """
        if self._closed:
            raise RuntimeError("generation session is closed")
        if self.context is None:
            raise RuntimeError("prepare a resolution before requesting workers")
        if isinstance(task_count, bool) or not isinstance(task_count, int) or task_count < 0:
            raise ValueError("task count must be a nonnegative integer")
        return self.executor

    def generate(self, scene_indices=None, *, include_indices: bool = False, progress_callback=None) -> dict:
        """Generate a selected subset of the frozen roster without restarting workers.

        Large range selections remain compact in the returned summary. The legacy
        wrapper requests explicit indices to retain its existing result contract.
        """
        if self._closed:
            raise RuntimeError("generation session is closed")
        if self.context is None:
            raise RuntimeError("prepare a resolution before generating scenes")
        context = self.context
        _validate_frozen_context(context)
        self._validate_inventory()
        selected = range(self._count) if scene_indices is None else scene_indices
        if isinstance(selected, range):
            if selected.step != 1:
                raise ValueError("scene index range must increase by one")
            if selected and not 0 <= selected.start <= selected[-1] < self._count:
                raise ValueError("scene index outside frozen roster")
        else:
            selected = list(selected)
            if any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < self._count for i in selected):
                raise ValueError("scene index outside frozen roster")
            if any(right <= left for left, right in zip(selected, selected[1:])):
                raise ValueError("scene indices must be unique ascending integers")
        selected_count = len(selected)
        totals = Counter()
        generated_count, aggregate_seconds = 0, 0.0

        def progress():
            if progress_callback is not None:
                progress_callback(sum(totals.values()), selected_count)

        def tasks():
            for index in selected:
                split = self._assignments[index]
                receipt = self.outputs / str(self.resolution) / split / "receipts" / f"{index:010d}.json"
                if receipt.exists():
                    validated = validate_scene_receipt(receipt.parent.parent, index, self.provenance_id)
                    if not _can_retry_infrastructure(validated, context):
                        totals[validated["scene_status"]] += 1
                        progress()
                        continue
                yield index, split, False

        batch_started = time.perf_counter()
        results = None
        try:
            results = (
                _execute_parallel(context, tasks(), self.workers, executor=self.executor)
                if self.executor is not None
                else (generate_successor_scene(context, *task) for task in tasks())
            )
            for _, _, status, duration in tqdm(results, total=selected_count):
                totals[status] += 1
                generated_count += 1
                aggregate_seconds += duration
                progress()
        except BaseException:
            try:
                if results is not None:
                    results.close()
            finally:
                self.close(cancel=True)
            raise
        batch = {
            "dataset_version": DATASET_VERSION,
            "provenance_id": self.provenance_id,
            "resolution": self.resolution,
            "requested_samples": self._count,
            "selected_samples": selected_count,
            "terminal_counts": dict(totals),
            "calibration_sha256": self.provenance["calibration_sha256"],
            "roster_identity": self._roster_identity,
            "timing": {
                "generated_this_run": generated_count,
                "aggregate_scene_seconds": aggregate_seconds,
                "wall_seconds": time.perf_counter() - batch_started,
            },
        }
        if include_indices:
            batch["indices"] = list(selected)
        elif isinstance(selected, range):
            batch["index_range"] = {"start": selected.start, "stop": selected.stop, "step": selected.step}
        _write_json(self.outputs / str(self.resolution) / "summary.json", batch)
        if sum(totals.values()) != selected_count:
            raise RuntimeError("generation terminal denominator differs from requested batch roster")
        return batch


def run_successor_generation(
    outputs_dirpath,
    config_filepath,
    tiles_dirpath,
    calibration_filepath,
    resolution=None,
    num_samples=None,
    num_cpus=None,
    disable_images=False,
    force=False,
    scene_indices=None,
):
    """Compatibility wrapper; orchestration can retain SuccessorGenerationSession."""
    with SuccessorGenerationSession(
        outputs_dirpath, config_filepath, tiles_dirpath, calibration_filepath, num_cpus=num_cpus, force=force
    ) as session:
        resolutions = (
            resolution
            if isinstance(resolution, list)
            else [resolution]
            if resolution
            else session.config["resolutions"]
        )
        batches = []
        for res in resolutions:
            session.prepare_resolution(res, requested_samples=num_samples)
            batches.append(session.generate(scene_indices, include_indices=True))
        return batches[0] if len(batches) == 1 else {"batches": batches, "provenance_id": session.provenance_id}
