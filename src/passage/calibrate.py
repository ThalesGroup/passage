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

"""Compute elevation calibration bound to a complete local DSM inventory."""

import fcntl
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import re
import tempfile

import cv2
from loguru import logger
from matplotlib import colormaps
import numpy as np
import rasterio
from tqdm import tqdm


CALIBRATION_SCHEMA = "passage-calibration-v2"
VALID_DATA_POLICY = {
    "band": 1,
    "raster_mask": True,
    "external_raster_dependencies": "rejected",
    "finite_only": True,
    "exclusive_minimum_elevation": -1000,
    "allowed_exclusions": ["no_valid_data"],
}


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fingerprint(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _receipt_hash(receipt: dict) -> str:
    return _canonical_hash({key: value for key, value in receipt.items() if key != "receipt_sha256"})


def validate_calibration_receipt(receipt: dict) -> tuple[float, float]:
    """Require an intact, complete v2 receipt before using its global range.

    This validates the recorded receipt; callers replaying source data must also
    compare its source inventory checksums with the bytes they actually consume.
    """
    return _validate_complete_receipt(receipt, require_current_policy=True)


def _validate_complete_receipt(receipt: dict, *, require_current_policy: bool) -> tuple[float, float]:
    try:
        if receipt.get("schema") != CALIBRATION_SCHEMA or receipt.get("complete") is not True:
            raise ValueError("Calibration lacks a complete provenance receipt")
        if receipt.get("receipt_sha256") != _receipt_hash(receipt):
            raise ValueError("Calibration receipt checksum mismatch")
        if require_current_policy and receipt.get("valid_data_policy") != VALID_DATA_POLICY:
            raise ValueError("Calibration valid-data policy mismatch")
        if not re.fullmatch("[a-f0-9]{64}", receipt["producer"]["code_sha256"]):
            raise ValueError("Calibration producer identity is invalid")
        inventory = receipt["source_inventory"]
        stats = receipt["tile_stats"]
        excluded = receipt["excluded_tiles"]
        inventory_names = [entry["filename"] for entry in inventory]
        stat_names = [entry["filename"] for entry in stats]
        excluded_names = [entry["filename"] for entry in excluded]
        if not inventory or inventory_names != sorted(set(inventory_names)):
            raise ValueError("Calibration inventory is empty, duplicated or unordered")
        if len(set(stat_names + excluded_names)) != len(stat_names + excluded_names):
            raise ValueError("Calibration outcomes contain duplicate tiles")
        if set(stat_names + excluded_names) != set(inventory_names):
            raise ValueError("Calibration outcomes do not cover the inventory")
        allowed_exclusions = receipt["valid_data_policy"]["allowed_exclusions"]
        if any(entry.get("reason") not in allowed_exclusions for entry in excluded):
            raise ValueError("Calibration includes an undeclared exclusion")
        if (
            receipt["tiles_expected"] != len(inventory)
            or receipt["tiles_processed"] != len(stats)
            or receipt["tiles_excluded"] != len(excluded)
            or receipt["tiles_failed"] != 0
            or receipt["failed_tiles"]
            or receipt["failures"]
            or receipt["calibration_errors"]
            or not stats
        ):
            raise ValueError("Calibration counts or failures contradict completeness")
        if receipt["source_inventory_sha256"] != _canonical_hash(inventory):
            raise ValueError("Calibration inventory checksum mismatch")
        sources = {entry["filename"]: entry for entry in inventory}
        for entry in inventory:
            if (
                not isinstance(entry["size_bytes"], int)
                or entry["size_bytes"] < 0
                or not re.fullmatch("[a-f0-9]{64}", entry["sha256"])
            ):
                raise ValueError("Calibration source identity is invalid")
        for entry in stats:
            if entry.get("source_sha256") != sources[entry["filename"]]["sha256"]:
                raise ValueError("Calibration tile statistics have a different source identity")
            if (
                not np.isfinite([entry["min"], entry["max"]]).all()
                or entry["min"] > entry["max"]
                or entry["min"] <= receipt["valid_data_policy"]["exclusive_minimum_elevation"]
            ):
                raise ValueError("Calibration tile extrema are invalid")
        minimum, maximum = float(receipt["global_min"]), float(receipt["global_max"])
        if minimum != min(entry["min"] for entry in stats) or maximum != max(entry["max"] for entry in stats):
            raise ValueError("Calibration range differs from complete tile extrema")
        return minimum, maximum
    except (KeyError, TypeError, OverflowError) as exc:
        raise ValueError("Malformed calibration provenance receipt") from exc


def _require_self_contained_raster(src, tile_file: Path) -> None:
    dependencies = [str(path) for path in src.files if Path(path).resolve() != tile_file.resolve()]
    if dependencies:
        raise ValueError(f"external_raster_dependency: {dependencies}")


def _external_sidecars(tile_file: Path) -> list[str]:
    return [str(path) for suffix in (".msk", ".aux.xml", ".ovr") if (path := Path(str(tile_file) + suffix)).exists()]


def _process_calibration_tile(job: tuple[Path, tuple[int, int, int, int, int], dict | None]) -> dict:
    tile_file, expected_fingerprint, cached = job
    source = {"filename": tile_file.name, "size_bytes": expected_fingerprint[2], "sha256": None}
    try:
        if _fingerprint(tile_file) != expected_fingerprint:
            raise ValueError("source_changed_during_calibration")
        source["sha256"] = _file_hash(tile_file)
        if sidecars := _external_sidecars(tile_file):
            raise ValueError(f"external_raster_dependency: {sidecars}")
        if cached is not None and cached.get("source_sha256") == source["sha256"]:
            with rasterio.open(tile_file) as src:
                _require_self_contained_raster(src, tile_file)
            minimum, maximum, error = cached["min"], cached["max"], None
        else:
            name, minimum, maximum, error = _process_tile_worker(tile_file)
            if name != tile_file.name:
                raise ValueError("worker_source_identity_mismatch")
        # A same-size write can preserve mtime and land within one ctime tick.
        if _fingerprint(tile_file) != expected_fingerprint or _file_hash(tile_file) != source["sha256"]:
            raise ValueError("source_changed_during_calibration")
        if error is None and (
            minimum is None or maximum is None or not np.isfinite([minimum, maximum]).all() or minimum > maximum
        ):
            raise ValueError("invalid_tile_extrema")
        return {"source": source, "min": minimum, "max": maximum, "error": error}
    except Exception as exc:
        return {"source": source, "min": None, "max": None, "error": str(exc)}


def run_calibrate(
    outputs_dirpath: str,
    tiles_dirpath: str,
    *,
    resume: bool = True,
    overwrite: bool = False,
    num_cpus: int | None = None,
) -> None:
    """Serialize calibration checks, source scans, and publication per output directory.

    A complete receipt is replaced only by explicit ``resume=False,
    overwrite=True``. Every complete version remains available by receipt hash.
    """
    outputs_path = Path(outputs_dirpath)
    outputs_path.mkdir(parents=True, exist_ok=True)
    with (outputs_path / ".calibrate.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            _run_calibrate_locked(
                outputs_dirpath,
                tiles_dirpath,
                resume=resume,
                overwrite=overwrite,
                num_cpus=num_cpus,
            )
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _run_calibrate_locked(
    outputs_dirpath: str,
    tiles_dirpath: str,
    *,
    resume: bool = True,
    overwrite: bool = False,
    num_cpus: int | None = None,
) -> None:
    """Scan every immediate ``*_DSM.tif`` source and atomically save a v2 receipt.

    ``no_valid_data`` is the sole predeclared exclusion. Read errors, changed
    inputs, interrupted processing, or zero usable tiles produce an incomplete
    receipt with null global extrema and raise. Legacy unbound statistics are
    recomputed. Resume checks current source bytes before reusing v2 statistics.
    An existing output is preserved when both resume and overwrite are false.
    """
    if num_cpus is not None and num_cpus < 0:
        raise ValueError("num_cpus must be nonnegative")
    outputs_path = Path(outputs_dirpath)
    calibrate_json = outputs_path / "calibrate.json"
    if calibrate_json.exists() and not resume and not overwrite:
        logger.info("Existing calibrate.json preserved; set overwrite=True to recompute")
        return
    tiles_path = Path(tiles_dirpath).resolve()
    if not tiles_path.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {tiles_path}")
    tile_files = sorted(tiles_path.glob("*_DSM.tif"))
    if not tile_files:
        raise ValueError(f"No DSM tiles found in {tiles_path}")
    fingerprints = {tile.name: _fingerprint(tile) for tile in tile_files}
    outputs_path.mkdir(parents=True, exist_ok=True)
    code_sha256 = _file_hash(Path(__file__))
    cached_stats = {}
    protected_receipt = None
    if resume and calibrate_json.exists():
        try:
            previous = json.loads(calibrate_json.read_text())
            try:
                _validate_complete_receipt(previous, require_current_policy=False)
                protected_receipt = previous
            except ValueError:
                pass
            if (
                previous.get("schema") == CALIBRATION_SCHEMA
                and previous.get("receipt_sha256") == _receipt_hash(previous)
                and previous.get("valid_data_policy") == VALID_DATA_POLICY
                and previous.get("producer", {}).get("code_sha256") == code_sha256
            ):
                cached_stats = {entry["filename"]: entry for entry in previous["tile_stats"]}
            else:
                logger.warning("Existing calibration is unbound or uses another producer; recomputing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logger.warning(f"Cannot resume previous receipt; recomputing: {exc}")
    jobs = [(tile, fingerprints[tile.name], cached_stats.get(tile.name)) for tile in tile_files]
    outcomes = {}
    interrupted = False
    cpus = num_cpus if num_cpus is not None else (os.cpu_count() or 1)
    logger.info(f"Calibrating {len(jobs)} source tiles with {cpus} workers into {outputs_path}")
    try:
        if num_cpus == 0:
            for outcome in tqdm(map(_process_calibration_tile, jobs), total=len(jobs), desc="Processing tiles"):
                outcomes[outcome["source"]["filename"]] = outcome
        else:
            # Spawn avoids inheriting raster handles and locks from threaded callers.
            with multiprocessing.get_context("spawn").Pool(cpus) as pool:
                for outcome in tqdm(
                    pool.imap_unordered(_process_calibration_tile, jobs), total=len(jobs), desc="Processing tiles"
                ):
                    outcomes[outcome["source"]["filename"]] = outcome
    except KeyboardInterrupt:
        interrupted = True
        logger.warning("Calibration interrupted; preserving an incomplete receipt")

    calibration_errors = []
    try:
        current_files = sorted(tiles_path.glob("*_DSM.tif"))
        if any(_external_sidecars(tile) for tile in current_files):
            calibration_errors.append("external_raster_dependency")
        if (
            [tile.name for tile in current_files] != [tile.name for tile in tile_files]
            or any(_fingerprint(tile) != fingerprints[tile.name] for tile in current_files)
            or any(
                (source_hash := outcomes.get(tile.name, {}).get("source", {}).get("sha256")) is not None
                and _file_hash(tile) != source_hash
                for tile in current_files
            )
        ):
            calibration_errors.append("source_inventory_changed_during_calibration")
    except OSError as exc:
        calibration_errors.append(f"source_inventory_verification_failed: {exc}")
    if interrupted:
        calibration_errors.append("interrupted")
    inventory, tile_stats, excluded_tiles, failures = [], [], [], []
    for tile in tile_files:
        outcome = outcomes.get(
            tile.name,
            {
                "source": {"filename": tile.name, "size_bytes": fingerprints[tile.name][2], "sha256": None},
                "min": None,
                "max": None,
                "error": "not_processed",
            },
        )
        inventory.append(outcome["source"])
        if outcome["error"] is None:
            entry = {
                "filename": tile.name,
                "min": outcome["min"],
                "max": outcome["max"],
                "source_sha256": outcome["source"]["sha256"],
            }
            match = re.search(r"([NS])(\d{3})([EW])(\d{3})", tile.name)
            if match:
                ns, lat, ew, lon = match.groups()
                entry.update(lat=int(lat) * (-1 if ns == "S" else 1), lon=int(lon) * (-1 if ew == "W" else 1))
            tile_stats.append(entry)
        elif outcome["error"] == "no_valid_data":
            excluded_tiles.append({"filename": tile.name, "reason": "no_valid_data"})
        else:
            failures.append({"filename": tile.name, "reason": outcome["error"]})
    complete = bool(tile_stats) and not failures and not calibration_errors
    results = {
        "schema": CALIBRATION_SCHEMA,
        "complete": complete,
        "inventory_scope": "all immediate *_DSM.tif files in tiles_dirpath; historical/download coverage not asserted",
        "valid_data_policy": VALID_DATA_POLICY,
        "producer": {
            "code_sha256": code_sha256,
            "numpy_version": np.__version__,
            "rasterio_version": rasterio.__version__,
            "gdal_version": rasterio.__gdal_version__,
        },
        "global_min": min(entry["min"] for entry in tile_stats) if complete else None,
        "global_max": max(entry["max"] for entry in tile_stats) if complete else None,
        "tiles_expected": len(tile_files),
        "tiles_processed": len(tile_stats),
        "tiles_excluded": len(excluded_tiles),
        "tiles_failed": len(failures),
        "failed_tiles": [entry["filename"] for entry in failures],
        "failures": failures,
        "excluded_tiles": excluded_tiles,
        "calibration_errors": calibration_errors,
        "tiles_dirpath": str(tiles_path),
        "source_inventory": inventory,
        "source_inventory_sha256": _canonical_hash(inventory),
        "tile_stats": tile_stats,
    }
    if complete:
        results["ptp_min"] = min(entry["max"] - entry["min"] for entry in tile_stats)
        results["ptp_max"] = max(entry["max"] - entry["min"] for entry in tile_stats)
    results["receipt_sha256"] = _receipt_hash(results)
    if complete:
        validate_calibration_receipt(results)
    # Re-read at publication as well as startup. The lock serializes cooperating
    # calibrators; this check also protects a complete receipt installed externally.
    if calibrate_json.exists():
        try:
            current = json.loads(calibrate_json.read_text())
            _validate_complete_receipt(current, require_current_policy=False)
        except (OSError, ValueError, TypeError):
            current = None
        if current is not None:
            if not (overwrite and not resume):
                protected_receipt = current
            # Preserve its version before an explicitly authorized replacement.
            _write_receipt_atomic(outputs_path / f"calibrate.{current['receipt_sha256']}.json", current)
    if protected_receipt is not None and results["receipt_sha256"] != protected_receipt["receipt_sha256"]:
        pending = outputs_path / f"calibrate.pending.{results['receipt_sha256']}.json"
        _write_receipt_atomic(pending, results)
        raise RuntimeError(
            f"A complete calibration already exists; explicit resume=False, overwrite=True "
            f"is required to replace it. New receipt preserved at {pending}"
        )
    _write_receipt_atomic(calibrate_json, results)
    if not complete:
        raise RuntimeError(
            f"Calibration incomplete: {len(failures)} failed tiles, "
            f"{len(tile_stats)} usable tiles, errors={calibration_errors}; receipt={calibrate_json}"
        )
    _write_maps(outputs_path, results)
    logger.info(
        f"Calibration complete: {len(tile_stats)} usable, {len(excluded_tiles)} excluded; "
        f"range=[{results['global_min']}, {results['global_max']}]"
    )


def _write_receipt_atomic(path: Path, results: dict) -> None:
    if path.exists() and path.read_text() == json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n":
        return
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, suffix=".json.tmp", delete=False) as stream:
            temp_path = Path(stream.name)
            json.dump(results, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        if results.get("complete"):
            temp_path.chmod(0o444)
            version_path = path.parent / f"calibrate.{results['receipt_sha256']}.json"
            if version_path.exists():
                if version_path.read_bytes() != temp_path.read_bytes():
                    raise ValueError("Versioned calibration artifact conflicts with its receipt identity")
            else:
                os.link(temp_path, version_path)
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _write_maps(outputs_path: Path, results: dict) -> None:
    for name, colormap, minimum, maximum in (
        ("max", "terrain", results["global_min"], results["global_max"]),
        ("ptp", "plasma", results["ptp_min"], results["ptp_max"]),
    ):
        image = np.zeros((181, 361, 3), dtype=np.uint8)
        for entry in results["tile_stats"]:
            lat, lon = entry.get("lat"), entry.get("lon")
            if lat is None or lon is None or not -90 <= lat <= 90 or not -180 <= lon <= 180:
                continue
            value = entry["max"] if name == "max" else entry["max"] - entry["min"]
            norm = (value - minimum) / (maximum - minimum) if maximum > minimum else 0.0
            rgba = colormaps[colormap](max(0.0, min(1.0, norm)))
            image[90 - lat, lon + 180] = tuple(int(channel * 255) for channel in rgba[:3][::-1])
        map_path = outputs_path / f"{name}_elevation_map.png"
        if not cv2.imwrite(str(map_path), image):
            logger.warning(f"Could not save calibration visualization: {map_path}")


def _process_tile_worker(tile_file: Path) -> tuple[str, float | None, float | None, str | None]:
    """Exclude GDAL nodata/masks, nonfinite cells, and the historical <= -1000 sentinel band."""
    try:
        with rasterio.open(tile_file) as src:
            _require_self_contained_raster(src, tile_file)
            data = np.ma.asarray(src.read(1, masked=True))
        values = data.compressed()
        valid_data = values[np.isfinite(values) & (values > -1000)]
        if not valid_data.size:
            return tile_file.name, None, None, "no_valid_data"
        return tile_file.name, float(valid_data.min()), float(valid_data.max()), None
    except Exception as exc:
        return tile_file.name, None, None, str(exc)
