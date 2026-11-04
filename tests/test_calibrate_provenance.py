"""Calibration must bind every source tile and preserve failed coverage."""

import hashlib
import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

import passage.calibrate as cal


def _tile(path, values, *, nodata=None):
    data = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs="EPSG:4326",
        transform=from_origin(0, 1, 1, 1),
        nodata=nodata,
    ) as raster:
        raster.write(data, 1)


def _receipt(path):
    return json.loads((path / "calibrate.json").read_text())


def test_declared_nodata_nan_and_infinity_cannot_set_range(tmp_path):
    tile = tmp_path / "N000E000_DSM.tif"
    _tile(tile, [[-999, 50, np.nan], [100, np.inf, -np.inf]], nodata=-999)
    assert cal._process_tile_worker(tile) == (tile.name, 50.0, 100.0, None)


def test_receipt_binds_sorted_complete_source_inventory(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N001E000_DSM.tif", [[70, 80]])
    _tile(tiles / "N000E000_DSM.tif", [[-10, 20]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["complete"] is True
    assert receipt["global_min"] == -10
    assert receipt["global_max"] == 80
    assert receipt["tiles_expected"] == 2
    assert [item["filename"] for item in receipt["source_inventory"]] == [
        "N000E000_DSM.tif",
        "N001E000_DSM.tif",
    ]
    for entry in receipt["source_inventory"]:
        assert entry["sha256"] == hashlib.sha256((tiles / entry["filename"]).read_bytes()).hexdigest()
    assert cal.validate_calibration_receipt(receipt) == (-10.0, 80.0)


def test_failed_tile_writes_incomplete_receipt_and_raises(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[0, 100]])
    (tiles / "N001E000_DSM.tif").write_bytes(b"corrupt raster")
    out = tmp_path / "calibrate"
    with pytest.raises(RuntimeError, match="incomplete"):
        cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["complete"] is False
    assert receipt["global_min"] is None
    assert receipt["global_max"] is None
    assert receipt["tiles_failed"] == 1
    assert receipt["failed_tiles"] == ["N001E000_DSM.tif"]
    assert len(receipt["source_inventory"]) == 2
    with pytest.raises(ValueError):
        cal.validate_calibration_receipt(receipt)


def test_all_nodata_exclusion_is_explicit_and_hashed(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[0, 100]])
    _tile(tiles / "N001E000_DSM.tif", [[-999, -999]], nodata=-999)
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["complete"] is True
    assert receipt["tiles_processed"] == 1
    assert receipt["tiles_excluded"] == 1
    assert receipt["tiles_failed"] == 0
    assert receipt["excluded_tiles"] == [{"filename": "N001E000_DSM.tif", "reason": "no_valid_data"}]
    assert receipt["valid_data_policy"]["allowed_exclusions"] == ["no_valid_data"]
    assert len(receipt["source_inventory"]) == 2


def test_no_usable_tiles_is_incomplete(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[-9999]], nodata=-9999)
    out = tmp_path / "calibrate"
    with pytest.raises(RuntimeError, match="incomplete"):
        cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    assert _receipt(out)["global_min"] is None


def test_resume_recomputes_changed_source_and_drops_absent_tiles(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    first = tiles / "N000E000_DSM.tif"
    removed = tiles / "N001E000_DSM.tif"
    _tile(first, [[0, 100]])
    _tile(removed, [[-20, 300]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    old_hash = _receipt(out)["source_inventory"][0]["sha256"]
    _tile(first, [[30, 40]])
    removed.unlink()
    cal.run_calibrate(str(out), str(tiles), resume=False, overwrite=True, num_cpus=0)
    receipt = _receipt(out)
    assert receipt["global_min"] == 30
    assert receipt["global_max"] == 40
    assert receipt["tiles_processed"] == 1
    assert receipt["source_inventory"][0]["sha256"] != old_hash


def test_legacy_resume_stats_do_not_override_actual_raster(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[30, 40]])
    out = tmp_path / "calibrate"
    out.mkdir()
    (out / "calibrate.json").write_text(
        json.dumps(
            {
                "global_min": 0,
                "global_max": 1000,
                "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0, "max": 1000}],
            }
        )
    )
    cal.run_calibrate(str(out), str(tiles), resume=True, num_cpus=0)
    assert _receipt(out)["global_min"] == 30
    assert _receipt(out)["global_max"] == 40


def test_serial_parallel_and_resumed_receipts_are_identical(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N001E000_DSM.tif", [[100, 200]])
    _tile(tiles / "N000E000_DSM.tif", [[-50, 0]])
    _tile(tiles / "N002E000_DSM.tif", [[-9999]], nodata=-9999)
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    cal.run_calibrate(str(serial), str(tiles), num_cpus=0)
    cal.run_calibrate(str(parallel), str(tiles), num_cpus=2)
    before = (serial / "calibrate.json").read_bytes()
    assert before == (parallel / "calibrate.json").read_bytes()
    cal.run_calibrate(str(serial), str(tiles), resume=True, num_cpus=2)
    assert before == (serial / "calibrate.json").read_bytes()


def test_receipt_validation_rejects_unbound_or_tampered_range(tmp_path):
    with pytest.raises(ValueError):
        cal.validate_calibration_receipt({"global_min": 0, "global_max": 1000})
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[-50, 0]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    receipt["global_min"] = -100
    with pytest.raises(ValueError):
        cal.validate_calibration_receipt(receipt)


def test_pytest_environment_does_not_redirect_explicit_relative_output(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[20, 30]])
    cal.run_calibrate("outputs/calibrate", str(tiles), num_cpus=0)
    assert _receipt(tmp_path / "outputs/calibrate")["global_min"] == 20


@pytest.mark.parametrize("missing", [False, True])
def test_missing_or_empty_inventory_fails_closed(tmp_path, missing):
    tiles = tmp_path / "tiles"
    if not missing:
        tiles.mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        cal.run_calibrate(str(tmp_path / "calibrate"), str(tiles), num_cpus=0)


def test_complete_calibration_is_readonly_and_requires_explicit_recompute(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    tile = tiles / "N000E000_DSM.tif"
    _tile(tile, [[10, 20]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    original = (out / "calibrate.json").read_bytes()
    old_receipt = _receipt(out)
    assert (out / "calibrate.json").stat().st_mode & 0o222 == 0
    assert (out / f"calibrate.{old_receipt['receipt_sha256']}.json").read_bytes() == original
    _tile(tile, [[30, 40]])
    with pytest.raises(RuntimeError, match="explicit"):
        cal.run_calibrate(str(out), str(tiles), resume=True, num_cpus=0)
    assert (out / "calibrate.json").read_bytes() == original
    cal.run_calibrate(str(out), str(tiles), resume=False, overwrite=True, num_cpus=0)
    assert _receipt(out)["global_min"] == 30
    assert (out / f"calibrate.{old_receipt['receipt_sha256']}.json").read_bytes() == original


def test_concurrent_scan_cannot_replace_complete_inventory(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import threading

    full = tmp_path / "full"
    reduced = tmp_path / "reduced"
    full.mkdir()
    reduced.mkdir()
    _tile(full / "N000E000_DSM.tif", [[0, 10]])
    _tile(full / "N001E000_DSM.tif", [[20, 30]])
    _tile(reduced / "N000E000_DSM.tif", [[0, 10]])
    out = tmp_path / "calibrate"
    scan_entered = threading.Event()
    release_scan = threading.Event()
    competing_lock_requested = threading.Event()
    original_worker = cal._process_tile_worker
    original_flock = cal.fcntl.flock

    def controlled_worker(path):
        if path.parent == full and path.name == "N000E000_DSM.tif":
            scan_entered.set()
            assert release_scan.wait(5)
        return original_worker(path)

    def observed_flock(fd, operation):
        if threading.current_thread().name.endswith("_1") and operation == cal.fcntl.LOCK_EX:
            competing_lock_requested.set()
        return original_flock(fd, operation)

    monkeypatch.setattr(cal, "_process_tile_worker", controlled_worker)
    monkeypatch.setattr(cal.fcntl, "flock", observed_flock)
    with ThreadPoolExecutor(max_workers=2) as executor:
        complete_run = executor.submit(cal.run_calibrate, str(out), str(full), num_cpus=0)
        assert scan_entered.wait(5)
        competing_run = executor.submit(cal.run_calibrate, str(out), str(reduced), num_cpus=0)
        try:
            assert competing_lock_requested.wait(5)
            assert not competing_run.done()
        finally:
            release_scan.set()
        complete_run.result(timeout=5)
        with pytest.raises(RuntimeError, match="explicit"):
            competing_run.result(timeout=5)
    receipt = _receipt(out)
    assert receipt["tiles_expected"] == 2
    assert receipt["global_max"] == 30
    assert receipt["complete"] is True


def test_external_mask_cannot_supply_unhashed_calibration_values(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    tile = tiles / "N000E000_DSM.tif"
    _tile(tile, [[10, 20, 900]])
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(tile, "r+") as raster:
            raster.write_mask(np.array([[255, 255, 0]], dtype=np.uint8))
    out = tmp_path / "calibrate"
    with pytest.raises(RuntimeError, match="incomplete"):
        cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["complete"] is False
    assert "external_raster_dependency" in receipt["failures"][0]["reason"]


def test_resume_rejects_new_external_mask_and_preserves_complete_artifact(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    tile = tiles / "N000E000_DSM.tif"
    _tile(tile, [[10, 20, 900]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    original = (out / "calibrate.json").read_bytes()
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=False):
        with rasterio.open(tile, "r+") as raster:
            raster.write_mask(np.array([[255, 255, 0]], dtype=np.uint8))
    with pytest.raises(RuntimeError, match="explicit"):
        cal.run_calibrate(str(out), str(tiles), resume=True, num_cpus=0)
    assert (out / "calibrate.json").read_bytes() == original
    pending = list(out.glob("calibrate.pending.*.json"))
    assert len(pending) == 1
    assert json.loads(pending[0].read_text())["complete"] is False


def test_new_policy_recomputes_sources_without_replacing_prior_complete_receipt(tmp_path, monkeypatch):
    full = tmp_path / "full"
    full.mkdir()
    _tile(full / "N000E000_DSM.tif", [[-50, 0]])
    _tile(full / "N001E000_DSM.tif", [[100, 200]])
    reduced = tmp_path / "reduced"
    reduced.mkdir()
    _tile(reduced / "N000E000_DSM.tif", [[30, 40]])
    out = tmp_path / "calibrate"
    previous_policy = {
        key: value for key, value in cal.VALID_DATA_POLICY.items() if key != "external_raster_dependencies"
    }
    with monkeypatch.context() as previous:
        previous.setattr(cal, "VALID_DATA_POLICY", previous_policy)
        cal.run_calibrate(str(out), str(full), num_cpus=0)
    original = (out / "calibrate.json").read_bytes()
    with pytest.raises(ValueError, match="policy"):
        cal.validate_calibration_receipt(_receipt(out))
    with pytest.raises(RuntimeError, match="explicit"):
        cal.run_calibrate(str(out), str(reduced), resume=True, num_cpus=0)
    assert (out / "calibrate.json").read_bytes() == original


@pytest.mark.parametrize("contradiction", ["failures", "minimum", "producer"])
def test_complete_receipt_rejects_self_consistent_digest_with_contradictory_fields(tmp_path, contradiction):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    _tile(tiles / "N000E000_DSM.tif", [[10, 20]])
    out = tmp_path / "calibrate"
    cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    if contradiction == "failures":
        receipt["failures"] = [{"filename": "N000E000_DSM.tif", "reason": "read_error"}]
    elif contradiction == "minimum":
        receipt["global_min"] = -9999
        receipt["tile_stats"][0]["min"] = -9999
    else:
        del receipt["producer"]
    content = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        cal.validate_calibration_receipt(receipt)


def test_timestamp_preserving_source_write_cannot_publish_stale_hash(tmp_path, monkeypatch):
    import os

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    tile = tiles / "N000E000_DSM.tif"
    _tile(tile, [[10, 20]])
    before = tile.stat()
    original_worker = cal._process_tile_worker

    def mutate_after_hash(path):
        with rasterio.open(path, "r+") as raster:
            raster.write(np.array([[30, 40]], dtype=np.float32), 1)
        os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
        return original_worker(path)

    monkeypatch.setattr(cal, "_process_tile_worker", mutate_after_hash)
    out = tmp_path / "calibrate"
    with pytest.raises(RuntimeError, match="incomplete"):
        cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["global_min"] is None
    assert receipt["complete"] is False
    assert "source_changed_during_calibration" in receipt["failures"][0]["reason"]


@pytest.mark.parametrize("mutation_phase", ["during_worker", "after_worker"])
def test_source_mutation_with_unchanged_metadata_fails_closed(tmp_path, monkeypatch, mutation_phase):
    import os
    from pathlib import Path

    tiles = tmp_path / "tiles"
    tiles.mkdir()
    first = tiles / "N000E000_DSM.tif"
    second = tiles / "N001E000_DSM.tif"
    _tile(first, [[10, 20]])
    _tile(second, [[50, 60]])
    before = first.stat()
    original_hash = hashlib.sha256(first.read_bytes()).hexdigest()
    original_stat = Path.stat
    original_worker = cal._process_tile_worker

    def coarse_stat(path, *args, **kwargs):
        stat = original_stat(path, *args, **kwargs)
        if path != first:
            return stat
        fields = {name: getattr(stat, name) for name in dir(stat) if name.startswith("st_")}
        return os.stat_result(stat, fields | {"st_ctime": before.st_ctime, "st_ctime_ns": before.st_ctime_ns})

    def mutate_source(path):
        trigger = first if mutation_phase == "during_worker" else second
        if path == trigger:
            with rasterio.open(first, "r+") as raster:
                raster.write(np.array([[30, 40]], dtype=np.float32), 1)
            os.utime(first, ns=(before.st_atime_ns, before.st_mtime_ns))
            after = first.stat()
            assert (after.st_size, after.st_mtime_ns, after.st_ctime_ns) == (
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            assert hashlib.sha256(first.read_bytes()).hexdigest() != original_hash
        return original_worker(path)

    # Model a filesystem whose ctime tick does not advance during the write.
    monkeypatch.setattr(Path, "stat", coarse_stat)
    monkeypatch.setattr(cal, "_process_tile_worker", mutate_source)
    out = tmp_path / "calibrate"
    with pytest.raises(RuntimeError, match="incomplete"):
        cal.run_calibrate(str(out), str(tiles), num_cpus=0)
    receipt = _receipt(out)
    assert receipt["complete"] is False
    assert receipt["global_min"] is None
    assert receipt["global_max"] is None
    assert receipt["source_inventory"][0]["sha256"] == original_hash
    if mutation_phase == "during_worker":
        assert receipt["failed_tiles"] == [first.name]
        assert "source_changed_during_calibration" in receipt["failures"][0]["reason"]
    else:
        assert "source_inventory_changed_during_calibration" in receipt["calibration_errors"]
