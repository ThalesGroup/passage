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

"""Bounded release orchestration keeps identity and deletion gates observable."""

import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image


def workflow():
    assert importlib.util.find_spec("passage.regenerate"), "bounded regeneration workflow missing"
    return importlib.import_module("passage.regenerate")


@pytest.fixture
def disk_space(monkeypatch):
    """Control capacity at the filesystem boundary while keeping admission real."""
    free_bytes = {}

    def disk_usage(path):
        total = 128 * 1024**3
        free = free_bytes.get(Path(path), total)
        return SimpleNamespace(total=total, used=total - free, free=free)

    monkeypatch.setattr(workflow().shutil, "disk_usage", disk_usage)
    return free_bytes


def inputs(tmp_path):
    tiles = tmp_path / "tiles"
    tiles.mkdir()
    (tiles / "N000E000_DSM.tif").write_bytes(b"fixture")
    calibration = tmp_path / "calibrate.json"
    from passage.calibrate import CALIBRATION_SCHEMA, VALID_DATA_POLICY, _canonical_hash, _receipt_hash

    source_hash = hashlib.sha256(b"fixture").hexdigest()
    inventory = [{"filename": "N000E000_DSM.tif", "size_bytes": 7, "sha256": source_hash}]
    receipt = {
        "schema": CALIBRATION_SCHEMA,
        "complete": True,
        "valid_data_policy": VALID_DATA_POLICY,
        "source_inventory": inventory,
        "source_inventory_sha256": _canonical_hash(inventory),
        "global_min": 1,
        "global_max": 10,
        "tile_stats": [
            {"filename": "N000E000_DSM.tif", "lat": 0, "lon": 0, "min": 1, "max": 10, "source_sha256": source_hash}
        ],
        "excluded_tiles": [],
        "tiles_expected": 1,
        "tiles_processed": 1,
        "tiles_excluded": 0,
        "tiles_failed": 0,
        "failed_tiles": [],
        "failures": [],
        "calibration_errors": [],
        "producer": {"code_sha256": "a" * 64},
    }
    receipt["receipt_sha256"] = _receipt_hash(receipt)
    calibration.write_text(json.dumps(receipt))
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    config["generation"]["dataset_version"] = "passage-successor-v1"
    config["resolutions"] = [64]
    config["num_samples"] = {64: 100}
    # Public exports omit the release mask; exercise enabled controls with local input.
    mask_path = tmp_path / "holdout.png"
    mask = Image.new("L", (361, 181))
    mask.putpixel((180, 90), 255)
    mask.save(mask_path)
    config["hold_out"] = {"enabled": True, "mask_path": str(mask_path)}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    return config_path, tiles, calibration


def test_cpu_default_uses_process_affinity(monkeypatch):
    w = workflow()
    monkeypatch.setattr(w.os, "sched_getaffinity", lambda _: set(range(72)))
    assert w.default_num_cpus() == 54


def test_full_plan_counts_and_byte_bound():
    w = workflow()
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    plan = w.build_plan(config)
    assert plan["core_scenes"] == 4_400_000
    assert plan["route_variants"] == 26_400_000
    assert len(plan["configs"]) == 14
    for item in plan["resolutions"]:
        assert sum(item["split_counts"].values()) == item["total_scenes"]
        assert item["batch_scenes"] * item["estimated_raw_scene_bytes"] <= plan["max_raw_bytes"]


def test_generation_stages_are_independent_of_export_shards():
    """Large admitted low-resolution work is not truncated to 4096 scenes."""
    w = workflow()
    config = yaml.safe_load(Path("config/config.yaml").read_text())
    plan = w.build_plan(config, max_raw_bytes=3 * 1024**4, num_cpus=54, export_memory_bytes=32 * 1024**3)
    by_resolution = {item["resolution"]: item for item in plan["resolutions"]}
    assert by_resolution[64]["batch_scenes"] == 1_000_000
    assert by_resolution[128]["batch_scenes"] == 1_000_000
    assert by_resolution[64]["export_batch_scenes"] == 1_000_000
    for item in plan["resolutions"]:
        assert item["export_batch_scenes"] * 64 * 1024 <= plan["export_coordinator_receipt_budget_bytes"]
    assert by_resolution[4096]["batch_scenes"] > 54
    assert plan["scheduling"] == "persistent_workers_per_resolution_staged_generation_then_parallel_export"


def test_measurement_stage_and_resume_boundaries_are_deterministic():
    w = workflow()
    assert list(w.generation_stage_bounds(150, 100)) == [(0, 64), (64, 150)]
    assert list(w.generation_stage_bounds(5, 2)) == [(0, 2), (2, 4), (4, 5)]
    assert list(w.generation_stage_bounds(4, 100)) == [(0, 4)]


def test_dry_run_validates_inputs_without_writes(tmp_path, disk_space):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    final, scratch = tmp_path / "final", tmp_path / "scratch"
    result = w.run_regenerate(
        config_filepath=config,
        tiles_dirpath=tiles,
        calibration_filepath=calibration,
        dataset_dirpath=final,
        scratch_dirpath=scratch,
        dry_run=True,
    )
    assert result["status"] == "planned"
    mask = tmp_path / "holdout.png"
    assert result["identity"]["frozen_controls"] == {
        "hold_out_enabled": True,
        "hold_out_mask_path": str(mask),
        "hold_out_mask_sha256": hashlib.sha256(mask.read_bytes()).hexdigest(),
    }
    assert not final.exists()
    assert not scratch.exists()


def test_dry_run_rejects_missing_enabled_holdout_without_writes(tmp_path, disk_space):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    (tmp_path / "holdout.png").unlink()
    final, scratch = tmp_path / "final", tmp_path / "scratch"
    with pytest.raises(FileNotFoundError, match="holdout[.]png"):
        w.run_regenerate(
            config_filepath=config,
            tiles_dirpath=tiles,
            calibration_filepath=calibration,
            dataset_dirpath=final,
            scratch_dirpath=scratch,
            dry_run=True,
        )
    assert not final.exists()
    assert not scratch.exists()


@pytest.mark.parametrize("low_space_root", ["scratch", "final"])
def test_dry_run_rejects_low_space_before_writes(tmp_path, disk_space, low_space_root):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    parents = {name: tmp_path / f"{name}-volume" for name in ("scratch", "final")}
    for parent in parents.values():
        parent.mkdir()
    final, scratch = parents["final"] / "final", parents["scratch"] / "scratch"
    disk_space[parents[low_space_root]] = 12 * 1024**3
    with pytest.raises(OSError, match=rf"insufficient free space at .*{low_space_root}:"):
        w.run_regenerate(
            config_filepath=config,
            tiles_dirpath=tiles,
            calibration_filepath=calibration,
            dataset_dirpath=final,
            scratch_dirpath=scratch,
            dry_run=True,
        )
    assert not final.exists()
    assert not scratch.exists()


def test_placeholder_calibration_and_overlapping_roots_rejected(tmp_path):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    (tiles / "N001E000_DSM.tif").write_bytes(b"missing calibration")
    with pytest.raises(ValueError, match="calibration.*coverage"):
        w.run_regenerate(
            config_filepath=config,
            tiles_dirpath=tiles,
            calibration_filepath=calibration,
            dataset_dirpath=tmp_path / "final",
            scratch_dirpath=tmp_path / "scratch",
            dry_run=True,
        )
    with pytest.raises(ValueError, match="separate"):
        w.run_regenerate(
            config_filepath=config,
            tiles_dirpath=tiles,
            calibration_filepath=calibration,
            dataset_dirpath=tmp_path / "final",
            scratch_dirpath=tmp_path / "final" / "scratch",
            dry_run=True,
        )


def test_cleanup_requires_verified_export_and_exact_owned_hashes(tmp_path):
    w = workflow()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    raw = scratch / "tensor.npy.zst"
    raw.write_bytes(b"owned")
    outside = tmp_path / "historical.parquet"
    outside.write_bytes(b"preserved")
    record = {"path": str(raw), "sha256": w.hash_file(raw)}
    persisted = tmp_path / "export-manifest.json"
    persisted.write_text("{}")
    verified = {"raw_cleanup_safe": True, "manifest_path": str(persisted), "manifest_sha256": w.hash_file(persisted)}
    with pytest.raises(ValueError, match="verified"):
        w.cleanup_owned_artifacts(scratch, [record], {"raw_cleanup_safe": False})
    with pytest.raises(ValueError, match="outside|owned"):
        w.cleanup_owned_artifacts(scratch, [{"path": str(outside), "sha256": w.hash_file(outside)}], verified)
    raw.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash"):
        w.cleanup_owned_artifacts(scratch, [record], verified)
    assert raw.exists() and outside.exists()


def test_resume_identity_rejects_changed_source(tmp_path):
    w = workflow()
    journal = tmp_path / "journal.json"
    w.open_journal(journal, {"code_sha256": "a"}, {"status": "planned"})
    with pytest.raises(ValueError, match="identity"):
        w.open_journal(journal, {"code_sha256": "b"}, {"status": "planned"})


def test_regenerate_cli_defaults(monkeypatch):
    import passage

    observed = {}
    monkeypatch.setattr(passage, "run_regenerate", lambda **kwargs: observed.update(kwargs), raising=False)
    passage.main(["regenerate", "--dry-run"])
    assert observed["dataset_dirpath"] == "outputs/release/dataset"
    assert observed["scratch_dirpath"] == "outputs/release/scratch"
    assert observed["dry_run"] is True
    assert observed["pilot"] is None


@pytest.mark.parametrize("pilot", [4, 100])
def test_interrupted_cleanup_resume_uses_committed_receipts(tmp_path, monkeypatch, pilot):
    """A crash after commit must resume cleanup without regenerating exported scenes."""
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    cfg = yaml.safe_load(config.read_text())
    cfg["export"]["chunk_size"][64] = 2
    config.write_text(yaml.safe_dump(cfg))
    import passage.export as export
    import passage.successor_generation as generation
    from passage.generate import _assign_splits

    # Import aliases must retain the complete physical producer source roster.
    producer_alias = tmp_path / "imports" / "passage"
    producer_alias.parent.mkdir()
    producer_alias.symlink_to(Path(export.__file__).resolve().parent, target_is_directory=True)
    monkeypatch.setattr(export, "__file__", str(producer_alias / "export.py"))

    calls = []

    def terminal_generation(outputs, config_path, raw_tiles, calibration_path, **kwargs):
        del raw_tiles, calibration_path
        cfg = yaml.safe_load(Path(config_path).read_text())
        assignments = _assign_splits(100, cfg["splits"]["ratios"], cfg["splits"]["seed"])
        calls.append(list(kwargs["scene_indices"]))
        for index in kwargs["scene_indices"]:
            split = assignments[index]
            receipt = {
                "schema_version": generation.SCENE_RECEIPT_VERSION,
                "dataset_version": generation.DATASET_VERSION,
                "index": index,
                "resolution": 64,
                "split": split,
                "scene_status": "rejected",
                "provenance_id": "fixture",
                "sample_seed": 1,
                "calibration_sha256": w.hash_file(Path(calibration)),
                "artifacts": {},
                "attempts": [{"reason": "fixture_terminal_support"}],
                "variants": {
                    f"{solver['name']}:{kind}": {"status": "failed", "cost": None, "length": 0}
                    for solver in cfg["solvers"]
                    for kind in ("free", "obstacles")
                },
            }
            target = Path(outputs) / "64" / split / "receipts" / f"{index:010d}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(receipt))
        return {
            "indices": kwargs["scene_indices"],
            "terminal_counts": {"rejected": len(kwargs["scene_indices"])},
            "provenance_id": "fixture",
        }

    class TerminalSession:
        def __init__(self, outputs, config_path, raw_tiles, calibration_path, **kwargs):
            self.args = outputs, config_path, raw_tiles, calibration_path

        def prepare_resolution(self, resolution):
            assert resolution == 64

        def generate(self, indices, **kwargs):
            result = terminal_generation(*self.args, scene_indices=list(indices))
            if kwargs.get("progress_callback"):
                kwargs["progress_callback"](len(indices), len(indices))
            return result

        def ensure_executor(self, task_count):
            return None

        def close(self, **kwargs):
            pass

    monkeypatch.setattr(generation, "SuccessorGenerationSession", TerminalSession)
    cleanup = w.cleanup_owned_artifacts
    monkeypatch.setattr(w, "cleanup_owned_artifacts", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    final, scratch = tmp_path / "final", tmp_path / "scratch"
    kwargs = dict(
        config_filepath=config,
        tiles_dirpath=tiles,
        calibration_filepath=calibration,
        dataset_dirpath=final,
        scratch_dirpath=scratch,
        pilot=pilot,
        batch_scenes=100,
        min_free_bytes=0,
        num_cpus=0,
    )
    with pytest.raises(KeyboardInterrupt):
        w.run_regenerate(**kwargs)
    journal = json.loads((final / "release-receipts" / "regeneration-journal.json").read_text())
    assert journal["status"] == "interrupted"
    assert next(iter(journal["batches"].values()))["stage"] == "exported"
    committed = list((final / "release-receipts").glob("batch-*.json"))
    assert len(committed) == 1
    manifest = json.loads(committed[0].read_text())
    assert manifest["producer_code_sha256"] == journal["identity"]["producer_code_sha256"]
    assert len(manifest["scenes"]) == min(8, pilot) and all(
        s["receipt"]["scene_status"] == "rejected" for s in manifest["scenes"]
    )
    monkeypatch.setattr(w, "cleanup_owned_artifacts", cleanup)
    # Export/cleanup-only replay must not demand another complete raw stage.
    space_checks = []
    check_space = w._check_space
    monkeypatch.setattr(
        w,
        "_check_space",
        lambda path, required, reserve: (space_checks.append((path, required)), check_space(path, required, reserve))[
            -1
        ],
    )
    result = w.run_regenerate(**kwargs)
    assert result["status"] == "pilot_complete"
    assert [len(indices) for indices in calls] == ([4] if pilot == 4 else [64, 36])
    raw_checks = [required for path, required in space_checks if path == scratch.resolve()]
    assert raw_checks == ([0] if pilot == 4 else [0, 36 * (64 * 64 * 128 + 1024 * 1024)])
    assert not list(scratch.rglob("receipts/*.json"))
    assert sum(result["verification"]["resolutions"]["64"]["scene_status_counts"].values()) == pilot
    w.run_regenerate(**kwargs)
    assert len(calls) == (1 if pilot == 4 else 2)
    ready = final / "release-receipts" / "ready-upload.json"
    ready.write_text(json.dumps({"status": "ready_for_review_and_upload"}))
    committed[0].write_text("corrupt manifest")
    with pytest.raises(ValueError):
        w.run_regenerate(**kwargs)
    assert json.loads(ready.read_text())["status"] == "failed"


def test_large_cleanup_roster_bounds_pinned_descriptors(tmp_path, monkeypatch):
    w = workflow()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    records = []
    for index in range(100):
        target = scratch / str(index)
        target.write_bytes(b"owned")
        records.append({"path": str(target), "sha256": w.hash_file(target)})
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")
    manifest = {
        "raw_cleanup_safe": True,
        "manifest_path": str(manifest_path),
        "manifest_sha256": w.hash_file(manifest_path),
    }
    monkeypatch.setattr(w.resource, "getrlimit", lambda option: (128, 128))
    original_open, original_dup, original_close = w.os.open, w.os.dup, w.os.close
    live, peak = set(), [0]

    def opened(function):
        def track(*args, **kwargs):
            descriptor = function(*args, **kwargs)
            live.add(descriptor)
            peak[0] = max(peak[0], len(live))
            return descriptor

        return track

    def close(descriptor):
        live.discard(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(w.os, "open", opened(original_open))
    monkeypatch.setattr(w.os, "dup", opened(original_dup))
    monkeypatch.setattr(w.os, "close", close)
    assert w.cleanup_owned_artifacts(scratch, records, manifest) == 100
    assert peak[0] <= 20 and not live


def test_legacy_calibration_receipt_is_rejected_before_writes(tmp_path):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    calibration.write_text(
        json.dumps(
            {"global_min": 0, "global_max": 10, "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0, "max": 10}]}
        )
    )
    with pytest.raises(ValueError, match="provenance"):
        w.run_regenerate(
            config_filepath=config,
            tiles_dirpath=tiles,
            calibration_filepath=calibration,
            dataset_dirpath=tmp_path / "final",
            scratch_dirpath=tmp_path / "scratch",
            dry_run=True,
        )


def test_selective_download_cli_passes_explicit_archives(monkeypatch):
    import passage

    observed = {}
    monkeypatch.setattr(passage, "run_download", lambda output, **kwargs: observed.update(output=output, **kwargs))
    passage.main(["download", "--archive", "-5", "25", "--archive", "20", "0"])
    assert observed == {
        "output": "outputs/download",
        "resume": True,
        "num_cpus": None,
        "archive_pairs": [(-5, 25), (20, 0)],
    }


def test_cleanup_refuses_unpersisted_verification_claim(tmp_path):
    w = workflow()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    raw = scratch / "raw"
    raw.write_bytes(b"owned")
    with pytest.raises(ValueError, match="persisted"):
        w.cleanup_owned_artifacts(
            scratch,
            [{"path": str(raw), "sha256": w.hash_file(raw)}],
            {"raw_cleanup_safe": True, "manifest_sha256": "a" * 64},
        )
    assert raw.exists()


def test_measured_storage_reserve_covers_remaining_resolution():
    w = workflow()
    cursor = {"max_observed_core_bytes_per_row": 10_000, "next_offset": 100}
    assert w.remaining_storage_reserve(1000, cursor, 1000) == 11_250_000
    assert w.remaining_storage_reserve(100, cursor, 1000) == 1000


def test_cleanup_anchors_unlink_to_owned_directory_when_ancestor_replaced(tmp_path, monkeypatch):
    w = workflow()
    scratch = tmp_path / "scratch"
    owned = scratch / "owned"
    owned.mkdir(parents=True)
    raw = owned / "tensor"
    raw.write_bytes(b"owned")
    outside = tmp_path / "outside"
    outside.mkdir()
    external = outside / "tensor"
    external.write_bytes(b"precious")
    persisted = tmp_path / "manifest.json"
    persisted.write_text("{}")
    manifest = {"raw_cleanup_safe": True, "manifest_path": str(persisted), "manifest_sha256": w.hash_file(persisted)}
    unlink = w.os.unlink
    swapped = []

    def replace_ancestor(path, *args, **kwargs):
        if Path(path).name == "tensor" and not swapped:
            owned.rename(scratch / "retained-owned")
            owned.symlink_to(outside, target_is_directory=True)
            swapped.append(True)
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(w.os, "unlink", replace_ancestor)
    assert w.cleanup_owned_artifacts(scratch, [{"path": str(raw), "sha256": w.hash_file(raw)}], manifest) == 1
    assert swapped and external.read_bytes() == b"precious"
    assert not (scratch / "retained-owned" / "tensor").exists()


def test_scratch_lock_is_exclusive_across_final_destinations(tmp_path):
    w = workflow()
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    descriptors = w.acquire_scratch_lock(scratch)
    try:
        with pytest.raises(BlockingIOError):
            w.acquire_scratch_lock(scratch)
    finally:
        for descriptor in descriptors:
            w.os.close(descriptor)
    descriptors = w.acquire_scratch_lock(scratch)
    for descriptor in descriptors:
        w.os.close(descriptor)


def test_holdout_mask_content_binds_readonly_plan_identity(tmp_path, disk_space):
    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    mask = tmp_path / "holdout.png"
    mask.write_bytes(b"first-control")
    cfg = yaml.safe_load(config.read_text())
    cfg["hold_out"] = {"enabled": True, "mask_path": str(mask)}
    config.write_text(yaml.safe_dump(cfg))
    kwargs = dict(
        config_filepath=config,
        tiles_dirpath=tiles,
        calibration_filepath=calibration,
        dataset_dirpath=tmp_path / "final",
        scratch_dirpath=tmp_path / "scratch",
        dry_run=True,
    )
    first = w.run_regenerate(**kwargs)
    mask.write_bytes(b"changed-control")
    second = w.run_regenerate(**kwargs)
    assert first["identity"] != second["identity"]


def verified_full_release(tmp_path, reference_status="exact"):
    import hashlib
    from collections import Counter
    import pyarrow as pa
    import pyarrow.parquet as pq

    w = workflow()
    scenes, files = [], []
    for index, split in enumerate(w.SPLITS):
        status = {"exact": "ok", "no_path": "infeasible"}.get(reference_status, "failed")
        variants = {
            f"{objective}:{kind}": {
                "status": status if index == 0 else "ok",
                "reference_status": reference_status if index == 0 else "exact",
            }
            for objective in ("elevation", "slope_uphill", "energy")
            for kind in ("free", "obstacles")
        }
        receipt = {
            "index": index,
            "split": split,
            "resolution": 64,
            "scene_status": "complete",
            "calibration_sha256": "a" * 64,
            "variants": variants,
            "source_windows": [{"lat": index, "lon": 0}],
        }
        scenes.append(
            {
                "index": index,
                "split": split,
                "receipt": receipt,
                "receipt_semantic_sha256": hashlib.sha256(
                    json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            }
        )
        for samples in (False, True):
            config = "64x64_samples" if samples else "64x64"
            path = tmp_path / config / split / "data.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.Table.from_pylist([{"metadata": {"index": index, "resolution": 64, "split": split}}]), path
            )
            files.append(
                {
                    "path": str(path.relative_to(tmp_path)),
                    "rows": 1,
                    "samples": samples,
                    "split": split,
                    "sha256": w.hash_file(path),
                    "size_bytes": path.stat().st_size,
                }
            )
    manifest = {
        "schema_version": "passage-export-batch-v1",
        "batch_id": "64-0000000000-0000000003",
        "resolution": 64,
        "scene_indices": list(range(4)),
        "scenes": scenes,
        "files": files,
        "requested_scenes": 4,
        "requested_variants": 24,
        "calibration_sha256": "a" * 64,
        "sample_ids": {split: [str(index)] for index, split in enumerate(w.SPLITS)},
        "scene_status_counts": dict(Counter(f"{s['split']}:complete" for s in scenes)),
        "variant_status_counts": dict(
            Counter(f"{s['split']}:{v['status']}" for s in scenes for v in s["receipt"]["variants"].values())
        ),
    }
    target = tmp_path / "release-receipts" / "batch-64-0000000000-0000000003.json"
    target.parent.mkdir()
    target.write_text(json.dumps(manifest))
    return {
        "scope": "full",
        "configs": ["64x64", "64x64_samples"],
        "resolutions": [
            {
                "resolution": 64,
                "total_scenes": 4,
                "requested_scenes": 4,
                "split_counts": {s: 1 for s in w.SPLITS},
                "sample_rows_each_split": 1,
            }
        ],
    }


@pytest.mark.parametrize("reference_status", ["censored", "numeric_error", "runtime_error"])
def test_full_data_verification_does_not_admit_failed_labels(tmp_path, reference_status):
    w = workflow()
    plan = verified_full_release(tmp_path, reference_status)
    result = w.verify_release(tmp_path, plan)
    assert result["data_status"] == "verified"
    assert result["admission"]["status"] == "blocked"
    assert result["admission"]["unresolved_variants"] == 6


def test_proved_no_path_is_retained_outside_successful_route_supervision(tmp_path):
    w = workflow()
    result = w.verify_release(tmp_path, verified_full_release(tmp_path, "no_path"))
    assert result["admission"]["status"] == "qualified"
    assert result["resolutions"]["64"]["proved_infeasible_variants"] == 6
    roster = [
        json.loads(line) for line in (tmp_path / "release-receipts" / "supervision-64.jsonl").read_text().splitlines()
    ]
    assert len(roster) == 18 and all(row["index"] != 0 for row in roster)
    assert all(row["calibration_sha256"] == "a" * 64 for row in roster)


def test_final_verification_rejects_untracked_config_tree(tmp_path):
    w = workflow()
    plan = verified_full_release(tmp_path)
    foreign = tmp_path / "128x128" / "train"
    foreign.mkdir(parents=True)
    (foreign / "data.parquet").write_bytes(b"foreign")
    with pytest.raises(ValueError, match="untracked|uncommitted"):
        w.verify_release(tmp_path, plan)


def test_final_verification_rejects_mixed_frozen_calibration(tmp_path):
    w = workflow()
    plan = verified_full_release(tmp_path)
    with pytest.raises(ValueError, match="provenance|calibration"):
        w.verify_release(tmp_path, plan, expected_identity={"calibration_sha256": "b" * 64})


def test_cleanup_roster_includes_archived_infrastructure_attempts(tmp_path):
    w = workflow()
    manifest = {
        "resolution": 64,
        "scenes": [
            {
                "index": 0,
                "split": "train",
                "receipt_file_sha256": "a" * 64,
                "receipt": {
                    "artifacts": {},
                    "attempt_history": [{"receipt": "failed-attempts/0000000000/0000.json", "sha256": "b" * 64}],
                },
            }
        ],
    }
    records = w._owned_records(tmp_path, manifest)
    assert {"path": str(tmp_path / "64/train/failed-attempts/0000000000/0000.json"), "sha256": "b" * 64} in records


def test_exported_failure_history_survives_raw_cleanup_and_tampering_fails(tmp_path):
    import copy
    from passage.export import _build_release_context, export_successor_batch, verify_successor_batch
    from passage.successor_generation import SCENE_RECEIPT_VERSION, _archive_failure

    w = workflow()
    config, tiles, calibration = inputs(tmp_path)
    del tiles
    raw, release = tmp_path / "scratch", tmp_path / "release"
    split_root = raw / "64/train"
    failure = {
        "schema_version": SCENE_RECEIPT_VERSION,
        "dataset_version": w.DATASET_VERSION,
        "index": 0,
        "resolution": 64,
        "split": "train",
        "sample_seed": 1,
        "provenance_id": "fixture",
        "calibration_sha256": w.hash_file(calibration),
        "scene_status": "failed",
        "retryable": True,
        "infrastructure_attempt": 1,
        "failure": {"type": "BrokenProcessPool", "message": "native exit 19"},
        "attempts": [{"reason": "native_worker_lost"}],
        "attempt_history": [],
        "artifacts": {},
        "variants": {
            f"{solver['name']}:{kind}": {"status": "failed", "cost": None, "length": 0}
            for solver in yaml.safe_load(config.read_text())["solvers"]
            for kind in ("free", "obstacles")
        },
    }
    attempt, history = _archive_failure(split_root, 0, failure)
    current = copy.deepcopy(failure)
    current.update(
        scene_status="rejected",
        retryable=False,
        infrastructure_attempt=attempt,
        attempt_history=history,
        attempts=[{"reason": "no_geographic_source_support"}],
    )
    receipt_path = split_root / "receipts/0000000000.json"
    receipt_path.parent.mkdir()
    receipt_path.write_text(json.dumps(current))
    context = _build_release_context(
        raw, config_filepath=config, calibration_filepath=calibration, export_dirpath=release
    )
    exported = export_successor_batch(context, 64, [0])
    historical = release / "64/train" / history[0]["receipt"]
    assert historical.read_bytes() == (split_root / history[0]["receipt"]).read_bytes()
    assert exported["history_files"][0]["source_path"] == history[0]["receipt"]
    exported["manifest_path"] = str(release / "release-receipts" / f"batch-{exported['batch_id']}.json")
    w.cleanup_owned_artifacts(raw, w._owned_records(raw, exported), exported)
    assert not receipt_path.exists() and not (split_root / history[0]["receipt"]).exists()
    verified = verify_successor_batch(release, exported["batch_id"])
    linked = release / "64/train" / verified["scenes"][0]["receipt"]["attempt_history"][0]["receipt"]
    assert json.loads(linked.read_text())["failure"]["message"] == "native exit 19"
    historical.write_text("tampered historical failure")
    with pytest.raises(ValueError, match="history|attempt"):
        verify_successor_batch(release, exported["batch_id"])
