"""Release export rejects incomplete routes and bounds large scene batches."""

import numpy as np
import pytest

from passage.export import _process_meta_for_prepare, parse_path_csv
from passage.utils import save_tensor


def _scene(tmp_path):
    split = tmp_path / "64" / "train"
    metadata = split / "metadata/N000/E000/0000000000.yaml"
    metadata.parent.mkdir(parents=True)
    metadata.write_text("index: 0\nresolution: 64\nsplit: train\n")
    tensor = split / "tensors/N000/E000/0000000000.npy.zst"
    tensor.parent.mkdir(parents=True)
    save_tensor(np.zeros((64, 64, 3), dtype=np.float32), tensor)
    return split, metadata


def test_missing_csv_is_never_an_infeasible_route(tmp_path):
    _, metadata = _scene(tmp_path)
    with pytest.raises(FileNotFoundError, match="path"):
        _process_meta_for_prepare(64, False, ["astar_elevation"], str(metadata))


@pytest.mark.parametrize("content", ["0\n", "0,0,7\n", "\n", "1.5,2\n"])
def test_malformed_csv_is_rejected_instead_of_dropping_vertices(tmp_path, content):
    path = tmp_path / "route.csv"
    path.write_text(content)
    with pytest.raises(ValueError):
        parse_path_csv(path)


def test_export_rejects_declared_identity_mismatch(tmp_path):
    _, metadata = _scene(tmp_path)
    metadata.write_text("index: 99\nresolution: 128\nsplit: test\n")
    with pytest.raises(ValueError, match="identity"):
        _process_meta_for_prepare(64, False, [], str(metadata))


def test_large_scene_row_budget_is_bounded():
    from passage.export import _export_row_budget

    assert _export_row_budget(4096, 10000) == 1
    assert 1 <= _export_row_budget(2048, 10000) <= 4
    assert _export_row_budget(64, 3) == 3


def _canonical_row(tmp_path):
    from passage_eval.protocol.contracts import GraphSpec
    from passage_eval.metrics.cost import path_cost

    tensor = np.zeros((64, 64, 3), dtype=np.float32)
    tensor[..., 1] = -1
    tensor[0, 0, 1] = 0
    tensor[0, 1, 1] = 1
    path = tmp_path / "canonical.npy.zst"
    save_tensor(tensor, path)
    graph = GraphSpec(cost_model="energy", cost_weight=2)
    route = [[0, 0], [0, 1]]
    cost = path_cost(tensor[..., 0], route, weight=2, cost_model="energy")
    stats = {"status": "ok", "length": 2, "cost": cost}
    return {
        "id": "0000000000",
        "tensor": path.read_bytes(),
        "metadata": {
            "resolution": 64,
            "split": "train",
            "calibration": {"min": 0, "max": 100},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 0, "j": 1}},
            "solvers": {"astar_energy": {"cost_model": "energy", "weight": 2, "graph_spec": graph.to_dict()}},
            "paths": {"astar_energy": {"free": dict(stats), "obstacles": dict(stats)}},
        },
        "path_astar_energy_free": route,
        "path_astar_energy_obstacles": route,
    }


def test_canonical_export_rejects_false_empty_success_and_wrong_cost(tmp_path):
    from passage.export import _validate_canonical_export_row

    row = _canonical_row(tmp_path)
    _validate_canonical_export_row(row, 64, "train", ["astar_energy"])
    row["metadata"]["paths"]["astar_energy"]["free"]["cost"] = 0.03
    with pytest.raises(ValueError, match="canonical route cost"):
        _validate_canonical_export_row(row, 64, "train", ["astar_energy"])
    row["metadata"]["paths"]["astar_energy"]["free"].update(length=0, cost=0)
    row["path_astar_energy_free"] = []
    with pytest.raises(ValueError, match="explicit valid route status"):
        _validate_canonical_export_row(row, 64, "train", ["astar_energy"])


def test_terminal_variant_remains_explicit_with_null_cost(tmp_path):
    from passage.export import _validate_canonical_export_row

    row = _canonical_row(tmp_path)
    row["metadata"]["paths"]["astar_energy"]["obstacles"].update(status="failed", length=0, cost=None)
    row["path_astar_energy_obstacles"] = []
    _validate_canonical_export_row(row, 64, "train", ["astar_energy"])


def test_batch_export_rejects_empty_index_roster():
    from passage.export import export_successor_batch

    with pytest.raises(ValueError, match="indices"):
        export_successor_batch(None, 64, [])


def _successor_scene(root, index, *, terminal=False, split_name="train"):
    import hashlib
    import json
    import yaml
    from passage.successor_generation import DATASET_VERSION, SCENE_RECEIPT_VERSION
    from passage_eval.protocol.contracts import GraphSpec
    from passage.generate import _derive_sample_seed

    sample_seed = _derive_sample_seed(42, 64, index, split_name)
    source_windows = [{"tile": "N000E000", "sha256": "1" * 64, "row": 0, "column": 0, "height": 64, "width": 64}]
    calibration_sha = hashlib.sha256(b"{}").hexdigest()
    provenance = {
        "generation_seed": 42,
        "sample_seed": sample_seed,
        "source_windows": source_windows,
        "provenance_id": "frozen-test-provenance",
        "calibration_sha256": calibration_sha,
    }
    split = root / "64" / split_name
    variants, artifacts, solvers, stats = {}, {}, {}, {}
    for model in ("elevation", "slope_uphill", "energy"):
        name = f"astar_{model}"
        graph = GraphSpec(cost_model=model, cost_weight=1)
        solvers[name] = {"name": name, "weight": 1, "cost_model": model, "graph_spec": graph.to_dict()}
        stats[name] = {}
        for variant in ("free", "obstacles"):
            status = "failed" if terminal else "ok"
            variants[f"{name}:{variant}"] = {
                "status": status,
                "cost": None if terminal else 30,
                "length": 0 if terminal else 2,
                "graph_id": graph.identity(),
            }
            stats[name][variant] = dict(variants[f"{name}:{variant}"])
            if not terminal:
                path = split / f"paths/{name}/{variant}/N000/E000/{index:010d}.csv"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("0,0\n0,1\n")
                artifacts[f"path:{name}:{variant}"] = {
                    "path": path.relative_to(split).as_posix(),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
    if not terminal:
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        tensor[..., 1] = -1
        tensor[0, 0, 1] = 0
        tensor[0, 1, 1] = 1
        tensor_path = split / f"tensors/N000/E000/{index:010d}.npy.zst"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        save_tensor(tensor, tensor_path)
        metadata_path = split / f"metadata/N000/E000/{index:010d}.yaml"
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(
            yaml.safe_dump(
                {
                    "index": index,
                    "resolution": 64,
                    "split": split_name,
                    "dataset_version": DATASET_VERSION,
                    "units": "metres",
                    "solvers": solvers,
                    "paths": stats,
                    "calibration": {"min": 0, "max": 100},
                    "provenance": provenance,
                    "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 0, "j": 1}},
                }
            )
        )
        for key, path in (("tensor", tensor_path), ("metadata", metadata_path)):
            artifacts[key] = {
                "path": path.relative_to(split).as_posix(),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    receipt = {
        "schema_version": SCENE_RECEIPT_VERSION,
        "dataset_version": DATASET_VERSION,
        "index": index,
        "resolution": 64,
        "split": split_name,
        "provenance_id": "frozen-test-provenance",
        "sample_seed": sample_seed,
        "source_windows": source_windows,
        "calibration_sha256": calibration_sha,
        "attempts": [{"reason": "accepted_input", "attempt": 0}],
        "scene_status": "rejected" if terminal else "complete",
        "variants": variants,
        "artifacts": artifacts,
    }
    path = split / f"receipts/{index:010d}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt))


def test_bounded_batch_export_retains_terminal_denominator_and_resumes_without_raw(tmp_path):
    import json
    import shutil
    import pyarrow.parquet as pq
    from passage.export import _build_release_context, export_successor_batch, verify_successor_batch

    raw, release = tmp_path / "raw", tmp_path / "release"
    _successor_scene(raw, 0)
    _successor_scene(raw, 1, terminal=True)
    config = tmp_path / "config.yaml"
    config.write_text(
        "export:\n  chunk_size: {64: 10}\n  sample: {num_samples: 1}\nsolvers:\n"
        " - {name: astar_elevation}\n - {name: astar_slope_uphill}\n - {name: astar_energy}\n"
    )
    calibration = tmp_path / "calibrate.json"
    calibration.write_text("{}")
    context = _build_release_context(
        raw, calibration_filepath=calibration, config_filepath=config, export_dirpath=release
    )
    first = export_successor_batch(context, 64, [0, 1])
    assert first["raw_cleanup_safe"] is True
    assert first["scene_status_counts"] == {"train:complete": 1, "train:rejected": 1}
    assert first["requested_variants"] == 12
    assert len(first["scenes"]) == 2
    core = next(record for record in first["files"] if not record["samples"])
    row = pq.read_table(release / core["path"]).to_pylist()[0]
    assert not any(name.startswith("image_") for name in row)
    assert json.loads(row["metadata_json"])["units"] == "metres"
    assert row["metadata"]["paths"]["astar_energy"]["free"]["status"] == "ok"
    _successor_scene(raw, 2)
    second = export_successor_batch(context, 64, [2])
    assert not any(record["samples"] for record in second["files"])
    shutil.rmtree(raw)
    resumed = export_successor_batch(context, 64, [0, 1])
    assert resumed["manifest_sha256"] == first["manifest_sha256"]
    (release / core["path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="content hash"):
        verify_successor_batch(release, first["batch_id"])


def test_batch_verification_rejects_tampered_scene_denominator(tmp_path):
    import json
    from passage.export import verify_successor_batch

    receipt_dir = tmp_path / "release-receipts"
    receipt_dir.mkdir()
    manifest = {
        "schema_version": "passage-export-batch-v1",
        "batch_id": "64-0000000000-0000000000",
        "scene_indices": [0],
        "scenes": [{"index": 99, "split": "train", "receipt": {"index": 99}, "receipt_semantic_sha256": "0" * 64}],
        "files": [],
    }
    (receipt_dir / "batch-64-0000000000-0000000000.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="identity"):
        verify_successor_batch(tmp_path, manifest["batch_id"])


@pytest.mark.parametrize("num_cpus", [0, 2])
def test_resumed_scene_history_survives_raw_removal_and_is_verified(tmp_path, num_cpus):
    import hashlib
    import json
    import shutil
    import pyarrow.parquet as pq
    from passage.export import _build_release_context, export_successor_batch, verify_successor_batch

    raw, release = tmp_path / "raw", tmp_path / "release"
    _successor_scene(raw, 0, terminal=True)
    receipt_path = raw / "64/train/receipts/0000000000.json"
    failed = json.loads(receipt_path.read_text())
    failed.update(scene_status="failed", retryable=True, infrastructure_attempt=1, attempt_history=[])
    history = raw / "64/train/failed-attempts/0000000000/0001.json"
    history.parent.mkdir(parents=True)
    history.write_text(json.dumps(failed))
    history_sha = hashlib.sha256(history.read_bytes()).hexdigest()
    _successor_scene(raw, 0)
    current = json.loads(receipt_path.read_text())
    current["attempt_history"] = [
        {"attempt": 1, "receipt": "failed-attempts/0000000000/0001.json", "sha256": history_sha}
    ]
    receipt_path.write_text(json.dumps(current))
    config = tmp_path / "config.yaml"
    config.write_text(
        "export:\n  chunk_size: {64: 10}\n  sample: {num_samples: 1}\nsolvers:\n"
        " - {name: astar_elevation}\n - {name: astar_slope_uphill}\n - {name: astar_energy}\n"
    )
    calibration = tmp_path / "calibrate.json"
    calibration.write_text("{}")
    context = _build_release_context(
        raw, calibration_filepath=calibration, config_filepath=config, export_dirpath=release
    )
    manifest = export_successor_batch(context, 64, [0], num_cpus=num_cpus)
    record = manifest["history_files"][0]
    assert record["path"] == "64/train/failed-attempts/0000000000/0001.json"
    assert record["source_path"] == current["attempt_history"][0]["receipt"]
    core = next(item for item in manifest["files"] if not item["samples"])
    exported = json.loads(pq.read_table(release / core["path"]).to_pylist()[0]["scene_receipt"])
    assert exported == current
    shutil.rmtree(raw)
    assert verify_successor_batch(release, manifest["batch_id"])["raw_cleanup_safe"]
    assert export_successor_batch(context, 64, [0])["manifest_sha256"] == manifest["manifest_sha256"]
    (release / record["path"]).write_text("{}")
    with pytest.raises(ValueError, match="history.*hash"):
        verify_successor_batch(release, manifest["batch_id"])


def test_arrow_export_preserves_calibration_and_objective_precision(tmp_path):
    from datasets import Dataset
    from passage.export import _features

    row = _canonical_row(tmp_path)
    row["metadata"]["calibration"] = {"min": 0.123456789012, "max": 100.987654321}
    row["metadata"]["solvers"]["astar_energy"]["weight"] = 62.4
    row.update(metadata_json=None, scene_receipt=None)
    decoded = Dataset.from_list([row], features=_features(False, ["astar_energy"]))[0]
    assert decoded["metadata"]["calibration"] == row["metadata"]["calibration"]
    assert decoded["metadata"]["solvers"]["astar_energy"]["weight"] == 62.4


def test_failed_compression_preserves_previously_committed_shards(tmp_path):
    from passage.export import compress_release

    raw = tmp_path / "raw"
    _scene(raw)
    config = tmp_path / "config.yaml"
    config.write_text(
        "export:\n  chunk_size: {64: 10}\n  sample: {num_samples: 1}\nsolvers:\n - {name: astar_elevation}\n"
    )
    existing = tmp_path / "release/64x64/train/data-00000.parquet"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"reviewed previously committed shard")
    assert (
        compress_release(
            dataset_dirpath=raw,
            config_filepath=config,
            export_dirpath=tmp_path / "release",
            resolutions=[64],
            num_cpus=0,
        )
        is False
    )
    assert existing.read_bytes() == b"reviewed previously committed shard"


def test_existing_empty_csv_requires_explicit_terminal_status(tmp_path):
    split, metadata = _scene(tmp_path)
    for variant in ("free", "obstacles"):
        path = split / f"paths/astar_elevation/{variant}/N000/E000/0000000000.csv"
        path.parent.mkdir(parents=True)
        path.write_text("")
    with pytest.raises(ValueError, match="terminal status"):
        _process_meta_for_prepare(64, False, ["astar_elevation"], str(metadata))


def test_batch_verification_rejects_wrong_status_denominator(tmp_path):
    import hashlib
    import json
    from passage.export import verify_successor_batch

    receipt = {
        "index": 0,
        "resolution": 64,
        "split": "train",
        "scene_status": "rejected",
        "variants": {
            f"astar_{model}:{variant}": {"status": "failed"}
            for model in ("elevation", "slope_uphill", "energy")
            for variant in ("free", "obstacles")
        },
    }
    semantic_hash = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    manifest = {
        "schema_version": "passage-export-batch-v1",
        "batch_id": "64-0000000000-0000000000",
        "resolution": 64,
        "scene_indices": [0],
        "requested_scenes": 2,
        "requested_variants": 12,
        "scene_status_counts": {"train:rejected": 2},
        "variant_status_counts": {"train:failed": 12},
        "scenes": [{"index": 0, "split": "train", "receipt": receipt, "receipt_semantic_sha256": semantic_hash}],
        "sample_ids": {"train": []},
        "files": [],
    }
    directory = tmp_path / "release-receipts"
    directory.mkdir()
    (directory / f"batch-{manifest['batch_id']}.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="denominator"):
        verify_successor_batch(tmp_path, manifest["batch_id"])


def _export_context(tmp_path, raw, release, *, chunk_size=2, sample_size=1):
    from passage.export import _build_release_context

    config = tmp_path / "config.yaml"
    config.write_text(
        f"export:\n  chunk_size: {{64: {chunk_size}}}\n"
        f"  sample: {{num_samples: {sample_size}}}\nsolvers:\n"
        " - {name: astar_elevation}\n - {name: astar_slope_uphill}\n - {name: astar_energy}\n"
    )
    calibration = tmp_path / "calibrate.json"
    calibration.write_text("{}")
    return _build_release_context(raw, calibration_filepath=calibration, config_filepath=config, export_dirpath=release)


def test_parallel_export_matches_serial_rows_and_reuses_external_executor(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    import json
    import multiprocessing as mp
    import pyarrow.parquet as pq
    from passage.export import export_successor_batch

    raw = tmp_path / "raw"
    for index in range(4):
        _successor_scene(raw, index, terminal=index == 3)
    for index, split in enumerate(("calibration", "validation", "test"), start=4):
        _successor_scene(raw, index, split_name=split)
    serial = _export_context(tmp_path, raw, tmp_path / "serial")
    parallel = _export_context(tmp_path, raw, tmp_path / "parallel")
    first = export_successor_batch(serial, 64, list(range(7)))
    with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
        second = export_successor_batch(parallel, 64, list(range(7)), executor=executor, num_cpus=2)
        assert executor.submit(abs, -3).result() == 3
        assert (
            export_successor_batch(parallel, 64, list(range(7)), executor=executor)["manifest_sha256"]
            == second["manifest_sha256"]
        )
    assert first["scene_status_counts"] == second["scene_status_counts"]
    assert first["variant_status_counts"] == second["variant_status_counts"]
    assert first["sample_ids_total"] == second["sample_ids_total"]
    assert len([f for f in second["files"] if not f["samples"]]) == 5
    assert sorted(f["rows"] for f in second["files"] if not f["samples"]) == [1, 1, 1, 1, 2]
    assert len([f for f in second["files"] if f["samples"]]) == 4
    assert [f["path"] for f in first["files"]] == [f["path"] for f in second["files"]]
    for a, b in zip(first["files"], second["files"]):
        assert (serial.export_dirpath / a["path"]).read_bytes() == (parallel.export_dirpath / b["path"]).read_bytes()
        assert (
            pq.read_table(serial.export_dirpath / a["path"]).to_pylist()
            == pq.read_table(parallel.export_dirpath / b["path"]).to_pylist()
        )
    for scene in second["scenes"]:
        assert scene["receipt_semantic_sha256"]
        assert (
            json.loads((raw / "64" / scene["split"] / "receipts" / f"{scene['index']:010d}.json").read_text())
            == scene["receipt"]
        )


@pytest.mark.parametrize("damage", ["missing_csv", "corrupt_tensor"])
def test_parallel_export_rejects_corrupt_raw_without_commit_and_retries(tmp_path, damage):
    from passage.export import export_successor_batch

    raw = tmp_path / "raw"
    _successor_scene(raw, 0)
    context = _export_context(tmp_path, raw, tmp_path / "release")
    if damage == "missing_csv":
        path = raw / "64/train/paths/astar_energy/free/N000/E000/0000000000.csv"
        path.unlink()
    else:
        path = raw / "64/train/tensors/N000/E000/0000000000.npy.zst"
        path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="corrupt artifact"):
        export_successor_batch(context, 64, [0], num_cpus=2)
    assert not list((context.export_dirpath / "release-receipts").glob("batch-*.json"))
    assert not list(context.export_dirpath.glob(".batch-*"))
    _successor_scene(raw, 0)
    assert export_successor_batch(context, 64, [0], num_cpus=2)["raw_cleanup_safe"]


def test_safe_metadata_loader_preserves_types_and_rejects_python_objects():
    import yaml
    from passage.export import _load_export_metadata

    document = "int: 7\nfloat: 0.123456789012\nboolean: true\nnull: null\ndate: 2026-10-11\nlist: [1, two]\n"
    assert _load_export_metadata(document) == yaml.safe_load(document)
    with pytest.raises(yaml.constructor.ConstructorError):
        _load_export_metadata("!!python/object/apply:os.system [echo unsafe]")


def test_safe_metadata_loader_falls_back_without_libyaml(monkeypatch):
    import yaml
    from passage.export import _load_export_metadata

    monkeypatch.delattr(yaml, "CSafeLoader", raising=False)
    assert _load_export_metadata("float: 0.123456789012\nnull: null\n") == yaml.safe_load(
        "float: 0.123456789012\nnull: null\n"
    )


@pytest.mark.parametrize("artifact", ["metadata", "tensor", "path", "receipt"])
def test_shard_loading_rejects_changes_after_admission(tmp_path, artifact):
    import hashlib
    from passage.export import _process_meta_for_prepare

    raw = tmp_path / "raw"
    _successor_scene(raw, 0)
    split = raw / "64/train"
    receipt = split / "receipts/0000000000.json"
    admitted_hash = hashlib.sha256(receipt.read_bytes()).hexdigest()
    paths = {
        "metadata": split / "metadata/N000/E000/0000000000.yaml",
        "tensor": split / "tensors/N000/E000/0000000000.npy.zst",
        "path": split / "paths/astar_energy/free/N000/E000/0000000000.csv",
        "receipt": receipt,
    }
    # Whitespace preserves decoded content but must still invalidate admitted bytes.
    paths[artifact].write_bytes(paths[artifact].read_bytes() + b" ")
    with pytest.raises(ValueError, match="content hash mismatch"):
        _process_meta_for_prepare(
            64,
            False,
            ["astar_elevation", "astar_slope_uphill", "astar_energy"],
            split / "metadata/N000/E000/0000000000.yaml",
            verified_receipt_sha256=admitted_hash,
        )


def _native_export_failure(job):
    import os

    os._exit(23)


def test_native_shard_worker_failure_preserves_commits_and_can_retry(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures.process import BrokenProcessPool
    import multiprocessing as mp
    from pathlib import Path
    import passage.export as export

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))
    raw = tmp_path / "raw"
    _successor_scene(raw, 0)
    context = _export_context(tmp_path, raw, tmp_path / "release")
    committed = export.export_successor_batch(context, 64, [0])
    previous_bytes = {f["path"]: (context.export_dirpath / f["path"]).read_bytes() for f in committed["files"]}
    _successor_scene(raw, 1)
    with monkeypatch.context() as patch:
        patch.setattr(export, "_write_successor_shard", _native_export_failure)
        with ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context("spawn")) as executor:
            executor.submit(abs, -1).result()
            processes = list(executor._processes.values())
            with pytest.raises(BrokenProcessPool):
                export.export_successor_batch(context, 64, [1], executor=executor, num_cpus=2)
        assert any(process.exitcode == 23 for process in processes)
    assert (raw / "64/train/receipts/0000000001.json").is_file()
    assert not (context.export_dirpath / "release-receipts/batch-64-0000000001-0000000001.json").exists()
    assert not list(context.export_dirpath.glob(".batch-*"))
    assert previous_bytes == {path: (context.export_dirpath / path).read_bytes() for path in previous_bytes}
    assert export.export_successor_batch(context, 64, [1], num_cpus=2)["raw_cleanup_safe"]


def test_parallel_preview_population_caps_at_64_across_checkpoints(tmp_path):
    from passage.export import export_successor_batch

    raw = tmp_path / "raw"
    for index in range(66):
        _successor_scene(raw, index)
    context = _export_context(tmp_path, raw, tmp_path / "release", chunk_size=33, sample_size=100)
    first = export_successor_batch(context, 64, list(range(66)), num_cpus=2)
    assert first["sample_size_per_split"] == 64
    assert first["sample_ids_total"]["train"] == [f"{i:010d}" for i in range(64)]
    assert sorted(f["rows"] for f in first["files"] if f["samples"]) == [31, 33]
    _successor_scene(raw, 66)
    second = export_successor_batch(context, 64, [66], num_cpus=2)
    assert not any(f["samples"] for f in second["files"])
    assert second["sample_ids_total"]["train"] == first["sample_ids_total"]["train"]


def test_export_workspace_caps_supplied_executor_concurrency():
    from concurrent.futures import ProcessPoolExecutor
    from passage.export import _successor_export_workers

    with ProcessPoolExecutor(max_workers=54) as executor:
        workers, workspace = _successor_export_workers(4096, 25, executor, 54, 8 * 1024**3)
        assert 1 <= workers < 54
        assert workers * workspace <= 8 * 1024**3
        assert _successor_export_workers(64, 20000, executor, 54, 32 * 1024**3)[0] == 54
        with pytest.raises(ValueError, match="cannot contain"):
            _successor_export_workers(4096, 25, executor, 54, workspace - 1)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_export_rejects_invalid_cpu_counts(tmp_path, value):
    from passage.export import export_successor_batch

    raw = tmp_path / "raw"
    _successor_scene(raw, 0)
    context = _export_context(tmp_path, raw, tmp_path / "release")
    with pytest.raises(ValueError, match="num_cpus"):
        export_successor_batch(context, 64, [0], num_cpus=value)


def _blocking_export_job(job, *, ready):
    import os
    import signal

    output = job[6]
    output.mkdir(parents=True, exist_ok=True)
    (output / f"blocked-{os.getpid()}").write_text("worker owns staging")
    ready.send(os.getpid())
    ready.close()
    while True:
        signal.pause()


def test_interrupt_stops_borrowed_native_workers_before_staging_cleanup(tmp_path, monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from functools import partial
    import multiprocessing as mp
    import os
    from pathlib import Path
    import signal
    import threading
    import passage.export as export
    from passage.successor_generation import _stop_executor

    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]))

    raw = tmp_path / "raw"
    for index in range(2):
        _successor_scene(raw, index)
    context = _export_context(tmp_path, raw, tmp_path / "release", chunk_size=1, sample_size=0)
    spawn = mp.get_context("spawn")
    reader, writer = spawn.Pipe(duplex=False)
    executor = ProcessPoolExecutor(max_workers=2, mp_context=spawn)
    completed = threading.Event()
    observed = {"processes": [], "second_interrupt": False, "startup_failed": False, "cleanup_alive": None}
    original_temporary_directory = export.tempfile.TemporaryDirectory

    class ObservedStaging(original_temporary_directory):
        def cleanup(self):
            observed["cleanup_alive"] = any(process.is_alive() for process in observed["processes"])
            super().cleanup()

    def interrupt_ready_workers():
        for _ in range(2):
            if not reader.poll(10):
                if completed.is_set():
                    return
                observed["startup_failed"] = True
                os.kill(os.getpid(), signal.SIGINT)
                return
            reader.recv()
        observed["processes"] = list(executor._processes.values())
        os.kill(os.getpid(), signal.SIGINT)
        # The previous wait(pending) would hang on the second native job.
        # A second interrupt bounds the regression failure without leaving workers.
        if not completed.wait(2):
            observed["second_interrupt"] = True
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(export, "_write_successor_shard", partial(_blocking_export_job, ready=writer))
    monkeypatch.setattr(export.tempfile, "TemporaryDirectory", ObservedStaging)
    interrupter = threading.Thread(target=interrupt_ready_workers, daemon=True)
    interrupter.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            export.export_successor_batch(context, 64, [0, 1], executor=executor, num_cpus=2)
        completed.set()
        interrupter.join(timeout=12)
        assert not interrupter.is_alive()
        assert not observed["startup_failed"]
        assert not observed["second_interrupt"], "export waited for running native work after interrupt"
        assert observed["processes"] and not any(process.is_alive() for process in observed["processes"])
        assert observed["cleanup_alive"] is False, "staging cleanup ran while its producer still lived"
        assert not list(context.export_dirpath.glob(".batch-*"))
        assert not list((context.export_dirpath / "release-receipts").glob("batch-*.json"))
        assert (raw / "64/train/receipts/0000000000.json").is_file()
    finally:
        completed.set()
        interrupter.join(timeout=12)
        _stop_executor(executor, cancel=True)
        reader.close()
        writer.close()
