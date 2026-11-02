"""Bounded projected access to real Parquet schemas without network fallback."""

import importlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def source(*args, **kwargs):
    return importlib.import_module("passage.dataset").DatasetSource(*args, **kwargs)


def write_rows(root, config, split, rows):
    directory = root / config / split
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), directory / "data-00000.parquet")


def test_local_projection_limit_and_sample_population_are_distinct(tmp_path):
    write_rows(
        tmp_path,
        "64x64",
        "validation",
        [{"id": "core0", "metadata": {"n": 0}, "tensor": b"x"}, {"id": "core1", "metadata": {"n": 1}, "tensor": b"y"}],
    )
    write_rows(tmp_path, "64x64_samples", "validation", [{"id": "preview", "tensor": b"z"}])
    data = source(tmp_path)
    assert list(data.iter_rows(64, "val", columns=["id"], limit=1)) == [{"id": "core0"}]
    assert data.get_row(64, "validation", sample_id="core1", columns=["id"]) == {"id": "core1"}
    assert data.get_row(64, "validation", samples=True, columns=["id"]) == {"id": "preview"}
    assert list(data.iter_rows(64, "validation", columns=["id"], limit=0)) == []


def test_missing_local_source_never_enters_remote_mode(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(source(tmp_path).iter_rows(64, "train", columns=["id"], limit=1))


def test_remote_requires_immutable_revision():
    for revision in (None, "main", "latest"):
        with pytest.raises(ValueError, match="revision"):
            source(revision=revision)


def test_metadata_count_reads_all_four_split_footers(tmp_path):
    for split, count in [("train", 3), ("validation", 2), ("calibration", 1), ("test", 4)]:
        write_rows(tmp_path, "64x64", split, [{"id": str(n)} for n in range(count)])
    assert source(tmp_path).split_counts(64) == {"train": 3, "validation": 2, "calibration": 1, "test": 4}


def test_invalid_projection_and_unknown_id_are_explicit(tmp_path):
    write_rows(tmp_path, "64x64", "test", [{"id": "one"}])
    with pytest.raises(ValueError, match="column"):
        list(source(tmp_path).iter_rows(64, "test", columns=["absent"]))
    with pytest.raises(KeyError, match="missing"):
        source(tmp_path).get_row(64, "test", sample_id="missing")


def test_high_resolution_caps_tensor_batch_and_rejects_invalid_limit():
    data = source("/unused")
    assert data.batch_size_for(4096, columns=["tensor"], requested=100) == 1
    assert data.batch_size_for(64, columns=["metadata"], requested=100) == 100
    with pytest.raises(ValueError, match="limit"):
        list(data.iter_rows(64, "train", limit=-1))


def test_remote_projection_revision_and_cap(monkeypatch):
    import datasets

    seen = {}
    consumed = []

    def rows(repo, **kwargs):
        seen.update(repo=repo, **kwargs)

        def stream():
            for n in range(5):
                consumed.append(n)
                yield {"id": str(n), "tensor": b"unwanted"}

        return stream()

    monkeypatch.setattr(datasets, "load_dataset", rows)
    data = source(revision="a" * 40)
    assert list(data.iter_rows(256, "val", columns=["id"], limit=2)) == [{"id": "0"}, {"id": "1"}]
    assert consumed == [0, 1]
    assert seen["revision"] == "a" * 40
    assert seen["columns"] == ["id"]
    assert seen["batch_size"] > 0


def test_scene_input_decoder_keeps_labels_out_and_checks_markers():
    import numpy as np
    from passage.utils import encode_tensor_blob

    tensor = np.zeros((2, 2, 3), dtype=np.float32)
    tensor[:, :, 0] = 0.5
    tensor[:, :, 1] = -1
    tensor[0, 0, 1] = 0
    tensor[1, 1, 1] = 1
    row = {
        "id": "task",
        "tensor": encode_tensor_blob(tensor),
        "metadata": {
            "resolution": 2,
            "split": "test",
            "calibration": {"min": 10, "max": 20},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 1, "j": 1}},
        },
    }
    decoder = importlib.import_module("passage.dataset").decode_scene_inputs
    inputs = decoder(row, resolution=2, split="test")
    np.testing.assert_allclose(inputs["elevation"], 15)
    assert inputs["start"] == (0, 0)
    assert inputs["goal"] == (1, 1)
    assert "oracle_cost" not in inputs
    row["metadata"]["markers"]["start_point"]["i"] = True
    with pytest.raises(ValueError, match="task"):
        decoder(row, resolution=2, split="test")


@pytest.mark.parametrize("revision", ["development", "v1.0", "abcdef", "a" * 39, "g" * 40])
def test_remote_revision_must_be_a_commit_sha(revision):
    with pytest.raises(ValueError, match="revision"):
        source(revision=revision)


def test_split_manifest_retains_missing_and_schema_failures(tmp_path):
    write_rows(tmp_path, "64x64", "train", [{"id": "0", "tensor": b"x", "metadata": {"resolution": 64}}])
    write_rows(tmp_path, "64x64", "test", [{"id": "1"}])
    manifest = source(tmp_path).split_manifest(64)
    assert manifest["train"]["count"] == 1
    assert manifest["train"]["status"] == "available"
    assert manifest["test"]["status"] == "invalid_schema"
    assert manifest["test"]["count"] == 1
    assert manifest["validation"]["count"] is None
    assert manifest["validation"]["status"] == "missing"
    assert manifest["train"]["content_verified"] is False
