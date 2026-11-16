"""Preparation uses real Parquet rows and independent canonical cost checks."""

import json
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage.utils import encode_tensor_blob
from passage_eval.candidates import prepare
from passage_eval.metrics.cost import path_cost
from passage_eval.protocol.candidate_package import file_sha256


def _row(identity, split, *, varied=False):
    tensor = np.zeros((256, 256, 3), np.float32)
    if varied:
        tensor[:, :, 0] = np.linspace(0, 1, 256)[None, :]
    tensor[:, :, 1] = -1
    tensor[0, 0, 1] = 0
    tensor[-1, -1, 1] = 1
    return {
        "id": identity,
        "tensor": encode_tensor_blob(tensor),
        "metadata": {
            "resolution": 256,
            "split": split,
            "calibration": {"min": 0.0, "max": 100.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 255, "j": 255}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
        },
    }


def _source(tmp_path, monkeypatch, *, overlap=False):
    root = tmp_path / "source"
    pinned = {}
    for split in ("train", "validation"):
        path = root / "256x256" / split / "data.parquet"
        path.parent.mkdir(parents=True)
        rows = [
            _row("same" if overlap else f"{split}-{i}", split, varied=split == "validation")
            for i in range(2 if not overlap else 1)
        ]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=1)
        pinned[path.relative_to(root).as_posix()] = file_sha256(path)
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "list_repo_tree",
        lambda *a, **kw: [
            SimpleNamespace(path=path, lfs=SimpleNamespace(sha256=digest)) for path, digest in pinned.items()
        ],
    )
    return root, pinned


def test_real_preparation_train_scale_identity_hash_and_independent_cost(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, monkeypatch)
    out = tmp_path / "prepared"
    prepare.main(
        [
            "--dataset",
            str(source),
            "--out",
            str(out),
            "--revision",
            "a" * 40,
            "--train-count",
            "2",
            "--validation-count",
            "2",
            "--workers",
            "1",
            "--seed",
            "2026",
        ]
    )
    train = json.loads((out / "train/index.json").read_text())
    val = json.loads((out / "validation/index.json").read_text())
    assert train["target_scale"] == max(r["target_max"] for r in train["scenes"])
    assert val["target_scale"] == train["target_scale"]
    assert max(r["target_max"] for r in val["scenes"]) > train["target_scale"]
    assert train["seed"] == 2026 and val["seed"] == 2027
    assert not ({r["sample_id"] for r in train["scenes"]} & {r["sample_id"] for r in val["scenes"]})
    for split, document in [("train", train), ("validation", val)]:
        sources = json.loads((out / split / "sources.json").read_text())
        assert sources["seed"] == document["seed"]
        assert sources["population_rows"] == 2
        assert sources["files"][0]["selected_rows"] == [0, 1]
        for record in document["scenes"]:
            file = out / split / record["file"]
            assert record["sha256"] == file_sha256(file)
            with np.load(file, allow_pickle=False) as scene:
                assert scene["sample_id"].item() == record["sample_id"]
                cost = path_cost(scene["terrain"], scene["expert_path"], weight=62.4, cost_model="elevation")
                assert np.isclose(cost, scene["cost_to_go"][tuple(scene["start"])], rtol=1e-6)
                assert scene["cost_to_go"][tuple(scene["goal"])] == 0
                np.testing.assert_array_equal(scene["reachable"], np.isfinite(scene["cost_to_go"]))
    with pytest.raises(FileExistsError):
        prepare.main(["--dataset", str(source), "--out", str(out), "--revision", "a" * 40])


def test_preparation_rejects_wrong_source_hash_and_test_split(tmp_path, monkeypatch):
    source, pinned = _source(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="hash mismatch"):
        prepare.prepare_split(source, tmp_path / "bad", "train", 1, seed=1, workers=1, pinned={})
    with pytest.raises(ValueError, match="train.*validation"):
        prepare.prepare_split(source, tmp_path / "test", "test", 1, seed=1, workers=1, pinned=pinned)


def test_preparation_rejects_train_validation_identity_overlap(tmp_path, monkeypatch):
    source, _ = _source(tmp_path, monkeypatch, overlap=True)
    with pytest.raises(ValueError, match="overlap"):
        prepare.main(
            [
                "--dataset",
                str(source),
                "--out",
                str(tmp_path / "out"),
                "--revision",
                "a" * 40,
                "--train-count",
                "1",
                "--validation-count",
                "1",
                "--workers",
                "1",
            ]
        )
