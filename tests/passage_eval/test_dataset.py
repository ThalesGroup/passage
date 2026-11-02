"""Exercise the released Parquet and compressed-tensor schema offline."""

import copy
import importlib

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _loader():
    return importlib.import_module("passage_eval.io.dataset")


def test_local_parquet_restores_metres_and_released_oracle(tmp_path, release_row):
    split = tmp_path / "3x3" / "validation"
    split.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([release_row]), split / "data-00000.parquet")
    scenes = _loader().load_val_scenes(resolution=3, cost_model="elevation", n_samples=1, dataset_path=tmp_path)
    scene = scenes[0]
    assert scene.sample_id == "0000000042"
    np.testing.assert_allclose(scene.elevation, 50.0)
    assert scene.obstacles[0, 2]
    assert scene.start == (0, 0) and scene.goal == (2, 2)
    assert scene.cost_weight == 62.4
    assert scene.oracle_cost == pytest.approx(60 * np.sqrt(2))
    assert scene.oracle_path == [(0, 0), (1, 1), (2, 2)]
    assert scene.input_tensor.shape == (3, 3, 3)


def test_huggingface_stream_uses_declared_repository_revision_and_split(monkeypatch, release_row):
    import datasets

    seen = {}

    def stream(repo, **kwargs):
        seen.update(repo=repo, **kwargs)
        return iter([release_row])

    monkeypatch.setattr(datasets, "load_dataset", stream)
    scenes = _loader().load_val_scenes(
        resolution=3, cost_model="elevation", n_samples=1, dataset_revision="a" * 40, split="val"
    )
    assert scenes[0].sample_id == release_row["id"]
    assert seen == {
        "repo": "thalesgroup/passage",
        "name": "3x3",
        "split": "validation",
        "revision": "a" * 40,
        "streaming": True,
        "columns": ["id", "tensor", "metadata", "path_astar_elevation_obstacles"],
        "batch_size": 1,
    }


@pytest.mark.parametrize(
    "mutation",
    ["bad_tensor", "bad_resolution", "bad_split", "missing_oracle", "bad_calibration", "bad_marker", "bad_oracle_cost"],
)
def test_corrupt_release_is_rejected_with_sample_identity(release_row, mutation):
    row = copy.deepcopy(release_row)
    if mutation == "bad_tensor":
        row["tensor"] = b"corrupt"
    elif mutation == "bad_resolution":
        row["metadata"]["resolution"] = 4
    elif mutation == "bad_split":
        row["metadata"]["split"] = "test"
    elif mutation == "missing_oracle":
        row["path_astar_elevation_obstacles"] = []
    elif mutation == "bad_calibration":
        row["metadata"]["calibration"]["max"] = -100
    elif mutation == "bad_marker":
        row["metadata"]["markers"]["start_point"]["i"] = -1
    else:
        row["metadata"]["paths"]["astar_elevation"]["obstacles"]["cost"] = -1
    with pytest.raises(ValueError, match="0000000042"):
        _loader().scene_from_row(row, resolution=3, cost_model="elevation", split="validation")


def test_weight_override_cannot_change_the_released_task(release_row):
    with pytest.raises(ValueError, match="weight"):
        _loader().scene_from_row(release_row, resolution=3, cost_model="elevation", split="validation", cost_weight=1.0)


def test_free_variant_removes_obstacles_from_input_tensor(release_row):
    scene = _loader().scene_from_row(
        release_row, resolution=3, cost_model="elevation", split="validation", task_variant="free"
    )
    assert not scene.obstacles.any()
    assert not scene.input_tensor[:, :, 2].any()


def test_missing_local_split_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        _loader().load_val_scenes(resolution=3, cost_model="elevation", dataset_path=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    ["unknown_variant", "bad_elevation", "bad_obstacle", "missing_solver", "bad_connectivity", "same_endpoints"],
)
def test_invalid_task_channels_and_solver_are_rejected(release_row, mutation):
    from passage.utils import decode_tensor_blob, encode_tensor_blob

    row = copy.deepcopy(release_row)
    tensor = decode_tensor_blob(row["tensor"], (3, 3, 3)).copy()
    variant = "obstacles"
    if mutation == "unknown_variant":
        variant = "other"
    elif mutation == "bad_elevation":
        tensor[1, 0, 0] = -1
    elif mutation == "bad_obstacle":
        tensor[1, 0, 2] = 0.5
    elif mutation == "missing_solver":
        row["metadata"]["solvers"] = {}
    elif mutation == "bad_connectivity":
        row["metadata"]["solvers"]["astar_elevation"]["diagonal_movement"] = "never"
    else:
        row["metadata"]["markers"]["end_point"] = {"i": 0, "j": 0}
    row["tensor"] = encode_tensor_blob(tensor)
    with pytest.raises(ValueError, match="0000000042"):
        _loader().scene_from_row(row, resolution=3, cost_model="elevation", split="validation", task_variant=variant)


def test_empty_and_duplicate_released_rows_abort(monkeypatch, release_row):
    import datasets

    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="empty"):
        _loader().load_val_scenes(resolution=3, cost_model="elevation", dataset_revision="a" * 40)
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: [release_row, release_row])
    with pytest.raises(ValueError, match="Duplicate"):
        _loader().load_val_scenes(resolution=3, cost_model="elevation", n_samples=2, dataset_revision="a" * 40)
    with pytest.raises(ValueError, match="positive"):
        _loader().load_val_scenes(resolution=3, cost_model="elevation", n_samples=0)


@pytest.mark.parametrize("mutation", ["missing_id", "empty_id", "extra_marker", "illegal_marker"])
def test_release_identity_and_marker_channel_must_be_unambiguous(release_row, mutation):
    from passage.utils import decode_tensor_blob, encode_tensor_blob

    row = copy.deepcopy(release_row)
    if mutation == "missing_id":
        del row["id"]
    elif mutation == "empty_id":
        row["id"] = ""
    else:
        tensor = decode_tensor_blob(row["tensor"], (3, 3, 3)).copy()
        tensor[1, 0, 1] = 0 if mutation == "extra_marker" else 0.5
        row["tensor"] = encode_tensor_blob(tensor)
    with pytest.raises(ValueError):
        _loader().scene_from_row(row, resolution=3, cost_model="elevation", split="validation")


def test_unpinned_remote_release_fails_before_network():
    with pytest.raises(ValueError, match="revision"):
        _loader().load_val_scenes(resolution=3, cost_model="elevation", n_samples=1)


def test_free_scene_records_its_task_variant(release_row):
    scene = _loader().scene_from_row(
        release_row, resolution=3, split="validation", cost_model="elevation", task_variant="free"
    )
    assert scene.metadata["task_variant"] == "free"


def test_local_loader_preserves_arbitrary_released_solver_name(tmp_path, release_row):
    original = "astar_elevation"
    replacement = "release_solver"
    release_row["metadata"]["solvers"][replacement] = release_row["metadata"]["solvers"].pop(original)
    release_row["metadata"]["paths"][replacement] = release_row["metadata"]["paths"].pop(original)
    for variant in ("free", "obstacles"):
        release_row[f"path_{replacement}_{variant}"] = release_row.pop(f"path_{original}_{variant}")
    directory = tmp_path / "3x3" / "validation"
    directory.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([release_row]), directory / "data.parquet")
    scene = _loader().load_val_scenes(resolution=3, cost_model="elevation", n_samples=1, dataset_path=tmp_path)[0]
    assert scene.sample_id == release_row["id"]
