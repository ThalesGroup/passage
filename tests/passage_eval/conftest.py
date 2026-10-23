"""Offline fixtures in the released PASSAGE row format."""

import numpy as np
import pytest
from passage.utils import encode_tensor_blob
from passage_eval import Scene


@pytest.fixture
def release_row():
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0] = 0.75
    tensor[:, :, 1] = -1
    tensor[0, 0, 1] = 0
    tensor[2, 2, 1] = 1
    tensor[0, 2, 2] = 1
    cost = float(60 * np.sqrt(2))
    return {
        "id": "0000000042",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_elevation_obstacles": [[0, 0], [1, 1], [2, 2]],
        "path_astar_elevation_free": [[0, 0], [1, 1], [2, 2]],
        "metadata": {
            "resolution": 3,
            "split": "validation",
            "index": 42,
            "calibration": {"min": -100.0, "max": 100.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
            "paths": {"astar_elevation": {"obstacles": {"cost": cost}, "free": {"cost": cost}}},
        },
    }


@pytest.fixture
def metre_scene():
    return Scene(
        sample_id="released-0",
        elevation=np.zeros((3, 3)),
        obstacles=np.zeros((3, 3), dtype=bool),
        start=(0, 0),
        goal=(2, 2),
        cost_model="elevation",
        cost_weight=62.4,
        oracle_path=[(0, 0), (1, 1), (2, 2)],
        oracle_cost=60 * np.sqrt(2),
    )


@pytest.fixture
def tensor_scene(metre_scene):
    from dataclasses import replace

    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 1] = -1
    tensor[metre_scene.start + (1,)] = 0
    tensor[metre_scene.goal + (1,)] = 1
    return replace(metre_scene, input_tensor=tensor, metadata={"calibration": {"min": 0.0, "max": 1.0}})
