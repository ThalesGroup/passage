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


@pytest.fixture(scope="session", autouse=True)
def _supervision_tree_hash_cache(tmp_path_factory):
    """One shared on-disk cache of environment tree digests (e.g. ``/usr``) for the whole session.

    Without it each supervised bwrap run would re-hash the bound interpreter environment. The cache is
    keyed by stat signature, and the tests that check it use their own directories.
    """
    import os

    previous = os.environ.get("PASSAGE_TREE_HASH_CACHE")
    os.environ["PASSAGE_TREE_HASH_CACHE"] = str(tmp_path_factory.mktemp("tree-hash-cache"))
    yield
    if previous is None:
        os.environ.pop("PASSAGE_TREE_HASH_CACHE", None)
    else:
        os.environ["PASSAGE_TREE_HASH_CACHE"] = previous


@pytest.fixture(scope="session", autouse=True)
def _harness_checkout_counts_as_committed():
    """Gate G2 refuses a harness checkout with uncommitted changes under ``HARNESS_TRACKED_PATHS``.

    The synthetic GREEN admission worlds are built from the running checkout, which may hold the edits under
    development, so the clean-tree check reports a clean tree here (for the whole session, module fixtures
    included). ``test_admission.py`` tests the real check on a temporary git repository and shows that a dirty tree
    turns G2 RED.
    """
    from passage_eval.protocol import admission

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(admission, "_uncommitted_harness_paths", lambda root=None: [])
        yield
