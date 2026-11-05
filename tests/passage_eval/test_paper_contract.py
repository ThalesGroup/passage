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

"""Regressions against the NeurIPS submission's path and cost definitions."""

from dataclasses import replace

import numpy as np
import pytest

from passage.pathfinding_utils import build_graph_from_elevation
from passage_eval import AStarPlanner, Planner, PlannerOutput, Scene, score_planner
from passage_eval.baselines.classical import _path_cost
from passage_eval.cli import BenchSpec, _CallablePlanner, _load_scenes
from passage_eval.metrics.geometric import path_assd, path_hausdorff_p95


def test_callable_oracle_path_is_scored_in_metres(metre_scene):
    planner = _CallablePlanner(lambda **_: metre_scene.oracle_path)
    metrics = score_planner(planner, [metre_scene])
    assert metrics["cost_regret_median"] == pytest.approx(0)


def test_candidate_cannot_supply_a_counterfeit_path_cost(metre_scene):
    class CheapPath(Planner):
        def predict_path(self, scene):
            return PlannerOutput(metre_scene.oracle_path, 0)

    assert score_planner(CheapPath(), [metre_scene])["cost_regret_median"] == pytest.approx(0)


@pytest.mark.parametrize(
    "path",
    [[], [(0, 0)], [(0, 0), (2, 2)], [(0, 0), (-1, 0), (0, 1), (1, 2), (2, 2)], [(0, 0), (0.5, 0.5), (1, 1), (2, 2)]],
)
def test_invalid_decodes_receive_censored_cost(metre_scene, path):
    metrics = score_planner(_CallablePlanner(lambda **_: path), [metre_scene])
    assert metrics["success_rate"] == 0
    assert metrics["cost_regret_median"] == pytest.approx(1e6 - 1)
    assert np.isnan(metrics["path_assd"])


def test_connected_obstacle_crossing_is_censored(metre_scene):
    obstacles = metre_scene.obstacles.copy()
    obstacles[1, 1] = True
    scene = replace(metre_scene, obstacles=obstacles)
    metrics = score_planner(_CallablePlanner(lambda **_: scene.oracle_path), [scene])
    assert metrics["success_rate"] == 1
    assert metrics["obstacle_violation_rate"] == 1
    assert metrics["cost_regret_median"] == pytest.approx(1e6 - 1)


@pytest.mark.parametrize("model", ["distance", "elevation", "slope_uphill", "energy", "absolute_elevation"])
def test_evaluation_cost_and_oracle_match_generation(model):
    import networkx as nx

    elevation = np.array([[0.0, 30.0, 10.0], [5.0, 15.0, 20.0], [0.0, 2.0, 4.0]])
    graph = build_graph_from_elevation(elevation, weight=2, cost_model=model, diagonal_movement="always")
    oracle = nx.shortest_path(graph, (0, 0), (2, 2), weight="weight")
    oracle_cost = nx.path_weight(graph, oracle, "weight")
    scene = Scene("terrain", elevation, np.zeros_like(elevation), (0, 0), (2, 2), model, 2.0, oracle, oracle_cost)
    output = AStarPlanner().predict_path(scene)
    assert output.predicted_cost == pytest.approx(oracle_cost)
    assert _path_cost(elevation, oracle, weight=2, cost_model=model) == pytest.approx(oracle_cost)


def test_geometry_uses_waypoint_count_and_pixel_hausdorff():
    oracle = [(0, 0), (1, 1), (2, 1), (3, 2), (4, 3), (4, 4)]
    candidate = [(0, 0), (1, 1), (2, 1), (3, 1), (4, 2), (4, 3), (4, 4)]
    # Both (3,1) and (4,2) deviate from the oracle; (3,2) deviates from
    # the candidate. Apply the displayed formula, not Appendix B's arithmetic.
    assert path_assd([candidate], [oracle])[0] == pytest.approx((2 / 7 + 1 / 6) / 2 / 6)
    assert path_hausdorff_p95([candidate], [oracle])[0] == pytest.approx(1.0)


def test_dataset_failure_is_never_a_synthetic_benchmark(monkeypatch):
    import sys
    from types import ModuleType

    loader = ModuleType("passage_eval.io.dataset")

    def failing_loader(**kwargs):
        raise OSError("dataset load failed")

    loader.load_val_scenes = failing_loader
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    with pytest.raises(OSError, match="dataset load failed"):
        _load_scenes(
            BenchSpec(
                "audit",
                "classical",
                "analytic",
                resolution=3,
                n_samples=1,
                dataset_revision="a" * 40,
            )
        )


def test_tensor_callable_receives_only_released_input(tensor_scene):
    from passage_eval.cli import _materialise_planner
    from unittest.mock import patch

    tensor = tensor_scene.input_tensor
    scene = tensor_scene
    received = []

    def plan(input_tensor):
        received.append(input_tensor)
        return scene.oracle_path

    with patch("passage_eval.cli._import_entry", return_value=plan):
        planner = _materialise_planner(BenchSpec("tensor", "segmentation", "example", inference="example:plan"))
    assert score_planner(planner, [scene])["cost_regret_median"] == pytest.approx(0)
    np.testing.assert_array_equal(received[0], tensor)


def test_scene_planner_cannot_read_oracle_labels(metre_scene):
    observed = []

    class Recorder(Planner):
        def predict_path(self, scene):
            observed.append((scene.oracle_path, scene.oracle_cost))
            return PlannerOutput(metre_scene.oracle_path, float("nan"))

    score_planner(Recorder(), [metre_scene])
    assert observed[0][0] == []
    assert np.isnan(observed[0][1])


def test_random_policy_harness_is_deterministic_and_contract_conformant(metre_scene):
    from passage_eval.baselines.classical import RandomPolicyPlanner

    first = RandomPolicyPlanner(seed=42, max_steps=4).predict_path(metre_scene)
    second = RandomPolicyPlanner(seed=42, max_steps=4).predict_path(metre_scene)
    assert first.predicted_path == second.predicted_path
    assert score_planner(RandomPolicyPlanner(seed=42, max_steps=4), [metre_scene])["success_rate"] == 1


@pytest.mark.parametrize(
    "weight,model,elevation",
    [
        (-1, "distance", np.zeros((2, 2))),
        (np.nan, "distance", np.zeros((2, 2))),
        (1, "unknown", np.zeros((2, 2))),
        (1, "distance", np.zeros(2)),
        (1, "distance", np.full((2, 2), np.nan)),
    ],
)
def test_invalid_analytic_task_is_rejected(weight, model, elevation):
    from passage_eval.metrics.cost import path_cost

    with pytest.raises(ValueError):
        path_cost(elevation, [(0, 0), (1, 1)], weight=weight, cost_model=model)


def test_corrupt_path_shape_is_censored_without_crashing(metre_scene):
    from passage_eval import Planner, PlannerOutput, score_planner

    class BrokenPlanner(Planner):
        def predict_path(self, scene):
            return PlannerOutput([1, 2, 3], 0)

    metrics = score_planner(BrokenPlanner(), [metre_scene])
    assert metrics["cost_regret_median"] == 999999
    assert metrics["success_rate"] == 0


def test_random_harness_uses_campaign_seed(metre_scene):
    from passage_eval import RandomPolicyPlanner

    paths = []
    for seed in (1234, 2026, 3407):
        np.random.seed(seed)
        paths.append(RandomPolicyPlanner(max_steps=8).predict_path(metre_scene).predicted_path)
    np.random.seed(1234)
    assert RandomPolicyPlanner(max_steps=8).predict_path(metre_scene).predicted_path == paths[0]
    assert len({tuple(path) for path in paths}) > 1
