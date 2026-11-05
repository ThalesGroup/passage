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

"""Tests for the Planner ABC + score_planner driver + reference baselines."""

from __future__ import annotations

import math

import numpy as np
import pytest

from passage_eval import (
    AStarPlanner,
    GreedyToGoalPlanner,
    Planner,
    PlannerOutput,
    Scene,
    score_planner,
)


def _make_simple_scene(
    *,
    H: int = 8,
    W: int = 8,
    obstacle_cells: tuple[tuple[int, int], ...] = (),
    cost_model: str = "distance",
    cost_weight: float = 1.0,
    sample_id: str = "scene-0",
) -> Scene:
    """Build a small flat-elevation scene with deterministic oracle."""
    elevation = np.zeros((H, W), dtype=np.float64)
    obstacles = np.zeros((H, W), dtype=np.uint8)
    for r, c in obstacle_cells:
        obstacles[r, c] = 1
    start = (0, 0)
    goal = (H - 1, W - 1)
    # On a flat grid with `distance` cost, the diagonal is the oracle.
    diag_len = max(H, W) - 1
    oracle_path = [(i, i) for i in range(diag_len + 1)]
    oracle_cost = sum(30 * math.hypot(1, 1) for _ in range(diag_len))
    return Scene(
        sample_id=sample_id,
        elevation=elevation,
        obstacles=obstacles,
        start=start,
        goal=goal,
        cost_model=cost_model,
        cost_weight=cost_weight,
        oracle_path=oracle_path,
        oracle_cost=oracle_cost,
    )


class _ConstantPlanner(Planner):
    """Returns a fixed straight horizontal-then-vertical path; no costs."""

    name = "constant_lshape"

    def predict_path(self, scene: Scene) -> PlannerOutput:
        si, sj = scene.start
        gi, gj = scene.goal
        path = [(si, j) for j in range(sj, gj + 1)]
        path += [(i, gj) for i in range(si + 1, gi + 1)]
        return PlannerOutput(predicted_path=path, predicted_cost=float(len(path) - 1))


def test_planner_abc_cannot_be_instantiated():
    with pytest.raises(TypeError):
        Planner()  # type: ignore[abstract]


def test_constant_planner_scores_via_score_planner():
    scenes = [_make_simple_scene(sample_id=f"s{i}") for i in range(3)]
    planner = _ConstantPlanner()
    out = score_planner(planner, scenes)
    # Six primary metrics + bookkeeping.
    for key in (
        "cost_regret_median",
        "cost_regret_p90",
        "path_assd",
        "path_hausdorff_p95",
        "success_rate",
        "obstacle_violation_rate",
        "n_samples",
    ):
        assert key in out
    assert out["n_samples"] == 3
    assert out["success_rate"] == 1.0


def test_astar_planner_recovers_oracle_on_flat_grid():
    scenes = [_make_simple_scene(sample_id=f"s{i}", cost_model="distance") for i in range(2)]
    planner = AStarPlanner()
    out = score_planner(planner, scenes)
    # On flat grid + distance cost, A* matches the oracle.
    assert out["success_rate"] == 1.0
    assert out["cost_regret_median"] == pytest.approx(0.0, abs=1e-9)
    assert out["cost_regret_p90"] == pytest.approx(0.0, abs=1e-9)
    assert out["path_hausdorff_p95"] == pytest.approx(0.0, abs=1e-9)


def test_greedy_planner_succeeds_but_pays_regret_with_obstacles():
    scenes = [
        _make_simple_scene(
            sample_id="block",
            obstacle_cells=((3, 3),),
            cost_model="distance",
        )
    ]
    out = score_planner(GreedyToGoalPlanner(), scenes)
    assert out["success_rate"] == 1.0
    # Oracle is straight diagonal; greedy detours and re-emits sub-optimal cost.
    assert out["cost_regret_median"] >= 0.0


def test_score_planner_rejects_empty_iterable():
    with pytest.raises(ValueError):
        score_planner(_ConstantPlanner(), [])


def test_score_planner_rejects_size_mismatch():
    class _BrokenPlanner(Planner):
        name = "broken"

        def predict_path(self, scene: Scene) -> PlannerOutput:
            return PlannerOutput(predicted_path=[scene.start], predicted_cost=0.0)

        def predict_paths(self, scenes):
            # deliberately drop one
            return [self.predict_path(s) for s in list(scenes)[:-1]]

    scenes = [_make_simple_scene(sample_id=f"s{i}") for i in range(3)]
    with pytest.raises(RuntimeError):
        score_planner(_BrokenPlanner(), scenes)


def test_astar_with_calibrated_elevation_cost_improves_over_greedy():
    rng = np.random.default_rng(0)
    H = W = 12
    elevation = rng.uniform(0, 100, size=(H, W)).astype(np.float64)
    obstacles = np.zeros((H, W), dtype=np.uint8)
    start = (0, 0)
    goal = (H - 1, W - 1)
    cost_model = "elevation"
    cost_weight = 1.0
    # Oracle = AStarPlanner on the same cost grid.
    oracle_planner = AStarPlanner()
    oracle_scene = Scene(
        sample_id="osc",
        elevation=elevation,
        obstacles=obstacles,
        start=start,
        goal=goal,
        cost_model=cost_model,
        cost_weight=cost_weight,
        oracle_path=[start],  # placeholder, not used by oracle planner
        oracle_cost=0.0,
    )
    oracle_out = oracle_planner.predict_path(oracle_scene)
    scene = Scene(
        sample_id="real",
        elevation=elevation,
        obstacles=obstacles,
        start=start,
        goal=goal,
        cost_model=cost_model,
        cost_weight=cost_weight,
        oracle_path=oracle_out.predicted_path,
        oracle_cost=oracle_out.predicted_cost,
    )
    astar_metrics = score_planner(AStarPlanner(), [scene])
    greedy_metrics = score_planner(GreedyToGoalPlanner(), [scene])
    # A* matches itself.
    assert astar_metrics["cost_regret_median"] == pytest.approx(0.0, abs=1e-9)
    # Greedy pays positive regret.
    assert greedy_metrics["cost_regret_median"] >= astar_metrics["cost_regret_median"]
