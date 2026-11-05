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

"""The second, independent implementation of ``c*`` (8-connected Dijkstra) against the A* reference.

Random synthetic grids only; no dataset is read.
"""

import math

import numpy as np
import pytest
from passage.audit import exact_reference
from passage.pathfinding_utils import PIXEL_SIZE_M
from passage_eval.protocol import second_reference as S
from passage_eval.protocol.planner import Scene

MODELS = ("distance", "elevation", "absolute_elevation", "energy", "slope_uphill")


def test_the_pixel_size_is_the_released_graph_constant():
    assert S.PIXEL_SIZE_M == PIXEL_SIZE_M == 30.0


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("seed", range(6))
def test_the_dijkstra_optimum_equals_the_astar_optimum(model, seed):
    rng = np.random.default_rng(seed)
    size = int(rng.integers(6, 14))
    elevation = rng.normal(0.0, 40.0, size=(size, size))
    obstacles = rng.random((size, size)) < 0.25
    start, goal = (0, 0), (size - 1, size - 1)
    obstacles[start] = obstacles[goal] = False
    weight = float(rng.uniform(0.5, 70.0))
    reference = exact_reference(
        elevation, obstacles, start, goal, cost_model=model, weight=weight, max_expansions=size * size
    )
    cost = S.dijkstra8_cost(elevation, obstacles, start, goal, cost_model=model, weight=weight)
    if reference["status"] == "no_path":
        assert cost is None
    else:
        assert reference["status"] == "exact"
        assert cost == pytest.approx(reference["cost"], rel=1e-9, abs=1e-9)


def test_blocked_endpoints_corner_cutting_and_a_wall():
    flat = np.zeros((3, 3))
    walls = np.zeros((3, 3), dtype=bool)
    walls[0, 1] = walls[1, 0] = True  # the diagonal step (0,0)->(1,1) cuts this corner: allowed by the release
    assert S.dijkstra8_cost(flat, walls, (0, 0), (2, 2), cost_model="distance", weight=0) == pytest.approx(
        2 * 30 * math.sqrt(2)
    )
    walls[1, 1] = True
    assert S.dijkstra8_cost(flat, walls, (0, 0), (2, 2), cost_model="distance", weight=0) is None
    walls[:] = False
    walls[0, 0] = True
    assert S.dijkstra8_cost(flat, walls, (0, 0), (2, 2), cost_model="distance", weight=0) is None
    walls[:] = False
    walls[2, 2] = True
    assert S.dijkstra8_cost(flat, walls, (0, 0), (2, 2), cost_model="distance", weight=0) is None


def test_the_slope_alias_and_invalid_inputs():
    grid = np.arange(16.0).reshape(4, 4)
    free = np.zeros((4, 4), dtype=bool)
    assert S.dijkstra8_cost(grid, free, (0, 0), (3, 3), cost_model="slope", weight=2.0) == S.dijkstra8_cost(
        grid, free, (0, 0), (3, 3), cost_model="slope_uphill", weight=2.0
    )
    with pytest.raises(ValueError, match="does not define"):
        S.dijkstra8_cost(grid, free, (0, 0), (3, 3), cost_model="unknown", weight=1.0)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        S.dijkstra8_cost(grid, free, (0, 0), (3, 3), cost_model="distance", weight=-1.0)
    with pytest.raises(ValueError, match="outside the grid"):
        S.dijkstra8_cost(grid, free, (0, 0), (4, 3), cost_model="distance", weight=1.0)
    with pytest.raises(ValueError, match="one shape"):
        S.dijkstra8_cost(grid, free[:3], (0, 0), (2, 2), cost_model="distance", weight=1.0)


def _scene(**changes):
    values = dict(
        sample_id="s",
        elevation=np.zeros((3, 3)),
        obstacles=np.zeros((3, 3), dtype=bool),
        start=(0, 0),
        goal=(2, 2),
        cost_model="elevation",
        cost_weight=62.4,
        oracle_path=[(0, 0), (1, 1), (2, 2)],
        oracle_cost=60 * math.sqrt(2),
    )
    values.update(changes)
    return Scene(**values)


def test_the_check_compares_exact_and_no_path_outcomes_within_the_graph_tolerance():
    scene = _scene()
    exact = {"reference_status": "exact", "optimal_cost": 60 * math.sqrt(2)}
    assert S.second_reference_check(scene, exact, rtol=1e-7, atol=1e-6)["status"] == "agrees"
    off = dict(exact, optimal_cost=60 * math.sqrt(2) * (1 + 1e-6))
    result = S.second_reference_check(scene, off, rtol=1e-7, atol=1e-6)
    assert result["status"] == "disagrees" and result["abs_difference"] > 1e-6
    assert S.second_reference_check(scene, {"reference_status": "no_path"}, rtol=1e-7, atol=1e-6)["status"] == (
        "disagrees"
    )
    blocked = _scene(obstacles=np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=bool))
    assert S.second_reference_check(blocked, {"reference_status": "no_path"}, rtol=1e-7, atol=1e-6)["status"] == (
        "agrees"
    )
    assert S.second_reference_check(blocked, exact, rtol=1e-7, atol=1e-6)["status"] == "disagrees"
    exhausted = S.second_reference_check(scene, {"reference_status": "budget_exhausted"}, rtol=1e-7, atol=1e-6)
    assert exhausted["status"] == "not_compared" and exhausted["implementation"] == S.IMPLEMENTATION
