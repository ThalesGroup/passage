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

import numpy as np
import pytest

import passage.generate as gen


@pytest.mark.unit
@pytest.mark.generate
def test_find_path_various_cost_models():
    """Ensure find_path runs across different cost models and returns a path."""
    crop = np.zeros((16, 16), dtype=np.float32)
    # create a mild ramp so different cost models have some effect
    for i in range(crop.shape[0]):
        crop[i, :] = float(i)

    solver_configs = [
        {"name": "networkx:astar", "diagonal_movement": "always", "weight": 1.0},
        {"name": "grid:astar", "diagonal_movement": "always", "weight": 1.0},
    ]

    cost_models = ["elevation", "distance", "absolute_elevation", "slope_uphill", "energy"]

    for cfg in solver_configs:
        for cm in cost_models:
            cfg_local = dict(cfg)
            cfg_local["cost_model"] = cm
            path = gen.find_path(crop, (0, 0), (15, 15), cfg_local)
            # Path should be a list (possibly empty for some pathological configs) and not raise
            assert isinstance(path, list)


@pytest.mark.unit
@pytest.mark.generate
def test_grid_solvers_with_cost_models():
    """Directly exercise internal grid solvers across cost models to increase coverage."""
    # A small gradient terrain
    crop = np.tile(np.linspace(0.0, 10.0, 8, dtype=np.float32), (8, 1))

    # A* (grid) should find a path
    path_astar = gen._grid_astar(
        crop, (0, 0), (7, 7), weight=5.0, diagonal_movement="always", heuristic_name="euclidean", cost_model="elevation"
    )
    assert len(path_astar) > 0

    # Dijkstra (grid) should find a path
    path_dijkstra = gen._grid_dijkstra(
        crop, (0, 0), (7, 7), weight=5.0, diagonal_movement="always", cost_model="absolute_elevation"
    )
    assert len(path_dijkstra) > 0
