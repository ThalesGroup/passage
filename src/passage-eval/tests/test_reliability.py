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

from passage_eval import (
    obstacle_cells_traversed_per_path,
    obstacle_violation_rate,
    success_rate,
)


def _connected_path(n=10):
    return np.stack([np.arange(n), np.arange(n)], axis=1)


def test_success_rate_full_when_paths_connect_endpoints():
    p = _connected_path()
    starts = [(0, 0)]
    goals = [(9, 9)]
    out = success_rate([p], starts=starts, goals=goals)
    assert out == 1.0


def test_success_rate_zero_for_disconnected_path():
    p = np.array([[0, 0], [3, 3]])  # consecutive cells differ by 3 (not 8-connected)
    starts = [(0, 0)]
    goals = [(3, 3)]
    out = success_rate([p], starts=starts, goals=goals)
    assert out == 0.0


def test_obstacle_violation_rate_zero_when_no_violation():
    p = _connected_path()
    obs = np.zeros((10, 10), dtype=bool)
    out = obstacle_violation_rate([p], [obs])
    assert out == 0.0


def test_obstacle_violation_rate_one_when_path_crosses_obstacle():
    p = _connected_path()
    obs = np.zeros((10, 10), dtype=bool)
    obs[5, 5] = True
    out = obstacle_violation_rate([p], [obs])
    assert out == 1.0


def test_obstacle_cells_traversed_conditional_on_violation():
    p = _connected_path()
    obs = np.zeros((10, 10), dtype=bool)
    obs[3, 3] = True
    obs[5, 5] = True
    out = obstacle_cells_traversed_per_path([p], [obs])
    assert out == 2.0
