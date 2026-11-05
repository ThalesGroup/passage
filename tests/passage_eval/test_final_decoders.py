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

"""Behavioural checks of the two frozen final-candidate decoders."""

import numpy as np
import pytest

from passage_eval.baselines.final_decoders import (
    REG_KAPPA,
    REG_TRAIN_SCALE,
    SEG_HEURISTIC_WEIGHT,
    SEG_STEP_WEIGHT,
    build_input,
    reg_bellman_descent,
    seg_logit_astar,
)


def _legal(path, obstacles, start, goal):
    steps = np.abs(np.diff(path, axis=0))
    return (
        tuple(path[0]) == start
        and tuple(path[-1]) == goal
        and (steps.max(axis=1) == 1).all()
        and not obstacles[path[:, 0], path[:, 1]].any()
    )


def _octile_length(path):
    steps = np.abs(np.diff(path, axis=0)).sum(axis=1)
    return float(np.sum(np.where(steps == 2, np.sqrt(2.0), 1.0)))


def test_frozen_parameters():
    assert (SEG_STEP_WEIGHT, SEG_HEURISTIC_WEIGHT, REG_KAPPA, REG_TRAIN_SCALE) == (0.25, 1.5, 8.0, 897803.25)


def test_build_input_channels():
    elevation = np.array([[-430.0, 8767.0], [0.0, 100.0]])
    obstacles = np.array([[False, True], [False, False]])
    x = build_input(elevation, obstacles, 0, 0, 1, 1, -430.0, 8767.0)
    assert x.shape == (1, 3, 2, 2) and x.dtype == np.float32
    assert x[0, 0, 0, 0] == 0.0 and x[0, 0, 0, 1] == 1.0
    assert x[0, 1].tolist() == [[0.0, -1.0], [-1.0, 1.0]]
    assert x[0, 2].tolist() == [[0.0, 1.0], [0.0, 0.0]]


def test_seg_constant_logit_gives_a_geometric_shortest_path():
    obstacles = np.zeros((12, 12), bool)
    result = seg_logit_astar(np.zeros((12, 12)), obstacles, (1, 1), (9, 6), eta=1.0)
    assert result["status"] == "ok" and _legal(result["path"], obstacles, (1, 1), (9, 6))
    assert _octile_length(result["path"]) == pytest.approx(5 * np.sqrt(2.0) + 3)


def test_seg_follows_the_logit_corridor_around_a_wall():
    obstacles = np.zeros((10, 10), bool)
    obstacles[2:, 5] = True  # wall down to the border: only the top rows pass
    logit = np.full((10, 10), -10.0)
    logit[1, :] = 5.0  # corridor above the wall
    result = seg_logit_astar(logit, obstacles, (8, 1), (8, 8))
    assert result["status"] == "ok" and _legal(result["path"], obstacles, (8, 1), (8, 8))
    assert result["path"][:, 0].min() <= 1


def test_reg_descent_on_exact_distance_needs_no_fallback():
    rows = cols = 9
    goal = (7, 2)
    yy, xx = np.mgrid[:rows, :cols]
    dr, dc = np.abs(yy - goal[0]), np.abs(xx - goal[1])
    field = (np.maximum(dr, dc) + (np.sqrt(2.0) - 1.0) * np.minimum(dr, dc)) * 30.0 / REG_TRAIN_SCALE
    obstacles = np.zeros((rows, cols), bool)
    result = reg_bellman_descent(field, obstacles, (0, 8), goal)
    assert result["status"] == "ok" and not result["fallback"]
    assert _legal(result["path"], obstacles, (0, 8), goal)


def test_reg_local_minimum_triggers_goal_rooted_reconnection():
    obstacles = np.zeros((8, 8), bool)
    field = np.full((8, 8), 1.0)
    field[0, 0], field[1, 1] = 0.9, 0.5  # local minimum at (1, 1), away from the goal
    result = reg_bellman_descent(field, obstacles, (0, 0), (6, 6))
    assert result["status"] == "ok" and result["fallback"]
    assert _legal(result["path"], obstacles, (0, 0), (6, 6))
    assert tuple(result["path"][1]) == (1, 1)


def test_bellman_term_prefers_a_cheaper_cardinal_step():
    obstacles = np.zeros((5, 5), bool)
    unit = 30.0 / REG_TRAIN_SCALE
    field = np.full((5, 5), 1.0)
    field[2, 2] = 0.5
    field[2, 3] = 0.5 - 1.0 * unit  # cardinal neighbour
    field[3, 3] = 0.5 - 1.2 * unit  # diagonal neighbour: lower V but a sqrt(2) step
    plain = reg_bellman_descent(field, obstacles, (2, 2), (4, 4), kappa=0.0)
    bellman = reg_bellman_descent(field, obstacles, (2, 2), (4, 4), kappa=1.0)
    assert tuple(plain["path"][1]) == (3, 3)
    assert tuple(bellman["path"][1]) == (2, 3)
    for result in (plain, bellman):
        assert result["status"] == "ok" and _legal(result["path"], obstacles, (2, 2), (4, 4))


def test_rejects_endpoints_on_obstacles():
    obstacles = np.zeros((4, 4), bool)
    obstacles[0, 0] = True
    with pytest.raises(ValueError):
        seg_logit_astar(np.zeros((4, 4)), obstacles, (0, 0), (3, 3))
