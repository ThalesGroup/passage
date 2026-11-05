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

"""Synthetic qualification of the bounded wrappers around the immutable final kernels."""

import threading

import numpy as np
import pytest

from passage_eval.baselines.final_bounded import RegFinalBounded, SegFinalBounded
from passage_eval.baselines.final_decoders import reg_bellman_descent, seg_logit_astar
from passage_eval.protocol.contracts import BudgetSpec


def test_n_limit_is_insufficient_and_n_plus_one_is_tight():
    field = np.ones((3, 3), np.float32)
    obstacles = np.zeros_like(field, dtype=bool)
    reference = reg_bellman_descent(field, obstacles, (0, 0), (2, 2))
    assert reference["fallback"] and reference["descent_steps"] == 1
    assert reference["fallback_expansions"] == 9
    with pytest.raises(ValueError, match="certified upper bound 10"):
        RegFinalBounded().decode(field, (0, 0), (2, 2), obstacles, BudgetSpec(max_expansions=9))
    actual = RegFinalBounded().decode(field, (0, 0), (2, 2), obstacles, BudgetSpec(max_expansions=10))
    assert actual.work["expansions"] == 10 and actual.status == reference["status"]
    np.testing.assert_array_equal(actual.path, reference["path"])


@pytest.mark.parametrize("seed", range(20))
def test_reg_random_fields_match_exactly_and_respect_bound(seed):
    rng = np.random.default_rng(seed)
    for n in range(2, 13):
        field = rng.random((n, n)).astype(np.float32)
        obstacles = rng.random((n, n)) < 0.2
        start, goal = (0, 0), (n - 1, n - 1)
        obstacles[start] = obstacles[goal] = False
        ref = reg_bellman_descent(field, obstacles, start, goal)
        got = RegFinalBounded().decode(field, start, goal, obstacles, BudgetSpec(max_expansions=n * n + 1))
        assert got.status == ref["status"]
        assert got.work["expansions"] == ref["descent_steps"] + ref["fallback_expansions"] <= n * n + 1
        assert got.diagnostics["fallback"] == ref["fallback"]
        np.testing.assert_array_equal(got.path, ref["path"])


def test_seg_matches_reference_under_common_bound():
    rng = np.random.default_rng(42)
    z = rng.normal(size=(11, 9)).astype(np.float32)
    obstacles = rng.random(z.shape) < 0.2
    start, goal = (0, 0), (10, 8)
    obstacles[start] = obstacles[goal] = False
    ref = seg_logit_astar(z, obstacles, start, goal)
    got = SegFinalBounded().decode(z, start, goal, obstacles, BudgetSpec(max_expansions=z.size + 1))
    assert got.status == ref["status"] and got.work["expansions"] == ref["expansions"]
    np.testing.assert_array_equal(got.path, ref["path"])


def test_exhaustive_binary_fields_on_three_by_three_grid():
    # Exercise flat regions, ties, forced-goal zero, every distinct endpoint,
    # and strict-descent/fallback transitions independently of random draws.
    obstacles = np.zeros((3, 3), dtype=bool)
    points = [(r, c) for r in range(3) for c in range(3)]
    for bits in range(1 << 9):
        field = np.array([(bits >> k) & 1 for k in range(9)], np.float32).reshape(3, 3)
        for start in points:
            for goal in points:
                if start == goal:
                    continue
                result = reg_bellman_descent(field, obstacles, start, goal)
                assert result["descent_steps"] + result["fallback_expansions"] <= 10


@pytest.mark.parametrize("cls", [RegFinalBounded, SegFinalBounded])
@pytest.mark.parametrize("name", ["max_queue_operations", "max_live_states"])
def test_unobservable_budgets_fail_closed(cls, name):
    field = np.ones((3, 3))
    obstacles = np.zeros((3, 3), bool)
    with pytest.raises(ValueError, match="unsupported decoder work budget"):
        cls().decode(field, (0, 0), (2, 2), obstacles, BudgetSpec(**{name: 1000}))


@pytest.mark.parametrize("family", ["seg", "reg"])
def test_wrapper_owns_readonly_input_snapshot_against_caller_mutation(family, monkeypatch):
    from passage_eval.baselines import final_bounded

    field = np.ones((3, 3), dtype=np.float64)
    obstacles = np.zeros((3, 3), dtype=np.bool_)
    start, goal = (0, 0), (2, 2)
    original = final_bounded.seg_logit_astar if family == "seg" else final_bounded.reg_bellman_descent
    expected = original(field.copy(), obstacles.copy(), start, goal)
    name = "seg_logit_astar" if family == "seg" else "reg_bellman_descent"
    mutation_requested, mutation_done = threading.Event(), threading.Event()

    def mutate_caller():
        assert mutation_requested.wait(5)
        field[:] = -10.0
        obstacles[1, 1] = True
        mutation_done.set()

    thread = threading.Thread(target=mutate_caller)
    thread.start()

    def mutate_caller_then_decode(private_field, private_obstacles, *args, **kwargs):
        mutation_requested.set()
        assert mutation_done.wait(5)
        assert not np.shares_memory(private_field, field)
        assert not np.shares_memory(private_obstacles, obstacles)
        assert not private_field.flags.writeable and not private_obstacles.flags.writeable
        return original(private_field, private_obstacles, *args, **kwargs)

    monkeypatch.setattr(final_bounded, name, mutate_caller_then_decode)
    cls = final_bounded.SegFinalBounded if family == "seg" else final_bounded.RegFinalBounded
    result = cls().decode(field, start, goal, obstacles, BudgetSpec(max_expansions=10))
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert result.status == expected["status"]
    np.testing.assert_array_equal(result.path, expected["path"])


@pytest.mark.parametrize("family", ["seg", "reg"])
def test_wrapper_snapshots_mutable_endpoints_before_decoding(family, monkeypatch):
    from passage_eval.baselines import final_bounded

    field = np.ones((3, 3), dtype=np.float64)
    obstacles = np.zeros((3, 3), dtype=np.bool_)
    start, goal = [0, 0], [2, 2]
    name = "seg_logit_astar" if family == "seg" else "reg_bellman_descent"
    original = getattr(final_bounded, name)
    expected = original(field, obstacles, tuple(start), tuple(goal))

    def mutate_endpoints_then_decode(private_field, private_obstacles, private_start, private_goal, **kwargs):
        start[0], goal[0] = 2, 0
        assert private_start == (0, 0) and private_goal == (2, 2)
        return original(private_field, private_obstacles, private_start, private_goal, **kwargs)

    monkeypatch.setattr(final_bounded, name, mutate_endpoints_then_decode)
    cls = final_bounded.SegFinalBounded if family == "seg" else final_bounded.RegFinalBounded
    result = cls().decode(field, start, goal, obstacles, BudgetSpec(max_expansions=10))
    assert result.status == expected["status"]
    np.testing.assert_array_equal(result.path, expected["path"])
