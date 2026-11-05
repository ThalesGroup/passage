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

"""The supervised wrappers must preserve the frozen candidate decoders."""

import numpy as np
import pytest

from passage_eval.baselines.final_decoders import reg_bellman_descent, seg_logit_astar
from passage_eval.protocol.contracts import BudgetSpec


def test_seg_wrapper_matches_reference_and_respects_expansion_budget():
    from passage_eval.baselines.final_operator_adapters import SegFinalDecoder

    rng = np.random.default_rng(2026)
    obstacles = rng.random((16, 16)) < 0.08
    obstacles[0, 0] = obstacles[-1, -1] = False
    field = rng.normal(size=obstacles.shape).astype(np.float32)
    reference = seg_logit_astar(field, obstacles, (0, 0), (15, 15), max_expansions=40)
    actual = SegFinalDecoder().decode(field, (0, 0), (15, 15), obstacles, BudgetSpec(max_expansions=40))
    assert actual.status == reference["status"]
    np.testing.assert_array_equal(actual.path, reference["path"])
    assert actual.work["expansions"] == reference["expansions"] <= 40


def test_reg_wrapper_matches_reference_including_fallback():
    from passage_eval.baselines.final_operator_adapters import RegFinalDecoder

    obstacles = np.zeros((8, 8), dtype=bool)
    field = np.ones((8, 8), dtype=np.float32)
    field[0, 0], field[1, 1] = 0.9, 0.5
    reference = reg_bellman_descent(field, obstacles, (0, 0), (6, 6))
    assert reference["fallback"]
    actual = RegFinalDecoder().decode(field, (0, 0), (6, 6), obstacles, BudgetSpec())
    assert actual.status == reference["status"] and actual.stage == "fallback"
    np.testing.assert_array_equal(actual.path, reference["path"])
    assert actual.work["expansions"] == reference["descent_steps"] + reference["fallback_expansions"]


def test_reg_direct_descent_does_not_report_a_fallback_stage():
    from passage_eval.baselines.final_operator_adapters import RegFinalDecoder

    field = np.arange(25, 0, -1, dtype=np.float32).reshape(5, 5)
    result = RegFinalDecoder().decode(field, (0, 0), (4, 4), np.zeros((5, 5), bool), BudgetSpec())
    assert result.status == "ok" and result.diagnostics["fallback"] is False
    assert result.stage == "primary" and "fallback" not in result.stage_work


@pytest.mark.parametrize("which", ["max_queue_operations", "max_live_states"])
@pytest.mark.parametrize("name", ["SegFinalDecoder", "RegFinalDecoder"])
def test_unsupported_resource_budgets_fail_closed(name, which):
    from passage_eval.baselines import final_operator_adapters

    decoder = getattr(final_operator_adapters, name)()
    with pytest.raises(ValueError, match="unsupported.*budget"):
        decoder.decode(np.ones((3, 3), np.float32), (0, 0), (2, 2), np.zeros((3, 3), bool), BudgetSpec(**{which: 1}))


@pytest.mark.parametrize("limit", [1, 25])
def test_reg_refuses_a_global_expansion_budget_below_the_certified_bound_even_without_fallback(limit):
    from passage_eval.baselines.final_operator_adapters import RegFinalDecoder

    field = np.arange(25, 0, -1, dtype=np.float32).reshape(5, 5)
    with pytest.raises(ValueError, match="certified upper bound 26"):
        RegFinalDecoder().decode(field, (0, 0), (4, 4), np.zeros((5, 5), bool), BudgetSpec(max_expansions=limit))


@pytest.mark.parametrize("extra", [0, 40_000])
def test_reg_accepts_the_protocol_budget_through_the_bounded_wrapper_with_identical_routes(extra):
    """N + 1 (65 537 at 256 x 256) is the certified bound: the frozen kernel then runs unchanged."""
    from passage_eval.baselines.final_operator_adapters import RegFinalDecoder

    rng = np.random.default_rng(2026)
    for size in (5, 12):
        obstacles = rng.random((size, size)) < 0.2
        obstacles[0, 0] = obstacles[-1, -1] = False
        field = rng.random((size, size)).astype(np.float32)
        reference = reg_bellman_descent(field, obstacles, (0, 0), (size - 1, size - 1))
        budget = BudgetSpec(wall_time_ms=120_000.0, max_expansions=size * size + 1 + extra)
        unlimited = RegFinalDecoder().decode(field, (0, 0), (size - 1, size - 1), obstacles, BudgetSpec())
        bounded = RegFinalDecoder().decode(field, (0, 0), (size - 1, size - 1), obstacles, budget)
        for result in (unlimited, bounded):
            assert result.status == reference["status"]
            np.testing.assert_array_equal(result.path, reference["path"])
            assert result.work["expansions"] == reference["descent_steps"] + reference["fallback_expansions"]
            assert result.stage == ("fallback" if reference["fallback"] else "primary")
        assert bounded.work["expansions"] <= size * size + 1
        assert bounded.diagnostics["global_expansions_certified_upper_bound"] == size * size + 1
        assert "global_expansions_certified_upper_bound" not in unlimited.diagnostics


def test_reg_bound_at_the_protocol_resolution_is_n_plus_one():
    from passage_eval.baselines.final_bounded import reg_certified_expansion_bound

    assert reg_certified_expansion_bound(256 * 256) == 65_537
