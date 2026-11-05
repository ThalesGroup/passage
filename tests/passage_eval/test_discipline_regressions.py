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

"""Counterexamples for the declared aggregate, seed and block estimands."""

import numpy as np
import pytest

from passage_eval.protocol import discipline


def test_nonlinear_contrast_is_difference_of_aggregates():
    result = discipline.paired_bootstrap_delta([0, 10, 11], [0, 1, 20], statistic=np.median, n_bootstrap=20)
    assert result.delta == 9


def test_seed_mean_is_averaged_inside_each_draw():
    result, draws = discipline.multiseed_paired_bootstrap(
        [np.full(3, i) for i in range(3)],
        [np.ones(3)] * 3,
        n_bootstrap_per_seed=20,
        return_draws=True,
    )
    assert (result.delta, result.ci_low, result.ci_high) == (0, 0, 0)
    assert result.n_bootstrap == 20
    assert draws.tolist() == [0] * 20


def test_seed_contrasts_share_rows_before_averaging():
    # Seed contrasts cancel row by row. Independent draws or separately sorting
    # each seed's draws would leave artificial geographic uncertainty.
    result, draws = discipline.multiseed_paired_bootstrap(
        [[0, 10, 20], [20, 10, 0], [10, 10, 10]],
        [[10, 10, 10]] * 3,
        n_bootstrap_per_seed=100,
        return_draws=True,
    )
    assert result.delta == 0
    assert np.allclose(draws, 0)
    assert result.ci_low == pytest.approx(0)
    assert result.ci_high == pytest.approx(0)


def test_whole_blocks_preserve_task_weights():
    result, draws = discipline.paired_bootstrap_delta(
        [0, 0, 9],
        [0, 0, 0],
        blocks=["west", "west", "east"],
        n_bootstrap=100,
        return_draws=True,
    )
    # Two west: 0; west+east: 3 (three tasks); two east: 9.
    assert result.delta == 3
    assert set(draws) == {0, 3, 9}
    assert result.n_blocks == 2


def test_geometry_retains_each_methods_own_valid_denominator():
    result, _ = discipline.paired_bootstrap_delta(
        [1, np.nan, 9],
        [2, 4, np.nan],
        nan_policy="omit",
        n_bootstrap=30,
        return_draws=True,
    )
    assert result.delta == 2  # mean(1, 9) - mean(2, 4)
    assert (result.n, result.n_a, result.n_b) == (3, 2, 2)
    # Some original-population draws have an empty method subset. They cannot
    # be dropped and called an unconditional percentile interval.
    assert np.isnan(result.ci_low)
    assert np.isnan(result.ci_high)


def test_nonfinite_values_require_explicit_omission_policy():
    with pytest.raises(ValueError, match="finite"):
        discipline.paired_bootstrap_delta([1, np.nan], [2, 3])


def test_missing_holm_value_stays_missing_and_keeps_family_size():
    result = discipline.holm_bonferroni({"measured": 0.001, "missing": np.nan})
    assert result["measured"].p_adjusted == 0.002
    assert np.isnan(result["missing"].p_adjusted)
    assert not result["missing"].rejected


@pytest.mark.parametrize("value", [-0.1, 1.1, np.inf, -np.inf])
def test_holm_rejects_invalid_probabilities(value):
    with pytest.raises(ValueError, match="p-value"):
        discipline.holm_bonferroni({"bad": value})


@pytest.mark.parametrize("kwargs", [{"n_bootstrap": 0}, {"n_bootstrap": True}, {"alpha": 1}])
def test_bootstrap_rejects_invalid_sampling_parameters(kwargs):
    with pytest.raises(ValueError):
        discipline.paired_bootstrap_delta([1, 2], [2, 3], **kwargs)


def test_shared_seed_bootstrap_rejects_incomplete_or_duplicate_seed_rosters():
    with pytest.raises(ValueError, match="shape|align"):
        discipline.multiseed_paired_bootstrap([[1], [1, 2], [1]], [[2], [2, 3], [2]])
    with pytest.raises(ValueError, match="distinct"):
        discipline.multiseed_paired_bootstrap([[1]] * 3, [[2]] * 3, seeds=[1, 1, 2])


def test_missing_draws_do_not_turn_into_zero_p_values():
    result = discipline.BootstrapDelta("geometry", np.nan, np.nan, np.nan, 3, 20, False)
    assert np.isnan(result.two_sided_p_value(np.array([np.nan, np.nan])))


def test_one_direction_with_inconclusive_metrics_is_partial_evidence():
    delta = discipline.BootstrapDelta
    decision, _ = discipline.decide_pairwise(
        [
            delta("regret", -1, -2, -0.1, 10, 20, False),
            delta("geometry", 0, -1, 1, 10, 20, False),
        ]
    )
    assert decision.value == "partial_evidence"


def _primary_arrays(regret):
    return {
        seed: {
            "task_id": np.array(["task-0", "task-1", "task-2"]),
            "regret": np.asarray(regret),
            "assd": np.array([1.0, 2.0, 3.0]),
            "h95": np.array([2.0, 4.0, 6.0]),
            "success": np.ones(3, dtype=bool),
            "obstacle_violation": np.zeros(3, dtype=bool),
            "valid_path": np.ones(3, dtype=bool),
        }
        for seed in discipline.DEFAULT_SEEDS
    }


def test_canonical_pipeline_keeps_six_metrics_and_requires_declared_test_for_claims():
    result = discipline.compare_primary_metrics(
        _primary_arrays([0, 10, 11]),
        _primary_arrays([0, 1, 20]),
        blocks=["west", "centre", "east"],
        n_bootstrap=30,
    )
    assert list(result.deltas) == [name for name, _, _ in discipline.PRIMARY_METRICS]
    assert result.deltas["cost_regret_median"].delta == 9
    assert result.deltas["cost_regret_p90"].delta == pytest.approx(-5.4)
    assert set(result.deltas["cost_regret_median"].per_seed) == set(discipline.DEFAULT_SEEDS)
    assert result.inference_status == "test_not_provided"
    assert all(np.isnan(record.p_adjusted) for record in result.holm.values())
    assert set(result.votes.values()) == {"tie"}


def test_canonical_pipeline_applies_holm_to_the_complete_declared_family():
    p_values = {name: 0.001 for name, _, _ in discipline.PRIMARY_METRICS}
    result = discipline.compare_primary_metrics(
        _primary_arrays([0, 0, 0]),
        _primary_arrays([1, 1, 1]),
        blocks=["west", "centre", "east"],
        n_bootstrap=30,
        p_values=p_values,
        paired_test="predeclared paired test",
        test_assumptions="exchangeable geographic blocks",
    )
    assert len(result.holm) == 6
    assert all(record.p_adjusted == 0.006 for record in result.holm.values())
    assert result.votes["cost_regret_median"] == "win_a"
    assert result.decision.value == "partial_evidence"


def test_canonical_geometry_masks_failures_and_reports_own_subset_sizes():
    a, b = _primary_arrays([0, 999999, 0]), _primary_arrays([0, 0, 999999])
    for columns in a.values():
        columns["success"][1] = columns["valid_path"][1] = False
        columns["assd"] = np.array([1.0, 123.0, 9.0])
    for columns in b.values():
        columns["success"][2] = columns["valid_path"][2] = False
        columns["assd"] = np.array([2.0, 4.0, 123.0])
    result = discipline.compare_primary_metrics(a, b, blocks=["w", "c", "e"], n_bootstrap=50)
    geometry = result.deltas["path_assd"]
    assert (geometry.n, geometry.n_a, geometry.n_b) == (9, 6, 6)
    assert geometry.delta == 2
    assert np.isnan(geometry.ci_low)
    assert result.deltas["cost_regret_median"].n_a == 9


def test_canonical_pipeline_rejects_missing_seed_blocks_or_test_metadata():
    a, b = _primary_arrays([0, 0, 0]), _primary_arrays([1, 1, 1])
    with pytest.raises(ValueError, match="seed"):
        discipline.compare_primary_metrics({1234: a[1234]}, b, blocks=["w", "c", "e"])
    with pytest.raises(ValueError, match="block"):
        discipline.compare_primary_metrics(a, b, blocks=["one"] * 3)
    with pytest.raises(ValueError, match="test"):
        discipline.compare_primary_metrics(a, b, blocks=["w", "c", "e"], p_values={})


def test_canonical_pipeline_rejects_reordered_or_duplicate_tasks():
    a, b = _primary_arrays([0, 0, 0]), _primary_arrays([1, 1, 1])
    b[2026]["task_id"] = np.array(["task-1", "task-0", "task-2"])
    with pytest.raises(ValueError, match="task"):
        discipline.compare_primary_metrics(a, b, blocks=["w", "c", "e"])
    a[1234]["task_id"] = np.array(["task-0", "task-0", "task-2"])
    with pytest.raises(ValueError, match="task"):
        discipline.compare_primary_metrics(a, b, blocks=["w", "c", "e"])


def test_canonical_pipeline_requires_known_geographic_blocks():
    a, b = _primary_arrays([0, 0, 0]), _primary_arrays([1, 1, 1])
    with pytest.raises(ValueError, match="block"):
        discipline.compare_primary_metrics(a, b, blocks=None, n_bootstrap=1)


@pytest.mark.parametrize("seed", [None, True, np.random.default_rng(1)])
def test_mutable_or_absent_resampling_seed_cannot_break_shared_metric_draws(seed):
    a, b = _primary_arrays([0, 0, 0]), _primary_arrays([1, 1, 1])
    with pytest.raises(ValueError, match="seed"):
        discipline.compare_primary_metrics(a, b, blocks=["w", "c", "e"], seed=seed, n_bootstrap=1)
