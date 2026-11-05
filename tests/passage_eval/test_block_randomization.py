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

from itertools import product

import numpy as np
import pytest
from passage_eval.protocol.block_randomization import block_randomization_p_values
from passage_eval.protocol.discipline import DEFAULT_SEEDS


def _methods():
    a, b = {}, {}
    for index, seed in enumerate(DEFAULT_SEEDS):
        ids = np.array(["a", "b", "c", "d"])
        success = np.ones(4, dtype=bool)
        common = {
            "task_id": ids,
            "success": success,
            "valid_path": success,
            "obstacle_violation": np.zeros(4, dtype=bool),
        }
        a[seed] = {
            **common,
            "regret": np.array([0.0, 2.0, 1.0, 5.0]) + index,
            "assd": np.array([0.0, 2.0, 1.0, 5.0]),
            "h95": np.array([2.0, 5.0, 0.0, 4.0]),
        }
        b[seed] = {
            **common,
            "regret": np.array([1.0, 1.0, 3.0, 2.0]) + index,
            "assd": np.array([1.0, 1.0, 3.0, 2.0]),
            "h95": np.array([1.0, 0.0, 3.0, 2.0]),
        }
    return a, b


def test_exact_permutation_swaps_whole_blocks_and_uses_difference_of_aggregates():
    a, b = _methods()
    a[2026]["regret"] = np.array([10.0, 10.0, 0.0, 0.0])
    b[2026]["regret"] = np.array([0.0, 0.0, 10.0, 10.0])
    result = block_randomization_p_values(a, b, blocks=["west", "west", "east", "east"], method="exact")
    observed = np.mean([np.median(a[s]["regret"]) - np.median(b[s]["regret"]) for s in DEFAULT_SEEDS])
    permuted = []
    for west, east in product((False, True), repeat=2):
        swap = np.array([west, west, east, east])
        contrasts = []
        for seed in DEFAULT_SEEDS:
            x = np.where(swap, b[seed]["regret"], a[seed]["regret"])
            y = np.where(swap, a[seed]["regret"], b[seed]["regret"])
            contrasts.append(np.median(x) - np.median(y))
        permuted.append(np.mean(contrasts))
    expected = sum(abs(x) >= abs(observed) - 1e-12 for x in permuted) / len(permuted)
    assert result["permutations"] == 4
    assert result["p_values"]["cost_regret_median"] == expected
    assert result["p_values"]["success_rate"] == 1
    assert result["p_values"]["obstacle_violation_rate"] == 1


def test_identical_methods_have_p_one_and_randomized_test_is_reproducible():
    a, _ = _methods()
    blocks = ["west", "west", "east", "east"]
    first = block_randomization_p_values(a, a, blocks=blocks, method="monte_carlo", permutations=39, seed=13)
    second = block_randomization_p_values(a, a, blocks=blocks, method="monte_carlo", permutations=39, seed=13)
    assert first == second
    assert set(first["p_values"].values()) == {1.0}
    assert first["permutations"] == 39


def test_monte_carlo_agrees_with_small_exact_reference_and_is_never_zero():
    a, b = _methods()
    blocks = ["west", "west", "east", "east"]
    exact = block_randomization_p_values(a, b, blocks=blocks, method="exact")
    sample = block_randomization_p_values(a, b, blocks=blocks, method="monte_carlo", permutations=999, seed=19)
    for name, value in exact["p_values"].items():
        assert abs(sample["p_values"][name] - value) < 0.07
        assert sample["p_values"][name] >= 1 / 1000


def test_exact_four_block_p_values_for_a_hand_computed_example():
    a, b = _methods()
    for seed in DEFAULT_SEEDS:
        for name in ("regret", "assd", "h95"):
            a[seed][name] = np.zeros(4)
            b[seed][name] = np.ones(4)
    result = block_randomization_p_values(a, b, blocks=["w", "x", "y", "z"], method="exact")
    # Of 16 label swaps, ten have a median contrast of magnitude one;
    # two retain the extreme p90 or mean contrast.
    assert result["p_values"]["cost_regret_median"] == 10 / 16
    assert result["p_values"]["cost_regret_p90"] == 2 / 16
    assert result["p_values"]["path_assd"] == 2 / 16
    assert result["p_values"]["path_hausdorff_p95"] == 2 / 16
    assert result["p_values"]["success_rate"] == 1
    assert result["p_values"]["obstacle_violation_rate"] == 1


def test_missing_geography_and_misaligned_seed_roster_are_rejected():
    a, b = _methods()
    with pytest.raises(ValueError, match="blocks"):
        block_randomization_p_values(a, b, blocks=None)
    b[3407]["task_id"] = np.array(["a", "b", "c", "other"])
    with pytest.raises(ValueError, match="task identities"):
        block_randomization_p_values(a, b, blocks=["west", "west", "east", "east"])


def test_undefined_conditional_geometry_does_not_become_a_p_value():
    a, b = _methods()
    for seed in DEFAULT_SEEDS:
        a[seed]["success"] = np.zeros(4, dtype=bool)
        a[seed]["valid_path"] = np.zeros(4, dtype=bool)
    result = block_randomization_p_values(a, b, blocks=["west", "west", "east", "east"], method="exact")
    assert np.isnan(result["p_values"]["path_assd"])
    assert np.isnan(result["p_values"]["path_hausdorff_p95"])
    assert np.isfinite(result["p_values"]["success_rate"])


def test_a_permutation_with_empty_geometry_also_makes_the_test_unavailable():
    a, b = _methods()
    for seed in DEFAULT_SEEDS:
        a[seed]["success"] = a[seed]["valid_path"] = np.array([True, True, False, False])
        b[seed]["success"] = b[seed]["valid_path"] = np.array([False, False, True, True])
    result = block_randomization_p_values(a, b, blocks=["west", "west", "east", "east"], method="exact")
    assert np.isnan(result["p_values"]["path_assd"])
    assert np.isnan(result["p_values"]["path_hausdorff_p95"])
    # A random sample can miss the undefined swaps; they still invalidate the
    # test statistic on the complete null orbit.
    sampled = block_randomization_p_values(
        a, b, blocks=["west", "west", "east", "east"], method="monte_carlo", permutations=1, seed=4
    )
    assert np.isnan(sampled["p_values"]["path_assd"])
    assert np.isnan(sampled["p_values"]["path_hausdorff_p95"])
