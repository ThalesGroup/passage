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

"""Tests for the App. J statistical-discipline primitives."""

from __future__ import annotations

import numpy as np
import pytest

from passage_eval import (
    DEFAULT_SEEDS,
    PRIMARY_METRICS,
    BootstrapDelta,
    PairwiseDecision,
    apply_censored_penalty,
    decide_pairwise,
    holm_bonferroni,
    multiseed_paired_bootstrap,
    paired_bootstrap_delta,
)


def test_default_seeds_match_paper():
    assert DEFAULT_SEEDS == (1234, 2026, 3407)


def test_primary_metrics_listed_in_paper_order():
    names = [m[0] for m in PRIMARY_METRICS]
    assert names == [
        "cost_regret_median",
        "cost_regret_p90",
        "path_assd",
        "path_hausdorff_p95",
        "success_rate",
        "obstacle_violation_rate",
    ]
    # higher_is_better only for success_rate
    higher = {n: hib for n, _, hib in PRIMARY_METRICS}
    assert higher["success_rate"] is True
    assert higher["cost_regret_median"] is False
    assert higher["obstacle_violation_rate"] is False


def test_apply_censored_penalty_rewrites_failed_decodes():
    pred = np.array([100.0, np.nan, 110.0, 120.0])
    orac = np.array([100.0, 100.0, 100.0, 100.0])
    succ = np.array([True, False, True, True])
    viol = np.array([False, False, True, False])
    out = apply_censored_penalty(pred, orac, decoder_succeeded=succ, obstacle_violated=viol, k_factor=1e6)
    assert out[0] == 100.0
    assert out[1] == 1e8  # nan + decoder failed -> 1e6 * oracle
    assert out[2] == 1e8  # obstacle violated -> 1e6 * oracle
    assert out[3] == 120.0


def test_paired_bootstrap_delta_zero_when_paired_identical():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    rec = paired_bootstrap_delta(
        a,
        a,
        metric="cost_regret_median",
        n_bootstrap=200,
        seed=DEFAULT_SEEDS[0],
    )
    assert rec.delta == 0.0
    assert rec.ci_low <= 0.0 <= rec.ci_high
    assert not rec.excludes_zero


def test_paired_bootstrap_delta_detects_a_better():
    rng = np.random.default_rng(0)
    n = 200
    # method A's regret is lower than B's by ~0.05 on every sample
    a = rng.uniform(0.0, 0.1, size=n)
    b = a + 0.05 + rng.normal(0, 0.005, size=n)
    rec = paired_bootstrap_delta(
        a,
        b,
        metric="cost_regret_median",
        higher_is_better=False,
        n_bootstrap=2000,
        seed=0,
    )
    # mean(a - b) ~= -0.05
    assert rec.delta < 0.0
    assert rec.excludes_zero
    assert rec.favours_a is True  # lower-is-better, delta < 0 => A wins


def test_multiseed_returns_one_seed_mean_per_shared_resample():
    rng = np.random.default_rng(0)
    n = 50
    by_seed_a = {s: rng.uniform(0, 0.1, size=n) for s in DEFAULT_SEEDS}
    by_seed_b = {s: rng.uniform(0.05, 0.15, size=n) for s in DEFAULT_SEEDS}
    rec, draws = multiseed_paired_bootstrap(
        by_seed_a,
        by_seed_b,
        metric="cost_regret_median",
        higher_is_better=False,
        n_bootstrap_per_seed=500,
        seeds=DEFAULT_SEEDS,
        return_draws=True,
    )
    assert draws.shape[0] == 500
    assert rec.n_bootstrap == 500
    assert rec.n == 3 * n


def test_holm_bonferroni_orders_and_rejects():
    # 4 hypotheses; smallest p must clear alpha/4.
    p = {"m1": 0.001, "m2": 0.04, "m3": 0.20, "m4": 0.50}
    out = holm_bonferroni(p, alpha=0.05)
    # m1: 0.001 <= 0.05/4 = 0.0125 -> reject
    # m2: 0.04  <= 0.05/3 = 0.0167 -> NOT reject -> stop
    assert out["m1"].rejected is True
    assert out["m2"].rejected is False
    assert out["m3"].rejected is False
    assert out["m4"].rejected is False
    # adjusted p is monotonic
    adj = [out[k].p_adjusted for k in ["m1", "m2", "m3", "m4"]]
    assert adj == sorted(adj)


def test_holm_bonferroni_all_reject():
    p = {"m1": 0.001, "m2": 0.002, "m3": 0.003, "m4": 0.004}
    out = holm_bonferroni(p, alpha=0.05)
    for r in out.values():
        assert r.rejected


def test_decide_pairwise_indistinguishable_when_no_metric_excludes_zero():
    deltas = [
        BootstrapDelta("cost_regret_median", 0.0, -0.1, 0.1, 100, 1000, False),
        BootstrapDelta("cost_regret_p90", 0.0, -0.1, 0.1, 100, 1000, False),
    ]
    decision, votes = decide_pairwise(deltas, holm=None)
    assert decision is PairwiseDecision.INDISTINGUISHABLE
    assert all(v == "tie" for v in votes.values())


def test_decide_pairwise_dominant_a_when_all_metrics_favour_a():
    deltas = [
        # For lower-is-better metrics, delta < 0 means A wins.
        BootstrapDelta("cost_regret_median", -0.05, -0.08, -0.02, 100, 1000, False),
        BootstrapDelta("path_assd", -0.10, -0.15, -0.05, 100, 1000, False),
        # For higher-is-better, delta > 0 means A wins.
        BootstrapDelta("success_rate", 0.05, 0.02, 0.08, 100, 1000, True),
    ]
    holm = holm_bonferroni({d.metric: 0.001 for d in deltas})
    decision, votes = decide_pairwise(deltas, holm=holm)
    assert decision is PairwiseDecision.PARETO_DOMINANT_A
    assert set(votes.values()) == {"win_a"}


def test_decide_pairwise_trade_off_when_directions_disagree():
    deltas = [
        BootstrapDelta("cost_regret_median", -0.05, -0.08, -0.02, 100, 1000, False),
        BootstrapDelta("path_assd", 0.10, 0.05, 0.15, 100, 1000, False),
    ]
    holm = holm_bonferroni({d.metric: 0.001 for d in deltas})
    decision, votes = decide_pairwise(deltas, holm=holm)
    assert decision is PairwiseDecision.TRADE_OFF
    assert set(votes.values()) == {"win_a", "win_b"}


def test_decide_pairwise_treats_holm_failure_as_tie():
    deltas = [
        BootstrapDelta("cost_regret_median", -0.001, -0.002, -0.0005, 100, 1000, False),
    ]
    # Holm fails to reject -> tie even though the CI excludes zero
    holm = holm_bonferroni({"cost_regret_median": 0.49})
    decision, votes = decide_pairwise(deltas, holm=holm)
    assert decision is PairwiseDecision.INDISTINGUISHABLE
    assert votes == {"cost_regret_median": "tie"}


def test_geometry_bootstrap_uses_each_methods_valid_subset():
    rec, draws = paired_bootstrap_delta(
        np.array([0.1, np.nan, 0.9]),
        np.array([0.2, 0.3, np.nan]),
        metric="path_assd",
        n_bootstrap=20,
        return_draws=True,
        nan_policy="omit",
    )
    assert (rec.n, rec.n_a, rec.n_b) == (3, 2, 2)
    assert rec.delta == pytest.approx(0.25)
    assert draws.shape == (20,)


def test_multiseed_geometry_reports_each_methods_sample_count():
    a = {s: np.array([0.1, np.nan]) for s in DEFAULT_SEEDS}
    b = {s: np.array([0.2, 0.3]) for s in DEFAULT_SEEDS}
    rec = multiseed_paired_bootstrap(a, b, metric="path_assd", n_bootstrap_per_seed=20, nan_policy="omit")
    assert (rec.n, rec.n_a, rec.n_b) == (6, 3, 6)
    assert rec.delta == pytest.approx(-0.15)


def test_unavailable_metric_does_not_allow_full_pareto_claim():
    deltas = [
        BootstrapDelta("regret", -0.1, -0.2, -0.01, 10, 20, False),
        BootstrapDelta("geometry", 0, -0.1, 0.1, 10, 20, False),
    ]
    decision, _ = decide_pairwise(deltas)
    assert decision is PairwiseDecision.PARTIAL_EVIDENCE


@pytest.mark.parametrize("a,b", [([], []), ([np.nan], [1.0])])
def test_unavailable_pairs_report_no_statistical_evidence(a, b):
    rec, draws = paired_bootstrap_delta(a, b, return_draws=True, n_bootstrap=20, nan_policy="omit")
    assert rec.n == len(a)
    assert np.isnan(rec.delta)
    assert not rec.excludes_zero
    assert draws.size == (20 if len(a) else 0)
    assert np.isnan(rec.two_sided_p_value(draws))
    pooled, draws = multiseed_paired_bootstrap(
        [a] * 3, [b] * 3, return_draws=True, n_bootstrap_per_seed=20, nan_policy="omit"
    )
    assert pooled.n == 3 * len(a)
    assert pooled.n_bootstrap == draws.size == (20 if len(a) else 0)


@pytest.mark.parametrize("a,b", [([1], [1, 2]), ([[1]], [[2]])])
def test_bootstrap_rejects_unpaired_or_multidimensional_arrays(a, b):
    with pytest.raises(ValueError):
        paired_bootstrap_delta(a, b)


def test_bootstrap_p_values_and_seed_alignment():
    rec = BootstrapDelta("regret", -0.1, -0.2, -0.01, 10, 20, False)
    assert 0 < rec.two_sided_p_value() < 1
    assert rec.two_sided_p_value(np.array([-1.0, -2.0, -3.0])) == 0
    assert rec.two_sided_p_value(np.array([-1.0, 1.0])) == 1
    unavailable = BootstrapDelta("regret", np.nan, np.nan, np.nan, 0, 0, False)
    assert np.isnan(unavailable.two_sided_p_value())
    with pytest.raises(ValueError, match="align"):
        multiseed_paired_bootstrap([[1]], [[2]])
    with pytest.raises(ValueError, match="shape"):
        multiseed_paired_bootstrap([[1]] * 3, [[1, 2]] * 3)
    assert holm_bonferroni({}) == {}
    corrected = holm_bonferroni({"measured": 0.001, "unavailable": np.nan})
    assert corrected["measured"].rejected
    assert not corrected["unavailable"].rejected
