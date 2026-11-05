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

"""Tests for the four trustworthiness axes (§5.4 / App. K)."""

from __future__ import annotations

import numpy as np
import pytest

from passage_eval import (
    geographic_generalisation,
    odd_boundary_delta,
    oracle_relative_efficiency,
    stability_cv,
)


def test_stability_cv_below_threshold_does_not_flag():
    # Three near-identical seeds -> tiny CV, axis does NOT fire.
    axis = stability_cv([0.143, 0.142, 0.145])
    assert axis.name == "Stab"
    assert not axis.flag
    assert axis.value < 0.25
    assert axis.n_samples == 3


def test_stability_cv_above_threshold_fires():
    axis = stability_cv([0.1, 0.5, 1.0], cv_threshold=0.25)
    assert axis.flag is True
    assert axis.value > 0.25


def test_stability_cv_handles_nan():
    axis = stability_cv([float("nan"), float("nan")])
    assert axis.n_samples == 0
    assert np.isnan(axis.value)


def test_geographic_generalisation_smaller_is_better_regression():
    # Hold-out is meaningfully *worse* than matched random for a smaller-is-better metric.
    holdout = np.array([0.18, 0.20, 0.22])
    random = np.array([0.10, 0.11, 0.12])
    axis = geographic_generalisation(holdout, random, higher_is_better=False)
    assert axis.flag is True  # hold-out worse -> axis fires
    assert axis.value > 0.0
    assert axis.extras["holdout_mean"] > axis.extras["matched_random_mean"]


def test_geographic_generalisation_no_drift_does_not_flag():
    arr = np.array([0.10, 0.11, 0.12])
    axis = geographic_generalisation(arr, arr, higher_is_better=False)
    assert axis.flag is False
    assert axis.value == pytest.approx(0.0, abs=1e-12)


def test_odd_boundary_top_decile_picks_envelope_edge():
    # 100 samples; top-decile stratifier corresponds to *worse* metric -> axis fires.
    metric = np.linspace(0.05, 0.50, 100)
    stratifier = metric.copy()  # bigger stratifier == bigger metric == worse for the planner
    axis = odd_boundary_delta(metric, stratifier, top_quantile=0.10, higher_is_better=False)
    assert axis.name == "ODDe"
    assert axis.flag is True
    assert axis.value > 0.0
    assert axis.extras["boundary_mean"] > axis.extras["full_mean"]


def test_odd_boundary_no_drift_does_not_flag():
    metric = np.full(100, 0.10)
    stratifier = np.linspace(0, 1, 100)
    axis = odd_boundary_delta(metric, stratifier, top_quantile=0.10)
    assert axis.flag is False
    assert axis.value == pytest.approx(0.0, abs=1e-12)


def test_oracle_relative_efficiency_speedup_above_threshold_does_not_flag():
    # Planner: 2 ms median; oracle: 50 ms median -> 25x speedup.
    axis = oracle_relative_efficiency([1.5, 2.0, 2.5, 3.0], [40.0, 50.0, 60.0, 55.0])
    # Median of [1.5, 2.0, 2.5, 3.0] = 2.25; median of [40, 50, 55, 60] = 52.5.
    assert axis.value == pytest.approx(52.5 / 2.25, rel=1e-6)
    assert axis.flag is False  # 22x > 1


def test_oracle_relative_efficiency_below_threshold_fires():
    axis = oracle_relative_efficiency(planner_latency_ms_per_sample=100.0, oracle_latency_ms_per_sample=10.0)
    assert axis.value == pytest.approx(0.10, rel=1e-6)
    assert axis.flag is True
