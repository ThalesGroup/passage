# MIT License
#
# Copyright (c) 2026 THALES

import math

import numpy as np

from passage_eval import cost_regret, cost_regret_summary


def test_cost_regret_zero_at_oracle_parity():
    pred = np.array([100.0, 200.0, 300.0])
    orac = np.array([100.0, 200.0, 300.0])
    r = cost_regret(pred, orac)
    np.testing.assert_allclose(r, 0.0)


def test_cost_regret_handles_nan_and_zero_oracle():
    pred = np.array([100.0, np.nan, 300.0, 50.0])
    orac = np.array([100.0, 200.0, 0.0, 100.0])
    r = cost_regret(pred, orac)
    assert np.isnan(r[1])
    assert np.isnan(r[2])
    np.testing.assert_allclose(r[0], 0.0)
    np.testing.assert_allclose(r[3], -0.5)


def test_cost_regret_summary_robust_to_outlier():
    # Median is robust; mean is not.
    base = np.zeros(99)
    outlier = np.array([1e6])
    r = np.concatenate([base, outlier])
    s = cost_regret_summary(r)
    assert s.median == 0.0
    assert s.p90 == 0.0
    assert s.mean > 100.0


def test_cost_regret_summary_geomean_definition():
    r = np.array([0.1, 0.2, 0.3])
    s = cost_regret_summary(r)
    expected = math.exp(np.mean(np.log(1.0 + r))) - 1.0
    np.testing.assert_allclose(s.geomean, expected, rtol=1e-9)


def test_cost_regret_summary_empty_returns_nan():
    s = cost_regret_summary(np.array([]))
    assert s.n_valid == 0
    assert math.isnan(s.median)
