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
