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
    astar_expansions_ratio,
    compute_primary_metrics,
    decoder_recovery_rate,
    paired_bootstrap_ci,
)


def _line(start, end, n):
    a = np.linspace(start[0], end[0], n)
    b = np.linspace(start[1], end[1], n)
    return np.stack([a, b], axis=1)


def test_compute_primary_metrics_returns_full_bundle():
    p = _line((0, 0), (9, 9), 10).astype(int)
    obs = np.zeros((10, 10), dtype=bool)
    out = compute_primary_metrics(
        predicted_paths=[p],
        oracle_paths=[p],
        predicted_costs=np.array([100.0]),
        oracle_costs=np.array([100.0]),
        obstacles=[obs],
        starts=[(0, 0)],
        goals=[(9, 9)],
    )
    assert out["cost_regret_median"] == 0.0
    assert out["cost_regret_p90"] == 0.0
    assert out["path_assd"] == 0.0
    assert out["path_hausdorff_p95"] == 0.0
    assert out["success_rate"] == 1.0
    assert out["obstacle_violation_rate"] == 0.0


def test_paired_bootstrap_ci_contains_zero_when_paired_identical():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)
    out = paired_bootstrap_ci(
        lambda x, y: float((x - y).mean()),
        a,
        a,
        n_bootstrap=200,
        seed=0,
    )
    assert out["point"] == 0.0
    assert out["ci_low"] <= 0.0 <= out["ci_high"]


def test_astar_expansions_ratio_summary():
    pred = np.array([10, 20, 100])
    orac = np.array([100, 100, 100])
    s = astar_expansions_ratio(pred, orac)
    assert s["median"] == 0.2
    assert s["n_valid"] == 3


def test_decoder_recovery_rate_only_counts_successes_without_fallback():
    succ = np.array([True, True, True, False])
    fb = np.array([False, True, False, False])
    out = decoder_recovery_rate(succ, fb)
    # 2 of 4 samples succeeded without triggering the fallback
    assert out == 0.5
