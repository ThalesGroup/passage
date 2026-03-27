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

from passage.obstacles import sample_n_from_config, sample_angle_from_config


def test_sample_n_within_bounds_and_distribution():
    rng = np.random.default_rng(123)
    cfg = {"shape": {"n": {"min": 2.0, "max": 100.0, "distribution": {"type": "beta", "a": 2.0, "b": 5.0}}}}
    samples = [sample_n_from_config(cfg, rng) for _ in range(1000)]
    assert all(2.0 <= s <= 100.0 for s in samples)

    cfg_uniform = {"shape": {"n": {"min": 2.0, "max": 10.0, "distribution": {"type": "uniform"}}}}
    rng = np.random.default_rng(1)
    samples_u = [sample_n_from_config(cfg_uniform, rng) for _ in range(200)]
    assert all(2.0 <= s <= 10.0 for s in samples_u)

    # Exponential distribution on [min, max]
    cfg_exp = {"shape": {"n": {"min": 2.0, "max": 10.0, "distribution": {"type": "exponential", "lambda": 2.0}}}}
    rng = np.random.default_rng(2)
    samples_e = [sample_n_from_config(cfg_exp, rng) for _ in range(500)]
    assert all(2.0 <= s <= 10.0 for s in samples_e)


def test_sample_angle_units():
    rng = np.random.default_rng(42)
    a = sample_angle_from_config(rng)
    assert 0.0 <= a <= 2 * np.pi
