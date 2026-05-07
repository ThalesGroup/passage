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

import passage.generate as gen
from passage.obstacles import sample_n_from_config


def test_sample_n_from_nested_config():
    cfg = {
        "shape": {
            "n": {
                "min": 2.0,
                "max": 4.0,
                "distribution": {"type": "uniform"},
            }
        }
    }
    rng = np.random.default_rng(0)
    val = sample_n_from_config(cfg, rng)
    assert 2.0 <= val <= 4.0


def test_sample_n_from_legacy_config():
    cfg = {"n_min": 1.0, "n_max": 3.0, "n_distribution": {"type": "uniform"}}
    rng = np.random.default_rng(1)
    val = sample_n_from_config(cfg, rng)
    assert 1.0 <= val <= 3.0


def test_resolve_obstacle_config_nested():
    cfg = {
        "obstacles": {
            "enabled": True,
            "ratio_max": 0.25,
            "shape": {"size": {"min_ratio": 0.1, "max_ratio": 0.2}},
            "generation": {"max_attempts": 5, "candidates_per_mask": 10},
            "obstacle_penalty": 123.0,
        }
    }
    resolved = gen._resolve_obstacle_config(cfg)
    assert resolved["enabled"] is True
    assert resolved["ratio_max"] == 0.25
    assert resolved["size_min_ratio"] == 0.1
    assert resolved["size_max_ratio"] == 0.2
    assert resolved["obstacle_penalty"] == 123.0


def test_generate_obstacles_mask_respects_markers():
    rng = np.random.default_rng(42)
    shape = (32, 32)
    marker_positions = [(10, 10), (20, 20)]
    obstacles_cfg = {"shape": {"n": {"min": 2.0, "max": 3.0}}, "generation": {"max_attempts": 3}}

    mask, ratio = gen._generate_obstacles_mask(
        shape,
        ratio_max=0.2,
        size_min_ratio=0.1,
        size_max_ratio=0.2,
        obstacles_cfg=obstacles_cfg,
        marker_positions=marker_positions,
        rng=rng,
    )

    for i, j in marker_positions:
        assert mask[i, j] == 0
    assert 0.0 <= ratio <= 0.2
