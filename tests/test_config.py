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

from pathlib import Path

import yaml


def test_config_has_obstacle_section():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    obstacles = cfg.get("obstacles", {})
    enabled = obstacles.get("enabled", None)
    ratio_max = obstacles.get("ratio_max", None)
    generation = obstacles.get("generation", {})

    assert enabled is not None, "config.yaml should define obstacles.enabled"
    assert isinstance(enabled, bool), "obstacles.enabled must be a boolean"

    assert ratio_max is not None, "config.yaml should define obstacles.ratio_max"
    ratio_max = float(ratio_max)
    assert 0.0 <= ratio_max <= 1.0, "obstacles.ratio_max must be between 0 and 1"

    # shape subsection
    shape = obstacles.get("shape", {})
    size_cfg = shape.get("size", {})
    size_min_ratio = size_cfg.get("min_ratio", None)
    size_max_ratio = size_cfg.get("max_ratio", None)
    assert size_min_ratio is not None, "config.yaml should define obstacles.shape.size.min_ratio"
    assert size_max_ratio is not None, "config.yaml should define obstacles.shape.size.max_ratio"
    size_min_ratio = float(size_min_ratio)
    size_max_ratio = float(size_max_ratio)
    assert 0.0 < size_min_ratio <= size_max_ratio <= 1.0, "obstacle size ratios must be within (0, 1]"

    # generation params
    assert isinstance(generation, dict), "obstacles.generation must be a mapping"
    candidates_per_mask = generation.get("candidates_per_mask", None)
    max_attempts = generation.get("max_attempts", None)

    assert candidates_per_mask is not None and int(candidates_per_mask) > 0, (
        "generation.candidates_per_mask must be > 0"
    )
    assert max_attempts is not None and int(max_attempts) > 0, "generation.max_attempts must be > 0"

    # ensure obstacle penalty present and positive
    obstacle_penalty = obstacles.get("obstacle_penalty", None)
    assert obstacle_penalty is not None and float(obstacle_penalty) > 0.0, "obstacle_penalty must be a positive number"

    # defaults should be reasonably sized
    assert size_min_ratio == 0.02 and size_max_ratio == 0.25, "Default size ratios should be 0.02 and 0.25"

    # Superellipse exponent limits (n)
    n_cfg = shape.get("n", {})
    n_min = n_cfg.get("min", None)
    n_max = n_cfg.get("max", None)
    assert n_min is not None and n_max is not None, "obstacles.shape.n.min/max must be defined"
    n_min = float(n_min)
    n_max = float(n_max)
    assert n_min > 0.0 and n_max >= n_min, "Invalid obstacles.shape.n min/max values"

    # n sampling distribution
    n_distribution = n_cfg.get("distribution", {})
    assert isinstance(n_distribution, dict), "obstacles.shape.n.distribution must be a mapping if present"
    dist_type = str(n_distribution.get("type", "log_uniform")).lower()
    assert dist_type in ("log_uniform", "beta", "uniform", "exponential"), (
        "obstacles.shape.n.distribution.type must be 'log_uniform', 'beta', 'uniform', or 'exponential'"
    )
