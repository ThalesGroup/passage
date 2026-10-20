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

"""Obstacle utilities: superellipse mask generation.

This module implements a small helper to create a binary mask for a rotated
superellipse (Lamé curve) given parameters. It documents the meaning of
``n`` and provides a reliable implementation that accepts angles in degrees
or radians.

Superellipse equation (centered coordinates):
    (|x|/a)^n + (|y|/b)^n <= 1

Notes about ``n``:
  - n = 2.0 -> ellipse (circle when a == b)
  - n < 2 -> "diamond"/pointed shapes (n = 1 -> diamond / rhombus)
  - n > 2 -> squarer shapes; as n -> large the curve approaches a rectangle
  - n must be > 0.0 to produce a closed shape

The function here keeps the implementation small and deterministic so it can
be used in tests and notebooks. Higher-level obstacle-generation routines can
sample ``n`` from the configuration to create varied collections of shapes.
"""

from __future__ import annotations

from typing import Tuple

import math
import numpy as np


def superellipse_mask(
    shape: Tuple[int, int],
    center: Tuple[float, float],
    a: float,
    b: float,
    n: float,
    angle: float = 0.0,
    angle_degrees: bool = True,
) -> np.ndarray:
    """Return a boolean mask of a rotated superellipse.

    Args:
        shape: (rows, cols) of the output mask.
        center: (y, x) center coordinates in pixel space (can be floats).
        a: semi-axis in x-direction (pixels). Must be > 0.
        b: semi-axis in y-direction (pixels). Must be > 0.
        n: superellipse exponent. Must be > 0. Larger values -> closer to rectangle.
        angle: rotation angle in degrees if ``angle_degrees`` is True, otherwise radians.
        angle_degrees: interpret ``angle`` as degrees when True.

    Returns:
        Boolean NumPy array with True inside the superellipse and False elsewhere.

    Raises:
        ValueError: if parameters are invalid (non-positive a/b/n or shape not 2-tuple).
    """
    if len(shape) != 2:
        raise ValueError("shape must be (rows, cols)")
    rows, cols = int(shape[0]), int(shape[1])

    if a <= 0 or b <= 0:
        raise ValueError("semi-axes 'a' and 'b' must be positive numbers")
    if n <= 0:
        raise ValueError("exponent 'n' must be positive")

    if angle_degrees:
        theta = float(np.deg2rad(angle))
    else:
        theta = float(angle)

    cos_a = np.cos(theta)
    sin_a = np.sin(theta)

    # grid of coordinates centered at (center_y, center_x)
    ys = np.arange(rows, dtype=np.float64) - float(center[0])
    xs = np.arange(cols, dtype=np.float64) - float(center[1])
    y_grid, x_grid = np.meshgrid(ys, xs, indexing="ij")

    # rotate coordinates by -theta (so angle is orientation of major axis)
    xr = cos_a * x_grid + sin_a * y_grid
    yr = -sin_a * x_grid + cos_a * y_grid

    with np.errstate(invalid="ignore"):
        val = (np.abs(xr) / float(a)) ** float(n) + (np.abs(yr) / float(b)) ** float(n)

    mask = val <= 1.0
    mask = np.asarray(mask, dtype=bool)
    return mask


def sample_n_from_config(obstacles_cfg: dict, rng: np.random.Generator) -> float:
    """Sample superellipse exponent `n` from configuration.

    Supports the following distribution types in `obstacles_cfg['shape']['n']['distribution']`:
      - 'uniform'      : uniform between n_min and n_max
      - 'log_uniform'  : uniform in log-space between n_min and n_max
      - 'beta'         : beta distribution (a,b) scaled to [n_min, n_max]

    If the nested `shape.n` structure is missing, the function reads
    `obstacles_cfg['n_min']`, `obstacles_cfg['n_max']`, and `obstacles_cfg['n_distribution']`.
    """
    shape_cfg = obstacles_cfg.get("shape", {}) or {}
    n_cfg = shape_cfg.get("n", {}) or {}
    if not n_cfg:
        # Backward compatibility
        n_cfg = {
            "min": obstacles_cfg.get("n_min", 2.0),
            "max": obstacles_cfg.get("n_max", 100.0),
            "distribution": obstacles_cfg.get("n_distribution", {}) or {},
        }

    n_min = float(n_cfg.get("min", 2.0))
    n_max = float(n_cfg.get("max", 100.0))
    if n_min <= 0 or n_max < n_min:
        raise ValueError("Invalid n_min/n_max in obstacles config")

    dist_cfg = n_cfg.get("distribution", {}) or {}
    dist_type = str(dist_cfg.get("type", "log_uniform")).lower()

    if dist_type == "uniform":
        return float(rng.uniform(n_min, n_max))
    if dist_type == "log_uniform":
        return float(np.exp(rng.uniform(np.log(n_min), np.log(n_max))))
    # Exponential distribution on [n_min, n_max] (parameterized by rate 'lambda' > 0).
    # Sampling is performed on a truncated exponential in [0,1] and then scaled to [n_min,n_max].
    if dist_type == "exponential":
        lam = float(dist_cfg.get("lambda", dist_cfg.get("rate", 1.0)))
        if lam <= 0:
            # Invalid parameter -> fallback to uniform
            return float(rng.uniform(n_min, n_max))
        # Sample u ~ Uniform(0,1) and invert truncated exponential CDF on [0,1]:
        # CDF_trunc(s) = (1 - exp(-lam * s)) / (1 - exp(-lam)), s in [0,1]
        u = float(rng.random())
        c = 1.0 - math.exp(-lam)
        s = -(1.0 / lam) * math.log(1.0 - u * c)
        s = min(max(s, 0.0), 1.0)
        return float(n_min + s * (n_max - n_min))
    # beta distribution (default)
    if dist_type == "beta":
        a = float(dist_cfg.get("a", 2.0))
        b = float(dist_cfg.get("b", 5.0))
        # clamp to sensible positive values
        a = max(0.01, a)
        b = max(0.01, b)
        u = float(rng.beta(a, b))
        return float(n_min + u * (n_max - n_min))

    # fallback to uniform
    return float(rng.uniform(n_min, n_max))


def sample_angle_from_config(rng: np.random.Generator) -> float:
    """Return an angle in radians uniformly sampled from [0, 2*pi).

    Angles are not configurable; they are always sampled uniformly in radians.
    """
    return float(rng.uniform(0.0, 2 * np.pi))
