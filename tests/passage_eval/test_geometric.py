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

from passage_eval import path_assd, path_hausdorff_p95, path_iou_diagnostic


def _line(start, end, n):
    a = np.linspace(start[0], end[0], n)
    b = np.linspace(start[1], end[1], n)
    return np.stack([a, b], axis=1)


def test_path_assd_is_zero_for_identical_paths():
    p = _line((0, 0), (10, 10), 50)
    out = path_assd([p], [p])
    np.testing.assert_allclose(out, 0.0)


def test_path_assd_handles_empty():
    out = path_assd([np.zeros((0, 2))], [_line((0, 0), (10, 10), 50)])
    assert math.isnan(out[0])


def test_path_hausdorff_p95_is_zero_for_identical_paths():
    p = _line((0, 0), (10, 10), 50)
    out = path_hausdorff_p95([p], [p])
    np.testing.assert_allclose(out, 0.0)


def test_path_assd_normalised_by_oracle_length():
    # Predicted shifted by one pixel uniformly along the row axis.
    p = np.array([(0, c) for c in range(11)])
    q = p + np.array([1.0, 0.0])
    out = path_assd([q], [p])
    # ASSD = 1 px, divided by the 11 oracle waypoints (paper App. B).
    np.testing.assert_allclose(out, 1 / 11)


def test_path_iou_diagnostic_is_one_for_identical_paths():
    p = _line((0, 0), (10, 10), 50).astype(int)
    out = path_iou_diagnostic([p], [p], grid_shape=(16, 16))
    np.testing.assert_allclose(out, 1.0)
