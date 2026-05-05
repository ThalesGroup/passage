# MIT License
#
# Copyright (c) 2026 THALES

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
    p = _line((0, 0), (10, 10), 50)
    q = p + np.array([1.0, 0.0])
    out = path_assd([q], [p])
    # ASSD = 1 px; oracle length ~ 10*sqrt(2) ~ 14.14
    assert 0.05 < float(out[0]) < 0.10


def test_path_iou_diagnostic_is_one_for_identical_paths():
    p = _line((0, 0), (10, 10), 50).astype(int)
    out = path_iou_diagnostic([p], [p], grid_shape=(16, 16))
    np.testing.assert_allclose(out, 1.0)
