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

"""Decoder-readiness diagnostics for the segmentation family.

The most diagnostic of these is `mask_endpoints_in_same_component`: if
it is True for ~99% of validation samples the A*-recovery never had to
bridge cuts, and the mask quality is what carries the decoded path. If
it is True for only 80% then 20% of cells the decoder touched were
structurally inadequate — useful triage signal.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def mask_connected_components_total(mask: np.ndarray) -> int:
    """Number of 8-connected components of ``mask > 0``."""
    structure = np.ones((3, 3), dtype=bool)
    _, n = ndi.label(np.asarray(mask) > 0, structure=structure)
    return int(n)


def mask_endpoints_in_same_component(
    mask: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
) -> bool:
    """Do start and goal cells share an 8-connected component of ``mask>0``?"""
    structure = np.ones((3, 3), dtype=bool)
    labels, _ = ndi.label(np.asarray(mask) > 0, structure=structure)
    si, sj = int(start[0]), int(start[1])
    gi, gj = int(goal[0]), int(goal[1])
    h, w = labels.shape
    if not (0 <= si < h and 0 <= sj < w and 0 <= gi < h and 0 <= gj < w):
        return False
    a = int(labels[si, sj])
    b = int(labels[gi, gj])
    return a != 0 and a == b


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(bool)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    return ndi.binary_dilation(mask, structure=structure)


def boundary_f_score(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    tolerance: int = 1,
) -> float:
    """F-score of boundary agreement at a tolerance of ``tolerance`` pixels.

    Standard segmentation-challenge boundary-F at tolerance ``d`` is
        precision = |pred ∩ dilate(gt, d)| / |pred|
        recall    = |gt   ∩ dilate(pred, d)| / |gt|
        F         = 2 PR / (P + R)
    Returns NaN when both masks are empty.
    """
    pred = np.asarray(pred_mask) > 0
    gt = np.asarray(gt_mask) > 0
    n_pred = int(pred.sum())
    n_gt = int(gt.sum())
    if n_pred == 0 and n_gt == 0:
        return float("nan")
    if n_pred == 0 or n_gt == 0:
        return 0.0
    gt_d = _dilate(gt, tolerance)
    pr_d = _dilate(pred, tolerance)
    precision = float(np.logical_and(pred, gt_d).sum() / n_pred)
    recall = float(np.logical_and(gt, pr_d).sum() / n_gt)
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)
