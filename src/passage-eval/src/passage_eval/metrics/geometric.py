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

"""Geometric-fidelity family: ASSD and percentile Hausdorff.

Headline:
    path_assd        : average symmetric surface distance, normalised by
                       the oracle path length, averaged over the batch.
    path_hausdorff_p95 : 95-th-percentile symmetric Hausdorff,
                       normalised by the oracle path length.

Diagnostic:
    path_iou_diagnostic : raw mask-overlap of the rasterised decoded
                       path against the rasterised oracle path. Kept for
                       App. B / App. C reporting; never headline.

Per-sample distances are normalised by the oracle path length so the
metric is commensurate across the seven PASSAGE resolutions
(64x64 to 4096x4096).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial import cKDTree

from passage_eval._typing import PathLike, _as_xy_array


def _per_sample_assd(predicted: np.ndarray, oracle: np.ndarray) -> float:
    """Symmetric average nearest-neighbour distance."""
    if predicted.size == 0 or oracle.size == 0:
        return float("nan")
    tree_p = cKDTree(predicted)
    tree_o = cKDTree(oracle)
    d_po, _ = tree_o.query(predicted, k=1)
    d_op, _ = tree_p.query(oracle, k=1)
    return 0.5 * (float(np.mean(d_po)) + float(np.mean(d_op)))


def _per_sample_hausdorff_p(predicted: np.ndarray, oracle: np.ndarray, percentile: float) -> float:
    if predicted.size == 0 or oracle.size == 0:
        return float("nan")
    tree_p = cKDTree(predicted)
    tree_o = cKDTree(oracle)
    d_po, _ = tree_o.query(predicted, k=1)
    d_op, _ = tree_p.query(oracle, k=1)
    h_p = float(np.percentile(d_po, percentile))
    h_q = float(np.percentile(d_op, percentile))
    return max(h_p, h_q)


def _oracle_length(oracle: np.ndarray) -> float:
    """Polyline length in pixel units; floor at 1.0 to keep normalisation finite."""
    if oracle.shape[0] < 2:
        return 1.0
    diffs = np.diff(oracle, axis=0)
    return max(float(np.sum(np.linalg.norm(diffs, axis=1))), 1.0)


def path_assd(
    predicted_paths: Sequence[PathLike],
    oracle_paths: Sequence[PathLike],
) -> np.ndarray:
    """Per-sample normalised ASSD; ``nan`` for empty paths.

    Returned array has shape ``(N,)`` and units of pixels per unit oracle
    path length. Aggregate with ``np.nanmean`` / ``np.nanmedian`` over a
    batch.
    """
    if len(predicted_paths) != len(oracle_paths):
        raise ValueError(f"Batch length mismatch: {len(predicted_paths)} vs {len(oracle_paths)}.")
    out = np.full(len(predicted_paths), np.nan, dtype=np.float64)
    for i, (pred, orac) in enumerate(zip(predicted_paths, oracle_paths)):
        p = _as_xy_array(pred)
        o = _as_xy_array(orac)
        if p.size == 0 or o.size == 0:
            continue
        d = _per_sample_assd(p, o)
        norm = _oracle_length(o)
        out[i] = d / norm
    return out


def path_hausdorff_p95(
    predicted_paths: Sequence[PathLike],
    oracle_paths: Sequence[PathLike],
    *,
    percentile: float = 95.0,
) -> np.ndarray:
    """Per-sample normalised symmetric percentile-Hausdorff."""
    if len(predicted_paths) != len(oracle_paths):
        raise ValueError(f"Batch length mismatch: {len(predicted_paths)} vs {len(oracle_paths)}.")
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100]; got {percentile}.")
    out = np.full(len(predicted_paths), np.nan, dtype=np.float64)
    for i, (pred, orac) in enumerate(zip(predicted_paths, oracle_paths)):
        p = _as_xy_array(pred)
        o = _as_xy_array(orac)
        if p.size == 0 or o.size == 0:
            continue
        d = _per_sample_hausdorff_p(p, o, percentile)
        norm = _oracle_length(o)
        out[i] = d / norm
    return out


def _rasterise_path(path: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if path.size == 0:
        return mask
    rr = np.clip(np.round(path[:, 0]).astype(int), 0, shape[0] - 1)
    cc = np.clip(np.round(path[:, 1]).astype(int), 0, shape[1] - 1)
    mask[rr, cc] = True
    return mask


def path_iou_diagnostic(
    predicted_paths: Sequence[PathLike],
    oracle_paths: Sequence[PathLike],
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Per-sample raw path-mask IoU on a fixed grid.

    Diagnostic only — drift between path_iou and the cost-regret family
    is what this metric is reported for, not as a promotion signal.
    """
    if len(predicted_paths) != len(oracle_paths):
        raise ValueError(f"Batch length mismatch: {len(predicted_paths)} vs {len(oracle_paths)}.")
    out = np.full(len(predicted_paths), np.nan, dtype=np.float64)
    for i, (pred, orac) in enumerate(zip(predicted_paths, oracle_paths)):
        mp = _rasterise_path(_as_xy_array(pred), grid_shape)
        mo = _rasterise_path(_as_xy_array(orac), grid_shape)
        union = np.logical_or(mp, mo).sum()
        if union == 0:
            continue
        out[i] = float(np.logical_and(mp, mo).sum() / union)
    return out
