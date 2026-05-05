# MIT License - Copyright (c) 2026 THALES
"""Decoder-readiness diagnostics for the cost-to-go regression family.

The most diagnostic of these is `descent_direction_agreement`: it is
the regression analogue of "decoder-readiness". A regression model with
0.99 here is operationally indistinguishable from the oracle to a
greedy decoder, even at high MAE.
"""

from __future__ import annotations

import numpy as np


def _eight_neighbours_offsets() -> np.ndarray:
    """8-neighbour (di, dj) offsets in canonical order."""
    return np.array(
        [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)],
        dtype=np.int64,
    )


def _argmin_neighbour(field: np.ndarray) -> np.ndarray:
    """Return an (H, W) array whose entries name the local 8-argmin index.

    Cells without any in-bounds neighbour return -1.
    """
    h, w = field.shape
    offsets = _eight_neighbours_offsets()
    n_off = offsets.shape[0]
    stack = np.full((n_off, h, w), np.inf, dtype=np.float64)
    for k, (di, dj) in enumerate(offsets):
        i_lo = max(0, di)
        i_hi = h + min(0, di)
        j_lo = max(0, dj)
        j_hi = w + min(0, dj)
        # Source rectangle in the *neighbour* grid we project from
        si_lo = i_lo - di
        si_hi = i_hi - di
        sj_lo = j_lo - dj
        sj_hi = j_hi - dj
        stack[k, i_lo:i_hi, j_lo:j_hi] = field[si_lo:si_hi, sj_lo:sj_hi]
    arg = np.argmin(stack, axis=0)
    valid = np.any(np.isfinite(stack), axis=0)
    arg = np.where(valid, arg, -1)
    return arg.astype(np.int64)


def descent_direction_agreement(
    predicted_field: np.ndarray,
    oracle_field: np.ndarray,
    *,
    reachable_mask: np.ndarray | None = None,
    obstacles: np.ndarray | None = None,
    goal: tuple[int, int] | None = None,
) -> float:
    """Fraction of cells where the local 8-argmin of the predicted field
    matches the oracle. Restricted to reachable, non-goal, non-obstacle cells.

    A regression model with `descent_direction_agreement = 0.99` produces
    oracle-equivalent paths under a greedy decoder regardless of MAE.
    """
    pred = np.asarray(predicted_field, dtype=np.float64)
    orac = np.asarray(oracle_field, dtype=np.float64)
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs oracle {orac.shape}.")
    h, w = pred.shape
    arg_pred = _argmin_neighbour(pred)
    arg_orac = _argmin_neighbour(orac)
    valid = (arg_pred >= 0) & (arg_orac >= 0)
    if reachable_mask is not None:
        valid &= np.asarray(reachable_mask).astype(bool)
    if obstacles is not None:
        valid &= ~np.asarray(obstacles).astype(bool)
    if goal is not None:
        gi, gj = int(goal[0]), int(goal[1])
        if 0 <= gi < h and 0 <= gj < w:
            valid[gi, gj] = False
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan")
    return float((arg_pred[valid] == arg_orac[valid]).mean())


def bellman_residual_reachable(
    predicted_field: np.ndarray,
    cost_grid: np.ndarray,
    *,
    reachable_mask: np.ndarray | None = None,
    goal: tuple[int, int] | None = None,
    eps: float = 1e-12,
) -> float:
    """Mean per-cell Bellman residual on reachable non-goal cells, normalised.

    For each reachable non-goal cell `s`,
        residual(s) = | V_hat(s) - min_n ( c(s, n) + V_hat(n) ) |
    Normalised by ``mean(|V_hat|)`` over the reachable subset.
    """
    pred = np.asarray(predicted_field, dtype=np.float64)
    cost = np.asarray(cost_grid, dtype=np.float64)
    if pred.shape != cost.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs cost {cost.shape}.")
    h, w = pred.shape
    offsets = _eight_neighbours_offsets()
    stack = np.full((offsets.shape[0], h, w), np.inf, dtype=np.float64)
    for k, (di, dj) in enumerate(offsets):
        # step length in pixels (1 for axis-aligned, sqrt(2) for diagonals)
        step = np.sqrt(di * di + dj * dj)
        i_lo = max(0, di)
        i_hi = h + min(0, di)
        j_lo = max(0, dj)
        j_hi = w + min(0, dj)
        si_lo = i_lo - di
        si_hi = i_hi - di
        sj_lo = j_lo - dj
        sj_hi = j_hi - dj
        # edge cost = cost(neighbour) * step ; +1 form to avoid zero division at flat regions
        edge_cost = cost[si_lo:si_hi, sj_lo:sj_hi] * step
        stack[k, i_lo:i_hi, j_lo:j_hi] = pred[si_lo:si_hi, sj_lo:sj_hi] + edge_cost
    rhs = np.min(stack, axis=0)
    res = np.abs(pred - rhs)
    valid = np.isfinite(rhs)
    if reachable_mask is not None:
        valid &= np.asarray(reachable_mask).astype(bool)
    if goal is not None:
        gi, gj = int(goal[0]), int(goal[1])
        if 0 <= gi < h and 0 <= gj < w:
            valid[gi, gj] = False
    n_valid = int(valid.sum())
    if n_valid == 0:
        return float("nan")
    norm = max(float(np.mean(np.abs(pred[valid]))), eps)
    return float(np.mean(res[valid]) / norm)


def local_ordinal_concordance(
    predicted_field: np.ndarray,
    oracle_field: np.ndarray,
    *,
    reachable_mask: np.ndarray | None = None,
    margin: float = 1e-3,
) -> float:
    """Fraction of neighbour pairs with non-trivial oracle margin where the
    predicted field ranks them the same as the oracle.

    A scale-invariant ordering check: complements MAE — MAE-good /
    ordinal-bad is exactly the failure mode the regression primer warns
    about.
    """
    pred = np.asarray(predicted_field, dtype=np.float64)
    orac = np.asarray(oracle_field, dtype=np.float64)
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs oracle {orac.shape}.")
    if reachable_mask is not None:
        mask = np.asarray(reachable_mask).astype(bool)
    else:
        mask = np.ones_like(pred, dtype=bool)

    h, w = pred.shape
    offsets = _eight_neighbours_offsets()
    n_pairs = 0
    n_concord = 0
    for di, dj in offsets:
        i_lo = max(0, di)
        i_hi = h + min(0, di)
        j_lo = max(0, dj)
        j_hi = w + min(0, dj)
        if i_hi <= i_lo or j_hi <= j_lo:
            continue
        si_lo = i_lo - di
        si_hi = i_hi - di
        sj_lo = j_lo - dj
        sj_hi = j_hi - dj
        m_a = mask[si_lo:si_hi, sj_lo:sj_hi]
        m_b = mask[i_lo:i_hi, j_lo:j_hi]
        joint = m_a & m_b
        if not bool(joint.any()):
            continue
        d_orac = orac[si_lo:si_hi, sj_lo:sj_hi] - orac[i_lo:i_hi, j_lo:j_hi]
        d_pred = pred[si_lo:si_hi, sj_lo:sj_hi] - pred[i_lo:i_hi, j_lo:j_hi]
        sig = (np.abs(d_orac) > margin) & joint
        if not bool(sig.any()):
            continue
        agree = np.sign(d_orac[sig]) == np.sign(d_pred[sig])
        n_pairs += int(sig.sum())
        n_concord += int(agree.sum())
    if n_pairs == 0:
        return float("nan")
    return n_concord / n_pairs


def field_ssi_mae(
    predicted_field: np.ndarray,
    oracle_field: np.ndarray,
    *,
    reachable_mask: np.ndarray | None = None,
    eps: float = 1e-12,
) -> float:
    """Scale-and-shift-invariant MAE.

    Solves for ``a, b = argmin || a * pred + b - oracle ||_1`` via a
    closed-form least-squares fit, then reports the L1 residual on the
    fitted scale. Cross-cost-model comparable; invariant to the
    per-sample V normalisation.
    """
    pred = np.asarray(predicted_field, dtype=np.float64).ravel()
    orac = np.asarray(oracle_field, dtype=np.float64).ravel()
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs oracle {orac.shape}.")
    if reachable_mask is not None:
        mask = np.asarray(reachable_mask).astype(bool).ravel()
        pred = pred[mask]
        orac = orac[mask]
    if pred.size == 0:
        return float("nan")
    var_p = float(np.var(pred))
    if var_p < eps:
        # degenerate predicted field — fall back to mean-shift only.
        b = float(np.mean(orac - pred))
        return float(np.mean(np.abs(pred + b - orac)))
    a = float(np.cov(pred, orac, bias=True)[0, 1] / var_p)
    b = float(np.mean(orac) - a * float(np.mean(pred)))
    return float(np.mean(np.abs(a * pred + b - orac)))
