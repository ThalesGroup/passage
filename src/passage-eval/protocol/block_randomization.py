"""Blockwise paired-label randomization for a prospective six-metric comparison.

For each metric, this test is valid only under that metric's stated blockwise
exchangeability null: its paired method outcomes can be independently exchanged
for entire geographic blocks, simultaneously across all three seeds. Geometry
also requires exchangeability of the paired success/validity indicators. It is not a general test of
equal means or quantiles. Geography, exchangeability, input provenance and
preregistration are scientific assumptions the caller must establish; merely
running this function does not establish them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product

import numpy as np
from passage_eval.protocol.discipline import DEFAULT_SEEDS, PRIMARY_METRICS, _block_groups, compare_primary_metrics

ASSUMPTIONS = (
    "For each tested metric's null, its paired outcomes across all three seeds are invariant to "
    "independently swapping method labels within each entire geographic block; geometry also "
    "requires exchangeability of the paired validity indicators. The same swap is used for all seeds; "
    "tasks, budgets and trained checkpoints are matched and the block definitions are justified. "
    "This exchangeability null is stronger than equality of a mean or quantile."
)


def _pair(a, b, name, swapped):
    left, right = np.asarray(a[name], dtype=np.float64), np.asarray(b[name], dtype=np.float64)
    return np.where(swapped, right, left), np.where(swapped, left, right)


def _geometry(left, right, valid_a, valid_b):
    if not valid_a.any() or not valid_b.any():
        return float("nan")
    return float(np.mean(left[valid_a]) - np.mean(right[valid_b]))


def _contrasts(a_by_seed: Mapping, b_by_seed: Mapping, swapped: np.ndarray) -> np.ndarray:
    results = []
    for seed in DEFAULT_SEEDS:
        a, b = a_by_seed[seed], b_by_seed[seed]

        regret_a, regret_b = _pair(a, b, "regret", swapped)
        success_a, success_b = _pair(a, b, "success", swapped)
        obstacle_a, obstacle_b = _pair(a, b, "obstacle_violation", swapped)
        valid_a, valid_b = _pair(a, b, "valid_path", swapped)
        assd_a, assd_b = _pair(a, b, "assd", swapped)
        h95_a, h95_b = _pair(a, b, "h95", swapped)
        valid_a, valid_b = valid_a.astype(bool), valid_b.astype(bool)

        results.append(
            [
                float(np.median(regret_a) - np.median(regret_b)),
                float(np.quantile(regret_a, 0.9) - np.quantile(regret_b, 0.9)),
                _geometry(assd_a, assd_b, valid_a, valid_b),
                _geometry(h95_a, h95_b, valid_a, valid_b),
                float(np.mean(success_a) - np.mean(success_b)),
                float(np.mean(obstacle_a) - np.mean(obstacle_b)),
            ]
        )
    return np.mean(np.asarray(results, dtype=np.float64), axis=0)


def block_randomization_p_values(
    a_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    b_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    *,
    blocks: Sequence[str] | np.ndarray,
    method: str = "monte_carlo",
    permutations: int = 10_000,
    seed: int = 1234,
) -> dict:
    """Return six two-sided block-swap p-values, conditional on exchangeability.

    ``exact`` enumerates all block swaps, capped at 16 blocks. ``monte_carlo``
    draws uniform independent swaps with replacement and adds the observed
    identity transformation to the numerator and denominator. Each draw uses
    one shared block swap across seeds and all metrics. The test statistic is
    the absolute three-seed mean of the per-method aggregate differences; the
    median and p90 are differences of aggregates, not medians of differences.
    Ties count as at least as extreme, with a small conservative rounding
    tolerance. Geometry must be defined over the *entire* block-swap orbit,
    including swaps not sampled by Monte Carlo. Otherwise its p-value is NaN
    and must remain unavailable to Holm.

    Before final test access, freeze the method, permutation count, seed,
    independent block assignment, null and comparison family. Evaluate whether
    the exchangeability null is justified; it is not implied by matched tasks.
    The caller may pass these p-values, test name and assumptions into
    ``compare_primary_metrics`` for its separate shared-draw bootstrap and Holm.
    """
    if blocks is None:
        raise ValueError("Known geographic blocks are required")
    if method not in {"exact", "monte_carlo"}:
        raise ValueError("method must be exact or monte_carlo")
    if type(permutations) is not int or permutations < 1:
        raise ValueError("permutations must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # Reuse the public comparison's strict roster, seed, metric and block checks.
    checked = compare_primary_metrics(a_by_seed, b_by_seed, blocks=blocks, n_bootstrap=1, seed=seed)
    groups = _block_groups(blocks, len(checked.task_ids))
    if groups is None:
        raise ValueError("Known geographic blocks are required")
    if method == "exact" and len(groups) > 16:
        raise ValueError("Exact enumeration is capped at 16 geographic blocks")
    labels = np.full(len(checked.task_ids), -1, dtype=np.int64)
    for index, indices in enumerate(groups):
        if np.any(labels[indices] != -1):
            raise ValueError("Geographic blocks must not overlap")
        labels[indices] = index
    if np.any(labels < 0):
        raise ValueError("Every task must belong to a geographic block")
    observed = _contrasts(a_by_seed, b_by_seed, np.zeros(len(labels), dtype=bool))
    extreme = np.zeros(len(PRIMARY_METRICS), dtype=np.int64)
    defined = np.isfinite(observed)
    # The minimum successful count attainable by either method over the whole
    # swap orbit is the sum of each block's smaller method-specific count.
    # This check cannot rely on whether Monte Carlo happened to sample a
    # problematic swap.
    geometry_defined = all(
        sum(
            min(
                np.count_nonzero(np.asarray(a_by_seed[training_seed]["valid_path"])[indices]),
                np.count_nonzero(np.asarray(b_by_seed[training_seed]["valid_path"])[indices]),
            )
            for indices in groups
        )
        > 0
        for training_seed in DEFAULT_SEEDS
    )
    if not geometry_defined:
        defined[2:4] = False
    tolerance = 1e-12 * np.maximum(1.0, np.abs(observed))
    rng = np.random.default_rng(seed)
    swaps = (
        product((False, True), repeat=len(groups))
        if method == "exact"
        else (rng.integers(0, 2, size=len(groups), dtype=np.int8).astype(bool) for _ in range(permutations))
    )
    draws = 0
    for group_swap in swaps:
        sample_swap = np.asarray(group_swap, dtype=bool)[labels]
        candidate = _contrasts(a_by_seed, b_by_seed, sample_swap)
        finite = np.isfinite(candidate)
        defined &= finite
        extreme += finite & (np.abs(candidate) >= np.abs(observed) - tolerance)
        draws += 1
    numerator = extreme + (1 if method == "monte_carlo" else 0)
    denominator = draws + (1 if method == "monte_carlo" else 0)
    values = np.where(defined, numerator / denominator, np.nan)
    names = [name for name, _, _ in PRIMARY_METRICS]
    return {
        "schema": "passage-paired-block-randomization/v1",
        "method": method,
        "status": "assumptions_not_verified",
        "assumptions": ASSUMPTIONS,
        "permutations": draws,
        "identity_added": method == "monte_carlo",
        "random_seed": seed,
        "number_of_blocks": len(groups),
        "geometry_defined_on_complete_orbit": geometry_defined,
        "task_ids": list(checked.task_ids),
        "blocks": list(blocks),
        "p_values": {name: float(value) for name, value in zip(names, values, strict=True)},
    }
