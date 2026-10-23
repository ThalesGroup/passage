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

"""Statistical-discipline primitives for the PASSAGE evaluation protocol.

The cross-family evaluation protocol of the PASSAGE paper declares
that every headline number is

  (i) a function of the *decoded path* (Rule 1: common anchor),
 (ii) computed under a pre-registered set of primary metrics
      (``PRIMARY_METRICS``),
(iii) aggregated over the seeds ``DEFAULT_SEEDS = (1234, 2026, 3407)``,
 (iv) estimated via shared geographic-block draws of aggregate contrasts,
  (v) corrected across the primary-metric family by Holm–Bonferroni at
      family-wise alpha = 0.05, and
 (vi) reduced to a categorical pairwise decision: Pareto-dominant /
      Trade-off / Partial evidence / No demonstrated difference.

This module is the canonical implementation of points (iii) - (vi). It is
deliberately small and depends only on numpy so a third-party submission
can re-derive the headline decision from its per-sample primary-metric
arrays without re-running its decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

#: Default seeds the PASSAGE paper pre-registers.
DEFAULT_SEEDS: tuple[int, int, int] = (1234, 2026, 3407)


class MetricFamily(str, Enum):
    """Three primary-metric families declared by the paper."""

    REGRET = "regret"
    GEOMETRIC = "geometric"
    RELIABILITY = "reliability"


#: Six primary decoded-path metrics in declared order. ``higher_is_better``
#: tells the decision rule which sign of $\\Delta = m_A - m_B$ favours
#: method $A$.
PRIMARY_METRICS: tuple[tuple[str, MetricFamily, bool], ...] = (
    ("cost_regret_median", MetricFamily.REGRET, False),
    ("cost_regret_p90", MetricFamily.REGRET, False),
    ("path_assd", MetricFamily.GEOMETRIC, False),
    ("path_hausdorff_p95", MetricFamily.GEOMETRIC, False),
    ("success_rate", MetricFamily.RELIABILITY, True),
    ("obstacle_violation_rate", MetricFamily.RELIABILITY, False),
)


# ---------------------------------------------------------------------------
# Censored penalty for failed decodes
# ---------------------------------------------------------------------------


def apply_censored_penalty(
    predicted_costs: Sequence[float] | np.ndarray,
    oracle_costs: Sequence[float] | np.ndarray,
    *,
    decoder_succeeded: Sequence[bool] | np.ndarray | None = None,
    obstacle_violated: Sequence[bool] | np.ndarray | None = None,
    k_factor: float = 1e6,
) -> np.ndarray:
    """Return predicted costs with a $K \\gg c^{\\star}$ penalty on failed decodes.

    The paper defines a censored penalty $c^{\\text{pred}}_i = K$
    "on a decode that fails to connect $s$ to $g$ or that crosses an
    obstacle". The canonical PASSAGE penalty value is
    ``K = k_factor * oracle_cost`` with ``k_factor = 1e6``.

    Parameters
    ----------
    predicted_costs, oracle_costs : array_like
        Per-sample costs in matching order. Pass ``np.nan`` (or the
        already-censored sentinel) for samples whose decode failed - this
        function will rewrite those entries.
    decoder_succeeded : array_like of bool, optional
        ``True`` iff the canonical decoder produced an $(s\\to g)$ path.
        Mutually consistent with ``success_rate``.
    obstacle_violated : array_like of bool, optional
        ``True`` iff the decoded path crosses an obstacle pixel.
    k_factor : float
        Multiplier on the oracle cost used as the censored penalty.

    Returns
    -------
    np.ndarray of shape ``(N,)``
        Predicted costs with failed decodes rewritten to
        ``k_factor * oracle_cost``. Entries with non-finite oracle cost
        remain non-finite.
    """
    pred = np.asarray(predicted_costs, dtype=np.float64).copy()
    orac = np.asarray(oracle_costs, dtype=np.float64)
    if pred.shape != orac.shape:
        raise ValueError(f"Shape mismatch: predicted {pred.shape} vs oracle {orac.shape}.")
    n = pred.shape[0]
    failed = ~np.isfinite(pred)
    if decoder_succeeded is not None:
        succ = np.asarray(decoder_succeeded, dtype=bool)
        if succ.shape[0] != n:
            raise ValueError("decoder_succeeded must align with predicted_costs.")
        failed |= ~succ
    if obstacle_violated is not None:
        viol = np.asarray(obstacle_violated, dtype=bool)
        if viol.shape[0] != n:
            raise ValueError("obstacle_violated must align with predicted_costs.")
        failed |= viol
    safe_oracle = np.where(np.isfinite(orac), orac, 0.0)
    pred = np.where(failed & np.isfinite(orac), k_factor * safe_oracle, pred)
    return pred


# ---------------------------------------------------------------------------
# Paired bootstrap on differences of aggregates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapDelta:
    """Paired-bootstrap result on a single primary metric.

    ``delta`` estimates ``T(A) - T(B)``. ``n`` counts the frozen task roster
    (summed across seeds for a seed mean); ``n_a`` and ``n_b`` count each
    method's finite outcomes. ``n_bootstrap`` counts shared draws, not seeds
    times draws. An undefined draw makes the interval unavailable (NaN).
    """

    metric: str
    delta: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int
    higher_is_better: bool
    n_a: int | None = None
    n_b: int | None = None
    n_blocks: int | None = None
    per_seed: Mapping[int, BootstrapDelta] | None = None

    @property
    def excludes_zero(self) -> bool:
        """``True`` iff zero is outside the CI - i.e. a directional difference."""
        return self.ci_low > 0.0 or self.ci_high < 0.0

    @property
    def favours_a(self) -> bool:
        """``True`` iff the directional CI favours method ``A``.

        Combines ``excludes_zero`` with the ``higher_is_better`` flag of
        the metric: e.g. for ``cost_regret_median`` (lower is better),
        $\\Delta < 0$ means $A$ is better than $B$.
        """
        if not self.excludes_zero:
            return False
        if self.higher_is_better:
            return self.ci_low > 0.0
        return self.ci_high < 0.0

    def two_sided_p_value(self, draws: np.ndarray | None = None) -> float:
        """Legacy descriptive tail proxy; not a declared confirmatory test.

        When ``draws`` (the resampled $\\Delta$ trajectory) is supplied the
        p-value is the empirical fraction of draws on the wrong side of
        zero, doubled for two-sided. With no draws available we fall back
        to a CI-coverage proxy: ``2 * Phi(-|delta| / se)`` where ``se`` is
        approximated from the symmetric 95% CI half-width. Neither proxy
        specifies a null-resampling test or its assumptions. The canonical
        comparison requires separately supplied, declared paired-test p-values.
        """
        if draws is not None:
            d = np.asarray(draws, dtype=np.float64)
            if d.ndim != 1:
                raise ValueError("Bootstrap draws must be one-dimensional")
            if d.size == 0 or not np.isfinite(d).all():
                return float("nan")
            # Empirical two-sided p: 2 * min(P(Δ <= 0), P(Δ >= 0)).
            p_lo = float(np.mean(d <= 0.0))
            p_hi = float(np.mean(d >= 0.0))
            return float(min(1.0, 2.0 * min(p_lo, p_hi)))
        # Fallback: convert the central CI to an approximate normal p-value.
        half = 0.5 * (self.ci_high - self.ci_low)
        if not np.isfinite(half) or half <= 0.0:
            return float("nan")
        # 95% normal CI half-width ~= 1.96 * se.
        se = half / 1.959964
        if se <= 0.0:
            return float("nan")
        from math import erfc, sqrt

        z = abs(self.delta) / se
        return float(erfc(z / sqrt(2.0)))


def _validate_bootstrap(n_bootstrap: int, alpha: float, nan_policy: str, seed: int) -> None:
    if isinstance(n_bootstrap, bool) or not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if nan_policy not in {"raise", "omit"}:
        raise ValueError("nan_policy must be 'raise' or 'omit'")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("The resampling seed must be a fixed nonnegative integer")


def _block_groups(blocks: Sequence[str] | np.ndarray | None, n: int) -> list[np.ndarray] | None:
    if blocks is None:
        return None
    labels = np.asarray(blocks, dtype=object)
    if labels.shape != (n,) or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise ValueError("Geographic blocks must be nonempty strings aligned with the task roster")
    names = list(dict.fromkeys(labels))
    if len(names) < 2:
        raise ValueError("At least two geographic blocks are required to estimate geographic uncertainty")
    return [np.flatnonzero(labels == name) for name in names]


def _paired_arrays(a, b, nan_policy: str) -> tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired shape mismatch: {a.shape} vs {b.shape}.")
    if a.ndim != 1:
        raise ValueError("Paired arrays must be one-dimensional")
    if (
        np.isinf(a).any()
        or np.isinf(b).any()
        or (nan_policy == "raise" and (not np.isfinite(a).all() or not np.isfinite(b).all()))
    ):
        raise ValueError("Paired arrays must be finite; use nan_policy='omit' for each method's defined subset")
    return a, b


def _contrast(a: np.ndarray, b: np.ndarray, statistic: Callable[[np.ndarray], float]) -> float:
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if not a.size or not b.size:
        return float("nan")
    value = float(statistic(a)) - float(statistic(b))
    if not np.isfinite(value):
        raise ValueError("statistic must return a finite scalar on finite nonempty arrays")
    return value


def _delta_record(metric, point, draws, n, higher_is_better, a, b, groups, alpha, *, per_seed=None):
    lo = hi = float("nan")
    # Do not condition on the subset of defined draws by silently dropping NaNs.
    if np.isfinite(point) and draws.size and np.isfinite(draws).all():
        lo, hi = (float(value) for value in np.quantile(draws, [alpha / 2, 1 - alpha / 2]))
    return BootstrapDelta(
        metric,
        float(point),
        lo,
        hi,
        n,
        len(draws),
        higher_is_better,
        n_a=int(np.isfinite(a).sum()),
        n_b=int(np.isfinite(b).sum()),
        n_blocks=len(groups) if groups is not None else None,
        per_seed=per_seed,
    )


def _shared_bootstrap(a_arrays, b_arrays, statistic, n_bootstrap, rng, groups):
    """Compute unsorted seed trajectories using the same complete task draw."""
    n = len(a_arrays[0])
    if not n:
        return np.empty((len(a_arrays), 0), dtype=np.float64)
    draws = np.empty((len(a_arrays), n_bootstrap), dtype=np.float64)
    for index in range(n_bootstrap):
        if groups is None:
            rows = rng.integers(0, n, size=n)
        else:
            selected = rng.integers(0, len(groups), size=len(groups))
            rows = np.concatenate([groups[group] for group in selected])
        for seed_index, (a, b) in enumerate(zip(a_arrays, b_arrays)):
            draws[seed_index, index] = _contrast(a[rows], b[rows], statistic)
    return draws


def paired_bootstrap_delta(
    per_sample_a: np.ndarray,
    per_sample_b: np.ndarray,
    *,
    metric: str = "metric",
    higher_is_better: bool = False,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEEDS[0],
    return_draws: bool = False,
    blocks: Sequence[str] | np.ndarray | None = None,
    nan_policy: Literal["raise", "omit"] = "raise",
) -> BootstrapDelta | tuple[BootstrapDelta, np.ndarray]:
    """Paired bootstrap on ``T(A) - T(B)``, preserving draw order.

    Pairs are anchored on the same sample index: the pairing cancels
    sample-difficulty variance because both methods are evaluated on the
    same indices. With ``blocks``, each draw samples the observed number
    of whole geographic blocks with replacement, retaining task weights.
    Without blocks this lower-level helper uses independent task resampling.

    For ``cost_regret_median`` and friends pass ``statistic=np.median``.
    The mean default agrees with the historical mean of paired differences.
    Nonlinear reductions intentionally use the difference of aggregates.

    Parameters
    ----------
    per_sample_a, per_sample_b : array_like, shape (N,)
        Per-sample metric values, paired by index.
    statistic : callable
        Reduction applied separately to each method, including inside draws.
    n_bootstrap, alpha, seed
        Bootstrap parameters; ``alpha = 0.05`` is the paper's default.
    return_draws : bool
        If ``True`` also return the ``(n_bootstrap,)`` draw vector for
        descriptive inspection. Ordinary bootstrap tails alone do not define
        the valid paired null test required for confirmatory Holm correction.
    nan_policy : {"raise", "omit"}
        Missing values raise by default. Explicit omission draws the complete
        roster first, then reduces each method's finite subset separately.
        An empty subset in any draw makes its CI unavailable; draws are retained.
    """
    _validate_bootstrap(n_bootstrap, alpha, nan_policy, seed)
    a, b = _paired_arrays(per_sample_a, per_sample_b, nan_policy)
    groups = _block_groups(blocks, len(a))
    draws = _shared_bootstrap([a], [b], statistic, n_bootstrap, np.random.default_rng(seed), groups)[0]
    rec = _delta_record(metric, _contrast(a, b, statistic), draws, len(a), higher_is_better, a, b, groups, alpha)
    if return_draws:
        return rec, draws
    return rec


def multiseed_paired_bootstrap(
    per_sample_a_by_seed: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    per_sample_b_by_seed: Mapping[int, np.ndarray] | Sequence[np.ndarray],
    *,
    metric: str = "metric",
    higher_is_better: bool = False,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap_per_seed: int = 10_000,
    alpha: float = 0.05,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    return_draws: bool = False,
    seed: int = DEFAULT_SEEDS[0],
    blocks: Sequence[str] | np.ndarray | None = None,
    nan_policy: Literal["raise", "omit"] = "raise",
) -> BootstrapDelta | tuple[BootstrapDelta, np.ndarray]:
    """Average seed contrasts *within* each shared task/block bootstrap draw.

    The result contains ``B = n_bootstrap_per_seed`` draws of the seed mean,
    and per-seed estimates/CIs in ``per_seed``. ``seed`` controls geographic
    resampling; ``seeds`` identifies fitted training seeds. This intentionally
    replaces the historical seed-union mixture of ``S * B`` draws.

    Inputs are mappings keyed by exactly ``seeds``, or sequences in that order.
    All arrays must represent the identical ordered task roster. Use
    ``compare_primary_metrics`` to validate explicit task identities and the
    complete preregistered seed/metric family.
    """
    _validate_bootstrap(n_bootstrap_per_seed, alpha, nan_policy, seed)
    if not len(seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and distinct")
    if isinstance(per_sample_a_by_seed, Mapping) != isinstance(per_sample_b_by_seed, Mapping):
        raise ValueError("per-seed arrays must align in the same input representation")
    if isinstance(per_sample_a_by_seed, Mapping):
        if set(per_sample_a_by_seed) != set(seeds) or set(per_sample_b_by_seed) != set(seeds):
            raise ValueError("per-seed mapping keys must align exactly with seeds")
        pairs = [(per_sample_a_by_seed[s], per_sample_b_by_seed[s]) for s in seeds]
    else:
        if len(per_sample_a_by_seed) != len(seeds) or len(per_sample_b_by_seed) != len(seeds):
            raise ValueError("per-seed arrays must align with seeds.")
        pairs = list(zip(per_sample_a_by_seed, per_sample_b_by_seed))
    validated = [_paired_arrays(a, b, nan_policy) for a, b in pairs]
    a_arrays, b_arrays = zip(*validated)
    if any(a.shape != a_arrays[0].shape for a in a_arrays):
        raise ValueError("All seeds must align with the same ordered task roster shape")
    groups = _block_groups(blocks, len(a_arrays[0]))
    trajectories = _shared_bootstrap(
        a_arrays, b_arrays, statistic, n_bootstrap_per_seed, np.random.default_rng(seed), groups
    )
    records = {
        s: _delta_record(metric, _contrast(a, b, statistic), draws, len(a), higher_is_better, a, b, groups, alpha)
        for s, a, b, draws in zip(seeds, a_arrays, b_arrays, trajectories)
    }
    all_draws = np.mean(trajectories, axis=0)
    rec = _delta_record(
        metric,
        np.mean([record.delta for record in records.values()]),
        all_draws,
        sum(len(a) for a in a_arrays),
        higher_is_better,
        np.concatenate(a_arrays),
        np.concatenate(b_arrays),
        groups,
        alpha,
        per_seed=records,
    )
    if return_draws:
        return rec, all_draws
    return rec


# ---------------------------------------------------------------------------
# Holm--Bonferroni correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HolmBonferroniResult:
    """Per-test outcome of a Holm–Bonferroni correction over a metric family."""

    metric: str
    p_value: float
    p_adjusted: float
    rejected: bool


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, HolmBonferroniResult]:
    """Holm–Bonferroni step-down correction on a family of two-sided p-values.

    Implements the standard step-down procedure: order p-values
    ascending, and reject at step $k$ iff $p_{(k)} \\leq \\alpha / (m - k + 1)$
    *and* every earlier hypothesis was rejected. Returns a per-metric
    record carrying the original p-value, the Holm-adjusted p-value
    ($\\min(1, (m - k + 1) p_{(k)})$ with monotonised cumulative max), and
    the reject decision at family-wise alpha.

    The order of the input mapping is ignored; results are returned in
    the original metric order (Python 3.11+ dicts preserve insertion).
    A NaN represents an unavailable test: it remains NaN and cannot reject,
    while still counting in the declared family size. Other values outside
    [0, 1], including infinities, raise instead of becoming evidence.
    """
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie strictly between zero and one")
    if not p_values:
        return {}
    items = [(name, float(value)) for name, value in p_values.items()]
    if any(not np.isnan(p) and (not np.isfinite(p) or not 0 <= p <= 1) for _, p in items):
        raise ValueError("Each p-value must be in [0, 1] or NaN for an unavailable test")
    m = len(items)
    # Stable sort by p-value ascending; tie-break on original order.
    order = sorted(range(m), key=lambda i: (float(items[i][1]) if np.isfinite(items[i][1]) else float("inf"), i))
    adjusted = np.full(m, np.nan, dtype=np.float64)
    decisions = np.zeros(m, dtype=bool)
    cumulative_max = -np.inf
    rejected_so_far = True
    for rank, idx in enumerate(order):
        name, p = items[idx]
        p = float(p)
        if np.isnan(p):
            # max(previous, NaN) would otherwise relabel missing evidence with
            # the previous test's adjusted probability.
            rejected_so_far = False
            continue
        factor = m - rank
        adj = min(1.0, factor * p) if np.isfinite(p) else float("nan")
        cumulative_max = max(cumulative_max, adj)
        adjusted[idx] = cumulative_max
        if rejected_so_far and np.isfinite(p) and p <= alpha / factor:
            decisions[idx] = True
        else:
            rejected_so_far = False
            decisions[idx] = False

    return {
        name: HolmBonferroniResult(
            metric=name,
            p_value=float(items[idx][1]),
            p_adjusted=float(adjusted[idx]) if np.isfinite(adjusted[idx]) else float("nan"),
            rejected=bool(decisions[idx]),
        )
        for idx, (name, _) in enumerate(items)
    }


# ---------------------------------------------------------------------------
# Pairwise decision rule
# ---------------------------------------------------------------------------


class PairwiseDecision(str, Enum):
    """Primary-metric outcomes; no-difference does not establish equivalence."""

    PARETO_DOMINANT_A = "pareto_dominant_a"
    PARETO_DOMINANT_B = "pareto_dominant_b"
    TRADE_OFF = "trade_off"
    PARTIAL_EVIDENCE = "partial_evidence"
    # Historical name/value retained for callers. This means only that the
    # declared superiority criteria were not met, never statistical equivalence.
    INDISTINGUISHABLE = "indistinguishable"


def decide_pairwise(
    deltas: Iterable[BootstrapDelta],
    holm: Mapping[str, HolmBonferroniResult] | None = None,
) -> tuple[PairwiseDecision, dict[str, str]]:
    """Reduce per-metric paired-bootstrap deltas + Holm flags to a verdict.

    Each metric independently votes:
      * ``win_a`` iff ``delta.favours_a`` *and* (if ``holm`` provided)
        the Holm-adjusted decision rejects the null.
      * ``win_b`` iff ``delta.favours_a`` is False, ``delta.excludes_zero``
        is True, and Holm rejects.
      * ``tie`` otherwise.

    Pareto-dominant: every metric votes the same direction.
    Indistinguishable: no metric votes either way.
    Trade-off: demonstrated wins in both directions. Partial evidence:
    demonstrated wins in one direction with inconclusive remaining metrics.
    This generic helper judges only the supplied family; use
    ``compare_primary_metrics`` for all six preregistered primary metrics.
    """
    votes: dict[str, str] = {}
    for d in deltas:
        if not d.excludes_zero:
            votes[d.metric] = "tie"
            continue
        if holm is not None:
            r = holm.get(d.metric)
            if r is None or not r.rejected:
                votes[d.metric] = "tie"
                continue
        votes[d.metric] = "win_a" if d.favours_a else "win_b"

    has_a = any(v == "win_a" for v in votes.values())
    has_b = any(v == "win_b" for v in votes.values())
    if has_a and has_b:
        return PairwiseDecision.TRADE_OFF, votes
    if has_a and all(v == "win_a" for v in votes.values()):
        return PairwiseDecision.PARETO_DOMINANT_A, votes
    if has_b and all(v == "win_b" for v in votes.values()):
        return PairwiseDecision.PARETO_DOMINANT_B, votes
    if has_a or has_b:
        return PairwiseDecision.PARTIAL_EVIDENCE, votes
    return PairwiseDecision.INDISTINGUISHABLE, votes


@dataclass(frozen=True)
class PrimaryMetricComparison:
    """Six-metric, three-seed comparison conditional on the fitted procedures.

    A Pareto label concerns primary metrics only. Deployment latency and
    footprint must still be assessed separately before an overall claim.
    ``inference_status`` records availability, not qualification of a caller's
    declared paired test or scientific eligibility of these inputs.
    """

    deltas: Mapping[str, BootstrapDelta]
    holm: Mapping[str, HolmBonferroniResult]
    decision: PairwiseDecision
    votes: Mapping[str, str]
    inference_status: str
    paired_test: str | None
    test_assumptions: str | None
    task_ids: tuple[str, ...]
    blocks: tuple[str, ...]
    resampling_seed: int
    alpha: float


def compare_primary_metrics(
    per_sample_a_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    per_sample_b_by_seed: Mapping[int, Mapping[str, np.ndarray]],
    *,
    blocks: Sequence[str] | np.ndarray,
    n_bootstrap: int = 10_000,
    alpha: float = 0.05,
    seed: int = DEFAULT_SEEDS[0],
    p_values: Mapping[str, float] | None = None,
    paired_test: str | None = None,
    test_assumptions: str | None = None,
) -> PrimaryMetricComparison:
    """Compare all primary metrics on an explicit, frozen three-seed roster.

    Each seed supplies ``task_id``, ``regret``, ``assd``, ``h95``, ``success``,
    ``obstacle_violation`` and ``valid_path`` arrays in identical task order.
    Regret must already retain censored failures; reliability uses all tasks.
    Geometry reduces each method's own successful, obstacle-free subset.
    The caller is responsible for matching data, operating conditions, work
    limits and frozen checkpoints; this function never loads or evaluates one.

    Geographic blocks are sampled identically for both methods, every seed
    and every metric. Within each draw, compute each seed's prescribed
    aggregate contrast and then average the three contrasts. Per-seed CIs and
    method-specific denominators accompany the seed-mean interval. A geometry
    draw with an empty subset makes the CI unavailable; no draw is discarded.

    Holm uses only the six explicitly supplied p-values from a predeclared
    paired test of this same block/seed-mean contrast. The test name and its
    assumptions are required metadata, not automatic evidence of validity.
    The legacy bootstrap-tail proxy is never used. Without a declared test,
    estimates/CIs are available but no confirmatory win is returned. NaN tests
    remain unavailable and retain their place in the six-test family. Claims
    across more method pairs require a larger prespecified correction family.
    """
    _validate_bootstrap(n_bootstrap, alpha, "raise", seed)
    if blocks is None:
        raise ValueError("Canonical comparisons require known geographic blocks")
    for method in (per_sample_a_by_seed, per_sample_b_by_seed):
        if not isinstance(method, Mapping) or set(method) != set(DEFAULT_SEEDS):
            raise ValueError("Canonical comparisons require exactly the three declared seeds")
    names = [name for name, _, _ in PRIMARY_METRICS]
    if p_values is not None:
        if not paired_test or not paired_test.strip() or not test_assumptions or not test_assumptions.strip():
            raise ValueError("Supplied p-values require a declared paired test and its assumptions")
        if set(p_values) != set(names):
            raise ValueError("The declared paired test must supply all six primary p-values (NaN if unavailable)")
        # Validate before bootstrapping, rather than failing after expensive work.
        holm_bonferroni(p_values, alpha=alpha)

    prepared = []
    task_ids = None
    for method in (per_sample_a_by_seed, per_sample_b_by_seed):
        metric_columns = {name: {} for name in names}
        for training_seed in DEFAULT_SEEDS:
            columns = method[training_seed]
            required = {"task_id", "regret", "assd", "h95", "success", "obstacle_violation", "valid_path"}
            if not required <= columns.keys():
                raise ValueError(f"Seed {training_seed} requires task identities and complete primary outcome columns")
            ids = np.asarray(columns["task_id"], dtype=object)
            if ids.ndim != 1 or not ids.size or any(not isinstance(item, str) or not item.strip() for item in ids):
                raise ValueError("task_id must be a nonempty one-dimensional array of nonempty strings")
            if len(set(ids)) != len(ids):
                raise ValueError("task_id values must be unique within the frozen roster")
            if task_ids is None:
                task_ids = tuple(ids)
                _block_groups(blocks, len(ids))
            elif tuple(ids) != task_ids:
                raise ValueError("Both methods and all seeds must have identical ordered task identities")
            values = {key: np.asarray(columns[key], dtype=np.float64) for key in required - {"task_id"}}
            if any(array.shape != ids.shape for array in values.values()):
                raise ValueError("Primary outcome arrays must align with task identities")
            for key in ("success", "obstacle_violation", "valid_path"):
                if not np.isin(values[key], [0, 1]).all():
                    raise ValueError(f"{key} must contain a complete binary outcome for every task")
            valid = values["valid_path"].astype(bool)
            if not np.array_equal(valid, values["success"].astype(bool) & ~values["obstacle_violation"].astype(bool)):
                raise ValueError("valid_path must equal success AND NOT obstacle_violation")
            if not np.isfinite(values["regret"]).all():
                raise ValueError("Regret must be finite on every task, including censored failures")
            for key in ("assd", "h95"):
                if not np.isfinite(values[key][valid]).all() or np.any(values[key][valid] < 0):
                    raise ValueError("Geometry must be finite and nonnegative on each method's valid subset")
                values[key] = np.where(valid, values[key], np.nan)
            for name, key in (
                ("cost_regret_median", "regret"),
                ("cost_regret_p90", "regret"),
                ("path_assd", "assd"),
                ("path_hausdorff_p95", "h95"),
                ("success_rate", "success"),
                ("obstacle_violation_rate", "obstacle_violation"),
            ):
                metric_columns[name][training_seed] = values[key]
        prepared.append(metric_columns)

    statistics = {"cost_regret_median": np.median, "cost_regret_p90": lambda array: np.quantile(array, 0.9)}
    deltas = {}
    for name, family, higher_is_better in PRIMARY_METRICS:
        deltas[name] = multiseed_paired_bootstrap(
            prepared[0][name],
            prepared[1][name],
            metric=name,
            higher_is_better=higher_is_better,
            statistic=statistics.get(name, np.mean),
            n_bootstrap_per_seed=n_bootstrap,
            alpha=alpha,
            seed=seed,
            blocks=blocks,
            nan_policy="omit" if family is MetricFamily.GEOMETRIC else "raise",
        )
    unavailable = p_values is None
    holm = holm_bonferroni({name: float("nan") for name in names} if unavailable else p_values, alpha=alpha)
    decision, votes = decide_pairwise(deltas.values(), holm)
    complete = all(np.isfinite(record.p_value) for record in holm.values()) and all(
        np.isfinite(record.ci_low) and np.isfinite(record.ci_high) for record in deltas.values()
    )
    return PrimaryMetricComparison(
        deltas,
        holm,
        decision,
        votes,
        "test_not_provided" if unavailable else "available" if complete else "incomplete",
        paired_test,
        test_assumptions,
        task_ids,
        tuple(blocks),
        seed,
        alpha,
    )
