# MIT License - Copyright (c) 2026 THALES
"""Statistical-discipline primitives for the PASSAGE evaluation protocol.

The PASSAGE paper (App. J, "Cross-Family Evaluation Protocol") declares
that every headline number is

  (i) a function of the *decoded path* (Rule 1: common anchor),
 (ii) computed under a pre-registered set of primary metrics
      (``PRIMARY_METRICS``),
(iii) aggregated over the seeds ``DEFAULT_SEEDS = (1234, 2026, 3407)``,
 (iv) tested via a paired bootstrap on the mean of paired differences,
  (v) corrected across the primary-metric family by Holm--Bonferroni at
      family-wise alpha = 0.05, and
 (vi) reduced to a categorical pairwise decision: Pareto-dominant /
      Trade-off / Indistinguishable.

This module is the canonical implementation of points (iii) - (vi). It is
deliberately small and depends only on numpy so a third-party submission
can re-derive the headline decision from its per-sample primary-metric
arrays without re-running its decoder.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Pre-registered constants
# ---------------------------------------------------------------------------

#: Default seeds the PASSAGE paper pre-registers (App. J, §Statistical Discipline).
DEFAULT_SEEDS: tuple[int, int, int] = (1234, 2026, 3407)


class MetricFamily(str, Enum):
    """Three primary-metric families declared in §5.2 of the paper."""

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

    §5.2 of the paper defines a censored penalty $c^{\\text{pred}}_i = K$
    "on a decode that fails to connect $s$ to $g$ or that crosses an
    obstacle". The canonical PASSAGE penalty value is
    ``K = k_factor * oracle_cost`` with ``k_factor = 1e6``, matching the
    obstacle-traversal cost the dataset itself uses.

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
# Paired bootstrap on the mean of paired differences
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapDelta:
    """Paired-bootstrap result on a single primary metric.

    ``delta`` is the point estimate of $\\Delta_m = \\operatorname{mean}(m_A - m_B)$
    (paired by sample index); ``ci_low`` and ``ci_high`` are the central
    paired-bootstrap interval at the requested confidence level. ``n`` is
    the number of (paired) samples and ``n_bootstrap`` the resample count.
    """

    metric: str
    delta: float
    ci_low: float
    ci_high: float
    n: int
    n_bootstrap: int
    higher_is_better: bool

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
        """Two-sided p-value associated with the directional test.

        When ``draws`` (the resampled $\\Delta$ trajectory) is supplied the
        p-value is the empirical fraction of draws on the wrong side of
        zero, doubled for two-sided. With no draws available we fall back
        to a CI-coverage proxy: ``2 * Phi(-|delta| / se)`` where ``se`` is
        approximated from the symmetric CI half-width.
        """
        if draws is not None:
            d = np.asarray(draws, dtype=np.float64)
            if d.size == 0:
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


def _resample_indices(n: int, n_bootstrap: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``(n_bootstrap, n)`` matrix of sample-paired resample indices."""
    return rng.integers(low=0, high=n, size=(n_bootstrap, n))


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
) -> BootstrapDelta | tuple[BootstrapDelta, np.ndarray]:
    """Paired bootstrap on $\\Delta = T(\\hat{m}_A - \\hat{m}_B)$.

    Pairs are anchored on the same sample index (Rule of §5.3 - "the
    pairing cancels sample-difficulty variance because both methods are
    evaluated on the same indices"). Each resample picks ``n`` indices
    *with replacement* and recomputes ``statistic`` on the per-sample
    differences.

    For ``cost_regret_median`` and friends pass ``statistic=np.median``.
    The default ``statistic=np.mean`` is the paper's $\\Delta_m$.

    Parameters
    ----------
    per_sample_a, per_sample_b : array_like, shape (N,)
        Per-sample metric values, paired by index.
    statistic : callable
        Reduction applied to the per-sample difference vector. Defaults
        to ``np.mean`` for the paper's $\\Delta_m = \\mathrm{mean}(m_A - m_B)$.
    n_bootstrap, alpha, seed
        Bootstrap parameters; ``alpha = 0.05`` is the paper's default.
    return_draws : bool
        If ``True`` also return the ``(n_bootstrap,)`` draw vector for
        downstream Holm--Bonferroni p-value computation.
    """
    a = np.asarray(per_sample_a, dtype=np.float64)
    b = np.asarray(per_sample_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"Paired shape mismatch: {a.shape} vs {b.shape}.")
    n = a.shape[0]
    if n == 0:
        rec = BootstrapDelta(
            metric=metric,
            delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            n_bootstrap=n_bootstrap,
            higher_is_better=higher_is_better,
        )
        return (rec, np.zeros((0,), dtype=np.float64)) if return_draws else rec

    diff = a - b
    point = float(statistic(diff))
    rng = np.random.default_rng(seed)
    idx = _resample_indices(n, n_bootstrap, rng)
    draws = np.asarray([statistic(diff[row]) for row in idx], dtype=np.float64)
    draws.sort()
    lo = float(np.quantile(draws, alpha / 2.0))
    hi = float(np.quantile(draws, 1.0 - alpha / 2.0))
    rec = BootstrapDelta(
        metric=metric,
        delta=point,
        ci_low=lo,
        ci_high=hi,
        n=n,
        n_bootstrap=n_bootstrap,
        higher_is_better=higher_is_better,
    )
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
) -> BootstrapDelta | tuple[BootstrapDelta, np.ndarray]:
    """Three-seed aggregation: pool $S \\times B$ paired-bootstrap draws.

    Each seed contributes ``n_bootstrap_per_seed`` resamples on its own
    paired arrays; the composite CI is the central interval over the
    pooled draws (App. J §"Three-seed aggregation"). The point estimate
    is the across-seed mean of per-seed point estimates.

    Inputs may be a dict ``{seed: (N_seed,) array}`` or a list aligned
    with ``seeds``.
    """
    if isinstance(per_sample_a_by_seed, Mapping):
        a_arrays = [np.asarray(per_sample_a_by_seed[s], dtype=np.float64) for s in seeds]
        b_arrays = [np.asarray(per_sample_b_by_seed[s], dtype=np.float64) for s in seeds]
    else:
        if len(per_sample_a_by_seed) != len(seeds) or len(per_sample_b_by_seed) != len(seeds):
            raise ValueError("per-seed arrays must align with seeds.")
        a_arrays = [np.asarray(x, dtype=np.float64) for x in per_sample_a_by_seed]
        b_arrays = [np.asarray(x, dtype=np.float64) for x in per_sample_b_by_seed]

    pooled: list[np.ndarray] = []
    point_estimates: list[float] = []
    n_total = 0
    for s, a, b in zip(seeds, a_arrays, b_arrays):
        if a.shape != b.shape:
            raise ValueError(f"Paired shape mismatch on seed {s}: {a.shape} vs {b.shape}.")
        if a.size == 0:
            continue
        rec, draws = paired_bootstrap_delta(
            a,
            b,
            metric=metric,
            higher_is_better=higher_is_better,
            statistic=statistic,
            n_bootstrap=n_bootstrap_per_seed,
            alpha=alpha,
            seed=int(s),
            return_draws=True,
        )
        pooled.append(draws)
        point_estimates.append(rec.delta)
        n_total += a.shape[0]

    if not pooled:
        rec = BootstrapDelta(
            metric=metric,
            delta=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n=0,
            n_bootstrap=0,
            higher_is_better=higher_is_better,
        )
        return (rec, np.zeros((0,), dtype=np.float64)) if return_draws else rec

    all_draws = np.concatenate(pooled, axis=0)
    all_draws.sort()
    lo = float(np.quantile(all_draws, alpha / 2.0))
    hi = float(np.quantile(all_draws, 1.0 - alpha / 2.0))
    point = float(np.mean(point_estimates))
    rec = BootstrapDelta(
        metric=metric,
        delta=point,
        ci_low=lo,
        ci_high=hi,
        n=n_total,
        n_bootstrap=int(all_draws.shape[0]),
        higher_is_better=higher_is_better,
    )
    if return_draws:
        return rec, all_draws
    return rec


# ---------------------------------------------------------------------------
# Holm--Bonferroni correction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HolmBonferroniResult:
    """Per-test outcome of a Holm--Bonferroni correction over a metric family."""

    metric: str
    p_value: float
    p_adjusted: float
    rejected: bool


def holm_bonferroni(
    p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> dict[str, HolmBonferroniResult]:
    """Holm--Bonferroni step-down correction on a family of two-sided p-values.

    Implements the standard step-down procedure: order p-values
    ascending, and reject at step $k$ iff $p_{(k)} \\leq \\alpha / (m - k + 1)$
    *and* every earlier hypothesis was rejected. Returns a per-metric
    record carrying the original p-value, the Holm-adjusted p-value
    ($\\min(1, (m - k + 1) p_{(k)})$ with monotonised cumulative max), and
    the reject decision at family-wise alpha.

    The order of the input mapping is ignored; results are returned in
    the original metric order (Python 3.11+ dicts preserve insertion).
    """
    if not p_values:
        return {}
    items = list(p_values.items())
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
    """Three possible overall outcomes (App. J §Decision Rules)."""

    PARETO_DOMINANT_A = "pareto_dominant_a"
    PARETO_DOMINANT_B = "pareto_dominant_b"
    TRADE_OFF = "trade_off"
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
    Trade-off: a mix of votes.
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
    if has_a:
        return PairwiseDecision.PARETO_DOMINANT_A, votes
    if has_b:
        return PairwiseDecision.PARETO_DOMINANT_B, votes
    return PairwiseDecision.INDISTINGUISHABLE, votes
