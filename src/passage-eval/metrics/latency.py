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

"""Censoring-aware complete-request latency quantiles.

**Proposed rule, not sealed** (``passage-censored-latency/v1``). The protocol leaves the treatment of
timeouts in latency percentiles to be fixed before sealing; this module implements one candidate rule so that
it can be reviewed and tested. It does not replace the legacy interpolated quantiles, which callers keep
under their existing keys.

1. Every request of the series is counted, failures included.
2. A request is *right-censored* when its status is ``timeout`` or when its latency was charged rather than
   measured (``latency_charged_not_measured``: a task never sent to a worker). Its recorded value is only a
   lower bound of its latency, so it is ordered as ``+inf``.
3. A quantile is an order statistic (``numpy`` method ``inverted_cdf``, no interpolation). When the selected
   rank falls on a censored request, the value is unknown: the result says ``censored: true`` and reports a
   lower bound, the same order statistic of the recorded lower bounds.
4. Counts are reported with the quantiles: requests, distinct scenes, repeats per scene, successes, timeouts,
   censored requests and failures that are not timeouts. The status counts partition the series
   (``n_ok + n_timeout + n_failed_non_timeout == n_requests``). A failure that is not a timeout is either
   charged without a measurement (censored, ``n_charged_not_measured``) or a fast failure that keeps its
   observed time (``n_failed_non_timeout_measured``). Quantiles over the successful requests alone are
   secondary and never represent the whole series.
5. ``meets_min_warm_requests`` flags a series with at least ``min_warm_requests`` *warm* requests (100 in the
   manuscript). It is a report flag, not an admission decision. A request is warm only after a declared
   warm-up, so the flag needs the warm-up evidence of the run (``warmup``, see
   :func:`passage_eval.execution.timing.manifest_warmup`), not the series name: it is ``false`` when the series
   is not ``warm`` or no warm-up was declared and performed, ``null`` when the caller gives no evidence, and
   otherwise counts the requests whose per-request warm-up state, when recorded, is complete.
   ``meets_min_warm_requests_reason`` says why the flag is not ``true``.

A repeated request is a timing observation, not an additional independent scene.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np

RULE_ID = "passage-censored-latency/v1"
RULE_STATUS = "proposed_not_sealed"
METHOD = "inverted_cdf"
DEFAULT_QUANTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
#: The manuscript's minimum number of warm timed requests per latency row.
MIN_WARM_REQUESTS = 100


def latency_censoring(status: Sequence[str], charged_not_measured: Sequence[bool] | None = None) -> np.ndarray:
    """Right-censoring flags of the proposed rule: a timeout, or a latency charged without a measurement."""
    status = np.asarray(status).astype(str)
    censored = status == "timeout"
    if charged_not_measured is not None:
        charged = np.asarray(charged_not_measured)
        if charged.shape != status.shape or charged.dtype.kind != "b":
            raise ValueError("charged-not-measured flags must be booleans aligned with the statuses")
        censored = censored | charged
    return censored


def _rank_index(n: int, q: float) -> int:
    # The 0-based order statistic numpy selects with method="inverted_cdf", without its interpolation.
    return int(np.quantile(np.arange(n, dtype=np.float64), q, method=METHOD))


def _order_statistics(values: np.ndarray, censored: np.ndarray, quantiles) -> dict:
    n = len(values)
    keys = np.where(censored, np.inf, values)
    order = np.argsort(keys, kind="stable")
    lower = np.sort(values, kind="stable")  # recorded values are lower bounds of every latency
    result = {}
    for name, q in quantiles:
        index = _rank_index(n, q)
        chosen = int(order[index])
        is_censored = bool(censored[chosen])
        result[name] = {
            "q": float(q),
            "rank": index + 1,
            "value_ms": None if is_censored else float(values[chosen]),
            "censored": is_censored,
            "lower_bound_ms": float(lower[index]) if is_censored else float(values[chosen]),
        }
    return result


def _warm_requests(
    series: str, warmup: Mapping | None, n: int, states: Sequence[bool | None] | None, minimum: int
) -> tuple[int | None, bool | None, str | None]:
    """``(warm requests, meets the minimum, reason when it does not)`` from the run's warm-up evidence."""
    if series != "warm":
        return 0, False, "not_a_warm_series"
    if warmup is None:
        return None, None, "warmup_evidence_not_given"
    if not warmup["declared"]:
        return 0, False, "no_declared_warmup"
    if not warmup["performed"]:
        return 0, False, "declared_warmup_not_performed_in_every_pass"
    warm = n if states is None else sum(state is True for state in states)
    if warm < minimum:
        return warm, False, "fewer_warm_requests_than_the_minimum"
    return warm, True, None


def censored_latency_quantiles(
    latency_ms: Sequence[float],
    *,
    status: Sequence[str],
    sample_id: Sequence[str],
    censored: Sequence[bool] | None = None,
    series: str = "warm",
    quantiles: Sequence[tuple[str, float]] = DEFAULT_QUANTILES,
    min_warm_requests: int = MIN_WARM_REQUESTS,
    warmup_complete: Sequence[bool | None] | None = None,
    warmup: Mapping | None = None,
) -> dict:
    """Quantiles of every request of one series under the proposed censoring rule (see the module docstring).

    ``censored`` defaults to the statuses' timeouts; pass :func:`latency_censoring` with the records'
    ``latency_charged_not_measured`` flags to censor charged latencies as well. ``warmup_complete`` takes the
    re-warm profile's ``warmup_complete_before_request`` values, when the records carry them; a ``None`` state
    (no warm-up applied) counts as a request without a complete warm-up. ``warmup`` takes the run's warm-up
    evidence (:func:`passage_eval.execution.timing.manifest_warmup`); without it the warm-request flag is
    ``None``.
    """
    values = np.asarray(latency_ms, dtype=np.float64)
    status = np.asarray(status).astype(str)
    sample_id = np.asarray(sample_id).astype(str)
    if values.ndim != 1 or not len(values):
        raise ValueError("latency quantiles need a nonempty one-dimensional series of requests")
    if status.shape != values.shape or sample_id.shape != values.shape:
        raise ValueError("statuses and sample ids must align with the latencies")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("recorded latencies must be finite and nonnegative (censoring is a separate flag)")
    flags = latency_censoring(status) if censored is None else np.asarray(censored)
    if flags.shape != values.shape or flags.dtype.kind != "b":
        raise ValueError("censoring flags must be booleans aligned with the latencies")
    if isinstance(min_warm_requests, bool) or not isinstance(min_warm_requests, int) or min_warm_requests < 1:
        raise ValueError("min_warm_requests must be a positive integer")
    for name, q in quantiles:
        if not isinstance(name, str) or not 0 < float(q) < 1:
            raise ValueError("quantiles must be named levels strictly between zero and one")
    ok = status == "ok"
    if np.any(ok & flags):
        raise ValueError("a successful request cannot be censored")
    if warmup is not None and (
        not isinstance(warmup, Mapping)
        or not isinstance(warmup.get("declared"), bool)
        or not isinstance(warmup.get("performed"), bool)
    ):
        raise ValueError("warm-up evidence must be a mapping with boolean 'declared' and 'performed'")
    states = None
    if warmup_complete is not None:
        states = list(warmup_complete)
        if len(states) != values.size or any(
            state is not None and not isinstance(state, (bool, np.bool_)) for state in states
        ):
            raise ValueError("warm-up states must be booleans or None aligned with the latencies")
        states = [None if state is None else bool(state) for state in states]
    repeats = Counter(sample_id.tolist())
    n = int(values.size)
    n_warm, meets, reason = _warm_requests(series, warmup, n, states, min_warm_requests)
    hot = np.ones(n, dtype=bool) if states is None else np.asarray([state is True for state in states], dtype=bool)
    n_rewarm_incomplete = int(n - hot.sum())
    block = {
        "rule": RULE_ID,
        "status": RULE_STATUS,
        "method": METHOD,
        "interpolation": False,
        "censoring": "timeout_or_charged_not_measured_right_censored_at_infinity",
        "series": series,
        "n_requests": n,
        "n_distinct_scenes": len(repeats),
        "n_repeats": next(iter(set(repeats.values()))) if len(set(repeats.values())) == 1 else None,
        "repeats_per_scene": {"min": min(repeats.values()), "max": max(repeats.values())},
        "n_ok": int(ok.sum()),
        "n_censored": int(flags.sum()),
        "n_timeout": int(np.sum(status == "timeout")),
        "n_charged_not_measured": int(np.sum(flags & (status != "timeout"))),
        "n_failed_non_timeout": int(np.sum(~ok & (status != "timeout"))),
        "n_failed_non_timeout_measured": int(np.sum(~ok & ~flags)),
        "quantiles": _order_statistics(values[hot], flags[hot], quantiles) if hot.any() else None,
        "hot_latency_scope": "complete_warmup_requests_only",
        "n_hot_latency_requests": int(hot.sum()),
        "n_rewarm_incomplete_requests": n_rewarm_incomplete,
        "hot_latency_final": n_rewarm_incomplete <= 0.01 * n,
        "success_only": {
            "secondary": True,
            "note": "successful requests only; never a substitute for the all-request quantiles",
            "n": int(ok.sum()),
            "quantiles": (
                {
                    name: {key: row[key] for key in ("q", "rank", "value_ms")}
                    for name, row in _order_statistics(values[ok], flags[ok], quantiles).items()
                }
                if ok.any()
                else None
            ),
        },
        "min_warm_requests": min_warm_requests,
        "warmup": None if warmup is None else dict(warmup),
        "n_warm_requests": n_warm,
        "meets_min_warm_requests": meets,
        "meets_min_warm_requests_reason": reason,
        "n_requests_without_complete_warmup": None if states is None else n_rewarm_incomplete,
    }
    return block


#: Record field of the supervisor's legality check, physical recost and budget accounting after a reply.
VALIDATION_RECOST_FIELD = "validation_recost_ms"


def validation_recost_values(records: Sequence[Mapping]) -> np.ndarray:
    """Per-record ``validation_recost_ms``, ``nan`` where a record has none (the in-process runner records none)."""
    values = []
    for row in records:
        value = row.get(VALIDATION_RECOST_FIELD)
        recorded = isinstance(value, (int, float)) and not isinstance(value, bool)
        values.append(float(value) if recorded else np.nan)
    return np.asarray(values, dtype=np.float64)


def validation_recost_summary(
    values: Sequence[float], *, inside_latency: bool, quantiles: Sequence[tuple[str, float]] = DEFAULT_QUANTILES
) -> dict:
    """The supervisor's validation and recost time, reported separately from the scored latency.

    The manuscript keeps validation and physical recosting outside the request latency and reports them
    separately. ``inside_latency`` states whether the run's timing profile nevertheless counts this time inside
    the scored latency (the legacy profile and the in-process runner do). Quantiles are order statistics
    (``inverted_cdf``) over the recorded values; a request killed before a reply records zero.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("validation and recost times must be one-dimensional")
    if not isinstance(inside_latency, bool):
        raise ValueError("inside_latency must be a boolean")
    recorded = ~np.isnan(values)
    observed = values[recorded]
    if not np.isfinite(observed).all() or np.any(observed < 0):
        raise ValueError("recorded validation and recost times must be finite and nonnegative")
    if not observed.size:
        status = "not_recorded"
    elif observed.size == values.size:
        status = "recorded"
    else:
        status = "partially_recorded"
    ordered = np.sort(observed)
    return {
        "field": VALIDATION_RECOST_FIELD,
        "reported_separately": True,
        "inside_scored_latency": inside_latency,
        "status": status,
        "method": METHOD,
        "interpolation": False,
        "n_requests": int(values.size),
        "n_recorded": int(observed.size),
        "mean_ms": float(observed.mean()) if observed.size else None,
        "max_ms": float(observed.max()) if observed.size else None,
        "quantiles_ms": {
            name: float(ordered[_rank_index(observed.size, q)]) if observed.size else None for name, q in quantiles
        },
    }
