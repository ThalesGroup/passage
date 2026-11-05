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

"""Batch sizing: dynamic request admission (the campaign guard) and the worst-case wall time (information only).

Dynamic admission (``admission_sizing``, the judge's R7). The campaign does not size a batch for its worst case: a
campaign session engages a request only while its window still holds the request's budget, grace, kill wait and
persistence (:mod:`passage_eval.supervision.session`), and defers the remaining tasks to the next window otherwise.
The plan therefore declares *measured* overheads and the observed warm request time of every procedure, and two
checks replace the worst-case refusal:

* margin  = ``lease_or_timeout_ms >= safety_margin * sum_p expected_p``, with ``safety_margin >= 3`` and
  ``expected_p = fixed + per_seed + (k + batch_size) * (observed_p + per_request_overhead) + persistence``: the
  window holds a whole batch, as observed, three times over;
* sanity  = for every procedure, ``floor_p <= lease_or_timeout_ms / n_procedures`` with ``floor_p = fixed +
  per_seed + roles * start + k * (observed_p + per_request_overhead) + reserve_p``: a session that runs as observed,
  given its equal share of a fresh window, can engage at least one measured request after a full start of its
  workers (``start = startup_timeout_s * 1000 + kill wait``; ``reserve_p`` = budget + grace + kill wait +
  persistence). Without it a batch could be deferred forever.

The worst case below stays available (``check-plan`` prints it) but no longer refuses a campaign plan.

Worst case. A *batch* is one supervised run (``python -m passage_eval.supervision run``) over ``batch_size`` roster
tasks. A launcher that gives each batch a lease or a timeout without dynamic admission must size it so that the batch
cannot be killed by that bound while the supervisor is still within its own limits; otherwise a slow but admissible
candidate turns into an ambiguous batch.

The run repeats every timed series once per seed of the spec, and each seed gets a fresh runner: its own
initial start, its own fresh workers per pass, its own restart counter and every measured, warm-up and re-warm
request. The bound is an upper bound under the supervisor's rules, not an estimate. In milliseconds, per seed:

* request   = ``wall_time_ms + deadline_grace_ms`` (``hard_timeout_ms`` without a wall budget)
  ``+ kill_wait_ms + per_request_overhead_ms``: one executed request, including a preemption kill and the
  validation and bookkeeping after the reply;
* start     = ``startup_timeout_s * 1000 + kill_wait_ms``: one worker start, including the kill of a worker
  that fails to start;
* measured  = ``(cold_passes + warm_passes) * batch_size`` timed requests;
* warm-up   = ``warm_passes * warmup_requests`` untimed requests;
* re-warms  = ``min(max_restarts, warm_passes * batch_size)`` under a re-warm profile with a warm-up (a re-warm
  follows a counted restart and precedes at most one measured request), else 0; re-warm = ``warmup_requests *
  re-warms``;
* restarts  = ``min(max_restarts, roles * (measured + warm-up + re-warm))``: counted restarts (every request may
  restart each role once), capped by the supervisor for each seed;
* outside   = ``roles * min(warm_passes, re-warms)``: restarts of workers lost during a failed re-warm, which the
  supervisor does not charge to ``max_restarts`` (at most one re-warm fails per pass);
* planned   = ``(1 + cold_passes + warm_passes) * roles``: the initial start and the fresh workers of every pass;
* seed      = ``per_seed_overhead_ms + (planned + restarts + outside) * start + (measured + warm-up + re-warm) *
  request``;
* total     = ``fixed_overhead_ms + seeds * seed``.

With one seed, one pass, no warm-up, one role and a restart cap at least the batch size, this reduces to
``batch_size * (budget + grace + kill + start + overhead) + fixed overhead + 2 * start``. ``fixed_overhead_ms``
(once per run: environment inspection, reference qualification), ``per_seed_overhead_ms`` (once per seed: echo
baseline, identity check, persistence) and ``per_request_overhead_ms`` are declared by the operator from
measurements on the target; they are required because the bound means nothing without them.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

from passage_eval.supervision import WORKER_KILL_WAIT_S

SCHEMA = "passage-supervised-batch-bound/v1"


class BatchPlanError(ValueError):
    """A batch whose worst-case wall time exceeds the lease or timeout that would bound it."""


def _nonnegative(name: str, value, *, integer: bool = False, positive: bool = False) -> None:
    kinds = (int,) if integer else (int, float)
    if isinstance(value, bool) or not isinstance(value, kinds) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite {'integer' if integer else 'number'}")
    if value < 0 or (positive and value == 0):
        raise ValueError(f"{name} must be {'positive' if positive else 'nonnegative'}")


@dataclass(frozen=True)
class SupervisedBatchPlan:
    """The settings that bound the wall time of one supervised batch."""

    batch_size: int
    wall_time_ms: float | None
    deadline_grace_ms: float
    hard_timeout_ms: float
    startup_timeout_s: float
    max_restarts: int
    fixed_overhead_ms: float
    per_request_overhead_ms: float
    #: Number of seeds of the run: every timed series, worker start and restart cap is repeated per seed.
    seeds: int
    per_seed_overhead_ms: float
    roles: int = 1
    cold_passes: int = 0
    warm_passes: int = 1
    warmup_requests: int = 0
    rewarm_after_restart: bool = False
    kill_wait_ms: float = WORKER_KILL_WAIT_S * 1000.0

    def __post_init__(self):
        _nonnegative("batch_size", self.batch_size, integer=True)
        if self.wall_time_ms is not None:
            _nonnegative("wall_time_ms", self.wall_time_ms, positive=True)
        for name in (
            "deadline_grace_ms",
            "fixed_overhead_ms",
            "per_request_overhead_ms",
            "per_seed_overhead_ms",
            "kill_wait_ms",
        ):
            _nonnegative(name, getattr(self, name))
        _nonnegative("hard_timeout_ms", self.hard_timeout_ms, positive=True)
        _nonnegative("startup_timeout_s", self.startup_timeout_s, positive=True)
        for name in ("max_restarts", "cold_passes", "warm_passes", "warmup_requests"):
            _nonnegative(name, getattr(self, name), integer=True)
        _nonnegative("roles", self.roles, integer=True, positive=True)
        _nonnegative("seeds", self.seeds, integer=True, positive=True)
        if self.cold_passes + self.warm_passes < 1:
            raise ValueError("a batch runs at least one timed pass")
        if not isinstance(self.rewarm_after_restart, bool):
            raise ValueError("rewarm_after_restart must be a boolean")

    @classmethod
    def from_run_settings(
        cls,
        spec,
        config,
        *,
        batch_size: int,
        split: bool,
        fixed_overhead_ms: float,
        per_request_overhead_ms: float,
        per_seed_overhead_ms: float,
        session: bool = False,
    ) -> "SupervisedBatchPlan":
        """The plan of a run with this ``BenchSpec``-like ``spec`` and ``SupervisorConfig``-like ``config``.

        Mirrors the supervisor: one runner per seed of ``spec.seeds``; per seed, a cold pass and
        ``latency_repeats`` warm passes when latency is measured, one warm pass otherwise; ``latency_warmup``
        warm-up requests per warm pass when latency is measured. A campaign session (``session``) has no cold pass.
        """
        from passage_eval.execution.timing import LEGACY, get_profile
        from passage_eval.protocol.contracts import BudgetSpec

        seeds = list(spec.seeds)
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("spec.seeds must be a nonempty list of distinct seeds")
        budget = BudgetSpec(**dict(spec.budget))
        measured = bool(spec.measure_latency)
        return cls(
            batch_size=batch_size,
            wall_time_ms=budget.wall_time_ms,
            deadline_grace_ms=config.deadline_grace_ms,
            hard_timeout_ms=config.hard_timeout_ms,
            startup_timeout_s=config.startup_timeout_s,
            max_restarts=config.max_restarts,
            fixed_overhead_ms=fixed_overhead_ms,
            per_request_overhead_ms=per_request_overhead_ms,
            seeds=len(seeds),
            per_seed_overhead_ms=per_seed_overhead_ms,
            roles=2 if split else 1,
            cold_passes=1 if measured and not session else 0,
            warm_passes=(spec.latency_repeats or 1) if measured else 1,
            warmup_requests=spec.latency_warmup if measured else 0,
            rewarm_after_restart=get_profile(getattr(config, "timing_profile", LEGACY)).rewarm_after_restart,
        )


def worst_case_batch_wall_ms(plan: SupervisedBatchPlan) -> dict:
    """The worst-case wall time of a batch and every term of the bound (see the module docstring)."""
    request_ms = (
        (plan.wall_time_ms + plan.deadline_grace_ms if plan.wall_time_ms is not None else plan.hard_timeout_ms)
        + plan.kill_wait_ms
        + plan.per_request_overhead_ms
    )
    start_ms = plan.startup_timeout_s * 1000.0 + plan.kill_wait_ms
    # Per seed: the supervisor builds a fresh runner, with its own starts and restart counter, for every seed.
    passes = plan.cold_passes + plan.warm_passes
    measured = passes * plan.batch_size
    warmup = plan.warm_passes * plan.warmup_requests
    rewarms = plan.rewarm_after_restart and plan.warmup_requests > 0
    rewarm_events = min(plan.max_restarts, plan.warm_passes * plan.batch_size) if rewarms else 0
    rewarm = plan.warmup_requests * rewarm_events
    executed = measured + warmup + rewarm
    # Every request may restart each role once before the cap.
    restarts = min(plan.max_restarts, plan.roles * executed)
    # A failed re-warm ends re-warming for its pass; the workers it lost restart outside the cap.
    outside_cap = plan.roles * min(plan.warm_passes, rewarm_events)
    planned = (1 + passes) * plan.roles
    per_seed = plan.per_seed_overhead_ms + (planned + restarts + outside_cap) * start_ms + executed * request_ms
    total = plan.fixed_overhead_ms + plan.seeds * per_seed
    return {
        "schema": SCHEMA,
        "plan": asdict(plan),
        "seeds": plan.seeds,
        "request_bound_ms": request_ms,
        "worker_start_bound_ms": start_ms,
        "measured_requests": measured,
        "warmup_requests": warmup,
        "rewarms_bound": rewarm_events,
        "rewarm_requests_bound": rewarm,
        "counted_restarts_bound": restarts,
        "rewarm_restarts_outside_cap_bound": outside_cap,
        "planned_worker_starts": planned,
        "per_seed_wall_ms": per_seed,
        "worst_case_wall_ms": total,
    }


def check_batch_plan(plan: SupervisedBatchPlan, limit_ms: float) -> dict:
    """Refuse a batch whose worst-case wall time exceeds ``limit_ms`` (a lease or batch timeout)."""
    _nonnegative("limit_ms", limit_ms, positive=True)
    bound = worst_case_batch_wall_ms(plan)
    bound.update(limit_ms=float(limit_ms), margin_ms=float(limit_ms) - bound["worst_case_wall_ms"])
    if bound["worst_case_wall_ms"] > limit_ms:
        raise BatchPlanError(
            f"worst-case batch wall time {bound['worst_case_wall_ms']:.0f} ms exceeds the bound {limit_ms:.0f} ms "
            f"({plan.batch_size} tasks and {plan.seeds} seed(s); per seed {bound['measured_requests']} timed and "
            f"{bound['warmup_requests'] + bound['rewarm_requests_bound']} warm-up requests at "
            f"{bound['request_bound_ms']:.0f} ms, "
            f"{bound['planned_worker_starts'] + bound['counted_restarts_bound'] + bound['rewarm_restarts_outside_cap_bound']}"
            f" worker starts at {bound['worker_start_bound_ms']:.0f} ms); use smaller batches, fewer seeds per batch "
            f"or tighter limits"
        )
    bound["admissible"] = True
    return bound


ADMISSION_SIZING_SCHEMA = "passage-dynamic-admission-sizing/v1"
#: The judge's R7: the lease holds the observed batch time at least three times.
MIN_SAFETY_MARGIN = 3.0


def admission_sizing(
    *,
    lease_ms: float,
    batch_size: int,
    warmup_requests: int,
    procedures: dict,
    fixed_overhead_ms: float,
    per_seed_overhead_ms: float,
    per_request_overhead_ms: float,
    persistence_ms: float,
    safety_margin: float,
    deadline_grace_ms: float,
    hard_timeout_ms: float,
    startup_timeout_s: float,
    kill_wait_ms: float = WORKER_KILL_WAIT_S * 1000.0,
) -> dict:
    """The margin and sanity checks of dynamic admission (module docstring); ``admissible`` says whether both hold.

    ``procedures`` maps each procedure to ``{"roles", "wall_time_ms", "observed_request_ms"}``.
    """
    for name, value in (
        ("lease_ms", lease_ms),
        ("fixed_overhead_ms", fixed_overhead_ms),
        ("per_seed_overhead_ms", per_seed_overhead_ms),
        ("per_request_overhead_ms", per_request_overhead_ms),
        ("persistence_ms", persistence_ms),
    ):
        _nonnegative(name, value, positive=name == "lease_ms")
    _nonnegative("batch_size", batch_size, integer=True, positive=True)
    _nonnegative("warmup_requests", warmup_requests, integer=True)
    _nonnegative("safety_margin", safety_margin, positive=True)
    if not procedures:
        raise ValueError("at least one procedure is required")
    start_ms = startup_timeout_s * 1000.0 + kill_wait_ms
    per_procedure, expected_total, sane = {}, 0.0, True
    share = lease_ms / len(procedures)
    for name in sorted(procedures):
        row = procedures[name]
        _nonnegative(f"{name}.observed_request_ms", row["observed_request_ms"], positive=True)
        step = row["observed_request_ms"] + per_request_overhead_ms
        wall = row["wall_time_ms"]
        reserve = (wall + deadline_grace_ms if wall is not None else hard_timeout_ms) + kill_wait_ms + persistence_ms
        expected = fixed_overhead_ms + per_seed_overhead_ms + (warmup_requests + batch_size) * step + persistence_ms
        floor = fixed_overhead_ms + per_seed_overhead_ms + row["roles"] * start_ms + warmup_requests * step + reserve
        sane = sane and floor <= share
        expected_total += expected
        per_procedure[name] = {
            "request_reserve_ms": reserve,
            "expected_session_ms": expected,
            "sanity_floor_ms": floor,
            "window_share_ms": share,
        }
    margin_ok = safety_margin >= MIN_SAFETY_MARGIN and lease_ms >= safety_margin * expected_total
    return {
        "schema": ADMISSION_SIZING_SCHEMA,
        "lease_ms": float(lease_ms),
        "batch_size": batch_size,
        "safety_margin": float(safety_margin),
        "min_safety_margin": MIN_SAFETY_MARGIN,
        "expected_batch_ms": expected_total,
        "required_lease_ms": safety_margin * expected_total,
        "margin_ok": margin_ok,
        "sanity_ok": sane,
        "admissible": margin_ok and sane,
        "per_procedure": per_procedure,
        "window_share": "equal split of the remaining window among the procedures still to run in it",
    }


def largest_batch_size(plan: SupervisedBatchPlan, limit_ms: float, *, ceiling: int = 1 << 20) -> int:
    """The largest batch size whose worst case fits ``limit_ms`` with the plan's other settings (0 if none)."""
    _nonnegative("limit_ms", limit_ms, positive=True)

    def fits(size: int) -> bool:
        return worst_case_batch_wall_ms(replace(plan, batch_size=size))["worst_case_wall_ms"] <= limit_ms

    if not fits(1):
        return 0
    low, high = 1, max(1, ceiling)
    if fits(high):
        return high
    while high - low > 1:  # the bound is nondecreasing in the batch size
        middle = (low + high) // 2
        low, high = (middle, high) if fits(middle) else (low, middle)
    return low
