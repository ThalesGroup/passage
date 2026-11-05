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

"""Campaign session mode of the supervisor: warm requests only, dynamic request admission, session invalidation.

A *session* is one supervised run of one seed over one roster inside one lease or timeout window. The campaign
runner (:mod:`passage_eval.campaign.runner`) starts every session in this mode; other callers keep the historical
cold-then-warm series. The mode (``SESSION_POLICY_ID``) is hashed into the procedure identity because it changes
what happens to a task when a worker cannot start (below); the window deadline and the persistence reserve are not,
since they never change the outcome of an executed request.

Series. Only one warm series is measured: fresh workers, then the untimed warm-up on disjoint inputs, then one timed
request per roster task. There is no cold series. The cold start is reported per session (``cold_start``): the
start time of each role's worker and the wall time of the first warm-up request. A session without a declared
warm-up is refused before any worker starts.

Dynamic request admission (``ADMISSION_RULE_ID``, the judge's R7). Before every request that the supervisor engages,
whether an initial warm-up request, a re-warm request or a measured request, the remaining time of the window must be
at least ``wall_time_ms + deadline_grace_ms`` (``hard_timeout_ms`` without a wall budget) ``+ kill wait +
persistence_ms``, plus ``startup_timeout_s + kill wait`` for every worker that must be restarted first. Otherwise the
session closes cleanly: the requests already measured are persisted, the remaining tasks are *deferred*, in roster
order, and the runner carries them to the next window. A measured request that overruns its budget is still a
penalised timeout; admission never turns it into a deferral.

Session invalidation. A worker that cannot start (initial start, the fresh start of the pass, a restart), an
initial warm-up whose request does not return ``ok`` or loses a worker, or a reached restart cap invalidates the
session: nothing it measured is written as a run artifact, no task is charged, and the restart cap is never
consumed by the warm-up. The runner treats an invalid session as an ambiguous batch attempt (relaunch rule R7), so
a candidate that cannot start never leaves the denominator: after the relaunches it becomes a candidate failure.

Receipt. Every session writes ``session.json`` (``SESSION_RECEIPT_SCHEMA``) at the top of its output directory:
status, executed and deferred tasks, closure point, invalidation reason, admission terms and cold start. The receipt
holds task identifiers and timings only; it is public run output like the run manifest.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from passage_eval.supervision import WORKER_KILL_WAIT_S

SESSION_POLICY_ID = "passage-campaign-session/v1"
ADMISSION_RULE_ID = "passage-dynamic-request-admission/v1"
SESSION_RECEIPT_SCHEMA = "passage-supervised-session/v1"
SESSION_RECEIPT = "session.json"

COMPLETE = "complete"
CLOSED = "closed_by_admission"
INVALID = "invalid"
STATUSES = (COMPLETE, CLOSED, INVALID)

#: Where a session can close: before its workers start, or before a request of each kind is engaged.
CLOSURE_POINTS = ("before_start", "before_warmup_request", "before_rewarm_request", "before_measured_request")
#: Why a session can be invalid. None of them is ever imputed to a task.
INVALID_REASONS = (
    "worker_start_failed",
    "initial_warmup_failed",
    "worker_restart_failed",
    "worker_restart_cap",
)


class SessionClosed(Exception):
    """The window cannot hold the next request: close the session cleanly and defer the remaining tasks."""

    def __init__(self, point: str, remaining_ms: float, required_ms: float):
        super().__init__(f"{point}: {remaining_ms:.0f} ms left, {required_ms:.0f} ms required")
        self.point, self.remaining_ms, self.required_ms = point, remaining_ms, required_ms


class SessionInvalid(Exception):
    """The session cannot produce a scored attempt (a worker cannot start, the initial warm-up failed)."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason, self.detail = reason, detail


def _positive(name: str, value: Any, *, allow_zero: bool = False) -> None:
    ok = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if not ok or value < 0 or (value == 0 and not allow_zero):
        raise ValueError(f"{name} must be a {'nonnegative' if allow_zero else 'positive'} finite number")


@dataclass(frozen=True)
class SessionPolicy:
    """The window of one session and the reserve kept after its last request.

    ``deadline_monotonic`` is an absolute ``time.monotonic()`` value of this host (the runner's window end minus its
    scoring reserve); ``None`` admits every request (a session without a lease, for diagnostics). ``persistence_ms``
    is the measured time to persist the session after its last request (identity re-check and artifact writes).
    """

    deadline_monotonic: float | None
    persistence_ms: float
    clock: Callable[[], float] = field(default=time.monotonic, compare=False, repr=False)

    def __post_init__(self):
        if self.deadline_monotonic is not None:
            _positive("deadline_monotonic", self.deadline_monotonic)
        _positive("persistence_ms", self.persistence_ms, allow_zero=True)

    def remaining_ms(self) -> float:
        if self.deadline_monotonic is None:
            return math.inf
        return 1000.0 * (self.deadline_monotonic - self.clock())


def request_reserve_ms(
    *,
    wall_time_ms: float | None,
    deadline_grace_ms: float,
    hard_timeout_ms: float,
    persistence_ms: float,
    kill_wait_ms: float = WORKER_KILL_WAIT_S * 1000.0,
) -> float:
    """The window that one engaged request needs: its budget and grace (or hard timeout), the kill wait, persistence."""
    bound = wall_time_ms + deadline_grace_ms if wall_time_ms is not None else hard_timeout_ms
    return float(bound + kill_wait_ms + persistence_ms)


def start_reserve_ms(startup_timeout_s: float, kill_wait_ms: float = WORKER_KILL_WAIT_S * 1000.0) -> float:
    """The window that one worker start needs: the start-up timeout and the kill of a worker that fails to start."""
    return float(startup_timeout_s * 1000.0 + kill_wait_ms)


def session_setup_reserve_ms(
    *,
    roles: int,
    wall_time_ms: float | None,
    deadline_grace_ms: float,
    hard_timeout_ms: float,
    startup_timeout_s: float,
    persistence_ms: float,
) -> float:
    """Reserve the fresh pass start and first dynamically admitted warm-up request."""
    if isinstance(roles, bool) or not isinstance(roles, int) or roles < 1:
        raise ValueError("roles must be a positive integer")
    request = request_reserve_ms(
        wall_time_ms=wall_time_ms,
        deadline_grace_ms=deadline_grace_ms,
        hard_timeout_ms=hard_timeout_ms,
        persistence_ms=persistence_ms,
    )
    return roles * start_reserve_ms(startup_timeout_s) + request


def admission_terms(
    policy: SessionPolicy,
    *,
    wall_time_ms,
    deadline_grace_ms,
    hard_timeout_ms,
    startup_timeout_s,
    roles: int = 1,
):
    """The terms of the admission rule as recorded in a manifest and a session receipt."""
    return {
        "rule": ADMISSION_RULE_ID,
        "deadline_kind": "host_monotonic_window" if policy.deadline_monotonic is not None else "none",
        "wall_time_ms": wall_time_ms,
        "deadline_grace_ms": deadline_grace_ms,
        "hard_timeout_ms_without_wall_budget": hard_timeout_ms,
        "kill_wait_ms": WORKER_KILL_WAIT_S * 1000.0,
        "persistence_ms": policy.persistence_ms,
        "request_reserve_ms": request_reserve_ms(
            wall_time_ms=wall_time_ms,
            deadline_grace_ms=deadline_grace_ms,
            hard_timeout_ms=hard_timeout_ms,
            persistence_ms=policy.persistence_ms,
        ),
        "worker_start_reserve_ms": start_reserve_ms(startup_timeout_s),
        "setup_reserve_ms": session_setup_reserve_ms(
            roles=roles,
            wall_time_ms=wall_time_ms,
            deadline_grace_ms=deadline_grace_ms,
            hard_timeout_ms=hard_timeout_ms,
            startup_timeout_s=startup_timeout_s,
            persistence_ms=policy.persistence_ms,
        ),
        "applies_to": ["initial_warmup_request", "rewarm_request", "measured_request", "worker_restart"],
        "on_insufficient_window": "close the session cleanly; defer the remaining tasks, in order, to the next window",
    }
