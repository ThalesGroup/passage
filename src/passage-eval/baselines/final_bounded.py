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

"""Bounded wrappers for the two immutable final decoders.

The wrappers call the frozen kernels of :mod:`passage_eval.baselines.final_decoders` without modifying them.
Expansion counters are exact. Queue and memory counters cannot be inferred from the immutable kernels and
are therefore rejected, not fabricated.

For Reg, a global expansion limit of at least N + 1 (N = number of cells) is a proven upper bound on descent
steps plus fallback expansions, so such a limit can never be reached and the frozen kernel runs unchanged.
Smaller global limits are rejected before the kernel runs, because the kernel does not enforce them: its own
limit applies to the fallback only, and changing it would change its semantics.
"""

from __future__ import annotations

import numpy as np

from passage_eval.baselines.final_decoders import reg_bellman_descent, seg_logit_astar
from passage_eval.protocol.contracts import BudgetSpec, PlanResult


def check_decoder_budget(budget: BudgetSpec) -> None:
    """Refuse the work limits that the frozen kernels cannot observe (queue operations, live states)."""
    if not isinstance(budget, BudgetSpec):
        raise TypeError("budget must be a BudgetSpec")
    unsupported = [name for name in ("max_queue_operations", "max_live_states") if getattr(budget, name) is not None]
    if unsupported:
        raise ValueError(f"unsupported decoder work budget(s): {', '.join(unsupported)}")


def _owned_inputs(field, obstacles):
    """Freeze private decoder inputs for the entire call and work-bound proof."""
    grid = np.array(field, dtype=np.float64, order="C", copy=True)
    blocked = np.array(obstacles, dtype=np.bool_, order="C", copy=True)
    grid.setflags(write=False)
    blocked.setflags(write=False)
    return grid, blocked


def reg_certified_expansion_bound(n_cells: int) -> int:
    """Proven upper bound on Reg descent steps plus fallback expansions on a grid of ``n_cells`` cells."""
    return int(n_cells) + 1


class SegFinalBounded:
    """Seg raw-logit A* with its native expansion limit (it retires every cell at most once)."""

    def decode(self, field, start, goal, obstacles, budget: BudgetSpec) -> PlanResult:
        check_decoder_budget(budget)
        start, goal = tuple(int(x) for x in start), tuple(int(x) for x in goal)
        grid, blocked = _owned_inputs(field, obstacles)
        result = seg_logit_astar(grid, blocked, start, goal, max_expansions=budget.max_expansions)
        return PlanResult(
            result["path"],
            status=result["status"],
            work={"expansions": result["expansions"]},
            diagnostics={"fallback": False, "decoder": "seg-raw-logit-astar"},
        )


class RegFinalBounded:
    """Reg Bellman descent with its frozen fallback; accepts a global expansion limit of at least N + 1."""

    def decode(self, field, start, goal, obstacles, budget: BudgetSpec) -> PlanResult:
        check_decoder_budget(budget)
        start, goal = tuple(int(x) for x in start), tuple(int(x) for x in goal)
        grid, blocked = _owned_inputs(field, obstacles)
        if grid.ndim != 2 or not grid.size:
            raise ValueError("expected one nonempty 2D predicted field")
        bound = reg_certified_expansion_bound(grid.size)
        if budget.max_expansions is not None and budget.max_expansions < bound:
            raise ValueError(f"global max_expansions below the certified upper bound {bound}")
        # Preserve the exact frozen fallback limit of N even if the global
        # limit is larger: lowering or raising it would alter its semantics.
        result = reg_bellman_descent(grid, blocked, start, goal)
        descent, fallback = result["descent_steps"], result["fallback_expansions"]
        if descent + fallback > bound:
            raise RuntimeError("frozen decoder exceeded the proven expansion bound")
        stage_work = {"primary": {"expansions": descent}}
        if result["fallback"]:
            stage_work["fallback"] = {"expansions": fallback}
        return PlanResult(
            result["path"],
            status=result["status"],
            stage="fallback" if result["fallback"] else "primary",
            work={"expansions": descent + fallback},
            stage_work=stage_work,
            diagnostics={
                "fallback": result["fallback"],
                "decoder": "reg-bellman-descent",
                "global_expansions_certified_upper_bound": bound,
            },
        )
