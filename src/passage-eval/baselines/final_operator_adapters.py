# MIT License
#
# Copyright (c) 2026 THALES

"""Adapters from the frozen candidate decoders to the supervised worker API.

These adapters expose only the predicted field, obstacles and endpoints to the
decoders. They do not implement or change either frozen algorithm. Resource
limits that the frozen kernels cannot enforce are refused before execution;
the caller must enforce wall time outside the worker. A Reg expansion limit is
accepted through :class:`~passage_eval.baselines.final_bounded.RegFinalBounded`
when it is at least the certified bound N + 1.
"""

from __future__ import annotations

from passage_eval.baselines.final_bounded import RegFinalBounded, check_decoder_budget
from passage_eval.baselines.final_decoders import reg_bellman_descent, seg_logit_astar
from passage_eval.protocol.contracts import BudgetSpec, PlanResult


class SegFinalDecoder:
    """Use the fixed raw-logit A* with its native expansion limit."""

    def decode(self, field, start, goal, obstacles, budget: BudgetSpec) -> PlanResult:
        check_decoder_budget(budget)
        result = seg_logit_astar(field, obstacles, start, goal, max_expansions=budget.max_expansions)
        return PlanResult(
            result["path"],
            status=result["status"],
            work={"expansions": result["expansions"]},
            diagnostics={"fallback": False, "decoder": "seg-raw-logit-astar"},
        )


class RegFinalDecoder:
    """Use the fixed Bellman descent and goal-rooted Dijkstra fallback.

    The native kernel's expansion limit applies only to its fallback, so it
    cannot implement an arbitrary global supervised expansion budget and is not
    mapped to one. A global limit of at least N + 1 (N = number of cells) bounds
    descent plus fallback work for every input; it is delegated to
    :class:`~passage_eval.baselines.final_bounded.RegFinalBounded`, which runs
    the same frozen kernel and refuses a smaller limit. The returned work
    counter includes both descent and fallback work.
    """

    def decode(self, field, start, goal, obstacles, budget: BudgetSpec) -> PlanResult:
        check_decoder_budget(budget)
        if budget.max_expansions is not None:
            return RegFinalBounded().decode(field, start, goal, obstacles, budget)
        result = reg_bellman_descent(field, obstacles, start, goal)
        descent, fallback = result["descent_steps"], result["fallback_expansions"]
        stage_work = {"primary": {"expansions": descent}}
        if result["fallback"]:
            stage_work["fallback"] = {"expansions": fallback}
        return PlanResult(
            result["path"],
            status=result["status"],
            stage="fallback" if result["fallback"] else "primary",
            work={"expansions": descent + fallback},
            stage_work=stage_work,
            diagnostics={"fallback": result["fallback"], "decoder": "reg-bellman-descent"},
        )
