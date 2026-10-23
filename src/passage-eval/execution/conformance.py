"""Bounded installed-harness checks; CPU contracts never certify a device."""

from dataclasses import replace

import numpy as np

from passage_eval.execution.runner import run_requests
from passage_eval.execution.scoring import score_run_artifact
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult, STATUSES


def harness_conformance(scene) -> dict[str, str]:
    """Exercise ordinary-input isolation, statuses, malformed paths and budgets.

    Reference access occurs only in the offline scorer. These checks exercise
    this installed runner; they do not attest a candidate's private dependencies,
    reference optimality over a population, device behavior or exclusive timing.
    """
    checks = {}
    original = scene.elevation.copy()
    poisoned = replace(
        scene,
        metadata={
            "split": "validation",
            "oracle_cost": 999,
            "calibration": {"min": 0.0, "max": 1.0, "cost_to_go_max": 999},
        },
    )
    route = [scene.start, (1, 1), scene.goal]

    class Isolated:
        def plan(self, request):
            if (
                hasattr(request, "oracle_cost")
                or hasattr(request, "oracle_path")
                or "oracle_cost" in request.metadata
                or "cost_to_go_max" in request.metadata["calibration"]
            ):
                raise ValueError("reference fields crossed the ordinary-input boundary")
            request.elevation[:] = 999
            return PlanResult(route)

    isolation = run_requests(Isolated(), [PlanRequest.from_scene(poisoned)])
    if isolation.records[0]["status"] != "ok" or not np.array_equal(scene.elevation, original):
        raise ValueError("ordinary-input isolation conformance failed")
    checks["ordinary_input_no_reference_fields"] = "passed"

    class Terminal:
        def __init__(self, status):
            self.status = status

        def plan(self, request):
            return PlanResult(route, status=self.status)

    for status in STATUSES - {"ok"}:
        artifact = run_requests(Terminal(status), [PlanRequest.from_scene(scene)])
        metrics, arrays = score_run_artifact(artifact, [scene], require_qualified_reference=False)
        if artifact.records[0]["status"] != status or metrics["legal_route_rate"] != 0 or arrays["regret"][0] != 999999:
            raise ValueError("terminal failure accounting conformance failed")
    checks["terminal_failure_accounting"] = "passed"

    ticks = [0.0]

    class Late:
        def plan(self, request):
            if request.deadline_monotonic != 0.001:
                raise ValueError("request admission deadline is missing")
            ticks[0] = 0.002
            return PlanResult(route)

    request = PlanRequest.from_scene(scene, budget=BudgetSpec(wall_time_ms=1))
    late = run_requests(Late(), [request], clock=lambda: ticks[0])
    if late.records[0]["status"] != "timeout" or late.records[0]["drain_ms"] != 1:
        raise ValueError("absolute deadline conformance failed")
    checks["absolute_deadline"] = "passed"

    class Stages:
        def plan(self, request):
            return PlanResult(
                route,
                stage="fallback",
                stage_work={"primary": {"expansions": 2}, "repair": {"expansions": 3}, "fallback": {"expansions": 4}},
            )

    budgeted = run_requests(Stages(), [PlanRequest.from_scene(scene, budget=BudgetSpec(max_expansions=8))])
    if budgeted.records[0]["status"] != "budget_exhausted" or budgeted.records[0]["work"]["expansions"] != 9:
        raise ValueError("stage work conformance failed")
    checks["primary_repair_fallback_work"] = "passed"

    class Malformed:
        def plan(self, request):
            return PlanResult([scene.start, (1.25, 1), scene.goal])

    if run_requests(Malformed(), [PlanRequest.from_scene(scene)]).records[0]["status"] != "invalid_path":
        raise ValueError("raw coordinate conformance failed")
    checks["no_rounding_or_clipping"] = "passed"
    return checks
