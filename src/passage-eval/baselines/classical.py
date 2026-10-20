# MIT License
#
# Copyright (c) 2026 THALES

"""Reference analytic, greedy and non-trainable policy adapters.

AStarPlanner uses the same analytic grid backend and per-scene weight as release
label generation. GreedyToGoalPlanner ignores terrain costs. RandomPolicyPlanner
implements Appendix A.2's uniform-neighbour/A* closure conformance harness.
These adapters depend on NumPy and the generator backend, without PyTorch.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from passage.pathfinding_utils import _grid_astar
from passage_eval.metrics.cost import canonical_cost_model, path_cost
from passage_eval.protocol.contracts import PlanRequest, PlanResult
from passage_eval.protocol.planner import Planner, PlannerOutput, Scene


__all__ = ["AStarPlanner", "GreedyToGoalPlanner", "RandomPolicyPlanner", "BoundedExactPlanner", "CanonicalAStarPlanner"]


_NEIGHBOUR_OFFSETS_8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)
_NEIGHBOUR_DISTS_8 = tuple(math.sqrt(dr * dr + dc * dc) for dr, dc in _NEIGHBOUR_OFFSETS_8)


def _heuristic_l2(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _path_cost(elevation: np.ndarray, path: list[tuple[int, int]], *, weight: float, cost_model: str) -> float:
    return path_cost(elevation, path, weight=weight, cost_model=cost_model)


@dataclass
class AStarPlanner(Planner):
    """Exact A$^{\\star}$ on the analytic cost grid.

    Implements :class:`Planner`. Acts as the PASSAGE oracle when
    invoked with the same cost model and weight as the dataset's
    label generation pipeline; with a perturbed weight it becomes a
    sub-optimal but valid candidate planner used as a sanity-check
    baseline.

    Uses the same grid backend, Euclidean heuristic and metre-valued edge
    costs as release generation.
    """

    name: str = "astar_classical"

    def predict_path(self, scene: Scene) -> PlannerOutput:
        path = _grid_astar(
            scene.elevation,
            scene.start,
            scene.goal,
            weight=scene.cost_weight,
            diagonal_movement="always",
            heuristic_name="euclidean",
            cost_model=canonical_cost_model(scene.cost_model),
            blocked_mask=np.asarray(scene.obstacles, dtype=bool),
        )
        cost = _path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
        return PlannerOutput(predicted_path=path, predicted_cost=cost)


@dataclass
class GreedyToGoalPlanner(Planner):
    """Diagonal-greedy descent toward the goal, ignoring costs.

    Useful as a weak-but-valid baseline. Always reaches the goal as
    long as no obstacle blocks every diagonal-then-axial route, and
    its cost is always $\\geq$ A$^{\\star}$ — so the cost-regret
    metric should always be positive when this planner is scored
    against the oracle.
    """

    name: str = "greedy_classical"
    max_steps_factor: int = 8

    def predict_path(self, scene: Scene) -> PlannerOutput:
        elevation = scene.elevation
        obstacles = scene.obstacles
        H, W = elevation.shape
        start = scene.start
        goal = scene.goal

        path: list[tuple[int, int]] = [start]
        current = start
        max_steps = self.max_steps_factor * (H + W)
        for _ in range(max_steps):
            if current == goal:
                break
            best = None
            best_dist = float("inf")
            cr, cc = current
            for dr, dc in _NEIGHBOUR_OFFSETS_8:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if obstacles[nr, nc]:
                    continue
                if (nr, nc) in path:
                    continue
                d = _heuristic_l2((nr, nc), goal)
                if d < best_dist:
                    best_dist = d
                    best = (nr, nc)
            if best is None:
                break
            path.append(best)
            current = best
        cost = _path_cost(elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
        return PlannerOutput(predicted_path=path, predicted_cost=cost)


@dataclass
class RandomPolicyPlanner(Planner):
    """Non-trainable App. A.2 harness: uniform legal neighbours, A* closure.

    The rollout is bounded by ``max_steps``. A fresh instance with the same
    seed reproduces the same sequence; this adapter makes no learned-RL claim.
    """

    name: str = "random_policy_harness"
    seed: int | None = None
    max_steps: int = 32

    def __post_init__(self):
        if self.max_steps < 0:
            raise ValueError("max_steps must be nonnegative")
        # An explicit seed is useful in direct calls; otherwise the CLI's
        # seeded numpy state supplies an independent reproducible run seed.
        seed = self.seed if self.seed is not None else int(np.random.randint(0, 2**32))
        self._rng = np.random.default_rng(seed)

    def predict_path(self, scene: Scene) -> PlannerOutput:
        height, width = scene.elevation.shape
        path = [scene.start]
        for _ in range(self.max_steps):
            if path[-1] == scene.goal:
                break
            row, col = path[-1]
            neighbours = [
                (row + dr, col + dc)
                for dr, dc in _NEIGHBOUR_OFFSETS_8
                if 0 <= row + dr < height and 0 <= col + dc < width and not scene.obstacles[row + dr, col + dc]
            ]
            if not neighbours:
                break
            path.append(neighbours[int(self._rng.integers(len(neighbours)))])
        if path[-1] != scene.goal:
            bridge = _grid_astar(
                scene.elevation,
                path[-1],
                scene.goal,
                weight=scene.cost_weight,
                diagonal_movement="always",
                heuristic_name="euclidean",
                cost_model=canonical_cost_model(scene.cost_model),
                blocked_mask=np.asarray(scene.obstacles, dtype=bool),
            )
            path.extend(bridge[1:])
        return PlannerOutput(
            path, _path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
        )


@dataclass
class BoundedExactPlanner:
    """Bounded Dijkstra prototype on the released graph, with charged work.

    Normalization, search and path reconstruction happen inside ``plan``. No
    release label or evaluator reference is consumed. This CPU prototype makes
    no claim about optimized native classical performance.
    """

    max_expansions: int = 10000
    name: str = "bounded_exact_dijkstra_prototype"

    def __post_init__(self):
        if isinstance(self.max_expansions, bool) or not isinstance(self.max_expansions, int) or self.max_expansions <= 0:
            raise ValueError("max_expansions must be a positive integer")

    def plan(self, request: PlanRequest) -> PlanResult:
        import heapq
        import time

        from passage_eval.metrics.cost import MODEL_CODES, directed_edge_cost
        started = time.perf_counter()
        deadline = request.deadline_monotonic
        if deadline is None and request.budget.wall_time_ms is not None:
            deadline = started + request.budget.wall_time_ms / 1000
        terrain = np.asarray(request.elevation, dtype=np.float64)
        minimum, maximum = float(terrain.min()), float(terrain.max())
        elevation_range, absolute_max = maximum - minimum, max(abs(minimum), abs(maximum))
        budget = request.budget
        expansion_limit = min(self.max_expansions, budget.max_expansions or self.max_expansions)
        best, parents = {request.start: 0.0}, {}
        queue = [(0.0, request.start)]
        expansions, operations, live_peak = 0, 1, 1

        def result(path, status):
            work = {"expansions": expansions, "queue_operations": operations, "live_states": live_peak}
            return PlanResult(path, status=status, work=work, precision="float64",
                              diagnostics={"algorithm": "bounded_dijkstra_prototype",
                                           "configured_max_expansions": self.max_expansions})

        height, width = request.elevation.shape
        while queue:
            if deadline is not None and time.perf_counter() >= deadline:
                return result([], "timeout")
            if expansions >= expansion_limit:
                return result([], "budget_exhausted")
            if budget.max_queue_operations is not None and operations >= budget.max_queue_operations:
                return result([], "budget_exhausted")
            distance, current = heapq.heappop(queue)
            operations += 1
            if distance != best[current]:
                continue
            expansions += 1
            if current == request.goal:
                path = [current]
                while path[-1] != request.start:
                    path.append(parents[path[-1]])
                return result(path[::-1], "ok")
            for (dr, dc), length in zip(_NEIGHBOUR_OFFSETS_8, _NEIGHBOUR_DISTS_8):
                nxt = current[0] + dr, current[1] + dc
                if not (0 <= nxt[0] < height and 0 <= nxt[1] < width) or request.obstacles[nxt]:
                    continue
                metres = 30.0 * length
                edge = directed_edge_cost(float(terrain[current]), float(terrain[nxt]), metres,
                                          float(request.cost_weight), MODEL_CODES[request.cost_model],
                                          minimum, elevation_range, absolute_max)
                proposal = distance + edge
                if not np.isfinite(edge) or not np.isfinite(proposal):
                    raise ValueError("nonfinite physical edge or accumulated path cost")
                if proposal < best.get(nxt, float("inf")):
                    if budget.max_queue_operations is not None and operations >= budget.max_queue_operations:
                        return result([], "budget_exhausted")
                    if nxt not in best and budget.max_live_states is not None and len(best) >= budget.max_live_states:
                        return result([], "budget_exhausted")
                    best[nxt], parents[nxt] = proposal, current
                    heapq.heappush(queue, (proposal, nxt))
                    operations += 1
                    live_peak = max(live_peak, len(best))
        return result([], "no_path")


@dataclass
class CanonicalAStarPlanner:
    """Current exact A* candidate on the shared ordinary-input graph.

    This invokes the deterministic canonical search on request terrain, never a
    saved reference. Expansion limits are enforced in the native search; wall,
    queue and live-state limits are checked at safe completion by the runner.
    Native calls remain cooperatively bounded and cannot be forcibly cancelled.
    Historical ``AStarPlanner`` remains a separate legacy procedure.
    """
    name: str = 'canonical_exact_astar'

    def plan(self, request: PlanRequest) -> PlanResult:
        from passage.reference import solve_reference
        remaining = request.remaining_wall_ms()
        if remaining is not None and remaining <= 0:
            return PlanResult([], status='timeout', precision='float64',
                              work={'expansions': 0, 'queue_operations': 0, 'max_live_states': 0})
        result = solve_reference(request.elevation, request.obstacles, request.start, request.goal,
                                 graph=request.graph, max_expansions=request.budget.max_expansions,
                                 algorithm='astar')
        return PlanResult(result.path, status={'exact': 'ok', 'no_path': 'no_path',
                                              'censored': 'budget_exhausted'}[result.status],
                          work=result.work, stage_work={'primary': result.work}, precision='float64',
                          diagnostics={'algorithm': 'canonical_astar', 'native_call': 'cooperative_safe_completion'})
