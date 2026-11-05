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

"""Integrate an existing breadth-first search without the reference solver.

The search optimizes step count, so physical-cost regret may be positive. Replace
``breadth_first_search`` with your algorithm and retain the ``plan`` boundary.
All attempts obey the same absolute deadline and charge actual queue/state work.
This CPU example establishes harness interoperability, not competitive latency.
"""
from collections import deque
import time

from passage_eval.protocol.contracts import PlanRequest, PlanResult


def breadth_first_search(request: PlanRequest) -> PlanResult:
    deadline = request.deadline_monotonic
    if deadline is None and request.budget.wall_time_ms is not None:
        deadline = time.perf_counter() + request.budget.wall_time_ms / 1000
    queue = deque([request.start])
    parents = {request.start: None}
    expansions, operations, peak = 0, 1, 1
    budget = request.budget

    def terminal(path, status):
        return PlanResult(path, status=status, precision='integer_coordinates',
                          work={'expansions': expansions, 'queue_operations': operations,
                                'live_states': peak},
                          stage_work={'primary': {'expansions': expansions, 'queue_operations': operations,
                                                  'live_states': peak}},
                          diagnostics={'algorithm': 'breadth_first_search', 'objective': 'step_count'})

    height, width = request.elevation.shape
    while queue:
        if deadline is not None and time.perf_counter() >= deadline:
            return terminal([], 'timeout')
        if ((budget.max_expansions is not None and expansions >= budget.max_expansions)
                or (budget.max_queue_operations is not None and operations >= budget.max_queue_operations)):
            return terminal([], 'budget_exhausted')
        current = queue.popleft()
        operations += 1
        expansions += 1
        if current == request.goal:
            path = [current]
            while parents[path[-1]] is not None:
                path.append(parents[path[-1]])
            return terminal(path[::-1], 'ok')
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nxt = current[0] + dr, current[1] + dc
                if ((dr == 0 and dc == 0) or not (0 <= nxt[0] < height and 0 <= nxt[1] < width)
                        or request.obstacles[nxt] or nxt in parents):
                    continue
                if ((budget.max_queue_operations is not None and operations >= budget.max_queue_operations)
                        or (budget.max_live_states is not None and len(parents) >= budget.max_live_states)):
                    return terminal([], 'budget_exhausted')
                parents[nxt] = current
                queue.append(nxt)
                operations += 1
                peak = max(peak, len(parents))
    return terminal([], 'no_path')


class ExistingAlgorithm:
    """CLI entry: ``examples.external_algorithm:ExistingAlgorithm``."""

    def plan(self, request: PlanRequest) -> PlanResult:
        return breadth_first_search(request)
