"""Small complete CPU procedure: unweighted eight-neighbour breadth-first search.

This is a packaging example. It seeks reachability, not minimum elevation cost.
"""
from collections import deque
from passage_eval.protocol.contracts import PlanResult


class MiniPlanner:
    def __init__(self, *, device='cpu'):
        if device != 'cpu':
            raise ValueError('MiniPlanner supports CPU only')

    def plan(self, request):
        start, goal = request.start, request.goal
        height, width = request.obstacles.shape
        frontier = deque([start])
        parents = {start: None}
        work = dict(expansions=0, queue_operations=1, live_states=1)

        def finish(status, path=None):
            return PlanResult(path=[] if path is None else path, status=status,
                              work=dict(work), stage_work={'primary': dict(work)},
                              precision='integer-search')

        while frontier:
            remaining = request.remaining_wall_ms()
            if remaining is not None and remaining <= 0:
                return finish('timeout')
            budget = request.budget
            if ((budget.max_expansions is not None and work['expansions'] >= budget.max_expansions)
                    or (budget.max_queue_operations is not None and work['queue_operations'] >= budget.max_queue_operations)):
                return finish('budget_exhausted')
            current = frontier.popleft()
            work['queue_operations'] += 1
            work['expansions'] += 1
            if current == goal:
                path = []
                while current is not None:
                    path.append(current)
                    current = parents[current]
                return finish('ok', path[::-1])
            for dr, dc in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbour = current[0] + dr, current[1] + dc
                r, c = neighbour
                if not (0 <= r < height and 0 <= c < width) or request.obstacles[r, c] or neighbour in parents:
                    continue
                if ((budget.max_queue_operations is not None and work['queue_operations'] >= budget.max_queue_operations)
                        or (budget.max_live_states is not None and len(parents) >= budget.max_live_states)):
                    return finish('budget_exhausted')
                parents[neighbour] = current
                frontier.append(neighbour)
                work['queue_operations'] += 1
                work['live_states'] = len(parents)
        return finish('no_path')

    def close(self):
        """No model, worker thread or device allocation is retained."""
