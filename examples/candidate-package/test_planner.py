"""Small CPU behavior checks for the example; no dataset or measured results."""
import unittest
import numpy as np
from passage_eval.protocol.contracts import PlanRequest, BudgetSpec
try:
    from mini_planner import MiniPlanner
except ModuleNotFoundError:
    MiniPlanner = None


class MiniPlannerTests(unittest.TestCase):
    def request(self, obstacles=None, budget=None):
        return PlanRequest('example', np.zeros((3, 3)), np.zeros((3, 3), dtype=bool) if obstacles is None else obstacles,
                           (0, 0), (2, 2), 'elevation', 62.4, budget=budget or BudgetSpec())

    def test_returns_materialized_legal_route_around_blocked_cells(self):
        self.assertIsNotNone(MiniPlanner, 'the example planner must exist')
        blocked = np.zeros((3, 3), dtype=bool); blocked[1, 1] = True
        result = MiniPlanner(device='cpu').plan(self.request(blocked))
        from passage_eval.metrics.reliability import path_status
        valid, violated, _ = path_status(result.path, start=(0, 0), goal=(2, 2), obstacles=blocked)
        self.assertEqual(result.status, 'ok')
        self.assertTrue(valid); self.assertFalse(violated)
        self.assertGreater(result.work['expansions'], 0)

    def test_reports_unreachable_and_expansion_budget_without_fake_route(self):
        self.assertIsNotNone(MiniPlanner, 'the example planner must exist')
        blocked = np.zeros((3, 3), dtype=bool); blocked[1, :] = True
        unreachable = MiniPlanner(device='cpu').plan(self.request(blocked))
        self.assertEqual(unreachable.status, 'no_path'); self.assertEqual(unreachable.path, [])
        limited = MiniPlanner(device='cpu').plan(self.request(budget=BudgetSpec(max_expansions=1)))
        self.assertEqual(limited.status, 'budget_exhausted'); self.assertEqual(limited.path, [])
        self.assertEqual(limited.work['expansions'], 1)


if __name__ == '__main__':
    unittest.main()
