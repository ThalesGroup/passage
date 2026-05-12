# MIT License
#
# Copyright (c) 2026 THALES

"""Reference planners used to demonstrate the family-agnostic protocol.

These are NOT the headline baselines reported in the paper (those live
in ``passage-vision`` and ``passage-graph``). They exist purely to
illustrate that any candidate implementing :class:`Planner` is scored
by the same canonical pipeline: dense-vision / GNN / classical /
hand-crafted heuristic — same code path.
"""

from passage_eval.baselines.classical import (
    AStarPlanner,
    GreedyToGoalPlanner,
)

__all__ = [
    "AStarPlanner",
    "GreedyToGoalPlanner",
]
