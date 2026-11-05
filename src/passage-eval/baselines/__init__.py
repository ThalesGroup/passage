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
    RandomPolicyPlanner,
    BoundedExactPlanner,
    CanonicalAStarPlanner,
)

__all__ = [
    "AStarPlanner",
    "GreedyToGoalPlanner",
    "RandomPolicyPlanner",
    "BoundedExactPlanner",
    "CanonicalAStarPlanner",
]
