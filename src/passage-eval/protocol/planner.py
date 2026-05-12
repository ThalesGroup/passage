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

"""Approach-agnostic planner interface for the PASSAGE benchmark.

The PASSAGE paper (App. A.2 / §4) declares the protocol to be
*family-agnostic*: any candidate planner --- dense-vision, graph-neural,
reinforcement-learning, classical --- only has to map the input tensor
to an ordered $s \\to g$ path. This module is the executable form of
that statement.

A concrete planner implements either :meth:`Planner.predict_path`
(single sample) or :meth:`Planner.predict_paths` (batch). Default
implementations forward to the other so most subclasses only need one.

The :func:`score_planner` driver runs a planner against an iterable of
PASSAGE-shaped scenes and emits the same primary-metric bundle that
:func:`passage_eval.metrics.summary.compute_primary_metrics` returns,
ready to drop into the headline table or feed
:func:`passage_eval.protocol.discipline.multiseed_paired_bootstrap`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

from passage_eval._typing import PathLike
from passage_eval.metrics.summary import compute_primary_metrics


__all__ = [
    "Scene",
    "PlannerOutput",
    "Planner",
    "score_planner",
]


@dataclass(frozen=True)
class Scene:
    """One PASSAGE evaluation scene.

    Mirrors the per-sample fields that the paper's metric definitions
    reference; populated by the dataset loader. ``oracle_path`` and
    ``oracle_cost`` come from the released A$^{\\star}$ labels and are
    treated as ground truth.
    """

    sample_id: str
    elevation: np.ndarray            #: (H, W) raw elevation, metres.
    obstacles: np.ndarray            #: (H, W) binary obstacle mask, 1 = forbidden.
    start: tuple[int, int]           #: (i, j) start grid cell.
    goal: tuple[int, int]            #: (i, j) goal grid cell.
    cost_model: str                  #: one of {"distance", "elevation", "absolute_elevation", "slope_uphill", "energy"}
    cost_weight: float               #: per-cost-model weight $w$ from the calibration JSON.
    oracle_path: PathLike            #: ordered list of (i, j) cells from A* on the analytic cost.
    oracle_cost: float               #: scalar cost of ``oracle_path`` under the analytic cost model.
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class PlannerOutput:
    """One predicted path + its cost under the scene's cost model.

    A planner that does not compute its own cost may set
    ``predicted_cost = float("nan")`` and let
    :func:`score_planner` re-evaluate it from ``predicted_path``.
    """

    predicted_path: PathLike
    predicted_cost: float


class Planner(ABC):
    """Abstract base class for any PASSAGE-scoreable planner.

    Subclasses MUST implement either :meth:`predict_path` or
    :meth:`predict_paths`. The default implementations forward to the
    other so subclasses only need one.

    The interface is intentionally minimal: a planner only needs to
    consume a :class:`Scene` and emit an ordered $s \\to g$ list of
    cells. The framework does not constrain the internal procedure —
    dense-vision, graph search, message-passing, RL roll-out, classical
    A$^{\\star}$ — anything that can produce a cell list is in scope.
    """

    #: Short identifier surfaced in result CSVs and pairwise decisions.
    name: str = "planner"

    @abstractmethod
    def predict_path(self, scene: Scene) -> PlannerOutput:
        """Return the predicted path + scalar cost for one scene."""

    def predict_paths(self, scenes: Sequence[Scene]) -> list[PlannerOutput]:
        """Vectorised default — override for batched inference."""
        return [self.predict_path(scene) for scene in scenes]


def score_planner(
    planner: Planner,
    scenes: Iterable[Scene],
    *,
    batch_size: int | None = None,
) -> dict[str, float]:
    """Score a :class:`Planner` against an iterable of :class:`Scene`s.

    Wraps :func:`compute_primary_metrics` so the same primary-metric
    dict is returned, regardless of whether the underlying planner is
    a CNN, a GNN, an RL policy or a classical search. ``batch_size``
    controls how many scenes are handed to ``predict_paths`` at a
    time; ``None`` materialises the full iterable in memory and is
    fine for the 1024-sample paper val split.
    """
    materialised: list[Scene] = list(scenes)
    if not materialised:
        raise ValueError("score_planner: empty scene iterable.")

    outputs: list[PlannerOutput] = []
    if batch_size is None:
        outputs = planner.predict_paths(materialised)
    else:
        for off in range(0, len(materialised), batch_size):
            outputs.extend(planner.predict_paths(materialised[off : off + batch_size]))

    if len(outputs) != len(materialised):
        raise RuntimeError(
            f"{type(planner).__name__}.predict_paths emitted "
            f"{len(outputs)} outputs for {len(materialised)} scenes."
        )

    predicted_paths = [out.predicted_path for out in outputs]
    oracle_paths = [scene.oracle_path for scene in materialised]
    predicted_costs = np.asarray([out.predicted_cost for out in outputs], dtype=np.float64)
    oracle_costs = np.asarray([scene.oracle_cost for scene in materialised], dtype=np.float64)
    obstacles = [scene.obstacles for scene in materialised]
    starts = [scene.start for scene in materialised]
    goals = [scene.goal for scene in materialised]

    return compute_primary_metrics(
        predicted_paths=predicted_paths,
        oracle_paths=oracle_paths,
        predicted_costs=predicted_costs,
        oracle_costs=oracle_costs,
        obstacles=obstacles,
        starts=starts,
        goals=goals,
    )
