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

A concrete planner implements :meth:`Planner.predict_path` for one sample.
It may override :meth:`Planner.predict_paths` for batched inference.

The :func:`score_planner` driver runs a planner against an iterable of
PASSAGE-shaped scenes and emits the same primary-metric bundle that
:func:`passage_eval.metrics.summary.compute_primary_metrics` returns,
ready to drop into the headline table or feed
:func:`passage_eval.protocol.discipline.multiseed_paired_bootstrap`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Iterable, Sequence

import numpy as np

from passage_eval._typing import PathLike
from passage_eval.metrics.summary import aggregate_primary_metrics, per_sample_metrics
from passage_eval.metrics.cost import path_cost


__all__ = [
    "Scene",
    "PlannerOutput",
    "Planner",
    "score_planner",
    "evaluate_planner",
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
    elevation: np.ndarray  #: (H, W) raw elevation, metres.
    obstacles: np.ndarray  #: (H, W) binary obstacle mask, 1 = forbidden.
    start: tuple[int, int]  #: (i, j) start grid cell.
    goal: tuple[int, int]  #: (i, j) goal grid cell.
    cost_model: str  #: one of {"distance", "elevation", "absolute_elevation", "slope_uphill", "energy"}
    cost_weight: float  #: per-cost-model weight $w$ from the calibration JSON.
    oracle_path: PathLike  #: ordered list of (i, j) cells from A* on the analytic cost.
    oracle_cost: float  #: scalar cost of ``oracle_path`` under the analytic cost model.
    metadata: dict = field(default_factory=dict)
    input_tensor: np.ndarray | None = None  #: Released (H,W,3) normalized input, without labels.


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

    Subclasses implement :meth:`predict_path`. The default :meth:`predict_paths`
    calls it for each scene; override that method for batched inference.

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


def candidate_scene(scene: Scene) -> Scene:
    """Isolate candidate-owned inputs from the evaluator's labels and arrays."""
    from passage_eval.protocol.contracts import PlanRequest
    ordinary = PlanRequest.from_scene(scene)
    return replace(
        scene,
        elevation=ordinary.elevation,
        obstacles=ordinary.obstacles,
        input_tensor=ordinary.input_tensor,
        oracle_path=[],
        oracle_cost=float("nan"),
        metadata=ordinary.metadata,
    )


def evaluate_planner(
    planner: Planner,
    scenes: Iterable[Scene],
    *,
    batch_size: int | None = None,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
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

    if batch_size is not None and batch_size <= 0:
        raise ValueError("batch_size must be positive")
    # Candidate inputs carry no oracle path or oracle costs, including metadata.
    outputs: list[PlannerOutput] = []
    if batch_size is None:
        outputs = planner.predict_paths([candidate_scene(s) for s in materialised])
    else:
        for off in range(0, len(materialised), batch_size):
            outputs.extend(planner.predict_paths([candidate_scene(s) for s in materialised[off : off + batch_size]]))

    if len(outputs) != len(materialised):
        raise RuntimeError(
            f"{type(planner).__name__}.predict_paths emitted {len(outputs)} outputs for {len(materialised)} scenes."
        )

    predicted_paths = [validated_raw_path(out.predicted_path) for out in outputs]
    oracle_paths = [scene.oracle_path for scene in materialised]
    # Candidate-reported costs are informational only: every scored cost is
    # recomputed under the scene's canonical analytic model and metre units.
    predicted_costs = np.asarray(
        [
            path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model)
            for scene, path in zip(materialised, predicted_paths)
        ],
        dtype=np.float64,
    )
    oracle_costs = np.asarray([scene.oracle_cost for scene in materialised], dtype=np.float64)
    obstacles = [scene.obstacles for scene in materialised]
    starts = [scene.start for scene in materialised]
    goals = [scene.goal for scene in materialised]

    arrays = per_sample_metrics(
        predicted_paths=predicted_paths,
        oracle_paths=oracle_paths,
        predicted_costs=predicted_costs,
        oracle_costs=oracle_costs,
        obstacles=obstacles,
        starts=starts,
        goals=goals,
    )

    arrays.update(
        sample_id=np.array([s.sample_id for s in materialised]),
        obs_density=np.array([float(np.mean(s.obstacles)) for s in materialised]),
        elev_std=np.array([float(np.std(s.elevation)) for s in materialised]),
        oracle_len=np.array([len(s.oracle_path) for s in materialised], dtype=np.int64),
        metric_contract=np.array("neurips-2026-submission-v1"),
    )
    return aggregate_primary_metrics(arrays), arrays


def score_planner(planner: Planner, scenes: Iterable[Scene], *, batch_size: int | None = None) -> dict[str, float]:
    """Compute headline metrics with costs derived from the emitted paths."""
    metrics, _ = evaluate_planner(planner, scenes, batch_size=batch_size)
    return metrics


def validated_raw_path(path: PathLike) -> np.ndarray:
    """Expose a numeric (N,2) route without reshaping, rounding or clipping.

    Invalid shape/type yields an empty scoring path. The complete-request
    runner separately persists numeric raw outputs even when malformed.
    """
    try:
        points = np.asarray(path)
        if points.size == 0:
            return np.empty((0, 2), dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or points.dtype.kind not in 'iuf':
            return np.empty((0, 2), dtype=np.float64)
        # Check coordinates in their original precision before downstream
        # metric helpers cast to float64. A tiny fractional long-double cell
        # must not become an integer through a lossy conversion.
        if not np.isfinite(points).all() or not np.equal(points, np.floor(points)).all():
            return np.empty((0, 2), dtype=np.float64)
        return points.copy()
    except (TypeError, ValueError):
        return np.empty((0, 2), dtype=np.float64)
