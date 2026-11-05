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

"""Scoring stops when a legal route costs less than the qualified optimum (judge R6); synthetic scenes only."""

import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest
from passage.audit import independently_qualified_reference_for_scene
from passage_eval.execution import NegativeRegretHalt, run_requests, score_run_artifact
from passage_eval.metrics.cost import path_cost
from passage_eval.protocol.contracts import PlanRequest, PlanResult

DIAGONAL = [(0, 0), (1, 1), (2, 2)]
DETOUR = [(0, 0), (0, 1), (1, 2), (2, 2)]
MANIFEST = {"reference_policy": "independent_analytic_reference", "reference_cohort_id": "c"}


def _rehash(receipt):
    payload = {key: value for key, value in receipt.items() if key not in {"reference_id", "hash"}}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
    return dict(receipt, reference_id=digest.hexdigest(), hash=digest.hexdigest())


def _reference(scene, path=None, cost_factor=1.0):
    """A self-consistent qualified record whose reference route is ``path`` (default: the true optimum)."""
    record = independently_qualified_reference_for_scene(scene, cohort_id="c", max_expansions=9)
    if path is None:
        return record
    cost = path_cost(scene.elevation, path, weight=scene.cost_weight, cost_model=scene.cost_model) * cost_factor
    receipt = _rehash(dict(record["reference"], path=[list(p) for p in path], cost=cost))
    return dict(
        record,
        optimal_path=[list(p) for p in path],
        optimal_cost=cost,
        reference=receipt,
        reference_id=receipt["reference_id"],
    )


def _run(tmp_path, scene, route, status_ok=True):
    class Planner:
        def plan(self, request):
            return PlanResult(route)

    return run_requests(Planner(), [PlanRequest.from_scene(scene)], output_dir=tmp_path, manifest=dict(MANIFEST))


def test_a_legal_route_cheaper_than_the_qualified_optimum_stops_scoring(metre_scene, tmp_path):
    scene = metre_scene
    run = _run(tmp_path / "cheaper", scene, DIAGONAL)
    forged = {scene.sample_id: _reference(scene, DETOUR)}  # a non-optimal "reference": the detour
    with pytest.raises(NegativeRegretHalt, match="1 legal route.*no win, no exclusion"):
        score_run_artifact(run, [scene], reference_store=forged)


def test_the_true_optimum_and_a_route_within_the_tolerance_are_scored(metre_scene, tmp_path):
    scene = metre_scene
    run = _run(tmp_path / "equal", scene, DIAGONAL)
    metrics, arrays = score_run_artifact(run, [scene], reference_store={scene.sample_id: _reference(scene)})
    assert arrays["regret"].tolist() == [0.0]
    # a reference within the graph tolerance above the candidate (rtol 1e-7) is not a diagnosis
    near = {scene.sample_id: _reference(scene, DIAGONAL, cost_factor=1 + 1e-8)}
    _, close = score_run_artifact(_run(tmp_path / "near", scene, DIAGONAL), [scene], reference_store=near)
    assert close["regret"][0] == pytest.approx(-1e-8, abs=1e-12)


def test_an_illegal_cheap_route_is_a_penalised_failure_not_a_diagnosis(metre_scene, tmp_path):
    blocked = np.zeros((3, 3), dtype=bool)
    blocked[1, 1] = True
    scene = replace(metre_scene, obstacles=blocked, oracle_path=DETOUR, oracle_cost=None)
    run = _run(tmp_path / "illegal", scene, DIAGONAL)  # crosses the obstacle at (1, 1): illegal
    _, arrays = score_run_artifact(run, [scene], reference_store={scene.sample_id: _reference(scene)})
    assert arrays["valid_path"].tolist() == [False] and arrays["regret"][0] == pytest.approx(1e6 - 1)
