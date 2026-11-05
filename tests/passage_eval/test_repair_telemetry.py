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

"""An explicitly reported repair is not an unrepaired direct decode."""

import hashlib
import json
from dataclasses import replace
import numpy as np
import pytest

from passage_eval.execution import run_requests, read_run_artifact, score_run_artifact
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult


def test_repair_and_fallback_rates_are_distinct_and_keep_every_repeat(metre_scene, tmp_path):
    class Mixed:
        calls = 0

        def plan(self, request):
            self.calls += 1
            stages = [
                ("primary", {"primary": {"expansions": 2}}),
                ("repair", {"primary": {"expansions": 2}, "repair": {"expansions": 3}}),
                (
                    "fallback",
                    {"primary": {"expansions": 2}, "repair": {"expansions": 3}, "fallback": {"expansions": 4}},
                ),
                ("primary", {"primary": {"expansions": 2}, "repair": {"expansions": 0}}),
            ]
            stage, work = stages[self.calls - 1]
            return PlanResult([(0, 0), (1, 1), (2, 2)], stage=stage, stage_work=work)

    run_requests(Mixed(), [PlanRequest.from_scene(metre_scene)], repeats=4, quality_repeat_index=1, output_dir=tmp_path)
    loaded = read_run_artifact(tmp_path)
    metrics, arrays = score_run_artifact(loaded, [metre_scene], require_qualified_reference=False)
    np.testing.assert_array_equal(arrays["repair_used"], [False, True, True, True])
    np.testing.assert_array_equal(arrays["fallback"], [False, False, True, False])
    assert metrics["repair_rate"] == 1.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["direct_success_rate"] == 0.0
    assert metrics["all_repeats"]["repair_rate"] == pytest.approx(0.75)
    assert metrics["all_repeats"]["fallback_rate"] == pytest.approx(0.25)
    assert metrics["all_repeats"]["direct_success_rate"] == pytest.approx(0.25)


def test_failed_repair_still_counts_in_request_denominator(metre_scene):
    class FailedRepair:
        def plan(self, request):
            return PlanResult(
                [],
                status="no_path",
                stage="repair",
                stage_work={"primary": {"expansions": 2}, "repair": {"expansions": 1}},
            )

    artifact = run_requests(FailedRepair(), [PlanRequest.from_scene(metre_scene)])
    metrics, arrays = score_run_artifact(artifact, [metre_scene], require_qualified_reference=False)
    assert arrays["repair_used"].tolist() == [True]
    assert metrics["repair_rate"] == 1.0
    assert metrics["direct_success_rate"] == 0.0
    assert metrics["success_rate"] == 0.0


@pytest.mark.parametrize(("budget", "expected"), [(5, "ok"), (4, "budget_exhausted")])
def test_declared_bounded_repair_is_charged_across_stages(metre_scene, budget, expected):
    class Repair:
        def plan(self, request):
            return PlanResult(
                metre_scene.oracle_path,
                stage="repair",
                stage_work={"primary": {"expansions": 2}, "repair": {"expansions": 3}},
            )

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(max_expansions=budget))
    manifest = {
        "component_control": {
            "variant": "bounded_repair",
            "guide": "learned",
            "checkpoint_id": "weights",
            "machinery_id": "two-pass-logit",
        }
    }
    artifact = run_requests(Repair(), [request], manifest=manifest)
    assert artifact.records[0]["status"] == expected
    assert artifact.records[0]["work"]["expansions"] == 5
    assert artifact.records[0]["fallback"] is False
    metrics, arrays = score_run_artifact(artifact, [metre_scene], require_qualified_reference=False)
    assert arrays["repair_used"].tolist() == [True]
    assert metrics["repair_rate"] == 1.0
    assert metrics["fallback_rate"] == 0.0
    assert metrics["direct_success_rate"] == 0.0


def test_comparison_derives_repair_for_legacy_and_rejects_forged_flags(metre_scene, tmp_path):
    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import score_run_artifact, write_scored_artifact
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.metrics.comparison import compare_artifacts

    reference = {metre_scene.sample_id: qualified_reference_for_scene(metre_scene)}
    identity = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    for name, stage in [("direct", "primary"), ("repaired", "repair")]:

        class Candidate:
            def plan(self, request):
                return PlanResult(metre_scene.oracle_path, stage=stage, stage_work={stage: {"expansions": 2}})

        folder = tmp_path / name
        run = run_requests(
            Candidate(),
            [PlanRequest.from_scene(metre_scene)],
            output_dir=folder,
            manifest={"data_identity": {"fixture": "v1"}, "reference_identity": identity},
        )
        metrics, arrays = score_run_artifact(run, [metre_scene], reference_store=reference)
        write_scored_artifact(run, metrics, arrays)

    # Simulate an older v1 score whose arrays did not include the new field.
    folder = tmp_path / "repaired"
    with np.load(folder / "per_sample.npz", allow_pickle=False) as store:
        arrays = {key: store[key] for key in store.files if key != "repair_used"}
    np.savez(folder / "per_sample.npz", **arrays)
    receipt_path = folder / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["per_sample_sha256"] = sha256_file(folder / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    result = compare_artifacts([tmp_path / "direct", folder], bootstrap_resamples=10)
    assert result["methods"][0]["rates"]["repair_rate"] == 0
    assert result["methods"][1]["rates"]["repair_rate"] == 1
    assert result["methods"][1]["rates"]["direct_success_rate"] == 0
    assert result["component_attribution"]["rows"][1]["repair_rate"] == 1

    # A newly scored artifact cannot hide a repair by forging its own flag.
    arrays["repair_used"] = np.array([False], dtype=bool)
    np.savez(folder / "per_sample.npz", **arrays)
    receipt["per_sample_sha256"] = sha256_file(folder / "per_sample.npz")
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="repair attribution"):
        compare_artifacts([tmp_path / "direct", folder], bootstrap_resamples=10)
