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

"""CPU fixture verification of complete-request artifacts and terminal outcomes."""

import json
from dataclasses import replace

import numpy as np
import pytest

from passage_eval.execution import run_requests, read_run_artifact, score_run_artifact
from passage_eval.protocol.contracts import PlanRequest, PlanResult, BudgetSpec
from passage_eval.protocol.planner import PlannerOutput


class Clock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def test_timed_route_is_persisted_and_every_repeat_is_scored(metre_scene, tmp_path):
    class Alternating:
        def __init__(self):
            self.calls = 0

        def plan(self, request):
            self.calls += 1
            path = [(0, 0), (1, 1), (2, 2)] if self.calls == 1 else [(0, 0), (1, 0), (2, 1), (2, 2)]
            return PlanResult(path, precision="float32", stage_work={"primary": {"expansions": 2}})

    artifact = run_requests(Alternating(), [PlanRequest.from_scene(metre_scene)], repeats=2, output_dir=tmp_path)
    loaded = read_run_artifact(tmp_path)
    assert len(loaded.records) == 2
    assert loaded.records[0]["route_hash"] != loaded.records[1]["route_hash"]
    np.testing.assert_array_equal(artifact.paths[1], loaded.paths[1])
    metrics, arrays = score_run_artifact(loaded, [metre_scene], require_qualified_reference=False)
    assert metrics["n_samples"] == 1
    assert metrics["all_repeats"]["n_samples"] == 2
    assert metrics["cost_regret_median"] == pytest.approx(0)
    assert metrics["all_repeats"]["cost_regret_median"] > 0
    assert arrays["repeat_index"].tolist() == [0, 1]
    assert arrays["status"].tolist() == ["ok", "ok"]
    assert arrays["regret"][1] > arrays["regret"][0]
    assert loaded.records[0]["work"]["expansions"] == 2
    assert loaded.records[0]["precision"] == "float32"


def test_deadline_includes_completion_sync_and_never_rescues_late_path(metre_scene):
    clock = Clock()

    class Late:
        def plan(self, request):
            clock.advance(0.003)
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    def sync():
        clock.advance(0.002)

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=4))
    run = run_requests(Late(), [request], clock=clock, synchronize=sync)
    record = run.records[0]
    assert record["status"] == "timeout"
    assert record["total_wall_ms"] == pytest.approx(5)
    assert record["failed_attempt_ms"] == 4
    assert record["drain_ms"] == pytest.approx(1)
    assert record["late_completion"] is True
    assert record["safe_completion"] is True
    metrics, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert metrics["success_rate"] == 0
    assert arrays["regret"][0] == 999999
    assert arrays["connected_path"][0]


@pytest.mark.parametrize(
    "path", [[[0, 0], [1.1, 1], [2, 2]], [[0, 0], [3, 3], [2, 2]], [[0, 0], [np.nan, 1], [2, 2]], [0, 0, 1, 1, 2, 2]]
)
def test_invalid_raw_coordinates_are_not_repaired(metre_scene, path, tmp_path):
    class Raw:
        def plan(self, request):
            return PlanResult(path)

    artifact = run_requests(Raw(), [PlanRequest.from_scene(metre_scene)], output_dir=tmp_path)
    assert artifact.records[0]["status"] == "invalid_path"
    np.testing.assert_equal(read_run_artifact(tmp_path).paths[0], np.asarray(path))


@pytest.mark.parametrize(
    "status", ["no_path", "invalid_path", "budget_exhausted", "timeout", "oom", "runtime_error", "cancelled"]
)
def test_candidate_failure_cannot_be_converted_to_success(metre_scene, status):
    class Failed:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)], status=status)

    run = run_requests(Failed(), [PlanRequest.from_scene(metre_scene)])
    assert run.records[0]["status"] == status
    _, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert not arrays["valid_path"][0]
    assert arrays["regret"][0] == 999999


def test_primary_repair_and_fallback_work_is_charged(metre_scene):
    class Repaired:
        def plan(self, request):
            return PlanResult(
                [(0, 0), (1, 1), (2, 2)],
                stage="fallback",
                stage_work={
                    "primary": {"expansions": 2},
                    "repair": {"expansions": 3},
                    "fallback": {"expansions": 4},
                },
            )

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(max_expansions=8))
    record = run_requests(Repaired(), [request]).records[0]
    assert record["work"]["expansions"] == 9
    assert record["fallback"] is True
    assert record["status"] == "budget_exhausted"


def test_legacy_predict_path_is_an_ordinary_input_adapter(metre_scene):
    class Legacy:
        def predict_path(self, scene):
            assert scene.oracle_path == []
            assert np.isnan(scene.oracle_cost)
            return PlannerOutput([(0, 0), (1, 1), (2, 2)], 123456)

    run = run_requests(Legacy(), [PlanRequest.from_scene(metre_scene)])
    assert run.records[0]["status"] == "ok"
    assert run.records[0]["recost"] == pytest.approx(metre_scene.oracle_cost)


def test_duplicate_roster_and_corrupt_offsets_are_rejected(metre_scene, tmp_path):
    request = PlanRequest.from_scene(metre_scene)

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    with pytest.raises(ValueError, match="unique"):
        run_requests(Direct(), [request, request])
    run_requests(Direct(), [request], output_dir=tmp_path)
    with np.load(tmp_path / "paths.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["offsets"][-1] += 100
    np.savez(tmp_path / "paths.npz", **arrays)
    with pytest.raises(ValueError):
        read_run_artifact(tmp_path)


def test_stored_reference_is_not_implicitly_qualified(metre_scene):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    with pytest.raises(ValueError, match="qualified"):
        score_run_artifact(run, [metre_scene])


def test_work_total_does_not_double_charge_reported_stage_subtotals(metre_scene):
    class ExplicitTotals:
        def plan(self, request):
            return PlanResult(
                [(0, 0), (1, 1), (2, 2)],
                work={"expansions": 5},
                stage_work={"primary": {"expansions": 2}, "repair": {"expansions": 3}},
            )

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(max_expansions=5))
    record = run_requests(ExplicitTotals(), [request]).records[0]
    assert record["work"]["expansions"] == 5
    assert record["status"] == "ok"


def test_legality_recost_is_inside_timer(metre_scene, monkeypatch):
    from passage_eval.execution import runner

    clock = Clock()
    actual_cost = runner.path_cost

    def timed_cost(*args, **kwargs):
        clock.advance(0.006)
        return actual_cost(*args, **kwargs)

    monkeypatch.setattr(runner, "path_cost", timed_cost)

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=5))
    record = run_requests(Direct(), [request], clock=clock).records[0]
    assert record["total_wall_ms"] == pytest.approx(6)
    assert record["status"] == "timeout"


@pytest.mark.parametrize("error,status", [(MemoryError("memory"), "oom"), (RuntimeError("failure"), "runtime_error")])
def test_failed_candidate_attempt_is_timed(metre_scene, error, status):
    clock = Clock()

    class Failed:
        def plan(self, request):
            clock.advance(0.007)
            raise error

    record = run_requests(Failed(), [PlanRequest.from_scene(metre_scene)], clock=clock).records[0]
    assert record["status"] == status
    assert record["total_wall_ms"] == pytest.approx(7)
    assert record["candidate_wall_ms"] == pytest.approx(7)
    assert record["failed_attempt_ms"] == pytest.approx(7)


def test_completion_failure_retains_lease_and_requires_explicit_safe_drain(metre_scene):
    from passage_eval.execution import PendingCompletionError, drain_pending_requests

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("work is still pending")

    with pytest.raises(PendingCompletionError, match="retained"):
        run_requests(Direct(), [PlanRequest.from_scene(metre_scene)], synchronize=sync)
    with pytest.raises(PendingCompletionError):
        run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    assert drain_pending_requests() == 1
    assert run_requests(Direct(), [PlanRequest.from_scene(metre_scene)]).records[0]["status"] == "ok"


def test_completion_precedes_route_snapshot_and_physical_recost(metre_scene):
    pending = np.array([[0, 0], [0, 0], [2, 2]])

    class AsyncRoute:
        def plan(self, request):
            return PlanResult(pending)

    calls = 0

    def sync():
        nonlocal calls
        calls += 1
        if calls == 2:
            pending[1] = [1, 1]

    run = run_requests(AsyncRoute(), [PlanRequest.from_scene(metre_scene)], synchronize=sync)
    assert run.records[0]["status"] == "ok"
    np.testing.assert_array_equal(run.paths[0], [[0, 0], [1, 1], [2, 2]])
    assert run.records[0]["recost"] == pytest.approx(metre_scene.oracle_cost)


@pytest.mark.parametrize("failed_sync", [False, True])
def test_decoder_field_owner_survives_until_safe_completion_or_drain(metre_scene, failed_sync):
    import gc
    import weakref
    from passage_eval.execution import DecoderPlannerAdapter, PendingCompletionError, drain_pending_requests

    owners = []

    def predictor(request):
        field = np.zeros(request.elevation.shape)
        owners.append(weakref.ref(field))
        return field

    def decoder(field, request):
        return PlanResult([(0, 0), (1, 1), (2, 2)])

    calls = [0]

    def synchronize():
        calls[0] += 1
        if calls[0] >= 2:
            gc.collect()
            assert owners[0]() is not None
        if failed_sync and calls[0] == 2:
            raise RuntimeError("pending field work")

    adapter = DecoderPlannerAdapter(predictor, decoder)
    if failed_sync:
        with pytest.raises(PendingCompletionError):
            run_requests(adapter, [PlanRequest.from_scene(metre_scene)], synchronize=synchronize)
        gc.collect()
        assert owners[0]() is not None
        assert drain_pending_requests() == 1
    else:
        run_requests(adapter, [PlanRequest.from_scene(metre_scene)], synchronize=synchronize)
    gc.collect()
    assert owners[0]() is None


def test_candidate_receives_one_absolute_deadline_including_copy_time(metre_scene, monkeypatch):
    clock = Clock()
    clock.advance(10)
    original = PlanRequest.candidate_copy

    def copying(request, **kwargs):
        clock.advance(0.002)
        return original(request, **kwargs)

    monkeypatch.setattr(PlanRequest, "candidate_copy", copying)

    class Staged:
        def plan(self, request):
            assert request.deadline_monotonic == pytest.approx(10.005)
            assert request.remaining_wall_ms(clock=clock) == pytest.approx(3)
            clock.advance(0.002)
            assert request.remaining_wall_ms(clock=clock) == pytest.approx(1)
            return PlanResult([(0, 0), (1, 1), (2, 2)], stage="repair")

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=5))
    assert run_requests(Staged(), [request], clock=clock).records[0]["status"] == "ok"


def test_offline_scoring_rejects_timed_recost_disagreement(metre_scene):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    run.records[0]["recost"] += 1
    with pytest.raises(ValueError, match="timed.*recost"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


@pytest.mark.parametrize(
    "mutation",
    [
        {"safe_completion": 1},
        {"legal_path": False},
        {"failed_attempt_ms": 3},
        {"late_completion": True},
        {"work": {"expansions": -1}},
    ],
)
def test_terminal_artifact_cannot_forge_completion_or_accounting(metre_scene, mutation):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    artifact = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    artifact.records[0].update(mutation)
    with pytest.raises(ValueError):
        artifact.validate()


def test_qualified_reference_roundtrip_and_input_content_binding(tensor_scene, tmp_path):
    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import write_scored_artifact

    reference = qualified_reference_for_scene(tensor_scene)
    assert reference["qualified"]

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    scene = tensor_scene
    run = run_requests(
        Direct(),
        [PlanRequest.from_scene(scene)],
        output_dir=tmp_path,
        manifest={"reference_qualification": {scene.sample_id: reference}},
    )
    metrics, arrays = score_run_artifact(run, [scene])
    assert metrics["reference_qualified"]
    assert arrays["reference_policy"].tolist() == ["qualified_exact"]
    write_scored_artifact(run, metrics, arrays)
    receipt = json.loads((tmp_path / "score_receipt.json").read_text())
    assert receipt["qualified_reference"]
    assert receipt["scored_records"] == 1
    changed = scene.input_tensor.copy()
    changed[:, :, 0] = 0.5
    mutated = replace(scene, input_tensor=changed, metadata={"calibration": {"min": -1.0, "max": 1.0}})
    with pytest.raises(ValueError, match="identity mismatch"):
        score_run_artifact(run, [mutated])


def test_unknown_geographic_block_is_explicit_and_contract_is_distinct(metre_scene):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    _, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert arrays["cluster_id"].tolist() == [""]
    assert arrays["cluster_provenance"].tolist() == ["unknown"]
    assert arrays["metric_contract"].item() == "passage-stored-diagnostic-v1"
    assert run.manifest["request_metadata"] == {metre_scene.sample_id: {}}


def test_mutated_reference_optimum_cannot_reuse_qualified_receipt(metre_scene):
    from passage.audit import qualified_reference_for_scene
    from passage_eval.metrics.cost import path_cost

    reference = qualified_reference_for_scene(metre_scene)
    reference["optimal_path"] = [(0, 0), (1, 0), (2, 1), (2, 2)]
    reference["optimal_cost"] = path_cost(
        metre_scene.elevation,
        reference["optimal_path"],
        weight=metre_scene.cost_weight,
        cost_model=metre_scene.cost_model,
    )

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    with pytest.raises(ValueError, match="receipt"):
        score_run_artifact(run, [metre_scene], reference_store={metre_scene.sample_id: reference})


@pytest.mark.parametrize("field,value", [("request_specs", {}), ("quality_repeat_index", 99)])
def test_invalid_run_schema_is_rejected(metre_scene, field, value):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)])
    run.manifest[field] = value
    with pytest.raises(ValueError):
        run.validate()


@pytest.mark.parametrize(
    "metadata",
    [{"diagnostics": {"bad": object()}}, {"stage_spans": {"bad": object()}}, {"resource_peaks": {"bad": object()}}],
)
def test_malformed_candidate_metadata_is_a_terminal_failure_not_campaign_abort(metre_scene, metadata):
    class BadMetadata:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)], **metadata)

    run = run_requests(BadMetadata(), [PlanRequest.from_scene(metre_scene)], repeats=2)
    assert [record["status"] for record in run.records] == ["runtime_error", "runtime_error"]
    assert len(run.paths) == 2


@pytest.mark.parametrize(
    "work,stage_work",
    [
        ({}, {"primary": {"expansions": 100}, "repair": {"expansions": -100}}),
        ({"expansions": -100}, {"primary": {"expansions": 100}}),
    ],
)
def test_negative_work_cannot_be_hidden_by_aggregation(metre_scene, work, stage_work):
    class BadWork:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)], work=work, stage_work=stage_work)

    record = run_requests(BadWork(), [PlanRequest.from_scene(metre_scene)]).records[0]
    assert record["status"] == "runtime_error"


def test_failed_completion_retains_planner_owner_until_safe_drain(metre_scene):
    import weakref
    import gc
    from passage_eval.execution import PendingCompletionError, drain_pending_requests

    owners = []

    class DevicePlanner:
        def __init__(self):
            owners.append(weakref.ref(self))

        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("pending")

    with pytest.raises(PendingCompletionError):
        run_requests(DevicePlanner(), [PlanRequest.from_scene(metre_scene)], synchronize=sync)
    gc.collect()
    assert owners[0]() is not None
    assert drain_pending_requests() == 1
    gc.collect()
    assert owners[0]() is None


def test_changed_ordinary_calibration_invalidates_scoring_identity(metre_scene):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    scene = replace(metre_scene, metadata={"calibration": {"min": 0, "max": 10}})
    run = run_requests(Direct(), [PlanRequest.from_scene(scene)])
    changed = replace(scene, metadata={"calibration": {"min": -100, "max": 100}})
    with pytest.raises(ValueError, match="identity mismatch"):
        score_run_artifact(run, [changed], require_qualified_reference=False)


def test_original_bad_output_remains_leased_until_device_drain(metre_scene):
    import gc
    import weakref
    from passage_eval.execution import PendingCompletionError, drain_pending_requests

    outputs = []

    class DevicePlanner:
        def plan(self, request):
            output = PlanResult([(0, 0), (1, 1), (2, 2)], diagnostics={"bad": object()})
            outputs.append(weakref.ref(output))
            return output

    calls = []

    def sync():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("pending")

    with pytest.raises(PendingCompletionError):
        run_requests(DevicePlanner(), [PlanRequest.from_scene(metre_scene)], synchronize=sync)
    gc.collect()
    assert outputs[0]() is not None
    assert drain_pending_requests() == 1
    gc.collect()
    assert outputs[0]() is None


@pytest.mark.parametrize(
    "work,peaks", [({"live_states": 1, "peak_live_states": 100}, {}), ({"live_states": 1}, {"peak_live_states": 100})]
)
def test_live_state_budget_uses_maximum_observed_peak(metre_scene, work, peaks):
    class HiddenPeak:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)], work=work, resource_peaks=peaks)

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(max_live_states=2))
    assert run_requests(HiddenPeak(), [request]).records[0]["status"] == "budget_exhausted"


@pytest.mark.parametrize("formulation", ["segmentation", "regression"])
def test_ordinary_input_field_adapters_include_prediction_and_decode(metre_scene, formulation):
    from passage_eval.execution import DecoderPlannerAdapter

    clock = Clock()

    def predictor(request):
        assert not hasattr(request, "oracle_path")
        clock.advance(0.003)
        return (
            np.eye(3, dtype=np.float32)
            if formulation == "segmentation"
            else np.arange(9, dtype=np.float32).reshape(3, 3)
        )

    def decoder(field, request):
        assert field.shape == request.elevation.shape
        clock.advance(0.004)
        return PlanResult([(0, 0), (1, 1), (2, 2)], work={"expansions": 2})

    planner = DecoderPlannerAdapter(predictor, decoder, precision="float32")
    run = run_requests(planner, [PlanRequest.from_scene(metre_scene)], clock=clock)
    assert run.records[0]["total_wall_ms"] == pytest.approx(7)
    assert run.records[0]["precision"] == "float32"
    assert run.records[0]["work"]["expansions"] == 2


def test_extended_precision_fractional_coordinates_are_rejected_before_conversion(metre_scene):
    fraction = np.nextafter(np.longdouble(1), np.longdouble(2))
    if fraction == np.longdouble(float(fraction)):
        pytest.skip("host has no floating type wider than float64")

    class Raw:
        def plan(self, request):
            return PlanResult(np.array([[0, 0], [fraction, 1], [2, 2]], dtype=np.longdouble))

    assert run_requests(Raw(), [PlanRequest.from_scene(metre_scene)]).records[0]["status"] == "invalid_path"


@pytest.mark.parametrize("contract", ["passage-stored-diagnostic-v1", "unknown-contract"])
def test_qualified_score_receipt_cannot_stamp_diagnostic_or_unknown_contract(metre_scene, tmp_path, contract):
    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import write_scored_artifact

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    reference = qualified_reference_for_scene(metre_scene)
    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)], output_dir=tmp_path)
    metrics, arrays = score_run_artifact(run, [metre_scene], reference_store={metre_scene.sample_id: reference})
    arrays["metric_contract"] = np.array(contract)
    with pytest.raises(ValueError, match="metric contract"):
        write_scored_artifact(run, metrics, arrays)
    assert not (tmp_path / "score_receipt.json").exists()


def test_score_receipt_binds_exact_metric_contract(metre_scene, tmp_path):
    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import write_scored_artifact

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)], output_dir=tmp_path)
    metrics, arrays = score_run_artifact(
        run, [metre_scene], reference_store={metre_scene.sample_id: qualified_reference_for_scene(metre_scene)}
    )
    write_scored_artifact(run, metrics, arrays)
    receipt = json.loads((tmp_path / "score_receipt.json").read_text())
    assert receipt["metric_contract"] == arrays["metric_contract"].item() == "passage-qualified-complete-v1"


def test_unqualified_score_receipt_rejects_qualified_contract(metre_scene, tmp_path):
    from passage_eval.execution import write_scored_artifact

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [PlanRequest.from_scene(metre_scene)], output_dir=tmp_path)
    metrics, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    arrays["metric_contract"] = np.array("passage-qualified-complete-v1")
    with pytest.raises(ValueError, match="metric contract"):
        write_scored_artifact(run, metrics, arrays)


@pytest.mark.parametrize("independent", [False, True])
def test_reference_variant_cannot_transfer_between_identical_free_and_obstacle_grids(metre_scene, independent):
    from passage.audit import qualified_reference_for_scene, independently_qualified_reference_for_scene

    source = replace(metre_scene, metadata={"task_variant": "free"})
    target = replace(metre_scene, metadata={"task_variant": "obstacles"})
    if independent:
        reference = independently_qualified_reference_for_scene(source, cohort_id="named-diagnostic")
        manifest = {"reference_policy": "independent_analytic_reference", "reference_cohort_id": "named-diagnostic"}
    else:
        reference = qualified_reference_for_scene(source)
        manifest = {"reference_policy": "released_labels"}
    run = run_requests(lambda request: [(0, 0), (1, 1), (2, 2)], [PlanRequest.from_scene(target)], manifest=manifest)
    with pytest.raises(ValueError, match="variant"):
        score_run_artifact(run, [target], reference_store={target.sample_id: reference})


@pytest.mark.parametrize(
    "change",
    [
        {"oracle_path": []},
        {"oracle_cost": float("nan")},
        {"oracle_path": [(0, 0), (1, 0), (2, 1), (2, 2)]},
        {"oracle_cost": 85.0},
    ],
)
def test_released_label_admission_binds_current_stored_path_and_cost(metre_scene, change):
    from passage.audit import qualified_reference_for_scene

    reference = qualified_reference_for_scene(metre_scene)
    target = replace(metre_scene, **change)
    run = run_requests(
        lambda request: [(0, 0), (1, 1), (2, 2)],
        [PlanRequest.from_scene(target)],
        manifest={"reference_policy": "released_labels"},
    )
    with pytest.raises(ValueError, match="stored label"):
        score_run_artifact(run, [target], reference_store={target.sample_id: reference})


def test_named_independent_cohort_accepts_its_invalid_source_labels(metre_scene):
    from passage.audit import independently_qualified_reference_for_scene

    target = replace(metre_scene, oracle_path=[], oracle_cost=float("nan"), metadata={"task_variant": "obstacles"})
    reference = independently_qualified_reference_for_scene(target, cohort_id="invalid-source-labels")
    run = run_requests(
        lambda request: [(0, 0), (1, 1), (2, 2)],
        [PlanRequest.from_scene(target)],
        manifest={"reference_policy": "independent_analytic_reference", "reference_cohort_id": "invalid-source-labels"},
    )
    metrics, arrays = score_run_artifact(run, [target], reference_store={target.sample_id: reference})
    assert metrics["legal_route_rate"] == 1
    assert arrays["regret"].tolist() == [0.0]
    assert reference["source_label_qualified"] is False


@pytest.mark.parametrize(
    "metadata,expected,provenance",
    [
        ({"source_footprint_id": "joined", "source_tile_id": "tile"}, "joined", "declared_source_footprint_id"),
        ({"source_tile_id": "tile", "tile_id": "crop"}, "tile", "declared_source_tile_id"),
        (
            {"cluster_id": "wide", "geographic": {"cluster_id": "nested"}, "source_footprint_id": "joined"},
            "wide",
            "declared_cluster_id",
        ),
        (
            {"geographic": {"cluster_id": "nested"}, "source_footprint_id": "joined"},
            "nested",
            "declared_geographic_cluster_id",
        ),
    ],
)
def test_score_support_precedence_preserves_declared_broader_blocks(metre_scene, metadata, expected, provenance):
    scene = replace(metre_scene, metadata=metadata)
    run = run_requests(lambda request: [(0, 0), (1, 1), (2, 2)], [PlanRequest.from_scene(scene)])
    _, arrays = score_run_artifact(run, [scene], require_qualified_reference=False)
    assert arrays["cluster_id"].tolist() == [expected]
    assert arrays["cluster_provenance"].tolist() == [provenance]


def test_partially_declared_broader_support_stays_unknown_across_roster(metre_scene):
    scenes = [
        replace(metre_scene, sample_id="a", metadata={"cluster_id": "wide", "tile_id": "tile-a"}),
        replace(metre_scene, sample_id="b", metadata={"tile_id": "tile-b"}),
    ]
    run = run_requests(lambda request: [(0, 0), (1, 1), (2, 2)], [PlanRequest.from_scene(scene) for scene in scenes])
    _, arrays = score_run_artifact(run, scenes, require_qualified_reference=False)
    assert arrays["cluster_id"].tolist() == ["", ""]
    assert arrays["cluster_provenance"].tolist() == ["unknown", "unknown"]


def test_released_receipt_cannot_claim_independent_named_admission(metre_scene):
    from passage.audit import qualified_reference_for_scene

    reference = qualified_reference_for_scene(metre_scene)
    run = run_requests(
        lambda request: [(0, 0), (1, 1), (2, 2)],
        [PlanRequest.from_scene(metre_scene)],
        manifest={"reference_policy": "independent_analytic_reference", "reference_cohort_id": "named"},
    )
    with pytest.raises(ValueError, match="policy"):
        score_run_artifact(run, [metre_scene], reference_store={metre_scene.sample_id: reference})


def test_updated_label_identity_cannot_hide_stale_released_optimality_flags(metre_scene):
    from passage.audit import qualified_reference_for_scene, stored_label_identity
    from passage_eval.metrics.cost import path_cost

    reference = qualified_reference_for_scene(metre_scene)
    route = [(0, 0), (1, 0), (2, 1), (2, 2)]
    cost = path_cost(metre_scene.elevation, route, weight=metre_scene.cost_weight, cost_model=metre_scene.cost_model)
    target = replace(metre_scene, oracle_path=route, oracle_cost=cost)
    reference.update(
        stored_path=route,
        stored_cost=cost,
        stored_label_id=stored_label_identity(route, cost, task_variant="obstacles"),
    )
    run = run_requests(lambda request: [(0, 0), (1, 1), (2, 2)], [PlanRequest.from_scene(target)])
    with pytest.raises(ValueError, match="stored label"):
        score_run_artifact(run, [target], reference_store={target.sample_id: reference})


# ---------------------------------------------------------------------------
# timing profiles (passage_eval.execution.timing)
# ---------------------------------------------------------------------------

SERVICE_MANIFEST = dict(
    schema="passage-complete-request-supervised-service-v2",
    timing_boundary="passage-timing-boundary/v4-supervised-ipc-service",
    timing_profile="supervised_service_v1",
    procedure={"supervision": {"timing_profile": "supervised_service_v1"}},
)
REWARM_MANIFEST = dict(
    schema="passage-complete-request-supervised-service-v3",
    timing_boundary="passage-timing-boundary/v5-supervised-ipc-service-rewarm",
    timing_profile="supervised_service_rewarm_v1",
    procedure={"supervision": {"timing_profile": "supervised_service_rewarm_v1"}},
)


def _direct_run(scene):
    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    return run_requests(Direct(), [PlanRequest.from_scene(scene)])


def _as_profiled(run, manifest, service_ms=None):
    run.manifest.update(manifest)
    for row in run.records:
        service = row["total_wall_ms"] / 2 if service_ms is None else service_ms
        row.update(
            service_wall_ms=service,
            validation_recost_ms=row["total_wall_ms"] - service,
            timing_boundary=manifest["timing_boundary"],
        )
    return run


def test_scoring_rejects_service_procedure_with_legacy_manifest(metre_scene):
    run = _direct_run(metre_scene)
    run.manifest["procedure"] = {"supervision": {"timing_profile": "supervised_service_v1"}}
    with pytest.raises(ValueError, match="timing profile"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


def test_service_profile_rejects_corrupt_timing_and_legacy_schema(metre_scene):
    run = _direct_run(metre_scene)
    run.manifest["timing_profile"] = "supervised_service_v1"
    run.manifest["procedure"] = {"supervision": {"timing_profile": "supervised_service_v1"}}
    run.records[0].update(service_wall_ms=100.0, validation_recost_ms=1.0, total_wall_ms=10.0)
    with pytest.raises(ValueError, match="schema|timing"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


def test_service_profile_rejects_corrupted_service_and_total_with_valid_schema(metre_scene):
    run = _direct_run(metre_scene)
    run.manifest.update(SERVICE_MANIFEST)
    run.records[0].update(
        service_wall_ms=100.0,
        validation_recost_ms=1.0,
        total_wall_ms=10.0,
        timing_boundary="passage-timing-boundary/v4-supervised-ipc-service",
    )
    with pytest.raises(ValueError, match="timing"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


@pytest.mark.parametrize(
    "boundary",
    ["passage-timing-boundary/v4-supervised-ipc-service", "passage-timing-boundary/v5-supervised-ipc-service-rewarm"],
)
def test_legacy_schema_cannot_claim_a_profiled_boundary(metre_scene, boundary):
    run = _direct_run(metre_scene)
    run.manifest["timing_boundary"] = boundary
    with pytest.raises(ValueError, match="timing|boundary"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


@pytest.mark.parametrize("procedure", [None, "bad", {"supervision": None}, {"supervision": "bad"}])
def test_malformed_procedure_is_rejected_cleanly(procedure):
    from passage_eval.execution.timing import latency_field

    with pytest.raises(ValueError, match="procedure|supervision"):
        latency_field({"procedure": procedure})


def test_service_artifact_rejects_record_with_legacy_boundary(metre_scene):
    run = _direct_run(metre_scene)
    run.manifest.update(SERVICE_MANIFEST)
    row = run.records[0]
    row["service_wall_ms"] = row["total_wall_ms"]
    row["validation_recost_ms"] = 0.0
    row["timing_boundary"] = "passage-timing-boundary/v3-supervised-ipc-fresh-pass"
    with pytest.raises(ValueError, match="boundary"):
        score_run_artifact(run, [metre_scene], require_qualified_reference=False)


def test_unknown_timing_profile_is_refused_not_mapped_to_a_default(metre_scene):
    from passage_eval.execution.timing import get_profile

    with pytest.raises(ValueError, match="unknown timing profile"):
        get_profile("supervised_service_v9")
    run = _direct_run(metre_scene)
    run.manifest.update(SERVICE_MANIFEST, timing_profile="supervised_service_v9")
    with pytest.raises(ValueError, match="unknown timing profile"):
        run.validate()


def test_profiles_bind_schema_boundary_latency_field_and_rewarm_together():
    from passage_eval.execution import timing
    from passage_eval.execution.artifacts import SCHEMA
    from passage_eval.supervision import READABLE_TIMING_BOUNDARY_IDS, TIMING_BOUNDARY_ID

    assert timing.PROFILES == (
        "supervised_total_v3",
        "supervised_service_v1",
        "supervised_service_rewarm_v1",
        "native_raw_path_persistent_v1",
    )
    legacy = timing.get_profile("supervised_total_v3")
    service = timing.get_profile("supervised_service_v1")
    rewarm = timing.get_profile("supervised_service_rewarm_v1")
    native = timing.get_profile("native_raw_path_persistent_v1")
    assert (legacy.schema, legacy.boundary, legacy.latency_field) == (SCHEMA, None, "total_wall_ms")
    assert legacy.validation_in_boundary and not service.validation_in_boundary
    assert (service.schema, service.latency_field, service.rewarm_after_restart) == (
        "passage-complete-request-supervised-service-v2",
        "service_wall_ms",
        False,
    )
    assert (rewarm.schema, rewarm.latency_field, rewarm.rewarm_after_restart) == (
        "passage-complete-request-supervised-service-v3",
        "service_wall_ms",
        True,
    )
    assert (native.schema, native.boundary, native.latency_field, native.rewarm_after_restart) == (
        "passage-complete-request-native-v1",
        "passage-timing-boundary/v6-native-raw-arrays-to-path",
        "native_wall_ms",
        False,
    )
    assert not native.validation_in_boundary
    assert len({legacy.schema, service.schema, rewarm.schema, native.schema}) == 4
    assert {service.boundary, rewarm.boundary, TIMING_BOUNDARY_ID} <= set(READABLE_TIMING_BOUNDARY_IDS)


@pytest.mark.parametrize("manifest", [SERVICE_MANIFEST, REWARM_MANIFEST])
def test_profiled_scores_use_the_service_time_and_keep_the_recost_out_of_the_deadline(metre_scene, manifest):
    run = _as_profiled(_direct_run(metre_scene), manifest)
    if manifest is REWARM_MANIFEST:
        run.records[0].update(rewarmed_before_request=False, warmup_complete_before_request=None)
    metrics, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert arrays["latency_ms"].tolist() == [run.records[0]["service_wall_ms"]]
    assert arrays["latency_ms"][0] < run.records[0]["total_wall_ms"]
    block = metrics["latency_all_repeats_censored"]
    assert block["timing_profile"] == manifest["timing_profile"] and block["latency_field"] == "service_wall_ms"
    # protocol: validation and physical recost stay outside the boundary and are reported separately
    recorded = run.records[0]["validation_recost_ms"]
    assert recorded > 0 and arrays["validation_recost_ms"].tolist() == [recorded]
    recost = metrics["validation_recost_all_repeats_ms"]
    assert recost["reported_separately"] is True and recost["inside_scored_latency"] is False
    assert recost["status"] == "recorded" and recost["n_recorded"] == 1
    assert recost["quantiles_ms"]["p50"] == recost["max_ms"] == recorded


def test_in_process_scores_state_that_validation_and_recost_are_inside_the_latency(metre_scene):
    run = _direct_run(metre_scene)
    metrics, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert np.isnan(arrays["validation_recost_ms"]).all()
    recost = metrics["validation_recost_all_repeats_ms"]
    assert recost["inside_scored_latency"] is True and recost["status"] == "not_recorded"
    assert recost["mean_ms"] is None and recost["quantiles_ms"] == {"p50": None, "p95": None, "p99": None}


def test_rewarm_records_must_state_their_warmup_state(metre_scene):
    run = _as_profiled(_direct_run(metre_scene), REWARM_MANIFEST)
    with pytest.raises(ValueError, match="warm-up state"):
        run.validate()
    for bad in (
        {"rewarmed_before_request": 1, "warmup_complete_before_request": None},
        {"rewarmed_before_request": False, "warmup_complete_before_request": "yes"},
    ):
        run.records[0].update(bad)
        with pytest.raises(ValueError, match="warm-up state"):
            run.validate()
    run.records[0].update(rewarmed_before_request=True, warmup_complete_before_request=False)
    run.validate()


def test_service_profile_deadline_uses_service_time_not_total(metre_scene):
    """Under v4 a request whose total exceeds the budget only because of the recost is on time."""
    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=50.0))

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    run = run_requests(Direct(), [request])
    run.manifest.update(SERVICE_MANIFEST)
    row = run.records[0]
    row.update(
        status="ok",
        total_wall_ms=80.0,
        service_wall_ms=40.0,
        validation_recost_ms=40.0,
        late_completion=False,
        failed_attempt_ms=0.0,
        drain_ms=0.0,
        deadline_ms=50.0,
        timing_boundary=SERVICE_MANIFEST["timing_boundary"],
    )
    run.validate()
    row["late_completion"] = True  # the legacy accounting (total > budget) contradicts the service profile
    with pytest.raises(ValueError, match="deadline"):
        run.validate()
