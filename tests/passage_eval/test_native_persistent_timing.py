"""Native persistent timing on synthetic requests; no released data or device is accessed."""

import pytest
from passage.audit import qualified_reference_for_scene
from passage_eval.execution import run_requests, score_run_artifact
from passage_eval.execution import timing
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult
from passage_eval.supervision import sandbox
from passage_eval.supervision.supervisor import SupervisorConfig

PROFILE = "native_raw_path_persistent_v1"
BOUNDARY = "passage-timing-boundary/v6-native-raw-arrays-to-path"
SCHEMA = "passage-complete-request-native-v1"


def native_artifact(scene):
    request = PlanRequest.from_scene(scene, budget=BudgetSpec(wall_time_ms=10))
    artifact = run_requests(lambda request: PlanResult(scene.oracle_path), [request])
    artifact.manifest.update(
        schema=SCHEMA,
        timing_profile=PROFILE,
        timing_boundary=BOUNDARY,
        procedure={"execution": {"timing_profile": PROFILE}},
        native_execution={
            "session_id": "process-1",
            "n_processes": 1,
            "n_restarts": 0,
            "warmup": {
                "k": 1,
                "repeats": 16,
                "performed": True,
                "input_roster_sha256": "a" * 64,
                "disjoint_by_content": True,
            },
        },
    )
    for row in artifact.records:
        row.update(
            native_wall_ms=3.0,
            validation_recost_ms=20.0,
            total_wall_ms=23.0,
            timing_boundary=BOUNDARY,
            candidate_wall_ms=3.0,
            status="ok",
            late_completion=False,
            failed_attempt_ms=0.0,
            drain_ms=0.0,
        )
    return artifact


def test_native_profile_scores_raw_path_time_and_excludes_host_validation(metre_scene):
    artifact = native_artifact(metre_scene)
    artifact.validate()
    metrics, arrays = score_run_artifact(
        artifact, [metre_scene], reference_store={metre_scene.sample_id: qualified_reference_for_scene(metre_scene)}
    )
    assert arrays["latency_ms"].tolist() == [3.0]
    assert arrays["validation_recost_ms"].tolist() == [20.0]
    assert arrays["success"].tolist() == [True]
    assert timing.latency_field(artifact.manifest) == "native_wall_ms"
    assert metrics["latency_all_repeats_censored"]["latency_field"] == "native_wall_ms"


def test_native_warmup_is_one_disjoint_scene_repeated_sixteen_times_once_per_process(metre_scene):
    manifest = native_artifact(metre_scene).manifest
    warmup = timing.manifest_warmup(manifest)
    assert warmup["declared"] is True and warmup["performed"] is True
    assert warmup["per_pass"] == 16
    manifest["native_execution"]["warmup"]["repeats"] = 15
    assert timing.manifest_warmup(manifest)["performed"] is False


def test_native_profile_does_not_allow_a_supervised_runner_to_mislabel_its_boundary():
    with pytest.raises(ValueError, match="native|supervised"):
        SupervisorConfig(worker_python="python", isolation=sandbox.IsolationConfig(), timing_profile=PROFILE)


def test_native_deadline_uses_native_time_even_when_host_validation_is_slower(metre_scene):
    artifact = native_artifact(metre_scene)
    artifact.records[0].update(
        native_wall_ms=11.0,
        total_wall_ms=31.0,
        late_completion=True,
        status="timeout",
        failed_attempt_ms=10.0,
        drain_ms=1.0,
    )
    artifact.validate()
    _, arrays = score_run_artifact(
        artifact, [metre_scene], reference_store={metre_scene.sample_id: qualified_reference_for_scene(metre_scene)}
    )
    assert arrays["success"].tolist() == [False]
    assert arrays["latency_ms"].tolist() == [11.0]
