import json
import numpy as np
import pytest
from passage_eval.protocol.contracts import PlanRequest, PlanResult, BudgetSpec
from .test_package import package_fixture


@pytest.mark.parametrize("descriptor_model", ["slope", "slope_uphill"])
@pytest.mark.parametrize("request_model", ["slope", "slope_uphill"])
def test_equivalent_objective_aliases_preserve_manifest_and_input_identity(tmp_path, descriptor_model, request_model):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, document = package_fixture(tmp_path)
    document["dataset"]["cost_model"] = descriptor_model
    path.write_text(json.dumps(document))
    sealed_bytes = path.read_bytes()
    package = load_candidate_package(path)
    sealed_digest = package.digest

    class Policy:
        def plan(self, request):
            return PlanResult([[0, 0], [0, 1]])

    request = PlanRequest("a", np.zeros((1, 2)), np.zeros((1, 2), bool), (0, 0), (0, 1), request_model, 62.4)
    journal = tmp_path / "journal.jsonl"
    report = run_candidate_session(Policy(), [request], manifest=package, journal_path=journal)
    start, result, complete = [json.loads(line) for line in journal.read_text().splitlines()]

    assert report["n_completed"] == complete["n_completed"] == 1
    assert result["status"] == "ok" and result["path"] == [[0, 0], [0, 1]]
    # This digest is fixed from the hand-encoded raw arrays, endpoints and sample ID.
    assert result["raw_input_sha256"] == "1708e0212ce28c50aefcb32c1033e6f1a89e292f4be42a45e40124c776814142"
    assert start["manifest_sha256"] == sealed_digest
    assert path.read_bytes() == sealed_bytes
    assert package.document == document and package.digest == sealed_digest
    package.verify()


@pytest.mark.parametrize(
    ("descriptor_model", "request_model", "request_weight"),
    [
        ("slope", "elevation", 62.4),
        ("slope_uphill", "elevation", 62.4),
        ("elevation", "slope", 62.4),
        ("elevation", "slope_uphill", 62.4),
        ("slope", "slope_uphill", 1.0),
        ("slope_uphill", "slope", 1.0),
    ],
)
def test_different_objective_model_or_weight_is_rejected(tmp_path, descriptor_model, request_model, request_weight):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, document = package_fixture(tmp_path)
    document["dataset"]["cost_model"] = descriptor_model
    path.write_text(json.dumps(document))

    class Policy:
        def plan(self, request):
            return PlanResult([[0, 0], [0, 1]])

    request = PlanRequest("a", np.zeros((1, 2)), np.zeros((1, 2), bool), (0, 0), (0, 1), request_model, request_weight)
    journal = tmp_path / "journal.jsonl"
    with pytest.raises(ValueError, match="request objective differs from candidate identity"):
        run_candidate_session(Policy(), [request], manifest=load_candidate_package(path), journal_path=journal)
    assert [json.loads(line)["kind"] for line in journal.read_text().splitlines()] == ["session_start"]


def test_complete_procedure_is_timed_through_synchronization(tmp_path):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)

    class Policy:
        ready = False

        def plan(self, request):
            assert not hasattr(request, "oracle_path")
            self.route = np.array([[0, 0], [0, 1]])
            return PlanResult(self.route, work={"policy_steps": 1})

        def synchronize(self):
            self.ready = True

    policy = Policy()
    ticks = iter([0, 10_000_000])
    req = PlanRequest("a", np.zeros((1, 2)), np.zeros((1, 2), bool), (0, 0), (0, 1), "elevation", 62.4)
    report = run_candidate_session(
        policy,
        [req],
        manifest=load_candidate_package(path),
        journal_path=tmp_path / "journal.jsonl",
        clock_ns=lambda: next(ticks),
    )
    row = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert policy.ready and report["n_completed"] == 1
    assert row["path"] == [[0, 0], [0, 1]] and row["work"] == {"policy_steps": 1}
    assert row["elapsed_ns"] == 10_000_000


def test_late_route_cannot_be_success(tmp_path):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)

    class Slow:
        def plan(self, r):
            return PlanResult([[0, 0], [0, 1]])

    req = PlanRequest(
        "a",
        np.zeros((1, 2)),
        np.zeros((1, 2), bool),
        (0, 0),
        (0, 1),
        "elevation",
        62.4,
        budget=BudgetSpec(wall_time_ms=1),
    )
    ticks = iter([0, 2_000_000])
    run_candidate_session(
        Slow(),
        [req],
        manifest=load_candidate_package(path),
        journal_path=tmp_path / "journal.jsonl",
        clock_ns=lambda: next(ticks),
    )
    row = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert row["status"] == "timeout" and row["path"] == [[0, 0], [0, 1]]


def test_reference_keys_are_rejected_before_candidate_load(tmp_path):
    from passage_eval.campaign.candidate.transport import load_input_requests
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)
    np.savez(
        tmp_path / "bad.npz",
        sample_ids=["a"],
        terrain=np.zeros((1, 256, 256)),
        obstacles=np.zeros((1, 256, 256), bool),
        starts=[[0, 0]],
        goals=[[0, 1]],
        oracle_cost=[1],
    )
    with pytest.raises(ValueError, match="input"):
        list(load_input_requests(tmp_path / "bad.npz", load_candidate_package(path)))


def test_failed_device_drain_invalidates_session_instead_of_reusing_buffers(tmp_path):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)

    class Broken:
        calls = 0

        def plan(self, r):
            return PlanResult([[0, 0], [0, 1]])

        def synchronize(self):
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("device drain failed")

    req = PlanRequest("a", np.zeros((1, 2)), np.zeros((1, 2), bool), (0, 0), (0, 1), "elevation", 62.4)
    with pytest.raises(RuntimeError, match="drain"):
        run_candidate_session(
            Broken(), [req], manifest=load_candidate_package(path), journal_path=tmp_path / "journal.jsonl"
        )


@pytest.mark.parametrize("bad", [-100, float("nan"), "100"])
def test_stage_counters_cannot_cancel_work_or_supply_nonfinite_values(tmp_path, bad):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)

    class Broken:
        def plan(self, request):
            return PlanResult(
                [[0, 0], [0, 1]],
                work={"expansions": 0},
                stage_work={"primary": {"expansions": 100}, "fallback": {"expansions": bad}},
            )

    req = PlanRequest(
        "a",
        np.zeros((1, 2)),
        np.zeros((1, 2), bool),
        (0, 0),
        (0, 1),
        "elevation",
        62.4,
        budget=BudgetSpec(max_expansions=10),
    )
    run_candidate_session(
        Broken(), [req], manifest=load_candidate_package(path), journal_path=tmp_path / "journal.jsonl"
    )
    row = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert row["status"] == "runtime_error"


def test_terminal_stage_precision_and_resource_evidence_are_retained(tmp_path):
    from passage_eval.campaign.candidate.session import run_candidate_session
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)

    class Fallback:
        def plan(self, request):
            return PlanResult(
                [[0, 0], [0, 1]],
                stage="fallback",
                precision="fp32",
                work={"live_states": 1},
                resource_peaks={"live_states": 20},
                stage_spans={"fallback": 0.5},
            )

    req = PlanRequest(
        "a",
        np.zeros((1, 2)),
        np.zeros((1, 2), bool),
        (0, 0),
        (0, 1),
        "elevation",
        62.4,
        budget=BudgetSpec(max_live_states=10),
    )
    run_candidate_session(
        Fallback(), [req], manifest=load_candidate_package(path), journal_path=tmp_path / "journal.jsonl"
    )
    row = json.loads((tmp_path / "journal.jsonl").read_text().splitlines()[1])
    assert row["stage"] == "fallback" and row["precision"] == "fp32"
    assert row["resource_peaks"]["live_states"] == 20 and row["stage_spans"] == {"fallback": 0.5}
    assert row["status"] == "budget_exhausted"
