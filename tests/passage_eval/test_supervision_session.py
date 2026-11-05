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

"""Campaign session mode of the supervisor: warm requests only, dynamic request admission, session invalidation.

Real worker processes (process isolation) on the synthetic behaviours of ``test_supervision_execution.py`` and on the
synthetic validation world of ``campaign_world.py``. Admission clocks are scripted, so every closure point is
deterministic. Nothing reads the released dataset.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from passage_eval.cli import BenchSpec
from passage_eval.execution import read_run_artifact
from passage_eval.protocol.contracts import BudgetSpec
from passage_eval.supervision import WORKER_KILL_WAIT_S, sandbox
from passage_eval.supervision import session as S
from passage_eval.supervision.supervisor import (
    RosterInput,
    SupervisionError,
    SupervisorConfig,
    WarmupInput,
    run_supervised,
)

HERE = Path(__file__).resolve().parent
REWARM = "supervised_service_rewarm_v1"


def _load(name, filename):
    key = f"passage_eval_tests_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


X = _load("supervision_execution", "test_supervision_execution.py")
W = _load("campaign_world", "campaign_world.py")


@pytest.fixture(scope="module")
def cand_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("session-cand")
    (root / "sup_candidates.py").write_text(X.CANDIDATES)
    return root


def scripted(*values, then=None):
    """An admission clock that returns ``values`` in order, then ``then`` (default: the last value) forever."""
    remaining = list(values)
    last = then if then is not None else values[-1]
    return lambda: remaining.pop(0) if remaining else last


GRACE = 20.0  # runner_for's deadline grace
KILL = WORKER_KILL_WAIT_S * 1000.0
START = 120.0 * 1000.0 + KILL  # runner_for's start-up timeout
START_FOR_ONE_ROLE = 600.0 * 1000.0 + KILL  # the CLI's default start-up timeout


def reserve(budget_ms=5000.0, persistence_ms=1000.0):
    return budget_ms + GRACE + KILL + persistence_ms


# --------------------------------------------------------------------------------------------- admission (runner)


def test_a_request_is_engaged_only_while_the_window_holds_its_budget_grace_kill_wait_and_persistence(cand_dir):
    required = reserve()
    deadline = 100.0
    # 1 microsecond more than the reserve is left, then 1 ms less than the reserve
    clock = scripted(deadline - required / 1000.0 - 1e-6, deadline - (required - 1.0) / 1000.0)
    with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
        runner.session = S.SessionPolicy(deadline, 1000.0, clock=clock)
        assert runner.admission_required_ms(X.make_request(0)) == required
        record = X.measured(runner, X.make_request(0, "s0"), 0)  # the reserve is left: admitted
        assert record["status"] == "ok"
        generations = {role: worker.generation for role, worker in runner.workers.items()}
        with pytest.raises(S.SessionClosed) as closed:  # 1 ms short: closed before anything is sent
            X.measured(runner, X.make_request(0, "s1"), 1)
        assert closed.value.point == "before_measured_request" and closed.value.required_ms == required
        assert {role: worker.generation for role, worker in runner.workers.items()} == generations
        assert runner.events[-1]["event"] == "session_closed"


def test_without_a_session_every_request_is_engaged(cand_dir):
    with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
        assert runner.session is None
        assert X.measured(runner, X.make_request(0, "s0"), 0)["status"] == "ok"


def test_a_dead_worker_adds_its_start_to_the_reserve_of_the_next_request(cand_dir):
    with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
        runner.session = S.SessionPolicy(None, 1000.0)
        X.measured(runner, X.make_request(7, "s0"), 0)  # kills its worker
        assert runner.admission_required_ms(X.make_request(0)) == reserve() + START
        # a session without a window admits it, and restarts the worker as usual
        assert X.measured(runner, X.make_request(0, "s1"), 1)["status"] == "ok" and runner.restarts == 1


def test_a_request_without_a_wall_budget_reserves_the_hard_timeout(cand_dir):
    with X.runner_for(cand_dir, timing_profile=REWARM, hard_timeout_ms=7000.0) as runner:
        runner.session = S.SessionPolicy(None, 0.0)
        assert runner.admission_required_ms(X.make_request(0, budget=BudgetSpec())) == 7000.0 + KILL


def test_every_rewarm_request_is_admitted_on_its_own(cand_dir):
    warm = [X.make_request(21, f"w{i}") for i in range(2)]
    timeout = BudgetSpec(wall_time_ms=200.0, max_expansions=10)
    with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
        X.start_warm_pass(runner, warm)
        ok = 1000.0
        runner.session = S.SessionPolicy(2000.0, 0.0, clock=scripted(ok, ok, 2000.0))
        assert X.measured(runner, X.make_request(20, "s0", budget=timeout), 0)["status"] == "timeout"
        with pytest.raises(S.SessionClosed) as closed:  # restart, then the first re-warm request is not admitted
            X.measured(runner, X.make_request(0, "s1"), 1)
        assert closed.value.point == "before_rewarm_request" and runner.restarts == 1


# ------------------------------------------------------------------------------------------ session invalidation


def test_in_session_mode_the_restart_cap_invalidates_the_session_instead_of_failing_tasks(cand_dir):
    for session in (None, S.SessionPolicy(None, 0.0)):
        with X.runner_for(cand_dir, timing_profile=REWARM, max_restarts=0) as runner:
            runner.session = session
            X.start_warm_pass(runner, [X.make_request(0, "w0")])
            assert X.measured(runner, X.make_request(7, "s0"), 0)["status"] == "runtime_error"  # the crash: its own
            if session is None:
                record = X.measured(runner, X.make_request(0, "s1"), 1)
                assert record["diagnostics"]["reason"] == "worker_restart_cap"
                assert runner.tasks_marked_not_executed == 1
            else:
                with pytest.raises(S.SessionInvalid, match="worker_restart_cap"):
                    X.measured(runner, X.make_request(0, "s1"), 1)
                assert runner.tasks_marked_not_executed == 0


def test_in_session_mode_a_worker_that_cannot_restart_invalidates_the_session(cand_dir):
    for session in (None, S.SessionPolicy(None, 0.0)):
        with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
            runner.session = session
            X.measured(runner, X.make_request(7, "s0"), 0)
            original = runner._start_worker

            def failing(role, *, first):
                raise SupervisionError("the model cannot be loaded again")

            runner._start_worker = failing
            if session is None:
                assert X.measured(runner, X.make_request(0, "s1"), 1)["completion_mode"].startswith("not_executed")
            else:
                with pytest.raises(S.SessionInvalid, match="worker_restart_failed"):
                    X.measured(runner, X.make_request(0, "s1"), 1)
            runner._start_worker = original


def test_in_session_mode_a_pass_whose_workers_cannot_start_is_invalid(cand_dir):
    for session in (None, S.SessionPolicy(None, 0.0)):
        with X.runner_for(cand_dir, timing_profile=REWARM) as runner:
            runner.session = session

            def failing(role, *, first):
                raise SupervisionError("cannot start")

            runner._start_worker = failing
            if session is None:
                runner.begin_pass("warm/0")
                assert runner.events[-1]["event"] == "pass_start_failed"
            else:
                with pytest.raises(S.SessionInvalid, match="worker_start_failed"):
                    runner.begin_pass("warm/0")


def test_in_session_mode_the_initial_warmup_stops_at_its_first_failure_and_never_consumes_the_cap(cand_dir):
    warm = [X.make_request(7, "w0"), X.make_request(0, "w1")]  # the first warm-up request kills the worker
    outcomes = {}
    for name, session in (("legacy", None), ("session", S.SessionPolicy(None, 0.0))):
        with X.runner_for(cand_dir, timing_profile=REWARM, max_restarts=5) as runner:
            runner.session = session
            summary = X.start_warm_pass(runner, warm)
            outcomes[name] = (summary["requests"], summary["complete"], runner.restarts)
            first = runner.first_warmup_request
    assert outcomes["legacy"] == (2, False, 1)  # every request runs; its restart is counted
    assert outcomes["session"] == (1, False, 0)  # stops at the failure: no restart, nothing counted
    assert first["status"] == "runtime_error" and len(first["input_sha256"]) == 64


# ------------------------------------------------------------------------------------------ whole sessions (CLI)


def _release_row():
    from passage.utils import encode_tensor_blob

    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0], tensor[:, :, 1] = 0.75, -1
    tensor[0, 0, 1], tensor[2, 2, 1], tensor[0, 2, 2] = 0, 1, 1
    cost = float(60 * np.sqrt(2))
    return {
        "id": "0000000042",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_elevation_obstacles": [[0, 0], [1, 1], [2, 2]],
        "path_astar_elevation_free": [[0, 0], [1, 1], [2, 2]],
        "metadata": {
            "resolution": 3,
            "split": "validation",
            "index": 42,
            "calibration": {"min": -100.0, "max": 100.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
            "paths": {"astar_elevation": {"obstacles": {"cost": cost}, "free": {"cost": cost}}},
        },
    }


def _session_run(tmp_path, cand_dir, *extra, **changes):
    spec, spec_path = X.write_spec(tmp_path, _release_row(), cand_dir, latency_repeats=1, **changes)
    run = tmp_path / "run"
    code = X.cli_run(spec_path, run, cand_dir, "process", "--timing-profile", REWARM, "--campaign-session", *extra)
    assert code == 0
    return spec, run, json.loads((run / "session.json").read_text())


def test_a_complete_session_measures_warm_requests_only_and_reports_its_cold_start(tmp_path, cand_dir):
    from passage_eval.cli import cmd_score
    from passage_eval.supervision.staging import stage_for_scoring

    spec, run, receipt = _session_run(tmp_path, cand_dir, "--session-persistence-ms", "500")
    assert receipt["schema"] == S.SESSION_RECEIPT_SCHEMA and receipt["status"] == "complete"
    assert receipt["executed_sample_ids"] == ["0000000042"] and receipt["deferred_sample_ids"] == []
    assert receipt["run_artifact"] == "seed-1234" and receipt["admission"]["persistence_ms"] == 500.0
    assert receipt["admission"]["rule"] == S.ADMISSION_RULE_ID
    assert not (run / "seed-1234" / "cold").exists()  # no cold series
    warm = read_run_artifact(run / "seed-1234")
    assert warm.manifest["series"] == "warm" and len(warm.records) == 1
    session = warm.manifest["supervision"]["session"]
    assert session["status"] == "complete" and session["executed_sample_ids"] == ["0000000042"]
    cold = session["cold_start"]
    assert set(cold["worker_load_ms"]) == {"planner"} and cold["worker_load_ms"]["planner"] > 0
    assert cold["first_warmup_request"]["status"] == "ok" and cold["first_warmup_request"]["latency_ms"] > 0
    assert warm.manifest["procedure"]["supervision"]["session_policy"] == S.SESSION_POLICY_ID
    assert warm.manifest["supervision"]["timing_policy"]["cold_series"].startswith("none")
    assert warm.manifest["cold_series_scope"].startswith("none")
    assert [p["warmup"]["complete"] for p in warm.manifest["supervision"]["passes"]] == [True]
    staged = tmp_path / "staged"
    stage_for_scoring(run, spec, staged)
    cmd_score(staged)
    block = json.loads((staged / "seed-1234" / "metrics.json").read_text())["latency_all_repeats_censored"]
    assert block["warmup"]["performed"] is True and block["n_warm_requests"] == 1


def test_the_session_mode_changes_the_procedure_identity(tmp_path, cand_dir):
    _, run, _ = _session_run(tmp_path / "session", cand_dir)
    spec, spec_path = X.write_spec(tmp_path / "plain", _release_row(), cand_dir, latency_repeats=1)
    plain = tmp_path / "plain" / "run"
    assert X.cli_run(spec_path, plain, cand_dir, "process", "--timing-profile", REWARM) == 0
    ids = [read_run_artifact(r / "seed-1234").manifest["procedure_id"] for r in (run, plain)]
    assert ids[0] != ids[1]
    assert "session_policy" not in read_run_artifact(plain / "seed-1234").manifest["procedure"]["supervision"]


def test_a_session_whose_window_cannot_hold_its_workers_and_one_request_closes_before_start(tmp_path, cand_dir):
    deadline = repr(time.monotonic() + 1.0)
    _, run, receipt = _session_run(tmp_path, cand_dir, "--session-deadline-monotonic", deadline)
    assert receipt["status"] == "closed_by_admission" and receipt["closure"]["point"] == "before_start"
    assert receipt["closure"]["required_ms"] == START_FOR_ONE_ROLE + 5000.0 + 10.0 + KILL
    assert receipt["executed_sample_ids"] == [] and receipt["deferred_sample_ids"] == ["0000000042"]
    assert receipt["run_artifact"] is None and not (run / "seed-1234").exists()


def test_an_initial_warmup_failure_invalidates_the_session_and_charges_no_task(tmp_path, cand_dir):
    _, run, receipt = _session_run(tmp_path, cand_dir, inference_kwargs={"force_code": 7})
    assert receipt["status"] == "invalid" and receipt["invalid"]["reason"] == "initial_warmup_failed"
    assert receipt["executed_sample_ids"] == [] and receipt["deferred_sample_ids"] == []
    assert receipt["run_artifact"] is None and not (run / "seed-1234").exists()  # nothing to score or charge
    assert "warmup_failed" in receipt["events"] and "restart" not in receipt["events"]


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"measure_latency": False}, "declared warm-up"),
        ({"latency_warmup": 0}, "declared warm-up"),
        ({"latency_repeats": 2}, "one measured request per task"),
        ({"seeds": [1234, 2026]}, "one seed per session"),
    ],
)
def test_a_campaign_session_refuses_a_spec_that_is_not_one_warm_series_of_one_seed(
    tmp_path, cand_dir, changes, message
):
    spec, _ = X.write_spec(tmp_path, _release_row(), cand_dir, latency_repeats=1)
    spec = replace(spec, **changes)
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="process"),
        candidate_paths=(str(cand_dir),),
        timing_profile=REWARM,
    )
    with pytest.raises(ValueError, match=message):
        run_supervised(
            spec, tmp_path / "out", config, warmup=WarmupInput(split="calibration"), session=S.SessionPolicy(None, 0)
        )
    assert not (tmp_path / "out").exists()


def test_a_campaign_session_needs_a_rewarm_profile(tmp_path, cand_dir):
    spec, _ = X.write_spec(tmp_path, _release_row(), cand_dir, latency_repeats=1)
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="process"),
        candidate_paths=(str(cand_dir),),
        timing_profile="supervised_service_v1",
    )
    with pytest.raises(ValueError, match="re-warm timing profile"):
        run_supervised(
            spec, tmp_path / "out", config, warmup=WarmupInput(split="calibration"), session=S.SessionPolicy(None, 0)
        )


def test_the_session_flags_need_the_campaign_session_mode(tmp_path, cand_dir):
    _, spec_path = X.write_spec(tmp_path, _release_row(), cand_dir, latency_repeats=1)
    with pytest.raises(SystemExit):
        X.cli_run(spec_path, tmp_path / "run", cand_dir, "process", "--session-persistence-ms", "5")


# ------------------------------------------------------------ a session closed mid-roster, scored and continued


@pytest.fixture
def roster_session(tmp_path):
    """A validation batch of two tasks run by the harness A* planner under a scripted admission clock."""
    campaign = W.campaign_world(tmp_path / "campaign", procedures=("astar",))
    plan = campaign.plan
    item = campaign.manifest["batches"][0]
    child = Path(plan["partition"]["dir"]) / item["file"]
    spec = BenchSpec(**json.loads(Path(plan["procedures"]["astar"]["spec"]["path"]).read_text()))
    spec = replace(spec, latency_warmup=2)
    roster = RosterInput(str(child), item["roster_sha256"], plan["dataset_root"])
    warm = plan["warmup"]["roster"]
    warmup = WarmupInput(roster=RosterInput(warm["path"], warm["roster_sha256"], plan["dataset_root"]))
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="process"),
        timing_profile=REWARM,
        ipc_baseline_pings=3,
        max_restarts=2,
    )
    # admission clock calls: before start, two warm-up requests, the first measured request; then the window is spent
    policy = S.SessionPolicy(1.0e6, 0.0, clock=scripted(0.0, 0.0, 0.0, 0.0, then=1.0e6))
    run = tmp_path / "run"
    run_supervised(spec, run, config, roster=roster, warmup=warmup, session=policy)
    ids = [row["sample_id"] for row in item["parent_mapping"]]
    return campaign, item, child, run, ids


def test_a_session_closed_mid_roster_writes_its_measured_prefix_that_score_roster_scores(roster_session, tmp_path):
    from passage_eval.supervision.roster_scoring import score_roster_run

    campaign, item, child, run, ids = roster_session
    receipt = json.loads((run / "session.json").read_text())
    assert receipt["status"] == "closed_by_admission" and receipt["closure"]["point"] == "before_measured_request"
    assert receipt["executed_sample_ids"] == ids[:1] and receipt["deferred_sample_ids"] == ids[1:]
    artifact = read_run_artifact(run / "seed-2026")
    assert artifact.manifest["roster"] == ids[:1] and [r["sample_id"] for r in artifact.records] == ids[:1]
    assert artifact.manifest["supervision"]["roster"]["roster_sha256"] == item["roster_sha256"]
    scored = score_roster_run(run, child, item["roster_sha256"], campaign.plan["dataset_root"], tmp_path / "scored")
    (directory,) = scored
    metrics = json.loads((directory / "metrics.json").read_text())
    assert metrics["n_samples"] == 1


@pytest.mark.parametrize(
    "change, message",
    [
        (
            lambda s: s.update(
                executed_sample_ids=list(reversed(s["executed_sample_ids"] + s["deferred_sample_ids"]))[:1]
            ),
            "measured prefix",
        ),
        (lambda s: s.update(deferred_sample_ids=[]), "measured prefix"),
        (lambda s: s.update(status="complete"), "measured prefix"),
        (lambda s: s.update(status="invalid"), "measured prefix"),
        (lambda s: s.update(n_executed=0, executed_sample_ids=[]), "measured prefix"),
        (lambda s: s.update(n_tasks=3), "measured prefix"),
        (lambda s: s.update(policy="passage-campaign-session/v0"), "unknown session policy"),
    ],
)
def test_score_roster_refuses_a_session_that_does_not_declare_an_ordered_prefix(roster_session, change, message):
    from passage_eval.supervision.roster_scoring import RosterScoringError, executed_task_ids

    _, _, _, run, ids = roster_session
    manifest = read_run_artifact(run / "seed-2026").manifest
    assert executed_task_ids(manifest, ids) == ids[:1]
    change(manifest["supervision"]["session"])
    with pytest.raises(RosterScoringError, match=message):
        executed_task_ids(manifest, ids)


def test_a_manifest_without_a_session_must_cover_every_roster_task():
    from passage_eval.supervision.roster_scoring import executed_task_ids

    assert executed_task_ids({"supervision": {}}, ["a", "b"]) == ["a", "b"]
