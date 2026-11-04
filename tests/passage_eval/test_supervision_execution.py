"""Supervised execution against real worker processes (process and bwrap isolation)."""

from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import socket
import struct
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage_eval.cli import BenchSpec, cmd_score
from passage_eval.execution import read_run_artifact, run_requests
from passage_eval.protocol.attribution import component_control
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest
from passage_eval.supervision import POLICY_DECODER_ONLY, POLICY_ORDINARY_REQUEST, TIMING_BOUNDARY_ID, identity, sandbox
from passage_eval.supervision.__main__ import main as supervision_main
from passage_eval.supervision.protocol import FrameLimits, StreamClosed, read_frame
from passage_eval.supervision.redaction import request_token
from passage_eval.supervision.staging import StagingError, stage_for_scoring
from passage_eval.supervision.supervisor import (
    SplitSpec,
    SupervisionError,
    SupervisorConfig,
    WorkerProcess,
    build_runner,
    inspect_environment,
    run_supervised,
)

BWRAP = sandbox.available_bwrap()
ISOLATIONS = [
    pytest.param("process", id="process"),
    pytest.param("bwrap", id="bwrap", marks=pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")),
]
SENTINEL = 4242.4242

CANDIDATES = r'''
import gc, os, socket, subprocess, sys, time
import numpy as np
from passage_eval.protocol.contracts import BudgetSpec, PlanResult

def route(request):
    return [request.start, (1, 1), request.goal]

class Behaviour:
    """The behaviour is selected by request.elevation[1, 0] (or kwargs force_code)."""
    def __init__(self, **kw):
        self.kw = kw

    def plan(self, request):
        code = int(self.kw.get("force_code", request.elevation[1, 0]))
        r = route(request)
        if code in (0, 50):
            return PlanResult(r, work={"expansions": 3}, precision="float32", diagnostics={"note": "good"})
        if code == 1:
            return PlanResult(r)
        if code == 2:
            return PlanResult(r, work={"expansions": 10 ** 6})
        if code == 3:
            return PlanResult([request.start, request.goal], work={"expansions": 1})
        if code == 4:
            return PlanResult([], work={"expansions": 1})
        if code == 5:
            raise RuntimeError("boom")
        if code == 6:
            sys.exit(3)
        if code == 7:
            os._exit(9)
        if code == 8:
            print("noise on stdout"); os.write(1, b"raw noise on fd 1"); sys.stdout.flush()
            return PlanResult(r, work={"expansions": 3})
        if code == 9:
            return PlanResult(np.array([[0, 0], [1.5, 1], [2, 2]]), work={"expansions": 1})
        if code == 10:
            return PlanResult(r, stage="fallback", stage_work={"primary": {"expansions": 2}, "fallback": {"expansions": 3}})
        if code == 11:
            return PlanResult(r, work={"expansions": float("nan")})
        if code == 12:
            return PlanResult([("a", "b")], work={"expansions": 1})
        if code == 13:
            return PlanResult(r, stage="repair", stage_work={"primary": {"expansions": 1}, "repair": {"expansions": 1}})
        if code == 14:
            raise MemoryError("simulated")
        if code == 20:
            if self.kw.get("pid_file"):
                child = subprocess.Popen(["sleep", "60"])
                with open(self.kw["pid_file"], "w") as stream:
                    stream.write(str(child.pid))
            time.sleep(30)
            return PlanResult(r, work={"expansions": 3})
        if code == 21:
            time.sleep(0.15)
            return PlanResult(r, work={"expansions": 3})
        if code == 30:
            return self.probe(request)
        if code == 31:
            for fd in range(3, 12):
                try:
                    os.write(fd, b"GARBAGE-NOT-A-FRAME-PREFIX-GARBAGE")
                except OSError:
                    pass
            return PlanResult(r, work={"expansions": 3})
        if code == 32:
            with open(self.kw["source_file"], "a") as stream:
                stream.write("\n# tampered while running\n")
            return PlanResult(r, work={"expansions": 3})
        if code == 33:
            return PlanResult(r, work={"expansions": 1}, diagnostics={
                "sample_id": request.sample_id, "metadata": request.metadata, "has_oracle": hasattr(request, "oracle_cost"),
                "budget": request.budget.to_dict(), "remaining_ms": request.remaining_wall_ms(),
                "has_tensor": request.input_tensor is not None, "cost_weight": request.cost_weight,
                "graph": request.graph.to_dict(), "elevation_shape": list(request.elevation.shape)})
        if code == 34:  # a warm-up input that succeeds until the marker exists, then kills its worker
            if os.path.exists(self.kw["marker"]):
                os._exit(9)
            return PlanResult(r, work={"expansions": 3})
        if code == 35:  # a roster input that creates the marker, then kills its worker
            open(self.kw["marker"], "w").close()
            os._exit(9)
        raise AssertionError("unknown behaviour code %r" % code)

    def probe(self, request):
        kw, out = {k: (v[2:] if isinstance(v, str) and v.startswith("p:") else v) for k, v in self.kw.items()}, {}
        def attempt(name, fn):
            try:
                out[name] = fn()
            except Exception as error:
                out[name] = type(error).__name__
        out["env"] = sorted(os.environ)
        out["secret_in_env_values"] = any(kw["secret"] in value for value in os.environ.values())
        attempt("secret_in_proc_environ", lambda: kw["secret"].encode() in open("/proc/self/environ", "rb").read())
        out["home"], out["cwd"], out["tmpdir"] = os.environ.get("HOME"), os.getcwd(), os.environ.get("TMPDIR")
        out["pid"], out["uid"] = os.getpid(), os.getuid()
        attempt("open_reference", lambda: (open(kw["reference_path"]).read(), "opened")[1])
        attempt("exists_dataset", lambda: os.path.exists(kw["dataset_dir"]))
        attempt("exists_real_ssh", lambda: os.path.exists(os.path.join(kw["real_home"], ".ssh")))
        attempt("root_entries", lambda: sorted(os.listdir("/")))
        attempt("write_source_dir", lambda: (open(os.path.join(kw["source_dir"], "x.txt"), "w").write("x"), "wrote")[1])
        attempt("write_home", lambda: (open(os.path.join(os.environ["HOME"], "probe.txt"), "w").write("x"), "wrote")[1])
        attempt("write_tmp", lambda: (open(os.path.join(os.environ["TMPDIR"], "probe.txt"), "w").write("x"), "wrote")[1])
        def connect(address):
            def run():
                s = socket.socket(); s.settimeout(2)
                try:
                    s.connect(address)
                    return "connected"
                finally:
                    s.close()
            return run
        attempt("connect_host_loopback", connect(("127.0.0.1", kw["port"])))
        attempt("connect_external", connect(("192.0.2.1", 9)))
        return PlanResult(route(request), work={"expansions": 1}, diagnostics=out)

class Predictor:
    def __init__(self, **kw):
        self.kw = kw
        self.calls = 0
    def predict(self, request):
        assert request.sample_id.startswith("req-"), request.sample_id
        mode = self.kw.get("mode", "ok")
        self.calls += 1
        if mode == "alternate":
            mode = "two_fields" if self.calls % 2 == 1 else "ok"
        shape = request.elevation.shape
        if mode == "two_fields":
            return (np.ones(shape, dtype=np.float32), np.zeros(shape, dtype=np.float16))
        if mode == "elevation_second_channel":      # the reviewer's attack: leak the real elevation as field 2
            return (np.zeros(shape, dtype=np.float32), request.elevation.astype(np.float64))
        if mode == "float64_field":
            return np.ones(shape, dtype=np.float64)
        if mode == "int_field":
            return np.ones(shape, dtype=np.int32)
        if mode == "wrong_shape":
            return np.ones((shape[0] + 1, shape[1]), dtype=np.float32)
        if mode == "nan_field":
            return np.full(shape, np.nan, dtype=np.float32)
        return np.ones(shape, dtype=np.float32)

class Decoder:
    def __init__(self, **kw):
        self.kw = kw
    def decode(self, field, start, goal, obstacles, budget):
        mode = self.kw.get("mode", "ok")
        if mode == "sleep":
            time.sleep(30)
        if mode == "raise":
            raise ValueError("decoder failed")
        fields = field if isinstance(field, tuple) else (field,)
        info = {"n_fields": len(fields), "field_shapes": [list(f.shape) for f in fields],
                "field_dtypes": [str(f.dtype) for f in fields], "start": list(start), "goal": list(goal),
                "start_type": type(start).__name__, "obstacles_dtype": str(obstacles.dtype),
                "budget_type": type(budget).__name__, "budget_wall_ms": budget.wall_time_ms,
                "budget_expansions": budget.max_expansions, "field_is_ndarray": isinstance(fields[0], np.ndarray),
                "pid": os.getpid()}
        return PlanResult([start, (1, 1), goal], work={"expansions": 5}, diagnostics=info)
'''


@pytest.fixture(scope="module")
def cand_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("supervision-cand")
    (root / "sup_candidates.py").write_text(CANDIDATES)
    return root


def make_request(code=0, sample_id="s0", budget="default", sentinel=False, metadata=None):
    elevation = np.zeros((3, 3))
    elevation[1, 0] = code
    if sentinel:
        elevation[2, 0] = SENTINEL
    if budget == "default":
        budget = BudgetSpec(wall_time_ms=5000.0, max_expansions=10)
    return PlanRequest(
        sample_id,
        elevation,
        np.zeros((3, 3), dtype=bool),
        (0, 0),
        (2, 2),
        "elevation",
        62.4,
        budget=budget,
        metadata=metadata or {},
    )


def bench_spec(entry, kwargs=None, **changes):
    spec = BenchSpec(
        "supervised-test",
        "classical",
        "fixture",
        inference=entry,
        inference_format="request",
        inference_kwargs=dict(kwargs or {}),
    )
    return replace(spec, **changes)


@contextlib.contextmanager
def runner_for(
    cand_dir,
    entry="sup_candidates:Behaviour",
    *,
    isolation="process",
    kwargs=None,
    split=None,
    env_set=None,
    ro_binds=(),
    protected=(),
    runtime_binds=(),
    **config_changes,
):
    spec = bench_spec(entry, kwargs)
    iso = sandbox.IsolationConfig(
        level=isolation,
        env_set=env_set or {},
        ro_binds=tuple(ro_binds),
        runtime_binds=tuple(runtime_binds),
        protected_paths=tuple(protected),
    )
    settings = dict(ipc_baseline_pings=3, ipc_baseline_warmup=1, startup_timeout_s=120.0, deadline_grace_ms=20.0)
    settings.update(config_changes)
    config = SupervisorConfig(worker_python=sys.executable, isolation=iso, candidate_paths=(str(cand_dir),), **settings)
    entries = [entry] if split is None else [split.predictor, split.decoder]
    environment = inspect_environment(config, entries)
    with build_runner(spec, split, config, environment, seed=1) as runner:
        yield runner


def run_one(runner, request, index=0, repeat=0, series="warm"):
    return runner.run_request(request, request_token(runner.secret, index), repeat, series)


def gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as stream:
            return stream.read().rsplit(")", 1)[1].split()[0] == "Z"
    except OSError:
        return True


def group_members(pgid):
    members = []
    for name in os.listdir("/proc"):
        if name.isdigit():
            try:
                with open(f"/proc/{name}/stat") as stream:
                    fields = stream.read().rsplit(")", 1)[1].split()
                if int(fields[2]) == pgid and fields[0] != "Z":
                    members.append(int(name))
            except (OSError, IndexError, ValueError):
                pass
    return members


def wait_until(predicate, seconds=5.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


# ---------------------------------------------------------------------------
# request/response, timing boundary, redaction at the worker
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_round_trip_timing_fields_and_boundary(cand_dir, isolation):
    with runner_for(cand_dir, isolation=isolation) as runner:
        records = [run_one(runner, make_request(0, f"s{i}"), index=i)[0] for i in range(3)]
        baseline = runner.measure_baseline("planner", 4096)
    for record in records:
        assert record["status"] == "ok" and record["legal_path"] and record["safe_completion"]
        assert record["timing_boundary"] == TIMING_BOUNDARY_ID == "passage-timing-boundary/v3-supervised-ipc-fresh-pass"
        for key in (
            "service_wall_ms",
            "candidate_reported_ms",
            "validation_recost_ms",
            "total_wall_ms",
            "inter_hop_ms",
            "candidate_wall_ms",
            "worker_restart_ms_before_request",
        ):
            assert np.isfinite(record[key]) and record[key] >= 0, key
        assert record["total_wall_ms"] == pytest.approx(
            record["service_wall_ms"] + record["validation_recost_ms"], abs=1e-9
        )
        assert record["service_wall_ms"] >= record["candidate_reported_ms"] > 0
        assert record["validation_recost_ms"] > 0 and record["preempted"] is False
        assert record["wire_policies"] == [POLICY_ORDINARY_REQUEST] and record["completion_mode"] == "worker_replied"
        (hop,) = record["hops"]
        assert hop["role"] == "planner" and hop["service_wall_ms"] <= record["service_wall_ms"]
        assert set(hop["worker_spans"]) == {"request_build_ms", "call_ms", "sync_ms", "pack_ms"}
        assert record["diagnostics"] == {"note": "good"} and record["precision"] == "float32"
    assert [row["mode"] for row in baseline] == ["round_trip_same_size", "request_size_out_small_back"]
    for row in baseline:
        assert row["payload_bytes"] == 4096 and row["n"] == 3 and 0 < row["min_ms"] <= row["median_ms"] <= row["max_ms"]
        assert row["iqr_ms"] >= 0 and row["std_ms"] >= 0


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_large_frames_cross_the_pipes_in_both_directions_without_deadlock(cand_dir, isolation):
    size = 24 << 20  # far beyond the 64 KiB pipe buffer
    with runner_for(cand_dir, isolation=isolation) as runner:
        rows = runner.measure_baseline("planner", size)
        record, _ = run_one(runner, make_request(0))
    assert [row["payload_bytes"] for row in rows] == [size, size] and rows[0]["max_ms"] < 5000.0
    assert rows[0]["median_ms"] > rows[1]["median_ms"] > 0  # returning the payload costs more than a short reply
    assert record["status"] == "ok"


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_candidate_sees_an_opaque_token_and_no_metadata_by_default(cand_dir, isolation):
    metadata = {"split": "validation", "index": 7, "source_id": "tile-x", "cluster_id": "c1"}
    with runner_for(cand_dir, isolation=isolation) as runner:
        record, _ = run_one(runner, make_request(33, "real-sample-id", metadata=metadata), index=4)
    seen = record["diagnostics"]
    assert seen["sample_id"] == request_token(runner.secret, 4) and "real-sample-id" not in json.dumps(seen)
    assert seen["metadata"] == {} and seen["has_oracle"] is False and seen["has_tensor"] is False
    assert seen["budget"]["max_expansions"] == 10 and seen["graph"]["connectivity"] == 8
    assert 0 < seen["remaining_ms"] <= 5000.0 and seen["cost_weight"] == 62.4


def test_metadata_allow_list_reaches_the_candidate_only_when_declared(cand_dir):
    metadata = {"split": "validation", "index": 7, "source_id": "tile-x"}
    with runner_for(cand_dir, metadata_allow=("index", "split")) as runner:
        record, _ = run_one(runner, make_request(33, "real-sample-id", metadata=metadata))
    assert record["diagnostics"]["metadata"] == {"index": 7, "split": "validation"}


# ---------------------------------------------------------------------------
# environment scrubbing and sandbox limits
# ---------------------------------------------------------------------------


def probe_kwargs(tmp_path, port, source_dir):
    reference = tmp_path / "private" / "reference.json"
    reference.parent.mkdir()
    reference.write_text('{"optimal_path": "SECRET_REFERENCE"}')
    dataset = tmp_path / "private" / "dataset"
    dataset.mkdir()
    # "p:" keeps the undeclared locations out of the kwargs path policy; the probe strips it.
    return dict(
        secret="hf_dummy_secret_value",
        reference_path="p:" + str(reference),
        dataset_dir="p:" + str(dataset),
        real_home="p:" + str(Path.home()),
        source_dir="p:" + str(source_dir),
        port=port,
        force_code=30,
    )


@pytest.fixture
def listener():
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(8)
    yield server.getsockname()[1]
    server.close()


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_supervisor_secrets_never_reach_the_worker_environment(cand_dir, tmp_path, monkeypatch, listener, isolation):
    for name, value in (
        ("HF_TOKEN", "hf_dummy_secret_value"),
        ("AWS_SECRET_ACCESS_KEY", "hf_dummy_secret_value"),
        ("PYTHONPATH", "/definitely/not/there"),
        ("WANDB_API_KEY", "hf_dummy_secret_value"),
        ("LD_PRELOAD", ""),
        ("MY_PRIVATE_SETTING", "hf_dummy_secret_value"),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("OMP_NUM_THREADS", "3")
    kwargs = probe_kwargs(tmp_path, listener, cand_dir)
    with runner_for(cand_dir, isolation=isolation, kwargs=kwargs, env_set={"MY_EXPLICIT": "yes"}) as runner:
        record, _ = run_one(runner, make_request(30))
    seen = record["diagnostics"]
    assert record["status"] == "ok"
    for leaked in (
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "PYTHONPATH",
        "WANDB_API_KEY",
        "LD_PRELOAD",
        "MY_PRIVATE_SETTING",
    ):
        assert leaked not in seen["env"]
    assert seen["secret_in_env_values"] is False and seen["secret_in_proc_environ"] is False
    assert "OMP_NUM_THREADS" in seen["env"] and "MY_EXPLICIT" in seen["env"]
    assert seen["home"] != os.environ.get("HOME", "")
    assert seen["cwd"] == seen["home"]
    assert not seen["tmpdir"].startswith(str(tmp_path))


@pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")
def test_bwrap_hides_references_home_dataset_and_network(cand_dir, tmp_path, listener):
    kwargs = probe_kwargs(tmp_path, listener, cand_dir)
    with runner_for(cand_dir, isolation="bwrap", kwargs=kwargs, protected=[str(tmp_path / "private")]) as runner:
        record, _ = run_one(runner, make_request(30))
    seen = record["diagnostics"]
    assert record["status"] == "ok"
    assert seen["open_reference"] == "FileNotFoundError"
    assert seen["exists_dataset"] is False and seen["exists_real_ssh"] is False
    assert seen["connect_host_loopback"] != "connected" and seen["connect_external"] != "connected"
    assert seen["write_source_dir"] in {"OSError", "PermissionError"}  # read-only bind
    assert seen["write_home"] == "wrote" and seen["write_tmp"] == "wrote"  # tmpfs HOME and /tmp
    assert seen["home"] == sandbox.SANDBOX_HOME and seen["tmpdir"] == "/tmp"
    assert seen["pid"] < 10  # own pid namespace
    assert set(seen["root_entries"]) <= {
        "bin",
        "dev",
        "etc",
        "home",
        "lib",
        "proc",
        "sbin",
        "tmp",
        "usr",
        "lib32",
        "lib64",
        "libx32",
    } | {Path(b.target).parts[1] for b in runner.workers["planner"].binds}
    isolation = runner.isolation_record
    assert isolation["level"] == "bwrap" and isolation["network"] == "unshared_loopback_only"
    assert str(tmp_path / "private") not in " ".join(isolation["read_only_binds"])


def test_process_isolation_is_only_fault_containment_and_says_so(cand_dir, tmp_path, listener):
    """Without bwrap the same uid can read files and reach the network; the record must not claim otherwise."""
    kwargs = probe_kwargs(tmp_path, listener, cand_dir)
    with runner_for(cand_dir, isolation="process", kwargs=kwargs) as runner:
        record, _ = run_one(runner, make_request(30))
    seen = record["diagnostics"]
    assert seen["open_reference"] == "opened" and seen["connect_host_loopback"] == "connected"
    isolation = runner.isolation_record
    assert isolation["level"] == "process" and isolation["same_uid_as_supervisor"] is True
    assert isolation["network"] == "shared_with_host" and isolation["read_only_binds"] == []
    assert "filesystem_isolation" in isolation["not_applied"]


@pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")
def test_a_bind_that_would_expose_a_protected_path_is_refused(cand_dir, tmp_path):
    (tmp_path / "dataset").mkdir()
    with pytest.raises(sandbox.SandboxError, match="overlaps"):
        with runner_for(cand_dir, isolation="bwrap", ro_binds=[tmp_path], protected=[str(tmp_path / "dataset")]):
            pass


def test_worker_applies_rlimits_and_the_manifest_reports_readback(cand_dir):
    with runner_for(cand_dir) as runner:
        ready = runner.ready["planner"]
    assert ready["rlimits"]["core"] == [0, 0] and ready["rlimits"]["nofile"] == [1024, 1024]
    assert ready["rlimits"]["fsize"] == [1 << 30, 1 << 30] and ready["pdeathsig"] is True


# ---------------------------------------------------------------------------
# preemption, failures and restarts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_overdue_candidate_is_killed_and_the_next_request_succeeds(cand_dir, tmp_path, isolation):
    pid_file = tmp_path / "child.pid"
    kwargs = {"pid_file": str(pid_file)} if isolation == "process" else {}
    with runner_for(cand_dir, isolation=isolation, kwargs=kwargs) as runner:
        first, _ = run_one(runner, make_request(0, "a", budget=BudgetSpec(wall_time_ms=400.0, max_expansions=10)))
        worker = runner.workers["planner"]
        pid = worker.proc.pid
        begun = time.monotonic()
        slow, raw = run_one(runner, make_request(20, "b", budget=BudgetSpec(wall_time_ms=400.0, max_expansions=10)), 1)
        elapsed = time.monotonic() - begun
        assert not worker.alive and worker.proc.returncode is not None
        assert wait_until(lambda: not group_members(pid))  # whole process group is gone
        if isolation == "process":
            assert wait_until(lambda: gone(int(pid_file.read_text())))  # grandchild killed with the group
        after, _ = run_one(runner, make_request(0, "c", budget=BudgetSpec(wall_time_ms=400.0, max_expansions=10)), 2)
        events = list(runner.events)
        assert worker.generation == 1 and worker.alive
    assert first["status"] == "ok"
    assert slow["status"] == "timeout" and slow["preempted"] is True and slow["late_completion"] is True
    assert slow["completion_mode"] == "worker_killed_after_deadline" and slow["diagnostics"]["preempted"] is True
    assert 400.0 < slow["total_wall_ms"] < 2000.0 and elapsed < 15  # not the candidate's 30 s
    # service time ends when the deadline+grace expired, before the kill and reaping housekeeping
    assert 420.0 <= slow["service_wall_ms"] < 420.0 + 250.0 and slow["service_wall_ms"] == slow["total_wall_ms"]
    assert slow["failed_attempt_ms"] == 400.0 and slow["drain_ms"] == pytest.approx(slow["total_wall_ms"] - 400.0)
    assert raw.shape == (0, 2) and slow["legal_path"] is False and slow["validation_recost_ms"] == 0.0
    assert after["status"] == "ok" and after["hops"][0]["worker_generation"] == 1
    kinds = [event["event"] for event in events]
    assert kinds == ["preempt_kill", "restart"]
    restart = events[1]
    assert restart["restart_ms"] > 0 and after["worker_restart_ms_before_request"] == pytest.approx(
        restart["restart_ms"]
    )
    assert after["total_wall_ms"] < restart["restart_ms"]  # restart is not charged to the request


def test_hard_timeout_applies_when_no_wall_budget_is_configured(cand_dir):
    with runner_for(cand_dir, hard_timeout_ms=700.0) as runner:
        record, _ = run_one(runner, make_request(20, budget=BudgetSpec()))
    assert record["status"] == "timeout" and record["preempted"] is True
    assert record["deadline_ms"] is None and record["late_completion"] is False
    assert 700.0 <= record["total_wall_ms"] < 3000.0


def test_every_failure_mode_is_a_terminal_record_and_the_worker_recovers(cand_dir):
    expected = [  # (code, status, worker restarted before the following request)
        (0, "ok", False),
        (5, "runtime_error", False),
        (14, "oom", False),
        (6, "runtime_error", True),
        (0, "ok", False),
        (7, "runtime_error", True),
        (0, "ok", False),
        (31, "runtime_error", True),
        (0, "ok", False),
        (8, "ok", False),
        (0, "ok", False),
    ]
    with runner_for(cand_dir) as runner:
        records = []
        for index, (code, _, _) in enumerate(expected):
            records.append(run_one(runner, make_request(code, f"s{index}"), index)[0])
        events = [event["event"] for event in runner.events]
    assert [r["status"] for r in records] == [status for _, status, _ in expected]
    restarted = [r["worker_restart_ms_before_request"] > 0 for r in records]
    assert restarted == [False] + [flag for _, _, flag in expected[:-1]]
    assert records[1]["diagnostics"]["error_type"] == "RuntimeError" and records[1]["diagnostics"]["error"] == "boom"
    assert records[3]["diagnostics"]["worker_fatal"] is True and records[3]["diagnostics"]["error_type"] == "SystemExit"
    assert records[5]["diagnostics"]["failure"] == "crashed" and records[5]["diagnostics"]["worker_returncode"] == 9
    assert records[5]["completion_mode"] == "worker_exited"
    assert records[7]["diagnostics"]["failure"] == "protocol_error" and "magic" in records[7]["diagnostics"]["error"]
    assert records[7]["completion_mode"] == "worker_killed_after_protocol_error"
    assert events == ["worker_fatal_error", "restart", "worker_gone", "restart", "protocol_error", "restart"]
    assert all(r["safe_completion"] is True for r in records)


def test_worker_rejects_malformed_or_misdirected_frames_and_exits(cand_dir):
    spec = bench_spec("sup_candidates:Behaviour")
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(),
        candidate_paths=(str(cand_dir),),
        startup_timeout_s=120.0,
    )
    environment = inspect_environment(config, [spec.inference])
    with build_runner(spec, None, config, environment, seed=1) as runner:
        from passage_eval.supervision.protocol import encode_frame

        worker = runner.workers["planner"]
        # a result frame is a worker->supervisor message; a worker must refuse it
        worker.send(encode_frame("result", worker.next_seq(), {}), time.perf_counter() + 5)
        reply = worker.receive(time.perf_counter() + 30)
        assert reply.type == "error" and reply.body["fatal"] is True and reply.body["phase"] == "protocol"
        assert wait_until(lambda: worker.proc.poll() == 3)
        worker.kill()
        runner.ensure_ready()
        # raw garbage is equally fatal
        worker.send(b"X" * 64, time.perf_counter() + 5)
        assert worker.receive(time.perf_counter() + 30).body["fatal"] is True
        assert wait_until(lambda: worker.proc.poll() == 3)
        worker.kill()
        runner.ensure_ready()
        # a predict message sent to a planner is not valid for its role
        worker.send(encode_frame("predict", worker.next_seq(), {}), time.perf_counter() + 5)
        assert worker.receive(time.perf_counter() + 30).body["fatal"] is True


def test_a_worker_that_cannot_load_the_candidate_aborts_the_run_with_context(cand_dir):
    with pytest.raises(SupervisionError, match="refused to load"):
        with runner_for(cand_dir, entry="sup_candidates:DoesNotExist"):
            pass


# ---------------------------------------------------------------------------
# legality / budget semantics identical to the in-process runner
# ---------------------------------------------------------------------------

PARITY_CODES = [0, 1, 2, 3, 4, 5, 9, 10, 11, 12]
COMPARED = (
    "status",
    "route_hash",
    "recost",
    "connected_path",
    "legal_path",
    "obstacle_violation",
    "obstacle_cells",
    "work",
    "stage",
    "stage_work",
    "fallback",
    "precision",
    "stage_spans",
    "resource_peaks",
    "deadline_ms",
    "late_completion",
)


def in_process_records(monkeypatch, cand_dir, codes, control=None):
    monkeypatch.syspath_prepend(str(cand_dir))
    module = importlib.import_module("sup_candidates")
    planner = module.Behaviour()
    sys.modules.pop("sup_candidates", None)  # keep the pytest process free of candidate modules
    artifact = run_requests(
        planner,
        [make_request(code, f"s{i}") for i, code in enumerate(codes)],
        manifest={"component_control": control} if control else None,
    )
    return artifact.records


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_statuses_costs_and_work_match_the_in_process_runner(cand_dir, monkeypatch, isolation):
    expected = in_process_records(monkeypatch, cand_dir, PARITY_CODES)
    with runner_for(cand_dir, isolation=isolation) as runner:
        actual = [run_one(runner, make_request(code, f"s{i}"), i)[0] for i, code in enumerate(PARITY_CODES)]
    assert [r["status"] for r in actual] == [
        "ok",
        "runtime_error",
        "budget_exhausted",
        "invalid_path",
        "no_path",
        "runtime_error",
        "invalid_path",
        "ok",
        "runtime_error",
        "invalid_path",
    ]
    for code, ours, theirs in zip(PARITY_CODES, actual, expected):
        for key in COMPARED:
            assert ours[key] == theirs[key], (code, key, ours[key], theirs[key])
        # durations differ by nature, but their accounting rule is the runner's
        assert ours["failed_attempt_ms"] == (0.0 if ours["status"] == "ok" else ours["total_wall_ms"])
        assert ours["drain_ms"] == 0.0 and theirs["drain_ms"] == 0.0
        assert theirs["diagnostics"].items() <= ours["diagnostics"].items(), code
    by_code = dict(zip(PARITY_CODES, actual))
    assert by_code[1]["diagnostics"]["unobservable_work_budgets"] == ["expansions"]
    assert by_code[2]["diagnostics"]["exceeded_work_budgets"] == ["expansions"]
    assert by_code[10]["fallback"] is True and by_code[10]["work"]["expansions"] == 5
    assert "malformed_raw_output" in by_code[12]["diagnostics"]


@pytest.mark.parametrize("variant", ["direct", "bounded_repair"])
def test_declared_repair_and_fallback_controls_match_the_in_process_runner(cand_dir, monkeypatch, variant):
    control = {"variant": variant, "guide": "learned", "checkpoint_id": "ckpt-1"}
    if variant == "bounded_repair":
        control["machinery_id"] = "decoder-1"
    codes = [0, 10, 13]
    expected = in_process_records(monkeypatch, cand_dir, codes, control)
    with runner_for(cand_dir) as runner:
        runner.control = component_control(control)
        actual = [run_one(runner, make_request(code, f"s{i}"), i)[0] for i, code in enumerate(codes)]
    for ours, theirs in zip(actual, expected):
        assert ours["status"] == theirs["status"] and ours["fallback"] == theirs["fallback"]
        assert ours["diagnostics"].get("component_control_violation") == theirs["diagnostics"].get(
            "component_control_violation"
        )
    assert actual[1]["status"] == "runtime_error" and actual[1]["diagnostics"]["component_control_violation"]


def test_a_reply_that_arrives_after_the_deadline_but_inside_the_grace_is_a_late_timeout(cand_dir):
    budget = BudgetSpec(wall_time_ms=100.0, max_expansions=10)
    with runner_for(cand_dir, deadline_grace_ms=3000.0) as runner:
        record, raw = run_one(runner, make_request(21, budget=budget))
        follow_up, _ = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=3000.0, max_expansions=10)), 1)
        assert runner.events == []  # drained, not killed
    assert record["status"] == "timeout" and record["preempted"] is False and record["late_completion"] is True
    assert (
        record["diagnostics"]["status_before_deadline_policy"] == "ok" and record["completion_mode"] == "worker_replied"
    )
    assert record["legal_path"] is True and raw.tolist() == [[0, 0], [1, 1], [2, 2]]  # kept, but never a success
    assert record["failed_attempt_ms"] == 100.0 and record["drain_ms"] == pytest.approx(record["total_wall_ms"] - 100.0)
    assert 150.0 <= record["service_wall_ms"] < 2500.0 and follow_up["status"] == "ok"


# ---------------------------------------------------------------------------
# split mode: a decoder never receives elevation
# ---------------------------------------------------------------------------


def capture_wire(monkeypatch):
    sent = []
    original = WorkerProcess.send

    def spy(self, data, deadline):
        sent.append((self.spec.role, bytes(data)))
        return original(self, data, deadline)

    monkeypatch.setattr(WorkerProcess, "send", spy)
    return sent


def parse(data):
    stream = io.BytesIO(data)

    def read_exact(count):
        chunk = stream.read(count)
        if len(chunk) < count:
            raise StreamClosed(len(chunk))
        return chunk

    return read_frame(read_exact)


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_split_decoder_receives_only_the_allowed_arguments_and_never_elevation(cand_dir, monkeypatch, isolation):
    sent = capture_wire(monkeypatch)
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder", precision="float32")
    with runner_for(cand_dir, isolation=isolation, split=split) as runner:
        record, raw = run_one(runner, make_request(0, "split-sample-77", sentinel=True))
        generations = {role: w.generation for role, w in runner.workers.items()}
    assert record["status"] == "ok" and record["legal_path"] and raw.tolist() == [[0, 0], [1, 1], [2, 2]]
    info = record["diagnostics"]
    assert info == {
        **info,
        "n_fields": 1,
        "field_shapes": [[3, 3]],
        "field_dtypes": ["float32"],
        "start": [0, 0],
        "goal": [2, 2],
        "start_type": "tuple",
        "obstacles_dtype": "bool",
        "budget_type": "BudgetSpec",
        "budget_expansions": 10,
        "field_is_ndarray": True,
    }
    assert 0 < info["budget_wall_ms"] <= 5000.0  # remaining wall time at dispatch
    assert record["precision"] == "float32" and record["wire_policies"] == [
        POLICY_ORDINARY_REQUEST,
        POLICY_DECODER_ONLY,
    ]
    assert [hop["role"] for hop in record["hops"]] == ["predictor", "decoder"] and generations == {
        "predictor": 0,
        "decoder": 0,
    }
    assert record["inter_hop_ms"] >= 0 and record["service_wall_ms"] >= sum(
        h["service_wall_ms"] for h in record["hops"]
    )
    assert record["candidate_reported_ms"] == pytest.approx(sum(h["candidate_reported_ms"] for h in record["hops"]))
    assert record["total_wall_ms"] == pytest.approx(record["service_wall_ms"] + record["validation_recost_ms"])
    pattern = struct.pack("<d", SENTINEL)
    frames = [(role, data, parse(data)) for role, data in sent]
    predict = [(data, frame) for _, data, frame in frames if frame.type == "predict"]
    decode = [(data, frame) for _, data, frame in frames if frame.type == "decode"]
    assert len(predict) == 1 and len(decode) == 1
    assert pattern in predict[0][0] and float(SENTINEL) in predict[0][1].arrays["elevation"]  # predictor sees elevation
    decode = [frame for _, frame in decode]
    assert set(decode[0].arrays) == {"field_0", "obstacles"}
    assert set(decode[0].body) == {"policy", "start", "goal", "budget", "field_kind", "field_count"}
    for role, data in sent:
        assert b"split-sample-77" not in data  # the real sample id never crosses the pipe
        if role == "decoder":
            assert pattern not in data  # no elevation bytes ever reach the decoder


@pytest.mark.parametrize(
    "mode, expected",
    [
        ("two_fields", "exactly one field"),
        ("float64_field", "dtype float64"),
        ("wrong_shape", "shape"),
        ("nan_field", "non-finite"),
        ("int_field", "floating-point"),
        ("elevation_second_channel", "exactly one field"),
    ],
)
def test_field_contract_violation_is_a_runtime_error_for_that_request_and_the_run_continues(
    cand_dir, monkeypatch, mode, expected
):
    """H3: a predictor that routes several channels (or float64, or a wrong shape) to the decoder is refused."""
    sent = capture_wire(monkeypatch)
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder")
    with runner_for(cand_dir, kwargs={"mode": mode}, split=split) as runner:
        bad, _ = run_one(runner, make_request(0, sentinel=True))
        events = list(runner.events)
    assert bad["status"] == "runtime_error"
    if mode == "int_field":  # the worker already refuses non-float output when it packs the reply
        assert bad["diagnostics"]["error_type"] == "TypeError" and expected in bad["diagnostics"]["error"]
    else:
        assert bad["diagnostics"]["error_type"] == "FieldContractViolation" and expected in bad["diagnostics"]["error"]
        assert bad["diagnostics"]["contract"] == "decoder-field/v1"
    assert not [
        1 for role, data in sent if role == "decoder" and parse(data).type == "decode"
    ]  # nothing reached the decoder
    assert bad["wire_policies"] == [POLICY_ORDINARY_REQUEST]
    assert mode == "int_field" or bad["completion_mode"].endswith("contract_violation")
    assert [hop["role"] for hop in bad["hops"]] == ["predictor"]
    assert events == []  # no kill, no restart


def test_a_valid_field_after_a_violation_still_reaches_the_decoder(cand_dir):
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder")
    with runner_for(cand_dir, kwargs={"mode": "alternate"}, split=split) as runner:
        rows = [run_one(runner, make_request(0, f"s{i}"), i)[0] for i in range(4)]
    assert [row["status"] for row in rows] == ["runtime_error", "ok", "runtime_error", "ok"]
    assert rows[1]["diagnostics"]["n_fields"] == 1 and rows[1]["diagnostics"]["field_dtypes"] == ["float32"]


def test_split_decoder_timeout_restarts_only_the_decoder(cand_dir):
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder", decoder_kwargs={"mode": "sleep"})
    with runner_for(cand_dir, split=split) as runner:
        slow, _ = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=400.0, max_expansions=10)))
        generations = {role: w.generation for role, w in runner.workers.items()}
        assert not runner.workers["decoder"].alive and runner.workers["predictor"].alive
    assert slow["status"] == "timeout" and slow["preempted"] is True
    assert slow["diagnostics"]["preempted_role"] == "decoder" and 400.0 < slow["total_wall_ms"] < 2500.0
    assert generations == {"predictor": 0, "decoder": 0}  # restart happens before the next request


def test_split_decoder_error_is_a_terminal_runtime_error(cand_dir):
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder", decoder_kwargs={"mode": "raise"})
    with runner_for(cand_dir, split=split) as runner:
        record, _ = run_one(runner, make_request(0))
        again, _ = run_one(runner, make_request(0), 1)
    assert record["status"] == again["status"] == "runtime_error"
    assert record["diagnostics"]["error_type"] == "ValueError" and record["diagnostics"]["error"] == "decoder failed"
    assert runner.events == []  # a plain exception does not kill the worker


# ---------------------------------------------------------------------------
# end to end: supervised run -> public artifacts -> scorer-side staging -> cmd_score
# ---------------------------------------------------------------------------


def write_dataset(root, release_row, **overrides):
    """Validation split (the evaluated roster) and calibration split (disjoint warm-up inputs)."""
    from passage.utils import decode_tensor_blob, encode_tensor_blob

    split = root / "data" / "3x3" / "validation"
    split.mkdir(parents=True, exist_ok=True)
    row = json.loads(json.dumps({k: v for k, v in release_row.items() if k != "tensor"}))
    row["tensor"] = release_row["tensor"]
    row.update(overrides)
    pq.write_table(pa.Table.from_pylist([row]), split / "data-00000.parquet")
    warm = root / "data" / "3x3" / "calibration"
    warm.mkdir(parents=True, exist_ok=True)
    tensor = np.array(decode_tensor_blob(release_row["tensor"], (3, 3, 3)), copy=True)
    tensor[:, :, 0] = 0.25  # a different elevation: disjoint by input identity
    warm_row = dict(
        row,
        id="0000000099",
        tensor=encode_tensor_blob(tensor),
        metadata=dict(row["metadata"], split="calibration", index=99),
    )
    pq.write_table(pa.Table.from_pylist([warm_row]), warm / "data-00000.parquet")
    return root / "data"


def write_spec(root, release_row, cand_dir, **changes):
    data = write_dataset(root, release_row)
    spec = BenchSpec(
        "supervised-e2e",
        "classical",
        "fixture",
        inference="sup_candidates:Behaviour",
        inference_format="request",
        inference_kwargs={"force_code": 50},
        resolution=3,
        n_samples=1,
        seeds=[1234],
        dataset_path=str(data),
        latency_repeats=2,
        latency_warmup=1,
        results_root=str(root / "results"),
        budget={"max_expansions": 10, "wall_time_ms": 5000},
    )
    spec = replace(spec, **changes)
    path = root / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return spec, path


def cli_run(spec_path, output, cand_dir, isolation, *extra):
    return supervision_main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--output",
            str(output),
            "--worker-python",
            sys.executable,
            "--isolation",
            isolation,
            "--candidate-path",
            str(cand_dir),
            "--ipc-baseline-pings",
            "5",
            "--warmup-split",
            "calibration",
            *extra,
        ]
    )


@pytest.fixture(scope="module")
def e2e(tmp_path_factory, cand_dir):
    # build the release row once for the module (the conftest fixture is function scoped)
    from passage.utils import encode_tensor_blob

    root = tmp_path_factory.mktemp("supervision-e2e")
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0], tensor[:, :, 1] = 0.75, -1
    tensor[0, 0, 1], tensor[2, 2, 1], tensor[0, 2, 2] = 0, 1, 1
    cost = float(60 * np.sqrt(2))
    row = {
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
    spec, spec_path = write_spec(root, row, cand_dir)
    run = root / "run-process"
    assert cli_run(spec_path, run, cand_dir, "process") == 0
    return dict(root=root, row=row, spec=spec, spec_path=spec_path, run=run)


def all_bytes(directory):
    return b"\n".join(
        path.read_bytes()
        for path in sorted(Path(directory).rglob("*"))
        if path.is_file() and path.suffix in {".json", ".jsonl"}
    )


def test_public_run_directory_carries_no_reference_or_dataset_location(e2e):
    run, root = e2e["run"], e2e["root"]
    blob = all_bytes(run).decode()
    for forbidden in (
        str(root / "data"),
        "optimal_path",
        "optimal_cost",
        "oracle",
        "stored_label",
        "stored_path",
        "path_astar",
        "diagnostic_reference",
        "SECRET",
    ):
        assert forbidden not in blob, forbidden
    assert not list(Path(run).rglob("score_*")) and not list(Path(run).rglob("qualification.json"))
    for manifest_path in Path(run).rglob("run_manifest.json"):
        manifest = json.loads(manifest_path.read_text())
        assert "reference_qualification" not in manifest and manifest["reference_qualification_withheld"] is True
        assert manifest["bench_spec"]["dataset_path"] is None and manifest["dataset"]["path"] is None
        assert manifest["dataset"]["path_redacted"] is True and len(manifest["reference_identity"]) == 64


def test_run_artifacts_are_standard_and_record_the_supervision_contract(e2e):
    run = e2e["run"] / "seed-1234"
    # v3: no recorded warm-up passes; the warm-up is untimed, unrecorded and drawn from disjoint inputs
    assert {p.name for p in run.iterdir()} == {"cold", "paths.npz", "requests.jsonl", "run_manifest.json"}
    for directory, repeats, series in ((run, 2, "warm"), (run / "cold", 1, "cold")):
        artifact = read_run_artifact(directory)  # hashes + structural validation
        manifest = artifact.manifest
        assert manifest["series"] == series and manifest["repeats"] == repeats
        assert manifest["timing_boundary"] == TIMING_BOUNDARY_ID
        assert manifest["budget_enforcement"] == "preemptive_wall_deadline_cooperative_work_counters"
        assert manifest["cancellation_policy"] == "kill_process_group_and_restart_worker"
        supervision = manifest["supervision"]
        assert supervision["schema"] == "passage-supervision-v1" and supervision["isolation"]["level"] == "process"
        assert supervision["redaction"]["policies"] == [POLICY_ORDINARY_REQUEST]
        assert (
            supervision["redaction"]["sample_id"] == "opaque_per_run_token"
            and supervision["redaction"]["metadata_allow_list"] == []
        )
        assert supervision["ipc_baseline_subtracted"] is False and "worker_restart" not in json.dumps(
            supervision["not_in_boundary"]
        )
        baseline = supervision["ipc_echo_baseline_ms"]["planner"]
        assert {row["mode"] for row in baseline} == {"round_trip_same_size", "request_size_out_small_back"}
        assert all(row["n"] == 5 and row["median_ms"] > 0 for row in baseline)
        assert supervision["observed_request_payload_bytes"]["planner"] == [baseline[0]["payload_bytes"]]
        assert supervision["environment"]["scrubbed"] is True and "HF_TOKEN" not in json.dumps(
            supervision["environment"]
        )
        procedure = manifest["procedure"]["supervision"]
        assert procedure["timing_boundary"] == TIMING_BOUNDARY_ID and procedure["isolation_level"] == "process"
        assert (
            procedure["purpose"] == "diagnostic"
            and procedure["pass_policy"] == "fresh_worker_and_tokens_per_timed_pass"
        )
        assert manifest["candidate_identity"]["identity_method"] == identity.IDENTITY_METHOD
        assert all(
            record["timing_boundary"] == TIMING_BOUNDARY_ID and record["status"] == "ok" for record in artifact.records
        )
        # one fresh worker per timed pass, recorded outside the boundary
        passes = supervision["passes"]
        assert [row["pass_id"] for row in passes] == [f"{series}/{i}" for i in range(repeats)]
        starts = [e for e in supervision["worker_events"] if e["event"] == "pass_start"]
        assert len(starts) == repeats and all(e["start_ms"] > 0 for e in starts)
        assert {record["pass_id"] for record in artifact.records} == {f"{series}/{i}" for i in range(repeats)}
        assert all(
            record["pass_worker_start_ms"] > 0 and record["counters_source"] == "candidate_self_reported"
            for record in artifact.records
        )
    warm = read_run_artifact(run).manifest["supervision"]
    policy = warm["timing_policy"]
    assert policy["warmup"]["policy"] == "disjoint_inputs" and policy["warmup"]["source"] == "split:calibration"
    assert policy["warmup"]["disjoint_from_roster"] is True and len(policy["warmup"]["input_set_sha256"]) == 64
    assert [row["warmup"]["requests"] for row in warm["passes"]] == [1, 1]  # latency_warmup=1 per warm pass
    cold = read_run_artifact(run / "cold").manifest["supervision"]
    assert [row["warmup"] for row in cold["passes"]] == [None]  # the cold pass is never warmed up
    assert warm["purpose"]["name"] == "diagnostic" and warm["purpose"]["verified_candidate_run"] is False
    assert warm["isolation"]["candidate_code_runs_unsandboxed"] is True
    assert warm["interpreter_probe"]["executed_in"] == "unsandboxed_process_level_same_as_worker"
    assert warm["counters"]["counters_source"] == "candidate_self_reported"
    assert (
        "diagnostics" in warm["counters"]["self_reported_fields"] and "work" in warm["counters"]["self_reported_fields"]
    )
    assert warm["roster"] is None and warm["forbidden_modules"]["names"] == ["onnxruntime"]
    assert warm["restart_cap"]["cap_reached"] is False if "restart_cap" in warm else True


def test_unstaged_artifacts_cannot_be_scored_and_staged_ones_score_unchanged(e2e, tmp_path):
    run = e2e["run"]
    before = sorted(str(p) for p in Path(run).rglob("*"))
    with pytest.raises(ValueError, match="dataset_revision"):
        cmd_score(run)  # no dataset location, no references in the run
    assert sorted(str(p) for p in Path(run).rglob("*")) == before
    staged = tmp_path / "private" / "staged"
    directories = stage_for_scoring(run, e2e["spec"], staged)
    assert {d.name for d in directories} == {"seed-1234", "cold"}
    assert (staged / "seed-1234" / "public_run_manifest.json").read_bytes() == (
        run / "seed-1234" / "run_manifest.json"
    ).read_bytes()
    scored = cmd_score(staged)
    assert len(scored) == 2
    for directory, repeats in ((staged / "seed-1234", 2), (staged / "seed-1234" / "cold", 1)):
        receipt = json.loads((directory / "score_receipt.json").read_text())
        assert receipt["qualified_reference"] is True and receipt["reference_policy"] == "qualified_exact"
        assert receipt["scored_repeats"] == repeats
        assert (directory / "score_reference_inputs.json").exists()
        with np.load(directory / "per_sample.npz", allow_pickle=False) as arrays:
            assert arrays["regret"].tolist() == pytest.approx([0] * repeats)
            assert set(arrays["status"].tolist()) == {"ok"}
            assert np.all(arrays["latency_ms"] > 0)
        staged_manifest = json.loads((directory / "run_manifest.json").read_text())
        assert staged_manifest["scoring_stage"]["schema"] == "passage-scoring-stage-v1"
        assert staged_manifest["dataset"]["path_redacted"] is False
    assert not list(Path(run).rglob("score_*"))  # the public run stays free of scorer outputs


def test_staging_refuses_inconsistent_inputs(e2e, tmp_path):
    run, spec = e2e["run"], e2e["spec"]
    with pytest.raises(FileExistsError):
        stage_for_scoring(run, spec, e2e["root"])
    with pytest.raises(StagingError, match="contain one another"):
        stage_for_scoring(run, spec, run / "inside")
    other = write_dataset(tmp_path / "other", e2e["row"], id="0000000043")
    with pytest.raises(StagingError, match="dataset content differs"):
        stage_for_scoring(run, replace(spec, dataset_path=str(other)), tmp_path / "s1")
    with pytest.raises(StagingError, match="bench_spec"):
        stage_for_scoring(run, replace(spec, method_name="renamed"), tmp_path / "s2")
    scored = tmp_path / "s3"
    stage_for_scoring(run, spec, scored)
    with pytest.raises(StagingError, match="supervised public manifest"):
        stage_for_scoring(scored, spec, tmp_path / "s4")  # an already-staged copy is not a public run
    with pytest.raises(StagingError, match="no run artifacts"):
        stage_for_scoring(e2e["root"] / "data", spec, tmp_path / "s5")


@pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")
def test_end_to_end_under_bwrap_scores_after_staging(e2e, cand_dir, tmp_path):
    run = tmp_path / "run-bwrap"
    assert cli_run(e2e["spec_path"], run, cand_dir, "bwrap") == 0
    manifest = json.loads((run / "seed-1234" / "run_manifest.json").read_text())
    isolation = manifest["supervision"]["isolation"]
    assert isolation["level"] == "bwrap" and isolation["network"] == "unshared_loopback_only"
    assert str(e2e["root"] / "data") not in " ".join(isolation["read_only_binds"])
    assert all(str(e2e["root"] / "data") not in line for line in all_bytes(run).decode().splitlines())
    staged = tmp_path / "private-staged"
    stage_for_scoring(run, e2e["spec"], staged)
    cmd_score(staged)
    with np.load(staged / "seed-1234" / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["regret"].tolist() == pytest.approx([0, 0])


@pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")
def test_a_candidate_path_that_would_expose_the_output_directory_is_refused(e2e, cand_dir, tmp_path):
    output = tmp_path / "published" / "run"
    spec, spec_path = write_spec(tmp_path, e2e["row"], cand_dir)
    (tmp_path / "sup_candidates.py").write_text(CANDIDATES)
    # a candidate path that contains both the dataset and the output directory
    with pytest.raises(sandbox.SandboxError, match="overlaps protected path"):
        cli_run(spec_path, output, tmp_path, "bwrap")
    assert not output.exists()


def test_timeouts_and_restarts_are_scored_as_failures_not_dropped(e2e, cand_dir, tmp_path):
    spec, spec_path = write_spec(
        tmp_path,
        e2e["row"],
        cand_dir,
        inference_kwargs={"force_code": 20},
        latency_repeats=1,
        latency_warmup=0,
        budget={"max_expansions": 10, "wall_time_ms": 150},
    )
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand_dir, "process", "--deadline-grace-ms", "20") == 0
    artifact = read_run_artifact(run / "seed-1234")
    (record,) = artifact.records
    assert record["status"] == "timeout" and record["preempted"] is True and record["late_completion"] is True
    # v3: the warm pass starts on a fresh worker (recorded, outside the boundary), so nothing was restarted mid-pass
    assert record["pass_worker_start_ms"] > 0 and record["worker_restart_ms_before_request"] == 0.0
    assert record["total_wall_ms"] < record["pass_worker_start_ms"] + 1000
    cold = read_run_artifact(run / "seed-1234" / "cold")
    assert cold.records[0]["status"] == "timeout" and cold.records[0]["worker_restart_ms_before_request"] == 0.0
    assert [e["event"] for e in cold.manifest["supervision"]["worker_events"]] == ["pass_start", "preempt_kill"]
    assert [e["event"] for e in artifact.manifest["supervision"]["worker_events"]] == ["pass_start", "preempt_kill"]
    staged = tmp_path / "private" / "staged"
    stage_for_scoring(run, spec, staged)
    cmd_score(staged)
    with np.load(staged / "seed-1234" / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["status"].tolist() == ["timeout"] and not arrays["success"].any()
        assert arrays["regret"][0] > 0  # censored penalty, not a dropped row
    metrics = json.loads((staged / "seed-1234" / "metrics.json").read_text())
    assert metrics["terminal_ok_rate"] == 0 and metrics["requested_count"] == 1


def test_candidate_that_edits_its_own_source_invalidates_the_run(e2e, tmp_path):
    cand = tmp_path / "cand"
    cand.mkdir()
    source = cand / "sup_candidates.py"
    source.write_text(CANDIDATES)
    spec, spec_path = write_spec(
        tmp_path,
        e2e["row"],
        cand,
        inference_kwargs={"force_code": 32, "source_file": str(source)},
        latency_repeats=1,
        latency_warmup=0,
    )
    with pytest.raises(SupervisionError, match="identity changed"):
        run_supervised(
            spec,
            tmp_path / "run",
            SupervisorConfig(
                worker_python=sys.executable,
                isolation=sandbox.IsolationConfig(),
                candidate_paths=(str(cand),),
                ipc_baseline_pings=2,
                startup_timeout_s=120.0,
            ),
        )
    assert not (tmp_path / "run" / "seed-1234" / "run_manifest.json").exists()


def test_module_entry_point_requires_an_explicit_isolation_and_runs_under_dash_i(e2e, cand_dir, tmp_path):
    missing = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "passage_eval.supervision",
            "run",
            "--spec",
            str(e2e["spec_path"]),
            "--output",
            str(tmp_path / "x"),
            "--worker-python",
            sys.executable,
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert missing.returncode == 2 and "--isolation" in missing.stderr
    run = tmp_path / "run"
    done = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "passage_eval.supervision",
            "run",
            "--spec",
            str(e2e["spec_path"]),
            "--output",
            str(run),
            "--worker-python",
            sys.executable,
            "--isolation",
            "process",
            "--candidate-path",
            str(cand_dir),
            "--ipc-baseline-pings",
            "3",
            "--warmup-split",
            "calibration",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith("seed-1234") and (run / "seed-1234" / "run_manifest.json").exists()
    staged = subprocess.run(
        [
            sys.executable,
            "-I",
            "-m",
            "passage_eval.supervision",
            "stage-score",
            "--run",
            str(run),
            "--spec",
            str(e2e["spec_path"]),
            "--output",
            str(tmp_path / "private"),
            "--score",
        ],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert staged.returncode == 0, staged.stderr
    assert (tmp_path / "private" / "seed-1234" / "score_receipt.json").exists()


def test_supervisor_process_never_imports_candidate_code(e2e, cand_dir, tmp_path):
    script = (
        "import sys\n"
        "import passage_eval.cli as cli\n"
        "def forbidden(*a, **k):\n"
        "    raise AssertionError('supervisor imported candidate code')\n"
        "cli._import_entry = cli._materialise_planner = cli._candidate_identity = forbidden\n"
        "from passage_eval.supervision.__main__ import main\n"
        "rc = main(['run', '--spec', sys.argv[1], '--output', sys.argv[2], '--worker-python', sys.executable,\n"
        "           '--isolation', 'process', '--candidate-path', sys.argv[3], '--ipc-baseline-pings', '2',\n"
        "           '--warmup-split', 'calibration'])\n"
        "assert rc == 0\n"
        "leaked = [m for m in sys.modules if m.split('.')[0] in {'sup_candidates', 'torch'}]\n"
        "assert not leaked, leaked\n"
        "print('supervisor-clean')\n"
    )
    done = subprocess.run(
        [sys.executable, "-I", "-c", script, str(e2e["spec_path"]), str(tmp_path / "run"), str(cand_dir)],
        capture_output=True,
        text=True,
        cwd=tmp_path,
    )
    assert done.returncode == 0 and "supervisor-clean" in done.stdout, done.stderr


def test_several_seeds_use_their_own_kwargs_and_directories(e2e, cand_dir, tmp_path):
    spec, spec_path = write_spec(
        tmp_path,
        e2e["row"],
        cand_dir,
        seeds=[1, 2],
        latency_repeats=1,
        latency_warmup=0,
        inference_kwargs_by_seed={"2": {"force_code": 0}},
    )
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand_dir, "process") == 0
    assert sorted(p.name for p in run.iterdir()) == ["seed-1", "seed-2"]
    manifests = {seed: json.loads((run / f"seed-{seed}" / "run_manifest.json").read_text()) for seed in (1, 2)}
    assert [manifests[seed]["seed"] for seed in (1, 2)] == [1, 2]
    assert [manifests[seed]["bench_spec"]["inference_kwargs"]["force_code"] for seed in (1, 2)] == [50, 0]
    assert manifests[1]["procedure_id"] != manifests[2]["procedure_id"]
    staged = tmp_path / "private"
    stage_for_scoring(run, spec, staged)
    assert len(cmd_score(staged)) == 4  # (cold + warm) x 2 seeds


def test_existing_output_directory_is_never_overwritten(e2e, cand_dir):
    with pytest.raises(FileExistsError):
        cli_run(e2e["spec_path"], e2e["run"], cand_dir, "process")


# ---------------------------------------------------------------------------
# timing profiles: supervised_service_v1 (boundary v4) and supervised_service_rewarm_v1 (boundary v5)
# ---------------------------------------------------------------------------

SERVICE = "supervised_service_v1"
REWARM = "supervised_service_rewarm_v1"


@pytest.fixture
def slow_recost(monkeypatch):
    from passage_eval.supervision import supervisor

    original = supervisor.validate_plan

    def slow(*args, **kwargs):
        result = original(*args, **kwargs)
        time.sleep(0.25)
        return result

    monkeypatch.setattr(supervisor, "validate_plan", slow)


@pytest.mark.parametrize("profile", [SERVICE, REWARM])
def test_supervised_service_profile_excludes_evaluator_recost_from_deadline(cand_dir, slow_recost, profile):
    with runner_for(cand_dir, timing_profile=profile) as runner:
        record, _ = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=200.0, max_expansions=10)))
    assert record["service_wall_ms"] < 200.0 < record["total_wall_ms"]
    assert record["status"] == "ok"
    assert not record["late_completion"]


def test_legacy_profile_still_charges_evaluator_recost(cand_dir, slow_recost):
    with runner_for(cand_dir) as runner:
        record, _ = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=200.0, max_expansions=10)))
    assert record["service_wall_ms"] < 200.0 < record["total_wall_ms"]
    assert record["status"] == "timeout" and record["late_completion"]
    assert "rewarmed_before_request" not in record  # legacy records are unchanged


def test_supervised_service_record_scored_with_service_not_recost(cand_dir, slow_recost, metre_scene):
    from passage_eval.execution import score_run_artifact
    from passage_eval.protocol.contracts import PlanResult

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=200.0, max_expansions=10))
    with runner_for(cand_dir, timing_profile=SERVICE) as runner:
        record, path = run_one(runner, request)

    class Fixture:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    artifact = run_requests(Fixture(), [request])
    artifact.records = [record]
    artifact.paths = [path]
    artifact.manifest["schema"] = "passage-complete-request-supervised-service-v2"
    artifact.manifest["timing_boundary"] = "passage-timing-boundary/v4-supervised-ipc-service"
    artifact.manifest["timing_profile"] = SERVICE
    artifact.manifest["procedure"] = {"supervision": {"timing_profile": SERVICE}}
    _, arrays = score_run_artifact(artifact, [metre_scene], require_qualified_reference=False)
    assert record["service_wall_ms"] < 200.0 < record["total_wall_ms"]
    assert arrays["success"].tolist() == [True]
    assert arrays["latency_ms"].tolist() == [record["service_wall_ms"]]
    assert artifact.records[0]["total_wall_ms"] > arrays["latency_ms"][0]


def test_service_profile_cli_persist_stage_and_score_synthetic_validation(e2e, cand_dir, tmp_path):
    run = tmp_path / "run-service"
    assert cli_run(e2e["spec_path"], run, cand_dir, "process", "--timing-profile", SERVICE) == 0
    public = read_run_artifact(run / "seed-1234")
    assert public.manifest["schema"] == "passage-complete-request-supervised-service-v2"
    assert public.manifest["procedure"]["supervision"]["timing_profile"] == SERVICE
    assert public.manifest["timing_profile"] == SERVICE
    assert "validation_and_physical_recost" in public.manifest["supervision"]["not_in_boundary"]
    staged = tmp_path / "staged-service"
    stage_for_scoring(run, e2e["spec"], staged)
    cmd_score(staged)
    record = read_run_artifact(staged / "seed-1234").records[0]
    with np.load(staged / "seed-1234" / "per_sample.npz", allow_pickle=False) as arrays:
        assert arrays["latency_ms"][0] == record["service_wall_ms"]


def test_legacy_manifest_timing_policy_retains_existing_fields(cand_dir):
    from passage_eval.supervision.supervisor import _timing_policy

    spec = bench_spec("sup_candidates:Behaviour")
    config = SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig(level="process"))
    legacy = _timing_policy(spec, {}, config)
    assert "latency_profile" not in legacy and "restart_rewarm" not in legacy
    service = _timing_policy(spec, {}, replace(config, timing_profile=SERVICE))
    assert service["latency_profile"] == SERVICE and "restart_rewarm" not in service
    rewarm = _timing_policy(spec, {}, replace(config, timing_profile=REWARM))
    assert rewarm["boundary"] == "passage-timing-boundary/v5-supervised-ipc-service-rewarm"
    assert rewarm["restart_rewarm"]["status"] == "proposed_not_sealed"
    with pytest.raises(ValueError, match="unknown timing profile"):
        replace(config, timing_profile="supervised_service_v9")


def test_service_profile_late_reply_crash_and_unexecuted_are_failures(cand_dir):
    with runner_for(cand_dir, timing_profile=SERVICE, deadline_grace_ms=1000.0, max_restarts=0) as runner:
        late, _ = run_one(runner, make_request(21, budget=BudgetSpec(wall_time_ms=100.0, max_expansions=10)))
        crash, _ = run_one(
            runner, make_request(7, "crash", budget=BudgetSpec(wall_time_ms=5000.0, max_expansions=10)), 1
        )
        skipped, _ = run_one(
            runner, make_request(0, "unexecuted", budget=BudgetSpec(wall_time_ms=5000.0, max_expansions=10)), 2
        )
    assert late["status"] == "timeout" and late["late_completion"] and not late["preempted"]
    assert late["failed_attempt_ms"] == 100.0 and late["drain_ms"] == pytest.approx(late["service_wall_ms"] - 100.0)
    assert crash["status"] == "runtime_error" and crash["completion_mode"] == "worker_exited"
    assert skipped["status"] == "runtime_error" and skipped["latency_charged_not_measured"]
    assert skipped["service_wall_ms"] == skipped["total_wall_ms"] == skipped["failed_attempt_ms"] == 5000.0
    assert skipped["timing_boundary"] == "passage-timing-boundary/v4-supervised-ipc-service"


def test_service_profile_preempts_overdue_worker(cand_dir):
    with runner_for(cand_dir, timing_profile=SERVICE) as runner:
        record, raw = run_one(runner, make_request(20, budget=BudgetSpec(wall_time_ms=200.0, max_expansions=10)))
    assert record["status"] == "timeout" and record["preempted"] and record["late_completion"]
    assert record["service_wall_ms"] == record["total_wall_ms"]
    assert record["failed_attempt_ms"] == 200.0 and record["drain_ms"] == pytest.approx(
        record["service_wall_ms"] - 200.0
    )
    assert raw.shape == (0, 2)


def test_service_profile_skips_decoder_if_field_arrives_after_deadline(cand_dir, monkeypatch):
    split = SplitSpec("sup_candidates:Predictor", "sup_candidates:Decoder")
    with runner_for(cand_dir, split=split, timing_profile=SERVICE) as runner:
        original = runner._hop

        def delayed_return(role, *args, **kwargs):
            result = original(role, *args, **kwargs)
            if role == "predictor":
                time.sleep(0.25)
            return result

        monkeypatch.setattr(runner, "_hop", delayed_return)
        record, raw = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=200.0, max_expansions=10)))
    assert record["status"] == "timeout" and record["late_completion"]
    assert record["diagnostics"]["decoder_skipped_after_deadline"] is True and len(record["hops"]) == 1
    assert record["failed_attempt_ms"] == 200.0 and record["drain_ms"] == pytest.approx(
        record["service_wall_ms"] - 200.0
    )
    assert raw.shape == (0, 2)


@pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")
def test_split_service_predictor_preemption_decoder_crash_and_contract_rejection_under_bwrap(tmp_path):
    candidate = tmp_path / "cand"
    candidate.mkdir()
    (candidate / "sup_candidates.py").write_text(
        CANDIDATES + "\nclass SlowPredictor(Predictor):\n    def predict(self, request):\n        time.sleep(30)\n"
        "        return super().predict(request)\n\nclass CrashingDecoder(Decoder):\n"
        "    def decode(self, field, start, goal, obstacles, budget):\n        os._exit(9)\n"
    )
    runtime = str(Path(__file__).resolve().parents[2] / "src")
    cases = [
        ("sup_candidates:SlowPredictor", "sup_candidates:Decoder", {}, "timeout"),
        ("sup_candidates:Predictor", "sup_candidates:CrashingDecoder", {}, "runtime_error"),
        ("sup_candidates:Predictor", "sup_candidates:Decoder", {"mode": "nan_field"}, "runtime_error"),
    ]
    records = []
    for predictor, decoder, kwargs, expected in cases:
        split = SplitSpec(predictor, decoder)
        with runner_for(
            candidate,
            isolation="bwrap",
            runtime_binds=(runtime,),
            kwargs=kwargs,
            split=split,
            timing_profile=SERVICE,
        ) as runner:
            budget = 300.0 if expected == "timeout" else 5000.0
            record, raw = run_one(runner, make_request(0, budget=BudgetSpec(wall_time_ms=budget, max_expansions=10)))
        assert record["status"] == expected and raw.shape == (0, 2)
        assert record["timing_boundary"] == "passage-timing-boundary/v4-supervised-ipc-service"
        records.append(record)
    assert records[0]["preempted"] and records[0]["diagnostics"]["preempted_role"] == "predictor"
    assert records[1]["completion_mode"] == "worker_exited" and records[1]["hops"][-1]["role"] == "decoder"
    assert records[2]["diagnostics"]["error_type"] == "FieldContractViolation"


# --- re-warm after a mid-pass restart (proposed policy of supervised_service_rewarm_v1) ----------------------


def measured(runner, request, index, pass_id="warm/0"):
    record, _ = runner.run_request(
        request, request_token(runner.secret, index, scope=pass_id), 0, "warm", pass_id=pass_id
    )
    return record


def start_warm_pass(runner, warm_requests, pass_id="warm/0"):
    runner.begin_pass(pass_id)
    runner.set_warm_plan(warm_requests, len(warm_requests), pass_id)
    return runner.run_warmup(warm_requests, len(warm_requests), pass_id)


def test_rewarm_profile_replays_the_warmup_after_a_mid_pass_timeout_outside_the_boundary(cand_dir):
    warm = [make_request(21, f"w{i}") for i in range(2)]  # 0.15 s each: visibly outside the boundary
    timeout = BudgetSpec(wall_time_ms=200.0, max_expansions=10)
    with runner_for(cand_dir, timing_profile=REWARM) as runner:
        initial = start_warm_pass(runner, warm)
        first = measured(runner, make_request(20, "s0", budget=timeout), 0)
        second = measured(runner, make_request(0, "s1"), 1)
        third = measured(runner, make_request(0, "s2"), 2)
        events = [event["event"] for event in runner.events]
        rewarms = list(runner.pass_rewarms)
        restarts = runner.restarts
    assert initial["complete"] and initial["succeeded"] == 2 and initial["workers_alive_after"]
    assert first["status"] == "timeout" and first["preempted"]
    assert (first["rewarmed_before_request"], first["warmup_complete_before_request"]) == (False, True)
    assert second["status"] == "ok" and second["worker_restart_ms_before_request"] > 0
    assert (second["rewarmed_before_request"], second["warmup_complete_before_request"]) == (True, True)
    assert (third["rewarmed_before_request"], third["warmup_complete_before_request"]) == (False, True)
    assert restarts == 1 and len(rewarms) == 1
    assert rewarms[0]["complete"] and rewarms[0]["requests"] == 2 and rewarms[0]["statuses"] == {"ok": 2}
    # the re-warm (two 0.15 s requests) is not inside the measured request's boundary
    assert rewarms[0]["wall_ms_total"] >= 300.0 > second["service_wall_ms"]
    assert second["timing_boundary"] == "passage-timing-boundary/v5-supervised-ipc-service-rewarm"
    assert events[events.index("preempt_kill") :] == ["preempt_kill", "restart", "rewarm"]


def test_rewarm_profile_initial_warmup_runs_every_request_like_the_other_profiles(cand_dir):
    warm = [make_request(7, "w0"), make_request(0, "w1")]  # the first warm-up request kills the worker
    with runner_for(cand_dir, timing_profile=REWARM, max_restarts=20) as runner:
        initial = start_warm_pass(runner, warm)
        rows = [measured(runner, make_request(0, f"s{i}"), i) for i in range(3)]
        events = [event["event"] for event in runner.events]
        rewarms = list(runner.pass_rewarms)
        restarts, outside_cap = runner.restarts, runner.rewarm_restarts
    # the initial warm-up keeps going, as in the other profiles; its failure is recorded, not merely counted
    assert initial["complete"] is False and initial["requests"] == 2 and initial["succeeded"] == 1
    assert initial["statuses"] == {"runtime_error": 1, "ok": 1} and initial["workers_alive_after"] is True
    assert all(row["status"] == "ok" for row in rows)
    # the worker restarted inside the warm-up served the rest of it: no re-warm, but no complete warm-up either
    assert all(row["rewarmed_before_request"] is False for row in rows)
    assert all(row["warmup_complete_before_request"] is False for row in rows)
    assert rewarms == [] and (restarts, outside_cap) == (1, 0)
    assert events.count("warmup_failed") == 1 and "rewarm" not in events


def _roster_outcomes(cand_dir, profile, warm_codes, roster_codes, max_restarts, kwargs=None):
    changes = {} if profile is None else {"timing_profile": profile}
    with runner_for(cand_dir, kwargs=kwargs, max_restarts=max_restarts, **changes) as runner:
        start_warm_pass(runner, [make_request(code, f"w{i}") for i, code in enumerate(warm_codes)])
        rows = [measured(runner, make_request(code, f"s{i}"), i) for i, code in enumerate(roster_codes)]
        return {
            "statuses": [row["status"] for row in rows],
            "reasons": [row["diagnostics"].get("reason") for row in rows],
            "counted_restarts": runner.restarts,
            "rewarm_restarts": runner.rewarm_restarts,
        }


# (warm-up codes, roster codes, max_restarts): 7 kills the worker, 0 succeeds, 34 kills it once 35 has run.
ROSTER_EQUIVALENCE = {
    "warmup-input-kills-the-worker": ([7], [0, 0], 1),
    "warmup-input-kills-then-succeeds": ([7, 0], [0, 0, 0], 1),
    "rewarm-fails-on-the-restarted-worker": ([34], [35, 0, 0], 1),
    "roster-tasks-reach-the-cap": ([0], [7, 7, 0], 1),
}


@pytest.mark.parametrize("scenario", list(ROSTER_EQUIVALENCE))
def test_rewarm_profile_roster_outcomes_and_counted_restarts_match_v3_and_v4(cand_dir, tmp_path, scenario):
    """A warm-up input outside the roster must not turn a roster task into a restart-cap failure under v5."""
    warm_codes, roster_codes, cap = ROSTER_EQUIVALENCE[scenario]
    outcomes = {}
    for profile in (None, SERVICE, REWARM):
        kwargs = {"marker": str(tmp_path / f"marker-{profile}")}
        outcomes[profile] = _roster_outcomes(cand_dir, profile, warm_codes, roster_codes, cap, kwargs)
    legacy, service, rewarm = outcomes[None], outcomes[SERVICE], outcomes[REWARM]
    for key in ("statuses", "reasons", "counted_restarts"):
        assert rewarm[key] == service[key] == legacy[key], (key, outcomes)
    assert legacy["rewarm_restarts"] == service["rewarm_restarts"] == 0
    if scenario == "rewarm-fails-on-the-restarted-worker":
        assert rewarm["rewarm_restarts"] == 1  # the worker lost in the re-warm, restarted outside the cap


def test_a_failed_rewarm_restarts_its_worker_outside_the_cap_and_ends_rewarming_for_the_pass(cand_dir, tmp_path):
    warm = [make_request(34, "w0")]  # succeeds on the first worker, kills every worker started after the marker
    roster = [make_request(code, f"s{i}") for i, code in enumerate([35, 0, 0, 7, 0])]
    with runner_for(cand_dir, timing_profile=REWARM, max_restarts=2, kwargs={"marker": str(tmp_path / "m")}) as runner:
        initial = start_warm_pass(runner, warm)
        rows = [measured(runner, request, i) for i, request in enumerate(roster)]
        events = [event["event"] for event in runner.events]
        rewarms = list(runner.pass_rewarms)
        restarts, outside_cap = runner.restarts, runner.rewarm_restarts
    assert initial["complete"] is True
    assert [row["status"] for row in rows] == ["runtime_error", "ok", "ok", "runtime_error", "ok"]
    assert [row["rewarmed_before_request"] for row in rows] == [False, True, False, False, False]
    assert [row["warmup_complete_before_request"] for row in rows] == [True, False, False, False, False]
    # s1: counted restart, failed re-warm, restart outside the cap; s4: counted restart, no further re-warm
    assert (restarts, outside_cap) == (2, 1)
    assert len(rewarms) == 1 and rewarms[0]["complete"] is False and rewarms[0]["requests"] == 1
    assert events.count("rewarm") == 1 and events.count("rewarm_failed") == 1
    assert events.count("rewarm_restart") == 1 and events.count("restart") == 2
    assert rows[1]["worker_restart_ms_before_request"] > 0 and rows[1]["service_wall_ms"] < 1000.0


@pytest.mark.parametrize("profile", [None, SERVICE])
def test_legacy_and_v4_profiles_never_rewarm_and_keep_their_records(cand_dir, profile):
    warm = [make_request(0, f"w{i}") for i in range(2)]
    timeout = BudgetSpec(wall_time_ms=200.0, max_expansions=10)
    changes = {} if profile is None else {"timing_profile": profile}
    with runner_for(cand_dir, **changes) as runner:
        start_warm_pass(runner, warm)
        measured(runner, make_request(20, "s0", budget=timeout), 0)
        second = measured(runner, make_request(0, "s1"), 1)
        events = [event["event"] for event in runner.events]
    assert second["status"] == "ok" and second["worker_restart_ms_before_request"] > 0
    assert "rewarm" not in events and "rewarmed_before_request" not in second
    assert "warmup_complete_before_request" not in second


def test_legacy_warmup_keeps_running_through_failures_and_records_them(cand_dir):
    warm = [make_request(7, "w0"), make_request(0, "w1")]
    with runner_for(cand_dir) as runner:
        summary = start_warm_pass(runner, warm)
        events = [event["event"] for event in runner.events]
    assert (
        summary["requests"] == 2 and summary["succeeded"] == 1 and summary["statuses"] == {"runtime_error": 1, "ok": 1}
    )
    assert summary["complete"] is False and summary["workers_alive_after"] is True
    assert events[-4:] == ["worker_gone", "restart", "warmup", "warmup_failed"]


def test_rewarm_profile_cli_writes_v5_artifacts_that_stage_and_score(e2e, cand_dir, tmp_path):
    run = tmp_path / "run-rewarm"
    assert cli_run(e2e["spec_path"], run, cand_dir, "process", "--timing-profile", REWARM) == 0
    warm, cold = read_run_artifact(run / "seed-1234"), read_run_artifact(run / "seed-1234" / "cold")
    for artifact in (warm, cold):
        assert artifact.manifest["schema"] == "passage-complete-request-supervised-service-v3"
        assert artifact.manifest["timing_boundary"] == "passage-timing-boundary/v5-supervised-ipc-service-rewarm"
        assert artifact.manifest["timing_profile"] == REWARM
    assert [(r["rewarmed_before_request"], r["warmup_complete_before_request"]) for r in warm.records] == [
        (False, True)
    ] * 2
    assert [(r["rewarmed_before_request"], r["warmup_complete_before_request"]) for r in cold.records] == [
        (False, None)
    ]
    supervision = warm.manifest["supervision"]
    assert [p["rewarms"] for p in supervision["passes"]] == [[], []]
    assert all(p["warmup"]["complete"] for p in supervision["passes"])
    assert supervision["timing_policy"]["restart_rewarm"]["status"] == "proposed_not_sealed"
    assert "rewarm_requests_after_restart" in supervision["not_in_boundary"]
    staged = tmp_path / "staged-rewarm"
    stage_for_scoring(run, e2e["spec"], staged)
    cmd_score(staged)
    metrics = json.loads((staged / "seed-1234" / "metrics.json").read_text())
    block = metrics["latency_all_repeats_censored"]
    assert block["timing_profile"] == REWARM and block["n_requests"] == 2 and block["n_repeats"] == 2
    assert block["n_requests_without_complete_warmup"] == 0 and block["meets_min_warm_requests"] is False
    assert block["warmup"]["declared"] is True and block["warmup"]["performed"] is True
    assert block["warmup"]["passes"] == block["warmup"]["passes_with_complete_warmup"] == 2
    assert block["n_warm_requests"] == 2
    assert block["meets_min_warm_requests_reason"] == "fewer_warm_requests_than_the_minimum"
    assert supervision["restart_cap"]["rewarm_restarts_outside_cap"] == 0
    cold_block = json.loads((staged / "seed-1234" / "cold" / "metrics.json").read_text())[
        "latency_all_repeats_censored"
    ]
    assert cold_block["meets_min_warm_requests"] is False
    assert cold_block["meets_min_warm_requests_reason"] == "not_a_warm_series"


def test_rewarm_profile_without_measured_latency_names_its_series_unwarmed(e2e, cand_dir, tmp_path):
    """measure_latency=false: no warm-up ran, so the measured series is called 'unwarmed', never 'warm'."""
    spec, spec_path = write_spec(tmp_path, e2e["row"], cand_dir, measure_latency=False)
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand_dir, "process", "--timing-profile", REWARM) == 0
    warm = read_run_artifact(run / "seed-1234")
    assert warm.manifest["series"] == "unwarmed" and not (run / "seed-1234" / "cold").exists()
    assert {record["series"] for record in warm.records} == {"unwarmed"}
    assert [(r["rewarmed_before_request"], r["warmup_complete_before_request"]) for r in warm.records] == [
        (False, None)
    ]
    assert warm.manifest["supervision"]["timing_policy"]["warmup"]["policy"] == "none"
    staged = tmp_path / "private" / "staged"
    stage_for_scoring(run, spec, staged)
    cmd_score(staged)
    block = json.loads((staged / "seed-1234" / "metrics.json").read_text())["latency_all_repeats_censored"]
    assert block["series"] == "unwarmed" and block["n_requests_without_complete_warmup"] == 1
    assert block["warmup"]["declared"] is False and block["warmup"]["passes_with_warmup"] == 0
    assert block["meets_min_warm_requests"] is False and block["n_warm_requests"] == 0
    assert block["meets_min_warm_requests_reason"] == "not_a_warm_series"


def test_rewarm_profile_cli_records_timeouts_failed_rewarms_and_censoring(e2e, cand_dir, tmp_path):
    spec, spec_path = write_spec(
        tmp_path,
        e2e["row"],
        cand_dir,
        inference_kwargs={"force_code": 20},
        latency_repeats=1,
        latency_warmup=1,
        budget={"max_expansions": 10, "wall_time_ms": 150},
    )
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand_dir, "process", "--deadline-grace-ms", "20", "--timing-profile", REWARM) == 0
    warm = read_run_artifact(run / "seed-1234")
    (record,) = warm.records
    assert record["status"] == "timeout" and record["preempted"]
    assert record["rewarmed_before_request"] is True and record["warmup_complete_before_request"] is False
    (entry,) = warm.manifest["supervision"]["passes"]
    assert entry["warmup"]["complete"] is False and [r["complete"] for r in entry["rewarms"]] == [False]
    events = [e["event"] for e in warm.manifest["supervision"]["worker_events"]]
    assert events.count("warmup_failed") == 1 and events.count("rewarm_failed") == 1
    # the restart before the re-warm is counted; the worker the re-warm lost restarts outside the cap
    assert events.count("restart") == 1 and events.count("rewarm_restart") == 1
    cap = warm.manifest["supervision"]["restart_cap"]
    assert (cap["unplanned_restarts_used"], cap["rewarm_restarts_outside_cap"]) == (1, 1)
    staged = tmp_path / "private" / "staged"
    stage_for_scoring(run, spec, staged)
    cmd_score(staged)
    metrics = json.loads((staged / "seed-1234" / "metrics.json").read_text())
    block = metrics["latency_all_repeats_censored"]
    assert block["n_censored"] == 1
    assert block["quantiles"] is None  # an incomplete re-warm cannot enter the hot latency series
    assert block["n_requests_without_complete_warmup"] == 1 and block["n_hot_latency_requests"] == 0
    assert block["hot_latency_final"] is False
    assert metrics["latency_all_repeats_ms"]["p50"] == record["service_wall_ms"]  # legacy keys unchanged


# --- procedure identity: every bound on a request is hashed --------------------------------------------------


def test_procedure_limits_bind_budget_profile_boundary_and_supervision_limits():
    from passage_eval.supervision.supervisor import procedure_limits

    config = SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig(level="process"))
    budget = BudgetSpec(wall_time_ms=120000.0, max_expansions=65537)
    limits = procedure_limits(budget, config)
    assert limits["budget"] == {
        "wall_time_ms": 120000.0,
        "max_expansions": 65537,
        "max_queue_operations": None,
        "max_live_states": None,
    }
    assert (limits["timing_profile"], limits["timing_boundary"], limits["latency_field"]) == (
        "supervised_total_v3",
        TIMING_BOUNDARY_ID,
        "total_wall_ms",
    )
    assert {
        key: limits[key] for key in ("deadline_grace_ms", "hard_timeout_ms", "startup_timeout_s", "max_restarts")
    } == {
        "deadline_grace_ms": 10.0,
        "hard_timeout_ms": 600000.0,
        "startup_timeout_s": 600.0,
        "max_restarts": 100,
    }
    assert limits["schema"] == "passage-procedure-limits/v2"
    assert limits["worker_rlimits"] == {
        "core": 0,
        "nofile": 1024,
        "fsize_bytes": 1 << 30,
        "address_space_bytes": None,
        "cpu_seconds": None,
    }
    assert limits["ipc_frame_limits"]["max_payload_bytes"] == 256 << 20
    legacy = procedure_limits(budget, config, schema="passage-procedure-limits/v1")
    assert legacy == {
        key: value for key, value in limits.items() if key not in ("worker_rlimits", "ipc_frame_limits")
    } | {"schema": "passage-procedure-limits/v1"}
    with pytest.raises(ValueError, match="schema"):
        procedure_limits(budget, config, schema="passage-procedure-limits/v9")
    rlimited = replace(
        config,
        isolation=replace(config.isolation, rlimits=sandbox.RlimitConfig(cpu_seconds=60, address_space_bytes=1 << 32)),
    )
    variants = [
        procedure_limits(budget, rlimited),
        procedure_limits(budget, replace(config, limits=FrameLimits(max_payload_bytes=1 << 20))),
        procedure_limits(replace(budget, wall_time_ms=180000.0), config),
        procedure_limits(replace(budget, max_expansions=65536), config),
        procedure_limits(budget, replace(config, timing_profile=SERVICE)),
        procedure_limits(budget, replace(config, timing_profile=REWARM)),
        procedure_limits(budget, replace(config, deadline_grace_ms=20.0)),
        procedure_limits(budget, replace(config, hard_timeout_ms=300000.0)),
        procedure_limits(budget, replace(config, startup_timeout_s=60.0)),
        procedure_limits(budget, replace(config, max_restarts=3)),
    ]
    digests = {json.dumps(item, sort_keys=True) for item in [limits, *variants]}
    assert len(digests) == 1 + len(variants)


def test_procedure_id_changes_with_the_startup_timeout_and_the_restart_cap(e2e, cand_dir, tmp_path):
    base = read_run_artifact(e2e["run"] / "seed-1234").manifest
    limits = base["procedure"]["supervision"]["limits"]
    assert limits["budget"] == {
        "wall_time_ms": 5000.0,
        "max_expansions": 10,
        "max_queue_operations": None,
        "max_live_states": None,
    }
    assert limits["startup_timeout_s"] == 600.0 and limits["max_restarts"] == 100
    protocol = base["procedure"]["supervision"]["latency_protocol"]
    assert protocol["latency_warmup_requests_per_warm_pass"] == 1 and protocol["warmup_source"] == "split:calibration"
    assert len(protocol["warmup_input_set_sha256"]) == 64
    run = tmp_path / "run-limits"
    assert cli_run(e2e["spec_path"], run, cand_dir, "process", "--startup-timeout-s", "300", "--max-restarts", "7") == 0
    other = read_run_artifact(run / "seed-1234").manifest
    assert other["procedure"]["supervision"]["limits"]["startup_timeout_s"] == 300.0
    assert other["procedure"]["supervision"]["limits"]["max_restarts"] == 7
    assert other["procedure_id"] != base["procedure_id"]
