import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import pytest


def test_budget_reserves_before_launch_and_releases_unused_time(tmp_path):
    from passage_eval.candidates.gpu_budget import GpuBudget

    ticks = iter([100.0, 110.0])
    budget = GpuBudget(tmp_path / "budget.json", limit_seconds=28 * 3600, clock=lambda: next(ticks))
    with budget.allocate("pilot", max_seconds=3600, gpus=["gpu-test"]):
        active = json.loads((tmp_path / "budget.json").read_text())
        assert active["allocations"][0]["reserved_gpu_seconds"] == 3600
    assert budget.remaining_seconds() == 28 * 3600 - 10


def test_concurrent_reservations_cannot_exceed_global_cap(tmp_path):
    from passage_eval.candidates.gpu_budget import GpuBudget

    budget = GpuBudget(tmp_path / "budget.json", limit_seconds=20)
    with budget.allocate("one", max_seconds=15, gpus=["a"]):
        with pytest.raises(ValueError, match="budget"):
            with budget.allocate("two", max_seconds=10, gpus=["b"]):
                pass


def test_exception_is_charged_and_limit_cannot_be_reset(tmp_path):
    from passage_eval.candidates.gpu_budget import GpuBudget

    ticks = iter([10.0, 13.0])
    budget = GpuBudget(tmp_path / "budget.json", limit_seconds=20, clock=lambda: next(ticks))
    with pytest.raises(RuntimeError):
        with budget.allocate("failure", max_seconds=15, gpus=["a"]):
            raise RuntimeError("candidate failed")
    assert budget.remaining_seconds() == 17
    with pytest.raises(ValueError, match="limit"):
        GpuBudget(tmp_path / "budget.json", limit_seconds=30)


def _live(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rpartition(")")[2].split()[0] != "Z"
    except FileNotFoundError:
        return False


def _command(tmp_path, *, wait=False, detached=False):
    child_pid = tmp_path / "child.pid"
    child = (
        "import os,signal,time;from pathlib import Path;"
        "signal.signal(signal.SIGTERM,signal.SIG_IGN);"
        f"Path({str(child_pid)!r}).write_text(str(os.getpid()));time.sleep(60)"
    )
    source = tmp_path / "leader.py"
    source.write_text(
        "import subprocess,sys,time,signal\nfrom pathlib import Path\n"
        "signal.signal(signal.SIGTERM,lambda *args:sys.exit(0))\n"
        f'subprocess.Popen([sys.executable,"-c",{child!r}],start_new_session={detached!r},'
        "stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)\n"
        f"while not Path({str(child_pid)!r}).exists():time.sleep(.005)\n" + ("time.sleep(60)\n" if wait else "")
    )
    return child_pid, [
        sys.executable,
        "-m",
        "passage_eval.candidates.gpu_budget",
        "--ledger",
        str(tmp_path / "ledger.json"),
        "--seconds",
        "15.2",
        "--gpu",
        "cpu-fixture",
        "--label",
        "cpu-process-drain",
        "--",
        sys.executable,
        str(source),
    ]


@pytest.mark.parametrize("detached", [False, True])
def test_normal_leader_exit_drains_remaining_child_before_budget_release(tmp_path, detached):
    pid_file, command = _command(tmp_path, detached=detached)
    child_pid = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=14)
        child_pid = int(pid_file.read_text())
        assert not _live(child_pid), "leader exited but child remained alive"
        assert result.returncode == 0, result.stderr
        allocation = json.loads((tmp_path / "ledger.json").read_text())["allocations"][0]
        assert allocation["state"] == "complete"
        assert allocation["process_drain_confirmed"] is True
        assert allocation["gpu_seconds"] <= allocation["reserved_gpu_seconds"]
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None and _live(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_timeout_drains_child_even_when_leader_exits_on_sigterm(tmp_path):
    pid_file, command = _command(tmp_path, wait=True)
    child_pid = None
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=16)
        child_pid = int(pid_file.read_text())
        assert not _live(child_pid), "SIGTERM made leader exit but left its child alive"
        assert result.returncode == 124, result.stderr
        allocation = json.loads((tmp_path / "ledger.json").read_text())["allocations"][0]
        assert allocation["state"] == "complete" and allocation["process_drain_confirmed"]
    finally:
        if child_pid is None and pid_file.exists():
            child_pid = int(pid_file.read_text())
        if child_pid is not None and _live(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_unknown_process_drain_keeps_full_reservation_active(tmp_path):
    from passage_eval.candidates.gpu_budget import GpuBudget

    budget = GpuBudget(tmp_path / "budget.json", limit_seconds=20)
    with budget.allocate("uncertain", max_seconds=15, gpus=["cpu-fixture"], require_drain=True):
        pass
    allocation = json.loads((tmp_path / "budget.json").read_text())["allocations"][0]
    assert allocation["state"] == "active"
    assert allocation["process_drain_confirmed"] is False
    assert budget.remaining_seconds() == 5


@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT])
def test_cli_signal_drains_detached_child_and_preserves_signal_exit(tmp_path, signum):
    pid_file, command = _command(tmp_path, wait=True, detached=True)
    child_pid = None
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.monotonic() + 5
        while not pid_file.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert pid_file.exists()
        child_pid = int(pid_file.read_text())
        proc.send_signal(signum)
        _, stderr = proc.communicate(timeout=10)
        assert proc.returncode == 128 + signum, stderr
        assert not _live(child_pid)
        allocation = json.loads((tmp_path / "ledger.json").read_text())["allocations"][0]
        assert allocation["state"] == "complete" and allocation["process_drain_confirmed"]
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        if child_pid is not None and _live(child_pid):
            os.kill(child_pid, signal.SIGKILL)


def test_cli_unknown_drain_keeps_reservation_and_returns_failure(tmp_path, monkeypatch):
    from passage_eval.candidates import gpu_budget

    monkeypatch.setattr(
        gpu_budget._ProcessDrain,
        "drain",
        lambda *args: {"confirmed": False, "remaining_pids": [], "errors": ["injected procfs failure"]},
    )
    ledger = tmp_path / "ledger.json"
    result = gpu_budget.main(
        [
            "--ledger",
            str(ledger),
            "--seconds",
            "15.2",
            "--gpu",
            "cpu-fixture",
            "--label",
            "uncertain",
            "--",
            sys.executable,
            "-c",
            "pass",
        ]
    )
    assert result == 125
    allocation = json.loads(ledger.read_text())["allocations"][0]
    assert allocation["state"] == "active"
    assert allocation["process_drain_confirmed"] is False
    assert "gpu_seconds" not in allocation
