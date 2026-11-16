"""Conservative, locked GPU allocation accounting. Crashes retain reservations."""

from __future__ import annotations
from contextlib import contextmanager
import ctypes
import fcntl
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
import time
import uuid

from passage_eval.candidates.catalog import atomic_json
from passage_eval.protocol.candidate_package import read_json


class GpuBudget:
    def __init__(self, path: str | Path, limit_seconds: float = 28 * 3600, *, clock=time.monotonic):
        self.path = Path(path)
        self.clock = clock
        if not math.isfinite(limit_seconds) or limit_seconds <= 0:
            raise ValueError("invalid GPU limit")
        self.limit_seconds = float(limit_seconds)
        with self._locked() as data:
            if data["limit_gpu_seconds"] != self.limit_seconds:
                raise ValueError("existing GPU limit cannot be reset")

    @contextmanager
    def _locked(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            data = (
                read_json(self.path)
                if self.path.exists()
                else {"schema": "passage-gpu-budget/v1", "limit_gpu_seconds": self.limit_seconds, "allocations": []}
            )
            yield data
            atomic_json(self.path, data)

    @staticmethod
    def _charged(data):
        return sum(
            a["reserved_gpu_seconds"] if a["state"] == "active" else a["gpu_seconds"] for a in data["allocations"]
        )

    def remaining_seconds(self) -> float:
        with self._locked() as data:
            return max(0.0, data["limit_gpu_seconds"] - self._charged(data))

    @contextmanager
    def allocate(self, label: str, *, max_seconds: float, gpus: list[str], require_drain: bool = False):
        if not label or not gpus or len(set(gpus)) != len(gpus) or not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError("invalid GPU allocation")
        record = {
            "id": uuid.uuid4().hex,
            "label": label,
            "gpus": gpus,
            "pid": os.getpid(),
            "state": "active",
            "reserved_gpu_seconds": float(max_seconds * len(gpus)),
            "started_unix": time.time(),
            "process_drain_required": require_drain,
        }
        with self._locked() as data:
            if self._charged(data) + record["reserved_gpu_seconds"] > data["limit_gpu_seconds"]:
                raise ValueError("GPU budget has insufficient unreserved time")
            if any(set(a["gpus"]) & set(gpus) for a in data["allocations"] if a["state"] == "active"):
                raise ValueError("GPU already allocated by this campaign")
            data["allocations"].append(record)
        begin = self.clock()
        try:
            yield record
        finally:
            elapsed = max(0.0, self.clock() - begin) * len(gpus)
            with self._locked() as data:
                actual = next(a for a in data["allocations"] if a["id"] == record["id"])
                confirmed = record.get("process_drain_confirmed") is True
                if require_drain and not confirmed:
                    actual.update(
                        process_drain_confirmed=False,
                        process_drain=record.get("process_drain"),
                        observed_gpu_seconds=elapsed,
                        last_checked_unix=time.time(),
                    )
                    # Keep the whole reservation and GPU lock active when cleanup is uncertain.
                else:
                    actual.update(
                        state="complete",
                        gpu_seconds=elapsed,
                        finished_unix=time.time(),
                        overrun=elapsed > record["reserved_gpu_seconds"],
                    )
                    if require_drain:
                        actual.update(process_drain_confirmed=True, process_drain=record.get("process_drain"))


@contextmanager
def _private_subreaper():
    """CLI-local Linux adoption of orphan descendants; never a host-wide setting."""
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(previous), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot inspect child subreaper state")
    if libc.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "cannot enable private child subreaper")
    try:
        yield
    finally:
        if libc.prctl(36, previous.value, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "cannot restore child subreaper state")


def _process_table():
    table = {}
    for path in Path("/proc").iterdir():
        if not path.name.isdigit():
            continue
        try:
            fields = (path / "stat").read_text().rpartition(")")[2].split()
            table[int(path.name)] = {
                "state": fields[0],
                "parent": int(fields[1]),
                "group": int(fields[2]),
                "start": int(fields[19]),
            }
        except (FileNotFoundError, ProcessLookupError):
            continue
        except PermissionError:
            continue
    if os.getpid() not in table:
        raise RuntimeError("cannot inspect supervising process in procfs")
    return table


class _ProcessDrain:
    """Track the command tree and subreaper-adopted children, including setsid."""

    def __init__(self):
        self.parent = os.getpid()
        self.baseline = {pid: row["start"] for pid, row in _process_table().items() if row["parent"] == self.parent}
        self.known = {}
        self.errors = []

    def _refresh(self, proc):
        table = _process_table()
        roots = {self.parent}
        changed = True
        while changed:
            changed = False
            for pid, row in table.items():
                if row["parent"] in roots and self.baseline.get(pid) != row["start"] and pid not in roots:
                    roots.add(pid)
                    changed = True
        for pid in roots - {self.parent}:
            self.known[pid] = table[pid]["start"]
        for pid, started in list(self.known.items()):
            if pid not in table or table[pid]["start"] != started:
                self.known.pop(pid)
                continue
            if pid != proc.pid:
                try:
                    os.waitpid(pid, os.WNOHANG)
                except ChildProcessError:
                    pass  # still has its original parent; adoption happens on parent exit
                except ProcessLookupError:
                    pass
        proc.poll()  # reap the leader through Popen so its return code remains accurate
        table = _process_table()
        self.known = {
            pid: started for pid, started in self.known.items() if pid in table and table[pid]["start"] == started
        }
        direct = Path(f"/proc/self/task/{self.parent}/children").read_text().split()
        for raw in direct:
            pid = int(raw)
            if pid in table and self.baseline.get(pid) != table[pid]["start"]:
                self.known[pid] = table[pid]["start"]
            elif pid not in table and Path(f"/proc/{pid}").exists():
                raise RuntimeError(f"cannot inspect adopted child {pid}")
        return table

    def _signal(self, proc, sig):
        table = self._refresh(proc)
        for pid, started in list(self.known.items()):
            row = table.get(pid)
            if row and row["start"] == started and row["state"] != "Z":
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except OSError as error:
                    self.errors.append(f"{type(error).__name__}: {error}")

    def drain(self, proc, deadline):
        begin = time.monotonic()
        try:
            self._signal(proc, signal.SIGTERM)
            grace = min(deadline, time.monotonic() + 2)
            while time.monotonic() < deadline:
                self._refresh(proc)
                if not self.known and proc.poll() is not None:
                    return {
                        "confirmed": True,
                        "elapsed_seconds": time.monotonic() - begin,
                        "remaining_pids": [],
                        "errors": self.errors,
                    }
                if time.monotonic() >= grace:
                    self._signal(proc, signal.SIGKILL)
                time.sleep(min(0.02, max(0, deadline - time.monotonic())))
            self._refresh(proc)
        except Exception as error:
            self.errors.append(f"{type(error).__name__}: {error}")
        return {
            "confirmed": False,
            "elapsed_seconds": time.monotonic() - begin,
            "remaining_pids": sorted(self.known),
            "errors": self.errors,
        }


class _BudgetSignal(BaseException):
    def __init__(self, signum):
        self.signum = signum


def main(argv=None):
    """Reserve before launch; drain every descendant before releasing accounting."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command or not math.isfinite(args.seconds) or args.seconds <= 15:
        parser.error("command and >15 seconds including shutdown reserve required")
    budget = GpuBudget(args.ledger)
    old_handlers = {}
    cleaning = False
    received_signal = None

    def handle_signal(signum, frame):
        nonlocal received_signal
        received_signal = signum
        if not cleaning:
            raise _BudgetSignal(signum)

    with _private_subreaper():
        supervisor = _ProcessDrain()
        try:
            for sig in (signal.SIGTERM, signal.SIGINT):
                old_handlers[sig] = signal.signal(sig, handle_signal)
            with budget.allocate(
                args.label, max_seconds=args.seconds, gpus=[args.gpu], require_drain=True
            ) as allocation:
                deadline = time.monotonic() + args.seconds
                proc = None
                returncode = 1
                try:
                    proc = subprocess.Popen(command, start_new_session=True)
                    try:
                        returncode = proc.wait(timeout=max(0.001, deadline - time.monotonic() - 10))
                    except subprocess.TimeoutExpired:
                        returncode = 124
                except _BudgetSignal as error:
                    returncode = 128 + error.signum
                finally:
                    cleaning = True
                    drain = (
                        supervisor.drain(proc, deadline)
                        if proc is not None
                        else {"confirmed": True, "remaining_pids": [], "errors": [], "reason": "command_not_started"}
                    )
                    allocation.update(process_drain_confirmed=drain["confirmed"], process_drain=drain)
                if not drain["confirmed"]:
                    print("Process drain unconfirmed; GPU reservation remains active", file=sys.stderr)
                    if received_signal is None:
                        returncode = 125
            return 128 + received_signal if received_signal is not None else returncode
        finally:
            for sig, handler in old_handlers.items():
                signal.signal(sig, handler)


if __name__ == "__main__":
    raise SystemExit(main())
