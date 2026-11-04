"""Observation on leased Tiger, before streaming input. Does not launch or stop jobs."""

import re
import subprocess
import time
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")


def _observe_once(target, job, rootfs, expected_argv, *, proc_root=Path("/proc"), cgroup_root=CGROUP_ROOT, query=None):
    if target != "tiger" or not re.fullmatch(r"job:[0-9a-f]{16}", job):
        raise ValueError("target/job")
    if (
        not isinstance(expected_argv, list)
        or not expected_argv
        or not all(isinstance(x, str) and x for x in expected_argv)
    ):
        raise ValueError("runner argument vector differs")
    scope = f"passage-edge-{target}-{job[4:]}.scope"
    if query is None:

        def query(unit):
            p = subprocess.run(
                ["systemctl", "--user", "show", unit, "--property=ActiveState", "--property=ControlGroup"],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)

    state = query(scope)
    if state.get("ActiveState") != "active":
        raise ValueError("scope inactive")
    group = state.get("ControlGroup", "")
    if not group.startswith("/") or ".." in Path(group).parts or not group.endswith("/" + scope):
        raise ValueError("scope cgroup")
    base = (Path(cgroup_root) / group.lstrip("/")).resolve()
    if not base.is_relative_to(Path(cgroup_root).resolve()) or not base.is_dir():
        raise ValueError("cgroup path")
    pids = set()
    for procfile in [base / "cgroup.procs", *base.rglob("cgroup.procs")]:
        pids.update(int(x) for x in procfile.read_text().split())
    root = Path(rootfs)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("rootfs path")
    source_stat = root.stat()
    matches = []
    diagnostics = []
    for pid in pids:
        try:
            p = proc_root / str(pid)
            argv = (p / "cmdline").read_bytes().rstrip(b"\0").split(b"\0")
            if argv != [arg.encode() for arg in expected_argv]:
                diagnostics.append((pid, "argv differs", [x.decode("utf8", "replace") for x in argv[:12]]))
                continue
            root_stat = (p / "root").stat()
            if (source_stat.st_dev, source_stat.st_ino) != (root_stat.st_dev, root_stat.st_ino):
                diagnostics.append(
                    (
                        pid,
                        "root inode differs",
                        (source_stat.st_dev, source_stat.st_ino),
                        (root_stat.st_dev, root_stat.st_ino),
                    )
                )
                continue
            lines = (p / "mountinfo").read_text().splitlines()
            roots = [
                line.split(" - ", 1)[0].split()
                for line in lines
                if " - " in line
                and len(line.split(" - ", 1)[0].split()) > 5
                and line.split(" - ", 1)[0].split()[4] == "/"
            ]
            if len(roots) != 1 or "ro" not in roots[0][5].split(","):
                diagnostics.append((pid, "root mount differs", roots))
                continue
            status = (p / "status").read_text()
            entries = [line.split()[1:] for line in status.splitlines() if line.startswith("NSpid:")]
            if len(entries) != 1 or not entries[0] or int(entries[0][0]) != pid:
                raise ValueError("namespace PID unavailable")
            namespace_pid = int(entries[0][-1])
            if namespace_pid <= 0:
                raise ValueError("namespace PID differs")
            matches.append(
                {
                    "pid": pid,
                    "namespace_pid": namespace_pid,
                    "argv": [x.decode("utf8", "strict") for x in argv],
                    "root_device": source_stat.st_dev,
                    "root_inode": source_stat.st_ino,
                    "root_mount_options": roots[0][5],
                }
            )
        except (OSError, UnicodeError, ValueError) as exc:
            diagnostics.append((pid, type(exc).__name__, str(exc)))
            continue
    if len(matches) != 1:
        raise ValueError(f"no unique scoped runner with matching read-only root; diagnostics={diagnostics}")
    return {
        "schema": "passage-bwrap-scope-observation/v1",
        "target": target,
        "job_id": job,
        "scope": scope,
        "cgroup": group,
        "runner": matches[0],
        "rootfs": str(root),
        "limitations": [
            "root observation is a point-in-time check",
            "runtime overrides and host device binds are separate from image contents",
            "does not authenticate OCI archive, attest GPU use or prove publication",
        ],
    }


def observe(
    target, job, rootfs, expected_argv, *, proc_root=Path("/proc"), cgroup_root=CGROUP_ROOT, query=None, timeout_s=45
):
    if not 0 <= timeout_s <= 60:
        raise ValueError("observation timeout differs")
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            return _observe_once(
                target, job, rootfs, expected_argv, proc_root=proc_root, cgroup_root=cgroup_root, query=query
            )
        except ValueError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.25)
