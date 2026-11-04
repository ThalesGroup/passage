"""Synthetic process observations, no live process or accelerator access."""

import pytest
from passage_eval.campaign.direct.bwrap_scope import observe


def scope_world(tmp_path):
    job = "job:0123456789abcdef"
    scope = "passage-edge-tiger-0123456789abcdef.scope"
    cg = tmp_path / "cg"
    base = cg / "user.slice" / scope
    base.mkdir(parents=True)
    (base / "cgroup.procs").write_text("42\n")
    proc = tmp_path / "proc"
    p = proc / "42"
    p.mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (p / "root").symlink_to(root)
    argv = ["/opt/venv/bin/python3", "-I", "-B", "/opt/passage-direct/direct_target.py"]
    (p / "status").write_text("NSpid:\t42\t2\n")
    (p / "cmdline").write_bytes(b"\0".join(x.encode() for x in argv) + b"\0")
    (p / "mountinfo").write_text("3 2 8:1 / / ro,relatime - ext4 /dev/a ro\n")

    def query(unit):
        return {"ActiveState": "active", "ControlGroup": "/user.slice/" + scope}

    return job, root, proc, cg, argv, query


def test_observes_exact_scoped_pid_namespace_and_readonly_root(tmp_path):
    job, root, proc, cg, argv, query = scope_world(tmp_path)
    result = observe("tiger", job, root, argv, proc_root=proc, cgroup_root=cg, query=query, timeout_s=0)
    assert result["runner"]["pid"] == 42 and result["runner"]["namespace_pid"] == 2


def test_refuses_changed_runner_or_writable_root(tmp_path):
    job, root, proc, cg, argv, query = scope_world(tmp_path)
    (proc / "42" / "mountinfo").write_text("3 2 8:1 / / rw,relatime - ext4 /dev/a rw\n")
    with pytest.raises(ValueError):
        observe("tiger", job, root, argv, proc_root=proc, cgroup_root=cg, query=query, timeout_s=0)
    with pytest.raises(ValueError):
        observe("tiger", job, root, argv + ["unexpected"], proc_root=proc, cgroup_root=cg, query=query, timeout_s=0)
