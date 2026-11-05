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

"""Synthetic tests; no target, lease, released split or model files are read."""

import hashlib

import pytest
from passage_eval.campaign.direct.bwrap_validation_relay import remote_command


def spec():
    return {
        "target_work_root": "/srv/passage/runs",
        "target_rootfs": "/srv/passage-images/rootfs",
        "target_readonly_mounts": ["/srv/passage/payload/model.onnx"],
        "runner_argv": [
            "/opt/venv/bin/python3",
            "-I",
            "-B",
            "/opt/passage-direct/direct_target.py",
            "--config",
            "{config}",
            "--config-sha256",
            hashlib.sha256(b"{}").hexdigest(),
            "--output",
            "{output}",
        ],
        "configuration_sha256": hashlib.sha256(b"{}").hexdigest(),
        "timeout_seconds": 1000,
    }


@pytest.fixture(autouse=True)
def portable_host_configuration(monkeypatch):
    for target in ("ORIN", "THOR", "TIGER"):
        monkeypatch.setenv("PASSAGE_EDGE_MACHINE_ID_" + target, "a" * 32)
        monkeypatch.setenv("PASSAGE_EDGE_SSH_HOST_" + target, "edge.example.invalid")


def test_one_scope_readonly_root_and_no_docker():
    command, scope, runner = remote_command(spec(), b"{}", "job:0123456789abcdef")
    assert scope == "passage-edge-tiger-0123456789abcdef.scope"
    assert "bwrap" in command and "--ro-bind" in command and "--unshare-net" in command
    assert "docker" not in command and command.count("systemd-run") == 1
    assert runner[-2:] == ["--output", "/srv/passage/runs/bwrap-session-0123456789abcdef/result"]


def test_bwrap_command_uses_operator_machine_identity(monkeypatch):
    monkeypatch.setenv("PASSAGE_EDGE_MACHINE_ID_TIGER", "b" * 32)
    command, _, _ = remote_command(spec(), b"{}", "job:0123456789abcdef")
    assert "b" * 32 in command
    monkeypatch.setenv("PASSAGE_EDGE_MACHINE_ID_TIGER", "invalid")
    with pytest.raises(ValueError, match="machine identity"):
        remote_command(spec(), b"{}", "job:0123456789abcdef")


def test_rejects_unbound_runner_and_invalid_job():
    p = spec()
    p["runner_argv"][-3] = "b" * 64
    with pytest.raises(ValueError):
        remote_command(p, b"{}", "job:0123456789abcdef")
    with pytest.raises(ValueError):
        remote_command(spec(), b"{}", "job:bad")


def test_lost_initial_lease_never_stages_or_launches(monkeypatch, tmp_path):
    from passage_eval.campaign.direct import bwrap_validation_relay as relay

    monkeypatch.setenv("AGENT_OPS_JOB", "job:0123456789abcdef")
    monkeypatch.setattr(relay, "load_spec", lambda *_: ({"output_root": str(tmp_path / "out")}, {}, {}))

    def lost(*_):
        raise RuntimeError("lease missing")

    monkeypatch.setattr(relay, "_lease", lost)
    monkeypatch.setattr(relay, "_stage_observers", lambda *_: pytest.fail("must not stage"))
    with pytest.raises(RuntimeError, match="lease missing"):
        relay._run_admitted("unused", "unused", object())
    assert not (tmp_path / "out").exists()


def test_cancel_fences_late_scope_before_drain(monkeypatch):
    from types import SimpleNamespace
    from passage_eval.campaign.direct import bwrap_validation_relay as relay

    commands = []
    monkeypatch.setattr(relay, "_ssh_argv", lambda _, command: [command])
    monkeypatch.setattr(
        relay.subprocess,
        "run",
        lambda argv, **kw: commands.append(argv[0]) or SimpleNamespace(returncode=0, stdout=b"FENCED"),
    )
    monkeypatch.setattr(relay, "_drain", lambda _, scope: commands.append("drain:" + scope))
    scope = "passage-edge-tiger-0123456789abcdef.scope"
    relay._cancel_and_drain({"target_work_root": "/srv/passage/runs"}, "job:0123456789abcdef", scope)
    assert "control.lock" in commands[0] and "CANCELLED" in commands[0]
    assert commands[0].index("CANCELLED") < commands[0].index("systemctl --user kill")
    assert commands[1] == "drain:" + scope


def test_preflight_checks_validation_projection_but_does_not_open_s(monkeypatch):
    from passage_eval.campaign.direct import bwrap_validation_relay as relay

    config = {"target": "tiger", "dataset_split": "validation", "index_sha256": "b" * 64, "warmup_index": []}
    spec = {key: None for key in relay._KEYS}
    spec.update(
        schema="passage-tiger-bwrap-relay/v1",
        target="tiger",
        configuration="config.json",
        configuration_sha256="a" * 64,
        prepared_root="validation",
        prepared_sha256="c" * 64,
    )
    prepared = {"split": "validation", "count": 128, "index_sha256": "b" * 64, "warmup_index": []}
    calls = []

    def read(path, digest):
        name = str(path)
        calls.append(name)
        return config if name.endswith("config.json") else prepared if name.endswith("prepared.json") else spec

    monkeypatch.setattr(relay.pinned, "read_pinned_json", read)
    monkeypatch.setattr(relay, "_validation_host_path", lambda path: path)
    monkeypatch.setattr(relay, "validate_config", lambda value: value)
    monkeypatch.setattr(relay, "_validation_provenance", lambda *_: calls.append("verified rosters"))
    relay.inspect_spec("spec.json", "d" * 64)
    assert "verified rosters" in calls
    prepared["count"] = 127
    with pytest.raises(ValueError, match="prepared validation"):
        relay.inspect_spec("spec.json", "d" * 64)
    config["dataset_split"] = "test"
    calls.clear()
    relay.inspect_spec("spec.json", "d" * 64)
    assert calls == ["spec.json", "config.json"]


def test_admitted_preflight_enforces_shared_g2_runner_and_rootfs(monkeypatch, tmp_path):
    import json
    from types import SimpleNamespace
    from passage_eval.campaign.direct import admitted_stream, bwrap_validation_relay as relay, runner_binding

    local = {}
    for name in (
        "known_hosts",
        "image_archive",
        "rootfs_export",
        "configuration",
        "bundle_manifest_host",
        "warmup_roster",
    ):
        path = tmp_path / name
        path.write_bytes(b"{}")
        local[name] = str(path)
    raw = b"{}"
    sha = hashlib.sha256(raw).hexdigest()
    config = {
        "target": "tiger",
        "procedure": "reg",
        "compiled_manifest": None,
        "grn_evidence": None,
        "artifact_paths": {k: "/srv/passage/payload/" + k for k in ("bundle", "checkpoint", "engine", "decoder")},
        "artifact_pins": {"bundle": sha},
    }
    spec = {key: None for key in relay._KEYS}
    spec.update(
        schema="passage-tiger-bwrap-relay/v1",
        target="tiger",
        known_hosts=local["known_hosts"],
        known_hosts_sha256=sha,
        image_archive=local["image_archive"],
        image_archive_sha256=sha,
        rootfs_export=local["rootfs_export"],
        rootfs_export_sha256=sha,
        configuration=local["configuration"],
        configuration_sha256=sha,
        warmup_roster=local["warmup_roster"],
        warmup_roster_sha256=sha,
        bundle_manifest_host=local["bundle_manifest_host"],
        prepared_root=str(tmp_path / "prepared"),
        validation_roster=str(tmp_path / "validation"),
        output_root=str(tmp_path / "out"),
        compiled_manifest_host=None,
        image_manifest_digest="sha256:" + "a" * 64,
        target_rootfs="/srv/passage-user/rootfs",
        target_rootfs_export="/srv/passage-user/root.tar",
        target_work_root="/srv/passage/runs",
        target_readonly_mounts=list(config["artifact_paths"].values()),
        runner_argv=["/opt/venv/bin/python3", "-I", "-B", "/opt/passage-direct/direct_target.py"],
        rootfs_exceptions=[],
        timeout_seconds=1000,
    )
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec))
    specsha = hashlib.sha256(spec_path.read_bytes()).hexdigest()
    session = SimpleNamespace(rule8_scope=object(), grant=object(), config_bytes=raw)
    monkeypatch.setattr(relay, "validate_config", lambda _: config)
    monkeypatch.setattr(admitted_stream, "bind_target_config", lambda *a, **kw: SimpleNamespace(config_bytes=raw))
    monkeypatch.setattr(relay, "_bundle_artifact_paths", lambda *a: [])
    comparisons = []

    def compare(*args, **kwargs):
        comparisons.append((args, kwargs))
        return {"rootfs_archive_sha256": sha, "exceptions": {}}

    monkeypatch.setattr(relay.bwrap_lineage, "compare", compare)
    calls = []
    monkeypatch.setattr(runner_binding, "verify_g2_runner", lambda *args, **kw: calls.append((args, kw)))
    relay.load_spec(spec_path, specsha, session)
    assert calls[0][0][1:3] == ("tiger", "reg") and calls[0][1]["rootfs_export_sha256"] == sha
    cache = {}
    relay.load_spec(spec_path, specsha, session, image_cache=cache)
    relay.load_spec(spec_path, specsha, session, image_cache=cache)
    assert len(comparisons) == 2  # one uncached call and one for both cached calls
    monkeypatch.setattr(
        runner_binding, "verify_g2_runner", lambda *a, **kw: (_ for _ in ()).throw(ValueError("G2 mismatch"))
    )
    with pytest.raises(ValueError, match="G2 mismatch"):
        relay.load_spec(spec_path, specsha, session)
    monkeypatch.setattr(runner_binding, "verify_g2_runner", lambda *args, **kw: None)
    (tmp_path / "rootfs_export").write_bytes(b"changed")
    with pytest.raises(ValueError, match="pin differs"):
        relay.load_spec(spec_path, specsha, session, image_cache=cache)


def test_validation_finish_requires_observed_exit_and_matching_session_markers():
    import json
    from passage_eval.campaign.direct.bwrap_validation_relay import validation_finish

    sid = "f" * 32
    journal = (
        json.dumps({"kind": "session_start", "namespace_pid": 2, "session_id": sid})
        + "\n"
        + json.dumps({"kind": "session_complete", "namespace_pid": 2, "session_id": sid})
        + "\n"
    ).encode()
    scope = "passage-edge-tiger-0123456789abcdef.scope"
    job = "job:0123456789abcdef"
    begin = {
        "pid": 42,
        "namespace_pid": 2,
        "container_digest": "sha256:" + "a" * 64,
        "scope": scope,
        "job_id": job,
        "configuration_sha256": "b" * 64,
        "input_index_sha256": "c" * 64,
    }
    receipt = {"journal_sha256": hashlib.sha256(journal).hexdigest()}
    spec = {"configuration_sha256": "b" * 64}
    config = {"index_sha256": "c" * 64}
    result = validation_finish(begin, receipt, spec, config, scope, job, journal, 0)
    assert result["exit_code"] == 0 and result["runner_session_uuid"] == sid and result["journal_markers"] == "matched"
    with pytest.raises(ValueError):
        validation_finish(begin, receipt, spec, config, scope, job, journal, 1)
    with pytest.raises(ValueError):
        validation_finish({**begin, "namespace_pid": 3}, receipt, spec, config, scope, job, journal, 0)


def test_session_tmp_uses_new_retained_scratch():
    command, _, _ = remote_command(spec(), b"{}", "job:0123456789abcdef")
    scratch = "/srv/passage/runs/bwrap-session-0123456789abcdef/scratch"
    assert "--bind " + scratch + " /tmp" in command
    assert "--tmpfs /tmp " not in command
    assert 'mkdir -m 700 -- "$d/scratch"' in command
    assert command.index('mkdir -m 700 -- "$d/scratch"') < command.index("systemd-run")


def test_numba_cache_is_set_before_clean_environment_entrypoint():
    command, _, _ = remote_command(spec(), b"{}", "job:0123456789abcdef")
    assert "--setenv NUMBA_CACHE_DIR /tmp/numba" in command
    assert 'mkdir -m 700 -- "$d/scratch/numba"' in command
    assert command.index('mkdir -m 700 -- "$d/scratch/numba"') < command.index("systemd-run")
