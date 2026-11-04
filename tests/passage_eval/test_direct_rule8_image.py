"""Synthetic host observations only; no daemon, device, data or cleanup."""

import hashlib
import json
import pytest
from passage_eval.campaign.direct.rule8_image import verify_finished_container, verify_running_container


def test_manifest_and_config_are_distinct_and_runner_is_the_container_init():
    config = "sha256:" + "a" * 64
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": config}}).encode()
    digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    argv = ["/usr/bin/python3", "-m", "passage_eval.campaign.direct.direct_target"]
    inspected = {
        "Id": "b" * 64,
        "Image": config,
        "Path": argv[0],
        "Args": argv[1:],
        "RestartCount": 0,
        "State": {"Running": True, "Pid": 42, "StartedAt": "2026-11-04T00:00:00Z"},
    }
    kwargs = {
        "scope": "job.scope",
        "process_in_scope": lambda pid, scope: (pid, scope) == (42, "job.scope"),
        "launch_reference": "registry/image@" + digest,
        "expected_runner_argv": argv,
    }
    start = verify_running_container(manifest, inspected, digest, **kwargs)
    stopped = {**inspected, "State": {"Running": False, "ExitCode": 0, "StartedAt": "2026-11-04T00:00:00Z"}}
    proof = verify_finished_container(start, stopped, exit_scope_drained=True)
    assert proof["container_digest"] == digest and proof["runner_pid"] == 42
    assert (proof["n_processes"], proof["n_restarts"]) == (1, 0)
    with pytest.raises(ValueError, match="immutable"):
        verify_running_container(manifest, inspected, digest, **{**kwargs, "launch_reference": "registry/image:latest"})
    with pytest.raises(ValueError, match="image differs"):
        verify_running_container(manifest, {**inspected, "Image": "sha256:" + "c" * 64}, digest, **kwargs)
    with pytest.raises(ValueError, match="expected direct runner"):
        verify_running_container(manifest, {**inspected, "Path": "/bin/sh"}, digest, **kwargs)
    with pytest.raises(ValueError, match="outside"):
        verify_running_container(manifest, inspected, digest, **{**kwargs, "process_in_scope": lambda *_: False})
    with pytest.raises(ValueError, match="completion"):
        verify_finished_container(start, {**stopped, "RestartCount": 1}, exit_scope_drained=True)
    with pytest.raises(ValueError, match="completion"):
        verify_finished_container(start, stopped, exit_scope_drained=False)


def test_python_command_cannot_smuggle_module_name_as_an_argument():
    from passage_eval.campaign.direct.rule8_image import _is_direct_python_command

    assert _is_direct_python_command(
        ["/usr/bin/python3", "-I", "-B", "-m", "passage_eval.campaign.direct.direct_target"]
    )
    assert _is_direct_python_command(["/usr/bin/python3", "-I", "-B", "/opt/passage-direct/direct_target.py", "--help"])
    assert not _is_direct_python_command(
        ["/usr/bin/python3", "-c", "pass", "passage_eval.campaign.direct.direct_target"]
    )
    assert not _is_direct_python_command(["/usr/bin/python3", "/tmp/other.py", "/opt/passage-direct/direct_target.py"])
