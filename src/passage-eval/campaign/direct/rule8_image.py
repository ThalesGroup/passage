"""Prototype host-only OCI identity checks; never treat synthetic observations as evidence.

The dispatcher must obtain these inputs from its own daemon inspection under its
lease, not from the candidate. Rootfs/bwrap execution is intentionally unsupported.
"""

import hashlib
import json
import re
from pathlib import PurePosixPath

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
_CONTAINER_ID = re.compile(r"[0-9a-f]{64}\Z")


def _is_direct_python_command(argv):
    """Accept only a Python module or script at Python's actual execution position."""
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(arg, str) and arg for arg in argv)
        or not re.fullmatch(r"python(?:[0-9]+(?:\.[0-9]+)?)?", PurePosixPath(argv[0]).name)
    ):
        return False
    index = 1
    while index < len(argv) and argv[index] in {"-I", "-B", "-E", "-s", "-S", "-u"}:
        index += 1
    if index >= len(argv):
        return False
    if argv[index] == "-m":
        return index + 1 < len(argv) and argv[index + 1] == "passage_eval.campaign.direct.direct_target"
    return argv[index].startswith("/") and PurePosixPath(argv[index]).name == "direct_target.py"


def verify_running_container(
    manifest_bytes, inspected, expected_digest, *, scope, process_in_scope, launch_reference, expected_runner_argv
):
    """Bind OCI manifest to the daemon-observed container, PID and executing scope.

    The trusted dispatcher performs daemon inspect and process scope lookup itself.
    The function does not prove publication, rebuild, GPU use or bwrap rootfs lineage.
    """
    if not isinstance(expected_digest, str) or not _DIGEST.fullmatch(expected_digest):
        raise ValueError("expected OCI manifest digest differs")
    if "sha256:" + hashlib.sha256(manifest_bytes).hexdigest() != expected_digest:
        raise ValueError("OCI manifest digest differs")
    try:
        manifest = json.loads(manifest_bytes)
        config = manifest["config"]["digest"]
        state = inspected["State"]
        pid = state["Pid"]
        container_id = inspected["Id"]
        actual_image = inspected["Image"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("OCI manifest or inspected container differs") from exc
    if manifest.get("schemaVersion") != 2 or not isinstance(config, str) or not _DIGEST.fullmatch(config):
        raise ValueError("OCI manifest config differs")
    if not isinstance(launch_reference, str) or not (
        launch_reference == expected_digest or launch_reference.endswith("@" + expected_digest)
    ):
        raise ValueError("host launch did not use the pinned immutable manifest reference")
    # The daemon may report the manifest or its config digest. The trusted host
    # must also have launched by the immutable manifest reference above.
    if actual_image not in (expected_digest, config):
        raise ValueError("container image differs from the pinned manifest and config")
    if (
        not _is_direct_python_command(expected_runner_argv)
        or inspected.get("Path") != expected_runner_argv[0]
        or inspected.get("Args") != expected_runner_argv[1:]
    ):
        raise ValueError("container init is not the expected direct runner")
    if type(inspected.get("RestartCount")) is not int or inspected["RestartCount"] != 0:
        raise ValueError("runner container has restarted")
    started_at = state.get("StartedAt")
    if not isinstance(started_at, str) or not started_at:
        raise ValueError("runner start identity is missing")
    if not isinstance(container_id, str) or not _CONTAINER_ID.fullmatch(container_id):
        raise ValueError("container identifier differs")
    if state.get("Running") is not True or type(pid) is not int or pid <= 0:
        raise ValueError("container process is not running")
    if not isinstance(scope, str) or not scope or process_in_scope(pid, scope) is not True:
        raise ValueError("container process is outside the leased session scope")
    return {
        "container_digest": expected_digest,
        "container_id": container_id,
        "observed_image_id": actual_image,
        "config_digest": config,
        "pid": pid,
        "scope": scope,
        "started_at": started_at,
        "restart_count": inspected["RestartCount"],
        "runner_argv_sha256": hashlib.sha256(
            json.dumps(expected_runner_argv, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def verify_finished_container(start, inspected, *, exit_scope_drained):
    """Require the same container to have exited; scope draining is independently checked."""
    state = inspected.get("State") if isinstance(inspected, dict) else None
    if (
        not isinstance(start, dict)
        or not isinstance(state, dict)
        or inspected.get("Id") != start.get("container_id")
        or inspected.get("Image") != start.get("observed_image_id")
        or inspected.get("RestartCount") != start.get("restart_count")
        or state.get("StartedAt") != start.get("started_at")
        or state.get("Running") is not False
        or type(state.get("ExitCode")) is not int
        or state["ExitCode"] != 0
        or exit_scope_drained is not True
    ):
        raise ValueError("image execution completion or drain is not established")
    return {
        "container_digest": start["container_digest"],
        "container_id": start["container_id"],
        "scope": start["scope"],
        "exit_code": state["ExitCode"],
        "runner_pid": start["pid"],
        "n_processes": 1,
        "n_restarts": 0,
    }
