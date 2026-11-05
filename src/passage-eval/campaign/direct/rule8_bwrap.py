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

"""Host-only evidence for an OCI rootfs executed by Tiger's rootless bwrap.

The dispatcher must perform all observations under the exact CAVIAR lease. This
checks their consistency, not producer authentication or transient rootfs changes.
"""

import re
import hashlib
import json

from passage_eval.campaign.direct.rule8_image import _is_direct_python_command

_ALLOWED_EXCEPTIONS = frozenset(
    {
        ".dockerenv",
        "dev/console",
        "dev/pts",
        "dev/shm",
        "etc/hostname",
        "etc/hosts",
        "etc/mtab",
        "etc/resolv.conf",
    }
)


def verify_running_bwrap(proof, expected_digest, *, config_sha256, index_sha256, process_in_scope):
    """Validate an independently observed OCI/export/tree/runner chain before PSI1."""
    if not isinstance(proof, dict) or set(proof) != {"comparison", "tree", "process", "job_id"}:
        raise ValueError("bwrap observation fields differ")
    comparison, tree, process, job = (proof[k] for k in ("comparison", "tree", "process", "job_id"))
    if not isinstance(job, str) or not re.fullmatch(r"job:[0-9a-f]{16}", job):
        raise ValueError("bwrap job identity differs")
    scope = "passage-edge-tiger-" + job[4:] + ".scope"
    exceptions = comparison.get("exceptions") if isinstance(comparison, dict) else None
    if (
        not isinstance(comparison, dict)
        or comparison.get("manifest_digest") != expected_digest
        or not re.fullmatch(r"sha256:[0-9a-f]{64}", str(comparison.get("config_digest")))
        or not re.fullmatch(r"[0-9a-f]{64}", str(comparison.get("rootfs_archive_sha256")))
        or not isinstance(exceptions, dict)
        or not set(exceptions) <= _ALLOWED_EXCEPTIONS
    ):
        raise ValueError("OCI and rootfs comparison differs")
    # Runtime-injected extras are allowed only as data or mountpoint entries.
    for name, diff in exceptions.items():
        if not isinstance(diff, dict) or set(diff) != {"oci", "export"}:
            raise ValueError("rootfs exception differs")
        item = diff["export"]
        if name in ("dev/pts", "dev/shm"):
            allowed = item == ["dir"] or item == ("dir",)
        elif name == "etc/mtab":
            allowed = item == ["symlink", "/proc/mounts"] or item == ("symlink", "/proc/mounts")
        else:
            allowed = isinstance(item, (list, tuple)) and len(item) == 4 and item[0] == "file" and item[1] == 0
        if not allowed:
            raise ValueError("rootfs exception includes unexpected content")
    if (
        not isinstance(tree, dict)
        or tree.get("rootfs_archive_sha256") != comparison["rootfs_archive_sha256"]
        or tree.get("file_count") != comparison.get("export_paths")
    ):
        raise ValueError("extracted rootfs verification differs")
    runner = process.get("runner") if isinstance(process, dict) else None
    if (
        not isinstance(process, dict)
        or process.get("target") != "tiger"
        or process.get("job_id") != job
        or process.get("scope") != scope
        or tree.get("rootfs_path") != process.get("rootfs")
        or not isinstance(runner, dict)
        or type(runner.get("pid")) is not int
        or runner["pid"] <= 0
        or type(runner.get("namespace_pid")) is not int
        or runner["namespace_pid"] <= 0
        or "ro" not in str(runner.get("root_mount_options", "")).split(",")
        or not _is_direct_python_command(runner.get("argv"))
        or process_in_scope(runner["pid"], scope) is not True
    ):
        raise ValueError("scoped bwrap runner observation differs")
    argv = runner["argv"]
    if "--config-sha256" not in argv or argv[argv.index("--config-sha256") + 1 : argv.index("--config-sha256") + 2] != [
        config_sha256
    ]:
        raise ValueError("runner configuration differs")
    return {
        "container_digest": expected_digest,
        "config_digest": comparison["config_digest"],
        "rootfs_archive_sha256": comparison["rootfs_archive_sha256"],
        "rootfs_exceptions": exceptions,
        "pid": runner["pid"],
        "namespace_pid": runner["namespace_pid"],
        "runner_argv": list(argv),
        "runner_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
        "scope": scope,
        "job_id": job,
        "configuration_sha256": config_sha256,
        "input_index_sha256": index_sha256,
        "limitations": ["point-in-time rootfs and process observation", "runtime mounts are outside the OCI rootfs"],
    }
