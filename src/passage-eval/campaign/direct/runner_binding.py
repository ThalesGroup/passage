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

"""Bind the admitted S runner to the command recorded by the G2 rehearsal."""

from __future__ import annotations

import hashlib
import json
import re

from passage_eval.io import pinned
from passage_eval.campaign.edge.lease import parse_job_id

_VARIABLE_OPTIONS = {"--config", "--config-sha256", "--output", "--split", "--dataset-split"}


def runner_identity(argv):
    """Keep program and functional options, masking only session-specific values."""
    if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) or not arg for arg in argv):
        raise ValueError("runner argument vector differs")
    result = []
    seen = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        result.append(token)
        if token in _VARIABLE_OPTIONS:
            if token in seen or index + 1 >= len(argv) or argv[index + 1].startswith("-"):
                raise ValueError("runner session argument differs")
            value = argv[index + 1]
            if token in {"--split", "--dataset-split"} and value not in {"validation", "test"}:
                raise ValueError("runner split differs")
            seen.add(token)
            result.append("<session-value>")
            index += 1
        index += 1
    return tuple(result)


def verify_tiger_observation(image):
    """Recheck the retained bwrap observation; this is not a new live probe."""
    from passage_eval.campaign.direct.rule8_bwrap import verify_running_bwrap

    checked = verify_running_bwrap(
        image.get("bwrap_proof"),
        image.get("container_digest"),
        config_sha256=image.get("configuration_sha256"),
        index_sha256=image.get("input_index_sha256"),
        process_in_scope=lambda pid, scope: pid == image.get("pid") and scope == image.get("scope"),
    )
    if any(pinned.canonical_json(image.get(key)) != pinned.canonical_json(value) for key, value in checked.items()):
        raise ValueError("recorded Tiger bwrap observation differs")
    return True


def verify_g2_runner(grant, target, procedure, runner_argv, *, rootfs_export_sha256=None, rootfs_exceptions=None):
    """Read the G2 receipt already pinned by a genuine admission grant."""
    if not grant.is_genuine():
        raise ValueError("genuine admission required")
    grant.ensure_unchanged()
    record = pinned.read_pinned_json(grant.record_path, grant.record_sha256)
    ref = record["evidence"]["edge"][target]
    path, digest = pinned.file_ref(ref)
    receipt = pinned.read_pinned_json(pinned.resolve_inside(grant.evidence_dir, path), digest)
    if receipt.get("target") != target or receipt.get("split") != "validation":
        raise ValueError("G2 validation receipt identity differs")
    entry = receipt["procedures"][procedure]
    validation = entry.get("runner_argv")
    image_path, image_sha = pinned.file_ref(entry.get("image_start"))
    image = pinned.read_pinned_json(pinned.resolve_inside(grant.evidence_dir, image_path), image_sha)
    observed = image.get("runner_argv")
    if (
        type(image.get("pid")) is not int
        or image["pid"] <= 0
        or type(image.get("namespace_pid")) is not int
        or image["namespace_pid"] <= 0
        or not isinstance(image.get("scope"), str)
        or not image["scope"]
        or (
            target != "tiger"
            and (
                not isinstance(image.get("container_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", image["container_id"])
            )
        )
        or image.get("target") != target
        or image.get("procedure") != procedure
        or observed != validation
        or not isinstance(observed, list)
        or hashlib.sha256(json.dumps(observed, separators=(",", ":")).encode()).hexdigest()
        != image.get("runner_argv_sha256")
        or image.get("container_digest") != record["candidates"][procedure]["targets"][target]["container_digest"]
    ):
        raise ValueError("G2 runner is not linked to observed image execution")
    from passage_eval.campaign.direct.direct_host_scoring import verify_drain
    from passage_eval.campaign.direct.oci_validation_relay import _journal_binding

    finish_path, finish_sha = pinned.file_ref(entry.get("image_finish"))
    finish = pinned.read_pinned_json(pinned.resolve_inside(grant.evidence_dir, finish_path), finish_sha)
    drain_path, drain_sha = pinned.file_ref(entry.get("drain"))
    drain = pinned.read_pinned_json(pinned.resolve_inside(grant.evidence_dir, drain_path), drain_sha)
    journal_path, journal_sha = pinned.file_ref(entry.get("journal"))
    journal = pinned.read_pinned(pinned.resolve_inside(grant.evidence_dir, journal_path), journal_sha)
    marks = _journal_binding(journal, image.get("namespace_pid"))
    job, observed_job = drain.get("job_id"), image.get("job_id")
    if (
        not isinstance(job, str)
        or not isinstance(observed_job, str)
        or parse_job_id(job) != job
        or parse_job_id(observed_job) != observed_job
        or job != observed_job
    ):
        raise ValueError("G2 drain job differs from observed image session")
    if (
        finish.get("scope") != image.get("scope")
        or finish.get("runner_pid") != image.get("pid")
        or finish.get("container_digest") != image.get("container_digest")
        or finish.get("exit_code") != 0
        or (target != "tiger" and finish.get("container_id") != image.get("container_id"))
        or marks.get("journal_markers") != "matched"
        or finish.get("runner_session_uuid") != marks.get("runner_session_uuid")
        or entry.get("native_execution", {}).get("session_id") != journal_sha
    ):
        raise ValueError("G2 runner observation is not linked to the validation journal")
    verify_drain(drain, target=target, procedure=procedure, job_id=drain.get("job_id"), scope=image["scope"])
    if runner_identity(validation) != runner_identity(runner_argv):
        raise ValueError("admitted runner differs from G2 validation")
    if target == "tiger":
        verify_tiger_observation(image)
        if (
            not pinned.is_hex64(rootfs_export_sha256)
            or not isinstance(rootfs_exceptions, dict)
            or entry.get("rootfs_export_sha256") != rootfs_export_sha256
            or image.get("rootfs_archive_sha256") != rootfs_export_sha256
            or pinned.canonical_json(entry.get("rootfs_exceptions")) != pinned.canonical_json(rootfs_exceptions)
            or pinned.canonical_json(image.get("rootfs_exceptions")) != pinned.canonical_json(rootfs_exceptions)
        ):
            raise ValueError("admitted Tiger rootfs differs from G2 validation")
    elif rootfs_export_sha256 is not None or rootfs_exceptions is not None:
        raise ValueError("rootfs binding only applies to Tiger")
    grant.ensure_unchanged()
    return True
