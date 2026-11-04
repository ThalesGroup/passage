"""One validation-only direct OCI relay under an already-held CAVIAR lease.

The image, input index, fragments, host keys and configuration are pinned. The
container is retained after exit. This relay does not admit or read test data.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import shlex
import signal
import struct
import subprocess
import tarfile
import threading
import time
from pathlib import Path, PurePosixPath

from passage_eval.campaign.direct import persistent_transport, rule8_image
from passage_eval.campaign.direct.direct_session import inspect_journal
from passage_eval.campaign.direct.direct_target import (
    SCHEMA as VALIDATION_SCHEMA,
    REPLICA_VALIDATION_SCHEMA,
    validate_config,
)
from passage_eval.io import pinned

_MACHINE_IDS = {"orin", "thor"}

_SPEC_KEYS = {
    "schema",
    "target",
    "known_hosts",
    "known_hosts_sha256",
    "image_archive",
    "image_archive_sha256",
    "image_manifest_digest",
    "prepared_root",
    "prepared_sha256",
    "configuration",
    "configuration_sha256",
    "compiled_manifest_host",
    "bundle_manifest_host",
    "target_work_root",
    "target_readonly_mounts",
    "runner_argv",
    "groups",
    "output_root",
    "cavops",
    "timeout_seconds",
    "user",
    "replica_catalog",
    "replica_catalog_sha256",
    "validation_roster", "validation_roster_sha256", "warmup_roster", "warmup_roster_sha256",
}


def _file_digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _remote_path(value):
    if not isinstance(value, str) or not value.startswith("/") or "," in value or "\n" in value:
        raise ValueError("target path must be an absolute bind-safe path")
    path = PurePosixPath(value)
    if ".." in path.parts or any(part.lower() == "test" for part in path.parts) or path == PurePosixPath("/"):
        raise ValueError("target path must not traverse or name the root")
    return str(path)


def _validation_host_path(value):
    """Refuse test components and symbolic links before opening host inputs."""
    path = Path(value).absolute()
    if any(part.lower() == "test" for part in path.parts):
        raise ValueError("validation input path contains a test component")
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            raise ValueError("validation input path contains a symbolic link")
    return path


def _image_signature(path):
    info = Path(path).stat()
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _verify_image_file(path, expected, cache):
    """Hash once per unchanged file within one queue process; fail on concurrent edits."""
    signature = _image_signature(path)
    key = (str(Path(path).absolute()), expected)
    if cache is None or cache.get(key) != signature:
        if _file_digest(path) != expected:
            raise ValueError("image_archive pin differs")
        if _image_signature(path) != signature:
            raise ValueError("image_archive changed during verification")
        if cache is not None:
            cache[key] = signature


def _bundle_artifact_paths(bundle, bundle_target):
    """Project only explicitly pinned bundle members into target file mounts."""
    entries = bundle.get("artifacts") if isinstance(bundle, dict) else None
    if not isinstance(entries, dict) or not entries:
        raise ValueError("bundle artifacts missing")
    base = PurePosixPath(_remote_path(bundle_target)).parent
    paths = []
    for name, digest in entries.items():
        if not isinstance(name, str) or not pinned.is_hex64(digest):
            raise ValueError("bundle artifact pin differs")
        relative = PurePosixPath(name)
        if relative.is_absolute() or not relative.parts or any(part in (".", "..") for part in relative.parts):
            raise ValueError("bundle artifact path differs")
        paths.append(PurePosixPath(_remote_path(str(base / relative))))
    return paths


def load_spec(path, sha256, *, image_cache=None, admitted_session=None):
    """Validate the local, host-only control before making any remote connection."""
    spec = pinned.read_pinned_json(_validation_host_path(path), sha256)
    if not isinstance(spec, dict) or set(spec) != _SPEC_KEYS or spec["schema"] != "passage-oci-validation-relay/v1":
        raise ValueError("OCI validation relay specification differs")
    target = spec["target"]
    if target not in _MACHINE_IDS:
        raise ValueError("only Orin and Thor support this OCI validation relay")
    _machine_id(target)
    _ssh_host(target)
    for name in (
        "known_hosts",
        "image_archive",
        "configuration",
        "prepared_root",
        "compiled_manifest_host",
        "bundle_manifest_host",
        "output_root",
    ):
        _validation_host_path(spec[name])
    for key in ("known_hosts_sha256", "image_archive_sha256", "prepared_sha256", "configuration_sha256"):
        if not pinned.is_hex64(spec[key]):
            raise ValueError(f"{key} must be a SHA-256")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(spec["image_manifest_digest"])):
        raise ValueError("image manifest digest differs")
    if type(spec["timeout_seconds"]) not in (int, float) or not 0 < spec["timeout_seconds"] <= 1000:
        raise ValueError("relay timeout differs")
    if not isinstance(spec["target_readonly_mounts"], list) or not spec["target_readonly_mounts"]:
        raise ValueError("read-only artifact mounts required")
    for item in spec["target_readonly_mounts"]:
        _remote_path(item)
    _remote_path(spec["target_work_root"])
    if not isinstance(spec["user"], str) or not re.fullmatch(r"[0-9]+:[0-9]+", spec["user"]):
        raise ValueError("container user must be numeric UID:GID")
    if not isinstance(spec["runner_argv"], list) or not all(isinstance(x, str) and x for x in spec["runner_argv"]):
        raise ValueError("runner argument vector differs")
    if not isinstance(spec["groups"], list) or any(type(x) is not int or x < 0 for x in spec["groups"]):
        raise ValueError("supplementary group IDs differ")
    for path_key, hash_key in (("known_hosts", "known_hosts_sha256"), ("configuration", "configuration_sha256")):
        if _file_digest(spec[path_key]) != spec[hash_key]:
            raise ValueError(f"{path_key} pin differs")
    image_signature = _image_signature(spec["image_archive"])
    _verify_image_file(spec["image_archive"], spec["image_archive_sha256"], image_cache)
    config = validate_config(pinned.read_pinned_json(spec["configuration"], spec["configuration_sha256"]))
    if admitted_session is not None:
        from passage_eval.campaign.direct.admitted_stream import bind_target_config

        bound = bind_target_config(admitted_session, config, compiled_host_path=spec["compiled_manifest_host"])
        if bound.config_bytes != pinned.read_pinned(spec["configuration"], spec["configuration_sha256"]):
            raise ValueError("target configuration bytes differ from admitted session")
        if admitted_session.config_bytes != bound.config_bytes or config["target"] != target:
            raise ValueError("target configuration is not bound to the admission")
        if admitted_session.rule8_scope is not None:
            from passage_eval.campaign.direct.runner_binding import verify_g2_runner

            verify_g2_runner(admitted_session.grant, target, config["procedure"], spec["runner_argv"])
    elif (
        config["schema"] not in {VALIDATION_SCHEMA, REPLICA_VALIDATION_SCHEMA}
        or config["dataset_split"] != "validation"
        or config["target"] != target
    ):
        raise ValueError("only the target validation configuration is permitted")
    if admitted_session is None and config["schema"] == REPLICA_VALIDATION_SCHEMA:
        from passage_eval.protocol.replicas import load_replica_catalog, replica_identity

        if not isinstance(spec["replica_catalog"], str) or not pinned.is_hex64(spec["replica_catalog_sha256"]):
            raise ValueError("replica catalogue pin required")
        catalog_path = _validation_host_path(spec["replica_catalog"])
        catalog = load_replica_catalog(catalog_path, spec["replica_catalog_sha256"])
        identity = replica_identity(catalog, config["procedure"], config["seed"])
        if identity["checkpoint_sha256"] != config["artifact_pins"]["checkpoint"]:
            raise ValueError("replica checkpoint differs from catalogue")
    elif admitted_session is None and (spec["replica_catalog"] is not None or spec["replica_catalog_sha256"] is not None):
        raise ValueError("2026 validation does not use a replica catalogue")
    if config["procedure"] not in ("seg", "reg"):
        raise ValueError("validation procedure differs")
    artifacts = config.get("artifact_paths")
    if not isinstance(artifacts, dict) or set(artifacts) != {"bundle", "checkpoint", "engine", "decoder"}:
        raise ValueError("candidate artifact paths differ")
    roots = [PurePosixPath(item) for item in spec["target_readonly_mounts"]]

    def require_mounted(value):
        path = PurePosixPath(_remote_path(value))
        if not any(path == root or root in path.parents for root in roots):
            raise ValueError("candidate artifact is not covered by a read-only mount")

    for value in artifacts.values():
        require_mounted(value)
    require_mounted(config["compiled_manifest"])
    if config["grn_evidence"] is not None:
        for key in ("rewrite_script", "input_archive", "result_archive"):
            require_mounted(config["grn_evidence"][key])
        reproduction = PurePosixPath(_remote_path(config["grn_evidence"]["reproduction_root"]))
        if reproduction != PurePosixPath("/tmp") and PurePosixPath("/tmp") not in reproduction.parents:
            raise ValueError("GRN reproductions must use the retained session scratch")
    compiled = pinned.read_pinned_json(spec["compiled_manifest_host"], config["compiled_sha256"])
    if compiled.get("engine_sha256") != config["artifact_pins"]["engine"]:
        raise ValueError("compiled engine differs from validation configuration")
    engine_name = compiled.get("engine")
    if not isinstance(engine_name, str) or PurePosixPath(engine_name).name != engine_name:
        raise ValueError("compiled engine filename differs")
    bundle = pinned.read_pinned_json(spec["bundle_manifest_host"], config["artifact_pins"]["bundle"])
    required = [PurePosixPath(value) for value in artifacts.values()]
    required.extend(_bundle_artifact_paths(bundle, artifacts["bundle"]))
    required.append(PurePosixPath(config["compiled_manifest"]))
    required.append(PurePosixPath(config["compiled_manifest"]).parent / engine_name)
    if config["grn_evidence"] is not None:
        required.extend(PurePosixPath(config["grn_evidence"][key]) for key in
                        ("rewrite_script", "input_archive", "result_archive"))
    if any(len(root.parts) < 5 or root not in required for root in roots):
        raise ValueError("read-only mount must be an individually declared artifact file")
    if any(path not in roots for path in required):
        raise ValueError("each declared artifact file must have its own read-only mount")
    prepared = Path(spec["prepared_root"])
    if admitted_session is None and _file_digest(prepared / "prepared.json") != spec["prepared_sha256"]:
        raise ValueError("prepared validation input pin differs")
    expected_sdk = (compiled.get("runtime_versions") or {}).get("tensorrt")
    if not isinstance(expected_sdk, str) or not expected_sdk:
        raise ValueError("compiled TensorRT version missing")
    with tarfile.open(spec["image_archive"]) as archive:
        index = json.load(archive.extractfile("index.json"))
        if len(index.get("manifests", [])) != 1 or index["manifests"][0].get("digest") != spec["image_manifest_digest"]:
            raise ValueError("OCI image manifest index differs")
        blob = archive.extractfile("blobs/sha256/" + spec["image_manifest_digest"][7:]).read()
    if "sha256:" + hashlib.sha256(blob).hexdigest() != spec["image_manifest_digest"]:
        raise ValueError("OCI manifest digest differs")
    if _image_signature(spec["image_archive"]) != image_signature:
        raise ValueError("image_archive changed during validation")
    return spec, config, expected_sdk, blob


def _lease(spec, job):
    data = json.loads(subprocess.check_output([spec["cavops"], "leases", "--json"], timeout=15))
    target = spec["target"]
    if (
        sum(
            x.get("resource") == "device:edge-" + target
            and x.get("activity_id") == job
            and x.get("holder_host") == "edge-" + target
            and x.get("holder_state") == "running"
            for x in data
        )
        != 1
    ):
        raise RuntimeError("matching CAVIAR lease not held")


def _machine_id(target):
    if target not in {"orin", "thor", "tiger"}:
        raise ValueError("unsupported machine identity target")
    value = os.environ.get("PASSAGE_EDGE_MACHINE_ID_" + target.upper(), "")
    if not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("explicit machine identity required")
    return value


def _ssh_host(target):
    if target not in {"orin", "thor", "tiger"}:
        raise ValueError("unsupported SSH host target")
    value = os.environ.get("PASSAGE_EDGE_SSH_HOST_" + target.upper(), "")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", value):
        raise ValueError("explicit SSH host required")
    return value


def _ssh_argv(spec, command):
    return [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=8",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={spec['known_hosts']}",
        _ssh_host(spec["target"]),
        "bash",
        "-c",
        shlex.quote(command),
    ]


def _ssh(spec, command):
    return subprocess.run(_ssh_argv(spec, command), capture_output=True, timeout=20, check=False)


def _record(path, value):
    payload = pinned.canonical_json(value)
    with Path(path).open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _stream_prepared(stream, root, expected_sha):
    prepared = pinned.read_pinned_json(root / "prepared.json", expected_sha)
    required = {"index_sha256", "count", "warmup_index", "warmup_sha256", "fragments", "split", "budget_ms"}
    if (
        not isinstance(prepared, dict)
        or set(prepared) != required
        or prepared["split"] != "validation"
        or type(prepared["count"]) is not int
        or not 0 < prepared["count"] <= 1024
        or not isinstance(prepared["fragments"], list)
        or sum(x["n"] for x in prepared["fragments"]) != prepared["count"]
    ):
        raise ValueError("prepared validation input scope differs")
    persistent_transport.write_task_index(stream, root / "index.jsonl", prepared["index_sha256"])
    persistent_transport.write_array_frame(stream, "warmup", 0, root / "warmup.npz", prepared["warmup_sha256"])
    for i, item in enumerate(prepared["fragments"]):
        if (
            set(item) != {"sequence", "path", "sha256", "n"}
            or item["sequence"] != i
            or type(item["n"]) is not int
            or item["n"] < 1
        ):
            raise ValueError("prepared fragment identity differs")
        path = _validation_host_path(item["path"])
        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
            raise ValueError("prepared fragment leaves the validation root")
        persistent_transport.write_array_frame(stream, "fragment", i, path, item["sha256"])
    persistent_transport.write_end(stream)


def _validation_provenance(spec, config, prepared):
    """Bind the input-only projection to two pinned validation rosters before reading scenes."""
    from passage_eval.campaign.direct.raw_identity import task_from_full
    from passage_eval.io.roster import validate_roster

    rosters = []
    for name in ("validation_roster", "warmup_roster"):
        path = _validation_host_path(spec[name])
        roster = validate_roster(pinned.read_pinned_json(path, spec[name + "_sha256"]))
        if roster["dataset"]["split"] != "validation":
            raise ValueError("roster is not validation")
        rosters.append(roster)
    measurement, warmup = rosters
    if len(measurement["tasks"]) != 128 or len(warmup["tasks"]) != 1 or prepared["count"] != 128:
        raise ValueError("validation requires 128 scenes and one warmup scene")
    expected = [task_from_full(t["sample_id"], t["input_identity"]) for t in measurement["tasks"]]
    warm = [task_from_full(t["sample_id"], t["input_identity"]) for t in warmup["tasks"]]
    root = _validation_host_path(spec["prepared_root"])
    index = _validation_host_path(root / "index.jsonl")
    _validation_host_path(root / "warmup.npz")
    raw = pinned.read_pinned(index, prepared["index_sha256"])
    framed = struct.pack("!4sIQ", b"PSI1", 0, len(raw)) + raw
    actual = persistent_transport.read_task_index(io.BytesIO(framed), prepared["index_sha256"])
    if actual != expected or config["warmup_index"] != warm or prepared["warmup_index"] != warm:
        raise ValueError("validation input projection differs from its roster")
    if any(t["sample_id"] == warm[0]["sample_id"] or t["raw_input_sha256"] == warm[0]["raw_input_sha256"] for t in expected):
        raise ValueError("validation warmup overlaps measured inputs")
    return prepared


def _verify_archive(data, output, config, config_sha, sdk, count, *, admitted_session=None):
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("target archive exceeds bound")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:") as archive:
        members = archive.getmembers()
        names = {"journal.jsonl", "target-receipt.json", "target-receipt.sha256"}
        if (
            len(members) != 3
            or {m.name for m in members} != names
            or any(not m.isfile() or m.size > 128 * 1024 * 1024 for m in members)
        ):
            raise ValueError("target archive members differ")
        files = {m.name: archive.extractfile(m).read() for m in members}
    if admitted_session is not None:
        from passage_eval.campaign.direct.admitted_stream import verify_admitted_target_receipt

        output.mkdir(mode=0o700)
        for name, raw in files.items():
            with (output / name).open("xb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        receipt = verify_admitted_target_receipt(output, admitted_session)
        audit = inspect_journal(output / "journal.jsonl")
        if (not audit["complete"] or not audit["runtime_released"]
                or audit["committed_prefix"] != count or audit["ambiguous_range"] is not None):
            raise ValueError("admitted journal incomplete or release unconfirmed")
        return receipt
    digest = files["target-receipt.sha256"].decode("ascii").strip()
    receipt = pinned.strict_json_loads(files["target-receipt.json"], name="target receipt")
    if (
        hashlib.sha256(files["target-receipt.json"]).hexdigest() != digest
        or receipt.get("schema") != "passage-direct-target-validation-result-prototype/v1"
        or receipt.get("configuration_sha256") != config_sha
        or receipt.get("input_index_sha256") != config["index_sha256"]
        or receipt.get("native_artifact_hashes") != config["artifact_pins"]
        or receipt.get("qualification_status") != "exploratory_unadmitted"
        or receipt.get("scope_gpu_drain_verified") is not False
        or receipt.get("timing", {}).get("status") != "complete"
        or receipt["timing"].get("n_completed") != count
        or receipt["timing"].get("warmups_completed") != 16
        or hashlib.sha256(files["journal.jsonl"]).hexdigest() != receipt.get("journal_sha256")
    ):
        raise ValueError("validation target receipt differs")
    identity = receipt.get("runtime_identity") or {}
    options = identity.get("options") or {}
    versions = (identity.get("versions") or {}).get("tensorrt") or {}
    if (
        identity.get("implementation") != "passage_inference.backends.TensorRTRuntime"
        or options.get("device_id") != 0
        or options.get("engine_sha256") != config["artifact_pins"]["engine"]
        or versions.get("loaded_module_version") != sdk
        or (versions.get("distributions") or {}).get("tensorrt") != sdk
    ):
        raise ValueError("observed native runtime differs")
    output.mkdir(mode=0o700)
    for name, raw in files.items():
        with (output / name).open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    audit = inspect_journal(output / "journal.jsonl")
    if (
        not audit["complete"]
        or not audit["runtime_released"]
        or audit["committed_prefix"] != count
        or audit["ambiguous_range"] is not None
    ):
        raise ValueError("direct journal incomplete or release unconfirmed")
    return receipt


def _remote_command(spec, config_bytes, job, *, session_key=None):
    """Create one retained container inside a job-specific cgroup and client scope."""
    target = spec["target"]
    if session_key is not None and not re.fullmatch(r"[0-9a-f]{12}", session_key):
        raise ValueError("session key differs")
    suffix = job[4:] + ("-" + session_key if session_key is not None else "")
    scope = f"passage-edge-{target}-{suffix}.scope"
    slice_name = f"passage-edge-{target}-{suffix}.slice"
    work = _remote_path(spec["target_work_root"]) + "/oci-validation-" + suffix
    container = "passage-oci-validation-" + suffix
    config_path = work + "/config.json"
    result_path = work + "/output/result"
    runner = [arg.replace("{config}", config_path).replace("{output}", result_path) for arg in spec["runner_argv"]]
    if not rule8_image._is_direct_python_command(runner):
        raise ValueError("runner is not a direct Python invocation")
    if runner[-6:] != [
        "--config",
        config_path,
        "--config-sha256",
        spec["configuration_sha256"],
        "--output",
        result_path,
    ]:
        raise ValueError("runner arguments do not bind the pinned configuration and output")
    groups = [item for group in spec["groups"] for item in ("--group-add", str(group))]
    mounts = [
        item
        for path in spec["target_readonly_mounts"]
        for item in ("--mount", f"type=bind,src={path},dst={path},readonly")
    ]
    image = spec["image_manifest_digest"]
    docker = [
        "sudo",
        "-n",
        "docker",
        "run",
        "--name",
        container,
        "--cidfile",
        work + "/container.cid",
        "--cgroup-parent",
        slice_name,
        "--network",
        "none",
        "--runtime",
        "nvidia",
        "--env",
        "NVIDIA_VISIBLE_DEVICES=all",
        "--env",
        "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        "--read-only",
        "--mount",
        f"type=bind,src={work}/scratch,dst=/tmp",
        "--user",
        spec["user"],
        *groups,
        *mounts,
        "--mount",
        f"type=bind,src={work},dst={work}",
        "--entrypoint",
        runner[0],
        "-i",
        image,
        *runner[1:],
    ]
    config64 = base64.b64encode(config_bytes).decode("ascii")
    script = (
        'set -eu; [ "$(cat /etc/machine-id)" = ' + _machine_id(target) + " ]; umask 077; "
        "for p in " + shlex.join(spec["target_readonly_mounts"]) + '; do [ -f "$p" ] && [ ! -L "$p" ] '
        '&& [ "$(readlink -e -- "$p")" = "$p" ] || exit 73; done; '
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        "d=" + shlex.quote(work) + '; mkdir -m 700 -- "$d"; mkdir -m 700 -- "$d/output"; mkdir -m 700 -- "$d/scratch"; '
        "set -C; printf %s " + shlex.quote(config64) + ' | base64 -d > "$d/config.json"; set +C; '
        "systemd-run --user --scope --quiet --unit=" + shlex.quote(scope) + " /usr/bin/timeout --signal=TERM "
        "--kill-after=20s " + str(int(spec["timeout_seconds"])) + "s " + shlex.join(docker) + " <&0 >&2 & runner=$!; "
        'if wait "$runner"; then rc=0; else rc=$?; fi; '
        'if [ -d "$d/output/result" ]; then tar -cf - -C "$d/output/result" journal.jsonl target-receipt.json target-receipt.sha256; fi; exit "$rc"'
    )
    return script, scope, slice_name, container, runner


def _container_cgroup(spec, pid):
    if type(pid) is not int or pid <= 0:
        raise ValueError("invalid observed container PID")
    result = _ssh(spec, "cat /proc/" + str(pid) + "/cgroup")
    if result.returncode:
        raise RuntimeError("container cgroup could not be read")
    paths = [line[3:] for line in result.stdout.decode("ascii").splitlines() if line.startswith("0::")]
    if len(paths) != 1 or not re.fullmatch(r"/[A-Za-z0-9_./:@-]+", paths[0]) or ".." in PurePosixPath(paths[0]).parts:
        raise ValueError("container cgroup differs")
    return paths[0]


def _signal_local(process, signum):
    """The SSH process may exit after poll and before the signal is sent."""
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass


def _drain(spec, container, scope, cgroup):
    path = cgroup or "/nonexistent-passage-oci-cgroup"
    command = (
        'set -eu; running=$(sudo -n docker inspect --format "{{.State.Running}}" ' + shlex.quote(container) + "); "
        '[ "$running" = false ]; export XDG_RUNTIME_DIR="/run/user/$(id -u)" '
        'DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        "state=$(systemctl --user show "
        + shlex.quote(scope)
        + ' --property=ActiveState --value); [ "$state" != active ]; '
        "for cg in " + shlex.quote(path) + '; do if [ -d "/sys/fs/cgroup$cg" ]; then '
        'while IFS= read -r f; do [ -z "$(sudo -n cat "$f")" ] || exit 72; done '
        '< <(sudo -n find "/sys/fs/cgroup$cg" -name cgroup.procs -type f); fi; done; printf DRAINED'
    )
    result = _ssh(spec, command)
    if result.returncode != 0 or result.stdout != b"DRAINED":
        raise RuntimeError("OCI container or cgroup drainage unproven")


def _journal_binding(raw, observed_namespace_pid):
    """Verify UUID/PID markers when present; old validation runners stay exploratory."""
    events = [json.loads(line) for line in raw.splitlines()]
    beginnings = [event for event in events if event.get("kind") == "session_start"]
    endings = [event for event in events if event.get("kind") == "session_complete"]
    if len(beginnings) != 1 or len(endings) != 1:
        raise ValueError("direct journal must contain one session start and completion")
    begin, end = beginnings[0], endings[0]
    marker_keys = ("session_id", "namespace_pid")
    if not any(key in begin or key in end for key in marker_keys):
        return {"journal_markers": "unavailable_in_exploratory_runner"}
    sid = begin.get("session_id")
    if (
        not isinstance(sid, str)
        or not re.fullmatch(r"[0-9a-f]{32}", sid)
        or sid != end.get("session_id")
        or type(observed_namespace_pid) is not int
        or observed_namespace_pid < 1
        or begin.get("namespace_pid") != observed_namespace_pid
        or end.get("namespace_pid") != observed_namespace_pid
    ):
        raise ValueError("runner session UUID or namespace PID markers differ")
    return {"runner_session_uuid": sid, "journal_markers": "matched"}


def _run_validation(spec_path, spec_sha256, *, session_key=None, image_cache=None, admitted_session=None):
    """Execute one validation session under the exact lease; leave all outputs intact."""
    spec, config, sdk, manifest = load_spec(
        spec_path, spec_sha256, image_cache=image_cache, admitted_session=admitted_session
    )
    job = os.environ.get("AGENT_OPS_JOB", "")
    if not re.fullmatch(r"job:[0-9a-f]{16}", job):
        raise ValueError("CAVIAR job identity missing")
    lease_lost = False
    def check_lease():
        nonlocal lease_lost
        if lease_lost:
            raise RuntimeError("lease previously lost")
        try:
            _lease(spec, job)
        except InterruptedError:
            raise
        except BaseException:
            lease_lost = True
            raise

    check_lease()
    prepared = None
    if admitted_session is None:
        prepared = pinned.read_pinned_json(Path(spec["prepared_root"]) / "prepared.json", spec["prepared_sha256"])
        if (
            prepared.get("split") != "validation"
            or prepared.get("index_sha256") != config["index_sha256"]
            or prepared.get("warmup_index") != config["warmup_index"]
        ):
            raise ValueError("prepared validation roster differs from target configuration")
        _validation_provenance(spec, config, prepared)
    script, scope, slice_name, container, runner = _remote_command(
        spec, pinned.read_pinned(spec["configuration"], spec["configuration_sha256"]), job, session_key=session_key
    )
    output = Path(spec["output_root"])
    output.mkdir(mode=0o700)
    process = None
    started = None
    failure = None
    actual_cgroup = None
    with (output / "response.tar").open("xb") as stdout, (output / "remote.stderr").open("xb") as stderr:
        try:
            process = subprocess.Popen(
                _ssh_argv(spec, script), stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True
            )
            deadline = time.monotonic() + 60
            while True:
                check_lease()
                query = _ssh(spec, "sudo -n docker inspect " + shlex.quote(container))
                if query.returncode == 0:
                    inspected = json.loads(query.stdout)[0]
                    if inspected.get("State", {}).get("Running") is True:
                        break
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError("validation container did not start")
                time.sleep(1)
            pid = inspected["State"]["Pid"]
            actual_cgroup = _container_cgroup(spec, pid)
            cid = inspected["Id"]
            if not (actual_cgroup.endswith("/docker-" + cid + ".scope") or actual_cgroup.endswith("/docker/" + cid)):
                raise ValueError("runner PID is outside its Docker cgroup")
            status = _ssh(spec, "cat /proc/" + str(pid) + "/status")
            if status.returncode:
                raise RuntimeError("runner namespace PID could not be observed")
            pid_lines = [line for line in status.stdout.decode("ascii").splitlines() if line.startswith("NSpid:")]
            if len(pid_lines) != 1:
                raise ValueError("runner namespace PID observation differs")
            try:
                pid_chain = [int(value) for value in pid_lines[0].split()[1:]]
            except ValueError as exc:
                raise ValueError("runner namespace PID observation differs") from exc
            if not pid_chain or pid_chain[0] != pid or any(value <= 0 for value in pid_chain):
                raise ValueError("runner namespace PID observation differs")
            started = rule8_image.verify_running_container(
                manifest,
                inspected,
                spec["image_manifest_digest"],
                scope=actual_cgroup,
                process_in_scope=lambda pid, cgroup: cgroup == actual_cgroup and _container_cgroup(spec, pid) == cgroup,
                launch_reference=spec["image_manifest_digest"],
                expected_runner_argv=runner,
            )
            started.update(
                {
                    "schema": "passage-oci-validation-start/v1",
                    "launch_reference": spec["image_manifest_digest"],
                    "job_id": job,
                    "target": spec["target"],
                    "procedure": config["procedure"],
                    "configuration_sha256": spec["configuration_sha256"],
                    "input_index_sha256": config["index_sha256"],
                    "client_scope": scope,
                    "configured_parent_slice": slice_name,
                    "namespace_pid": pid_chain[-1],
                    "runner_argv": runner,
                }
            )
            if admitted_session is None:
                _record(output / "image-start.json", started)
            errors = []
            cancelled = threading.Event()

            class LeasedStream:
                def write(self, data):
                    if cancelled.is_set():
                        raise RuntimeError("lease no longer held")
                    return process.stdin.write(data)

                def flush(self):
                    if cancelled.is_set():
                        raise RuntimeError("lease no longer held")
                    return process.stdin.flush()

            def sender():
                try:
                    if admitted_session is None:
                        _stream_prepared(LeasedStream(), Path(spec["prepared_root"]), spec["prepared_sha256"])
                    else:
                        from passage_eval.campaign.direct.admitted_stream import stream_admitted_session

                        callbacks = {}
                        if admitted_session.rule8_scope is not None:
                            callbacks = {
                                "image_inspect": lambda: (manifest, inspected, actual_cgroup,
                                    spec["image_manifest_digest"], runner),
                                "process_in_scope": lambda p, c: c == actual_cgroup and _container_cgroup(spec, p) == c,
                                "image_evidence_path": output / "image-start.json",
                            }
                        stream_admitted_session(LeasedStream(), admitted_session, **callbacks)
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    try:
                        process.stdin.close()
                    except (OSError, ValueError) as exc:
                        errors.append(exc)

            thread = threading.Thread(target=sender, daemon=True)
            thread.start()
            deadline = time.monotonic() + spec["timeout_seconds"] + 10
            while process.poll() is None:
                check_lease()
                if errors:
                    raise RuntimeError("input sender failed") from errors[0]
                if time.monotonic() > deadline:
                    raise TimeoutError("validation session timed out")
                time.sleep(3)
            thread.join(timeout=10)
            if thread.is_alive() or errors or process.returncode:
                raise RuntimeError("validation process or sender failed")
            check_lease()
        except BaseException as exc:
            failure = exc
            if "cancelled" in locals():
                cancelled.set()
        finally:
            # Cleanup has its own bounded SSH and wait deadlines. Do not let
            # a late alarm or a repeated TERM interrupt stop/drain halfway.
            signal.setitimer(signal.ITIMER_REAL, 0)
            for signum in (signal.SIGTERM, signal.SIGALRM, signal.SIGINT, signal.SIGHUP):
                signal.signal(signum, signal.SIG_IGN)
            lease_valid = True
            try:
                check_lease()
            except BaseException as lease_error:
                lease_valid = False
                failure = RuntimeError("lease lost: remote process and drain state unknown")
                failure.__cause__ = lease_error
            if lease_valid and process and process.poll() is None:
                try:
                    _ssh(spec, "sudo -n docker stop --time 5 " + shlex.quote(container) + " >/dev/null 2>&1 || true")
                except (OSError, subprocess.TimeoutExpired) as stop_error:
                    if failure is None:
                        failure = RuntimeError("remote stop could not be observed")
                        failure.__cause__ = stop_error
                if process.poll() is None:
                    _signal_local(process, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        _signal_local(process, signal.SIGKILL)
                        process.wait(timeout=10)
            stdout.flush()
            os.fsync(stdout.fileno())
            stderr.flush()
            os.fsync(stderr.fileno())
    if not lease_valid:
        if process and process.poll() is None:
            try:
                _signal_local(process, signal.SIGTERM)
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        if "thread" in locals():
            thread.join(timeout=10)
            if thread.is_alive():
                _record(output / "failure.json", {"job_id": job, "remote_state": "unknown", "reason": "producer_blocked"})
                raise RuntimeError("lease lost; local producer did not stop") from failure
        _record(output / "failure.json", {"job_id": job, "remote_state": "unknown", "reason": "lease_lost"})
        raise failure
    if "thread" in locals():
        thread.join(timeout=10)
        if thread.is_alive():
            _record(output / "failure.json", {"job_id": job, "remote_state": "unknown", "reason": "producer_blocked"})
            raise RuntimeError("local producer did not stop")
    check_lease()
    try:
        _drain(spec, container, scope, actual_cgroup)
    except BaseException:
        _record(output / "failure.json", {"job_id": job, "remote_state": "unknown", "reason": "drain_unproven"})
        raise
    if failure:
        _record(output / "failure.json", {"job_id": job, "remote_state": "drained", "reason": type(failure).__name__})
        raise failure
    check_lease()
    final = _ssh(spec, "sudo -n docker inspect " + shlex.quote(container))
    if final.returncode:
        raise RuntimeError("final container inspection failed")
    finished = rule8_image.verify_finished_container(started, json.loads(final.stdout)[0], exit_scope_drained=True)
    receipt = _verify_archive(
        (output / "response.tar").read_bytes(),
        output / "collected",
        config,
        spec["configuration_sha256"],
        sdk,
        prepared["count"] if prepared is not None else len(admitted_session.tasks),
        admitted_session=admitted_session,
    )
    markers = _journal_binding((output / "collected" / "journal.jsonl").read_bytes(), pid_chain[-1])
    if admitted_session is not None and markers.get("journal_markers") != "matched":
        raise ValueError("admitted session requires matching runner session and PID markers")
    finished.update(markers)
    _record(output / "image-finish.json", finished)
    drain = {
            "schema": "passage-direct-session-drain/v1",
            "job_id": job,
            "target": spec["target"],
            "procedure": config["procedure"],
            "scope": actual_cgroup,
            "client_scope": scope,
            "container_slice": slice_name,
            "actual_container_cgroup": actual_cgroup,
            "container_stopped": True,
            "scope_drained": True,
            "unit_inactive": True,
            "cgroup_empty": True,
            "session_processes": [],
            "handle_evidence": "process_exit_closes_session_descriptors",
            "runtime_synchronize_close_recorded": True,
            "driver_context_introspection": False,
            "gpu_evidence": "native_runtime_reported",
        }
    _record(output / "drain.json", drain)
    if admitted_session is not None and admitted_session.rule8_scope is not None:
        from passage_eval.campaign.direct.admitted_stream import verify_rule8_completed_session

        start_path = output / "image-start.json"
        verify_rule8_completed_session(
            admitted_session, output / "collected", start_path, _file_digest(start_path),
            container_inspect=json.loads(final.stdout)[0], drain=drain, job_id=job,
        )
    return {
        "job_id": job,
        "target": spec["target"],
        "procedure": config["procedure"],
        "journal_sha256": receipt["journal_sha256"],
        "status": "admitted_session_collected" if admitted_session is not None else "exploratory_validation_collected",
    }


def run_validation(spec_path, spec_sha256, *, session_key=None, image_cache=None, admitted_session=None,
                   deadline_monotonic=None):
    """Handle termination while the lease is held, allowing bounded stop and drain."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("relay must run in the main thread to handle termination")
    handled = (signal.SIGTERM, signal.SIGALRM, signal.SIGINT, signal.SIGHUP)
    old_handlers = {signum: signal.getsignal(signum) for signum in handled}
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("relay requires an unused process alarm")
    deadline = time.monotonic() + 1100 if deadline_monotonic is None else deadline_monotonic
    remaining = deadline - time.monotonic()
    if remaining <= 0 or remaining > 1100:
        raise TimeoutError("relay preparation exhausted its global session budget")

    def on_term(signum, frame):
        for signum in handled:
            signal.signal(signum, signal.SIG_IGN)
        raise InterruptedError("relay terminated; stopping under the held lease")

    for signum in handled:
        signal.signal(signum, on_term)
    signal.setitimer(signal.ITIMER_REAL, remaining)
    try:
        return _run_validation(spec_path, spec_sha256, session_key=session_key,
                               image_cache=image_cache, admitted_session=admitted_session)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def run_validation_queue(queue_path, queue_sha256):
    """One session per job and lease; orchestration belongs to the host dispatcher."""
    raise ValueError("multi-session jobs disabled: schedule one drained session per job and lease")


def main():
    deadline = time.monotonic() + 1100
    parser = argparse.ArgumentParser(allow_abbrev=False)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec")
    group.add_argument("--queue")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--record")
    parser.add_argument("--evidence-dir")
    parser.add_argument("--roster")
    parser.add_argument("--dataset-root")
    parser.add_argument("--warmup-npz")
    parser.add_argument("--warmup-npz-sha256")
    parser.add_argument("--scope")
    parser.add_argument("--scope-sha256")
    args = parser.parse_args()
    admission_args = (args.record, args.evidence_dir, args.roster, args.dataset_root, args.warmup_npz,
                      args.warmup_npz_sha256)
    if any(admission_args):
        if not args.spec or not all(admission_args) or bool(args.scope) != bool(args.scope_sha256):
            parser.error("admitted mode needs spec, record, evidence, roster, dataset and pinned warmup")
        from passage_eval.campaign.direct.admitted_stream import bind_target_config, prepare_admitted_session
        from passage_eval.campaign.direct.rule8_scope import load_rule8_scope
        from passage_eval.io.roster import validate_roster
        from passage_eval.protocol.contracts import BudgetSpec

        spec = pinned.read_pinned_json(_validation_host_path(args.spec), args.sha256)
        config = validate_config(pinned.read_pinned_json(_validation_host_path(spec["configuration"]),
                                                         spec["configuration_sha256"]))
        budget = BudgetSpec(**config["budget"])
        warm_roster = validate_roster(pinned.read_pinned_json(_validation_host_path(spec["warmup_roster"]),
                                                             spec["warmup_roster_sha256"]))
        frame = io.BytesIO()
        persistent_transport.write_array_frame(frame, "warmup", 0, _validation_host_path(args.warmup_npz),
                                               args.warmup_npz_sha256)
        frame.seek(0)
        warmup = persistent_transport.read_warmup(frame, budget, config["warmup_index"])[0]
        scope = load_rule8_scope(args.scope, args.scope_sha256) if args.scope else None
        session = prepare_admitted_session(args.record, args.evidence_dir, args.roster, args.dataset_root,
                                           budget, warmup, warm_roster, rule8_scope=scope)
        session = bind_target_config(session, config, compiled_host_path=spec["compiled_manifest_host"])
        result = run_validation(args.spec, args.sha256, admitted_session=session, deadline_monotonic=deadline)
    else:
        if args.scope or args.scope_sha256:
            parser.error("scope requires admitted mode")
        result = (run_validation(args.spec, args.sha256, deadline_monotonic=deadline)
                  if args.spec else run_validation_queue(args.queue, args.sha256))
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
