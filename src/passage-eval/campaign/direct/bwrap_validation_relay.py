"""One admitted Tiger bwrap session under a pre-existing CAVIAR lease.

OCI/archive and extracted rootfs checks run before any S data leaves the host.
The session keeps a single runner and never automatically resumes.
"""

from __future__ import annotations

import base64
import hashlib
import io
import tarfile
import json
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from pathlib import Path, PurePosixPath

from passage_eval.campaign.direct import bwrap_lineage
from passage_eval.campaign.direct.direct_target import (
    SCHEMA as VALIDATION_SCHEMA,
    REPLICA_VALIDATION_SCHEMA,
    validate_config,
)
from passage_eval.campaign.direct.oci_validation_relay import (
    _bundle_artifact_paths,
    _file_digest,
    _image_signature,
    _verify_image_file,
    _lease,
    _machine_id,
    _ssh_host,
    _record,
    _remote_path,
    _stream_prepared,
    _validation_provenance,
    _journal_binding,
    _ssh,
    _ssh_argv,
    _validation_host_path,
    _verify_archive,
)
from passage_eval.campaign.direct.rule8_image import _is_direct_python_command
from passage_eval.io import pinned

_KEYS = {
    "schema",
    "target",
    "known_hosts",
    "known_hosts_sha256",
    "image_archive",
    "image_archive_sha256",
    "image_manifest_digest",
    "rootfs_export",
    "rootfs_export_sha256",
    "target_rootfs_export",
    "target_rootfs",
    "rootfs_exceptions",
    "configuration",
    "configuration_sha256",
    "prepared_root",
    "prepared_sha256",
    "validation_roster",
    "validation_roster_sha256",
    "replica_catalog",
    "replica_catalog_sha256",
    "warmup_roster",
    "warmup_roster_sha256",
    "compiled_manifest_host",
    "bundle_manifest_host",
    "target_work_root",
    "target_readonly_mounts",
    "runner_argv",
    "output_root",
    "cavops",
    "timeout_seconds",
}


def inspect_spec(path, digest):
    """Preflight a pinned control and validation projection without opening S."""
    spec = pinned.read_pinned_json(_validation_host_path(path), digest)
    if (
        not isinstance(spec, dict)
        or set(spec) != _KEYS
        or spec.get("schema") != "passage-tiger-bwrap-relay/v1"
        or spec.get("target") != "tiger"
    ):
        raise ValueError("Tiger bwrap specification differs")
    _machine_id("tiger")
    _ssh_host("tiger")
    if not pinned.is_hex64(spec.get("configuration_sha256")):
        raise ValueError("configuration digest differs")
    config = validate_config(
        pinned.read_pinned_json(_validation_host_path(spec["configuration"]), spec["configuration_sha256"])
    )
    if config.get("target") != "tiger":
        raise ValueError("configuration target differs")
    if config.get("dataset_split") == "validation":
        prepared = pinned.read_pinned_json(
            _validation_host_path(Path(spec["prepared_root"]) / "prepared.json"), spec["prepared_sha256"]
        )
        if (
            prepared.get("split") != "validation"
            or prepared.get("count") != 128
            or prepared.get("index_sha256") != config["index_sha256"]
            or prepared.get("warmup_index") != config["warmup_index"]
        ):
            raise ValueError("prepared validation identity differs")
        _validation_provenance(spec, config, prepared)
    return spec, config


def load_spec(path, digest, session=None, *, image_cache=None):
    from passage_eval.campaign.direct.admitted_stream import bind_target_config

    spec = pinned.read_pinned_json(_validation_host_path(path), digest)
    if (
        not isinstance(spec, dict)
        or set(spec) != _KEYS
        or spec["schema"] != "passage-tiger-bwrap-relay/v1"
        or spec["target"] != "tiger"
    ):
        raise ValueError("Tiger bwrap specification differs")
    _machine_id("tiger")
    _ssh_host("tiger")
    if session is not None and session.rule8_scope is None:
        raise ValueError("admitted Rule8 session required")
    for key in (
        "known_hosts",
        "image_archive",
        "rootfs_export",
        "configuration",
        "warmup_roster",
        "prepared_root",
        "validation_roster",
        "bundle_manifest_host",
        "output_root",
    ):
        _validation_host_path(spec[key])
    for key in ("known_hosts", "image_archive", "rootfs_export", "configuration", "warmup_roster"):
        if not pinned.is_hex64(spec[key + "_sha256"]):
            raise ValueError(f"{key} pin differs")
        if key in ("image_archive", "rootfs_export"):
            _verify_image_file(spec[key], spec[key + "_sha256"], image_cache)
        elif _file_digest(spec[key]) != spec[key + "_sha256"]:
            raise ValueError(f"{key} pin differs")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(spec["image_manifest_digest"])):
        raise ValueError("OCI manifest differs")
    if type(spec["timeout_seconds"]) not in (int, float) or not 0 < spec["timeout_seconds"] <= 1000:
        raise ValueError("timeout differs")
    if spec["compiled_manifest_host"] is not None:
        raise ValueError("Tiger does not use a compiled manifest")
    for key in ("target_rootfs_export", "target_rootfs", "target_work_root"):
        _remote_path(spec[key])
    if not isinstance(spec["target_readonly_mounts"], list) or not spec["target_readonly_mounts"]:
        raise ValueError("read-only mounts required")
    roots = [PurePosixPath(_remote_path(p)) for p in spec["target_readonly_mounts"]]
    config = validate_config(pinned.read_pinned_json(spec["configuration"], spec["configuration_sha256"]))
    if session is not None:
        bound = bind_target_config(session, config, compiled_host_path=spec["compiled_manifest_host"])
        if (
            config["target"] != "tiger"
            or session.config_bytes != bound.config_bytes
            or bound.config_bytes != pinned.read_pinned(spec["configuration"], spec["configuration_sha256"])
        ):
            raise ValueError("admitted configuration bytes differ")
    elif (
        config["target"] != "tiger"
        or config["dataset_split"] != "validation"
        or config["schema"] not in (VALIDATION_SCHEMA, REPLICA_VALIDATION_SCHEMA)
    ):
        raise ValueError("only pinned Tiger validation is permitted without admission")
    if session is None and config["schema"] == REPLICA_VALIDATION_SCHEMA:
        from passage_eval.protocol.replicas import load_replica_catalog, replica_identity

        catalog = load_replica_catalog(_validation_host_path(spec["replica_catalog"]), spec["replica_catalog_sha256"])
        if (
            replica_identity(catalog, config["procedure"], config["seed"])["checkpoint_sha256"]
            != config["artifact_pins"]["checkpoint"]
        ):
            raise ValueError("validation replica differs")
    elif session is None and (spec["replica_catalog"] is not None or spec["replica_catalog_sha256"] is not None):
        raise ValueError("unexpected validation replica catalog")
    paths = config.get("artifact_paths")
    if not isinstance(paths, dict) or set(paths) != {"bundle", "checkpoint", "engine", "decoder"}:
        raise ValueError("artifact paths differ")
    required = [PurePosixPath(_remote_path(p)) for p in paths.values()]
    bundle = pinned.read_pinned_json(spec["bundle_manifest_host"], config["artifact_pins"]["bundle"])
    required.extend(_bundle_artifact_paths(bundle, paths["bundle"]))
    if config.get("compiled_manifest") is not None or config.get("grn_evidence") is not None:
        raise ValueError("unexpected Tiger compiled or GRN evidence")
    if any(len(root.parts) < 5 or root not in required for root in roots) or any(p not in roots for p in required):
        raise ValueError("each declared candidate artifact needs an individual read-only mount")
    if not _is_direct_python_command(spec["runner_argv"]):
        raise ValueError("direct runner differs")
    exceptions = spec["rootfs_exceptions"]
    if (
        not isinstance(exceptions, list)
        or not all(isinstance(x, str) for x in exceptions)
        or len(set(exceptions)) != len(exceptions)
    ):
        raise ValueError("rootfs exceptions differ")
    signature = (_image_signature(spec["image_archive"]), _image_signature(spec["rootfs_export"]))
    cache_key = ("bwrap-comparison", str(Path(spec["image_archive"]).absolute()), spec["image_archive_sha256"],
                 str(Path(spec["rootfs_export"]).absolute()), spec["rootfs_export_sha256"],
                 spec["image_manifest_digest"], tuple(exceptions))
    cached = image_cache.get(cache_key) if image_cache is not None else None
    if cached is not None and cached[0] == signature:
        comparison = cached[1]
    else:
        comparison = bwrap_lineage.compare(
            spec["image_archive"], spec["image_manifest_digest"][7:], spec["rootfs_export"], allowed_differences=exceptions
        )
        if signature != (_image_signature(spec["image_archive"]), _image_signature(spec["rootfs_export"])):
            raise ValueError("image or rootfs changed during comparison")
        if image_cache is not None:
            image_cache[cache_key] = (signature, comparison)
    if comparison["rootfs_archive_sha256"] != spec["rootfs_export_sha256"]:
        raise ValueError("rootfs export changed")
    if session is not None:
        from passage_eval.campaign.direct.runner_binding import verify_g2_runner

        verify_g2_runner(
            session.grant,
            "tiger",
            config["procedure"],
            spec["runner_argv"],
            rootfs_export_sha256=comparison["rootfs_archive_sha256"],
            rootfs_exceptions=comparison["exceptions"],
        )
    return spec, config, comparison


def _verify_tiger_validation_archive(data, output, config, config_sha, count):
    from passage_eval.campaign.direct.direct_session import inspect_journal

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
    receipt = pinned.strict_json_loads(files["target-receipt.json"], name="target receipt")
    digest = files["target-receipt.sha256"].decode("ascii").strip()
    timing = receipt.get("timing") or {}
    identity = receipt.get("runtime_identity") or {}
    options = identity.get("options") or {}
    version = (identity.get("versions") or {}).get("openvino") or {}
    sdk = (version.get("distributions") or {}).get("openvino")
    if (
        hashlib.sha256(files["target-receipt.json"]).hexdigest() != digest
        or receipt.get("schema") != "passage-direct-target-validation-result-prototype/v1"
        or receipt.get("configuration_sha256") != config_sha
        or receipt.get("input_index_sha256") != config["index_sha256"]
        or receipt.get("native_artifact_hashes") != config["artifact_pins"]
        or receipt.get("qualification_status") != "exploratory_unadmitted"
        or receipt.get("scope_gpu_drain_verified") is not False
        or timing.get("status") != "complete"
        or timing.get("n_completed") != count
        or timing.get("warmups_completed") != 16
        or hashlib.sha256(files["journal.jsonl"]).hexdigest() != receipt.get("journal_sha256")
        or identity.get("implementation") != "passage_inference.backends.OpenVinoRuntime"
        or options.get("execution_devices") != ["GPU.0"]
        or options.get("device") not in ("GPU", "GPU.0")
        or not isinstance(sdk, str)
        or not sdk
        or not str(version.get("loaded_module_version", "")).startswith(sdk)
    ):
        raise ValueError("Tiger validation receipt or runtime differs")
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


def remote_command(spec, config_bytes, job):
    if not re.fullmatch(r"job:[0-9a-f]{16}", job):
        raise ValueError("job identity differs")
    if hashlib.sha256(config_bytes).hexdigest() != spec["configuration_sha256"]:
        raise ValueError("configuration bytes differ")
    scope = "passage-edge-tiger-" + job[4:] + ".scope"
    work = _remote_path(spec["target_work_root"]) + "/bwrap-session-" + job[4:]
    config = work + "/config.json"
    output = work + "/result"
    runner = [arg.replace("{config}", config).replace("{output}", output) for arg in spec["runner_argv"]]
    if not _is_direct_python_command(runner) or runner[-6:] != [
        "--config",
        config,
        "--config-sha256",
        spec["configuration_sha256"],
        "--output",
        output,
    ]:
        raise ValueError("runner binding differs")
    mounts = [x for path in spec["target_readonly_mounts"] for x in ("--ro-bind", path, path)]
    cmd = [
        "bwrap",
        "--unshare-user",
        "--unshare-pid",
        "--unshare-net",
        "--ro-bind",
        spec["target_rootfs"],
        "/",
        "--dev",
        "/dev",
        "--dev-bind",
        "/dev/dri",
        "/dev/dri",
        "--proc",
        "/proc",
        "--bind",
        work + "/scratch",
        "/tmp",
        "--tmpfs",
        "/home",
        "--clearenv",
        "--setenv",
        "HOME",
        "/tmp",
        "--setenv",
        "XDG_CACHE_HOME",
        "/tmp",
        "--setenv",
        "NUMBA_CACHE_DIR",
        "/tmp/numba",
        "--setenv",
        "PYTHONDONTWRITEBYTECODE",
        "1",
        "--setenv",
        "PYTHONNOUSERSITE",
        "1",
        "--setenv",
        "PATH",
        "/opt/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "--setenv",
        "LD_LIBRARY_PATH",
        "/opt/intel/openvino/runtime/3rdparty/tbb/lib:/opt/intel/openvino/runtime/lib/intel64:/opt/intel/openvino/tools/compile_tool",
        "--setenv",
        "INTEL_OPENVINO_DIR",
        "/opt/intel/openvino",
        "--setenv",
        "LIBVA_DRIVER_NAME",
        "iHD",
        "--setenv",
        "LIBVA_DRIVERS_PATH",
        "/usr/lib/x86_64-linux-gnu/dri",
        *mounts,
        "--bind",
        work,
        work,
        "--chdir",
        "/tmp",
        *runner,
    ]
    encoded = base64.b64encode(config_bytes).decode("ascii")
    script = (
        'set -eu; [ "$(cat /etc/machine-id)" = ' + _machine_id("tiger") + " ]; umask 077; "
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        "d="
        + shlex.quote(work)
        + '; mkdir -p -m 700 -- "$d"; exec 9>"$d/control.lock"; flock -x -w 45 9; test ! -e "$d/CANCELLED"; mkdir -m 700 -- "$d/scratch"; mkdir -m 700 -- "$d/scratch/numba"; set -C; printf %s '
        + shlex.quote(encoded)
        + ' | base64 -d > "$d/config.json"; set +C; '
        "systemd-run --user --scope --quiet --unit="
        + shlex.quote(scope)
        + " /usr/bin/timeout --signal=TERM --kill-after=20s "
        + str(int(spec["timeout_seconds"]))
        + "s "
        + shlex.join(cmd)
        + " <&0 >&2 9>&- & runner=$!; "
        "start=$SECONDS; while ! timeout 3s systemctl --user is-active --quiet " + shlex.quote(scope) + "; do "
        'if ! kill -0 "$runner" 2>/dev/null; then break; fi; '
        'if (( SECONDS - start >= 35 )); then kill -TERM "$runner" 2>/dev/null || true; wait "$runner" || true; touch "$d/FINISHED"; exit 125; fi; sleep 1; done; '
        'touch "$d/STARTED"; flock -u 9; exec 9>&-; '
        'if wait "$runner"; then rc=0; else rc=$?; fi; touch "$d/FINISHED"; '
        'if [ -d "$d/result" ]; then tar -cf - -C "$d/result" journal.jsonl target-receipt.json target-receipt.sha256; fi; exit "$rc"'
    )
    return script, scope, runner


def _stage_observers(spec, job):
    root = Path(__file__).resolve().parent
    remote = _remote_path(spec["target_work_root"]) + "/bwrap-probe-" + job[4:]
    sources = {name: (root / name).read_text() for name in ("bwrap_lineage.py", "bwrap_tree.py", "bwrap_scope.py")}
    sources["bwrap_tree.py"] = sources["bwrap_tree.py"].replace(
        "from passage_eval.campaign.direct.bwrap_lineage import", "from bwrap_lineage import"
    )
    writes = []
    for name, text in sources.items():
        writes.append(
            "printf %s "
            + shlex.quote(base64.b64encode(text.encode()).decode())
            + " | base64 -d > "
            + shlex.quote(remote + "/" + name)
        )
    script = (
        'set -eu; [ "$(cat /etc/machine-id)" = '
        + _machine_id("tiger")
        + " ]; umask 077; mkdir -m 700 -- "
        + shlex.quote(remote)
        + "; set -C; "
        + "; ".join(writes)
    )
    result = _ssh(spec, script)
    if result.returncode:
        raise RuntimeError("observer staging failed: " + result.stderr.decode(errors="replace")[-4000:])
    return remote


def _observe(spec, job, runner, remote, *, include_tree):
    program = (
        "import sys,json;sys.path.insert(0,"
        + repr(remote)
        + ");from bwrap_scope import observe;from bwrap_tree import compare_tree;"
        'p=observe("tiger",'
        + repr(job)
        + ","
        + repr(spec["target_rootfs"])
        + ","
        + repr(runner)
        + ");"
        + (
            "t=compare_tree("
            + repr(spec["target_rootfs"])
            + ","
            + repr(spec["target_rootfs_export"])
            + ","
            + repr(spec["rootfs_export_sha256"])
            + ");"
            if include_tree
            else "t=None;"
        )
        + 'print(json.dumps({"process":p,"tree":t},sort_keys=True))'
    )
    shell = (
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        + shlex.join(["/usr/bin/python3", "-I", "-B", "-c", program])
    )
    result = subprocess.run(
        _ssh_argv(spec, shell), capture_output=True, timeout=300 if include_tree else 60, check=False
    )
    if result.returncode:
        raise RuntimeError("bwrap observer failed: " + result.stderr.decode(errors="replace")[-4000:])
    if len(result.stdout) > 1_000_000:
        raise ValueError("observation too large")
    return json.loads(result.stdout)


def _drain(spec, scope):
    shell = (
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        "load=$(systemctl --user show " + shlex.quote(scope) + " --property=LoadState --value); "
        "state=$(systemctl --user show " + shlex.quote(scope) + " --property=ActiveState --value); "
        'case "$load:$state" in not-found:inactive|loaded:inactive|loaded:failed) ;; *) exit 71 ;; esac; '
        "cg=$(systemctl --user show " + shlex.quote(scope) + " --property=ControlGroup --value); "
        'if [ -n "$cg" ] && [ -d "/sys/fs/cgroup$cg" ]; then '
        'while IFS= read -r f; do [ -z "$(cat "$f")" ] || exit 72; done < <(find "/sys/fs/cgroup$cg" -name cgroup.procs -type f); fi; printf DRAINED'
    )
    result = _ssh(spec, shell)
    if result.returncode or result.stdout != b"DRAINED":
        raise RuntimeError("scope drain unproven")


def _cancel_and_drain(spec, job, scope):
    """Fence a late startup before stopping only this session's scope."""
    work = _remote_path(spec["target_work_root"]) + "/bwrap-session-" + job[4:]
    script = (
        'set -eu; [ "$(cat /etc/machine-id)" = '
        + _machine_id("tiger")
        + " ]; umask 077; d="
        + shlex.quote(work)
        + '; mkdir -p -m 700 -- "$d"; '
        'export XDG_RUNTIME_DIR="/run/user/$(id -u)" DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"; '
        'exec 9>"$d/control.lock"; flock -x -w 50 9; touch "$d/CANCELLED"; '
        "timeout 5s systemctl --user kill --kill-who=all --signal=TERM " + shlex.quote(scope) + " 2>/dev/null || true; "
        "sleep 2; timeout 5s systemctl --user kill --kill-who=all --signal=KILL "
        + shlex.quote(scope)
        + " 2>/dev/null || true; "
        "printf FENCED"
    )
    result = subprocess.run(_ssh_argv(spec, script), capture_output=True, timeout=80, check=False)
    if result.returncode or result.stdout != b"FENCED":
        raise RuntimeError("could not fence target startup")
    _drain(spec, scope)


def validation_finish(begin, receipt, spec, config, scope, job, journal, exit_code):
    """Bind validation completion to observed runner and durable journal."""
    if type(exit_code) is not int or exit_code != 0:
        raise ValueError("runner exit was not successful")
    if (
        hashlib.sha256(journal).hexdigest() != receipt.get("journal_sha256")
        or begin.get("scope") != scope
        or begin.get("job_id") != job
        or begin.get("configuration_sha256") != spec["configuration_sha256"]
        or begin.get("input_index_sha256") != config["index_sha256"]
    ):
        raise ValueError("validation session identity differs")
    markers = _journal_binding(journal, begin["namespace_pid"])
    if markers.get("journal_markers") != "matched":
        raise ValueError("validation runner session markers are missing")
    return {
        "container_digest": begin["container_digest"],
        "configuration_sha256": spec["configuration_sha256"],
        "input_index_sha256": config["index_sha256"],
        "journal_sha256": receipt["journal_sha256"],
        "scope": scope,
        "job_id": job,
        "runner_pid": begin["pid"],
        "n_processes": 1,
        "n_restarts": 0,
        "exit_code": exit_code,
        **markers,
    }


def _run_admitted(spec_path, spec_sha256, session):
    """Run exactly one admitted target session; stop and retain evidence on failure."""
    from passage_eval.campaign.direct.admitted_stream import stream_admitted_session, verify_rule8_completed_session

    spec, config, comparison = load_spec(spec_path, spec_sha256, session)
    prepared = None
    if session is None:
        prepared = pinned.read_pinned_json(
            _validation_host_path(Path(spec["prepared_root"]) / "prepared.json"), spec["prepared_sha256"]
        )
        if (
            prepared.get("split") != "validation"
            or prepared.get("count") != 128
            or prepared.get("index_sha256") != config["index_sha256"]
            or prepared.get("warmup_index") != config["warmup_index"]
        ):
            raise ValueError("prepared validation identity differs")
        _validation_provenance(spec, config, prepared)
    job = os.environ.get("AGENT_OPS_JOB", "")
    if not re.fullmatch(r"job:[0-9a-f]{16}", job):
        raise ValueError("CAVIAR job identity missing")

    lost = False

    def lease():
        nonlocal lost
        if lost:
            raise RuntimeError("lease previously lost")
        try:
            _lease(spec, job)
        except InterruptedError:
            raise
        except BaseException:
            lost = True
            raise

    lease()
    output = Path(spec["output_root"])
    output.mkdir(mode=0o700)
    _record(
        output / "dispatch.json", {"schema": "passage-tiger-bwrap-dispatch/v1", "job_id": job, "replay_allowed": False}
    )
    remote = _stage_observers(spec, job)
    script, scope, runner = remote_command(
        spec, pinned.read_pinned(spec["configuration"], spec["configuration_sha256"]), job
    )
    process = None
    failure = None
    errors = []
    cancel = threading.Event()
    thread = None
    with (output / "response.tar").open("xb") as stdout, (output / "remote.stderr").open("xb") as stderr:
        try:
            lease()
            process = subprocess.Popen(
                _ssh_argv(spec, script), stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, start_new_session=True
            )
            observed = _observe(spec, job, runner, remote, include_tree=True)
            lease()
            proof = {"comparison": comparison, "tree": observed["tree"], "process": observed["process"], "job_id": job}

            class LeasedStream:
                def write(self, data):
                    if cancel.is_set():
                        raise RuntimeError("lease unavailable")
                    return process.stdin.write(data)

                def flush(self):
                    if cancel.is_set():
                        raise RuntimeError("lease unavailable")
                    return process.stdin.flush()

            def sender():
                try:

                    def check_scope(pid, expected_scope):
                        lease()
                        current = _observe(spec, job, runner, remote, include_tree=False)["process"]
                        return current["scope"] == expected_scope and current["runner"]["pid"] == pid

                    if session is not None:
                        stream_admitted_session(
                            LeasedStream(),
                            session,
                            bwrap_inspect=lambda: proof,
                            process_in_scope=check_scope,
                            image_evidence_path=output / "image-start.json",
                        )
                    else:
                        from passage_eval.campaign.direct.rule8_bwrap import verify_running_bwrap

                        start = verify_running_bwrap(
                            proof,
                            spec["image_manifest_digest"],
                            config_sha256=spec["configuration_sha256"],
                            index_sha256=config["index_sha256"],
                            process_in_scope=check_scope,
                        )
                        start.update(
                            schema="passage-bwrap-validation-start/v1",
                            target="tiger",
                            procedure=config["procedure"],
                            seed=config["seed"],
                            mechanism="bwrap",
                            bwrap_proof=proof,
                        )
                        _record(output / "image-start.json", start)
                        _stream_prepared(LeasedStream(), Path(spec["prepared_root"]), spec["prepared_sha256"])
                except BaseException as exc:
                    errors.append(exc)
                finally:
                    try:
                        process.stdin.close()
                    except (ValueError, OSError):
                        pass

            thread = threading.Thread(target=sender, daemon=True)
            thread.start()
            deadline = time.monotonic() + spec["timeout_seconds"] + 10
            while process.poll() is None:
                lease()
                if errors:
                    raise RuntimeError("input sender failed") from errors[0]
                if time.monotonic() > deadline:
                    raise TimeoutError("session deadline")
                time.sleep(3)
            thread.join(timeout=10)
            if thread.is_alive() or errors or process.returncode:
                raise RuntimeError("session or input sender failed")
            lease()
        except BaseException as exc:
            failure = exc
            cancel.set()
        finally:
            try:
                lease()
            except BaseException as exc:
                lost = True
                failure = exc
                cancel.set()
            if process and failure is not None and not lost:
                try:
                    _cancel_and_drain(spec, job, scope)
                except BaseException as exc:
                    failure = RuntimeError("scope cancellation or drain unproven")
                    failure.__cause__ = exc
            if process and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait()
            if thread:
                thread.join(timeout=10)
                if thread.is_alive():
                    failure = RuntimeError("input producer did not stop")
            stdout.flush()
            os.fsync(stdout.fileno())
            stderr.flush()
            os.fsync(stderr.fileno())
    if lost:
        raise RuntimeError("lease lost; remote scope state unknown") from failure
    lease()
    _drain(spec, scope)
    if failure:
        raise failure
    if session is None:
        receipt = _verify_tiger_validation_archive(
            (output / "response.tar").read_bytes(),
            output / "collected",
            config,
            spec["configuration_sha256"],
            prepared["count"],
        )
    else:
        receipt = _verify_archive(
            (output / "response.tar").read_bytes(),
            output / "collected",
            config,
            spec["configuration_sha256"],
            None,
            len(session.tasks),
            admitted_session=session,
        )
    drain = {
        "schema": "passage-direct-session-drain/v1",
        "target": "tiger",
        "procedure": config["procedure"],
        "job_id": job,
        "scope": scope,
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    _record(output / "drain.json", drain)
    start = output / "image-start.json"
    if session is not None:
        verified = verify_rule8_completed_session(
            session, output / "collected", start, _file_digest(start), container_inspect=None, drain=drain, job_id=job
        )
    else:
        begin = pinned.read_pinned_json(start, _file_digest(start))
        verified = validation_finish(
            begin,
            receipt,
            spec,
            config,
            scope,
            job,
            (output / "collected" / "journal.jsonl").read_bytes(),
            process.returncode,
        )
    # The SSH process returns the actual systemd-run/bwrap exit status.
    verified["exit_code"] = process.returncode
    _record(output / "image-finish.json", verified)
    return {
        "job_id": job,
        "target": "tiger",
        "procedure": config["procedure"],
        "status": "admitted_session_collected" if session is not None else "exploratory_validation_collected",
        "journal_sha256": receipt["journal_sha256"],
    }


def run_admitted(spec_path, spec_sha256, session):
    """Bound cancellation while preserving uncertain leases and all artifacts."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("relay must run in the main thread")
    old_term, old_alarm = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGALRM)
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError("relay requires an unused process alarm")

    def interrupt(signum, frame):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGALRM, signal.SIG_IGN)
        raise InterruptedError("relay interrupted")

    signal.signal(signal.SIGTERM, interrupt)
    signal.signal(signal.SIGALRM, interrupt)
    signal.setitimer(signal.ITIMER_REAL, 1050)
    try:
        return _run_admitted(spec_path, spec_sha256, session)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGTERM, old_term)
        signal.signal(signal.SIGALRM, old_alarm)


def main():
    """Prepare a single admitted session, then dispatch it under the existing lease."""
    import argparse
    import io

    from passage_eval.campaign.direct import persistent_transport
    from passage_eval.campaign.direct.admitted_stream import bind_target_config, prepare_admitted_session
    from passage_eval.campaign.direct.rule8_scope import load_rule8_scope
    from passage_eval.io.roster import validate_roster
    from passage_eval.protocol.contracts import BudgetSpec

    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in (
        "record",
        "evidence-dir",
        "roster",
        "dataset-root",
        "warmup-npz",
        "warmup-npz-sha256",
        "scope",
        "scope-sha256",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    admission_args = (
        args.record,
        args.evidence_dir,
        args.roster,
        args.dataset_root,
        args.warmup_npz,
        args.warmup_npz_sha256,
        args.scope,
        args.scope_sha256,
    )
    if not any(admission_args):
        print(json.dumps(run_admitted(args.spec, args.sha256, None), sort_keys=True), flush=True)
        return
    if not all(admission_args):
        parser.error("admitted mode requires all admission and scope arguments")
    spec = pinned.read_pinned_json(_validation_host_path(args.spec), args.sha256)
    config = validate_config(
        pinned.read_pinned_json(_validation_host_path(spec["configuration"]), spec["configuration_sha256"])
    )
    budget = BudgetSpec(**config["budget"])
    warm_roster = validate_roster(
        pinned.read_pinned_json(_validation_host_path(spec["warmup_roster"]), spec["warmup_roster_sha256"])
    )
    frame = io.BytesIO()
    persistent_transport.write_array_frame(
        frame, "warmup", 0, _validation_host_path(args.warmup_npz), args.warmup_npz_sha256
    )
    frame.seek(0)
    warmup = persistent_transport.read_warmup(frame, budget, config["warmup_index"])[0]
    scope = load_rule8_scope(args.scope, args.scope_sha256)
    session = prepare_admitted_session(
        args.record, args.evidence_dir, args.roster, args.dataset_root, budget, warmup, warm_roster, rule8_scope=scope
    )
    session = bind_target_config(session, config, compiled_host_path=spec["compiled_manifest_host"])
    print(json.dumps(run_admitted(args.spec, args.sha256, session), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
