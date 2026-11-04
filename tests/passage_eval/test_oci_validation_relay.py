"""Local validation-only OCI relay checks; no daemon, target, split test or cleanup."""

import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from passage_eval.campaign.direct import oci_validation_relay as relay
from passage_eval.campaign.direct.direct_target import SCHEMA


def write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    with path.open("xb") as stream:
        stream.write(raw)
    return hashlib.sha256(raw).hexdigest()


def fixture(root):
    root.mkdir()
    known = root / "known-hosts"
    known.write_bytes(b"synthetic-host-key")
    engine = "1" * 64
    compiled = root / "compiled.json"
    compiled_sha = write(
        compiled, {"engine": "engine", "engine_sha256": engine, "runtime_versions": {"tensorrt": "10.16.2.10"}}
    )
    bundle = root / "bundle.json"
    bundle_sha = write(bundle, {"artifacts": {"model.onnx": "c" * 64}})
    config = {
        "schema": SCHEMA,
        "dataset_split": "validation",
        "target": "orin",
        "procedure": "reg",
        "seed": 2026,
        "artifact_paths": {
            "bundle": "/var/lib/passage/payload/bundle.json",
            "checkpoint": "/var/lib/passage/payload/checkpoint",
            "engine": "/var/lib/passage/payload/engine",
            "decoder": "/var/lib/passage/payload/decoder",
        },
        "artifact_pins": {"engine": engine, "bundle": bundle_sha},
        "compiled_manifest": "/var/lib/passage/payload/compiled.json",
        "compiled_sha256": compiled_sha,
        "exploratory_unadmitted_precision": True,
        "budget": {},
        "warmup_index": [{"sample_id": "warm", "raw_input_sha256": "a" * 64}],
        "index_sha256": "b" * 64,
        "checkpoint_size": 32,
        "grn_evidence": None,
    }
    config_path = root / "config.json"
    config_sha = write(config_path, config)
    prepared = root / "prepared"
    prepared.mkdir()
    prepared_sha = write(
        prepared / "prepared.json",
        {"split": "validation", "count": 1, "index_sha256": "b" * 64, "warmup_index": config["warmup_index"]},
    )
    manifest = json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "c" * 64}}).encode()
    image_digest = "sha256:" + hashlib.sha256(manifest).hexdigest()
    index = json.dumps({"manifests": [{"digest": image_digest}]}).encode()
    archive = root / "image.tar"
    with tarfile.open(archive, "w") as tar:
        for name, data in [("index.json", index), ("blobs/sha256/" + image_digest[7:], manifest)]:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    image_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    spec = {
        "schema": "passage-oci-validation-relay/v1",
        "target": "orin",
        "known_hosts": str(known),
        "known_hosts_sha256": hashlib.sha256(known.read_bytes()).hexdigest(),
        "image_archive": str(archive),
        "image_archive_sha256": image_sha,
        "image_manifest_digest": image_digest,
        "prepared_root": str(prepared),
        "prepared_sha256": prepared_sha,
        "configuration": str(config_path),
        "configuration_sha256": config_sha,
        "compiled_manifest_host": str(compiled),
        "bundle_manifest_host": str(bundle),
        "target_work_root": "/var/lib/passage/runs",
        "target_readonly_mounts": [
            f"/var/lib/passage/payload/{name}"
            for name in ("bundle.json", "checkpoint", "engine", "decoder", "compiled.json", "model.onnx")
        ],
        "runner_argv": [
            "/usr/bin/python3",
            "-I",
            "-B",
            "/opt/passage-direct/direct_target.py",
            "--config",
            "{config}",
            "--config-sha256",
            config_sha,
            "--output",
            "{output}",
        ],
        "groups": [44, 993],
        "user": "1000:1000",
        "output_root": str(root / "output"),
        "cavops": "/usr/bin/cavops",
        "timeout_seconds": 1000,
        "replica_catalog": None,
        "replica_catalog_sha256": None,
        "validation_roster": None,
        "validation_roster_sha256": None,
        "warmup_roster": None,
        "warmup_roster_sha256": None,
    }
    spec_path = root / "spec.json"
    spec_sha = write(spec_path, spec)
    return spec_path, spec_sha, spec, config, manifest


@pytest.fixture(autouse=True)
def portable_host_configuration(monkeypatch):
    for target in ("ORIN", "THOR", "TIGER"):
        monkeypatch.setenv("PASSAGE_EDGE_MACHINE_ID_" + target, "a" * 32)
        monkeypatch.setenv("PASSAGE_EDGE_SSH_HOST_" + target, "edge.example.invalid")


def test_validation_spec_builds_persistent_scratch_and_job_scope(tmp_path):
    path, sha, spec, config, manifest = fixture(tmp_path / "fixtures")
    loaded, observed, sdk, blob = relay.load_spec(path, sha)
    assert observed == config and sdk == "10.16.2.10" and blob == manifest
    command, scope, slice_name, container, argv = relay._remote_command(
        loaded, path.read_bytes(), "job:0123456789abcdef"
    )
    assert scope == "passage-edge-orin-0123456789abcdef.scope"
    assert slice_name == "passage-edge-orin-0123456789abcdef.slice"
    assert (
        "--cgroup-parent" in command and "dst=/tmp" in command and "--tmpfs" not in command and "--rm " not in command
    )
    assert container in command
    assert argv[-6] == "--config" and argv[-4] == "--config-sha256" and argv[-2] == "--output"


def test_test_scope_rejected_before_reading_validation_payload(tmp_path, monkeypatch):
    path, sha, spec, _, _ = fixture(tmp_path / "fixtures")
    value = json.loads(Path(spec["configuration"]).read_bytes())
    value["dataset_split"] = "test"
    changed = tmp_path / "wrong-config.json"
    spec["configuration_sha256"] = write(changed, value)
    spec["configuration"] = str(changed)
    new = tmp_path / "wrong-spec.json"
    new_sha = write(new, spec)
    original = relay._file_digest

    def checked(path):
        if Path(path).name == "prepared.json":
            pytest.fail("prepared input read before split guard")
        return original(path)

    monkeypatch.setattr(relay, "_file_digest", checked)
    with pytest.raises(ValueError, match="validation"):
        relay.load_spec(new, new_sha)


def test_uuid_pid_markers_are_distinct_from_journal_hash_and_must_match():
    sid = "a" * 32

    def line(value):
        return json.dumps(value).encode() + b"\n"

    old = line({"kind": "session_start"}) + line({"kind": "session_complete"})
    assert relay._journal_binding(old, 1) == {"journal_markers": "unavailable_in_exploratory_runner"}
    data = line({"kind": "session_start", "session_id": sid, "namespace_pid": 1}) + line(
        {"kind": "session_complete", "session_id": sid, "namespace_pid": 1}
    )
    got = relay._journal_binding(data, 1)
    assert got == {"runner_session_uuid": sid, "journal_markers": "matched"} and sid != hashlib.sha256(data).hexdigest()
    with pytest.raises(ValueError, match="markers"):
        relay._journal_binding(data, 2)
    with pytest.raises(ValueError, match="one session"):
        relay._journal_binding(data + line({"kind": "session_complete"}), 1)


@pytest.mark.parametrize("sender_fails", [False, True])
@pytest.mark.parametrize("interrupt_drain", [None, 1, 2, 14, 15])
def test_synthetic_host_sequence_observes_image_before_input_and_matches_uuid(
    tmp_path, monkeypatch, sender_fails, interrupt_drain
):
    path, sha, spec, config, manifest = fixture(tmp_path / "fixtures")
    monkeypatch.setenv("AGENT_OPS_JOB", "job:0123456789abcdef")
    monkeypatch.setattr(relay, "_lease", lambda *args: None)
    monkeypatch.setattr(relay, "_validation_provenance", lambda *args: None)
    _, _, _, _, runner = relay._remote_command(spec, path.read_bytes(), "job:0123456789abcdef")
    cid = "b" * 64
    image = spec["image_manifest_digest"]
    running = {
        "Id": cid,
        "Image": image,
        "Path": runner[0],
        "Args": runner[1:],
        "RestartCount": 0,
        "State": {"Running": True, "Pid": 42, "StartedAt": "2026-11-04T00:00:00Z"},
    }

    class Input(io.BytesIO):
        def __init__(self, parent):
            super().__init__()
            self.parent = parent

        def close(self):
            self.parent.finished = True
            super().close()

    class Process:
        def __init__(self, *args, **kwargs):
            self.finished = False
            self.returncode = None
            self.pid = 42
            self.stdin = Input(self)

        def poll(self):
            if self.finished:
                self.returncode = 0
            return self.returncode

    process = Process()
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *args, **kwargs: process)

    class Result:
        def __init__(self, out=b""):
            self.returncode = 0
            self.stdout = out

    def ssh(_spec, command):
        if command.startswith("sudo -n docker inspect "):
            state = (
                running
                if not process.finished
                else {**running, "State": {"Running": False, "ExitCode": 0, "StartedAt": "2026-11-04T00:00:00Z"}}
            )
            return Result(json.dumps([state]).encode())
        if command == "cat /proc/42/cgroup":
            return Result(("0::/system.slice/docker-" + cid + ".scope\n").encode())
        if command == "cat /proc/42/status":
            return Result(b"Name: python3\nNSpid:\t42\t1\n")
        if "DRAINED" in command:
            if interrupt_drain:
                import os

                os.kill(os.getpid(), interrupt_drain)
            return Result(b"DRAINED")
        raise AssertionError(command)

    monkeypatch.setattr(relay, "_ssh", ssh)
    monkeypatch.setattr(relay.time, "sleep", lambda _: None)

    def send(stream, *args):
        assert (Path(spec["output_root"]) / "image-start.json").is_file()
        if sender_fails:
            raise ValueError("synthetic sender failure")
        stream.write(b"synthetic-only")

    monkeypatch.setattr(relay, "_stream_prepared", send)

    def verify(_data, out, *args, **kwargs):
        out.mkdir()
        events = [
            {"kind": "session_start", "session_id": "c" * 32, "namespace_pid": 1},
            {"kind": "session_complete", "session_id": "c" * 32, "namespace_pid": 1},
        ]
        raw = b"".join(json.dumps(x).encode() + b"\n" for x in events)
        (out / "journal.jsonl").write_bytes(raw)
        return {"journal_sha256": hashlib.sha256(raw).hexdigest()}

    monkeypatch.setattr(relay, "_verify_archive", verify)
    if sender_fails:
        with pytest.raises(RuntimeError, match="sender failed"):
            relay.run_validation(path, sha)
        assert process.stdin.closed
        return
    result = relay.run_validation(path, sha)
    finish = json.loads((Path(spec["output_root"]) / "image-finish.json").read_bytes())
    assert result["status"] == "exploratory_validation_collected"
    assert (
        finish["runner_session_uuid"] == "c" * 32
        and finish["runner_pid"] == 42
        and finish["journal_markers"] == "matched"
    )


@pytest.mark.parametrize("seed", [1234, 3407])
@pytest.mark.parametrize("procedure", ["seg", "reg"])
def test_replica_validation_requires_catalogue_and_exact_checkpoint(tmp_path, seed, procedure):
    from passage_eval.campaign.direct.direct_target import REPLICA_VALIDATION_SCHEMA
    from passage_eval.protocol.replicas import load_replica_catalog, replica_identity

    path, sha, spec, config, _ = fixture(tmp_path / "fixtures")
    catalog = Path(__file__).resolve().parents[2] / "docs/benchmark/replica-candidates-256.json"
    catalog_sha = hashlib.sha256(catalog.read_bytes()).hexdigest()
    identity = replica_identity(load_replica_catalog(catalog, catalog_sha), procedure, seed)
    config.update(
        schema=REPLICA_VALIDATION_SCHEMA,
        seed=seed,
        procedure=procedure,
        artifact_pins={**config["artifact_pins"], "checkpoint": identity["checkpoint_sha256"]},
    )
    if procedure == "seg":
        spec["target_readonly_mounts"].extend(f"/var/lib/passage/payload/{name}" for name in ("a", "b", "c"))
        config["grn_evidence"] = {
            "rewrite_script": "/var/lib/passage/payload/a",
            "input_archive": "/var/lib/passage/payload/b",
            "result_archive": "/var/lib/passage/payload/c",
            "reproduction_root": "/tmp",
        }
    config_path = tmp_path / "replica-config.json"
    config_sha = write(config_path, config)
    spec.update(
        configuration=str(config_path),
        configuration_sha256=config_sha,
        replica_catalog=str(catalog),
        replica_catalog_sha256=catalog_sha,
    )
    spec_path = tmp_path / "replica-spec.json"
    spec_sha = write(spec_path, spec)
    assert relay.load_spec(spec_path, spec_sha)[1]["seed"] == seed
    wrong = {**config, "artifact_pins": {**config["artifact_pins"], "checkpoint": "f" * 64}}
    wrong_config = tmp_path / "wrong-config.json"
    wrong_sha = write(wrong_config, wrong)
    wrong_spec = {**spec, "configuration": str(wrong_config), "configuration_sha256": wrong_sha}
    wrong_path = tmp_path / "wrong-spec.json"
    wrong_pin = write(wrong_path, wrong_spec)
    with pytest.raises(ValueError, match="checkpoint differs"):
        relay.load_spec(wrong_path, wrong_pin)
    missing = {**spec, "replica_catalog": None, "replica_catalog_sha256": None}
    missing_path = tmp_path / "missing-spec.json"
    missing_pin = write(missing_path, missing)
    with pytest.raises(ValueError, match="catalogue pin"):
        relay.load_spec(missing_path, missing_pin)


def test_queue_refuses_multiple_sessions_in_one_job(tmp_path):
    with pytest.raises(ValueError, match="one drained session per job"):
        relay.run_validation_queue(tmp_path / "queue.json", "a" * 64)


def test_lease_loss_does_not_touch_remote_target(tmp_path, monkeypatch):
    path, sha, _, _, _ = fixture(tmp_path / "fixtures")
    monkeypatch.setenv("AGENT_OPS_JOB", "job:0123456789abcdef")
    monkeypatch.setattr(relay, "_validation_provenance", lambda *args: None)
    calls = []

    def lease(*args):
        calls.append("lease")
        if len(calls) > 1:
            raise RuntimeError("expired")

    monkeypatch.setattr(relay, "_lease", lease)

    class Process:
        stdin = io.BytesIO()
        pid = 42

        def poll(self):
            return None

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(relay.os, "killpg", lambda *args: None)
    monkeypatch.setattr(relay.subprocess, "Popen", lambda *args, **kwargs: Process())
    monkeypatch.setattr(relay, "_ssh", lambda *args: pytest.fail("remote command after lease loss"))
    monkeypatch.setattr(relay, "_drain", lambda *args: pytest.fail("remote drain after lease loss"))
    with pytest.raises(RuntimeError, match="lease lost"):
        relay.run_validation(path, sha)


def test_validation_provenance_pins_rosters_and_checks_128_inputs(tmp_path, monkeypatch):
    from passage_eval.campaign.direct import raw_identity
    from passage_eval.io import roster as roster_io

    measured = [{"sample_id": str(i), "input_identity": f"{i:064x}"} for i in range(128)]
    warm = [{"sample_id": "warm", "input_identity": "f" * 64}]
    main_roster = {"dataset": {"split": "validation"}, "tasks": measured}
    warm_roster = {"dataset": {"split": "validation"}, "tasks": warm}
    main_file, warm_file = tmp_path / "main.json", tmp_path / "warm.json"
    main_sha, warm_sha = write(main_file, main_roster), write(warm_file, warm_roster)
    monkeypatch.setattr(roster_io, "validate_roster", lambda value: value)
    monkeypatch.setattr(
        raw_identity, "task_from_full", lambda sid, identity: {"sample_id": sid, "raw_input_sha256": identity}
    )
    rows = [raw_identity.task_from_full(t["sample_id"], t["input_identity"]) for t in measured]
    raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows)
    root = tmp_path / "inputs"
    root.mkdir()
    (root / "index.jsonl").write_bytes(raw)
    (root / "warmup.npz").write_bytes(b"synthetic")
    warm_rows = [raw_identity.task_from_full(t["sample_id"], t["input_identity"]) for t in warm]
    spec = {
        "validation_roster": str(main_file),
        "validation_roster_sha256": main_sha,
        "warmup_roster": str(warm_file),
        "warmup_roster_sha256": warm_sha,
        "prepared_root": str(root),
    }
    prepared = {"count": 128, "index_sha256": hashlib.sha256(raw).hexdigest(), "warmup_index": warm_rows}
    config = {"warmup_index": warm_rows}
    relay._validation_provenance(spec, config, prepared)
    with pytest.raises(ValueError, match="128 scenes"):
        relay._validation_provenance(spec, config, {**prepared, "count": 127})
    with pytest.raises(ValueError):
        relay._validation_provenance({**spec, "validation_roster_sha256": "0" * 64}, config, prepared)
    with pytest.raises(ValueError, match="projection"):
        relay._validation_provenance(spec, {"warmup_index": rows[:1]}, prepared)


def test_admitted_spec_requires_bound_configuration(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from passage_eval.campaign.direct import admitted_stream
    from passage_eval.campaign.direct.direct_target import RULE8_SCHEMA

    _, _, spec, config, _ = fixture(tmp_path / "fixtures")
    config.update(schema=RULE8_SCHEMA, dataset_split="test", exploratory_unadmitted_precision=False)
    config_path = tmp_path / "admitted-config.json"
    config_sha = write(config_path, config)
    spec.update(configuration=str(config_path), configuration_sha256=config_sha)
    spec_path = tmp_path / "admitted-spec.json"
    spec_sha = write(spec_path, spec)
    with pytest.raises(ValueError, match="only the target validation"):
        relay.load_spec(spec_path, spec_sha)
    raw = config_path.read_bytes()
    session = SimpleNamespace(config_bytes=raw, rule8_scope=None)
    monkeypatch.setattr(
        admitted_stream, "bind_target_config", lambda *_args, **_kwargs: SimpleNamespace(config_bytes=raw)
    )
    assert relay.load_spec(spec_path, spec_sha, admitted_session=session)[1]["dataset_split"] == "test"
    with pytest.raises(ValueError, match="not bound"):
        relay.load_spec(
            spec_path, spec_sha, admitted_session=SimpleNamespace(config_bytes=b"different", rule8_scope=None)
        )


def test_relay_signal_runs_cleanup_and_restores_handlers(monkeypatch):
    import os
    import signal

    events = []
    old = signal.getsignal(signal.SIGTERM)

    def interrupted(*args, **kwargs):
        try:
            os.kill(os.getpid(), signal.SIGTERM)
        finally:
            events.append("cleanup")

    monkeypatch.setattr(relay, "_run_validation", interrupted)
    with pytest.raises(InterruptedError):
        relay.run_validation("unused", "a" * 64)
    assert events == ["cleanup"]
    assert signal.getsignal(signal.SIGTERM) == old
    assert signal.getitimer(signal.ITIMER_REAL) == (0.0, 0.0)


def test_broad_home_mount_is_rejected(tmp_path):
    _, _, spec, _, _ = fixture(tmp_path / "fixtures")
    spec["target_readonly_mounts"] = ["/srv/passage-user"]
    path = tmp_path / "wide.json"
    sha = write(path, spec)
    with pytest.raises(ValueError, match="read-only mount"):
        relay.load_spec(path, sha)


def test_lease_loss_unblocks_local_sender_without_remote_cleanup(tmp_path, monkeypatch):
    import subprocess
    import sys
    import threading

    path, sha, _, _, _ = fixture(tmp_path / "fixtures")
    monkeypatch.setenv("AGENT_OPS_JOB", "job:0123456789abcdef")
    monkeypatch.setattr(relay, "_validation_provenance", lambda *args: None)
    sending = threading.Event()
    lost = threading.Event()

    def lease(*args):
        if sending.is_set():
            lost.set()
            raise RuntimeError("lost")

    monkeypatch.setattr(relay, "_lease", lease)
    original_popen = subprocess.Popen
    spawned = []

    def popen(*args, **kwargs):
        proc = original_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(relay.subprocess, "Popen", popen)
    cid = "b" * 64
    scope = "/system.slice/docker-" + cid + ".scope"
    monkeypatch.setattr(relay, "_container_cgroup", lambda *_args: scope)
    monkeypatch.setattr(relay.rule8_image, "verify_running_container", lambda *_args, **_kwargs: {})

    class Result:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    def ssh(_spec, command):
        if lost.is_set():
            pytest.fail("remote call after lease loss")
        if command.startswith("sudo -n docker inspect"):
            return Result(json.dumps([{"Id": cid, "State": {"Running": True, "Pid": spawned[0].pid}}]).encode())
        if command.startswith("cat /proc/"):
            return Result(f"NSpid:\t{spawned[0].pid}\t1\n".encode())
        pytest.fail("unexpected remote command")

    monkeypatch.setattr(relay, "_ssh", ssh)

    def send(stream, *_args):
        sending.set()
        stream.write(b"x" * (16 * 1024 * 1024))

    monkeypatch.setattr(relay, "_stream_prepared", send)
    with pytest.raises(RuntimeError, match="lease lost"):
        relay.run_validation(path, sha)
    assert spawned[0].poll() is not None
    assert lost.is_set()


def test_preparation_time_is_charged_to_global_budget(monkeypatch):
    monkeypatch.setattr(relay.time, "monotonic", lambda: 1200.0)
    monkeypatch.setattr(relay, "_run_validation", lambda *args, **kwargs: pytest.fail("session started too late"))
    with pytest.raises(TimeoutError, match="preparation exhausted"):
        relay.run_validation("unused", "a" * 64, deadline_monotonic=1100.0)


@pytest.mark.parametrize("exit_race", [False, True])
def test_stop_timeout_unblocks_sender_and_records_unproven_drain(tmp_path, monkeypatch, exit_race):
    import os
    import signal
    import subprocess
    import sys
    import threading

    path, sha, spec, _, _ = fixture(tmp_path / "fixtures")
    monkeypatch.setenv("AGENT_OPS_JOB", "job:0123456789abcdef")
    monkeypatch.setattr(relay, "_validation_provenance", lambda *args: None)
    monkeypatch.setattr(relay, "_lease", lambda *args: None)
    if exit_race:
        killpg = os.killpg

        def exiting_killpg(pid, signum):
            killpg(pid, signum)
            raise ProcessLookupError("process exited after poll")

        monkeypatch.setattr(relay.os, "killpg", exiting_killpg)
    original_popen = subprocess.Popen
    spawned = []

    def popen(*args, **kwargs):
        proc = original_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdin=subprocess.PIPE,
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
            start_new_session=True,
        )
        spawned.append(proc)
        return proc

    monkeypatch.setattr(relay.subprocess, "Popen", popen)
    cid = "b" * 64
    scope = "/system.slice/docker-" + cid + ".scope"
    monkeypatch.setattr(relay, "_container_cgroup", lambda *_args: scope)
    monkeypatch.setattr(relay.rule8_image, "verify_running_container", lambda *_args, **_kwargs: {})

    class Result:
        returncode = 0

        def __init__(self, out):
            self.stdout = out

    def ssh(_spec, command):
        if command.startswith("sudo -n docker inspect"):
            return Result(json.dumps([{"Id": cid, "State": {"Running": True, "Pid": spawned[0].pid}}]).encode())
        if command.startswith("cat /proc/"):
            return Result(f"NSpid:\t{spawned[0].pid}\t1\n".encode())
        if command.startswith("sudo -n docker stop"):
            raise subprocess.TimeoutExpired(command, 1)
        pytest.fail("unexpected remote command")

    monkeypatch.setattr(relay, "_ssh", ssh)
    monkeypatch.setattr(relay, "_drain", lambda *_args: (_ for _ in ()).throw(RuntimeError("not proven")))

    def send(stream, *_args):
        threading.Timer(0.1, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
        stream.write(b"x" * (16 * 1024 * 1024))

    monkeypatch.setattr(relay, "_stream_prepared", send)
    with pytest.raises(RuntimeError, match="not proven"):
        relay.run_validation(path, sha)
    assert spawned[0].poll() is not None
    receipt = json.loads((Path(spec["output_root"]) / "failure.json").read_bytes())
    assert receipt["remote_state"] == "unknown" and receipt["reason"] == "drain_unproven"


def test_connection_settings_require_explicit_host_and_machine_identity(monkeypatch):
    from passage_eval.campaign.direct import oci_validation_relay as relay

    monkeypatch.delenv("PASSAGE_EDGE_SSH_HOST_THOR", raising=False)
    monkeypatch.delenv("PASSAGE_EDGE_MACHINE_ID_THOR", raising=False)
    with pytest.raises(ValueError, match="SSH host"):
        relay._ssh_argv({"target": "thor", "known_hosts": "/tmp/known-hosts"}, "true")
    with pytest.raises(ValueError, match="machine identity"):
        relay._machine_id("thor")
    monkeypatch.setenv("PASSAGE_EDGE_SSH_HOST_THOR", "edge.example.invalid")
    monkeypatch.setenv("PASSAGE_EDGE_MACHINE_ID_THOR", "a" * 32)
    assert "edge.example.invalid" in relay._ssh_argv({"target": "thor", "known_hosts": "/tmp/known-hosts"}, "true")
    assert relay._machine_id("thor") == "a" * 32
    monkeypatch.setenv("PASSAGE_EDGE_SSH_HOST_THOR", "-oProxyCommand=bad")
    with pytest.raises(ValueError, match="SSH host"):
        relay._ssh_argv({"target": "thor", "known_hosts": "/tmp/known-hosts"}, "true")
