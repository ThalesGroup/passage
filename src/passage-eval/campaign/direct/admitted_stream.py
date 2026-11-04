"""Host-side admitted S index and bounded input-only stream production."""

from __future__ import annotations

import hashlib
import io
import json
import os
import struct
from pathlib import Path
from dataclasses import dataclass, replace

import numpy as np

from passage_eval.campaign.direct.admitted_input import admitted_requests
from passage_eval.campaign.direct.persistent_transport import _array_payload, _write_all, read_task_index, write_end
from passage_eval.campaign.direct.raw_identity import raw_digest, task_from_full
from passage_eval.io import pinned
from passage_eval.io import roster as roster_io
from passage_eval.protocol.admission import DESCRIPTIVE_PROFILES, RULE8_NATIVE_PROFILE, grant_test_admission
from passage_eval.protocol.guards import refuse_test_roster


@dataclass(frozen=True)
class AdmittedSession:
    """Host-only binding: never serialize the grant or the canonical roster."""

    record_path: object
    evidence_dir: object
    roster_path: object
    dataset_root: object
    budget: object
    grant: object
    index: bytes
    index_sha256: str
    tasks: tuple
    warmup: object
    warmup_index: tuple
    warmup_roster_sha256: str
    config_bytes: bytes = b""
    rule8_scope: object = None
    compiled_host_path: object = None


def prepare_admitted_session(
    record_path, evidence_dir, roster_path, dataset_root, budget, warmup, warmup_roster, *, rule8_scope=None
):
    """Check all admission gates before opening the S roster; do not read dataset rows."""
    grant = grant_test_admission(record_path, evidence_dir)
    if grant.report.admitted.get("profile") == RULE8_NATIVE_PROFILE:
        from passage_eval.campaign.direct.rule8_scope import Rule8Scope

        if not isinstance(rule8_scope, Rule8Scope):
            raise ValueError("Rule8 requires the pinned global plan before opening the roster")
        rule8_scope.check_record(grant)
    elif rule8_scope is not None:
        raise ValueError("global Rule8 plan cannot be applied to a descriptive grant")
    roster = roster_io.load_roster(roster_path, expected_sha256=grant.admissible_roster_sha256)
    refuse_test_roster(
        roster, allow_test=True, admission=grant, require_admitted_roster=True, context="the direct native session"
    )
    if roster["dataset"]["split"] != "test" or len(roster["tasks"]) != 1024:
        raise ValueError("direct campaign requires exactly the admitted S roster of 1024 tasks")
    warmup_roster = roster_io.validate_roster(warmup_roster)
    refuse_test_roster(warmup_roster, context="the direct native warmup")
    if warmup_roster["dataset"]["split"] != "validation" or len(warmup_roster["tasks"]) != 1:
        raise ValueError("warmup must be one rostered validation scene")
    if warmup.metadata or warmup.input_tensor is not None:
        raise ValueError("warmup must contain only raw inputs")
    tasks = tuple(task_from_full(t["sample_id"], t["input_identity"]) for t in roster["tasks"])
    warm = {"sample_id": warmup.sample_id, "raw_input_sha256": raw_digest(warmup)}
    warm_task = warmup_roster["tasks"][0]
    if warm != task_from_full(warm_task["sample_id"], warm_task["input_identity"]):
        raise ValueError("warmup input differs from its validation roster")
    if any(t["sample_id"] == warm["sample_id"] or t["raw_input_sha256"] == warm["raw_input_sha256"] for t in tasks):
        raise ValueError("warmup must be disjoint from S by ID and input content")
    raw = b"".join(json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in tasks)
    digest = hashlib.sha256(raw).hexdigest()
    read_task_index(io.BytesIO(struct.pack("!4sIQ", b"PSI1", 0, len(raw)) + raw), digest)
    grant.ensure_unchanged()
    return AdmittedSession(
        record_path,
        evidence_dir,
        roster_path,
        dataset_root,
        budget,
        grant,
        raw,
        digest,
        tasks,
        warmup,
        (warm,),
        warmup_roster["roster_sha256"],
        rule8_scope=rule8_scope,
    )


def _runtime_expectation(environment, target):
    """Read the SDK version pinned by the host evidence, without guessing from prose."""
    value = environment.get("runtime_expectation")
    sdk = "openvino" if target == "tiger" else "tensorrt"
    if (
        not isinstance(value, dict)
        or set(value) != {"sdk", "version"}
        or value["sdk"] != sdk
        or not isinstance(value["version"], str)
        or not value["version"].strip()
    ):
        raise ValueError("pinned native SDK expectation is missing or differs")
    return value


def _verify_runtime_identity(identity, expectation, target, engine_sha256):
    """Match observable SDK, accelerator and, where reported, loaded engine hash."""
    if not isinstance(identity, dict):
        raise ValueError("observed runtime identity is missing")
    sdk = expectation["sdk"]
    options = identity.get("options")
    versions = identity.get("versions")
    if not isinstance(options, dict) or not isinstance(versions, dict):
        raise ValueError("observed runtime options or versions are missing")
    observed = versions.get(sdk)
    if not isinstance(observed, dict) or not isinstance(observed.get("distributions"), dict):
        raise ValueError("observed SDK version is missing")
    version = expectation["version"]
    if version is None and target == "tiger":
        version = observed["distributions"].get("openvino")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("observed SDK distribution version is missing")
    loaded = observed.get("loaded_module_version")
    if (
        observed["distributions"].get(sdk) != version
        or not isinstance(loaded, str)
        or not (loaded == version or (sdk == "openvino" and loaded.startswith(version + "-")))
    ):
        raise ValueError("loaded SDK version differs from admitted evidence")
    if target == "tiger":
        if (
            identity.get("implementation") != "passage_inference.backends.OpenVinoRuntime"
            or options.get("device") not in ("GPU", "GPU.0")
            or not isinstance(options.get("execution_devices"), list)
            or "GPU.0" not in options["execution_devices"]
        ):
            raise ValueError("observed OpenVINO accelerator differs")
    else:
        if (
            identity.get("implementation") != "passage_inference.backends.TensorRTRuntime"
            or type(options.get("device_id")) is not int
            or options["device_id"] != 0
            or options.get("engine_sha256") != engine_sha256
        ):
            raise ValueError("observed TensorRT accelerator or engine differs")
    return True


def bind_target_config(session, config, *, compiled_host_path=None):
    """Bind the input-only config to the candidate and budget in the all-GREEN record.

    Record and evidence hashes remain on the host; only artifact pins and declared
    precision/runtime facts enter the target configuration.
    """
    from passage_eval.campaign.direct.direct_target import ADMITTED_SCHEMA, RULE8_SCHEMA, validate_config
    from passage_eval.execution import timing

    session.grant.ensure_unchanged()
    record = pinned.read_pinned_json(session.grant.record_path, session.grant.record_sha256)
    decisions = record["protocol"]["decisions"]
    profile = record["protocol"]["profile"]
    rule8 = profile == RULE8_NATIVE_PROFILE
    if (
        profile not in (*DESCRIPTIVE_PROFILES, RULE8_NATIVE_PROFILE)
        or decisions["timing_profile"]["value"] != timing.NATIVE_PERSISTENT
    ):
        raise ValueError("a sealed native timing profile is required")
    if rule8:
        if session.rule8_scope is None:
            raise ValueError("Rule8 requires the pinned global plan")
        session.rule8_scope.check_record(session.grant)
    validate_config(config)
    if config["schema"] != (RULE8_SCHEMA if rule8 else ADMITTED_SCHEMA):
        raise ValueError("admitted S requires the endpoint schema matching its profile")
    if rule8:
        from passage_eval.campaign.direct.rule8_scope import Rule8Scope

        if not isinstance(session.rule8_scope, Rule8Scope):
            raise ValueError("Rule8 requires the pinned global scope")
        session.rule8_scope.check_record(session.grant, target=config["target"])
    warm_ref = record["evidence"]["warmup"]["roster"]
    warm_path, warm_digest = pinned.file_ref(warm_ref)
    sealed_warm = pinned.read_pinned_json(pinned.resolve_inside(session.grant.evidence_dir, warm_path), warm_digest)
    sealed_warm = roster_io.validate_roster(sealed_warm)
    if sealed_warm["roster_sha256"] != session.warmup_roster_sha256:
        raise ValueError("warmup differs from the admitted warmup roster")
    candidate = record["candidates"][config["procedure"]]
    target = config["target"]
    if target not in decisions["targets"]["value"] or config["seed"] not in decisions["seeds"]["value"]:
        raise ValueError("target or seed not admitted")
    entry = candidate["targets"][target]
    expected = {
        "bundle": entry["bundle_sha256"],
        "engine": entry["engine_sha256"],
        "checkpoint": candidate["checkpoint_sha256"],
        "decoder": candidate["final_decoders_sha256"],
    }
    if config["artifact_pins"] != expected:
        raise ValueError("candidate artifact pins differ from admission record")
    if (
        config["budget"] != decisions["budget"]["value"]
        or config["budget"] != session.budget.to_dict()
        or config["index_sha256"] != session.index_sha256
        or config["warmup_index"] != list(session.warmup_index)
    ):
        raise ValueError("budget or input projection differs from admitted session")
    compiled_host_path = compiled_host_path or session.compiled_host_path
    if rule8 and target in ("orin", "thor"):
        if compiled_host_path is None:
            raise ValueError("Rule8 TensorRT needs its pinned compiled manifest on the host")
        compiled = pinned.read_pinned_json(compiled_host_path, config["compiled_sha256"])
        if (
            compiled.get("engine_sha256") != expected["engine"]
            or not isinstance(compiled.get("runtime_versions"), dict)
            or not isinstance(compiled["runtime_versions"].get("tensorrt"), str)
        ):
            raise ValueError("compiled engine or SDK expectation differs")
    elif not rule8:
        path, digest = pinned.file_ref(entry["native_environment"])
        environment = pinned.read_pinned_json(pinned.resolve_inside(session.grant.evidence_dir, path), digest)
        _runtime_expectation(environment, target)
        facts = {"precision": environment["stack"]["precision"], "runtime": environment["stack"]["runtime"]}
        if config["native_stack"] != facts:
            raise ValueError("declared native precision or runtime differs from admitted evidence")
    session.grant.ensure_unchanged()
    return replace(session, config_bytes=pinned.canonical_json(config), compiled_host_path=compiled_host_path)


def _frame(stream, kind, sequence, requests):
    arrays = {
        "sample_ids": np.asarray([r.sample_id for r in requests]),
        "terrain": np.stack([r.elevation for r in requests]).astype(np.float64, copy=False),
        "obstacles": np.stack([r.obstacles for r in requests]).astype(np.bool_, copy=False),
        "starts": np.asarray([r.start for r in requests], dtype=np.int64),
        "goals": np.asarray([r.goal for r in requests], dtype=np.int64),
    }
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    raw = buf.getvalue()
    checked = _array_payload(raw)
    header = json.dumps(
        {
            "kind": kind,
            "sequence": sequence,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "ids": checked["sample_ids"].tolist(),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(header) > 65536:
        raise ValueError("frame header exceeds bound")
    _write_all(stream, struct.pack("!4sIQ", b"PSF2", len(header), len(raw)))
    _write_all(stream, header)
    _write_all(stream, raw)
    stream.flush()


def stream_admitted_session(
    stream,
    session,
    *,
    fragment_size=8,
    image_inspect=None,
    bwrap_inspect=None,
    process_in_scope=None,
    image_evidence_path=None,
):
    """Send PSI1, one validation warmup and bounded S fragments; references stay on host."""
    if not isinstance(session, AdmittedSession) or type(fragment_size) is not int or not 1 <= fragment_size <= 256:
        raise ValueError("session or fragment size differs")
    session.grant.ensure_unchanged()
    rule8_record = session.rule8_scope.check_record(session.grant) if session.rule8_scope is not None else None
    if not session.config_bytes:
        raise ValueError("target config has not been bound to the admission record")
    config = pinned.strict_json_loads(session.config_bytes, name="admitted target config")
    if bind_target_config(session, config).config_bytes != session.config_bytes:
        raise ValueError("admitted target config changed")
    if hashlib.sha256(session.index).hexdigest() != session.index_sha256:
        raise ValueError("input index changed")
    image_start = None
    if rule8_record is not None:
        if (
            not callable(process_in_scope)
            or image_evidence_path is None
            or callable(image_inspect) == callable(bwrap_inspect)
        ):
            raise ValueError("Rule8 requires exactly one host image observation before any S input")
        digest = rule8_record["candidates"][config["procedure"]]["targets"][config["target"]]["container_digest"]
        if bwrap_inspect is not None:
            if config["target"] != "tiger":
                raise ValueError("bwrap observation is restricted to Tiger")
            from passage_eval.campaign.direct.rule8_bwrap import verify_running_bwrap

            bwrap_proof = bwrap_inspect()
            image_start = verify_running_bwrap(
                bwrap_proof,
                digest,
                config_sha256=hashlib.sha256(session.config_bytes).hexdigest(),
                index_sha256=session.index_sha256,
                process_in_scope=process_in_scope,
            )
            image_start["mechanism"] = "bwrap"
            image_start["bwrap_proof"] = bwrap_proof
        else:
            from passage_eval.campaign.direct.rule8_image import verify_running_container

            manifest_bytes, inspected, scope, launch_reference, expected_runner_argv = image_inspect()
            image_start = verify_running_container(
                manifest_bytes,
                inspected,
                digest,
                scope=scope,
                process_in_scope=process_in_scope,
                launch_reference=launch_reference,
                expected_runner_argv=expected_runner_argv,
            )
        image_start.update(
            {
                "schema": "passage-rule8-host-image-start/v1",
                "target": config["target"],
                "procedure": config["procedure"],
                "seed": config["seed"],
                "configuration_sha256": hashlib.sha256(session.config_bytes).hexdigest(),
                "input_index_sha256": session.index_sha256,
            }
        )
        path = Path(image_evidence_path)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        fd = os.open(path, flags, 0o600)
        with os.fdopen(fd, "wb") as evidence:
            evidence.write(pinned.canonical_json(image_start))
            evidence.flush()
            os.fsync(evidence.fileno())
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    elif (
        image_inspect is not None
        or bwrap_inspect is not None
        or process_in_scope is not None
        or image_evidence_path is not None
    ):
        raise ValueError("image observation arguments require Rule8")
    session.grant.ensure_unchanged()
    if session.rule8_scope is not None:
        session.rule8_scope.check_record(session.grant)
    _write_all(stream, struct.pack("!4sIQ", b"PSI1", 0, len(session.index)))
    _write_all(stream, session.index)
    _frame(stream, "warmup", 0, [session.warmup])
    sequence = 0
    batch = []
    count = 0
    for request in admitted_requests(
        session.record_path,
        session.evidence_dir,
        session.roster_path,
        session.dataset_root,
        session.budget,
        **({"rule8_scope": session.rule8_scope} if rule8_record is not None else {}),
    ):
        if (
            count >= len(session.tasks)
            or request.sample_id != session.tasks[count]["sample_id"]
            or raw_digest(request) != session.tasks[count]["raw_input_sha256"]
        ):
            raise ValueError("admitted request differs from pinned session index")
        batch.append(request)
        count += 1
        if len(batch) == fragment_size:
            _frame(stream, "fragment", sequence, batch)
            sequence += 1
            batch = []
    if count != len(session.tasks):
        raise ValueError("admitted input stream is incomplete")
    if batch:
        _frame(stream, "fragment", sequence, batch)
    session.grant.ensure_unchanged()
    write_end(stream)
    return image_start


def verify_admitted_target_receipt(root, session):
    """Verify a returned S receipt against the same still-valid host admission binding."""
    from pathlib import Path
    from passage_eval.campaign.edge.target import _sha256_regular_file

    if not session.config_bytes:
        raise ValueError("target config is not bound to admission")
    config = pinned.strict_json_loads(session.config_bytes, name="admitted target config")
    if bind_target_config(session, config).config_bytes != session.config_bytes:
        raise ValueError("admission binding changed")
    root = Path(root)
    paths = [root / "target-receipt.json", root / "target-receipt.sha256", root / "journal.jsonl"]
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() or not p.is_file() for p in paths):
        raise ValueError("target receipt files are not regular")
    receipt_sha = paths[1].read_text().strip()
    receipt = pinned.read_pinned_json(paths[0], receipt_sha)
    rule8 = session.rule8_scope is not None
    if rule8:
        session.rule8_scope.check_record(session.grant)
    expected = {
        "schema": "passage-direct-target-rule8-s-result/v1" if rule8 else "passage-direct-target-admitted-s-result/v1",
        "configuration_sha256": hashlib.sha256(session.config_bytes).hexdigest(),
        "input_index_sha256": session.index_sha256,
        "native_artifact_hashes": config["artifact_pins"],
        "qualification_status": "host_admission_verification_required",
        "scope_gpu_drain_verified": False,
        "precision_profile": "native_observed" if rule8 else "native_descriptive",
        "native_stack": config.get("native_stack"),
    }
    if any(receipt.get(key) != value for key, value in expected.items()):
        raise ValueError("admitted target receipt identity differs")
    record = pinned.read_pinned_json(session.grant.record_path, session.grant.record_sha256)
    entry = record["candidates"][config["procedure"]]["targets"][config["target"]]
    if rule8 and config["target"] in ("orin", "thor"):
        if session.compiled_host_path is None:
            raise ValueError("compiled manifest is not available on the host")
        compiled = pinned.read_pinned_json(session.compiled_host_path, config["compiled_sha256"])
        expectation = {"sdk": "tensorrt", "version": compiled["runtime_versions"]["tensorrt"]}
    elif rule8:
        expectation = {"sdk": "openvino", "version": None}
    else:
        relative, digest = pinned.file_ref(entry["native_environment"])
        environment = pinned.read_pinned_json(pinned.resolve_inside(session.grant.evidence_dir, relative), digest)
        expectation = _runtime_expectation(environment, config["target"])
    _verify_runtime_identity(
        receipt.get("runtime_identity"), expectation, config["target"], config["artifact_pins"]["engine"]
    )
    if _sha256_regular_file(paths[2]) != receipt.get("journal_sha256"):
        raise ValueError("admitted target journal digest differs")
    session.grant.ensure_unchanged()
    return receipt


def verify_rule8_completed_session(
    session, target_root, image_start_path, image_start_sha256, *, container_inspect, drain, job_id
):
    """Bind a collected target receipt to the same observed OCI process and drained job.

    ``container_inspect`` must come directly from the trusted host dispatcher.
    This checks OCI execution identity, not publication, rebuild or rootfs provenance.
    """
    from passage_eval.campaign.direct.direct_host_scoring import verify_drain
    from passage_eval.campaign.direct.rule8_image import verify_finished_container

    if session.rule8_scope is None or not session.config_bytes:
        raise ValueError("completed Rule8 session requires its bound global scope")
    record = session.rule8_scope.check_record(session.grant)
    config = pinned.strict_json_loads(session.config_bytes, name="Rule8 target configuration")
    start = pinned.read_pinned_json(image_start_path, image_start_sha256)
    expected = {
        "schema": "passage-rule8-host-image-start/v1",
        "target": config["target"],
        "procedure": config["procedure"],
        "seed": config["seed"],
        "configuration_sha256": hashlib.sha256(session.config_bytes).hexdigest(),
        "input_index_sha256": session.index_sha256,
        "container_digest": record["candidates"][config["procedure"]]["targets"][config["target"]]["container_digest"],
    }
    if not isinstance(start, dict) or any(start.get(k) != v for k, v in expected.items()):
        raise ValueError("host image observation differs from the admitted session")
    verify_drain(drain, target=config["target"], procedure=config["procedure"], job_id=job_id, scope=start["scope"])
    receipt = verify_admitted_target_receipt(target_root, session)
    if start.get("mechanism") == "bwrap":
        if config["target"] != "tiger" or container_inspect is not None or start.get("job_id") != job_id:
            raise ValueError("bwrap completion identity differs")
        from passage_eval.campaign.direct.oci_validation_relay import _journal_binding
        from passage_eval.campaign.direct.rule8_bwrap import verify_running_bwrap

        checked = verify_running_bwrap(
            start.get("bwrap_proof"),
            expected["container_digest"],
            config_sha256=expected["configuration_sha256"],
            index_sha256=session.index_sha256,
            process_in_scope=lambda pid, scope: True,
        )
        if any(start.get(key) != value for key, value in checked.items()):
            raise ValueError("recorded bwrap observation differs")
        markers = _journal_binding((Path(target_root) / "journal.jsonl").read_bytes(), start.get("namespace_pid"))
        if markers.get("journal_markers") != "matched":
            raise ValueError("bwrap runner markers are missing")
        image = {
            "container_digest": start["container_digest"],
            "runner_pid": start["pid"],
            "n_processes": 1,
            "n_restarts": 0,
            **markers,
        }
    else:
        image = verify_finished_container(start, container_inspect, exit_scope_drained=True)
    session.rule8_scope.check_record(session.grant)
    return {
        "container_digest": image["container_digest"],
        "configuration_sha256": expected["configuration_sha256"],
        "input_index_sha256": session.index_sha256,
        "journal_sha256": receipt["journal_sha256"],
        "scope": start["scope"],
        "job_id": job_id,
        "runner_pid": image["runner_pid"],
        **(
            {
                "runner_argv": start["runner_argv"],
                "journal_markers": image["journal_markers"],
                "runner_session_uuid": image["runner_session_uuid"],
            }
            if start.get("mechanism") == "bwrap"
            else {}
        ),
        "n_processes": image["n_processes"],
        "n_restarts": image["n_restarts"],
    }
