"""Assemble a G2 receipt from existing observed direct-validation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from passage_eval.campaign.direct.admitted_stream import _verify_runtime_identity
from passage_eval.campaign.direct.direct_host_scoring import verify_drain
from passage_eval.campaign.direct.direct_session import inspect_journal
from passage_eval.campaign.direct.oci_validation_relay import _journal_binding
from passage_eval.campaign.direct.rule8_image import _is_direct_python_command
from passage_eval.campaign.edge.lease import parse_job_id
from passage_eval.io import pinned


def _file(evidence_dir, ref):
    name, digest = pinned.file_ref(ref)
    return pinned.resolve_inside(evidence_dir, name), digest


def _json(evidence_dir, ref):
    path, digest = _file(evidence_dir, ref)
    return pinned.read_pinned_json(path, digest)


def _procedure(evidence_dir, record, target, procedure, refs, plan_sha256):
    keys = {
        "configuration",
        "image_start",
        "image_finish",
        "drain",
        "target_receipt",
        "journal",
        "aggregate",
        "run_manifest",
        "roster",
        "warmup_roster",
        "compiled_manifest",
    }
    if not isinstance(refs, dict) or set(refs) != keys:
        raise ValueError("validation evidence references differ")
    config, start, finish, drain, receipt, aggregate, manifest = (
        _json(evidence_dir, refs[key])
        for key in (
            "configuration",
            "image_start",
            "image_finish",
            "drain",
            "target_receipt",
            "aggregate",
            "run_manifest",
        )
    )
    for key in ("roster", "warmup_roster"):
        _json(evidence_dir, refs[key])
    journal_path, journal_sha = _file(evidence_dir, refs["journal"])
    journal = pinned.read_pinned(journal_path, journal_sha)
    candidate = record["candidates"][procedure]
    target_candidate = candidate["targets"][target]
    expected_pins = {
        "bundle": target_candidate["bundle_sha256"],
        "checkpoint": candidate["checkpoint_sha256"],
        "decoder": candidate["final_decoders_sha256"],
        "engine": target_candidate["engine_sha256"],
    }
    if (
        config.get("dataset_split") != "validation"
        or config.get("target") != target
        or config.get("procedure") != procedure
        or config.get("artifact_pins") != expected_pins
    ):
        raise ValueError("validation configuration differs from recorded candidate")
    config_sha = refs["configuration"]["sha256"]
    image = target_candidate["container_digest"]
    if (
        type(start.get("pid")) is not int
        or start["pid"] <= 0
        or type(start.get("namespace_pid")) is not int
        or start["namespace_pid"] <= 0
        or not isinstance(start.get("scope"), str)
        or not start["scope"]
        or (
            target != "tiger"
            and (
                not isinstance(start.get("container_id"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", start["container_id"])
            )
        )
        or start.get("target") != target
        or start.get("procedure") != procedure
        or start.get("configuration_sha256") != config_sha
        or start.get("input_index_sha256") != config["index_sha256"]
        or start.get("container_digest") != image
        or finish.get("container_digest") != image
        or finish.get("scope") != start.get("scope")
        or finish.get("runner_pid") != start.get("pid")
        or finish.get("exit_code") != 0
        or (target != "tiger" and finish.get("container_id") != start.get("container_id"))
        or finish.get("n_processes") != 1
        or finish.get("n_restarts") != 0
        or finish.get("journal_markers") != "matched"
    ):
        raise ValueError("observed image, runner or process continuity differs")
    argv = start.get("runner_argv")
    if (
        not isinstance(argv, list)
        or not _is_direct_python_command(argv)
        or hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest()
        != start.get("runner_argv_sha256")
    ):
        raise ValueError("observed runner argument vector differs")
    job = drain.get("job_id")
    observed_job = start.get("job_id")
    if (
        not isinstance(job, str)
        or not isinstance(observed_job, str)
        or parse_job_id(job) != job
        or parse_job_id(observed_job) != observed_job
        or job != observed_job
    ):
        raise ValueError("drain job differs from observed image session")
    verify_drain(drain, target=target, procedure=procedure, job_id=job, scope=start["scope"])
    markers = _journal_binding(journal, start.get("namespace_pid"))
    if markers.get("journal_markers") != "matched" or finish.get("runner_session_uuid") != markers.get(
        "runner_session_uuid"
    ):
        raise ValueError("runner session identity differs from observed PID")
    audit = inspect_journal(journal_path)
    if (
        not audit["complete"]
        or not audit["runtime_released"]
        or audit["committed_prefix"] != 128
        or audit["ambiguous_range"] is not None
    ):
        raise ValueError("validation journal is incomplete")
    timing = receipt.get("timing") or {}
    if (
        receipt.get("schema") != "passage-direct-target-validation-result-prototype/v1"
        or receipt.get("qualification_status") != "exploratory_unadmitted"
        or receipt.get("configuration_sha256") != config_sha
        or receipt.get("input_index_sha256") != config["index_sha256"]
        or receipt.get("native_artifact_hashes") != expected_pins
        or receipt.get("journal_sha256") != journal_sha
        or timing.get("status") != "complete"
        or timing.get("n_completed") != 128
        or timing.get("warmups_completed") != 16
    ):
        raise ValueError("target receipt is not linked to completed validation")
    if target in ("orin", "thor"):
        compiled = _json(evidence_dir, refs["compiled_manifest"])
        if (
            refs["compiled_manifest"]["sha256"] != config.get("compiled_sha256")
            or compiled.get("engine_sha256") != expected_pins["engine"]
        ):
            raise ValueError("compiled manifest differs")
        expectation = {"sdk": "tensorrt", "version": compiled["runtime_versions"]["tensorrt"]}
    else:
        if refs["compiled_manifest"] is not None:
            raise ValueError("Tiger must not declare a TensorRT manifest")
        expectation = {"sdk": "openvino", "version": None}
    _verify_runtime_identity(receipt.get("runtime_identity"), expectation, target, expected_pins["engine"])
    execution = manifest.get("native_execution")
    if (
        not isinstance(execution, dict)
        or execution.get("session_id") != journal_sha
        or execution.get("container_digest") != image
        or execution.get("n_processes") != 1
        or execution.get("n_restarts") != 0
        or execution.get("artifacts")
        != {
            "checkpoint_sha256": expected_pins["checkpoint"],
            "engine_sha256": expected_pins["engine"],
            "final_decoders_sha256": expected_pins["decoder"],
        }
        or aggregate.get("plan_sha256") != plan_sha256
    ):
        raise ValueError("scored manifest or aggregate differs from observed session")
    result = {
        "completed": True,
        "accelerator_used": True,
        "checkpoint_sha256": expected_pins["checkpoint"],
        "engine_sha256": expected_pins["engine"],
        "native_execution": execution,
        "aggregate": refs["aggregate"],
        "run_manifest": refs["run_manifest"],
        "roster": refs["roster"],
        "warmup_roster": refs["warmup_roster"],
        "runner_argv": argv,
        "source_qualification_status": receipt.get("qualification_status"),
        "configuration": refs["configuration"],
        "image_start": refs["image_start"],
        "image_finish": refs["image_finish"],
        "drain": refs["drain"],
        "target_receipt": refs["target_receipt"],
        "journal": refs["journal"],
    }
    if target == "tiger":
        from passage_eval.campaign.direct.runner_binding import verify_tiger_observation

        verify_tiger_observation(start)
        rootfs = start.get("rootfs_archive_sha256")
        exceptions = start.get("rootfs_exceptions")
        if not pinned.is_hex64(rootfs) or not isinstance(exceptions, dict):
            raise ValueError("observed Tiger rootfs identity missing")
        result.update(rootfs_export_sha256=rootfs, rootfs_exceptions=exceptions)
    return result


def build(record_ref, evidence_dir, target, plan_sha256, procedures, output):
    if target not in ("orin", "thor", "tiger") or not pinned.is_hex64(plan_sha256):
        raise ValueError("validation target or plan differs")
    record = _json(evidence_dir, record_ref)
    if not isinstance(procedures, dict) or set(procedures) != {"seg", "reg"}:
        raise ValueError("exactly Seg and Reg evidence required")
    entries = {
        name: _procedure(evidence_dir, record, target, name, procedures[name], plan_sha256) for name in ("seg", "reg")
    }
    receipt = {
        "schema": "passage-native-persistent-validation/v1",
        "target": target,
        "split": "validation",
        "harness": record["harness"],
        "plan_sha256": plan_sha256,
        "procedures": entries,
    }
    raw = pinned.canonical_json(receipt)
    with Path(output).open("xb") as file:
        file.write(raw)
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    inputs = pinned.read_pinned_json(args.inputs, args.inputs_sha256)
    if not isinstance(inputs, dict) or set(inputs) != {"record", "evidence_dir", "target", "plan_sha256", "procedures"}:
        raise ValueError("G2 inputs differ")
    print(
        build(
            inputs["record"],
            inputs["evidence_dir"],
            inputs["target"],
            inputs["plan_sha256"],
            inputs["procedures"],
            args.output,
        )
    )


if __name__ == "__main__":
    main()
