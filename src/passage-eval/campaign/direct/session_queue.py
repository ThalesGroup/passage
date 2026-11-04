"""Host-only sequential per-target dispatch; separate targets run concurrently.

A cell is never replayed after an ambiguous job. The retained ledger requires a
new plan and explicit operator decision after any interruption or failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from passage_eval.campaign.direct.direct_host_scoring import verify_drain
from passage_eval.campaign.direct.direct_target import validate_config
from passage_eval.campaign.edge.lease import holder_host_for, new_effect_key, parse_job_id, resource_for
from passage_eval.io import pinned


_ADMISSION = {
    "record": "--record", "evidence_dir": "--evidence-dir", "roster": "--roster",
    "dataset_root": "--dataset-root", "warmup_npz": "--warmup-npz",
    "warmup_npz_sha256": "--warmup-npz-sha256", "scope": "--scope", "scope_sha256": "--scope-sha256",
}


def _record(path, value):
    import os
    raw = pinned.canonical_json(value)
    with Path(path).open("xb") as file:
        file.write(raw)
        file.flush()
        os.fsync(file.fileno())
    fd = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _adapter(target):
    if target in ("orin", "thor"):
        from passage_eval.campaign.direct import oci_validation_relay as adapter
        return "passage_eval.campaign.direct.oci_validation_relay", adapter
    if target == "tiger":
        from passage_eval.campaign.direct import bwrap_validation_relay as adapter
        return "passage_eval.campaign.direct.bwrap_validation_relay", adapter
    raise ValueError("unknown target")


def load_queue(path, sha256):
    """Preflight all validation cells before any job, without opening S data."""
    plan = pinned.read_pinned_json(path, sha256)
    return validate_queue(plan)


def validate_queue(plan):
    """Apply the same preflight to a proposed in-memory plan before publication."""
    if (not isinstance(plan, dict) or set(plan) != {"schema", "sessions", "ledger", "cavops"}
            or plan["schema"] != "passage-direct-session-queue/v1"):
        raise ValueError("queue specification differs")
    entries = plan["sessions"]
    wanted = {(t, s, p) for t in ("orin", "thor", "tiger") for s in (1234, 2026, 3407) for p in ("seg", "reg")}
    if not isinstance(entries, list) or len(entries) != len(wanted):
        raise ValueError("queue requires exactly 18 cells")
    if len({entry.get("admission") is None for entry in entries if isinstance(entry, dict)}) != 1:
        raise ValueError("validation and admitted sessions cannot share a queue")
    seen = set()
    pair = None
    catalog = None
    scope_pin = None
    candidate_runtime = {}
    cache = {}
    admissions = {}
    pair_records = {}
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"target", "seed", "procedure", "spec", "sha256", "admission"}:
            raise ValueError("queue cell differs")
        cell = (entry["target"], entry["seed"], entry["procedure"])
        if cell not in wanted or cell in seen or not pinned.is_hex64(entry["sha256"]):
            raise ValueError("duplicate or invalid queue cell")
        seen.add(cell)
        module, adapter = _adapter(entry["target"])
        admission = entry["admission"]
        if admission is None:
            if entry["target"] == "tiger":
                spec, config = adapter.inspect_spec(entry["spec"], entry["sha256"])
                checked_spec, checked_config, *_ = adapter.load_spec(entry["spec"], entry["sha256"], image_cache=cache)
                if checked_spec != spec or checked_config != config:
                    raise ValueError("Tiger full preflight differs from the input projection")
            else:
                spec, config, *_ = adapter.load_spec(entry["spec"], entry["sha256"], image_cache=cache)
            prepared = pinned.read_pinned_json(Path(spec["prepared_root"]) / "prepared.json", spec["prepared_sha256"])
            if (prepared.get("index_sha256") != config["index_sha256"]
                    or prepared.get("warmup_index") != config["warmup_index"]
                    or prepared.get("count") != 128):
                raise ValueError("prepared validation inputs differ from target configuration")
            if entry["target"] != "tiger":
                adapter._validation_provenance(spec, config, prepared)
            if entry["seed"] != 2026:
                current = spec["replica_catalog_sha256"]
                if catalog is None:
                    catalog = current
                elif catalog != current:
                    raise ValueError("validation catalogue differs across cells")
        else:
            if not isinstance(admission, dict) or set(admission) != set(_ADMISSION):
                raise ValueError("admission arguments differ")
            if any(not isinstance(value, str) or not value for value in admission.values()):
                raise ValueError("admission arguments missing")
            if not pinned.is_hex64(admission["warmup_npz_sha256"]) or not pinned.is_hex64(admission["scope_sha256"]):
                raise ValueError("admission argument pins differ")
            if scope_pin is None:
                scope_pin = admission["scope_sha256"]
            elif scope_pin != admission["scope_sha256"]:
                raise ValueError("Rule8 global scope differs across cells")
            from passage_eval.campaign.direct.rule8_scope import load_rule8_scope
            from passage_eval.protocol.admission import grant_test_admission

            admission_key = (admission["record"], admission["evidence_dir"], admission["scope"],
                             admission["scope_sha256"])
            if admission_key not in admissions:
                grant = grant_test_admission(admission["record"], admission["evidence_dir"])
                scope = load_rule8_scope(admission["scope"], admission["scope_sha256"])
                admissions[admission_key] = (grant, scope)
            grant, scope = admissions[admission_key]
            sealed_record = scope.check_record(grant, target=entry["target"])
            if sealed_record["protocol"]["decisions"]["seeds"]["value"] != [entry["seed"]]:
                raise ValueError("Rule8 record seed differs from queued cell")
            if not pinned.is_hex64(grant.record_sha256):
                raise ValueError("Rule8 record digest differs")
            pair_key = (entry["target"], entry["seed"])
            previous_record = pair_records.setdefault(pair_key, grant.record_sha256)
            if previous_record != grant.record_sha256:
                raise ValueError("Seg and Reg must use the same admitted record per target and seed")
            # Actual grant and roster are checked by the worker before any dataset row is read.
            if entry["target"] == "tiger":
                spec, config = adapter.inspect_spec(entry["spec"], entry["sha256"])
            else:
                spec = pinned.read_pinned_json(entry["spec"], entry["sha256"])
                config = validate_config(pinned.read_pinned_json(spec["configuration"], spec["configuration_sha256"]))
            if config["dataset_split"] != "test":
                raise ValueError("admitted job must use a test configuration")
        if (config["target"], config["seed"], config["procedure"]) != cell:
            raise ValueError("queue and target configuration differ")
        if entry["target"] in ("orin", "thor", "tiger"):
            argv = spec.get("runner_argv")
            if not isinstance(argv, list) or argv.count("--config-sha256") != 1:
                raise ValueError("candidate runner identity differs")
            argv = list(argv)
            position = argv.index("--config-sha256") + 1
            if position >= len(argv) or argv[position] != spec["configuration_sha256"]:
                raise ValueError("candidate runner configuration pin differs")
            argv[position] = "<configuration-sha256>"
            identity = (spec.get("image_manifest_digest"), tuple(argv))
            candidate_key = (entry["target"], entry["procedure"])
            if candidate_key in candidate_runtime and candidate_runtime[candidate_key] != identity:
                raise ValueError("candidate image or runner differs across seeds")
            candidate_runtime[candidate_key] = identity
        key = (config["index_sha256"], pinned.canonical_json(config["warmup_index"]))
        if pair is None:
            pair = key
        elif pair != key:
            raise ValueError("cells must use the same input and warmup projections")
        if not isinstance(spec.get("output_root"), str) or not spec["output_root"]:
            raise ValueError("output root missing")
        entry["_module"] = module
        entry["_output"] = spec["output_root"]
        entry["_config_sha256"] = spec["configuration_sha256"]
        entry["_index_sha256"] = config["index_sha256"]
    if len({entry["_output"] for entry in entries}) != len(entries):
        raise ValueError("output roots must be distinct")
    return plan


def _run_target(plan, target, ledger, *, run=subprocess.run):
    for position, entry in enumerate(e for e in plan["sessions"] if e["target"] == target):
        output_root = Path(entry["_output"])
        if output_root.exists() or output_root.is_symlink():
            raise FileExistsError("session output already exists; no automatic replay")
        key = new_effect_key(target)
        command = [sys.executable, "-B", "-m", entry["_module"], "--spec", entry["spec"], "--sha256", entry["sha256"]]
        if entry["admission"] is not None:
            for name, flag in _ADMISSION.items():
                command.extend((flag, entry["admission"][name]))
        fingerprint = hashlib.sha256(pinned.canonical_json({"target": target, "command": command})).hexdigest()
        stem = f"{target}-{position:02d}"
        _record(ledger / (stem + "-started.json"), {"cell": [target, entry["seed"], entry["procedure"]],
                                                      "effect_key": key, "effect_fingerprint": fingerprint})
        argv = [plan["cavops"], "job-run", "--host", holder_host_for(target), "--resource", resource_for(target),
                "--effect-key", key, "--effect-fingerprint", fingerprint, "--task", "passage-edge-qualification",
                "--claim", "passage-edge-qualification", "--wait", "--", "/usr/bin/timeout", "--signal=TERM",
                "--kill-after=15s", "1180s", *command]
        try:
            result = run(argv, capture_output=True, text=True, check=False, timeout=1300)
            job = parse_job_id(result.stdout)
            if result.returncode:
                raise RuntimeError("job failed; target status must be checked")
            drain = pinned.strict_json_loads((Path(entry["_output"]) / "drain.json").read_bytes(), name="drain")
            start = pinned.strict_json_loads((output_root / "image-start.json").read_bytes(), name="image start")
            if (start.get("scope") != drain.get("scope") or start.get("target") != target
                    or start.get("procedure") != entry["procedure"]
                    or start.get("configuration_sha256") != entry["_config_sha256"]
                    or start.get("input_index_sha256") != entry["_index_sha256"]):
                raise ValueError("drain does not match the observed image and input")
            collected = output_root / "collected"
            receipt_sha = (collected / "target-receipt.sha256").read_text(encoding="ascii").strip()
            receipt = pinned.read_pinned_json(collected / "target-receipt.json", receipt_sha)
            if (receipt.get("configuration_sha256") != entry["_config_sha256"]
                    or receipt.get("input_index_sha256") != entry["_index_sha256"]
                    or hashlib.sha256((collected / "journal.jsonl").read_bytes()).hexdigest() != receipt.get("journal_sha256")):
                raise ValueError("target receipt does not match the observed session")
            verify_drain(drain, target=target, procedure=entry["procedure"], job_id=job, scope=drain["scope"])
            leases = run([plan["cavops"], "leases", "--json"], capture_output=True, text=True, check=True, timeout=20)
            if any(x.get("resource") == resource_for(target) for x in json.loads(leases.stdout)):
                raise RuntimeError("target lease not released after drainage")
        except BaseException as exc:
            _record(ledger / (stem + "-failed.json"), {"error_type": type(exc).__name__})
            raise
        _record(ledger / (stem + "-complete.json"), {"job_id": job, "scope": drain["scope"], "effect_key": key})


def run_queue(path, sha256):
    plan = load_queue(path, sha256)
    ledger = Path(plan["ledger"])
    ledger.mkdir(mode=0o700)
    _record(ledger / "queue-start.json", {"plan_sha256": sha256})
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(_run_target, plan, target, ledger) for target in ("orin", "thor", "tiger")]
        for future in futures:
            future.result()
    _record(ledger / "queue-complete.json", {"sessions": 18, "plan_sha256": sha256})


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    run_queue(args.plan, args.sha256)


if __name__ == "__main__":
    main()
