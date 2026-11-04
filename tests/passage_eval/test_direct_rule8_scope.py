"""Synthetic host scope checks only; no datasets, devices, or cleanup."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from passage_eval.campaign.direct import admitted_stream, rule8_scope


def scope_and_grant(monkeypatch, root, *, record_catalog="a" * 64):
    analysis = {
        "a": "seg",
        "b": "reg",
        "target": "orin",
        "seeds": [1234, 2026, 3407],
        "alpha": 0.05,
        "draws": 10000,
        "permutations": 10000,
        "bootstrap_seed": 1,
        "permutation_seed": 2,
        "latency_bootstrap_seed": 3,
    }
    plan = {
        "schema": "passage-direct-rule8-global-scope/v1",
        "replica_catalog_sha256": "a" * 64,
        "confirmatory_analysis": analysis,
    }
    record = {
        "protocol": {
            "profile": "rule8_native_persistent",
            "decisions": {"seeds": {"value": [2026]}, "confirmatory_analysis": {"value": analysis}},
        },
        "evidence": {"replica_catalog": {"path": "catalog.json", "sha256": record_catalog}},
    }

    def write_new(name, value):
        data = json.dumps(value).encode()
        path = root / name
        with path.open("xb") as stream:
            stream.write(data)
        return path, hashlib.sha256(data).hexdigest()

    plan_path, plan_sha = write_new("plan.json", plan)
    record_path, record_sha = write_new("record.json", record)

    class Grant:
        admissible_roster_sha256 = "b" * 64
        report = SimpleNamespace(admitted={"profile": "rule8_native_persistent"})

        def __init__(self):
            self.record_path = record_path
            self.record_sha256 = record_sha
            self.evidence_dir = root

        def ensure_unchanged(self):
            pass

        def is_genuine(self):
            return True

        def reference_ids(self):
            return {"measured": "sealed-reference"}

        def reference_policy(self):
            return {
                "reference_policy": "independent_analytic_reference",
                "reference_cohort_id": "synthetic",
                "reference_max_expansions": 9,
            }

    monkeypatch.setattr(rule8_scope, "AdmissionGrant", Grant)
    return rule8_scope.load_rule8_scope(plan_path, plan_sha), Grant()


def test_scope_checks_catalog_and_analysis(monkeypatch, tmp_path):
    scope, grant = scope_and_grant(monkeypatch, tmp_path)
    assert scope.check_record(grant)["protocol"]["decisions"]["seeds"]["value"] == [2026]
    assert scope.check_record(grant, target="orin")
    with pytest.raises(ValueError, match="record target differs"):
        scope.check_record(grant, target="thor")


def test_different_catalogue_refused_before_roster_open(monkeypatch, tmp_path):
    scope, grant = scope_and_grant(monkeypatch, tmp_path, record_catalog="c" * 64)
    monkeypatch.setattr(admitted_stream, "grant_test_admission", lambda *a: grant)
    monkeypatch.setattr(admitted_stream.roster_io, "load_roster", lambda *a, **k: pytest.fail("roster opened"))
    with pytest.raises(ValueError, match="catalogue differs"):
        admitted_stream.prepare_admitted_session(
            "record", "evidence", "roster", "dataset", None, None, None, rule8_scope=scope
        )


def test_rule8_without_global_plan_refused_before_roster_open(monkeypatch, tmp_path):
    _, grant = scope_and_grant(monkeypatch, tmp_path)
    monkeypatch.setattr(admitted_stream, "grant_test_admission", lambda *a: grant)
    monkeypatch.setattr(admitted_stream.roster_io, "load_roster", lambda *a, **k: pytest.fail("roster opened"))
    with pytest.raises(ValueError, match="global plan"):
        admitted_stream.prepare_admitted_session("record", "evidence", "roster", "dataset", None, None, None)


def test_rule8_bind_uses_compiled_sdk_without_new_attestation_but_stream_stays_closed(monkeypatch, tmp_path):
    from passage_eval.campaign.direct.direct_target import RULE8_SCHEMA
    from passage_eval.protocol.contracts import BudgetSpec

    scope, grant = scope_and_grant(monkeypatch, tmp_path)
    budget = BudgetSpec(wall_time_ms=1000)
    pins = {"bundle": "1" * 64, "engine": "2" * 64, "checkpoint": "3" * 64, "decoder": "4" * 64}
    warm = {"roster_sha256": "e" * 64}
    warm_raw = json.dumps(warm).encode()
    with (tmp_path / "warm.json").open("xb") as f:
        f.write(warm_raw)
    warm_ref = {"path": "warm.json", "sha256": hashlib.sha256(warm_raw).hexdigest()}
    compiled = {"engine_sha256": pins["engine"], "runtime_versions": {"tensorrt": "10.16.2.10"}}
    compiled_raw = json.dumps(compiled).encode()
    compiled_path = tmp_path / "compiled.json"
    with compiled_path.open("xb") as f:
        f.write(compiled_raw)
    record = json.loads(grant.record_path.read_text())
    record["protocol"]["decisions"].update(
        {
            "timing_profile": {"value": "native_raw_path_persistent_v1"},
            "targets": {"value": ["orin"]},
            "budget": {"value": budget.to_dict()},
        }
    )
    record["evidence"]["warmup"] = {"roster": warm_ref}
    manifest_bytes = json.dumps({"schemaVersion": 2, "config": {"digest": "sha256:" + "a" * 64}}).encode()
    image_digest = "sha256:" + hashlib.sha256(manifest_bytes).hexdigest()
    record["candidates"] = {
        "reg": {
            "checkpoint_sha256": pins["checkpoint"],
            "final_decoders_sha256": pins["decoder"],
            "targets": {
                "orin": {
                    "bundle_sha256": pins["bundle"],
                    "engine_sha256": pins["engine"],
                    "container_digest": image_digest,
                }
            },
        }
    }
    record_raw = json.dumps(record).encode()
    grant.record_path.write_bytes(record_raw)
    grant.record_sha256 = hashlib.sha256(record_raw).hexdigest()
    monkeypatch.setattr(admitted_stream.roster_io, "validate_roster", lambda r: r)
    session = admitted_stream.AdmittedSession(
        "record",
        "evidence",
        "roster",
        "dataset",
        budget,
        grant,
        b"index",
        hashlib.sha256(b"index").hexdigest(),
        (),
        None,
        ({"sample_id": "warm", "raw_input_sha256": "d" * 64},),
        "e" * 64,
        rule8_scope=scope,
    )
    cfg = {
        "schema": RULE8_SCHEMA,
        "dataset_split": "test",
        "target": "orin",
        "procedure": "reg",
        "seed": 2026,
        "artifact_paths": {},
        "artifact_pins": pins,
        "compiled_manifest": "/target/compiled.json",
        "compiled_sha256": hashlib.sha256(compiled_raw).hexdigest(),
        "exploratory_unadmitted_precision": False,
        "budget": budget.to_dict(),
        "warmup_index": list(session.warmup_index),
        "index_sha256": hashlib.sha256(b"index").hexdigest(),
        "checkpoint_size": 32,
        "grn_evidence": None,
    }
    bound = admitted_stream.bind_target_config(session, cfg, compiled_host_path=compiled_path)
    assert bound.config_bytes and bound.compiled_host_path == compiled_path
    import io

    with pytest.raises(ValueError, match="host image observation"):
        admitted_stream.stream_admitted_session(io.BytesIO(), bound)
    output = io.BytesIO()
    evidence = tmp_path / "image-start.json"
    with pytest.raises(ValueError, match="manifest digest differs"):
        admitted_stream.stream_admitted_session(
            output,
            bound,
            image_inspect=lambda: (b"{}", {}, "scope", "bad", []),
            process_in_scope=lambda *_: True,
            image_evidence_path=evidence,
        )
    assert output.getvalue() == b"" and not evidence.exists()

    # The host must persist its actual image observation before consuming any input.
    from dataclasses import replace
    import numpy as np
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.campaign.direct.raw_identity import raw_digest

    warm_request = PlanRequest(
        "warm", np.zeros((3, 3)), np.zeros((3, 3), bool), (0, 0), (2, 2), "elevation", 62.4, budget=budget
    )
    measured = PlanRequest(
        "measured", np.ones((3, 3)), np.zeros((3, 3), bool), (0, 0), (2, 2), "elevation", 62.4, budget=budget
    )
    bound = replace(
        bound, warmup=warm_request, tasks=({"sample_id": "measured", "raw_input_sha256": raw_digest(measured)},)
    )

    def inputs_after_observation(*_, **kwargs):
        assert kwargs.get("rule8_scope") is scope
        assert evidence.is_file()
        return iter([measured])

    monkeypatch.setattr(admitted_stream, "admitted_requests", inputs_after_observation)
    argv = ["/usr/bin/python3", "-m", "passage_eval.campaign.direct.direct_target"]
    running = {
        "Id": "b" * 64,
        "Image": "sha256:" + "a" * 64,
        "Path": argv[0],
        "Args": argv[1:],
        "RestartCount": 0,
        "State": {"Running": True, "Pid": 42, "StartedAt": "2026-11-04T00:00:00Z"},
    }
    start = admitted_stream.stream_admitted_session(
        io.BytesIO(),
        bound,
        fragment_size=1,
        image_inspect=lambda: (manifest_bytes, running, "job.scope", "image@" + image_digest, argv),
        process_in_scope=lambda pid, scope: (pid, scope) == (42, "job.scope"),
        image_evidence_path=evidence,
    )
    assert start["container_digest"] == image_digest
    start_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    target_root = tmp_path / "target-output"
    target_root.mkdir()
    journal = target_root / "journal.jsonl"
    events = [
        {"kind": "session_start", "procedure": "reg", "n_expected": 1, "warmup_required": 16},
        {"kind": "warmup_complete", "count": 16, "distinct_inputs": 1},
        {"kind": "reservation", "start": 0, "end": 1},
        {"kind": "request", "sequence": 0},
        {"kind": "checkpoint", "n_completed": 1},
        {"kind": "session_complete", "n_completed": 1},
        {"kind": "runtime_released"},
    ]
    journal.write_text("".join(json.dumps(event) + "\n" for event in events))
    journal_sha = hashlib.sha256(journal.read_bytes()).hexdigest()
    receipt = {
        "schema": "passage-direct-target-rule8-s-result/v1",
        "configuration_sha256": hashlib.sha256(bound.config_bytes).hexdigest(),
        "input_index_sha256": bound.index_sha256,
        "native_artifact_hashes": pins,
        "qualification_status": "host_admission_verification_required",
        "scope_gpu_drain_verified": False,
        "precision_profile": "native_observed",
        "native_stack": None,
        "journal_sha256": journal_sha,
        "runtime_identity": {
            "implementation": "passage_inference.backends.TensorRTRuntime",
            "options": {"device_id": 0, "engine_sha256": pins["engine"]},
            "versions": {
                "tensorrt": {"distributions": {"tensorrt": "10.16.2.10"}, "loaded_module_version": "10.16.2.10"}
            },
        },
    }
    receipt_raw = json.dumps(receipt).encode()
    (target_root / "target-receipt.json").write_bytes(receipt_raw)
    (target_root / "target-receipt.sha256").write_text(hashlib.sha256(receipt_raw).hexdigest())
    drain = {
        "schema": "passage-direct-session-drain/v1",
        "target": "orin",
        "procedure": "reg",
        "job_id": "job:0123456789abcdef",
        "scope": "job.scope",
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    stopped = {**running, "State": {"Running": False, "ExitCode": 0, "StartedAt": "2026-11-04T00:00:00Z"}}
    result = admitted_stream.verify_rule8_completed_session(
        bound, target_root, evidence, start_sha, container_inspect=stopped, drain=drain, job_id="job:0123456789abcdef"
    )
    assert result["container_digest"] == image_digest and result["journal_sha256"] == journal_sha
    assert (result["n_processes"], result["n_restarts"], result["runner_pid"]) == (1, 0, 42)

    # Exercise the real image/receipt verifier through the scorer boundary.
    from passage_eval.campaign.direct import direct_host_scoring as scoring
    from passage_eval.io import roster as roster_io
    from passage_eval.protocol import admission

    scene = SimpleNamespace(sample_id="measured")
    roster = {
        "tasks": [{"sample_id": "measured", "input_identity": {"x": 1}}],
        "roster_sha256": "r",
        "dataset": {"commit": "revision"},
        "dataset_root_identity": {"sha256": "root"},
        "cell": {"task_variant": "fixture"},
    }
    monkeypatch.setattr(roster_io, "load_roster", lambda *args, **kwargs: roster)
    monkeypatch.setattr(roster_io, "request_input_identity", lambda request: {"x": 1})
    monkeypatch.setattr(scoring.PlanRequest, "from_scene", lambda scene, budget: measured)

    def projection(value):
        if value.get("roster_sha256") == "e" * 64:
            return {"tasks": [{"sample_id": "warm", "content_identity_sha256": "w"}]}
        return {"tasks": [{"sample_id": "measured", "content_identity_sha256": "m"}]}

    monkeypatch.setattr(admission, "native_input_projection", projection)
    artifact = SimpleNamespace(manifest={})
    monkeypatch.setattr(scoring, "host_score_direct", lambda *args, **kwargs: ({}, {}, artifact))
    scoring.score_rule8_completed_session(
        bound,
        target_root,
        evidence,
        start_sha,
        [scene],
        {"measured": {"reference_id": "sealed-reference"}},
        container_inspect=stopped,
        drain=drain,
        job_id="job:0123456789abcdef",
        reference_cohort_id="synthetic",
    )
    assert artifact.manifest["rule8_session_evidence"]["runner_pid"] == 42
    assert artifact.manifest["native_execution"]["container_digest"] == image_digest
    bad = {**stopped, "RestartCount": 1}
    with pytest.raises(ValueError):
        scoring.score_rule8_completed_session(
            bound,
            target_root,
            evidence,
            start_sha,
            [scene],
            {"measured": {"reference_id": "sealed-reference"}},
            container_inspect=bad,
            drain=drain,
            job_id="job:0123456789abcdef",
            reference_cohort_id="synthetic",
        )


def test_direct_reader_itself_refuses_rule8_without_global_scope(monkeypatch, tmp_path):
    from passage_eval.campaign.direct import admitted_input

    _, grant = scope_and_grant(monkeypatch, tmp_path)
    monkeypatch.setattr(admitted_input, "grant_test_admission", lambda *args: grant)
    monkeypatch.setattr(admitted_input.roster_io, "load_roster", lambda *args, **kwargs: pytest.fail("roster opened"))
    with pytest.raises(ValueError, match="global plan"):
        next(admitted_input.admitted_requests("record", "evidence", "roster", "dataset", None))


def test_v2_one_scope_binds_nine_target_seed_records(monkeypatch, tmp_path):
    original, grant = scope_and_grant(monkeypatch, tmp_path)
    record = json.loads(grant.record_path.read_bytes())
    analysis = record["protocol"]["decisions"]["confirmatory_analysis"]["value"]
    families = {target: {**analysis, "target": target} for target in ("orin", "thor", "tiger")}
    plan = {
        "schema": "passage-direct-rule8-global-scope/v2",
        "replica_catalog_sha256": "a" * 64,
        "confirmatory_analyses": families,
    }
    raw = json.dumps(plan, sort_keys=True).encode()
    path = tmp_path / "scope-v2.json"
    path.write_bytes(raw)
    scope = rule8_scope.load_rule8_scope(path, hashlib.sha256(raw).hexdigest())
    assert original.check_record(grant)  # v1 behavior remains supported
    for target in ("orin", "thor", "tiger"):
        for seed in (1234, 2026, 3407):
            current = json.loads(json.dumps(record))
            current["protocol"]["decisions"]["seeds"]["value"] = [seed]
            current["protocol"]["decisions"]["confirmatory_analysis"]["value"] = families[target]
            data = json.dumps(current, sort_keys=True).encode()
            grant.record_path = tmp_path / f"{target}-{seed}.json"
            grant.record_path.write_bytes(data)
            grant.record_sha256 = hashlib.sha256(data).hexdigest()
            assert scope.check_record(grant, target=target) == current
            wrong = "thor" if target != "thor" else "orin"
            with pytest.raises(ValueError, match="record target differs"):
                scope.check_record(grant, target=wrong)


def test_v2_rejects_missing_or_substituted_analysis_before_roster(monkeypatch, tmp_path):
    _, grant = scope_and_grant(monkeypatch, tmp_path)
    record = json.loads(grant.record_path.read_bytes())
    analysis = record["protocol"]["decisions"]["confirmatory_analysis"]["value"]
    families = {target: {**analysis, "target": target} for target in ("orin", "thor", "tiger")}
    families["tiger"]["seeds"] = [1234, 2026]
    plan = {
        "schema": "passage-direct-rule8-global-scope/v2",
        "replica_catalog_sha256": "a" * 64,
        "confirmatory_analyses": families,
    }
    raw = json.dumps(plan).encode()
    path = tmp_path / "substituted-scope.json"
    path.write_bytes(raw)
    scope = rule8_scope.load_rule8_scope(path, hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(admitted_stream, "grant_test_admission", lambda *a: grant)
    monkeypatch.setattr(admitted_stream.roster_io, "load_roster", lambda *a, **k: pytest.fail("roster opened"))
    with pytest.raises(ValueError, match="target analysis differs"):
        admitted_stream.prepare_admitted_session(
            "record", "evidence", "roster", "dataset", None, None, None, rule8_scope=scope
        )
