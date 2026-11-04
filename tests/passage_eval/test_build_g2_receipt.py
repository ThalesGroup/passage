"""Synthetic G2 receipt wiring and substitution checks."""
import hashlib
import json

import pytest

from passage_eval.campaign.direct import build_g2_receipt as builder
from passage_eval.io import pinned


def _ref(root, name, value):
    raw = value if isinstance(value, bytes) else pinned.canonical_json(value)
    (root / name).write_bytes(raw)
    return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}


def test_g2_copies_observed_runner_and_rejects_substitution(tmp_path, monkeypatch):
    from passage_eval.campaign.direct import runner_binding
    monkeypatch.setattr(runner_binding, "verify_tiger_observation", lambda *_args: True)
    target = "tiger"
    image = "sha256:" + "a" * 64
    plan_sha = "b" * 64
    candidates, entries = {}, {}
    for proc in ("seg", "reg"):
        pins = {"bundle": "1" * 64, "checkpoint": ("2" if proc == "seg" else "3") * 64,
                "decoder": "4" * 64, "engine": "5" * 64}
        candidates[proc] = {"checkpoint_sha256": pins["checkpoint"], "final_decoders_sha256": pins["decoder"],
                            "targets": {target: {"bundle_sha256": pins["bundle"], "engine_sha256": pins["engine"],
                                                 "container_digest": image}}}
        argv = ["python", "-m", "passage_eval.campaign.direct.direct_target", "--config", "/cfg", "--output", "/out"]
        session = "c" * 32
        events = [{"kind": kind, "session_id": session, "namespace_pid": 1}
                  for kind in ("session_start", "session_complete")]
        journal = _ref(tmp_path, proc + "-journal", b"".join(json.dumps(e).encode() + b"\n" for e in events))
        config = _ref(tmp_path, proc + "-config", {"target": target, "procedure": proc, "dataset_split": "validation",
                                                   "artifact_pins": pins, "index_sha256": "6" * 64})
        start = _ref(tmp_path, proc + "-start", {"target": target, "procedure": proc, "container_digest": image,
                    "configuration_sha256": config["sha256"], "input_index_sha256": "6" * 64, "scope": "/scope",
                    "pid": 42, "namespace_pid": 1, "job_id": "job:0123456789abcdef", "runner_argv": argv,
                    "runner_argv_sha256": hashlib.sha256(json.dumps(argv, separators=(",", ":")).encode()).hexdigest(),
                    "rootfs_archive_sha256": "7" * 64, "rootfs_exceptions": {"entry": {"oci": "a", "export": "b"}}})
        finish = _ref(tmp_path, proc + "-finish", {"container_digest": image, "scope": "/scope", "runner_pid": 42,
                     "n_processes": 1, "n_restarts": 0, "journal_markers": "matched", "runner_session_uuid": session, "exit_code": 0})
        drain = _ref(tmp_path, proc + "-drain", {"schema": "passage-direct-session-drain/v1", "target": target,
                    "procedure": proc, "job_id": "job:0123456789abcdef", "scope": "/scope", "unit_inactive": True,
                    "cgroup_empty": True, "session_processes": [], "handle_evidence": "process_exit_closes_session_descriptors",
                    "runtime_synchronize_close_recorded": True, "driver_context_introspection": False})
        runtime = {"implementation": "passage_inference.backends.OpenVinoRuntime", "options": {"device": "GPU.0",
                   "execution_devices": ["GPU.0"]}, "versions": {"openvino": {"loaded_module_version": "2026.1",
                   "distributions": {"openvino": "2026.1"}}}}
        receipt = _ref(tmp_path, proc + "-receipt", {"schema": "passage-direct-target-validation-result-prototype/v1", "configuration_sha256": config["sha256"],
                     "input_index_sha256": "6" * 64, "native_artifact_hashes": pins, "journal_sha256": journal["sha256"],
                     "qualification_status": "exploratory_unadmitted", "runtime_identity": runtime,
                     "timing": {"status": "complete", "n_completed": 128, "warmups_completed": 16}})
        execution = {"session_id": journal["sha256"], "container_digest": image, "n_processes": 1, "n_restarts": 0,
                     "artifacts": {"checkpoint_sha256": pins["checkpoint"], "engine_sha256": pins["engine"],
                                   "final_decoders_sha256": pins["decoder"]}}
        manifest = _ref(tmp_path, proc + "-manifest", {"native_execution": execution})
        aggregate = _ref(tmp_path, proc + "-aggregate", {"plan_sha256": plan_sha})
        roster = _ref(tmp_path, proc + "-roster", {})
        warm = _ref(tmp_path, proc + "-warm", {})
        entries[proc] = {"configuration": config, "image_start": start, "image_finish": finish, "drain": drain,
                         "target_receipt": receipt, "journal": journal, "aggregate": aggregate, "run_manifest": manifest,
                         "roster": roster, "warmup_roster": warm, "compiled_manifest": None}
    record = _ref(tmp_path, "record", {"harness": {"commit": "d" * 40}, "candidates": candidates})
    monkeypatch.setattr(builder, "inspect_journal", lambda *_: {"complete": True, "runtime_released": True,
                        "committed_prefix": 128, "ambiguous_range": None})
    out = tmp_path / "g2.json"
    digest = builder.build(record, tmp_path, target, plan_sha, entries, out)
    result = pinned.read_pinned_json(out, digest)
    assert result["procedures"]["seg"]["configuration"] == entries["seg"]["configuration"]
    assert result["procedures"]["seg"]["runner_argv"] == argv
    assert result["procedures"]["seg"]["rootfs_export_sha256"] == "7" * 64
    assert result["procedures"]["seg"]["source_qualification_status"] == "exploratory_unadmitted"
    altered = dict(entries["seg"])
    bad = json.loads((tmp_path / "seg-start").read_bytes())
    bad["runner_argv"] = argv + ["--unexpected"]
    altered["image_start"] = _ref(tmp_path, "bad-start", bad)
    with pytest.raises(ValueError, match="runner argument vector"):
        builder._procedure(tmp_path, pinned.read_pinned_json(tmp_path / "record", record["sha256"]),
                           target, "seg", altered, plan_sha)
    altered = dict(entries["seg"])
    bad_drain = json.loads((tmp_path / "seg-drain").read_bytes())
    bad_drain["job_id"] = "job:fedcba9876543210"
    altered["drain"] = _ref(tmp_path, "bad-drain", bad_drain)
    with pytest.raises(ValueError, match="drain job differs"):
        builder._procedure(tmp_path, pinned.read_pinned_json(tmp_path / "record", record["sha256"]),
                           target, "seg", altered, plan_sha)
    altered = dict(entries["seg"])
    bad_finish = json.loads((tmp_path / "seg-finish").read_bytes())
    bad_finish["exit_code"] = 1
    altered["image_finish"] = _ref(tmp_path, "bad-finish", bad_finish)
    with pytest.raises(ValueError, match="observed image, runner or process"):
        builder._procedure(tmp_path, pinned.read_pinned_json(tmp_path / "record", record["sha256"]),
                           target, "seg", altered, plan_sha)
    for idx, bad_job in enumerate((None, "job:broken")):
        altered = dict(entries["seg"])
        bad_start = json.loads((tmp_path / "seg-start").read_bytes())
        bad_drain = json.loads((tmp_path / "seg-drain").read_bytes())
        bad_start["job_id"] = bad_job
        bad_drain["job_id"] = bad_job
        altered["image_start"] = _ref(tmp_path, f"bad-job-start-{idx}", bad_start)
        altered["drain"] = _ref(tmp_path, f"bad-job-drain-{idx}", bad_drain)
        with pytest.raises(ValueError):
            builder._procedure(tmp_path, pinned.read_pinned_json(tmp_path / "record", record["sha256"]),
                               target, "seg", altered, plan_sha)
    altered = dict(entries["seg"])
    bad_start = json.loads((tmp_path / "seg-start").read_bytes())
    bad_start["pid"] = None
    altered["image_start"] = _ref(tmp_path, "missing-pid-start", bad_start)
    with pytest.raises(ValueError, match="observed image, runner or process"):
        builder._procedure(tmp_path, pinned.read_pinned_json(tmp_path / "record", record["sha256"]),
                           target, "seg", altered, plan_sha)
