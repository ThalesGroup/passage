"""G2 runner equivalence, no data or target access."""
import hashlib
import json

import pytest

from passage_eval.campaign.direct.runner_binding import runner_identity, verify_g2_runner
from passage_eval.io import pinned


def _write(path, value):
    raw = pinned.canonical_json(value)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_runner_normalizes_only_session_values():
    valid = ["python", "-m", "runner", "--config", "/val/config", "--config-sha256", "a" * 64,
             "--output", "/val/out", "--split", "validation", "--batch", "1"]
    test = ["python", "-m", "runner", "--config", "/test/config", "--config-sha256", "b" * 64,
            "--output", "/test/out", "--split", "test", "--batch", "1"]
    assert runner_identity(valid) == runner_identity(test)
    assert runner_identity(test[:-1] + ["2"]) != runner_identity(valid)
    with pytest.raises(ValueError, match="session argument"):
        runner_identity(["python", "--config", "--batch", "1"])


def test_pinned_g2_runner_and_tiger_rootfs(tmp_path, monkeypatch):
    from passage_eval.campaign.direct import runner_binding
    monkeypatch.setattr(runner_binding, "verify_tiger_observation", lambda *_args: True)
    class Grant:
        evidence_dir = tmp_path
        def is_genuine(self):
            return True
        def ensure_unchanged(self):
            return None
    grant = Grant()
    validation = ["python", "-m", "runner", "--config", "/validation", "--output", "/out-a"]
    image_digest = "sha256:" + "f" * 64
    image = {"target": "tiger", "procedure": "seg", "runner_argv": validation,
             "runner_argv_sha256": hashlib.sha256(json.dumps(validation, separators=(",", ":")).encode()).hexdigest(),
             "container_digest": image_digest, "rootfs_archive_sha256": "c" * 64,
             "rootfs_exceptions": {"entry": {"oci": "a", "export": "b"}},
             "scope": "/scope", "pid": 42, "namespace_pid": 1, "job_id": "job:0123456789abcdef"}
    image_ref = {"path": "image.json", "sha256": _write(tmp_path / "image.json", image)}
    sid = "a" * 32
    events = [{"kind": kind, "session_id": sid, "namespace_pid": 1} for kind in ("session_start", "session_complete")]
    raw = b"".join(json.dumps(event).encode() + b"\n" for event in events)
    (tmp_path / "journal.jsonl").write_bytes(raw)
    journal_sha = hashlib.sha256(raw).hexdigest()
    finish = {"scope": "/scope", "runner_pid": 42, "container_digest": image_digest, "runner_session_uuid": sid, "exit_code": 0}
    finish_ref = {"path": "finish.json", "sha256": _write(tmp_path / "finish.json", finish)}
    drain = {"schema": "passage-direct-session-drain/v1", "target": "tiger", "procedure": "seg", "scope": "/scope",
             "job_id": "job:0123456789abcdef", "unit_inactive": True, "cgroup_empty": True, "session_processes": [],
             "handle_evidence": "process_exit_closes_session_descriptors", "runtime_synchronize_close_recorded": True,
             "driver_context_introspection": False}
    drain_ref = {"path": "drain.json", "sha256": _write(tmp_path / "drain.json", drain)}
    entry = {"runner_argv": validation, "rootfs_export_sha256": "c" * 64,
             "rootfs_exceptions": {"entry": {"oci": "a", "export": "b"}}, "image_start": image_ref,
             "image_finish": finish_ref, "drain": drain_ref, "journal": {"path": "journal.jsonl", "sha256": journal_sha},
             "native_execution": {"session_id": journal_sha}}
    receipt = {"target": "tiger", "split": "validation", "procedures": {"seg": entry}}
    digest = _write(tmp_path / "receipt.json", receipt)
    record = {"evidence": {"edge": {"tiger": {"path": "receipt.json", "sha256": digest}}},
              "candidates": {"seg": {"targets": {"tiger": {"container_digest": image_digest}}}}}
    grant.record_path = tmp_path / "record.json"
    grant.record_sha256 = _write(grant.record_path, record)
    argv = ["python", "-m", "runner", "--config", "/test", "--output", "/out-b"]
    assert verify_g2_runner(grant, "tiger", "seg", argv, rootfs_export_sha256="c" * 64,
                            rootfs_exceptions={"entry": {"oci": "a", "export": "b"}})
    with pytest.raises(ValueError, match="rootfs differs"):
        verify_g2_runner(grant, "tiger", "seg", argv, rootfs_export_sha256="d" * 64,
                         rootfs_exceptions={"entry": {"oci": "a", "export": "b"}})
    with pytest.raises(ValueError, match="runner differs"):
        verify_g2_runner(grant, "tiger", "seg", argv + ["--new-option"], rootfs_export_sha256="c" * 64,
                         rootfs_exceptions={"entry": {"oci": "a", "export": "b"}})
    substituted = {**entry, "runner_argv": validation + ["--unobserved"]}
    receipt2 = {"target": "tiger", "split": "validation", "procedures": {"seg": substituted}}
    digest2 = _write(tmp_path / "receipt-substituted.json", receipt2)
    record2 = {**record, "evidence": {"edge": {"tiger": {"path": "receipt-substituted.json", "sha256": digest2}}}}
    grant.record_path = tmp_path / "record-substituted.json"
    grant.record_sha256 = _write(grant.record_path, record2)
    with pytest.raises(ValueError, match="not linked to observed"):
        verify_g2_runner(grant, "tiger", "seg", argv, rootfs_export_sha256="c" * 64,
                         rootfs_exceptions={"entry": {"oci": "a", "export": "b"}})
    for index, job in enumerate((None, "job:broken")):
        image_bad = {**image, "job_id": job}
        drain_bad = {**drain, "job_id": job}
        image_bad_ref = {"path": f"bad-image-{index}", "sha256": _write(tmp_path / f"bad-image-{index}", image_bad)}
        drain_bad_ref = {"path": f"bad-drain-{index}", "sha256": _write(tmp_path / f"bad-drain-{index}", drain_bad)}
        entry_bad = {**entry, "image_start": image_bad_ref, "drain": drain_bad_ref}
        receipt_bad = {"target": "tiger", "split": "validation", "procedures": {"seg": entry_bad}}
        receipt_name = f"bad-receipt-{index}"
        receipt_sha = _write(tmp_path / receipt_name, receipt_bad)
        record_bad = {**record, "evidence": {"edge": {"tiger": {"path": receipt_name, "sha256": receipt_sha}}}}
        grant.record_path = tmp_path / f"bad-record-{index}"
        grant.record_sha256 = _write(grant.record_path, record_bad)
        with pytest.raises(ValueError):
            verify_g2_runner(grant, "tiger", "seg", argv, rootfs_export_sha256="c" * 64,
                             rootfs_exceptions={"entry": {"oci": "a", "export": "b"}})
    orin_image = {**image, "target": "orin"}
    orin_image_ref = {"path": "orin-image", "sha256": _write(tmp_path / "orin-image", orin_image)}
    orin_entry = {**entry, "image_start": orin_image_ref}
    orin_receipt = {"target": "orin", "split": "validation", "procedures": {"seg": orin_entry}}
    orin_receipt_sha = _write(tmp_path / "orin-receipt", orin_receipt)
    orin_record = {"evidence": {"edge": {"orin": {"path": "orin-receipt", "sha256": orin_receipt_sha}}},
                   "candidates": {"seg": {"targets": {"orin": {"container_digest": image_digest}}}}}
    grant.record_path = tmp_path / "orin-record"
    grant.record_sha256 = _write(grant.record_path, orin_record)
    with pytest.raises(ValueError, match="not linked to observed"):
        verify_g2_runner(grant, "orin", "seg", argv)


def test_tiger_observation_is_revalidated_against_retained_proof():
    tiger = pytest.importorskip("passage_eval.campaign.direct.rule8_bwrap")
    from passage_eval.campaign.direct.runner_binding import verify_tiger_observation

    job = "job:0123456789abcdef"
    scope = "passage-edge-tiger-0123456789abcdef.scope"
    digest = "sha256:" + "a" * 64
    config, index = "b" * 64, "c" * 64
    argv = ["python", "-m", "passage_eval.campaign.direct.direct_target", "--config", "/cfg",
            "--config-sha256", config, "--output", "/out"]
    proof = {"job_id": job,
             "comparison": {"manifest_digest": digest, "config_digest": "sha256:" + "d" * 64,
                            "rootfs_archive_sha256": "e" * 64, "exceptions": {}, "export_paths": 3},
             "tree": {"rootfs_archive_sha256": "e" * 64, "file_count": 3, "rootfs_path": "/rootfs"},
             "process": {"target": "tiger", "rootfs": "/rootfs", "job_id": job, "scope": scope,
                         "runner": {"pid": 42, "namespace_pid": 2, "root_mount_options": "ro,nosuid", "argv": argv}}}
    image = tiger.verify_running_bwrap(proof, digest, config_sha256=config, index_sha256=index,
                                       process_in_scope=lambda *_: True)
    image["bwrap_proof"] = proof
    assert verify_tiger_observation(image)
    image["rootfs_archive_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="bwrap observation differs"):
        verify_tiger_observation(image)
