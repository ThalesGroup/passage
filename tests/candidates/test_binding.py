import json

import pytest

from .test_scoring import score_fixture


def receipt_fixture(run):
    from passage_eval.protocol.candidate_package import read_json

    doc = read_json(run / "run.json")
    doc["input_file_sha256"] = "e" * 64
    (run / "run.json").write_text(json.dumps(doc))
    receipt = {
        "schema": "passage-rootless-oci-run/v1",
        "manifest_sha256": doc["manifest_sha256"],
        "input_file_sha256": "e" * 64,
        "device": "cpu",
        "gpu_uuid": None,
        "rootfs_receipt_sha256": "a" * 64,
        "image_inspect_id": "sha256:" + "b" * 64,
        "config_sha256": "c" * 64,
        "returncode": 0,
        "timed_out": False,
        "rootfs_unchanged": True,
        "child_namespace_drained": True,
        "private_runtime_state_empty": True,
        "cleanup_errors": [],
        "strict_admission": False,
    }
    (run / "oci-run.json").write_text(json.dumps(receipt))
    return receipt


def test_binding_requires_actual_successful_matching_oci_receipt(tmp_path):
    from passage_eval.campaign.candidate.binding import bind_container_evidence, verify_container_evidence
    from passage_eval.protocol.candidate_package import read_json, file_sha256

    run, _ = score_fixture(tmp_path)
    receipt_fixture(run)
    bind_container_evidence(run)
    doc = read_json(run / "run.json")
    assert doc["container"]["oci_receipt_sha256"] == file_sha256(run / "oci-run.json")
    assert doc["container"]["strict_admission"] is False
    verify_container_evidence(run, doc)
    receipt = read_json(run / "oci-run.json")
    receipt["image_inspect_id"] = "sha256:" + "f" * 64
    (run / "oci-run.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="receipt"):
        verify_container_evidence(run, doc)


@pytest.mark.parametrize(
    "change",
    [
        {"manifest_sha256": "0" * 64},
        {"returncode": 1},
        {"timed_out": True},
        {"child_namespace_drained": None},
        {"private_runtime_state_empty": False},
        {"cleanup_errors": ["unknown"]},
    ],
)
def test_binding_rejects_unproven_execution(tmp_path, change):
    from passage_eval.campaign.candidate.binding import bind_container_evidence

    run, _ = score_fixture(tmp_path)
    receipt = receipt_fixture(run)
    receipt.update(change)
    (run / "oci-run.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        bind_container_evidence(run)
