import pytest
from passage_eval.campaign.direct.direct_host_scoring import verify_drain


def test_drain_is_bound_to_same_job_and_scope():
    good = {
        "schema": "passage-direct-session-drain/v1",
        "target": "orin",
        "procedure": "reg",
        "job_id": "job:0123456789abcdef",
        "scope": "passage-edge-orin-0123456789abcdef.scope",
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    assert (
        verify_drain(
            good,
            target="orin",
            procedure="reg",
            job_id="job:0123456789abcdef",
            scope="passage-edge-orin-0123456789abcdef.scope",
        )
        is True
    )
    with pytest.raises(ValueError, match="drain"):
        verify_drain(
            {**good, "job_id": "job:aaaaaaaaaaaaaaaa"},
            target="orin",
            procedure="reg",
            job_id="job:0123456789abcdef",
            scope="passage-edge-orin-0123456789abcdef.scope",
        )


def test_target_receipt_links_config_index_native_and_journal(tmp_path):
    import json
    import hashlib
    from passage_eval.campaign.direct.direct_host_scoring import verify_target_receipt

    journal = tmp_path / "journal.jsonl"
    journal.write_text("sample\n")

    def digest(b):
        return hashlib.sha256(b).hexdigest()

    pins = {"bundle": "a" * 64, "checkpoint": "b" * 64, "engine": "c" * 64, "decoder": "d" * 64}
    receipt = {
        "schema": "passage-direct-target-validation-result-prototype/v1",
        "configuration_sha256": "e" * 64,
        "input_index_sha256": "f" * 64,
        "journal_sha256": digest(journal.read_bytes()),
        "native_artifact_hashes": pins,
        "runtime_identity": {},
        "qualification_status": "exploratory_unadmitted",
        "timing": {},
        "scope_gpu_drain_verified": False,
    }
    data = json.dumps(receipt).encode()
    (tmp_path / "target-receipt.json").write_bytes(data)
    (tmp_path / "target-receipt.sha256").write_text(digest(data) + "\n")
    assert verify_target_receipt(tmp_path, config_sha256="e" * 64, index_sha256="f" * 64, native_pins=pins) == receipt
    journal.write_text("changed\n")
    with pytest.raises(ValueError, match="journal"):
        verify_target_receipt(tmp_path, config_sha256="e" * 64, index_sha256="f" * 64, native_pins=pins)
