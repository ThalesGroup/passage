"""Collection verifies native identity before host scoring."""

from __future__ import annotations

import pytest

from tests.passage_eval import edge_world as world


def test_identity_mismatch_is_rejected_before_host_scorer(tmp_path):
    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.collect import score_collected
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    receipt = world.receipt(package.sha256, artifacts)
    result = run_target(
        package.root,
        package.sha256,
        receipt=receipt,
        output=tmp_path / "result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    receipt["final_decoders_sha256"] = "0" * 64
    raw = __import__("json").dumps({**result.payload, "receipt": receipt}).encode()
    (result.root / "target-result.json").write_bytes(raw)
    (result.root / "target-result.sha256").write_text(__import__("hashlib").sha256(raw).hexdigest() + "\n")
    called = False

    def scorer(_run):
        nonlocal called
        called = True
        return {"score": 1.0}

    with pytest.raises(EdgeRelayError, match="decoder"):
        score_collected(result.root, package=package, scorer=scorer)
    assert not called


def test_tampered_target_result_is_rejected(tmp_path):
    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.collect import verify_collected_run
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    result = run_target(
        package.root,
        package.sha256,
        receipt=world.receipt(package.sha256, artifacts),
        output=tmp_path / "result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    (result.root / "target-result.json").write_text("{}")
    with pytest.raises(EdgeRelayError, match="result digest"):
        verify_collected_run(result.root, package=package)


def test_collector_rejects_private_data_even_when_a_target_reseals_its_result(tmp_path):
    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.collect import verify_collected_run
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    result = run_target(
        package.root,
        package.sha256,
        receipt=world.receipt(package.sha256, artifacts),
        output=tmp_path / "result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    payload = {**result.payload, "outputs": {"paths": [[[0, 0], [0, 1]]], "labels": ["private"]}}
    raw = __import__("json").dumps(payload).encode()
    (result.root / "target-result.json").write_bytes(raw)
    (result.root / "target-result.sha256").write_text(__import__("hashlib").sha256(raw).hexdigest() + "\n")
    with pytest.raises(EdgeRelayError, match="schema"):
        verify_collected_run(result.root, package=package)


def test_collector_binds_receipt_target_to_package_target(tmp_path):
    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.collect import verify_collected_run
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    result = run_target(
        package.root,
        package.sha256,
        receipt=world.receipt(package.sha256, artifacts),
        output=tmp_path / "result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    payload = {**result.payload, "receipt": {**result.payload["receipt"], "target": "orin"}}
    raw = __import__("json").dumps(payload).encode()
    (result.root / "target-result.json").write_bytes(raw)
    (result.root / "target-result.sha256").write_text(__import__("hashlib").sha256(raw).hexdigest() + "\n")
    with pytest.raises(EdgeRelayError, match="target"):
        verify_collected_run(result.root, package=package)
