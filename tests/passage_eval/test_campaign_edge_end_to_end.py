"""A synthetic target round trip never moves scoring to the target."""

from __future__ import annotations

import json
import copy

import pytest

from tests.passage_eval import edge_world as world


def _validation_request(tmp_path, monkeypatch, *, split="validation", candidate_override=None):
    """A two-procedure native rehearsal without a campaign roster or plan reader."""
    from passage_eval.campaign.plan import ValidationEdgeRelayRequest
    from passage_eval.campaign.edge.package import build_input_package
    from passage_eval.campaign.edge.target import run_target
    from passage_eval.campaign.edge import targets
    from passage_eval.campaign.edge import validation_control
    from passage_eval.io import roster as roster_io
    from tests.passage_eval import campaign_world

    artifacts = world.native_artifacts(tmp_path)
    validation_roster = campaign_world.reference_world(tmp_path / "validation-roster", split="validation").admissible
    validation_dataset = tmp_path / "validation-roster" / "dataset"
    scene = roster_io.load_roster_scenes(validation_roster, validation_dataset)[0]
    expected_candidate = {
        "checkpoint_sha256": __import__("hashlib").sha256(artifacts["checkpoint"].read_bytes()).hexdigest(),
        "seed": 2026,
    }
    monkeypatch.setattr(
        targets, "frozen_candidate_identity", lambda procedure: dict(expected_candidate, procedure=procedure)
    )
    monkeypatch.setattr(
        validation_control,
        "_CONTROL",
        validation_control.ValidationRelayControl(
            roster_path=tmp_path / "validation-roster" / "evidence" / "admissible-roster.json",
            roster_file_sha256=campaign_world._sha(
                tmp_path / "validation-roster" / "evidence" / "admissible-roster.json"
            ),
            roster_sha256=validation_roster["roster_sha256"],
            dataset_root=validation_dataset,
            pinned_receipt_sha256=validation_roster["dataset_root_identity"]["sha256"],
        ),
        raising=False,
    )
    inputs = {
        "sample_ids": [scene.sample_id],
        "terrain": [scene.elevation.tolist()],
        "obstacles": [scene.obstacles.tolist()],
        "starts": [list(scene.start)],
        "goals": [list(scene.goal)],
        "input_tensor": [scene.input_tensor.tolist()],
    }
    packages = {
        procedure: build_input_package(
            tmp_path / f"package-{procedure}",
            target="thor",
            plan_sha256=world.PLAN_SHA256,
            procedure=procedure,
            inputs=inputs,
            validation_roster=validation_roster,
            candidate=candidate_override or expected_candidate,
        )
        for procedure in ("seg", "reg")
    }

    def native_relay(procedure, package):
        receipt = dict(world.receipt(package.sha256, artifacts), procedure=procedure)
        return run_target(
            package.root,
            package.sha256,
            receipt=receipt,
            output=tmp_path / f"result-{procedure}",
            engine_or_model=artifacts["engine_or_model"],
            checkpoint=artifacts["checkpoint"],
            final_decoders=artifacts["final_decoders"],
            adapter_command=world.adapter(tmp_path),
        ).root

    scores = []

    def scorer(procedure, outputs):
        scores.append(procedure)
        return {"procedure": procedure, "paths": outputs["paths"]}

    return ValidationEdgeRelayRequest(
        split=split,
        target="thor",
        procedures=("seg", "reg"),
        seed=2026,
        candidates={procedure: expected_candidate for procedure in ("seg", "reg")},
        packages=packages,
        native_relay=native_relay,
        scorer=scorer,
    ), scores


def test_validation_edge_relay_runs_the_fixed_scope_and_scores_collected_native_results_on_host(tmp_path, monkeypatch):
    """The pre-G3 relay has no plan or roster input: only sealed validation packages reach the target."""
    from passage_eval.campaign.runner import run_validation_edge_relay

    request, scores = _validation_request(tmp_path, monkeypatch)
    result = run_validation_edge_relay(request)

    assert result.target == "thor" and result.seed == 2026 and result.plan_sha256 == world.PLAN_SHA256
    assert scores == ["seg", "reg"]
    assert result.scores == {
        "seg": {"procedure": "seg", "paths": [[[0, 0], [0, 1]]]},
        "reg": {"procedure": "reg", "paths": [[[0, 0], [0, 1]]]},
    }


def test_validation_edge_relay_refuses_test_scope_without_invoking_a_native_relay(tmp_path, monkeypatch):
    """The validation-only record has no test-roster or test-plan escape hatch."""
    from passage_eval.campaign.plan import PlanError
    from passage_eval.campaign.runner import run_validation_edge_relay

    request, scores = _validation_request(tmp_path, monkeypatch, split="test")
    with pytest.raises(PlanError, match="validation split only"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_relay_control_requires_configured_paths_and_keeps_fixed_pins(tmp_path, monkeypatch):
    from passage_eval.campaign.edge import validation_control

    monkeypatch.delenv("PASSAGE_VALIDATION_ROSTER", raising=False)
    monkeypatch.delenv("PASSAGE_VALIDATION_DATASET_ROOT", raising=False)
    assert validation_control.validation_relay_control() is None
    roster = tmp_path / "validation-roster.json"
    dataset = tmp_path / "validation-data"
    monkeypatch.setenv("PASSAGE_VALIDATION_ROSTER", str(roster))
    monkeypatch.setenv("PASSAGE_VALIDATION_DATASET_ROOT", str(dataset))
    control = validation_control.validation_relay_control()
    assert control.roster_path == roster and control.dataset_root == dataset
    assert control.roster_file_sha256 == "0f29ab361c2313038a5c17f736d57245b1d067b0569f90b5f2a13c568f94cb49"
    assert control.roster_sha256 == "4a4869a18dd7e5ec099149069c74321a11cdf40450f95bac81d74e8fe054b556"
    assert control.pinned_receipt_sha256 == "78c745d0c51d36d34da99705e0389b3cf4fe717802ffdee7d208aa2285110ed4"
    monkeypatch.setenv("PASSAGE_VALIDATION_ROSTER", "relative.json")
    with pytest.raises(ValueError, match="absolute"):
        validation_control.validation_relay_control()


def test_validation_edge_relay_fails_closed_without_a_reviewed_harness_control(tmp_path, monkeypatch):
    from passage_eval.campaign.edge import validation_control
    from passage_eval.campaign.runner import run_validation_edge_relay
    from passage_eval.io import pinned, roster as roster_io

    request, scores = _validation_request(tmp_path, monkeypatch)
    monkeypatch.setattr(validation_control, "_CONTROL", None, raising=False)
    monkeypatch.delenv("PASSAGE_VALIDATION_ROSTER", raising=False)
    monkeypatch.delenv("PASSAGE_VALIDATION_DATASET_ROOT", raising=False)
    monkeypatch.setattr(pinned, "read_pinned", lambda *_args, **_kwargs: pytest.fail("roster was opened"))
    monkeypatch.setattr(roster_io, "load_roster_scenes", lambda *_args, **_kwargs: pytest.fail("dataset was opened"))

    with pytest.raises(ValueError, match="unconfigured"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_edge_relay_rejects_a_control_root_receipt_mismatch_before_dataset_read(tmp_path, monkeypatch):
    from dataclasses import replace

    from passage_eval.campaign.edge import validation_control
    from passage_eval.campaign.runner import run_validation_edge_relay
    from passage_eval.io import roster as roster_io

    request, scores = _validation_request(tmp_path, monkeypatch)
    control = validation_control.validation_relay_control()
    monkeypatch.setattr(
        validation_control,
        "_CONTROL",
        replace(control, pinned_receipt_sha256="0" * 64),
        raising=False,
    )
    monkeypatch.setattr(roster_io, "load_roster_scenes", lambda *_args, **_kwargs: pytest.fail("dataset was opened"))

    with pytest.raises(ValueError, match="root receipt"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_edge_relay_rejects_a_decoder_identity_mismatch_before_host_scoring(tmp_path, monkeypatch):
    """Collection remains the only path to scoring, including for the pre-G3 rehearsal."""
    from passage_eval.campaign.edge.target import run_target
    from passage_eval.campaign.plan import ValidationEdgeRelayRequest
    from passage_eval.campaign.runner import run_validation_edge_relay
    from passage_eval.io import pinned

    request, scores = _validation_request(tmp_path, monkeypatch)
    package = request.packages["seg"]
    artifacts = world.native_artifacts(tmp_path)
    target = run_target(
        package.root,
        package.sha256,
        receipt=world.receipt(package.sha256, artifacts),
        output=tmp_path / "mismatched-result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    payload = dict(target.payload, receipt=dict(target.payload["receipt"], final_decoders_sha256="0" * 64))
    raw = pinned.canonical_json(payload)
    (target.root / "target-result.json").write_bytes(raw)
    (target.root / "target-result.sha256").write_text(pinned.sha256_bytes(raw) + "\n")

    request = ValidationEdgeRelayRequest(
        split=request.split,
        target=request.target,
        procedures=request.procedures,
        seed=request.seed,
        candidates=request.candidates,
        packages=request.packages,
        native_relay=lambda procedure, _: target.root,
        scorer=request.scorer,
    )
    with pytest.raises(ValueError, match="decoder SHA-256 differs"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_edge_package_refuses_a_test_roster_before_it_can_reach_the_relay(tmp_path):
    from passage_eval.campaign.edge import PackageError
    from passage_eval.campaign.edge.package import build_input_package
    from tests.passage_eval import campaign_world

    test_roster = campaign_world.reference_world(tmp_path / "test-roster", split="test").admissible
    with pytest.raises(PackageError, match="validation roster split"):
        build_input_package(
            tmp_path / "test-package",
            target="thor",
            plan_sha256=world.PLAN_SHA256,
            procedure="seg",
            inputs=dict(world.inputs(), sample_ids=[test_roster["tasks"][0]["sample_id"]]),
            validation_roster=test_roster,
            candidate={"checkpoint_sha256": "0" * 64, "seed": 2026},
        )


def test_validation_edge_relay_rejects_an_arbitrary_candidate_before_native_execution_or_scoring(tmp_path, monkeypatch):
    from passage_eval.campaign.plan import ValidationEdgeRelayRequest
    from passage_eval.campaign.runner import run_validation_edge_relay

    request, scores = _validation_request(tmp_path, monkeypatch)
    request = ValidationEdgeRelayRequest(
        split=request.split,
        target=request.target,
        procedures=request.procedures,
        seed=request.seed,
        candidates={procedure: {"checkpoint_sha256": "0" * 64, "seed": 2026} for procedure in request.procedures},
        packages=request.packages,
        native_relay=request.native_relay,
        scorer=request.scorer,
    )
    with pytest.raises(ValueError, match="request candidate differs"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_edge_relay_rejects_a_rehashed_test_roster_relabelled_as_validation_without_a_control_match(
    tmp_path, monkeypatch
):
    from passage_eval.campaign.edge.package import build_input_package
    from passage_eval.campaign.runner import run_validation_edge_relay
    from passage_eval.io import roster as roster_io
    from tests.passage_eval import campaign_world

    request, scores = _validation_request(tmp_path, monkeypatch)
    test_world = campaign_world.reference_world(tmp_path / "test-roster", split="test")
    relabelled = copy.deepcopy(test_world.admissible)
    relabelled["dataset"]["split"] = "validation"
    relabelled["roster_sha256"] = roster_io.compute_roster_sha256(relabelled)
    test_inputs = dict(world.inputs(), sample_ids=[relabelled["tasks"][0]["sample_id"]])
    forged = build_input_package(
        tmp_path / "forged-test-package",
        target=request.target,
        plan_sha256=world.PLAN_SHA256,
        procedure="seg",
        inputs=test_inputs,
        validation_roster=relabelled,
        candidate=request.candidates["seg"],
    )
    request = request.__class__(
        split=request.split,
        target=request.target,
        procedures=request.procedures,
        seed=request.seed,
        candidates=request.candidates,
        packages={**request.packages, "seg": forged},
        native_relay=request.native_relay,
        scorer=request.scorer,
    )
    with pytest.raises(ValueError, match="trusted validation roster"):
        run_validation_edge_relay(request)
    assert scores == []


def test_validation_edge_relay_rejects_altered_tensors_under_a_validation_id_before_native_execution(
    tmp_path, monkeypatch
):
    from passage_eval.campaign.edge.package import build_input_package, verify_package
    from passage_eval.campaign.plan import ValidationEdgeRelayRequest
    from passage_eval.campaign.runner import run_validation_edge_relay
    from passage_eval.io import roster as roster_io

    request, scores = _validation_request(tmp_path, monkeypatch)
    from passage_eval.campaign.edge import validation_control

    control = validation_control.validation_relay_control()
    roster = roster_io.load_roster(control.roster_path, expected_sha256=control.roster_sha256)
    inputs = copy.deepcopy(verify_package(request.packages["seg"].root, request.packages["seg"].sha256).inputs)
    inputs["terrain"][0][0][0] += 1.0
    tampered = build_input_package(
        tmp_path / "altered-package",
        target=request.target,
        plan_sha256=world.PLAN_SHA256,
        procedure="seg",
        inputs=inputs,
        validation_roster=roster,
        candidate=request.candidates["seg"],
    )
    request = ValidationEdgeRelayRequest(
        split=request.split,
        target=request.target,
        procedures=request.procedures,
        seed=request.seed,
        candidates=request.candidates,
        packages={**request.packages, "seg": tampered},
        native_relay=request.native_relay,
        scorer=request.scorer,
    )
    with pytest.raises(ValueError, match="ordinary inputs differ"):
        run_validation_edge_relay(request)
    assert scores == []


def test_synthetic_target_round_trip_scores_only_after_collection(tmp_path):
    from passage_eval.campaign.edge.collect import score_collected
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    target_result = run_target(
        package.root,
        package.sha256,
        receipt=world.receipt(package.sha256, artifacts),
        output=tmp_path / "result",
        engine_or_model=artifacts["engine_or_model"],
        checkpoint=artifacts["checkpoint"],
        final_decoders=artifacts["final_decoders"],
        adapter_command=world.adapter(tmp_path),
    )
    assert "score" not in target_result.payload
    assert target_result.payload["adapter"]["artifact_sha256"] == {
        "engine_or_model_sha256": target_result.payload["receipt"]["engine_or_model_sha256"],
        "checkpoint_sha256": target_result.payload["receipt"]["checkpoint_sha256"],
        "final_decoders_sha256": target_result.payload["receipt"]["final_decoders_sha256"],
    }
    assert str(artifacts["checkpoint"]) not in json.dumps(target_result.payload)
    score = score_collected(target_result.root, package=package, scorer=lambda run: {"scored_paths": run["paths"]})
    assert score == {"scored_paths": [[[0, 0], [0, 1]]]}


def test_target_entry_seals_only_adapter_output(tmp_path):
    from passage_eval.campaign.edge.__main__ import main

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps(world.receipt(package.sha256, artifacts)))
    assert (
        main(
            [
                "target-run",
                "--package",
                str(package.root),
                "--package-sha256",
                package.sha256,
                "--receipt",
                str(receipt),
                "--engine-or-model",
                str(artifacts["engine_or_model"]),
                "--checkpoint",
                str(artifacts["checkpoint"]),
                "--final-decoders",
                str(artifacts["final_decoders"]),
                "--adapter",
                world.adapter(tmp_path)[0],
                "--adapter-arg",
                world.adapter(tmp_path)[1],
                "--output",
                str(tmp_path / "result"),
            ]
        )
        == 0
    )
    assert (tmp_path / "result" / "target-result.json").exists()


def test_target_rejects_native_artifact_mutated_by_adapter_without_writing_a_result(tmp_path):
    import pytest

    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.target import run_target

    package = world.package(tmp_path / "package")
    artifacts = world.native_artifacts(tmp_path)
    with pytest.raises(EdgeRelayError, match="changed during adapter"):
        run_target(
            package.root,
            package.sha256,
            receipt=world.receipt(package.sha256, artifacts),
            output=tmp_path / "result",
            engine_or_model=artifacts["engine_or_model"],
            checkpoint=artifacts["checkpoint"],
            final_decoders=artifacts["final_decoders"],
            adapter_command=world.mutating_adapter(tmp_path),
        )
    assert not (tmp_path / "result").exists()
