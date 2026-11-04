"""Replica catalogue contracts use synthetic checkpoint identities, never model or dataset contents."""

import copy
from pathlib import Path

import pytest
from passage_eval.io import pinned
from passage_eval.protocol import replicas


FROZEN = {
    "reference_decoders": {"module_sha256": "a" * 64},
    "candidates": {"seg": {"checkpoint_256": {"sha256": "b" * 64}}, "reg": {"checkpoint_256": {"sha256": "c" * 64}}},
}


def catalog():
    return {
        "schema": replicas.SCHEMA,
        "status": "sealed",
        "final_candidates_sha256": "d" * 64,
        "final_decoders_sha256": "a" * 64,
        "replicas": {
            "1234": {"status": "sealed", "seg_checkpoint_sha256": "1" * 64, "reg_checkpoint_sha256": "2" * 64},
            "3407": {"status": "pending", "seg_checkpoint_sha256": None, "reg_checkpoint_sha256": None},
        },
    }


def validate(value):
    return replicas.validate_replica_catalog(value, final_candidates=FROZEN, final_candidates_sha256="d" * 64)


def test_one_replica_seed_can_be_frozen_without_blocking_on_another_seed_or_changing_2026():
    before = copy.deepcopy(FROZEN)
    checked = validate(catalog())
    assert replicas.replica_identity(checked, "seg", 1234)["checkpoint_sha256"] == "1" * 64
    with pytest.raises(replicas.ReplicaCatalogError, match="pending"):
        replicas.replica_identity(checked, "seg", 3407)
    assert FROZEN == before


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.update(status="draft"),
        lambda c: c.update(final_candidates_sha256="e" * 64),
        lambda c: c.update(final_decoders_sha256="e" * 64),
        lambda c: c["replicas"]["1234"].update(seg_checkpoint_sha256="b" * 64),
        lambda c: c["replicas"]["1234"].update(reg_checkpoint_sha256="1" * 64),
        lambda c: c["replicas"]["3407"].update(seg_checkpoint_sha256="3" * 64),
    ],
)
def test_catalogue_refuses_unsealed_unknown_or_relabelled_identity(change):
    value = catalog()
    change(value)
    with pytest.raises(replicas.ReplicaCatalogError):
        validate(value)


def released_catalog():
    path = Path(__file__).resolve().parents[2] / "docs/benchmark/replica-candidates-256.json"
    return path, pinned.sha256_bytes(path.read_bytes())


@pytest.mark.parametrize("seed", [1234, 3407])
def test_committed_catalogue_links_real_accepted_identities_without_rewriting_2026(seed):
    path, digest = released_catalog()
    checked = replicas.load_replica_catalog(path, digest)
    for name in ("seg", "reg"):
        identity = replicas.replica_identity(checked, name, seed)
        assert pinned.is_hex64(identity["checkpoint_sha256"])
        assert identity["seed"] == seed
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        replicas.load_replica_catalog(path, "0" * 64)


@pytest.mark.parametrize("seed", [1234, 3407])
def test_adapter_receipt_requires_the_explicit_pinned_seed(seed, tmp_path):
    from passage_eval.campaign.edge import EdgeRelayError
    from passage_eval.campaign.edge.targets import validate_native_adapter
    from tests.passage_eval import edge_world

    receipt = edge_world.receipt("b" * 64, edge_world.native_artifacts(tmp_path))
    receipt["seed"] = seed
    with pytest.raises(EdgeRelayError, match="expected seed"):
        validate_native_adapter(receipt)
    assert validate_native_adapter(receipt, expected_seed=seed)["seed"] == seed


@pytest.mark.parametrize("seed", [1234, 3407])
def test_validation_package_requires_catalogue_and_sends_only_candidate_identity(seed, tmp_path):
    from passage_eval.campaign.edge import PackageError
    from passage_eval.campaign.edge.package import build_input_package, verify_package
    from passage_eval.campaign.edge.targets import frozen_candidate_identity
    from tests.passage_eval import edge_world

    path, digest = released_catalog()
    identity = frozen_candidate_identity("seg", seed=seed, replica_catalog=path, replica_catalog_sha256=digest)
    candidate = {key: identity[key] for key in ("seed", "checkpoint_sha256")}
    # Metadata-only synthetic validation identity; no dataset is opened by the package boundary.
    from passage_eval.io import roster as roster_io
    from unittest.mock import patch

    roster = {"dataset": {"split": "validation"}, "roster_sha256": "a" * 64, "tasks": [{"sample_id": "validation-000"}]}
    args = dict(
        target="thor",
        plan_sha256=edge_world.PLAN_SHA256,
        procedure="seg",
        inputs=edge_world.inputs(),
        validation_roster=roster,
        candidate=candidate,
    )
    with patch.object(roster_io, "validate_roster", return_value=roster):
        with pytest.raises(PackageError, match="pinned catalogue"):
            build_input_package(tmp_path / "missing", **args)
        assert not (tmp_path / "missing").exists()
        package = build_input_package(tmp_path / "package", **args, replica_catalog=path, replica_catalog_sha256=digest)
    view = verify_package(package.root, package.sha256)
    assert view.manifest["candidate"] == candidate
    assert "replica_catalog" not in view.manifest
