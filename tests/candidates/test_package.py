"""A modified artifact or reused version must never silently execute different code."""

import hashlib
import json
import sys

import pytest


def package_fixture(tmp_path):
    python_minor = f"{sys.version_info.major}.{sys.version_info.minor}"
    (tmp_path / "planner.py").write_text('raise RuntimeError("must never import on validation")\n')
    (tmp_path / "uv.lock").write_text("version = 1\n")
    (tmp_path / "Containerfile").write_text(f"FROM python:{python_minor}\n")
    doc = {
        "schema": "passage-candidate-package/v1",
        "candidate_id": "test-graph",
        "version": "1",
        "family": "graph",
        "entrypoint": "planner:Planner",
        "kwargs": {},
        "python": python_minor,
        "backends": ["cpu"],
        "seed": 2026,
        "dataset": {
            "id": "thalesgroup/passage",
            "revision": "c" * 40,
            "config": "256x256",
            "cost_model": "elevation",
            "cost_weight": 62.4,
        },
        "budget": {"wall_time_ms": 5000},
        "precision": "fp32",
        "training": {"split": "train", "test_usage": "not used for training"},
        "license": "MIT",
        "image": {"recipe": "Containerfile", "reference": None},
        "artifacts": [
            {"path": p, "sha256": hashlib.sha256((tmp_path / p).read_bytes()).hexdigest(), "role": role}
            for p, role in [("planner.py", "code"), ("uv.lock", "lock"), ("Containerfile", "recipe")]
        ],
    }
    path = tmp_path / "candidate.json"
    path.write_text(json.dumps(doc))
    return path, doc


def test_passive_validation_and_idempotent_registration(tmp_path):
    from passage_eval.protocol.candidate_package import load_candidate_package
    from passage_eval.candidates.catalog import register_candidate

    path, _ = package_fixture(tmp_path)
    package = load_candidate_package(path)
    catalog = tmp_path / "catalog.json"
    assert register_candidate(package, catalog)["added"] is True
    assert register_candidate(package, catalog)["added"] is False
    assert len(json.loads(catalog.read_text())["candidates"]) == 1


def test_modified_code_rejected_before_import(tmp_path):
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, _ = package_fixture(tmp_path)
    (tmp_path / "planner.py").write_text('print("changed")')
    with pytest.raises(ValueError, match="hash"):
        load_candidate_package(path)


@pytest.mark.parametrize("change", ["traversal", "symlink", "unknown", "revision", "backend", "budget"])
def test_invalid_packages_fail_closed(tmp_path, change):
    from passage_eval.protocol.candidate_package import load_candidate_package

    path, doc = package_fixture(tmp_path)
    if change == "traversal":
        doc["artifacts"][0]["path"] = "../planner.py"
    if change == "symlink":
        (tmp_path / "planner.py").unlink()
        (tmp_path / "planner.py").symlink_to("/etc/hostname")
    if change == "unknown":
        doc["oracle_cost"] = 0
    if change == "revision":
        doc["dataset"]["revision"] = "main"
    if change == "backend":
        doc["backends"] = []
    if change == "budget":
        doc["budget"]["wall_time_ms"] = float("nan")
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError):
        load_candidate_package(path)


def test_reusing_version_with_new_bytes_is_refused(tmp_path):
    from passage_eval.protocol.candidate_package import load_candidate_package
    from passage_eval.candidates.catalog import register_candidate

    path, doc = package_fixture(tmp_path)
    catalog = tmp_path / "catalog.json"
    register_candidate(load_candidate_package(path), catalog)
    doc["kwargs"] = {"other": True}
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="version"):
        register_candidate(load_candidate_package(path), catalog)


def test_in_memory_manifest_mutation_cannot_relabel_registration(tmp_path):
    from passage_eval.protocol.candidate_package import load_candidate_package
    from passage_eval.candidates.catalog import register_candidate

    path, _ = package_fixture(tmp_path)
    package = load_candidate_package(path)
    package.document["candidate_id"] = "substituted-name"
    with pytest.raises(ValueError, match="manifest"):
        register_candidate(package, tmp_path / "catalog.json")
