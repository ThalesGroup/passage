"""Validation and evaluated snapshots of one revision have distinct receipts."""

import datetime as dt
from pathlib import Path
from passage_eval.protocol import admission as A


def checker(tmp_path):
    rec = A.template_record()
    rec["protocol"]["profile"] = "rule8_native_persistent"
    d = rec["protocol"]["decisions"]

    def put(k, v):
        d[k]["value"] = v
        d[k]["status"] = "sealed"

    put("timing_profile", "native_raw_path_persistent_v1")
    put("dataset_root_sha256", "a" * 64)
    put("warmup", {"salt": "fixed", "k": 1, "repeats": 16, "dataset_root_sha256": "b" * 64})
    put("dataset_revision", {"repo": "repo", "commit": "c" * 40, "config": "256x256", "resolution": 256})
    put("evaluation_cell", {"cost_model": "elevation", "cost_weight": 62.4, "task_variant": "obstacles"})
    return A._Checker(rec, tmp_path, Path("unused"), dt.datetime.now(dt.timezone.utc))


def roster(root, split="validation", commit="c" * 40):
    return {
        "dataset": {"repo": "repo", "commit": commit, "config": "256x256", "resolution": 256, "split": split},
        "dataset_root_identity": {"sha256": root},
        "cell": {"cost_model": "elevation", "cost_weight": 62.4, "task_variant": "obstacles"},
    }


def test_validation_root_is_distinct_from_source_root_and_revision_remains_bound(tmp_path):
    c = checker(tmp_path)
    g = A._Gate("G2")
    assert A.decision_problem("warmup", c.decision("warmup"), c.decisions, profile="rule8_native_persistent") is None
    c._roster_identity_problems(g, "native_measured", roster("b" * 64))
    assert not g.reasons
    c._roster_identity_problems(g, "native_measured", roster("d" * 64))
    assert any(r["code"] == "native_measured_roster_dataset_root_differs" for r in g.reasons)
    c._roster_identity_problems(g, "native_measured", roster("b" * 64, commit="d" * 40))
    assert any(r["code"] == "native_measured_roster_revision_differs" for r in g.reasons)


def test_source_root_remains_independently_pinned(tmp_path):
    c = checker(tmp_path)
    g = A._Gate("G4")
    c._roster_identity_problems(g, "source", roster("a" * 64, split="test"))
    assert not g.reasons
    c._roster_identity_problems(g, "source", roster("b" * 64, split="test"))
    assert any(r["code"] == "source_roster_dataset_root_differs" for r in g.reasons)


def test_host_verifier_only_commit_keeps_original_runtime_pin(tmp_path):
    import hashlib
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init", "-q")
    git("config", "user.name", "Synthetic")
    git("config", "user.email", "fixture@example.invalid")
    files = {
        "src/passage/__init__.py": b"generator\n",
        "src/passage-eval/__init__.py": b"evaluator\n",
        "src/passage-eval/protocol/admission.py": b"verifier old\n",
    }
    for name, data in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    git("add", ".")
    git("commit", "-qm", "base")
    base = git("rev-parse", "HEAD")
    digest = hashlib.sha256()
    for package, prefix in [("passage", "src/passage/"), ("passage_eval", "src/passage-eval/")]:
        for name in sorted(k for k in files if k.startswith(prefix)):
            digest.update(f"{package}/{name[len(prefix) :]}".encode())
            digest.update(files[name])
    (tmp_path / "src/passage-eval/protocol/admission.py").write_bytes(b"verifier corrected\n")
    git("add", ".")
    git("commit", "-qm", "verifier")
    new = git("rev-parse", "HEAD")
    binding = {"commit": base, "source_sha256": digest.hexdigest()}
    current = hashlib.sha256()
    current_files = {**files, "src/passage-eval/protocol/admission.py": b"verifier corrected\n"}
    for package, prefix in [("passage", "src/passage/"), ("passage_eval", "src/passage-eval/")]:
        for name in sorted(k for k in current_files if k.startswith(prefix)):
            current.update(f"{package}/{name[len(prefix) :]}".encode())
            current.update(current_files[name])
    assert (
        A.host_verifier_runtime_binding(binding, new, verifier_source_sha256=current.hexdigest(), root=tmp_path)
        is not None
    )
    assert A.host_verifier_runtime_binding(binding, new, verifier_source_sha256="f" * 64, root=tmp_path) is None
    assert (
        A.host_verifier_runtime_binding(
            {**binding, "source_sha256": "0" * 64}, new, verifier_source_sha256=current.hexdigest(), root=tmp_path
        )
        is None
    )
    (tmp_path / "src/passage-eval/__init__.py").write_bytes(b"runtime altered\n")
    git("add", ".")
    git("commit", "-qm", "runtime")
    runtime = git("rev-parse", "HEAD")
    assert (
        A.host_verifier_runtime_binding(binding, runtime, verifier_source_sha256=current.hexdigest(), root=tmp_path)
        is None
    )


def test_committed_decoder_tree_uses_runtime_tree(tmp_path):
    import subprocess
    from passage_eval.supervision import treehash

    def git(*args):
        return subprocess.check_output(["git", "-C", str(tmp_path), *args], stderr=subprocess.DEVNULL).decode().strip()

    git("init", "-q")
    git("config", "user.name", "Synthetic")
    git("config", "user.email", "fixture@example.invalid")
    package = tmp_path / "src/passage-eval"
    (package / "protocol").mkdir(parents=True)
    (package / "__init__.py").write_text("runtime\n")
    (package / "protocol/admission.py").write_text("original verifier\n")
    git("add", ".")
    git("commit", "-qm", "runtime")
    base = git("rev-parse", "HEAD")
    expected = treehash.hash_tree(
        package, role="decoder_entry", symlinks="refuse", strict=True, exclude_dirs=treehash.CANDIDATE_EXCLUDES
    ).sha256
    (package / "protocol/admission.py").write_text("updated verifier\n")
    git("add", ".")
    git("commit", "-qm", "verifier")
    assert A.committed_decoder_tree_sha256(base, root=tmp_path) == expected
    (package / "__init__.py").write_text("different runtime\n")
    git("add", ".")
    git("commit", "-qm", "changed runtime")
    assert A.committed_decoder_tree_sha256(git("rev-parse", "HEAD"), root=tmp_path) != expected


def test_reconstruction_tag_does_not_hide_non_evidence_markers():
    good = {"build_command": ["docker", "build", "-t", "local/image:ef68-unqualified", "."]}
    assert A.reconstruction_non_evidence_markers(good) == []
    assert A.reconstruction_non_evidence_markers(
        {"build_command": ["docker", "build", "--tag", "local/fake-unqualified", "."]}
    ) == ["fake"]
    assert "unqualified" in A.reconstruction_non_evidence_markers(
        {"build_command": ["docker", "build", "--build-arg", "unqualified"]}
    )
    assert "synthetic" in A.reconstruction_non_evidence_markers({**good, "description": "synthetic"})
    assert "unqualified" in A.reconstruction_non_evidence_markers({**good, "description": "unqualified"})
    assert A._flags_unqualified({**good, "qualified": False})


def test_supervisor_only_decisions_explicitly_nonapplicable_only_for_native(tmp_path):
    c = checker(tmp_path)
    value = {"not_applicable": "native_direct_persistent"}
    for name in ("deadline_grace_ms", "hard_timeout_ms", "startup_timeout_s", "worker_rlimits", "batching"):
        assert A.decision_problem(name, value, c.decisions, profile="rule8_native_persistent") is None
        assert A.decision_problem(name, value, c.decisions, profile="strict") is not None
        assert (
            A.decision_problem(name, {"not_applicable": "other"}, c.decisions, profile="rule8_native_persistent")
            is not None
        )
    assert (
        A.decision_problem("reference_exclusion_caps", value, c.decisions, profile="rule8_native_persistent")
        is not None
    )
