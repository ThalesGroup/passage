"""Candidate identity is measured from file bytes; the supervisor never imports candidate code."""

import hashlib
import json
import sys

import pytest

from pathlib import Path

from passage_eval.supervision import identity, sandbox, treehash


@pytest.fixture
def candidate_tree(tmp_path):
    root = tmp_path / "cand"
    (root / "pkg" / "sub").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text("raise RuntimeError('package imported by the supervisor')\n")
    (root / "pkg" / "sub" / "__init__.py").write_text("raise RuntimeError('subpackage imported by the supervisor')\n")
    (root / "pkg" / "sub" / "mod.py").write_text("raise RuntimeError('module imported by the supervisor')\n")
    (root / "flat.py").write_text("raise RuntimeError('flat module imported by the supervisor')\n")
    return root


def probe_for(root, entries):
    return sandbox.probe_interpreter(
        sys.executable, candidate_paths=[root], top_levels=identity.top_level_names(entries)
    )


def test_files_are_hashed_without_importing_the_candidate(candidate_tree):
    entries = ["pkg.sub.mod:Planner", "flat:fn"]
    probe = probe_for(candidate_tree, entries)
    nested = identity.measure_component(entries[0], {}, probe)
    flat = identity.measure_component(entries[1], {}, probe)

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    assert nested["modules"] == {
        "pkg": {
            "path": str((candidate_tree / "pkg" / "__init__.py").resolve()),
            "sha256": sha(candidate_tree / "pkg" / "__init__.py"),
        },
        "pkg.sub": {
            "path": str((candidate_tree / "pkg" / "sub" / "__init__.py").resolve()),
            "sha256": sha(candidate_tree / "pkg" / "sub" / "__init__.py"),
        },
        "pkg.sub.mod": {
            "path": str((candidate_tree / "pkg" / "sub" / "mod.py").resolve()),
            "sha256": sha(candidate_tree / "pkg" / "sub" / "mod.py"),
        },
    }
    assert list(flat["modules"]) == ["flat"]
    for name in ("pkg", "flat", "pkg.sub.mod"):
        assert name not in sys.modules  # nothing was imported here either


def test_identity_record_is_scorer_compatible_and_changes_with_file_bytes(candidate_tree):
    entry = "flat:fn"
    probe = probe_for(candidate_tree, [entry])

    def build():
        return identity.build_identity(
            {"planner": identity.measure_component(entry, {}, probe)},
            entry_label=entry,
            bundle=None,
            declared_source_sha256=None,
        )

    before = build()
    assert before["schema"] == "passage-candidate-identity-v1" and before["identity_method"] == identity.IDENTITY_METHOD
    assert before["source_verification"] == "file_content_measured" and before["loaded_code_sha256"] == {}
    assert json.loads(json.dumps(before)) == before  # survives the manifest round trip
    (candidate_tree / "flat.py").write_text("# changed\n")
    assert identity.identity_digest(build()) != identity.identity_digest(before)


def test_unlocatable_modules_fail_closed(candidate_tree):
    probe = probe_for(candidate_tree, ["flat:fn", "pkg.missing:fn", "nowhere:fn"])
    with pytest.raises(identity.IdentityError, match="cannot locate files"):
        identity.measure_component("pkg.missing:fn", {}, probe)
    with pytest.raises(identity.IdentityError, match="cannot locate top-level"):
        identity.measure_component("nowhere:fn", {}, probe)
    for bad in ("no_colon", "a:b:c", ":x", "x:", "a-b:c", "1a:b"):
        with pytest.raises(identity.IdentityError):
            identity.split_entry(bad)


def test_installed_modules_are_found_through_the_worker_interpreters_own_finders(tmp_path):
    entry = "passage_eval.baselines.classical:BoundedExactPlanner"
    probe = probe_for(tmp_path, [entry])
    component = identity.measure_component(entry, {}, probe)
    assert list(component["modules"]) == ["passage_eval", "passage_eval.baselines", "passage_eval.baselines.classical"]


def test_checkpoint_files_are_hashed_and_must_exist(candidate_tree, tmp_path):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"weights")
    probe = probe_for(candidate_tree, ["flat:fn"])
    component = identity.measure_component("flat:fn", {"checkpoint": str(checkpoint), "lr": 0.1}, probe)
    assert component["checkpoints"]["checkpoint"]["sha256"] == hashlib.sha256(b"weights").hexdigest()
    with pytest.raises(identity.IdentityError, match="readable local artifact"):
        identity.measure_component("flat:fn", {"bundle_path": str(tmp_path / "absent")}, probe)


def test_runtime_bundle_is_verified(tmp_path):
    helper = tmp_path / "helper.bin"
    helper.write_bytes(b"one")
    bundle = tmp_path / "bundle.json"

    def write(digest):
        bundle.write_text(
            json.dumps({"schema": "passage-runtime-bundle-v1", "files": [{"path": helper.name, "sha256": digest}]})
        )

    write(hashlib.sha256(b"one").hexdigest())
    record = identity.runtime_bundle(str(bundle))
    assert record["files"] == {helper.resolve().as_posix(): hashlib.sha256(b"one").hexdigest()}
    assert identity.runtime_bundle(None) is None
    helper.write_bytes(b"two")
    with pytest.raises(identity.IdentityError, match="identity mismatch"):
        identity.runtime_bundle(str(bundle))
    bundle.write_text(json.dumps({"schema": "wrong", "files": []}))
    with pytest.raises(identity.IdentityError, match="passage-runtime-bundle-v1"):
        identity.runtime_bundle(str(bundle))


def test_declared_source_hash_format_is_enforced(candidate_tree):
    probe = probe_for(candidate_tree, ["flat:fn"])
    components = {"planner": identity.measure_component("flat:fn", {}, probe)}
    for bad in ("A" * 64, "a" * 10, 5):
        with pytest.raises(identity.IdentityError, match="64-character"):
            identity.build_identity(components, entry_label="flat:fn", bundle=None, declared_source_sha256=bad)
    assert (
        identity.build_identity(components, entry_label="flat:fn", bundle=None, declared_source_sha256="a" * 64)[
            "declared_source_sha256"
        ]
        == "a" * 64
    )


def test_loaded_files_must_be_inside_the_hashed_set(candidate_tree):
    probe = probe_for(candidate_tree, ["flat:fn"])
    record = identity.build_identity(
        {"planner": identity.measure_component("flat:fn", {}, probe)},
        entry_label="flat:fn",
        bundle=None,
        declared_source_sha256=None,
    )
    flat = str((candidate_tree / "flat.py").resolve())
    identity.check_loaded_files("planner", [flat], record)
    with pytest.raises(identity.IdentityError, match="outside the hashed identity"):
        identity.check_loaded_files("planner", [flat, "/usr/lib/python3/other.py"], record)
    with pytest.raises(identity.IdentityError):
        identity.check_loaded_files("planner", [], record)


def test_probe_reports_runtime_without_candidate_execution(candidate_tree):
    probe = probe_for(candidate_tree, ["flat:fn"])
    assert probe["modules"]["flat"]["origin"] == str(candidate_tree / "flat.py")
    assert str(candidate_tree) not in probe["sys_path"]  # candidate dirs are not runtime binds
    layout = sandbox.static_runtime_layout(sys.executable, trusted_roots=sandbox.evaluator_roots())  # read from files
    assert "/usr" in layout.roots and all(Path(root).is_absolute() for root in layout.roots)
    assert not any(candidate_tree == Path(root) or candidate_tree in Path(root).parents for root in layout.roots)
    assert sandbox.uncovered_probe_paths(probe, [*layout.roots, candidate_tree]) == []
    assert probe["executed_in"] == "unsandboxed_process_level_same_as_worker" and probe["startup_output"] == ""
    assert [str(p) for p in sandbox.uncovered_probe_paths(probe, ["/usr"])]  # the venv is not under /usr


def test_path_view_maps_original_paths_to_the_bound_bytes(tmp_path):
    view = identity.PathView()
    view.add("/orig/pkg", tmp_path / "staged" / "pkg")
    view.add("/orig/w.bin", tmp_path / "staged" / "w.bin")
    assert view("/orig/pkg/sub/m.py") == tmp_path / "staged" / "pkg" / "sub" / "m.py"
    assert view("/orig/pkg") == tmp_path / "staged" / "pkg" and view("/orig/w.bin") == tmp_path / "staged" / "w.bin"
    assert view("/elsewhere/x.py") == Path("/elsewhere/x.py") and view("/orig/pkg2/x.py") == Path("/orig/pkg2/x.py")
    assert view.covers("/orig/pkg/a") and not view.covers("/orig/pkg2/a")


def test_module_files_are_hashed_through_the_view_so_the_bound_bytes_are_the_hashed_bytes(candidate_tree, tmp_path):
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "flat.py").write_text("# the bytes that are bound\n")
    view = identity.PathView()
    view.add(candidate_tree / "flat.py", staged / "flat.py")
    probe = probe_for(candidate_tree, ["flat:fn"])
    component = identity.measure_component("flat:fn", {}, probe, view)
    assert component["modules"]["flat"]["sha256"] == hashlib.sha256(b"# the bytes that are bound\n").hexdigest()
    assert (
        component["modules"]["flat"]["sha256"] != hashlib.sha256((candidate_tree / "flat.py").read_bytes()).hexdigest()
    )


def test_loaded_files_inside_a_bound_tree_are_covered_by_its_digest(candidate_tree, tmp_path):
    probe = probe_for(candidate_tree, ["flat:fn"])
    component = identity.measure_component("flat:fn", {}, probe)
    helper = candidate_tree / "helper.py"
    helper.write_text("1\n")
    plain = identity.build_identity(
        {"planner": component}, entry_label="flat:fn", bundle=None, declared_source_sha256=None
    )
    with pytest.raises(identity.IdentityError, match="outside the hashed identity"):
        identity.check_loaded_files("planner", [str(helper)], plain)
    tree = treehash.hash_tree(candidate_tree, role="candidate_in_place", exclude_dirs=("__pycache__",))
    covered = identity.build_identity(
        {"planner": component},
        entry_label="flat:fn",
        bundle=None,
        declared_source_sha256=None,
        bound_trees={"binding": "in_place_hash_rechecked", "candidate": [tree.to_manifest()], "environment": []},
        binding="in_place_hash_rechecked",
    )
    identity.check_loaded_files("planner", [str(helper), str(candidate_tree / "flat.py")], covered)
    with pytest.raises(identity.IdentityError):
        identity.check_loaded_files("planner", [str(helper), "/usr/lib/python3/other.py"], covered)
    with pytest.raises(identity.IdentityError):
        identity.build_identity(
            {"planner": component}, entry_label="x", bundle=None, declared_source_sha256=None, binding="other"
        )


def test_runtime_bundle_members_are_declared_then_verified_against_the_bound_copy(tmp_path):
    helper = tmp_path / "helper.bin"
    helper.write_bytes(b"one")
    bundle = tmp_path / "bundle.json"
    bundle.write_text(
        json.dumps(
            {
                "schema": "passage-runtime-bundle-v1",
                "files": [{"path": helper.name, "sha256": hashlib.sha256(b"one").hexdigest()}],
            }
        )
    )
    assert identity.declared_bundle_members(None) == []
    assert identity.declared_bundle_members(str(bundle)) == [bundle.resolve(), helper.resolve()]
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / "helper.bin").write_bytes(b"two")  # the bound copy differs from what the manifest lists
    (staged / "bundle.json").write_bytes(bundle.read_bytes())
    view = identity.PathView()
    view.add(helper, staged / "helper.bin")
    view.add(bundle, staged / "bundle.json")
    with pytest.raises(identity.IdentityError, match="identity mismatch"):
        identity.runtime_bundle(str(bundle), view)
    (staged / "helper.bin").write_bytes(b"one")
    assert identity.runtime_bundle(str(bundle), view)["files"] == {
        helper.resolve().as_posix(): hashlib.sha256(b"one").hexdigest()
    }
    bundle.write_text("{ not json")
    with pytest.raises(identity.IdentityError, match="unreadable"):
        identity.declared_bundle_members(str(bundle))


def test_decoder_entry_tree_is_the_top_level_package_or_module_file(candidate_tree):
    entries = ["pkg.sub.mod:Planner", "flat:fn"]
    probe = probe_for(candidate_tree, entries)
    assert identity.entry_package_path(entries[0], probe) == (candidate_tree / "pkg").resolve()
    assert identity.entry_package_path(entries[1], probe) == (candidate_tree / "flat.py").resolve()
    package = identity.decoder_entry_tree(entries[0], probe)
    assert package.role == "decoder_entry" and package.kind == "dir" and package.files == 3
    assert package.sha256 == treehash.hash_tree(candidate_tree / "pkg", exclude_dirs=("__pycache__",)).sha256
    single = identity.decoder_entry_tree(entries[1], probe)
    assert single.kind == "file" and single.path == str((candidate_tree / "flat.py").resolve())
    (candidate_tree / "pkg" / "sub" / "mod.py").write_text("# helper of the decoder package changed\n")
    assert identity.decoder_entry_tree(entries[0], probe).sha256 != package.sha256
    with pytest.raises(identity.IdentityError, match="cannot locate top-level"):
        identity.entry_package_path("nowhere:fn", probe_for(candidate_tree, ["flat:fn"]))
