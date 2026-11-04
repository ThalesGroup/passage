"""C2 and C3: tree hashes, staging, static interpreter layout and the sandboxed interpreter probe."""

from __future__ import annotations

import json
import os
import stat
import sys
import venv
from pathlib import Path

import pytest

from passage_eval.supervision import sandbox, treehash
from passage_eval.supervision.environment import build_bound_set
from passage_eval.supervision.treehash import StageArea, TreeError, TreeHashCache, hash_tree

BWRAP = sandbox.available_bwrap()


def shared_cache():
    """The session-wide digest cache (see conftest): /usr is hashed once per test session, not once per test."""
    return TreeHashCache(os.environ["PASSAGE_TREE_HASH_CACHE"])


needs_bwrap = pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")


def make_tree(root: Path, files: dict[str, str | bytes], executable: tuple[str, ...] = ()) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
        if name in executable:
            path.chmod(0o755)
    return root


# ---------------------------------------------------------------------------
# deterministic tree hash
# ---------------------------------------------------------------------------


def test_tree_hash_is_deterministic_and_independent_of_location_and_timestamps(tmp_path):
    files = {"pkg/__init__.py": "x = 1\n", "pkg/mod.py": "y = 2\n", "data/w.bin": b"\x00\x01\x02"}
    first = hash_tree(make_tree(tmp_path / "a", files))
    other = make_tree(tmp_path / "elsewhere" / "b", files)
    os.utime(other / "pkg" / "mod.py", (1, 1))
    second = hash_tree(other)
    assert first.sha256 == second.sha256 and first.files == 3 and first.directories == 2
    assert first.bytes == len("x = 1\n") + len("y = 2\n") + 3 and first.hash_source == "computed"


@pytest.mark.parametrize("change", ["content", "rename", "new_file", "mode", "empty_dir", "symlink"])
def test_every_kind_of_change_alters_the_digest(tmp_path, change):
    root = make_tree(tmp_path / "t", {"a.py": "1\n", "sub/b.py": "2\n"})
    before = hash_tree(root).sha256
    if change == "content":
        (root / "sub" / "b.py").write_text("3\n")
    elif change == "rename":
        (root / "sub" / "b.py").rename(root / "sub" / "c.py")
    elif change == "new_file":
        (root / "extra.txt").write_text("")
    elif change == "mode":
        (root / "a.py").chmod(0o755)
    elif change == "empty_dir":
        (root / "pkg").mkdir()
    else:
        (root / "link").symlink_to("a.py")
    assert hash_tree(root).sha256 != before


def test_a_symlink_is_recorded_by_target_and_never_followed(tmp_path):
    root = make_tree(tmp_path / "t", {"a.py": "1\n"})
    (root / "link").symlink_to("/nonexistent/target")
    record = hash_tree(root)
    assert record.symlinks == 1 and record.symlinks_listed == [{"path": "link", "target": "/nonexistent/target"}]
    assert record.external_symlinks == [{"path": "link", "target": "/nonexistent/target"}]
    (root / "link").unlink()
    (root / "link").symlink_to("/nonexistent/other")
    assert hash_tree(root).sha256 != record.sha256  # the target is part of the digest
    with pytest.raises(TreeError, match="symbolic link refused"):
        hash_tree(root, symlinks="refuse")


def test_pycache_can_be_excluded_so_stale_bytecode_never_stands_in_for_source(tmp_path):
    root = make_tree(tmp_path / "t", {"m.py": "1\n"})
    clean = hash_tree(root, exclude_dirs=treehash.CANDIDATE_EXCLUDES)
    make_tree(root, {"__pycache__/m.cpython-312.pyc": b"forged"})
    assert hash_tree(root, exclude_dirs=treehash.CANDIDATE_EXCLUDES).sha256 == clean.sha256
    assert hash_tree(root).sha256 != clean.sha256  # included when hashing an environment


def test_a_single_file_is_a_tree_of_one_entry(tmp_path):
    path = tmp_path / "weights.onnx"
    path.write_bytes(b"w" * 100)
    record = hash_tree(path)
    assert record.kind == "file" and record.files == 1 and record.bytes == 100
    path.write_bytes(b"w" * 99 + b"x")
    assert hash_tree(path).sha256 != record.sha256


def test_unreadable_files_are_recorded_for_environments_and_refused_when_strict(tmp_path):
    root = make_tree(tmp_path / "t", {"ok.py": "1\n", "secret.bin": "s"})
    (root / "secret.bin").chmod(0)
    try:
        if os.access(root / "secret.bin", os.R_OK):
            pytest.skip("running as a user that can read mode 000 files")
        record = hash_tree(root)
        assert record.unreadable == 1 and record.unreadable_listed == [{"path": "secret.bin"}]
        with pytest.raises(TreeError, match="unreadable"):
            hash_tree(root, strict=True)
    finally:
        (root / "secret.bin").chmod(0o600)


# ---------------------------------------------------------------------------
# staging: a private, read-only, content-addressed copy that is hashed while it is copied
# ---------------------------------------------------------------------------


def test_staged_copy_has_the_digest_of_the_source_and_is_read_only(tmp_path):
    source = make_tree(
        tmp_path / "cand",
        {"pkg/__init__.py": "x\n", "pkg/impl.py": "y\n", "run.sh": "#!/bin/sh\n"},
        executable=("run.sh",),
    )
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        staged = Path(record.staged_path)
        assert (
            record.role == "candidate_staged"
            and record.path == str(source)
            and record.sha256 == hash_tree(source).sha256
        )
        assert staged != source and record.sha256 in str(staged)  # content addressed
        assert hash_tree(staged).sha256 == record.sha256
        assert stat.S_IMODE((staged / "run.sh").stat().st_mode) == 0o500  # executable bit kept
        assert stat.S_IMODE((staged / "pkg" / "impl.py").stat().st_mode) == 0o400
        with pytest.raises(PermissionError):
            (staged / "pkg" / "impl.py").write_text("tamper")
        with pytest.raises(PermissionError):
            (staged / "new.py").write_text("tamper")
        area.verify(record)
    assert not Path(record.staged_path).exists()  # close() removes the area


def test_editing_the_original_after_staging_changes_nothing_that_is_bound(tmp_path):
    """No TOCTOU: the bytes that were hashed are the bytes in the copy, whatever happens to the original."""
    source = make_tree(tmp_path / "cand", {"impl.py": "MARK = 'original'\n"})
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        (source / "impl.py").write_text("MARK = 'edited after staging'\n")
        assert (Path(record.staged_path) / "impl.py").read_text() == "MARK = 'original'\n"
        area.verify(record)  # the copy still matches its digest
        assert hash_tree(source).sha256 != record.sha256


def test_tampering_with_the_staged_copy_is_detected_by_the_recheck(tmp_path):
    source = make_tree(tmp_path / "cand", {"impl.py": "MARK = 1\n"})
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        staged = Path(record.staged_path)
        os.chmod(staged, 0o700)
        os.chmod(staged / "impl.py", 0o600)
        (staged / "impl.py").write_text("MARK = 2\n")
        with pytest.raises(TreeError, match="changed after it was hashed"):
            area.verify(record)


def test_identical_content_is_stored_once(tmp_path):
    files = {"m.py": "1\n"}
    one, two = make_tree(tmp_path / "one", files), make_tree(tmp_path / "two", files)
    with StageArea(tmp_path) as area:
        first, second = area.stage(one), area.stage(two)
        assert first.sha256 == second.sha256 and first.staged_path == second.staged_path
        assert sorted(p.name for p in area.root.iterdir()) == [first.sha256]


def test_staging_refuses_symlinks_special_files_and_symlink_roots(tmp_path):
    source = make_tree(tmp_path / "cand", {"m.py": "1\n"})
    (source / "link").symlink_to("m.py")
    with StageArea(tmp_path) as area:
        with pytest.raises(TreeError, match="symbolic link refused"):
            area.stage(source)
        (source / "link").unlink()
        os.mkfifo(source / "pipe")
        with pytest.raises(TreeError, match="special file refused"):
            area.stage(source)
        (source / "pipe").unlink()
        (tmp_path / "alias").symlink_to(source)
        with pytest.raises(TreeError, match="symbolic link"):
            area.stage(tmp_path / "alias")
        assert not list(area.root.glob(".incoming-*"))  # failed staging leaves nothing behind


def test_staging_excludes_pycache(tmp_path):
    source = make_tree(tmp_path / "cand", {"m.py": "1\n", "__pycache__/m.cpython-312.pyc": b"forged"})
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        assert not (Path(record.staged_path) / "__pycache__").exists() and record.excluded_dirs == ("__pycache__",)


# ---------------------------------------------------------------------------
# stat-signature cache for large environment trees
# ---------------------------------------------------------------------------


def test_cache_reuses_the_digest_only_while_the_stat_signature_is_unchanged(tmp_path):
    root = make_tree(tmp_path / "env", {"lib/a.py": "aaaa\n", "lib/b.py": "bbbb\n"})
    cache = TreeHashCache(tmp_path / "cache")
    first = hash_tree(root, cache=cache)
    second = hash_tree(root, cache=cache)
    assert (first.hash_source, second.hash_source) == ("computed", "stat_signature_cache")
    assert first.sha256 == second.sha256 and second.signature == first.signature
    # same size, same modification time, different content: ctime still moves, so the cache must miss
    before = (root / "lib" / "a.py").stat()
    (root / "lib" / "a.py").write_text("XXXX\n")
    os.utime(root / "lib" / "a.py", ns=(before.st_atime_ns, before.st_mtime_ns))
    third = hash_tree(root, cache=cache)
    assert third.hash_source == "computed" and third.sha256 != first.sha256


def test_cache_is_not_trusted_for_another_path_or_after_corruption(tmp_path):
    one = make_tree(tmp_path / "one", {"a.py": "1\n"})
    two = make_tree(tmp_path / "two", {"a.py": "1\n"})
    cache = TreeHashCache(tmp_path / "cache")
    hash_tree(one, cache=cache)
    assert hash_tree(two, cache=cache).hash_source == "computed"
    for entry in (tmp_path / "cache").glob("*.json"):
        entry.write_text("{ not json")
    assert hash_tree(one, cache=cache).hash_source == "computed"


def test_current_signature_detects_a_change_without_reading_files(tmp_path):
    root = make_tree(tmp_path / "env", {"a.py": "1\n"})
    record = hash_tree(root)
    assert treehash.current_signature(root) == record.signature
    (root / "a.py").write_text("22\n")
    assert treehash.current_signature(root) != record.signature


# ---------------------------------------------------------------------------
# static runtime layout: derived from files, nothing is executed
# ---------------------------------------------------------------------------


def make_venv(path: Path, *, pth_lines: tuple[str, ...] = (), finder_mapping: dict[str, str] | None = None) -> Path:
    venv.EnvBuilder(with_pip=False, symlinks=True).create(path)
    site = next((path / "lib").glob("python*/site-packages"))
    if finder_mapping is not None:
        (site / "__editable___demo_finder.py").write_text(
            f"from __future__ import annotations\nMAPPING: dict[str, str] = {finder_mapping!r}\nNAMESPACES: dict[str, list[str]] = {{}}\n"
            "raise SystemExit('the finder must be read, never run')\n"
        )
        pth_lines = (*pth_lines, "import __editable___demo_finder; __editable___demo_finder.install()")
    (site / "demo.pth").write_text("\n".join(pth_lines) + "\n")
    return path


def test_layout_reads_pth_files_and_editable_finders_without_running_anything(tmp_path):
    marker = tmp_path / "marker"
    extra, editable = tmp_path / "extra-libs", tmp_path / "editable-src"
    extra.mkdir()
    editable.mkdir()
    env = make_venv(
        tmp_path / "venv",
        pth_lines=(
            str(extra),
            "# comment",
            "",
            "relative-and-missing",
            f"import os; open({str(marker)!r}, 'w').write('ran')",
        ),
        finder_mapping={"demo": str(editable)},
    )
    layout = sandbox.static_runtime_layout(env / "bin" / "python", trusted_roots=[extra, editable])
    assert not marker.exists()  # the .pth line was counted, not run
    assert layout.pth_import_lines == 2
    assert layout.venv_root == str(env.resolve()) and "/usr" in layout.roots
    assert str(extra.resolve()) in layout.roots and str(editable.resolve()) in layout.roots
    assert layout.editable_roots == {"demo": str(editable)} and layout.needs_declaration == ()
    assert all(
        Path(a) == Path(b) or Path(a) not in Path(b).parents for a in layout.roots for b in layout.roots if a != b
    )  # a minimal cover: no root inside another


def test_a_location_named_by_the_candidates_environment_is_not_bound_on_its_say_so(tmp_path):
    """A .pth line or an editable finder could otherwise expose any directory of the host to the candidate."""
    secret, editable = tmp_path / "somebody-elses-data", tmp_path / "editable-src"
    secret.mkdir()
    editable.mkdir()
    env = make_venv(tmp_path / "venv", pth_lines=(str(secret),), finder_mapping={"demo": str(editable)})
    layout = sandbox.static_runtime_layout(env / "bin" / "python")
    assert str(secret.resolve()) not in layout.roots and str(editable.resolve()) not in layout.roots
    assert {path for path, _ in layout.needs_declaration} == {str(secret.resolve()), str(editable.resolve())}
    assert set(layout.pth_roots) == {str(secret.resolve()), str(editable.resolve())}
    with pytest.raises(sandbox.SandboxError, match="not bound on its say-so"):
        build_bound_set(
            layout, isolation="bwrap", declared=[], protected=[], staging_base=tmp_path, cache=shared_cache()
        )
    assert not list(tmp_path.glob("passage-stage-*"))
    # the operator can declare what is meant to be visible, as a runtime tree or as a candidate path
    declared = sandbox.static_runtime_layout(env / "bin" / "python", extra_roots=[secret])
    bound = build_bound_set(
        declared, isolation="bwrap", declared=[editable], protected=[], staging_base=tmp_path, cache=shared_cache()
    )
    try:
        assert str(secret.resolve()) in [r.path for r in bound.environment]
        assert str(editable.resolve()) in [r.path for r in bound.candidate]
    finally:
        bound.close()
    # without a sandbox nothing is bound, so what the environment names is hashed instead of refused
    process = build_bound_set(layout, isolation="process", declared=[], protected=[], staging_base=tmp_path)
    try:
        assert {r.path for r in process.environment} >= {str(secret.resolve()), str(editable.resolve())}
    finally:
        process.close()


def test_the_home_line_of_pyvenv_cfg_does_not_choose_what_is_bound(tmp_path):
    victim = tmp_path / "victim" / "bin"
    victim.mkdir(parents=True)
    env = make_venv(tmp_path / "venv")
    config = env / "pyvenv.cfg"
    config.write_text(config.read_text().replace("home = /usr/bin", f"home = {victim}"))
    layout = sandbox.static_runtime_layout(env / "bin" / "python")
    assert layout.base_prefix == "/usr" and str(victim.parent) not in layout.roots
    assert layout.needs_declaration == ()


def test_an_interpreter_whose_prefix_is_outside_usr_must_be_declared(tmp_path):
    prefix = tmp_path / "python-prefix"
    (prefix / "bin").mkdir(parents=True)
    (prefix / "bin" / "python3").symlink_to(sys.executable)
    real_prefix = Path(sys.executable).resolve().parent.parent
    layout = sandbox.static_runtime_layout(prefix / "bin" / "python3")
    assert layout.base_prefix == str(real_prefix)
    if not str(real_prefix).startswith("/usr"):
        assert (str(real_prefix), "prefix of the interpreter executable, outside /usr") in layout.needs_declaration
        assert str(real_prefix) in sandbox.static_runtime_layout(prefix / "bin" / "python3", [real_prefix]).roots


def test_layout_of_the_running_interpreter_covers_the_evaluator_sources():
    layout = sandbox.static_runtime_layout(sys.executable, trusted_roots=sandbox.evaluator_roots())
    import passage_eval

    source = str(Path(passage_eval.__file__).resolve().parent)
    assert "/usr" in layout.roots
    assert any(source == root or source.startswith(root + os.sep) for root in layout.roots)
    assert "passage_eval" in layout.editable_roots or any(source.startswith(r + os.sep) for r in layout.roots)
    assert layout.needs_declaration == ()  # the evaluator is the supervisor's own


def test_layout_refuses_a_missing_interpreter_or_root(tmp_path):
    with pytest.raises(sandbox.SandboxError, match="does not exist"):
        sandbox.static_runtime_layout(tmp_path / "no-python")
    with pytest.raises(sandbox.SandboxError, match="runtime root does not exist"):
        sandbox.static_runtime_layout(sys.executable, [tmp_path / "missing"])


# ---------------------------------------------------------------------------
# C2: the interpreter probe runs inside the sandbox of the worker, never on the host
# ---------------------------------------------------------------------------


def evil_environment(tmp_path: Path):
    """The reviewer's attack: a candidate venv whose .pth hook runs at interpreter start-up.

    The hook records what it can see. It is inert for the supervisor's run itself: the shims below
    only make ``passage_eval``, ``passage`` and ``numpy`` importable so that the probe can complete.
    """
    import numpy

    import passage
    import passage_eval

    host_marker = tmp_path / "host-marker.json"
    private = tmp_path / "private-dataset"
    private.mkdir()
    env = make_venv(tmp_path / "evilvenv")
    site = next((env / "lib").glob("python*/site-packages"))
    for name, module in (("passage_eval", passage_eval), ("passage", passage)):
        (site / name).mkdir()
        (site / name / "__init__.py").write_text(f"__path__ = [{str(Path(module.__file__).resolve().parent)!r}]\n")
    numpy_site = str(Path(numpy.__file__).resolve().parent.parent)
    (site / "evil_hook.py").write_text(
        "import json, os, sys\n"
        f"seen = dict(sees_dataset=os.path.exists({str(private)!r}), sees_home=os.path.exists(os.path.expanduser('~/.ssh')),\n"
        "            net=open('/proc/net/dev').read().count(':') if os.path.exists('/proc/net/dev') else -1, pid=os.getpid())\n"
        "sys.stdout.write('HOOK-SAW:' + json.dumps(seen) + chr(10))\n"
        "try:\n"
        f"    open({str(host_marker)!r}, 'w').write(json.dumps(seen))\n"
        "except OSError:\n"
        "    pass\n"
    )
    (site / "zz_candidate_dep.pth").write_text(f"{numpy_site}\nimport evil_hook\n")
    return env, host_marker, private


def probe_binds(env: Path, layout: sandbox.RuntimeLayout):
    return [sandbox.Bind(root, root) for root in layout.roots]


def test_process_level_probe_runs_the_hook_unsandboxed_and_the_manifest_says_so(tmp_path):
    env, marker, private = evil_environment(tmp_path)
    probe = sandbox.probe_interpreter(env / "bin" / "python", isolation=sandbox.IsolationConfig(level="process"))
    assert marker.exists() and json.loads(marker.read_text())["sees_dataset"] is True  # the documented limit
    assert probe["executed_in"] == "unsandboxed_process_level_same_as_worker"
    record = sandbox.isolation_record(sandbox.IsolationConfig(level="process"), [], bwrap_version=None)
    assert record["candidate_code_runs_unsandboxed"] is True and record["candidate_code_confined"] is False
    assert record["interpreter_probe"] == "executed_unsandboxed_process_level_same_as_worker"
    assert "unsandboxed" in record["statement"] and "not admissible for a verified run" in record["statement"]


@needs_bwrap
def test_bwrap_probe_confines_the_hook_it_cannot_see_the_dataset_the_home_or_write_to_the_host(tmp_path):
    """C2 regression: the .pth hook of the candidate environment must run inside the worker's sandbox."""
    env, marker, private = evil_environment(tmp_path)
    layout = sandbox.static_runtime_layout(
        env / "bin" / "python", [Path(__import__("numpy").__file__).resolve().parent.parent]
    )
    isolation = sandbox.IsolationConfig(level="bwrap")
    probe = sandbox.probe_interpreter(env / "bin" / "python", isolation=isolation, binds=probe_binds(env, layout))
    assert probe["executed_in"] == "bwrap_same_sandbox_as_worker"
    assert not marker.exists()  # the hook could not write to the host
    saw = json.loads(probe["startup_output"].split("HOOK-SAW:", 1)[1].splitlines()[0])
    assert saw["sees_dataset"] is False and saw["sees_home"] is False
    assert saw["net"] <= 1 and saw["pid"] < 10  # loopback only, own pid namespace
    record = sandbox.isolation_record(isolation, [sandbox.Bind("/usr", "/usr")], bwrap_version="bubblewrap 0.9.0")
    assert (
        record["candidate_code_confined"] is True
        and record["interpreter_probe"] == "executed_in_bwrap_same_sandbox_as_worker"
    )


@needs_bwrap
def test_a_probe_that_reports_unbound_locations_fails_closed(tmp_path):
    from passage_eval.supervision.supervisor import SupervisorConfig, inspect_environment

    env, marker, private = evil_environment(tmp_path)  # numpy's site is only reachable through a .pth line
    numpy_site = str(Path(__import__("numpy").__file__).resolve().parent.parent)
    python = str(env / "bin" / "python")
    undeclared = SupervisorConfig(
        worker_python=python, isolation=sandbox.IsolationConfig(level="bwrap"), staging_dir=str(tmp_path)
    )
    with pytest.raises(sandbox.SandboxError, match="not bound on its say-so.*" + numpy_site.replace(".", r"\.")):
        inspect_environment(undeclared, ["m:fn"])  # refused before anything ran, hook included
    assert not marker.exists()
    config = SupervisorConfig(
        worker_python=python,
        staging_dir=str(tmp_path),
        isolation=sandbox.IsolationConfig(level="bwrap", runtime_binds=(numpy_site,)),
    )
    environment = inspect_environment(config, ["m:fn"])  # declared by the operator: bound
    try:
        assert not marker.exists()
        assert environment.uncovered_paths == ()
        assert environment.probe["executed_in"] == "bwrap_same_sandbox_as_worker"
    finally:
        environment.close()
    # a hook that adds a location the static reading cannot see is reported by the probe and refused
    site = next((env / "lib").glob("python*/site-packages"))
    hidden = tmp_path / "hidden-libs"
    hidden.mkdir()
    (site / "zz_dynamic.pth").write_text(f"import sys; sys.path.append({str(hidden)!r})\n")
    with pytest.raises(sandbox.SandboxError, match="no bind covers"):
        inspect_environment(config, ["m:fn"])


@needs_bwrap
def test_the_probe_output_is_bounded_and_a_chatty_hook_cannot_exhaust_memory(tmp_path):
    """Standard output is a size-limited file that is never parsed; the result has its own bounded channel."""
    env, marker, private = evil_environment(tmp_path)
    site = next((env / "lib").glob("python*/site-packages"))
    (site / "zz_flood.pth").write_text("import sys; sys.stdout.write('X' * (4 << 20))\n")
    layout = sandbox.static_runtime_layout(
        env / "bin" / "python", [Path(__import__("numpy").__file__).resolve().parent.parent]
    )
    probe = sandbox.probe_interpreter(
        env / "bin" / "python", isolation=sandbox.IsolationConfig(level="bwrap"), binds=probe_binds(env, layout)
    )
    assert len(probe["startup_output"]) <= 1024 and probe["channel"]["trailing_bytes_ignored"] == 0
    assert probe["modules"]["passage_eval"]["origin"]  # the real result still arrived
    # the result channel is bounded as well: a hook that floods it is a failure of the probe
    (site / "flood.py").write_text(
        "import os\nfor fd in range(3, 64):\n"
        "    try:\n        os.write(fd, b'Z' * (2 << 20))\n    except OSError:\n        pass\n"
    )
    (site / "zz_flood.pth").write_text("import flood\n")
    with pytest.raises(sandbox.SandboxError, match="more than the limit"):
        sandbox.probe_interpreter(
            env / "bin" / "python", isolation=sandbox.IsolationConfig(level="bwrap"), binds=probe_binds(env, layout)
        )


# ---------------------------------------------------------------------------
# N3: the probe result travels on a dedicated pipe descriptor; only the first complete message counts
# ---------------------------------------------------------------------------


def _message(payload: dict) -> bytes:
    body = json.dumps(payload).encode()
    return b"PASSAGE-PROBE/v2\n" + str(len(body)).encode() + b"\n" + body


def test_parse_probe_message_takes_the_first_complete_message_and_counts_what_follows():
    first, second = {"modules": {"a": 1}, "id": "first"}, {"modules": {"a": 2}, "id": "second"}
    probe, trailing = sandbox.parse_probe_message(_message(first))
    assert probe == first and trailing == 0
    probe, trailing = sandbox.parse_probe_message(_message(first) + _message(second))
    assert probe == first and trailing == len(_message(second))  # the second message is never parsed
    probe, trailing = sandbox.parse_probe_message(_message(first) + b"\nPASSAGE-PROBE:" + json.dumps(second).encode())
    assert probe == first and trailing > 0  # the old stdout marker means nothing


@pytest.mark.parametrize(
    "data, match",
    [
        (b"", "does not start"),
        (b"PASSAGE-PROBE:{}", "does not start"),
        (b"PASSAGE-PROBE/v2\n", "valid length"),
        (b"PASSAGE-PROBE/v2\nabc\n{}", "valid length"),
        (b"PASSAGE-PROBE/v2\n999999999\n{}", "larger than the limit"),
        (b'PASSAGE-PROBE/v2\n50\n{"modules": {}}', "incomplete"),
        (b"PASSAGE-PROBE/v2\n5\n[1,2]", "module table"),
        (b"PASSAGE-PROBE/v2\n4\n\xff\xfe\xfd\xfc", "not valid JSON"),
    ],
)
def test_parse_probe_message_refuses_everything_that_is_not_a_complete_message(data, match):
    with pytest.raises(sandbox.SandboxError, match=match):
        sandbox.parse_probe_message(data)


FORGERY = (
    "import atexit, json, os, sys\n"
    "saved = []\n"
    "for fd in range(3, 64):              # a start-up hook keeps its own copies of every descriptor it inherited\n"
    "    try:\n        saved.append(os.dup(fd))\n    except OSError:\n        pass\n"
    "def _forge():\n"
    "    import importlib.util as u\n"
    "    out = dict(executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix, exec_prefix=sys.exec_prefix,\n"
    "               base_exec_prefix=sys.base_exec_prefix, version=list(sys.version_info[:3]), sys_path=[p for p in sys.path if p], modules={})\n"
    "    for n in ['passage_eval', 'passage', 'numpy']:\n"
    "        s = u.find_spec(n); out['modules'][n] = dict(origin=s.origin, locations=list(s.submodule_search_locations or []))\n"
    "    out['modules']['realdec'] = dict(origin=%(decoy_init)r, locations=[%(decoy)r])\n"
    "    payload = json.dumps(out).encode()\n"
    "    data = b'PASSAGE-PROBE/v2' + bytes([10]) + str(len(payload)).encode() + bytes([10]) + payload\n"
    "    sys.stdout.write(chr(10) + 'PASSAGE-PROBE:' + json.dumps(out) + chr(10)); sys.stdout.flush()\n"
    "    for fd in saved:\n"
    "        try:\n            os.write(fd, data)\n        except OSError:\n            pass\n"
    "atexit.register(_forge)\n"
)


def forged_environment(tmp_path: Path, hook: str = FORGERY):
    """The re-review's probe: a venv whose exit hook forges the probe result so that a decoy looks like the decoder."""
    env, marker, private = evil_environment(tmp_path)
    site = next((env / "lib").glob("python*/site-packages"))
    real = tmp_path / "reviewed-and-installed" / "realdec"
    real.mkdir(parents=True)
    (real / "__init__.py").write_text("# the copy that is really imported\n")
    decoy = site / "decoy_reviewed_pkg"
    decoy.mkdir()
    (decoy / "__init__.py").write_text("# the copy the operator reviewed\n")
    (site / "forge.py").write_text(hook % {"decoy": str(decoy), "decoy_init": str(decoy / "__init__.py")})
    (site / "zzz_forge.pth").write_text("import forge\n")
    return env, real


@needs_bwrap
def test_an_exit_hook_cannot_forge_the_probe_result_by_writing_a_second_message_or_a_stdout_marker(tmp_path):
    """N3 (re-review): with the result on stdout the LAST marker won, so an atexit hook could relocate the decoder."""
    env, real = forged_environment(tmp_path)
    numpy_site = Path(__import__("numpy").__file__).resolve().parent.parent
    layout = sandbox.static_runtime_layout(env / "bin" / "python", [numpy_site, real.parent])
    probe = sandbox.probe_interpreter(
        env / "bin" / "python",
        isolation=sandbox.IsolationConfig(level="bwrap"),
        binds=probe_binds(env, layout),
        candidate_paths=[real.parent],
        top_levels=["realdec"],
    )
    assert (
        Path(probe["modules"]["realdec"]["origin"]).resolve() == (real / "__init__.py").resolve()
    )  # the first message
    assert "decoy_reviewed_pkg" not in json.dumps(probe["modules"])
    assert probe["channel"]["kind"] == "dedicated_pipe_descriptor" and probe["channel"]["trailing_bytes_ignored"] > 100
    assert "PASSAGE-PROBE:" in probe["startup_output"] and probe["channel"]["stdout_never_parsed"] is True


STARTUP_FORGERY = (
    "import json, os, sys\n"
    "import importlib.util as u\n"
    "out = dict(executable=sys.executable, prefix=sys.prefix, base_prefix=sys.base_prefix, exec_prefix=sys.exec_prefix,\n"
    "           base_exec_prefix=sys.base_exec_prefix, version=list(sys.version_info[:3]), sys_path=[p for p in sys.path if p], modules={})\n"
    "for n in ['passage_eval', 'passage', 'numpy']:\n"
    "    s = u.find_spec(n); out['modules'][n] = dict(origin=s.origin, locations=list(s.submodule_search_locations or []))\n"
    "out['modules']['realdec'] = dict(origin=%(decoy_init)r, locations=[%(decoy)r])\n"
    "payload = json.dumps(out).encode()\n"
    "data = b'PASSAGE-PROBE/v2' + bytes([10]) + str(len(payload)).encode() + bytes([10]) + payload\n"
    "for fd in range(3, 64):\n"
    "    try:\n        os.write(fd, data)\n    except OSError:\n        pass\n"
)


@needs_bwrap
def test_documented_limit_a_start_up_hook_that_writes_first_pre_empts_the_probe_and_leaves_a_trace(tmp_path):
    """The gap that remains, on purpose: a hook that runs BEFORE the probe and finds the descriptor wins 'first message'.

    The real probe's own message then follows, so the trace is visible (``trailing_bytes_ignored`` > 0) and a verified
    run refuses it. The defence that does not depend on a trace is the pin of the interpreter environment, which a
    verified run requires: the hook is part of the environment tree whose digest the operator reviewed.
    """
    env, real = forged_environment(tmp_path, STARTUP_FORGERY)
    numpy_site = Path(__import__("numpy").__file__).resolve().parent.parent
    layout = sandbox.static_runtime_layout(env / "bin" / "python", [numpy_site, real.parent])
    probe = sandbox.probe_interpreter(
        env / "bin" / "python",
        isolation=sandbox.IsolationConfig(level="bwrap"),
        binds=probe_binds(env, layout),
        candidate_paths=[real.parent],
        top_levels=["realdec"],
    )
    assert "decoy_reviewed_pkg" in probe["modules"]["realdec"]["origin"]  # the forgery won
    assert probe["channel"]["trailing_bytes_ignored"] > 100  # ... and left the real message behind


# ---------------------------------------------------------------------------
# bound set: staged candidate trees, hashed environment trees, covered locations
# ---------------------------------------------------------------------------


def layout_for(env: Path, extra=()):
    return sandbox.static_runtime_layout(env / "bin" / "python", extra)


def test_bound_set_stages_candidates_and_hashes_the_environment_in_place(tmp_path):
    env = make_venv(tmp_path / "venv")
    cand = make_tree(tmp_path / "cand", {"pkg/__init__.py": "", "pkg/impl.py": "1\n"})
    weights = tmp_path / "weights.onnx"
    weights.write_bytes(b"w")
    bound = build_bound_set(
        layout_for(env),
        isolation="bwrap",
        declared=[cand, weights],
        protected=[],
        cache=shared_cache(),
        staging_base=tmp_path,
    )
    try:
        roles = {record.path: record.role for record in bound.candidate}
        assert roles == {str(cand.resolve()): "candidate_staged", str(weights.resolve()): "candidate_staged"}
        assert {Path(r.path).name for r in bound.environment} >= {"venv", "usr"}
        assert bound.binding == "staged_read_only_copy" and bound.os_layer["hashed"] is True
        binds = {b.target: b for b in bound.binds}
        assert binds[str(cand.resolve())].role == "candidate_staged" and binds[str(cand.resolve())].source != str(
            cand.resolve()
        )
        assert binds[str(env.resolve())].role == "runtime" and binds[str(env.resolve())].source == str(env.resolve())
        staged_cand = next(r for r in bound.candidate if r.path == str(cand.resolve()))
        assert bound.view(cand / "pkg" / "impl.py") == Path(staged_cand.staged_path) / "pkg" / "impl.py"
        manifest = bound.manifest()
        assert json.loads(json.dumps(manifest)) == manifest and manifest["binding"] == "staged_read_only_copy"
        assert all(len(row["sha256"]) == 64 for row in manifest["candidate"] + manifest["environment"])
        bound.recheck()
    finally:
        bound.close()


def test_environment_change_during_the_run_is_detected(tmp_path):
    env = make_venv(tmp_path / "venv")
    bound = build_bound_set(layout_for(env), isolation="process", declared=[], protected=[], staging_base=tmp_path)
    try:
        bound.recheck()
        site = next((env / "lib").glob("python*/site-packages"))
        (site / "injected.py").write_text("import os\n")
        from passage_eval.supervision.identity import IdentityError

        with pytest.raises(IdentityError, match="runtime tree .* changed"):
            bound.recheck()
        assert bound.os_layer["hashed"] is False and "process isolation" in bound.os_layer["reason"]
    finally:
        bound.close()


def test_process_isolation_hashes_candidates_in_place_and_detects_edits(tmp_path):
    env = make_venv(tmp_path / "venv")
    cand = make_tree(tmp_path / "cand", {"impl.py": "1\n"})
    bound = build_bound_set(layout_for(env), isolation="process", declared=[cand], protected=[], staging_base=tmp_path)
    try:
        (record,) = bound.candidate
        assert record.role == "candidate_in_place" and record.staged_path is None and bound.binds == []
        assert bound.binding == "in_place_hash_rechecked"
        (cand / "impl.py").write_text("2\n")
        from passage_eval.supervision.identity import IdentityError

        with pytest.raises(IdentityError, match="declared candidate tree .* changed"):
            bound.recheck()
    finally:
        bound.close()


def test_a_declared_path_that_overlaps_a_protected_path_is_refused_before_anything_is_copied(tmp_path):
    env = make_venv(tmp_path / "venv")
    dataset = make_tree(tmp_path / "data", {"3x3/validation/x.parquet": b"labels"})
    with pytest.raises(sandbox.SandboxError, match="overlaps protected path"):
        build_bound_set(
            layout_for(env), isolation="bwrap", declared=[tmp_path], protected=[str(dataset)], staging_base=tmp_path
        )
    assert not list(tmp_path.glob("passage-stage-*"))  # the staging area was removed again


def test_declared_paths_nested_in_a_staged_tree_are_covered_by_it(tmp_path):
    env = make_venv(tmp_path / "venv")
    cand = make_tree(tmp_path / "cand", {"sub/impl.py": "1\n", "w.bin": "w"})
    bound = build_bound_set(
        layout_for(env), isolation="bwrap", declared=[cand], protected=[], cache=shared_cache(), staging_base=tmp_path
    )
    try:
        extended = bound.with_declared([cand / "w.bin", cand / "sub"], [])
        assert [r.path for r in extended.candidate] == [str(cand.resolve())]
        assert extended.is_covered(cand / "w.bin") and not extended.is_covered(tmp_path / "elsewhere.bin")
        with pytest.raises(Exception, match="contains an already staged tree"):
            extended.with_declared([tmp_path], [])
    finally:
        bound.close()


def test_runtime_roots_inside_a_declared_candidate_tree_are_served_by_its_staged_copy(tmp_path):
    """The pilot's case: an editable install whose source directory is also declared as --candidate-path."""
    editable = make_tree(tmp_path / "editable-src", {"mod.py": "1\n"})
    env = make_venv(tmp_path / "venv", pth_lines=(str(editable),))
    bound = build_bound_set(
        layout_for(env),
        isolation="bwrap",
        declared=[editable],
        protected=[],
        cache=shared_cache(),
        staging_base=tmp_path,
    )
    try:
        assert str(editable.resolve()) not in [r.path for r in bound.environment]
        assert [r.role for r in bound.candidate] == ["candidate_staged"]
        targets = [b.target for b in bound.binds if b.target == str(editable.resolve())]
        assert len(targets) == 1
    finally:
        bound.close()
