"""Third review round: pinned workers (N1, N2), device binds (N4), staged-copy re-checks and bytecode (N5), ONNX Runtime.

The probe channel (N3) is tested beside the other probe tests in ``test_supervision_tree.py``; the whole verified
chain (environment pin, narrowed views, planner pin) runs in ``test_supervision_hardening.py``. Here the pieces are
exercised one by one, against real worker processes where the claim is about what a worker can see.
"""

from __future__ import annotations

import importlib.util
import marshal
import os
import stat
import struct
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from passage_eval.supervision import PURPOSE_VERIFIED, identity, sandbox, treehash
from passage_eval.supervision.__main__ import main as supervision_main
from passage_eval.supervision.environment import build_bound_set, forbidden_runtime_name
from passage_eval.supervision.identity import IdentityError
from passage_eval.supervision.supervisor import PurposeError, SplitSpec, SupervisionError, _evaluator_extra_files
from passage_eval.supervision.treehash import StageArea, TreeError


def _load(name: str):
    path = Path(__file__).resolve().parent / name
    spec = importlib.util.spec_from_file_location(f"{path.stem}_for_pins", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


H = _load("test_supervision_hardening.py")
T = _load("test_supervision_tree.py")
BWRAP = sandbox.available_bwrap()
needs_bwrap = pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")

# --------------------------------------------------------------------------------------------------------------
# a candidate with siblings: the decoder's package, a predictor, a helper module, weights and an unrelated package
# --------------------------------------------------------------------------------------------------------------

DECODER = r"""
import os, sys
from passage_eval.protocol.contracts import PlanResult


class Decoder:
    def __init__(self, **kw):
        self.kw = kw

    def decode(self, field, start, goal, obstacles, budget):
        package = os.path.dirname(os.path.abspath(__file__))
        parent = os.path.dirname(package)
        return PlanResult([start, (1, 1), goal], work={"expansions": 4}, precision="float32",
                          diagnostics={"siblings": sorted(os.listdir(parent)), "package_files": sorted(os.listdir(package)),
                                       "sys_path_parent": parent in sys.path})
"""
SIBLING_DECODER = r"""
from helper import decode_impl            # a top-level module next to the decoder's package, outside the reviewed tree


class Decoder:
    def __init__(self, **kw):
        pass

    def decode(self, field, start, goal, obstacles, budget):
        return decode_impl(field, start, goal, obstacles, budget)
"""
HELPER = r"""
from passage_eval.protocol.contracts import PlanResult


def decode_impl(field, start, goal, obstacles, budget):
    return PlanResult([start, (1, 1), goal], work={"expansions": 1}, diagnostics={"helper": "present"})
"""
PREDICTOR = r"""
import numpy as np


class P:
    def __init__(self, **kw):
        pass

    def predict(self, request):
        return np.ones(request.elevation.shape, dtype=np.float32)
"""
PLANNER_WITH_ARTIFACT = r"""
from passage_eval.protocol.contracts import PlanResult


class Planner:
    def __init__(self, **kw):
        pass

    def plan(self, request):
        return PlanResult([request.start, (1, 1), request.goal], work={"expansions": 2})
"""


def write_candidate(root: Path) -> Path:
    for name in ("decpkg", "other_pkg", "nspkg"):
        (root / name).mkdir(parents=True)
    (root / "decpkg" / "__init__.py").write_text("")
    (root / "decpkg" / "dec.py").write_text(DECODER)
    (root / "decpkg" / "sibling_user.py").write_text(SIBLING_DECODER)
    (root / "other_pkg" / "__init__.py").write_text("")
    (root / "nspkg" / "dec.py").write_text(DECODER)  # no __init__.py: a namespace package
    (root / "predpkg.py").write_text(PREDICTOR)
    (root / "helper.py").write_text(HELPER)
    (root / "planpkg.py").write_text(PLANNER_WITH_ARTIFACT)
    (root / "weights.bin").write_bytes(b"weights")
    return root


@pytest.fixture
def cand(tmp_path):
    return write_candidate(tmp_path / "cand")


def pin_of(path: Path) -> str:
    return treehash.hash_tree(
        path, role="decoder_entry", symlinks="refuse", strict=True, exclude_dirs=treehash.CANDIDATE_EXCLUDES
    ).sha256


SPLIT = SplitSpec("predpkg:P", "decpkg.dec:Decoder")


def targets(worker) -> set[str]:
    return {bind.target for bind in worker.binds}


# --------------------------------------------------------------------------------------------------------------
# N1: a pinned worker is given the pinned tree and the environment, nothing else
# --------------------------------------------------------------------------------------------------------------


@needs_bwrap
def test_a_pinned_decoder_sees_only_its_own_package_while_the_predictor_still_sees_the_candidate_tree(cand):
    """The re-review's probe: the allow-list hashed ``decpkg`` but the decoder could import every sibling of the root."""
    with H.runner_for(
        cand, "predpkg:P", isolation="bwrap", split=SPLIT, decoder_allow_sha256=(pin_of(cand / "decpkg"),)
    ) as runner:
        record = H.run_one(runner, H.request_for(0))
        assert record["status"] == "ok" and record["counters_source"] == "operator_decoder"
        assert record["diagnostics"]["siblings"] == ["decpkg"]  # the decoder sees nothing else
        assert record["diagnostics"]["package_files"] == ["__init__.py", "dec.py", "sibling_user.py"]
        assert record["diagnostics"]["sys_path_parent"] is True
        decoder, predictor = runner.workers["decoder"], runner.workers["predictor"]
        assert str(cand.resolve() / "decpkg") in targets(decoder) and str(cand.resolve()) not in targets(decoder)
        assert str(cand.resolve()) in targets(predictor) and str(cand.resolve() / "decpkg") not in targets(predictor)
        assert decoder.import_paths == [str(cand.resolve())] or decoder.import_paths == [str(cand)]
        assert {b.role for b in decoder.binds if b.target.endswith("decpkg")} == {"candidate_staged"}
        view = runner.facts["worker_views"]["decoder"]
        assert view["pinned"] and view["narrowed_to_pinned_tree"] and view["enforced_by"] == "bwrap_mount_namespace"
        assert runner.facts["worker_views"]["predictor"]["narrowed_to_pinned_tree"] is False


@needs_bwrap
def test_a_pinned_decoder_cannot_import_a_sibling_module_that_the_pin_does_not_cover(cand):
    """Before: ``operator_decoder`` with a swapped sibling. Now the sibling is not there, and the decoder does not load."""
    split = SplitSpec("predpkg:P", "decpkg.sibling_user:Decoder")
    pin = pin_of(cand / "decpkg")
    with pytest.raises(SupervisionError, match="refused to load.*No module named 'helper'"):
        with H.runner_for(cand, "predpkg:P", isolation="bwrap", split=split, decoder_allow_sha256=(pin,)):
            pass
    # unpinned, the decoder keeps the candidate's whole tree and its counters say so
    with H.runner_for(cand, "predpkg:P", isolation="bwrap", split=split) as runner:
        record = H.run_one(runner, H.request_for(0))
        assert record["status"] == "ok" and record["counters_source"] == "candidate_self_reported"
        assert record["diagnostics"]["helper"] == "present"
        assert runner.facts["worker_views"]["decoder"]["narrowed_to_pinned_tree"] is False


def test_process_isolation_records_that_the_pinned_view_is_not_enforced(cand):
    with H.runner_for(
        cand, "predpkg:P", isolation="process", split=SPLIT, decoder_allow_sha256=(pin_of(cand / "decpkg"),)
    ) as runner:
        view = runner.facts["worker_views"]["decoder"]
        assert view["pinned"] is True and view["narrowed_to_pinned_tree"] is False
        assert view["enforced_by"] == "not_enforced_process_isolation_shares_the_file_system"
        assert view["import_paths"] == [str(cand.resolve())]
        assert "interpreter_aliases" not in view
        assert H.run_one(runner, H.request_for(0))["counters_source"] == "operator_decoder"


def test_a_namespace_package_decoder_is_refused(cand):
    """No ``__init__.py``: its content is a union of directories and no single digest describes it."""
    probe = sandbox.probe_interpreter(sys.executable, candidate_paths=[cand], top_levels=["nspkg", "decpkg", "predpkg"])
    with pytest.raises(IdentityError, match="namespace package"):
        identity.entry_package_path("nspkg.dec:Decoder", probe)
    assert identity.entry_package_path("decpkg.dec:Decoder", probe) == (cand / "decpkg").resolve()
    with pytest.raises(IdentityError, match="namespace package"):
        with H.runner_for(cand, "predpkg:P", isolation="process", split=SplitSpec("predpkg:P", "nspkg.dec:Decoder")):
            pass


def test_a_package_that_resolves_to_several_locations_is_refused(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "__init__.py").write_text("")
    probe = {
        "modules": {
            "split": {
                "origin": str(tmp_path / "a" / "__init__.py"),
                "locations": [str(tmp_path / "a"), str(tmp_path / "elsewhere")],
            }
        }
    }
    with pytest.raises(IdentityError, match="several locations"):
        identity.entry_package_path("split.mod:fn", probe)
    probe["modules"]["split"]["locations"] = [str(tmp_path / "a")]
    assert identity.entry_package_path("split.mod:fn", probe) == (tmp_path / "a").resolve()


def test_the_kwargs_of_a_pinned_worker_may_not_declare_files_its_sandbox_would_not_hold(cand):
    split = SplitSpec("predpkg:P", "decpkg.dec:Decoder", {"checkpoint_path": str(cand / "weights.bin")})
    with pytest.raises(IdentityError, match="self-contained"):
        with H.runner_for(
            cand, "predpkg:P", isolation="process", split=split, decoder_allow_sha256=(pin_of(cand / "decpkg"),)
        ):
            pass
    unpinned = H.runner_for(cand, "predpkg:P", isolation="process", split=split)
    with unpinned as runner:  # an unpinned decoder may declare its files as before
        assert runner.facts["worker_views"]["decoder"]["pinned"] is False


# --------------------------------------------------------------------------------------------------------------
# N2: planner mode is pinned the same way, and the family in the spec decides nothing
# --------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("isolation", [pytest.param("process"), pytest.param("bwrap", marks=needs_bwrap)])
def test_a_pinned_planner_reports_operator_planner_counters_and_an_unpinned_one_does_not(cand, isolation):
    entry = "planpkg:Planner"
    pin = pin_of(cand / "planpkg.py")  # a single module file is a tree of one entry
    with H.runner_for(cand, entry, isolation=isolation, planner_allow_sha256=(pin,)) as runner:
        record = H.run_one(runner, H.request_for(0))
        assert runner.counters_source == "operator_planner" == record["counters_source"]
        assert runner.facts["pinned_role"] == "planner" and runner.facts["planner_entry_tree"]["sha256"] == pin
        view = runner.facts["worker_views"]["planner"]
        assert view["pinned"] is True and view["narrowed_to_pinned_tree"] is (isolation == "bwrap")
        if isolation == "bwrap":
            assert targets(runner.workers["planner"]) >= {str(cand.resolve() / "planpkg.py")}
            assert str(cand.resolve()) not in targets(runner.workers["planner"])
    with H.runner_for(cand, entry, isolation=isolation, planner_allow_sha256=("e" * 64,)) as runner:
        assert runner.counters_source == "candidate_self_reported" and runner.facts["pinned_role"] is None
    with H.runner_for(cand, entry, isolation=isolation) as runner:  # diagnostic, no pin at all
        assert runner.counters_source == "candidate_self_reported" and runner.facts["planner_entry_tree"] is None


@pytest.mark.parametrize("isolation", [pytest.param("process"), pytest.param("bwrap", marks=needs_bwrap)])
def test_a_classical_baseline_in_the_interpreter_environment_is_pinned_by_its_package_tree(cand, isolation):
    """A* of the evaluator itself: no candidate tree is involved, the worker gets the environment and nothing else."""
    import passage_eval

    package = Path(passage_eval.__file__).resolve().parent
    entry = "passage_eval.baselines.classical:CanonicalAStarPlanner"
    with H.runner_for(cand, entry, isolation=isolation, planner_allow_sha256=(pin_of(package),)) as runner:
        record = H.run_one(runner, H.request_for(0))
        assert record["status"] == "ok" and record["counters_source"] == "operator_planner"
        view = runner.facts["worker_views"]["planner"]
        assert view["pinned_tree"] == str(package) and view["pinned"] is True
        if isolation == "bwrap":
            assert view["narrowed_to_pinned_tree"] is True and view["import_paths"] == []
            assert str(cand.resolve()) not in view["read_only_binds"]  # the candidate tree is not mounted
            assert "only the interpreter environment is mounted" in view["note"]


@needs_bwrap
def test_a_verified_run_with_an_unpinned_planner_or_decoder_is_refused(cand):
    with pytest.raises(PurposeError, match="planner entry tree sha256 [0-9a-f]{64} is not in the operator allow-list"):
        with H.verified_runner_for(cand, "planpkg:Planner", planner_allow_sha256=("e" * 64,)):
            pass
    with pytest.raises(PurposeError, match="decoder entry tree sha256 [0-9a-f]{64} is not in the operator allow-list"):
        with H.verified_runner_for(cand, "predpkg:P", split=SPLIT, decoder_allow_sha256=("e" * 64,)):
            pass


# --------------------------------------------------------------------------------------------------------------
# environment allow-list and the supervisor-side ONNX Runtime scan
# --------------------------------------------------------------------------------------------------------------


def warm_session_cache(cand, entries) -> None:
    """Make sure the session-wide digest cache holds the environment trees, so that the next inspection reuses it."""
    config = H.SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="bwrap", runtime_binds=H.interpreter_runtime_binds()),
        candidate_paths=(str(cand),),
    )
    H.inspect_environment(config, entries).close()


@needs_bwrap
def test_a_verified_run_needs_every_environment_tree_in_the_operator_allow_list(cand):
    pin = pin_of(cand / "planpkg.py")
    warm_session_cache(cand, ["planpkg:Planner"])
    with pytest.raises(PurposeError) as refused:
        with H.verified_runner_for(
            cand, "planpkg:Planner", planner_allow_sha256=(pin,), environment_digests=("a" * 64,)
        ):
            pass
    text = str(refused.value)
    assert (
        "not in the operator allow-list (--environment-allow-sha256)" in text and "/usr" in text
    )  # the OS layer is bound
    with pytest.raises(PurposeError, match="not computed in this run"):  # a digest taken from the cache is not evidence
        with H.verified_runner_for(cand, "planpkg:Planner", planner_allow_sha256=(pin,)):
            pass


def test_tree_hash_environment_prints_the_digest_that_the_bound_set_records(tmp_path, capsys):
    env = T.make_venv(tmp_path / "venv")
    site = next((env / "lib").glob("python*/site-packages"))
    (site / "__pycache__").mkdir()
    (site / "__pycache__" / "mod.cpython-312.pyc").write_bytes(b"bytecode")
    bound = build_bound_set(T.layout_for(env), isolation="process", declared=[], protected=[], staging_base=tmp_path)
    try:
        recorded = next(r for r in bound.environment if r.path == str(env.resolve()))
    finally:
        bound.close()
    assert supervision_main(["tree-hash", "--environment", str(env)]) == 0
    assert capsys.readouterr().out.strip() == recorded.sha256
    with pytest.raises(TreeError, match="symbolic link refused"):  # the candidate rule cannot even read a venv
        supervision_main(["tree-hash", str(env)])
    plain = T.make_tree(tmp_path / "plain", {"pkg/m.py": "1\n", "pkg/__pycache__/m.pyc": b"x"})
    assert supervision_main(["tree-hash", str(plain)]) == 0
    candidate_rule = capsys.readouterr().out.strip()
    assert supervision_main(["tree-hash", "--environment", str(plain)]) == 0
    assert capsys.readouterr().out.strip() != candidate_rule  # the candidate rule excludes __pycache__


@pytest.mark.parametrize(
    "name, flagged",
    [
        ("onnxruntime", True),
        ("onnxruntime_gpu-1.20.0.dist-info", True),
        ("onnxruntime-1.20.0.dist-info", True),
        ("libonnxruntime.so.1.20.0", True),
        ("libonnxruntime_providers_shared.so", True),
        ("onnxruntime_pybind11_state.cpython-312-aarch64-linux-gnu.so", True),
        ("onnx", False),
        ("onnx-1.23.2.dist-info", False),
        ("myonnxruntime", False),
        ("runtime", False),
        ("numpy", False),
    ],
)
def test_forbidden_runtime_names(name, flagged):
    assert forbidden_runtime_name(name.lower()) is flagged


def test_the_supervisor_scan_finds_onnxruntime_in_the_environment_without_asking_a_worker(tmp_path):
    env = T.make_venv(tmp_path / "venv")
    site = next((env / "lib").glob("python*/site-packages"))
    clean = build_bound_set(T.layout_for(env), isolation="process", declared=[], protected=[], staging_base=tmp_path)
    try:
        assert clean.forbidden_scan()["clean"] is True and clean.forbidden_scan()["found"] == []
    finally:
        clean.close()
    (site / "onnxruntime").mkdir()
    (site / "onnxruntime" / "__init__.py").write_text("")
    (site / "onnxruntime_gpu-1.20.0.dist-info").mkdir()
    (site / "onnxruntime" / "capi").mkdir()
    (site / "onnxruntime" / "capi" / "libonnxruntime_providers_shared.so").write_bytes(b"\x7fELF")
    bound = build_bound_set(T.layout_for(env), isolation="process", declared=[], protected=[], staging_base=tmp_path)
    try:
        scan = bound.forbidden_scan()
        assert scan["clean"] is False and scan["evidence"] == "supervisor_walk_of_the_hashed_trees"
        (row,) = [r for r in scan["found"] if r["tree"] == str(env.resolve())]
        assert row["count"] == 3 and any(e.endswith("libonnxruntime_providers_shared.so") for e in row["entries"])
        assert "onnxruntime_gpu-1.20.0.dist-info" in " ".join(row["entries"])
    finally:
        bound.close()


@needs_bwrap
def test_a_verified_run_refuses_onnxruntime_in_a_staged_candidate_tree_and_in_a_runtime_tree(cand, tmp_path):
    (cand / "onnxruntime").mkdir()
    (cand / "onnxruntime" / "__init__.py").write_text("")
    pin = pin_of(cand / "planpkg.py")
    with pytest.raises(PurposeError, match="onnxruntime distributions or libraries.*cand"):
        with H.verified_runner_for(cand, "planpkg:Planner", planner_allow_sha256=(pin,)):
            pass
    # the same candidate without the directory is accepted by this rule (the cached digests are refused for another reason)
    (cand / "onnxruntime" / "__init__.py").unlink()
    (cand / "onnxruntime").rmdir()
    runtime = tmp_path / "extra-runtime"
    runtime.mkdir()
    (runtime / "libonnxruntime.so").write_bytes(b"\x7fELF")
    iso = sandbox.IsolationConfig(level="bwrap", runtime_binds=(str(runtime), *H.interpreter_runtime_binds()))
    config = H.SupervisorConfig(
        worker_python=sys.executable, isolation=iso, candidate_paths=(str(cand),), staging_dir=str(tmp_path)
    )
    environment = H.inspect_environment(config, ["planpkg:Planner"])
    try:
        digests = tuple(r.sha256 for r in environment.bound.environment)
        verified = replace(
            config, purpose=PURPOSE_VERIFIED, environment_allow_sha256=digests, planner_allow_sha256=(pin,)
        )
        with pytest.raises(PurposeError, match="onnxruntime distributions or libraries.*extra-runtime"):
            H.build_runner(H.bench("planpkg:Planner"), None, verified, environment, seed=1)
        diagnostic = H.build_runner(
            H.bench("planpkg:Planner"), None, config, environment, seed=1
        )  # recorded, not refused
        assert diagnostic.facts["forbidden_scan"]["clean"] is False
    finally:
        environment.close()


# --------------------------------------------------------------------------------------------------------------
# N4: device binds are existing character-device nodes and nothing else
# --------------------------------------------------------------------------------------------------------------


def test_a_character_device_node_is_accepted():
    config = sandbox.IsolationConfig(level="bwrap", dev_binds=("/dev/null", "/dev/zero"))
    argv = sandbox.bwrap_command(config, [], ["/usr/bin/true"])
    assert [argv[i + 1] for i, token in enumerate(argv) if token == "--dev-bind"] == ["/dev/null", "/dev/zero"]


@pytest.mark.parametrize(
    "device, match",
    [
        ("/dev/shm", "directory"),  # the re-review: became a shared read-write directory
        ("/dev/pts", "directory"),
        ("/dev/stdout", "symbolic link"),
        ("/dev/fd", "symbolic link"),
        ("/dev/no-such-node", "does not exist"),
        ("/dev/shm/../null", "below /dev"),
        ("/dev/", "directory"),
    ],
)
def test_directories_links_and_missing_nodes_are_refused(device, match):
    if not os.path.lexists(device) and match != "does not exist" and "below" not in match:
        pytest.skip(f"{device} does not exist on this host")
    with pytest.raises(ValueError, match=match):
        sandbox.IsolationConfig(level="bwrap", dev_binds=(device,))


def test_other_file_types_and_links_in_the_path_are_refused(monkeypatch):
    real_lstat = os.lstat

    def fake(kind):
        def lstat(path, *args, **kwargs):
            if str(path) == "/dev/fake":
                return os.stat_result((kind | 0o600, 1, 1, 1, 0, 0, 0, 0, 0, 0))
            return real_lstat(path, *args, **kwargs)

        return lstat

    for kind, match in (
        (stat.S_IFREG, "not a character device"),
        (stat.S_IFBLK, "not a character device"),
        (stat.S_IFIFO, "not a character device"),
        (stat.S_IFSOCK, "not a character device"),
    ):
        monkeypatch.setattr(sandbox.os, "lstat", fake(kind))
        with pytest.raises(ValueError, match=match):
            sandbox.check_device_node("/dev/fake")
    monkeypatch.setattr(sandbox.os, "lstat", fake(stat.S_IFCHR))
    assert sandbox.check_device_node("/dev/fake") == "/dev/fake"  # a character device is what is accepted
    monkeypatch.setattr(sandbox.os.path, "realpath", lambda path: "/dev/elsewhere")
    with pytest.raises(ValueError, match="passes through a symbolic link"):
        sandbox.check_device_node("/dev/fake")


def test_the_node_is_checked_again_when_the_command_line_is_built(monkeypatch):
    config = sandbox.IsolationConfig(level="bwrap", dev_binds=("/dev/null",))
    real_lstat = os.lstat

    def replaced(path, *args, **kwargs):  # the node was swapped for a directory meanwhile
        if str(path) == "/dev/null":
            return real_lstat("/tmp")
        return real_lstat(path, *args, **kwargs)

    monkeypatch.setattr(sandbox.os, "lstat", replaced)
    with pytest.raises(ValueError, match="directory"):
        sandbox.bwrap_command(config, [], ["/usr/bin/true"])


def test_the_isolation_record_states_the_device_policy():
    record = sandbox.isolation_record(
        sandbox.IsolationConfig(level="bwrap", dev_binds=("/dev/null",)), [], bwrap_version="x"
    )
    assert record["device_binds"] == ["/dev/null"]
    assert record["device_bind_policy"] == "existing_character_device_nodes_only_no_links_no_directories"


# --------------------------------------------------------------------------------------------------------------
# N5: a staged copy is checked by stat signature as well as by content
# --------------------------------------------------------------------------------------------------------------


def test_a_staged_copy_that_was_changed_and_put_back_is_caught_by_its_stat_signature(tmp_path):
    source = T.make_tree(tmp_path / "cand", {"impl.py": "MARK = 'original'\n", "sub/data.bin": b"d"})
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        staged = Path(record.staged_path)
        area.verify(record)
        target = staged / "impl.py"
        before = target.stat()
        original = target.read_bytes()
        # Give even a coarse filesystem clock a new tick before the transient
        # write. A same-tick ctime cannot attest that the file was touched.
        deadline_ns = before.st_ctime_ns + 1_000_000_000
        time.sleep(max(0, deadline_ns - time.time_ns()) / 1_000_000_000)
        os.chmod(target, 0o600)
        target.write_bytes(b"MARK = 'evil'\n")  # a transient change ...
        target.write_bytes(original)  # ... undone: same bytes
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))  # ... same modification time
        os.chmod(target, 0o400)  # ... same mode
        assert treehash.hash_tree(staged).sha256 == record.sha256  # the content check alone passes
        assert target.stat().st_ctime_ns != before.st_ctime_ns  # the change time cannot be put back
        with pytest.raises(TreeError, match="stat signature"):
            area.verify(record)


def test_a_new_entry_in_a_staged_copy_is_caught_even_when_it_is_a_pycache_directory(tmp_path):
    source = T.make_tree(tmp_path / "cand", {"impl.py": "1\n"})
    with StageArea(tmp_path) as area:
        record = area.stage(source)
        staged = Path(record.staged_path)
        os.chmod(staged, 0o700)
        (staged / "__pycache__").mkdir()  # excluded when copying, never when checking
        os.chmod(staged, 0o500)
        with pytest.raises(TreeError, match="changed after it was hashed"):
            area.verify(record)


def test_staging_the_same_content_again_checks_the_copy_that_is_already_there(tmp_path):
    files = {"impl.py": "1\n"}
    one, two = T.make_tree(tmp_path / "one", files), T.make_tree(tmp_path / "two", files)
    with StageArea(tmp_path) as area:
        record = area.stage(one)
        staged = Path(record.staged_path)
        os.chmod(staged, 0o700)
        os.chmod(staged / "impl.py", 0o600)
        (staged / "impl.py").write_text("tampered\n")
        with pytest.raises(TreeError, match="no longer holds the bytes"):
            area.stage(two)


def test_an_in_place_candidate_tree_that_was_touched_is_caught_by_its_signature(tmp_path):
    env = T.make_venv(tmp_path / "venv")
    cand = T.make_tree(tmp_path / "cand", {"impl.py": "1\n"})
    bound = build_bound_set(
        T.layout_for(env), isolation="process", declared=[cand], protected=[], staging_base=tmp_path
    )
    try:
        bound.recheck()
        target = cand / "impl.py"
        before = target.stat()
        original = target.read_bytes()
        target.write_bytes(b"2\n")
        target.write_bytes(original)
        os.utime(target, ns=(before.st_atime_ns, before.st_mtime_ns))
        with pytest.raises(identity.IdentityError, match="touched during the run"):
            bound.recheck()
    finally:
        bound.close()


# --------------------------------------------------------------------------------------------------------------
# N5: workers start with a fresh private bytecode prefix
# --------------------------------------------------------------------------------------------------------------

PYC_CANDIDATE = {
    "pycpkg/__init__.py": "",
    "pycpkg/impl.py": "MARK = 'source'\n",
    "pycpkg/entry.py": (
        "from passage_eval.protocol.contracts import PlanResult\n"
        "from pycpkg import impl\n\n\n"
        "class Planner:\n"
        "    def __init__(self, **kw):\n        pass\n\n"
        "    def plan(self, request):\n"
        "        return PlanResult([request.start, (1, 1), request.goal], work={'expansions': 1}, diagnostics={'mark': impl.MARK})\n"
    ),
}


def forge_bytecode(source: Path, forged_source: str) -> Path:
    """A .pyc for ``source`` that holds other code and carries the source's own recorded mtime and size."""
    info = source.stat()
    code = compile(forged_source, str(source), "exec")
    pyc = Path(importlib.util.cache_from_source(str(source)))
    pyc.parent.mkdir(exist_ok=True)
    header = (
        importlib.util.MAGIC_NUMBER
        + struct.pack("<I", 0)
        + struct.pack("<II", int(info.st_mtime) & 0xFFFFFFFF, info.st_size & 0xFFFFFFFF)
    )
    pyc.write_bytes(header + marshal.dumps(code))
    return pyc


def test_a_forged_pyc_beside_a_hashed_source_is_not_loaded_by_a_worker(tmp_path):
    cand = T.make_tree(tmp_path / "cand", PYC_CANDIDATE)
    forge_bytecode(cand / "pycpkg" / "impl.py", "MARK = 'forged'\n")
    control = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import sys; sys.path.insert(0, {str(cand)!r}); import pycpkg.impl as m; print(m.MARK)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert control.stdout.strip() == "forged"  # a plain interpreter trusts the bytecode beside the source
    with H.runner_for(cand, "pycpkg.entry:Planner", isolation="process") as runner:  # candidate hashed in place
        assert runner.identity["identity_binding"] == "in_place_hash_rechecked"
        record = H.run_one(runner, H.request_for(0))
    assert record["status"] == "ok" and record["diagnostics"]["mark"] == "source"


@pytest.mark.parametrize("isolation", [pytest.param("process"), pytest.param("bwrap", marks=needs_bwrap)])
def test_workers_and_the_probe_start_with_a_private_pycache_prefix(cand, isolation):
    with H.runner_for(cand, "planpkg:Planner", isolation=isolation) as runner:
        worker = runner.workers["planner"]
        argv = Path(f"/proc/{worker.proc.pid}/cmdline").read_bytes().split(b"\0")
        flags = [a.decode() for a in argv if a.startswith(b"pycache_prefix=")]
        assert len(flags) == 1
        prefix = flags[0].split("=", 1)[1]
        if isolation == "bwrap":
            assert prefix == sandbox.SANDBOX_PYCACHE  # a tmpfs inside the sandbox
        else:
            assert prefix.startswith(str(worker.private)) and not os.path.exists(prefix)  # a fresh, private directory
        assert (
            b"-B" in argv and b"-X" in argv and argv.index(b"-X") < argv.index(b"-m")
        )  # read-only prefix: nothing written
        assert runner.environment.probe["channel"]["kind"] == "dedicated_pipe_descriptor"
    assert sandbox.pycache_args("/p") == ["-B", "-X", "pycache_prefix=/p"]


# --------------------------------------------------------------------------------------------------------------
# N5: non-source files of the evaluator packages
# --------------------------------------------------------------------------------------------------------------


def make_package(root: Path, files: dict[str, bytes | str]) -> Path:
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    return root


def test_non_source_files_must_exist_identically_on_the_supervisors_side(tmp_path):
    own = make_package(
        tmp_path / "own",
        {
            "__init__.py": "x = 1\n",
            "py.typed": b"",
            "data/table.csv": b"a,b\n1,2\n",
            "results/big.npz": b"z" * 1000,
            "__pycache__/__init__.cpython-312.pyc": b"own",
        },
    )
    # a plain installed copy: fewer files than the checkout, source may even differ (that is the source hash's concern)
    clean = make_package(
        tmp_path / "clean", {"__init__.py": "x = 2\n", "py.typed": b"", "__pycache__/x.cpython-312.pyc": b"ignored"}
    )
    report = treehash.unmatched_non_source_files(clean, own)
    assert report == {"checked": 1, "unmatched_count": 0, "unmatched": []}
    shadow = make_package(
        tmp_path / "shadow",
        {
            "__init__.py": "x = 1\n",
            "py.typed": b"",
            "protocol/contracts.cpython-312-aarch64-linux-gnu.so": b"\x7fELF",
            "sourceless.pyc": b"x",
            "hook.pth": b"import os\n",
            "data/table.csv": b"a,b\n9,9\n",
        },
    )
    (shadow / "linked").symlink_to("/etc/passwd")
    (shadow / "linked_dir").symlink_to(tmp_path)
    report = treehash.unmatched_non_source_files(shadow, own)
    assert sorted(report["unmatched"]) == [
        "data/table.csv",
        "hook.pth",
        "linked",
        "linked_dir",
        "protocol/contracts.cpython-312-aarch64-linux-gnu.so",
        "sourceless.pyc",
    ]
    assert report["unmatched_count"] == 6 and report["checked"] == 7


def test_the_evaluator_check_reads_the_worker_copy_through_the_view_and_skips_the_supervisors_own_directory(tmp_path):
    import passage
    import passage_eval

    own = {
        "passage": Path(passage.__file__).resolve().parent,
        "passage_eval": Path(passage_eval.__file__).resolve().parent,
    }
    probe = {
        "modules": {name: {"origin": str(path / "__init__.py"), "locations": [str(path)]} for name, path in own.items()}
    }
    report = _evaluator_extra_files(probe, Path)
    assert all(row["same_directory_as_the_supervisor"] and row["unmatched_count"] == 0 for row in report.values())
    copies = {name: make_package(tmp_path / name, {"__init__.py": ""}) for name in own}
    (copies["passage_eval"] / "native.cpython-312-aarch64-linux-gnu.so").write_bytes(b"\x7fELF")
    probe = {
        "modules": {
            name: {"origin": str(path / "__init__.py"), "locations": [str(path)]} for name, path in copies.items()
        }
    }
    report = _evaluator_extra_files(probe, Path)
    assert report["passage"]["unmatched_count"] == 0
    assert report["passage_eval"]["unmatched"] == ["native.cpython-312-aarch64-linux-gnu.so"]
