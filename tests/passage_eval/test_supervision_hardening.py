"""Regression tests for the adversarial review of the supervised execution chain.

Each class of test reproduces one attack of the review against real worker processes (``process`` and,
where the attack is about confinement, ``bwrap``) and shows that it no longer works.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from passage_eval.cli import BenchSpec
from passage_eval.execution import read_run_artifact
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest
from passage_eval.supervision import PURPOSE_VERIFIED, TIMING_BOUNDARY_ID, identity, sandbox, treehash
from passage_eval.supervision.__main__ import main as supervision_main
from passage_eval.supervision.redaction import request_token
from passage_eval.supervision.staging import StagingError, stage_for_scoring
from passage_eval.supervision.supervisor import (
    PurposeError,
    RosterInput,
    SplitSpec,
    SupervisionError,
    SupervisorConfig,
    WarmupInput,
    WorkerProcess,
    assert_spec_has_no_dataset_location,
    build_runner,
    check_purpose_static,
    dataset_location_fragments,
    inspect_environment,
    run_supervised,
)

BWRAP = sandbox.available_bwrap()
ISOLATIONS = [
    pytest.param("process", id="process"),
    pytest.param("bwrap", id="bwrap", marks=pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")),
]
needs_bwrap = pytest.mark.skipif(not BWRAP, reason="unprivileged bwrap unavailable")

# --------------------------------------------------------------------------------------------------------------
# candidate package: entry -> impl (helper module), a decoder, a fake onnxruntime
# --------------------------------------------------------------------------------------------------------------

IMPL = r"""
import gc, hashlib, inspect, json, os, struct, sys, time
import numpy as np
from passage_eval.protocol.contracts import PlanResult

MARK = "original"
_CACHE = {}
MODES = ["ok", "memo", "mark", "evil_frame", "die", "big_diag", "stderr_flood", "big_table", "late_import", "leak",
         "daemon"]


def key(request):
    digest = hashlib.sha256(np.ascontiguousarray(request.elevation).tobytes() + repr((request.start, request.goal)).encode())
    return digest.hexdigest()


def route(request):
    return [request.start, (1, 1), request.goal]


def plan(request, kw):
    code = int(request.elevation[1, 0])
    mode = kw.get("force_mode") or MODES[code]
    if mode == "ok":
        return PlanResult(route(request), work={"expansions": 1}, diagnostics={"mark": MARK})
    if mode == "memo":                                # the reviewer's cache attack
        k = key(request)
        hit = k in _CACHE
        if not hit:
            time.sleep(0.05)
            _CACHE[k] = True
        return PlanResult(route(request), work={"expansions": 1},
                          diagnostics={"cache_hit": hit, "token": request.sample_id, "pid": os.getpid()})
    if mode == "mark":
        weights = kw.get("weights", "")
        weights = weights[2:] if weights.startswith("p:") else weights
        content = open(weights).read() if weights else None
        return PlanResult(route(request), work={"expansions": 1}, diagnostics={"mark": MARK, "weights": content, "pid": os.getpid()})
    if mode == "evil_frame":                          # the reviewer's crafted frame: zero elements, absurd dimension
        chan = next(o for o in gc.get_objects() if type(o).__name__ == "_Channel")
        seq = next(f.frame.f_locals["frame"].seq for f in inspect.stack()
                   if "frame" in f.frame.f_locals and hasattr(f.frame.f_locals["frame"], "seq"))
        header = json.dumps({"proto": "passage-ipc/v1", "type": "result", "seq": seq, "body": {},
                             "buffers": [{"name": "path", "dtype": "<f8", "shape": [0, 2 ** 70], "nbytes": 0}]}).encode()
        os.write(chan.ipc_out, struct.pack("<4sIQ", b"PIPC", len(header), 0) + header)
        time.sleep(5)
        return PlanResult(route(request), work={"expansions": 1})
    if mode == "die":
        os._exit(9)
    if mode == "big_diag":
        return PlanResult(route(request), work={"expansions": 1},
                          diagnostics={"blob": "A" * 400_000, "many": {str(i): i for i in range(5000)},
                                       "nested": {"a": {"b": {"c": {"d": {"e": {"f": {"g": "deep"}}}}}}},
                                       "list": list(range(10_000))})
    if mode == "stderr_flood":
        sys.stderr.write("E" * 200_000 + " SECRET-DATASET-ROOT:" + kw["dataset"][2:])
        sys.stderr.flush()
        os._exit(3)
    if mode == "big_table":
        return PlanResult(route(request), work={**{f"k{i}": 1 for i in range(100)}, "expansions": 1})
    if mode == "late_import":
        import onnxruntime  # noqa: F401  (a fake module in the candidate directory)
        return PlanResult(route(request), work={"expansions": 1})
    if mode == "leak":
        return PlanResult(route(request), work={"expansions": 1}, diagnostics={"where": kw["dataset"][2:], "list": [kw["dataset"][2:]]})
    if mode == "daemon":                              # escape the process group with a double fork + setsid
        pid = os.fork()
        if pid == 0:
            os.setsid()
            if os.fork() == 0:
                os.execvp("sleep", ["sleep", kw["marker"]])
            os._exit(0)
        os.waitpid(pid, 0)
        return PlanResult(route(request), work={"expansions": 1})
    raise AssertionError(mode)
"""

ENTRY = r"""
import numpy as np
from hpkg import impl
from passage_eval.protocol.contracts import PlanResult


class Planner:
    def __init__(self, **kw):
        self.kw = kw

    def plan(self, request):
        return impl.plan(request, self.kw)


class Predictor:
    def __init__(self, **kw):
        self.kw = kw

    def predict(self, request):
        return np.ones(request.elevation.shape, dtype=np.float32)


class Decoder:
    def __init__(self, **kw):
        self.kw = kw

    def decode(self, field, start, goal, obstacles, budget):
        return PlanResult([start, (1, 1), goal], work={"expansions": 4}, precision="float32",
                          diagnostics={"decoder_mark": impl.MARK})
"""

FAKE_ONNXRUNTIME = "VERSION = 'fake'\n"
LATE_IMPORT_AT_LOAD = "import onnxruntime\nfrom hpkg.entry import Planner, Predictor, Decoder\n"


def write_candidate(root: Path, *, import_banned_at_load: bool = False, fake_onnxruntime: bool = True) -> Path:
    package = root / "hpkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "impl.py").write_text(IMPL)
    (package / "entry.py").write_text(ENTRY)
    if import_banned_at_load:
        (package / "greedy.py").write_text(LATE_IMPORT_AT_LOAD)
    if fake_onnxruntime:  # a verified run refuses a tree that holds one (supervisor-side scan)
        (root / "onnxruntime").mkdir()
        (root / "onnxruntime" / "__init__.py").write_text(FAKE_ONNXRUNTIME)
    return root


@pytest.fixture
def cand(tmp_path):
    return write_candidate(tmp_path / "cand")


def request_for(code: int = 0, sample_id: str = "s0", *, budget="default", size=3, elevation_value=None) -> PlanRequest:
    elevation = np.zeros((size, size))
    elevation[1, 0] = code
    if elevation_value is not None:
        elevation[2, 0] = elevation_value
    if budget == "default":
        budget = BudgetSpec(wall_time_ms=20000.0, max_expansions=10)
    return PlanRequest(
        sample_id,
        elevation,
        np.zeros((size, size), dtype=bool),
        (0, 0),
        (size - 1, size - 1),
        "elevation",
        62.4,
        budget=budget,
    )


def bench(entry="hpkg.entry:Planner", kwargs=None, **changes):
    return replace(
        BenchSpec(
            "hardening",
            "classical",
            "fixture",
            inference=entry,
            inference_format="request",
            inference_kwargs=dict(kwargs or {}),
        ),
        **changes,
    )


@contextlib.contextmanager
def runner_for(
    cand_dir,
    entry="hpkg.entry:Planner",
    *,
    isolation="process",
    kwargs=None,
    split=None,
    ro_binds=(),
    protected=(),
    spec_changes=None,
    **config_changes,
):
    spec = bench(entry, kwargs, **(spec_changes or {}))
    iso = sandbox.IsolationConfig(
        level=isolation, ro_binds=tuple(str(p) for p in ro_binds), protected_paths=tuple(str(p) for p in protected)
    )
    settings = dict(ipc_baseline_pings=2, ipc_baseline_warmup=0, startup_timeout_s=120.0, deadline_grace_ms=20.0)
    settings.update(config_changes)
    config = SupervisorConfig(worker_python=sys.executable, isolation=iso, candidate_paths=(str(cand_dir),), **settings)
    entries = [entry] if split is None else [split.predictor, split.decoder]
    environment = inspect_environment(config, entries)
    try:
        with build_runner(spec, split, config, environment, seed=1) as runner:
            runner.environment = environment
            yield runner
    finally:
        environment.close()


@contextlib.contextmanager
def verified_runner_for(
    cand_dir,
    entry="hpkg.entry:Planner",
    *,
    split=None,
    kwargs=None,
    spec_changes=None,
    environment_digests=None,
    **config_changes,
):
    """A runner under ``verified-candidate-run`` on an environment that was hashed with the session cache.

    ``inspect_environment`` (diagnostic, cached) measures the trees once; the verified rules are then applied by
    ``build_runner`` with the digests of exactly those trees as the operator's environment allow-list (or the list given).
    A runner built this way is only for tests of the refusals: the digests are not recomputed here, which a real
    verified run always does, so ``build_runner`` also refuses on that ground unless the test says otherwise.
    """
    spec = bench(entry, kwargs, **(spec_changes or {}))
    diagnostic = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="bwrap"),
        candidate_paths=(str(cand_dir),),
        ipc_baseline_pings=2,
        ipc_baseline_warmup=0,
        startup_timeout_s=120.0,
        deadline_grace_ms=20.0,
    )
    entries = [entry] if split is None else [split.predictor, split.decoder]
    environment = inspect_environment(diagnostic, entries)
    try:
        digests = (
            tuple(r.sha256 for r in environment.bound.environment)
            if environment_digests is None
            else environment_digests
        )
        verified = replace(diagnostic, purpose=PURPOSE_VERIFIED, environment_allow_sha256=digests, **config_changes)
        with build_runner(spec, split, verified, environment, seed=1) as runner:
            runner.environment = environment
            yield runner
    finally:
        environment.close()


def run_one(runner, request, index=0, repeat=0, series="warm"):
    return runner.run_request(request, request_token(runner.secret, index), repeat, series)[0]


# --------------------------------------------------------------------------------------------------------------
# C3: identity covers every byte that is bound
# --------------------------------------------------------------------------------------------------------------


def tree_digest_of(identity_record, path: Path) -> str:
    rows = identity_record["bound_trees"]["candidate"]
    return next(row["sha256"] for row in rows if row["path"] == str(path.resolve()))


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_identity_covers_helper_modules_weights_and_bound_directories(cand, tmp_path, isolation):
    """The reviewer's ``identity`` probe: only entry.py was hashed; impl.py and everything else was not."""
    data = tmp_path / "data"
    data.mkdir()
    (data / "extra.bin").write_bytes(b"extra")
    weights = tmp_path / "weights.txt"
    weights.write_text("W1")
    with runner_for(cand, isolation=isolation, kwargs={"weights": str(weights)}, ro_binds=[data, weights]) as runner:
        ident = runner.identity
        files = {Path(p).name for c in ident["components"].values() for p in c["files"]}
        assert files == {"__init__.py", "entry.py"}  # per-module hashes still list the entry chain only ...
        trees = {Path(row["path"]).name: row for row in ident["bound_trees"]["candidate"]}
        assert set(trees) == {"cand", "data", "weights.txt"}  # ... but the bound trees cover helpers, data and weights
        assert trees["cand"]["sha256"] == treehash.hash_tree(cand, exclude_dirs=("__pycache__",)).sha256
        assert trees["cand"]["files"] == 4 and trees["data"]["files"] == 1 and trees["weights.txt"]["kind"] == "file"
        assert ident["identity_method"] == identity.IDENTITY_METHOD
        assert ident["identity_binding"] == (
            "staged_read_only_copy" if isolation == "bwrap" else "in_place_hash_rechecked"
        )
        assert all(len(row["sha256"]) == 64 for row in ident["bound_trees"]["environment"])
        assert any(Path(row["path"]).name == "usr" for row in ident["bound_trees"]["environment"]) == (
            isolation == "bwrap"
        )
        assert json.loads(json.dumps(ident)) == ident
        before = identity.identity_digest(ident)
    (cand / "hpkg" / "impl.py").write_text(IMPL.replace('"original"', '"swapped"'))  # a helper changes
    with runner_for(cand, isolation=isolation, kwargs={"weights": str(weights)}, ro_binds=[data, weights]) as runner:
        assert tree_digest_of(runner.identity, cand) != trees["cand"]["sha256"]
        assert identity.identity_digest(runner.identity) != before
    (cand / "hpkg" / "impl.py").write_text(IMPL)
    weights.write_text("W2")  # weights change
    with runner_for(cand, isolation=isolation, kwargs={"weights": str(weights)}, ro_binds=[data, weights]) as runner:
        assert identity.identity_digest(runner.identity) != before


def test_weights_under_an_undeclared_kwarg_name_are_refused_until_they_are_declared(cand, tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"weights")
    for key in ("onnx_path", "weights", "model", "table"):
        with pytest.raises(identity.IdentityError, match="outside the declared candidate and runtime trees"):
            with runner_for(cand, kwargs={key: str(weights)}):
                pass
    with pytest.raises(identity.IdentityError, match="outside the declared"):  # nested kwargs too
        with runner_for(cand, kwargs={"options": {"files": [str(weights)]}}):
            pass
    with runner_for(
        cand, kwargs={"onnx_path": str(weights)}, ro_binds=[weights]
    ) as runner:  # declared: staged or hashed
        assert any(Path(row["path"]) == weights.resolve() for row in runner.identity["bound_trees"]["candidate"])
    with runner_for(cand, kwargs={"checkpoint": str(weights)}) as runner:  # checkpoint keys auto-declare
        assert runner.identity["components"]["planner"]["checkpoints"]["checkpoint"]["sha256"]
        assert any(Path(row["path"]) == weights.resolve() for row in runner.identity["bound_trees"]["candidate"])


@needs_bwrap
def test_bwrap_runs_the_staged_copy_whatever_happens_to_the_original_afterwards(cand, tmp_path):
    """No TOCTOU between hashing and use: editing the source after staging does not change what a restarted worker runs."""
    weights = tmp_path / "weights.txt"
    weights.write_text("W-original")
    with runner_for(
        cand, isolation="bwrap", kwargs={"weights": f"p:{weights}", "force_mode": "mark"}, ro_binds=[weights]
    ) as runner:
        first = run_one(runner, request_for(2))
        (cand / "hpkg" / "impl.py").write_text(IMPL.replace('MARK = "original"', 'MARK = "edited-after-staging"'))
        weights.write_text("W-edited")
        runner.workers["planner"].kill()  # force a restart: new process, new import
        second = run_one(runner, request_for(2), 1)
        runner.identity_check()  # the staged copy still matches its digest
    assert first["diagnostics"]["mark"] == second["diagnostics"]["mark"] == "original"
    assert first["diagnostics"]["weights"] == second["diagnostics"]["weights"] == "W-original"
    assert runner.workers["planner"].generation == 1  # a genuinely new process imported the code again


def test_process_isolation_cannot_stage_so_an_edit_during_the_run_invalidates_it(cand):
    with runner_for(cand, isolation="process") as runner:
        run_one(runner, request_for(0))
        (cand / "hpkg" / "impl.py").write_text(IMPL + "\n# tampered\n")
        with pytest.raises(identity.IdentityError, match="changed during the run"):
            runner.identity_check()


@needs_bwrap
def test_the_venv_and_the_os_layer_are_hashed_and_recorded(cand):
    with runner_for(cand, isolation="bwrap") as runner:
        env = {Path(row["path"]).name: row for row in runner.identity["bound_trees"]["environment"]}
        assert "usr" in env and ".venv" in env or any("venv" in name for name in env)
        os_layer = runner.bound.os_layer
        assert os_layer["path"] == "/usr" and os_layer["hashed"] is True and len(os_layer["sha256"]) == 64
        assert os_layer["hash_source"] in {"computed", "stat_signature_cache"}
        for row in runner.identity["bound_trees"]["environment"]:
            assert row["hash_source"] in {"computed", "stat_signature_cache"} and len(row["stat_signature"]) == 64
        runner.bound.recheck()


def test_kwargs_path_policy_unit(tmp_path):
    weights = tmp_path / "weights.bin"
    weights.write_bytes(b"w")
    with pytest.raises(identity.IdentityError, match="outside"):
        identity.check_kwargs_paths("planner", {"w": str(weights)}, ["/nonexistent-root"])
    assert identity.check_kwargs_paths(
        "planner", {"w": str(weights), "n": 3, "s": "relu", "slash": "/"}, [tmp_path]
    ) == ["w"]
    assert identity.check_kwargs_paths("planner", {"t": "/does/not/exist"}, []) == []  # only existing files matter
    with pytest.raises(identity.IdentityError, match="outside"):
        identity.check_kwargs_paths("planner", {"home": "~/"}, [tmp_path])
    with pytest.raises(identity.IdentityError, match="outside"):
        identity.check_kwargs_paths("decoder", {"nested": {"a": [str(weights)]}}, ["/usr"])
    link = tmp_path / "link.bin"
    link.symlink_to(weights)
    with pytest.raises(identity.IdentityError, match="outside"):  # a symlink does not smuggle a file in
        identity.check_kwargs_paths("planner", {"w": str(link)}, [tmp_path / "other"])


def test_loaded_files_are_only_a_cross_check_and_the_manifest_labels_them_self_reported(cand):
    with runner_for(cand) as runner:
        ready = runner.ready["planner"]
    assert [Path(p).name for p in ready["loaded_files"]] == ["entry.py"]
    assert "numpy" in ready["sys_modules_top_level"] and "hpkg" in ready["sys_modules_top_level"]
    assert ready["forbidden_modules"] == ["onnxruntime"]


# --------------------------------------------------------------------------------------------------------------
# H1: purpose
# --------------------------------------------------------------------------------------------------------------


def make_config(isolation="bwrap", **changes):
    return SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig(level=isolation), **changes)


def fake_roster(sha="a" * 64):
    return RosterInput("roster.json", sha, "/nonexistent-dataset")


PINNED = dict(decoder_allow_sha256=("b" * 64,), planner_allow_sha256=("c" * 64,), environment_allow_sha256=("d" * 64,))


@pytest.mark.parametrize(
    "changes, spec_changes, split, roster, match",
    [
        (dict(purpose=PURPOSE_VERIFIED, **PINNED), {}, None, fake_roster(), "unsandboxed"),  # process isolation
        (dict(purpose=PURPOSE_VERIFIED, **PINNED), {}, None, None, "pinned roster"),
        (
            dict(purpose=PURPOSE_VERIFIED),
            {},
            SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder"),
            fake_roster(),
            "decoder allow-list",
        ),
        (
            dict(purpose=PURPOSE_VERIFIED, environment_allow_sha256=("d" * 64,)),
            {},
            SplitSpec("a:b", "c:d"),
            fake_roster(),
            "decoder allow-list",
        ),
        (
            dict(purpose=PURPOSE_VERIFIED, decoder_allow_sha256=("b" * 64,)),
            {},
            SplitSpec("a:b", "c:d"),
            fake_roster(),
            "environment allow-list",
        ),
        # N2: the decision never depends on the family named in the spec; planner mode needs a pinned planner whatever it is
        (
            dict(purpose=PURPOSE_VERIFIED, environment_allow_sha256=("d" * 64,)),
            {},
            None,
            fake_roster(),
            "planner allow-list",
        ),
        (
            dict(purpose=PURPOSE_VERIFIED, environment_allow_sha256=("d" * 64,)),
            {"family": "classical"},
            None,
            fake_roster(),
            "planner allow-list",
        ),
        (
            dict(purpose=PURPOSE_VERIFIED, environment_allow_sha256=("d" * 64,)),
            {"family": "dense_segmentation"},
            None,
            fake_roster(),
            "planner allow-list",
        ),
        (
            dict(purpose=PURPOSE_VERIFIED, environment_allow_sha256=("d" * 64,)),
            {"family": "an unknown family name"},
            None,
            fake_roster(),
            "planner allow-list",
        ),
    ],
)
def test_verified_candidate_run_refuses_what_it_cannot_vouch_for(changes, spec_changes, split, roster, match):
    isolation = "process" if match == "unsandboxed" else "bwrap"
    config = make_config(isolation, **changes)
    with pytest.raises(PurposeError, match=match):
        check_purpose_static(config, bench(**spec_changes), split, roster)


@pytest.mark.parametrize(
    "family", ["classical", "dense_segmentation", "dense_regression", "learned_field", "anything", ""]
)
def test_the_family_in_the_spec_text_never_decides_what_a_verified_run_needs(family):
    """N2: the 0.2 rule asked for split mode when the family was one of six names and nothing otherwise."""
    planner = make_config(
        "bwrap", purpose=PURPOSE_VERIFIED, planner_allow_sha256=("c" * 64,), environment_allow_sha256=("d" * 64,)
    )
    split_config = make_config(
        "bwrap", purpose=PURPOSE_VERIFIED, decoder_allow_sha256=("b" * 64,), environment_allow_sha256=("d" * 64,)
    )
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    spec = bench(family=family)
    planner_checks = check_purpose_static(planner, spec, None, fake_roster())
    split_checks = check_purpose_static(split_config, spec, split, fake_roster())
    assert {"check": "planner_allow_list_present", "passed": True, "applies": True} in planner_checks
    assert {"check": "decoder_allow_list_present", "passed": True, "applies": True} in split_checks
    with pytest.raises(PurposeError, match="planner allow-list"):
        check_purpose_static(split_config, spec, None, fake_roster())  # a decoder pin does not pin a planner
    with pytest.raises(PurposeError, match="decoder allow-list"):
        check_purpose_static(planner, spec, split, fake_roster())  # a planner pin does not pin a decoder


def test_verified_runs_refuse_a_tree_hash_cache_and_never_read_one_from_the_environment(monkeypatch):
    """N5: a cache keyed by stat signature can be filled by an earlier process, so a verified run recomputes everything."""
    monkeypatch.setenv("PASSAGE_TREE_HASH_CACHE", "/tmp/some-cache")
    assert make_config(purpose=PURPOSE_VERIFIED).cache_setting() is None
    with pytest.raises(PurposeError, match="tree-hash cache is not allowed"):
        make_config(purpose=PURPOSE_VERIFIED, tree_hash_cache="/explicit")
    assert make_config(purpose=PURPOSE_VERIFIED, tree_hash_cache="off").cache_setting() is None
    assert make_config().cache_setting() == "/tmp/some-cache"
    assert make_config(tree_hash_cache="/explicit").cache_setting() == "/explicit"
    assert make_config(tree_hash_cache="off").cache_setting() == "off"


def test_verified_runs_refuse_to_switch_off_the_default_forbidden_modules(tmp_path):
    """ONNX Runtime: --no-default-forbidden-modules is a diagnostic option only."""
    with pytest.raises(PurposeError, match="forbidden modules"):
        make_config(purpose=PURPOSE_VERIFIED, forbidden_modules=())
    with pytest.raises(PurposeError, match="forbidden modules"):
        make_config(purpose=PURPOSE_VERIFIED, forbidden_modules=("torch",))
    assert make_config(purpose=PURPOSE_VERIFIED, forbidden_modules=("onnxruntime", "torch")).forbidden_modules == (
        "onnxruntime",
        "torch",
    )
    assert make_config(forbidden_modules=()).forbidden_modules == ()  # a diagnostic run may
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(bench().to_dict()))
    with pytest.raises(PurposeError, match="forbidden modules"):
        supervision_main(
            [
                "run",
                "--spec",
                str(spec_path),
                "--output",
                str(tmp_path / "out"),
                "--worker-python",
                sys.executable,
                "--isolation",
                "bwrap",
                "--purpose",
                PURPOSE_VERIFIED,
                "--no-default-forbidden-modules",
                "--planner-allow-sha256",
                "c" * 64,
                "--environment-allow-sha256",
                "d" * 64,
            ]
        )
    assert not (tmp_path / "out").exists()


def test_verified_candidate_run_accepts_a_complete_configuration_and_diagnostic_accepts_anything():
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    config = make_config(
        "bwrap", purpose=PURPOSE_VERIFIED, decoder_allow_sha256=("b" * 64,), environment_allow_sha256=("d" * 64,)
    )
    checks = check_purpose_static(config, bench(family="dense_segmentation"), split, fake_roster())
    assert {"check": "isolation_is_bwrap", "passed": True} in checks
    assert {"check": "environment_allow_list_present", "passed": True} in checks and {
        "check": "no_tree_hash_cache",
        "passed": True,
    } in checks
    assert check_purpose_static(make_config("process"), bench(family="dense_segmentation"), None, None) == [
        {"check": "purpose", "value": "diagnostic"}
    ]
    with pytest.raises(ValueError, match="purpose"):
        make_config(purpose="verified")
    for label in ("decoder", "planner", "environment"):
        with pytest.raises(ValueError, match=f"{label} allow-list.*64-character"):
            make_config(**{f"{label}_allow_sha256": ("xyz",)})


def test_run_supervised_enforces_the_purpose_before_loading_any_scene(tmp_path):
    spec = bench(family="dense_regression")
    with pytest.raises(PurposeError):
        run_supervised(
            spec, tmp_path / "out", make_config("bwrap", purpose=PURPOSE_VERIFIED)
        )  # planner mode, no roster
    assert not (tmp_path / "out").exists()


@needs_bwrap
def test_a_decoder_outside_the_operator_allow_list_is_refused_for_a_verified_run(cand):
    """Hashed once with the session cache; the verified-only requirements are then applied to that environment."""
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    with pytest.raises(PurposeError, match="not in the operator allow-list"):
        with verified_runner_for(cand, "hpkg.entry:Predictor", split=split, decoder_allow_sha256=("c" * 64,)):
            pass


# --------------------------------------------------------------------------------------------------------------
# H3 / H4: decoder allow-list and the source of the counters
# --------------------------------------------------------------------------------------------------------------


def decoder_pin(cand: Path) -> str:
    return treehash.hash_tree(
        cand / "hpkg", role="decoder_entry", symlinks="refuse", strict=True, exclude_dirs=treehash.CANDIDATE_EXCLUDES
    ).sha256


@pytest.mark.parametrize("pinned, source", [(True, "operator_decoder"), (False, "candidate_self_reported")])
def test_counters_source_follows_the_decoder_allow_list(cand, pinned, source):
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    allow = (decoder_pin(cand),) if pinned else ("d" * 64,)
    with runner_for(cand, "hpkg.entry:Predictor", split=split, decoder_allow_sha256=allow) as runner:
        record = run_one(runner, request_for(0))
        decoder_tree = runner.facts["decoder_entry_tree"]
    assert record["status"] == "ok" and record["counters_source"] == source == runner.counters_source
    assert decoder_tree["sha256"] == decoder_pin(cand) and decoder_tree["role"] == "decoder_entry"


def test_the_decoder_pin_covers_helper_modules_of_its_package_and_the_tree_hash_command_prints_it(cand, capsys):
    split = SplitSpec("hpkg.entry:Predictor", "hpkg.entry:Decoder")
    pin = decoder_pin(cand)
    assert supervision_main(["tree-hash", str(cand / "hpkg")]) == 0
    assert capsys.readouterr().out.strip() == pin
    (cand / "hpkg" / "impl.py").write_text(IMPL.replace('"original"', '"changed helper"'))
    assert decoder_pin(cand) != pin
    with runner_for(cand, "hpkg.entry:Predictor", split=split, decoder_allow_sha256=(pin,)) as runner:
        assert runner.counters_source == "candidate_self_reported"  # a changed helper no longer matches the pin


def test_the_evaluator_packages_in_the_worker_environment_must_match_the_supervisors(cand):
    with runner_for(cand) as runner:
        assert runner.facts["evaluator_matches_supervisor"] is True
        assert len(runner.facts["evaluator_source_hash"]) == 64


# --------------------------------------------------------------------------------------------------------------
# M1: a crafted frame or a restart storm never aborts the run
# --------------------------------------------------------------------------------------------------------------


def test_a_crafted_frame_is_a_runtime_error_for_that_request_and_the_run_continues(cand):
    """The reviewer's ``evil_frame``: shape [0, 2**70] used to raise a bare ValueError that aborted the whole run."""
    with runner_for(cand) as runner:
        rows = [run_one(runner, request_for(code, f"s{i}"), i) for i, code in enumerate([0, 3, 0])]
        events = [event["event"] for event in runner.events]
    assert [row["status"] for row in rows] == ["ok", "runtime_error", "ok"]
    assert rows[1]["diagnostics"]["failure"] == "protocol_error" and "dimension" in rows[1]["diagnostics"]["error"]
    assert rows[1]["completion_mode"] == "worker_killed_after_protocol_error"
    assert events == ["protocol_error", "restart"] and rows[2]["worker_restart_ms_before_request"] > 0


def test_after_the_restart_cap_every_remaining_task_is_a_runtime_error_with_the_reason(cand):
    requests = [request_for(4, f"s{i}") for i in range(6)]  # the candidate dies on every request
    with runner_for(cand, max_restarts=2) as runner:
        rows = [run_one(runner, request, i) for i, request in enumerate(requests)]
        cap, marked = runner.cap, runner.tasks_marked_not_executed
        restarts = runner.restarts
    assert [row["status"] for row in rows] == ["runtime_error"] * 6
    assert [row["diagnostics"].get("failure") for row in rows] == ["crashed", "crashed", "crashed"] + [
        "not_executed"
    ] * 3
    assert all(row["diagnostics"]["reason"] == "worker_restart_cap" for row in rows[3:])
    assert (
        rows[3]["diagnostics"]["error_type"] == "WorkerRestartCap"
        and rows[3]["completion_mode"] == "not_executed_worker_restart_cap"
    )
    assert cap["reason"] == "worker_restart_cap" and restarts == 2 and marked == 3
    for row in rows[3:]:  # latency of a never-run task is charged, not measured
        assert row["latency_charged_not_measured"] is True and row["total_wall_ms"] == 20000.0
        assert row["failed_attempt_ms"] == 20000.0 and row["hops"] == [] and row["safe_completion"] is True


def test_the_restart_cap_still_writes_the_complete_artifact_for_every_series(tmp_path, cand):
    spec, spec_path = write_spec(
        tmp_path, cand, inference_kwargs={"force_mode": "die"}, latency_repeats=2, latency_warmup=0
    )
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand, "process", "--max-restarts", "1") == 0  # the run is not aborted
    cold, warm = read_run_artifact(run / "seed-1234" / "cold"), read_run_artifact(run / "seed-1234")
    for artifact, repeats in ((cold, 1), (warm, 2)):  # complete and structurally valid
        assert len(artifact.records) == repeats * len(artifact.manifest["roster"])
        assert {r["status"] for r in artifact.records} == {"runtime_error"}
    # cold: both tasks crash (one restart used); warm/0: first task crashes, the cap hits; warm/1: nothing is started
    assert [r["diagnostics"].get("reason") for r in cold.records] == [None, None]
    assert [r["diagnostics"].get("reason") for r in warm.records] == [
        None,
        "worker_restart_cap",
        "worker_restart_cap",
        "worker_restart_cap",
    ]
    assert [r["diagnostics"]["failure"] for r in warm.records] == ["crashed"] + ["not_executed"] * 3
    # latency_warmup is 0: the measured series ran without a warm-up, so it is called unwarmed, never warm
    assert warm.manifest["series"] == "unwarmed"
    assert [(r["pass_id"], r["sample_id"]) for r in warm.records] == [
        ("unwarmed/0", "0000000000"),
        ("unwarmed/0", "0000000001"),
        ("unwarmed/1", "0000000000"),
        ("unwarmed/1", "0000000001"),
    ]
    assert cold.manifest["supervision"]["restart_cap"]["cap_reached"] is False
    cap = warm.manifest["supervision"]["restart_cap"]
    assert cap["cap_reached"] is True and cap["reason"] == "worker_restart_cap" and cap["max_restarts"] == 1
    assert cap["unplanned_restarts_used"] == 1 and cap["tasks_marked_not_executed_in_this_series"] == 3
    assert [p["records"] for p in warm.manifest["supervision"]["passes"]] == [2, 2]
    assert "restart_cap_reached" in [e["event"] for e in warm.manifest["supervision"]["worker_events"]]


def test_a_restart_that_fails_is_a_runtime_error_for_that_request_not_an_abort(cand, monkeypatch):
    calls = {"n": 0}
    original = WorkerProcess.start

    def flaky(self):
        calls["n"] += 1
        if calls["n"] == 1:  # the first restart cannot load the candidate
            raise SupervisionError("planner worker refused to load hpkg.entry:Planner: simulated")
        return original(self)

    with runner_for(cand) as runner:
        monkeypatch.setattr(WorkerProcess, "start", flaky)
        rows = [run_one(runner, request_for(code, f"s{i}"), i) for i, code in enumerate([4, 0, 0])]
        events = [event["event"] for event in runner.events]
    assert [row["status"] for row in rows] == ["runtime_error"] * 2 + ["ok"]
    assert (
        rows[1]["diagnostics"]["reason"] == "worker_restart_failed"
        and rows[1]["completion_mode"] == "not_executed_worker_restart_failed"
    )
    assert events == ["worker_gone", "restart_failed", "restart"]


# --------------------------------------------------------------------------------------------------------------
# M4: candidate-controlled text is bounded
# --------------------------------------------------------------------------------------------------------------


def test_diagnostics_are_bounded_and_scrubbed_of_the_dataset_location(cand, tmp_path):
    secret_root = tmp_path / "secret-dataset-root"
    secret_root.mkdir()
    with runner_for(cand, kwargs={"dataset": f"p:{secret_root}"}) as runner:
        runner.scrub = tuple(dataset_location_fragments(secret_root))
        big, leak = run_one(runner, request_for(5, "big"), 0), run_one(runner, request_for(9, "leak"), 1)
    blob = json.dumps(big["diagnostics"])
    assert big["status"] == "ok" and len(blob.encode()) <= 4096
    assert big["diagnostics"].get("_truncated") is True or len(blob) < 4096  # the 1 MB blob is gone
    assert "A" * 2000 not in json.dumps(big)
    assert str(secret_root) not in json.dumps(leak) and "<redacted:protected-location>" in json.dumps(
        leak["diagnostics"]
    )


def test_a_stderr_flood_is_cut_to_a_small_scrubbed_tail(cand, tmp_path):
    secret_root = tmp_path / "secret-dataset-root"
    secret_root.mkdir()
    with runner_for(cand, kwargs={"dataset": f"p:{secret_root}"}) as runner:
        runner.scrub = tuple(dataset_location_fragments(secret_root))
        row = run_one(runner, request_for(6))
    tail = row["diagnostics"]["stderr_tail"]
    assert row["status"] == "runtime_error" and row["diagnostics"]["failure"] == "crashed"
    assert 0 < len(tail) <= 1024 and str(secret_root) not in tail
    assert len(json.dumps(row).encode()) < 8192


def test_an_oversized_counter_table_is_a_protocol_error(cand):
    with runner_for(cand) as runner:
        row = run_one(runner, request_for(7))
        again = run_one(runner, request_for(0), 1)
    assert row["status"] == "runtime_error" and row["diagnostics"]["failure"] == "protocol_error"
    assert "small mapping" in row["diagnostics"]["error"] and again["status"] == "ok"


def test_sanitizer_unit_limits():
    from passage_eval.supervision import sanitize

    assert sanitize.cap_text("x" * 5000, 100).endswith("[truncated]") and len(sanitize.cap_text("x" * 5000, 100)) == 100
    assert sanitize.cap_text("see /data/root/file", 1000, ["/data/root"]) == f"see {sanitize.REDACTED}/file"
    capped = sanitize.cap_json({"a": [1] * 500, "b": {str(i): i for i in range(500)}, "c": "z" * 5000})
    assert len(capped["a"]) == 65 and capped["b"]["_truncated_items"] == 436 and len(capped["c"]) <= 1024
    whole = sanitize.cap_json({"k": "v" * 1000, "l": ["w" * 1000] * 60})  # each part is fine, the sum is not
    assert (
        whole["_truncated"] is True and whole["limit_bytes"] == 4096 and whole["original_bytes_after_item_caps"] > 4096
    )
    assert sanitize.cap_json({"d": {"d": {"d": {"d": {"d": {"d": {"d": 1}}}}}}}) == {
        "d": {"d": {"d": {"d": {"d": {"d": "<depth-limit>"}}}}}
    }
    for bad in ({"x" * 65: 1}, {str(i): 1 for i in range(65)}, [1]):
        with pytest.raises(ValueError):
            sanitize.check_table("t", bad)


# --------------------------------------------------------------------------------------------------------------
# Pilot findings: forbidden modules and the spec / dataset location
# --------------------------------------------------------------------------------------------------------------


def test_a_forbidden_module_imported_at_load_stops_the_worker_before_it_serves_anything(tmp_path):
    cand = write_candidate(tmp_path / "cand", import_banned_at_load=True)
    with pytest.raises(SupervisionError, match="refused to load.*forbidden module"):
        with runner_for(cand, "hpkg.greedy:Planner"):
            pass
    with runner_for(cand, "hpkg.greedy:Planner", forbidden_modules=()) as runner:  # the guard is what stops it
        assert "onnxruntime" in runner.ready["planner"]["sys_modules_top_level"]
        assert runner.ready["planner"]["forbidden_modules"] == []


def test_a_forbidden_module_imported_during_a_request_is_a_runtime_error_and_the_worker_is_killed(cand):
    with runner_for(cand) as runner:
        rows = [run_one(runner, request_for(code, f"s{i}"), i) for i, code in enumerate([0, 8, 0])]
        events = [event for event in runner.events]
    assert [row["status"] for row in rows] == ["ok", "runtime_error", "ok"]
    assert (
        rows[1]["diagnostics"]["error_type"] == "ForbiddenModule" and "onnxruntime" in rows[1]["diagnostics"]["error"]
    )
    assert rows[1]["diagnostics"]["worker_fatal"] is True
    assert [e["event"] for e in events] == ["worker_fatal_error", "restart"] and events[0]["phase"] == "call"


def test_additional_forbidden_modules_are_enforced_too(cand):
    # numpy is imported by the worker itself, so a candidate cannot even load once it is forbidden
    with pytest.raises(SupervisionError, match="forbidden module"):
        with runner_for(cand, forbidden_modules=("numpy",)):
            pass


def spec_with(**fields):
    return bench(**fields)


def test_the_spec_may_not_contain_the_dataset_location_anywhere_but_dataset_path(tmp_path):
    root = tmp_path / "datasets" / "successor"
    (root / "3x3" / "validation").mkdir(parents=True)
    (root / "3x3" / "validation" / "data-3-0000000000-0000000063-00000.parquet").write_bytes(b"x")
    link = tmp_path / "link-to-dataset"
    link.symlink_to(root)
    for given in (root, link):
        fragments = dataset_location_fragments(given)
        assert str(root) in fragments and "3x3/validation/data-3-0000000000-0000000063-00000.parquet" in fragments
        assert_spec_has_no_dataset_location(
            spec_with(dataset_path=str(given), dataset_version="local export, path withheld"), fragments
        )  # dataset_path itself is allowed
    fragments = dataset_location_fragments(link)
    leaks = [
        dict(dataset_version=f"LOCAL export at {root}, validation split"),  # the resolved target
        dict(dataset_version=f"LOCAL export at {link}"),  # the symlink as given
        dict(extra={"notes": {"where": [f"read {root}/3x3/validation"]}}),  # nested
        dict(hardware_stack="data-3-0000000000-0000000063-00000.parquet was read"),  # a file name
        dict(method_name=f"m@{root}"),
    ]
    for fields in leaks:
        with pytest.raises(ValueError, match="contains the dataset location") as caught:
            assert_spec_has_no_dataset_location(spec_with(dataset_path=str(link), **fields), fragments)
        assert str(root) not in str(caught.value)  # the message does not repeat it


def test_run_supervised_refuses_to_start_when_a_spec_string_names_the_dataset(tmp_path, monkeypatch):
    root = tmp_path / "ds"
    (root / "3x3" / "validation").mkdir(parents=True)
    spec = bench(dataset_path=str(root), resolution=3, dataset_version=f"export at {root}", latency_warmup=0)
    started = []
    monkeypatch.setattr("passage_eval.supervision.supervisor.inspect_environment", lambda *a, **k: started.append(a))
    with pytest.raises(ValueError, match="'dataset_version' contains the dataset location"):
        run_supervised(spec, tmp_path / "out", make_config("process"))
    assert started == [] and not (tmp_path / "out").exists()  # nothing was started or written


# --------------------------------------------------------------------------------------------------------------
# end-to-end helpers: a tiny released-format dataset with a validation and a calibration split
# --------------------------------------------------------------------------------------------------------------


def release_row(i: int, *, split="validation", elevation=0.75):
    from passage.utils import encode_tensor_blob

    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0], tensor[:, :, 1] = elevation + 0.001 * i, -1
    tensor[0, 0, 1], tensor[2, 2, 1], tensor[0, 2, 2] = 0, 1, 1
    cost = float(60 * np.sqrt(2))
    return {
        "id": f"{i:010d}",
        "tensor": encode_tensor_blob(tensor),
        "path_astar_elevation_obstacles": [[0, 0], [1, 1], [2, 2]],
        "path_astar_elevation_free": [[0, 0], [1, 1], [2, 2]],
        "metadata": {
            "resolution": 3,
            "split": split,
            "index": i,
            "calibration": {"min": -100.0, "max": 100.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
            "paths": {"astar_elevation": {"obstacles": {"cost": cost}, "free": {"cost": cost}}},
        },
    }


def write_split(root: Path, split: str, rows: list[dict]) -> None:
    folder = root / "data" / "3x3" / split
    folder.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), folder / "data-00000.parquet")


def write_spec(root: Path, cand_dir: Path, *, n_rows=2, warm_rows=None, **changes):
    write_split(root, "validation", [release_row(i) for i in range(n_rows)])
    write_split(
        root,
        "calibration",
        warm_rows
        if warm_rows is not None
        else [release_row(100 + i, split="calibration", elevation=0.25) for i in range(3)],
    )
    spec = BenchSpec(
        "hardening-e2e",
        "classical",
        "fixture",
        inference="hpkg.entry:Planner",
        inference_format="request",
        inference_kwargs={"force_mode": "ok"},
        resolution=3,
        n_samples=n_rows,
        seeds=[1234],
        dataset_path=str(root / "data"),
        latency_repeats=2,
        latency_warmup=2,
        results_root=str(root / "results"),
        budget={"max_expansions": 10, "wall_time_ms": 20000},
    )
    spec = replace(spec, **changes)
    path = root / "spec.json"
    path.write_text(json.dumps(spec.to_dict()))
    return spec, path


def cli_run(spec_path, output, cand_dir, isolation, *extra):
    return supervision_main(
        [
            "run",
            "--spec",
            str(spec_path),
            "--output",
            str(output),
            "--worker-python",
            sys.executable,
            "--isolation",
            isolation,
            "--candidate-path",
            str(cand_dir),
            "--ipc-baseline-pings",
            "2",
            *extra,
        ]
    )


# --------------------------------------------------------------------------------------------------------------
# H2: fresh worker and fresh tokens for every timed pass; warm-up on disjoint inputs
# --------------------------------------------------------------------------------------------------------------


def test_control_one_worker_for_every_pass_lets_a_memoising_candidate_hit_its_cache(cand):
    """What the review measured: with one long-lived worker the 'warm' repeats are cache lookups."""
    with runner_for(cand, kwargs={"force_mode": "memo"}) as runner:
        rows = [run_one(runner, request_for(0, "task-A"), 0, repeat) for repeat in range(3)]
    assert [row["diagnostics"]["cache_hit"] for row in rows] == [False, True, True]


@pytest.mark.parametrize("isolation", ISOLATIONS)
def test_a_memoising_candidate_never_hits_its_cache_in_a_timed_pass(tmp_path, cand, isolation):
    spec, spec_path = write_spec(tmp_path, cand, inference_kwargs={"force_mode": "memo"})
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand, isolation, "--warmup-split", "calibration") == 0
    cold, warm = read_run_artifact(run / "seed-1234" / "cold"), read_run_artifact(run / "seed-1234")
    rows = cold.records + warm.records
    assert len(rows) == 2 + 2 * 2 and [r["diagnostics"]["cache_hit"] for r in rows] == [False] * 6  # no timed cache hit
    by_pass = {}
    for row in rows:
        by_pass.setdefault(row["pass_id"], []).append(row)
    assert sorted(by_pass) == ["cold/0", "warm/0", "warm/1"]
    pids = {name: {r["diagnostics"]["pid"] for r in group} for name, group in by_pass.items()}
    assert all(len(group) == 1 for group in pids.values())  # one worker serves a whole pass
    if isolation == "process":  # (bwrap pids are namespace-local)
        assert len({next(iter(g)) for g in pids.values()}) == 3  # three passes, three workers
    tokens = [r["diagnostics"]["token"] for r in rows]
    assert len(set(tokens)) == 6 and all(t.startswith("req-") for t in tokens)  # fresh per pass
    sample_ids = [r["sample_id"] for r in rows]
    assert set(sample_ids) == {"0000000000", "0000000001"}  # real ids stay supervisor-side
    assert all(t not in sample_ids for t in tokens)
    for row in warm.records:
        assert row["pass_worker_start_ms"] > 0 and row["worker_restart_ms_before_request"] == 0.0
    passes = warm.manifest["supervision"]["passes"]
    assert [p["warmup"]["requests"] for p in passes] == [2, 2] and all(
        p["warmup"]["statuses"] == {"ok": 2} for p in passes
    )
    assert cold.manifest["supervision"]["passes"][0]["warmup"] is None
    generations = [p["worker_generations_at_start"]["planner"] for p in cold.manifest["supervision"]["passes"] + passes]
    assert generations == sorted(generations) and len(set(generations)) == 3


def test_the_warmup_is_not_a_task_and_its_policy_is_recorded(tmp_path, cand):
    spec, spec_path = write_spec(tmp_path, cand)
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand, "process", "--warmup-split", "calibration") == 0
    warm = read_run_artifact(run / "seed-1234")
    assert set(warm.manifest["roster"]) == {"0000000000", "0000000001"}  # no warm-up id (…100…) anywhere
    assert "0000000100" not in (run / "seed-1234" / "requests.jsonl").read_text()
    policy = warm.manifest["supervision"]["timing_policy"]
    assert policy["boundary"] == TIMING_BOUNDARY_ID == "passage-timing-boundary/v3-supervised-ipc-fresh-pass"
    assert policy["warmup"]["policy"] == "disjoint_inputs" and policy["warmup"]["n_inputs"] == 2
    assert policy["warmup"]["disjoint_from_roster"] is True and policy["warmup_series_artifact"] == "not_produced"
    assert policy["worker_per_timed_pass"].startswith("fresh_workers") and "baseline" in " ".join(policy)
    assert "warmup_requests" in warm.manifest["supervision"]["not_in_boundary"]


def test_warmup_inputs_that_reuse_a_roster_id_are_refused(tmp_path, cand):
    spec, spec_path = write_spec(tmp_path, cand, warm_rows=[release_row(0, split="calibration", elevation=0.25)])
    with pytest.raises(ValueError, match="not disjoint"):
        cli_run(spec_path, tmp_path / "run", cand, "process", "--warmup-split", "calibration")
    assert not (tmp_path / "run").exists() or not any((tmp_path / "run").rglob("run_manifest.json"))


def test_warmup_inputs_that_repeat_a_roster_input_under_another_id_are_refused(monkeypatch):
    """Disjointness is checked by the ordinary-input identity as well, not only by the sample id."""
    from passage_eval import cli
    from passage_eval.supervision import supervisor

    roster = [request_for(0, "r0"), request_for(0, "r1", elevation_value=3.0)]
    renamed = PlanRequest(
        "warm-0",
        roster[1].elevation,
        roster[1].obstacles,
        roster[1].start,
        roster[1].goal,
        roster[1].cost_model,
        roster[1].cost_weight,
    )
    scene = type("S", (), {})()
    monkeypatch.setattr(cli, "_load_scenes", lambda spec: [scene])
    monkeypatch.setattr(PlanRequest, "from_scene", classmethod(lambda cls, s, **k: renamed))
    monkeypatch.setattr(supervisor, "candidate_scene", lambda s: s)
    with pytest.raises(ValueError, match="1 shared inputs"):
        supervisor._load_warmup_requests(bench(split="validation"), WarmupInput(split="calibration"), 1, "/x", roster)


def test_the_warmup_split_must_differ_from_the_evaluated_split(tmp_path, cand):
    spec, spec_path = write_spec(tmp_path, cand)
    for name in ("validation", "val"):
        with pytest.raises(ValueError, match="must differ"):
            cli_run(spec_path, tmp_path / f"run-{name}", cand, "process", "--warmup-split", name)


def test_latency_warmup_without_disjoint_inputs_is_refused_before_any_worker_starts(tmp_path, cand, monkeypatch):
    spec, spec_path = write_spec(tmp_path, cand)
    monkeypatch.setattr(
        "passage_eval.supervision.supervisor.inspect_environment", lambda *a, **k: pytest.fail("started")
    )
    with pytest.raises(ValueError, match="disjoint from the roster"):
        cli_run(spec_path, tmp_path / "run", cand, "process")
    spec0, spec0_path = write_spec(tmp_path / "zero", cand, latency_warmup=0)  # warm-up 0 needs none
    monkeypatch.undo()
    assert cli_run(spec0_path, tmp_path / "zero" / "run", cand, "process") == 0
    manifest = read_run_artifact(tmp_path / "zero" / "run" / "seed-1234").manifest
    assert manifest["supervision"]["timing_policy"]["warmup"]["policy"] == "none"
    assert [p["warmup"] for p in manifest["supervision"]["passes"]] == [None, None]


def test_v2_artifacts_stay_readable_and_unknown_boundaries_are_refused(tmp_path, cand):
    from passage_eval.supervision import LEGACY_TIMING_BOUNDARY_IDS, READABLE_TIMING_BOUNDARY_IDS

    assert LEGACY_TIMING_BOUNDARY_IDS == ("passage-timing-boundary/v2-supervised-ipc",)
    assert set(READABLE_TIMING_BOUNDARY_IDS) == {
        "passage-timing-boundary/v2-supervised-ipc",
        TIMING_BOUNDARY_ID,
        "passage-timing-boundary/v4-supervised-ipc-service",
        "passage-timing-boundary/v5-supervised-ipc-service-rewarm",
    }
    spec, spec_path = write_spec(tmp_path, cand, latency_warmup=0, measure_latency=False)
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand, "process") == 0
    staged_v3 = tmp_path / "staged-v3"
    stage_for_scoring(run, spec, staged_v3)
    for boundary, readable in (
        ("passage-timing-boundary/v2-supervised-ipc", True),
        ("request_copy_through_validation_recost_completion", False),
    ):
        copy = tmp_path / f"copy-{readable}"
        subprocess.run(["cp", "-r", str(run), str(copy)], check=True)
        manifest_path = copy / "seed-1234" / "run_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["timing_boundary"] = boundary
        manifest_path.write_text(json.dumps(manifest))
        if readable:
            stage_for_scoring(copy, spec, tmp_path / f"out-{readable}")
        else:
            with pytest.raises(StagingError, match="unknown timing boundary"):
                stage_for_scoring(copy, spec, tmp_path / f"out-{readable}")


# --------------------------------------------------------------------------------------------------------------
# H5: pinned roster
# --------------------------------------------------------------------------------------------------------------


def _load_roster_fixture():
    import importlib.util

    path = Path(__file__).resolve().parent / "test_roster.py"
    spec = importlib.util.spec_from_file_location("roster_fixture_for_supervision", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def roster_world(tmp_path_factory):
    """A pinned synthetic dataset root (PINNED.json), its roster (6 of 12 rows) and a disjoint calibration split.

    The evaluated split is validation: the supervisor refuses an evaluated test roster without a test admission.
    """
    fixture = _load_roster_fixture()
    root = tmp_path_factory.mktemp("roster-world")
    dataset = fixture.write_dataset(root / "dataset", split="validation")
    roster, _ = fixture.build(dataset, split="validation")
    roster_path = root / "roster.json"
    roster_path.write_text(json.dumps(roster))
    warm_dir = dataset / "3x3" / "calibration"
    warm_dir.mkdir()
    selected = {task["sample_id"] for task in roster["tasks"]}
    spare = [i for i in range(12) if f"{i:010d}" not in selected][:2]  # population rows the roster did not select
    rows = [fixture._row(i) for i in spare]
    for row in rows:  # unlisted in PINNED.json; other ids and other tensors
        row["metadata"] = dict(row["metadata"], split="calibration")
    pq.write_table(pa.Table.from_pylist(rows), warm_dir / "data-3-0000.parquet")
    cand_dir = write_candidate(root / "cand", fake_onnxruntime=False)
    spec = BenchSpec(
        "roster-e2e",
        "classical",
        "fixture",
        inference="hpkg.entry:Planner",
        inference_format="request",
        inference_kwargs={"force_mode": "ok"},
        resolution=3,
        split="validation",
        n_samples=6,
        seeds=[1234],
        latency_repeats=1,
        latency_warmup=1,
        budget={"max_expansions": 10, "wall_time_ms": 20000},
        results_root=str(root / "results"),
        dataset_version="pinned synthetic roster dataset",
        reference_policy="independent_analytic_reference",
        reference_cohort_id="synthetic-cohort",
    )
    spec_path = root / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    return dict(
        root=root,
        dataset=dataset,
        roster=roster,
        roster_path=roster_path,
        cand=cand_dir,
        spec=spec,
        spec_path=spec_path,
        fixture=fixture,
    )


def roster_args(world, **overrides):
    values = dict(roster=world["roster_path"], sha=world["roster"]["roster_sha256"], root=world["dataset"])
    values.update(overrides)
    return [
        "--roster",
        str(values["roster"]),
        "--roster-sha256",
        values["sha"],
        "--dataset-root",
        str(values["root"]),
        "--warmup-split",
        "calibration",
    ]


@pytest.fixture(scope="module")
def roster_run(roster_world):
    run = roster_world["root"] / "run"
    assert (
        cli_run(
            roster_world["spec_path"],
            run,
            roster_world["cand"],
            "process",
            *roster_args(roster_world),
            "--roster-id",
            "synthetic-public-test",
        )
        == 0
    )
    return run


def test_roster_run_records_the_roster_and_produces_a_record_for_every_task(roster_world, roster_run):
    roster = roster_world["roster"]
    expected = [task["sample_id"] for task in roster["tasks"]]
    for directory in (roster_run / "seed-1234", roster_run / "seed-1234" / "cold"):
        artifact = read_run_artifact(directory)
        assert artifact.manifest["roster"] == expected  # roster order, every task
        assert [r["sample_id"] for r in artifact.records] == expected * artifact.manifest["repeats"]
        block = artifact.manifest["supervision"]["roster"]
        assert block["roster_id"] == "synthetic-public-test" and block["benchmark_id"] == roster["benchmark_id"]
        assert block["roster_sha256"] == roster["roster_sha256"] and block["n_expected"] == 6 == block["n_loaded"]
        assert (
            block["dataset_root_identity"] == roster["dataset_root_identity"]
            and block["dataset_root_path_recorded"] is False
        )
        assert block["dataset"]["commit"] == roster["dataset"]["commit"] and block["labels"].startswith(
            "supervisor_side_only"
        )
        assert artifact.manifest["dataset"]["revision"] == roster["dataset"]["commit"]
        assert artifact.manifest["procedure"]["supervision"]["roster"]["roster_sha256"] == roster["roster_sha256"]
        assert str(roster_world["dataset"]) not in json.dumps(artifact.manifest)
        assert all(r["status"] == "ok" for r in artifact.records)


def test_roster_run_artifacts_hold_no_dataset_location_and_no_labels(roster_world, roster_run):
    blob = b"\n".join(
        p.read_bytes() for p in sorted(roster_run.rglob("*")) if p.is_file() and p.suffix in {".json", ".jsonl"}
    ).decode()
    for forbidden in (
        str(roster_world["dataset"]),
        "path_astar",
        "optimal_path",
        "oracle",
        roster_world["fixture"].MARKER,
    ):
        assert forbidden not in blob


def test_a_wrong_pin_a_foreign_cell_or_a_missing_dataset_root_is_refused(roster_world, tmp_path):
    world = roster_world
    with pytest.raises(ValueError, match="roster_sha256 differs"):
        cli_run(world["spec_path"], tmp_path / "a", world["cand"], "process", *roster_args(world, sha="0" * 64))
    other = replace(world["spec"], resolution=4)
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps(other.to_dict()))
    with pytest.raises(ValueError, match="does not match the roster's cell.*resolution"):
        cli_run(other_path, tmp_path / "b", world["cand"], "process", *roster_args(world))
    wrong_split = tmp_path / "split.json"
    wrong_split.write_text(json.dumps(replace(world["spec"], split="calibration").to_dict()))
    with pytest.raises(ValueError, match="split"):
        cli_run(wrong_split, tmp_path / "c", world["cand"], "process", *roster_args(world))
    with pytest.raises(ValueError, match="no readable PINNED.json"):
        cli_run(world["spec_path"], tmp_path / "d", world["cand"], "process", *roster_args(world, root=tmp_path))
    for args in (["--roster", str(world["roster_path"])], ["--dataset-root", str(world["dataset"])]):
        with pytest.raises(SystemExit):
            cli_run(world["spec_path"], tmp_path / "e", world["cand"], "process", *args)


def test_roster_runs_are_staged_for_scoring_but_the_unchanged_scorer_cannot_score_them(
    roster_world, roster_run, tmp_path
):
    world = roster_world
    roster = RosterInput(str(world["roster_path"]), world["roster"]["roster_sha256"], str(world["dataset"]))
    staged = tmp_path / "staged"
    directories = stage_for_scoring(roster_run, world["spec"], staged, roster=roster)
    assert {d.name for d in directories} == {"seed-1234", "cold"}
    manifest = json.loads((staged / "seed-1234" / "run_manifest.json").read_text())
    assert manifest["supervision"]["roster"]["roster_sha256"] == world["roster"]["roster_sha256"]
    with pytest.raises(StagingError, match="disagree about the roster"):
        stage_for_scoring(roster_run, world["spec"], tmp_path / "no-roster")
    with pytest.raises(SystemExit):
        supervision_main(
            [
                "stage-score",
                "--run",
                str(roster_run),
                "--spec",
                str(world["spec_path"]),
                "--output",
                str(tmp_path / "x"),
                "--roster",
                str(world["roster_path"]),
                "--roster-sha256",
                world["roster"]["roster_sha256"],
                "--dataset-root",
                str(world["dataset"]),
                "--score",
            ]
        )
    assert (
        supervision_main(
            [
                "stage-score",
                "--run",
                str(roster_run),
                "--spec",
                str(world["spec_path"]),
                "--output",
                str(tmp_path / "staged-cli"),
                "--roster",
                str(world["roster_path"]),
                "--roster-sha256",
                world["roster"]["roster_sha256"],
                "--dataset-root",
                str(world["dataset"]),
            ]
        )
        == 0
    )


def test_the_warmup_can_come_from_a_second_pinned_roster_and_must_be_disjoint_from_the_evaluated_one(
    roster_world, tmp_path
):
    world, fixture = roster_world, roster_world["fixture"]
    # A root that pins the evaluated validation split and a calibration split with the same rows: the warm-up roster
    # comes from calibration. (The supervisor refuses a test roster as warm-up source: test_warmup.py.)
    dataset = fixture.write_dataset(tmp_path / "shared", split="validation")
    receipt = json.loads((dataset / "PINNED.json").read_text())
    shard = dataset / "3x3" / "calibration" / "data-3-0000.parquet"
    shard.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([fixture._row(i, split="calibration") for i in range(12)]), shard)
    data = shard.read_bytes()
    receipt["files"].append(
        {"path": "3x3/calibration/data-3-0000.parquet", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    )
    fixture.write_receipt(dataset, receipt["files"])
    evaluated, _ = fixture.build(dataset, split="validation")
    evaluated_path = tmp_path / "evaluated-roster.json"
    evaluated_path.write_text(json.dumps(evaluated))
    selected = {task["sample_id"] for task in evaluated["tasks"]}

    def validation_roster(benchmark_id, n):  # a warm-up roster of the calibration copy
        return fixture.build(dataset, benchmark_id=benchmark_id, n_expected=n, split="calibration")[0]

    for attempt in range(200):  # a validation roster sharing no task id with the evaluated one
        second = validation_roster(f"warmup-roster-{attempt}", 2)
        if not selected & {task["sample_id"] for task in second["tasks"]}:
            break
    else:
        pytest.skip("no disjoint second roster for this tiny population")
    second_path = tmp_path / "warmup-roster.json"
    second_path.write_text(json.dumps(second))
    base = ["--roster", str(evaluated_path), "--roster-sha256", evaluated["roster_sha256"], "--dataset-root"]
    base.append(str(dataset))
    run = tmp_path / "run"
    assert (
        cli_run(
            world["spec_path"],
            run,
            world["cand"],
            "process",
            *base,
            "--warmup-roster",
            str(second_path),
            "--warmup-roster-sha256",
            second["roster_sha256"],
        )
        == 0
    )
    policy = read_run_artifact(run / "seed-1234").manifest["supervision"]["timing_policy"]["warmup"]
    assert policy["source"] == f"roster:{second['roster_sha256']}" and policy["disjoint_from_roster"] is True
    from passage_eval.io.roster import derive_roster

    # validation copies of exactly the evaluated rows: same ids and content (the warm-up reads only its first rows)
    every = derive_roster(validation_roster("every-validation-row", 12), sorted(selected), rule="evaluated rows")
    every_path = tmp_path / "every.json"
    every_path.write_text(json.dumps(every))
    with pytest.raises(ValueError, match="not disjoint"):
        cli_run(
            world["spec_path"],
            tmp_path / "overlap",
            world["cand"],
            "process",
            *base,
            "--warmup-roster",
            str(every_path),
            "--warmup-roster-sha256",
            every["roster_sha256"],
        )
    with pytest.raises(ValueError, match="not disjoint"):  # the evaluated roster itself
        cli_run(
            world["spec_path"],
            tmp_path / "itself",
            world["cand"],
            "process",
            *base,
            "--warmup-roster",
            str(evaluated_path),
            "--warmup-roster-sha256",
            evaluated["roster_sha256"],
        )
    with pytest.raises(SystemExit):  # a warm-up roster needs its own pin
        cli_run(
            world["spec_path"],
            tmp_path / "unpinned",
            world["cand"],
            "process",
            *base,
            "--warmup-roster",
            str(second_path),
        )


def test_warmup_requests_carry_the_wall_budget_so_a_hung_warmup_is_preempted(tmp_path, cand):
    spec, spec_path = write_spec(
        tmp_path,
        cand,
        inference_kwargs={"force_mode": "ok"},
        latency_repeats=1,
        budget={"max_expansions": 10, "wall_time_ms": 7000},
    )
    from passage_eval.supervision import supervisor

    requests, _ = supervisor._load_warmup_requests(
        spec, WarmupInput(split="calibration"), 2, spec.dataset_path, [request_for(0)]
    )
    assert [r.budget.wall_time_ms for r in requests] == [7000.0, 7000.0]


def environment_pins(cand_dir, entries, *, isolation="bwrap") -> list[str]:
    """Digests of every interpreter-environment tree the sandbox binds, as an operator reads them from a diagnostic run."""
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level=isolation),
        candidate_paths=(str(cand_dir),),
    )
    environment = inspect_environment(config, entries)
    try:
        return sorted({record.sha256 for record in environment.bound.environment})
    finally:
        environment.close()


def pin_args(digests) -> list[str]:
    return [arg for digest in digests for arg in ("--environment-allow-sha256", digest)]


@needs_bwrap
def test_a_complete_verified_candidate_run_in_split_mode_with_a_roster_and_an_operator_decoder(
    roster_world, tmp_path, monkeypatch
):
    """The whole chain under bwrap: roster, split, pinned decoder and environment, fresh workers, staged bytes.

    Nothing is cached: ``/usr``, the virtual environment and the evaluator sources are hashed again by the run itself,
    and their digests must equal the ones read from a (cached) diagnostic inspection.
    """
    monkeypatch.setattr(sys, "dont_write_bytecode", True)  # the evaluator trees are hashed twice: keep them still
    world = roster_world
    cand = world["cand"]
    spec = replace(world["spec"], family="dense_segmentation", latency_warmup=1)
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(spec.to_dict()))
    split_args = [
        "--split-predictor",
        "hpkg.entry:Predictor",
        "--split-decoder",
        "hpkg.entry:Decoder",
        "--split-precision",
        "float32_field",
    ]
    pin = decoder_pin(cand)
    digests = environment_pins(cand, ["hpkg.entry:Predictor", "hpkg.entry:Decoder"])
    run = tmp_path / "verified"
    assert (
        cli_run(
            spec_path,
            run,
            cand,
            "bwrap",
            *roster_args(world),
            *split_args,
            "--purpose",
            PURPOSE_VERIFIED,
            "--decoder-allow-sha256",
            pin,
            *pin_args(digests),
        )
        == 0
    )
    warm = read_run_artifact(run / "seed-1234")
    supervision = warm.manifest["supervision"]
    assert (
        supervision["purpose"]["name"] == PURPOSE_VERIFIED and supervision["purpose"]["verified_candidate_run"] is True
    )
    assert (
        supervision["isolation"]["effective_level"] == "bwrap"
        and supervision["isolation"]["candidate_code_runs_unsandboxed"] is False
    )
    assert supervision["interpreter_probe"]["executed_in"] == "bwrap_same_sandbox_as_worker"
    assert supervision["interpreter_probe"]["channel"]["kind"] == "dedicated_pipe_descriptor"
    assert supervision["interpreter_probe"]["channel"]["trailing_bytes_ignored"] == 0
    assert (
        supervision["counters"]["counters_source"] == "operator_decoder"
        and supervision["counters"]["pinned_role"] == "decoder"
    )
    assert supervision["counters"]["decoder_entry_tree"]["sha256"] == pin and supervision["counters"][
        "decoder_allow_list"
    ] == [pin]
    assert supervision["evaluator"]["matches_supervisor"] is True
    assert (
        supervision["bound_trees"]["binding"] == "staged_read_only_copy"
        and supervision["bound_trees"]["os_layer"]["hashed"]
    )
    assert all(row["hash_source"] == "computed" for row in supervision["bound_trees"]["environment"])  # no cache
    environment_pin = supervision["environment_pin"]
    assert environment_pin["all_trees_allow_listed"] is True and sorted(environment_pin["allow_list"]) == digests
    assert {row["sha256"] for row in environment_pin["trees"]} <= set(digests)
    views = supervision["worker_views"]
    assert (
        views["decoder"]["narrowed_to_pinned_tree"] is True
        and views["decoder"]["enforced_by"] == "bwrap_mount_namespace"
    )
    assert (
        str(cand / "hpkg") in views["decoder"]["read_only_binds"]
        and str(cand) not in views["decoder"]["read_only_binds"]
    )
    assert views["predictor"]["narrowed_to_pinned_tree"] is False and str(cand) in views["predictor"]["read_only_binds"]
    assert supervision["forbidden_modules"]["supervisor_scan"]["clean"] is True
    assert supervision["redaction"]["decoder_field_contract"] == "decoder-field/v1"
    assert supervision["forbidden_modules"]["evidence"] == "self_reported_by_the_worker"
    assert all(r["counters_source"] == "operator_decoder" and r["status"] == "ok" for r in warm.records)
    assert supervision["timing_policy"]["warmup"]["policy"] == "disjoint_inputs"
    assert "-X pycache_prefix=<private>" in supervision["interpreter_probe"]["flags"]
    # the same run is refused as soon as one requirement is dropped
    for args, match in (
        (["--decoder-allow-sha256", "e" * 64, *pin_args(digests)], "decoder allow-list|not in the operator allow-list"),
        (["--decoder-allow-sha256", pin], "environment allow-list"),
        (
            ["--decoder-allow-sha256", pin, *pin_args(["f" * 64])],
            r"not in the operator allow-list \(--environment-allow-sha256\)",
        ),
    ):
        with pytest.raises(PurposeError, match=match):
            cli_run(
                spec_path,
                tmp_path / f"refused-{len(args)}",
                cand,
                "bwrap",
                *roster_args(world),
                *split_args,
                "--purpose",
                PURPOSE_VERIFIED,
                *args,
            )
    with pytest.raises(PurposeError, match="tree-hash cache is not allowed"):
        cli_run(
            spec_path,
            tmp_path / "refused-cache",
            cand,
            "bwrap",
            *roster_args(world),
            *split_args,
            "--purpose",
            PURPOSE_VERIFIED,
            "--decoder-allow-sha256",
            pin,
            *pin_args(digests),
            "--tree-hash-cache",
            os.environ["PASSAGE_TREE_HASH_CACHE"],
        )
    with pytest.raises(PurposeError, match="unsandboxed"):
        cli_run(
            spec_path,
            tmp_path / "refused-process",
            cand,
            "process",
            *roster_args(world),
            *split_args,
            "--purpose",
            PURPOSE_VERIFIED,
            "--decoder-allow-sha256",
            pin,
            *pin_args(digests),
        )


@needs_bwrap
def test_a_complete_verified_candidate_run_in_planner_mode_with_a_pinned_planner(roster_world, tmp_path, monkeypatch):
    """N2: a classical baseline is verified through a planner pin, whatever the spec says about its family."""
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    world = roster_world
    cand = world["cand"]
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps(replace(world["spec"], family="classical", latency_warmup=1).to_dict()))
    pin = decoder_pin(cand)  # the digest of the planner's package tree, same algorithm
    digests = environment_pins(cand, ["hpkg.entry:Planner"])
    run = tmp_path / "verified-planner"
    assert (
        cli_run(
            spec_path,
            run,
            cand,
            "bwrap",
            *roster_args(world),
            "--purpose",
            PURPOSE_VERIFIED,
            "--planner-allow-sha256",
            pin,
            *pin_args(digests),
        )
        == 0
    )
    warm = read_run_artifact(run / "seed-1234")
    supervision = warm.manifest["supervision"]
    assert supervision["purpose"]["verified_candidate_run"] is True and supervision["mode"] == "planner"
    assert (
        supervision["counters"]["counters_source"] == "operator_planner"
        and supervision["counters"]["pinned_role"] == "planner"
    )
    assert supervision["counters"]["planner_entry_tree"]["sha256"] == pin and supervision["counters"][
        "planner_allow_list"
    ] == [pin]
    assert supervision["worker_views"]["planner"]["narrowed_to_pinned_tree"] is True
    assert all(r["counters_source"] == "operator_planner" and r["status"] == "ok" for r in warm.records)
    with pytest.raises(PurposeError, match="planner allow-list"):  # no pin, no verified run
        cli_run(
            spec_path,
            tmp_path / "unpinned",
            cand,
            "bwrap",
            *roster_args(world),
            "--purpose",
            PURPOSE_VERIFIED,
            *pin_args(digests),
        )


# --------------------------------------------------------------------------------------------------------------
# C2 at the level of a whole run, and the process escape that bwrap closes
# --------------------------------------------------------------------------------------------------------------


@needs_bwrap
def test_a_bwrap_run_probes_the_interpreter_in_the_sandbox_and_records_it(tmp_path, cand):
    spec, spec_path = write_spec(tmp_path, cand, latency_warmup=0, latency_repeats=1)
    run = tmp_path / "run"
    assert cli_run(spec_path, run, cand, "bwrap") == 0
    supervision = read_run_artifact(run / "seed-1234").manifest["supervision"]
    assert supervision["interpreter_probe"]["executed_in"] == "bwrap_same_sandbox_as_worker"
    assert supervision["interpreter_probe"]["flags"] == ["-I", "-B", "-X pycache_prefix=<private>"]
    assert supervision["interpreter_probe"]["uncovered_paths"] == []
    assert supervision["isolation"]["interpreter_probe"] == "executed_in_bwrap_same_sandbox_as_worker"
    assert supervision["runtime_layout"]["executable_pth_lines_run_inside_the_worker_sandbox"] >= 0
    assert supervision["isolation"]["read_only_bind_sources"]
    sources = {row["target"]: row["source"] for row in supervision["isolation"]["read_only_bind_sources"]}
    assert sources[str(cand.resolve())] == "private_staged_copy"


def find_marker(marker: str) -> list[int]:
    hits = []
    for pid in filter(str.isdigit, os.listdir("/proc")):
        try:
            if open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0")[:2] == [b"sleep", marker.encode()]:
                hits.append(int(pid))
        except OSError:
            pass
    return hits


@pytest.mark.parametrize(
    "isolation, survives",
    [
        pytest.param("process", True, id="process-documented-limit"),
        pytest.param("bwrap", False, id="bwrap-closes-it", marks=needs_bwrap),
    ],
)
def test_a_daemonising_candidate_survives_process_isolation_but_not_bwrap(cand, isolation, survives):
    """The reviewer's ``daemon`` probe: a double-forked setsid child escapes the worker's process group."""
    marker = f"3601.{os.getpid()}{len(isolation)}"
    with runner_for(cand, isolation=isolation, kwargs={"force_mode": "daemon", "marker": marker}) as runner:
        assert run_one(runner, request_for(0))["status"] == "ok"
    time.sleep(1.0)
    alive = find_marker(marker)
    try:
        assert bool(alive) is survives
    finally:
        for pid in alive:
            os.kill(pid, 9)
