# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Every guarded entry point refuses the test split before anything under a ``test`` directory is touched.

A synthetic dataset root holds a validation directory and a canary ``3x3/test`` directory (mode 000, one canary
file). ``os.open``, ``os.stat``, ``os.lstat``, ``os.scandir``, ``os.listdir`` and ``open`` record any call on a path
under the canary; the refusal must leave that record empty. The canary is a synthetic fixture: no file of the
released dataset exists here, and the released test split is never read.
"""

import builtins
import importlib.util
import io
import json
import os
import sys
from pathlib import Path

import pytest
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent
SCRIPT = HERE.parents[1] / "scripts" / "build_public_test_roster.py"


def _world():
    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


W = _world()


def _builder():
    spec = importlib.util.spec_from_file_location("build_public_test_roster_for_canary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _builder()


@pytest.fixture
def canary(tmp_path, monkeypatch):
    root = tmp_path / "root"
    (root / "3x3" / "validation").mkdir(parents=True)
    test_dir = root / "3x3" / "test"
    test_dir.mkdir()
    (test_dir / "data-3-0000.parquet").write_bytes(b"canary: this file must never be read")
    (root / "PINNED.json").write_text(
        json.dumps({"schema": "passage-pinned-hub-snapshot-v1", "revision": "1" * 40, "files": []})
    )
    roster = W.green_world().world.source  # a synthetic roster of a split named test (metadata only)
    roster_path = tmp_path / "test-roster.json"
    roster_path.write_text(json.dumps(roster))
    touched = []
    prefix = str(test_dir.resolve())

    def recorder(function):
        def wrapper(path, *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                text = os.fsdecode(path)
                if os.path.abspath(text).startswith(prefix):
                    touched.append((function.__name__, text))
            return function(path, *args, **kwargs)

        return wrapper

    for name in ("open", "stat", "lstat", "scandir", "listdir"):
        monkeypatch.setattr(os, name, recorder(getattr(os, name)))
    monkeypatch.setattr(builtins, "open", recorder(builtins.open))
    monkeypatch.setattr(io, "open", recorder(io.open))
    test_dir.chmod(0)
    yield root, roster, roster_path, touched
    test_dir.chmod(0o700)


def red_record(tmp_path) -> Path:
    from passage_eval.protocol.admission import template_record

    path = tmp_path / "red-record.json"
    path.write_text(json.dumps(template_record()))
    return path


def refused(touched, call, match="refuses the test split|not all GREEN"):
    with pytest.raises(TestSplitRefused, match=match):
        call()
    assert touched == []


def test_the_public_roster_builder(canary, tmp_path):
    root, _, _, touched = canary
    refused(touched, lambda: BUILD.build_roster(root, benchmark_id="b", resolution=3, n_expected=2))
    refused(touched, lambda: BUILD.build_roster(root, benchmark_id="b", resolution=3, n_expected=2, allow_test=True))
    args = ["--dataset-root", str(root), "--out-dir", str(tmp_path / "o"), "--resolution", "3", "--n", "2"]
    refused(touched, lambda: BUILD.main(args))
    refused(
        touched, lambda: BUILD.main([*args, "--allow-test-split", "--test-admission-record", str(red_record(tmp_path))])
    )
    assert not (tmp_path / "o").exists()


def test_the_reference_admission(canary, tmp_path):
    from passage_eval.protocol import reference_admission as A

    root, roster, roster_path, touched = canary
    refused(touched, lambda: A.admission_policy(roster, reference_policy="released_labels", reference_max_expansions=9))
    refused(
        touched, lambda: A.build_reference_inventory(roster, lambda payload: A.load_admission_scenes(payload, root), {})
    )
    args = ["--roster", str(roster_path), "--roster-sha256", roster["roster_sha256"], "--dataset-root", str(root)]
    args += [
        "--reference-policy",
        "released_labels",
        "--reference-max-expansions",
        "9",
        "--output-dir",
        str(tmp_path / "o"),
    ]
    refused(touched, lambda: A.main(args))
    refused(
        touched, lambda: A.main([*args, "--allow-test-split", "--test-admission-record", str(red_record(tmp_path))])
    )
    assert not (tmp_path / "o").exists()


def test_score_roster(canary, tmp_path):
    from passage_eval.supervision.__main__ import main
    from passage_eval.supervision.roster_scoring import score_roster_run

    root, roster, roster_path, touched = canary
    run = tmp_path / "run"
    run.mkdir()
    refused(touched, lambda: score_roster_run(run, roster_path, roster["roster_sha256"], root, tmp_path / "o"))
    args = ["score-roster", "--run", str(run), "--roster", str(roster_path), "--roster-sha256", roster["roster_sha256"]]
    args += ["--dataset-root", str(root), "--output", str(tmp_path / "o")]
    refused(touched, lambda: main(args))
    refused(touched, lambda: main([*args, "--allow-test-split", "--test-admission-record", str(red_record(tmp_path))]))
    assert not (tmp_path / "o").exists()


def test_the_supervisor_for_a_spec_run_a_roster_run_and_a_warmup_source(canary, tmp_path):
    from passage_eval.supervision.__main__ import main

    root, roster, roster_path, touched = canary
    spec = tmp_path / "spec.json"
    payload = {"method_name": "m", "family": "f", "architecture": "a", "inference": "x:Y", "split": "test"}
    payload.update(resolution=3, dataset_path=str(root), measure_latency=False)
    spec.write_text(json.dumps(payload))
    base = ["run", "--spec", str(spec), "--output", str(tmp_path / "run"), "--worker-python", sys.executable]
    base += ["--isolation", "process", "--candidate-path", str(tmp_path)]
    refused(touched, lambda: main(base))  # a spec run on test is never admitted
    with_roster = [*base, "--roster", str(roster_path), "--roster-sha256", roster["roster_sha256"], "--dataset-root"]
    with_roster.append(str(root))
    refused(touched, lambda: main(with_roster))
    record = str(red_record(tmp_path))
    refused(touched, lambda: main([*with_roster, "--allow-test-split", "--test-admission-record", record]))
    validation = dict(payload, split="validation", measure_latency=True, latency_warmup=1)
    spec.write_text(json.dumps(validation))
    refused(touched, lambda: main([*base, "--warmup-split", "test"]), match="warm-up refuses the test split")
    assert not (tmp_path / "run").exists()


def test_the_warmup_selection(canary):
    from passage_eval.protocol.warmup import select_warmup_scenes
    from passage_eval.supervision.supervisor import WarmupInput

    _, _, _, touched = canary
    refused(touched, lambda: select_warmup_scenes(["a", "b"], "salt", 1, split="test", evaluated_ids=[]))
    refused(touched, lambda: WarmupInput(split="test"))


def test_the_campaign_planner_and_partition(canary, tmp_path):
    from passage_eval.campaign import partition as P
    from passage_eval.campaign.plan import read_plan

    root, roster, roster_path, touched = canary
    campaign = W.campaign_world(tmp_path / "campaign")
    plan = dict(campaign.plan, split="test", dataset_root=str(root))
    W.write_plan(campaign, plan)
    refused(touched, lambda: read_plan(campaign.plan_path, campaign.sha), match="admission record")
    record = red_record(tmp_path)
    admission = {"record": str(record), "record_sha256": W._sha(record), "evidence_dir": str(tmp_path)}
    W.write_plan(campaign, dict(plan, admission=admission))
    refused(touched, lambda: read_plan(campaign.plan_path, campaign.sha))
    refused(touched, lambda: read_plan(campaign.plan_path, campaign.sha, allow_test=True))
    file_sha = W._sha(roster_path)
    refused(touched, lambda: P.partition(roster_path, file_sha, roster["roster_sha256"], tmp_path / "parts", 2))
    assert not (tmp_path / "parts").exists()


def test_the_canary_records_a_traversal_and_an_unguarded_listing_touches_it(canary):
    root, _, _, touched = canary
    prefix = str((root / "3x3" / "test").resolve())  # resolving stats the canary path: do it before listing
    touched.clear()
    list(root.rglob("*"))  # what an unguarded listing of the dataset root would do
    seen = list(touched)
    assert seen and all(path.startswith(prefix) for _, path in seen)


def test_a_preparation_grant_opens_nothing_outside_its_readers(canary, tmp_path):
    """A preparation grant opens only its named readers and only the sealed roster-S scope.

    The canary's deliberately fixture-sized roster is therefore refused before a census can inspect it.  Runs,
    scoring, warm-up sources and campaign tools also refuse the grant before touching the test directory.
    """
    from passage_eval.campaign import partition as P
    from passage_eval.protocol.block_census import block_census
    from passage_eval.supervision.roster_scoring import score_roster_run
    from passage_eval.supervision.supervisor import RosterInput, WarmupInput, _guard_evaluated_split

    root, roster, roster_path, touched = canary
    grant = W.preparation_grant(W.preparation_world(tmp_path / "preparation"))
    touched.clear()
    refused(touched, lambda: block_census(roster, allow_test=True, admission=grant), match="only roster S")
    run = tmp_path / "run"
    run.mkdir()
    match = "authorises only"
    sha = roster["roster_sha256"]
    refused(
        touched,
        lambda: score_roster_run(run, roster_path, sha, root, tmp_path / "o", allow_test=True, admission=grant),
        match=match,
    )
    pinned_roster = RosterInput(path=str(roster_path), sha256=sha, dataset_root=str(root))
    from passage_eval import cli

    spec = cli.BenchSpec(method_name="m", family="f", architecture="a", inference="x:Y", split="test", resolution=3)
    refused(touched, lambda: _guard_evaluated_split(spec, pinned_roster, True, grant), match=match)
    refused(touched, lambda: WarmupInput(split="test", allow_test=True, admission=grant), match=match)
    refused(
        touched,
        lambda: P.partition(
            roster_path, W._sha(roster_path), sha, tmp_path / "parts", 2, allow_test=True, admission=grant
        ),
        match=match,
    )
    assert not (tmp_path / "o").exists() and not (tmp_path / "parts").exists()
