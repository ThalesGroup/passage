"""The ``passage-native-chain-rehearsal/v1`` receipt builder: only a qualifying validation rehearsal gets a receipt.

The campaign runs with the fake sessions of ``test_campaign_runner.py`` (paused, resumed, one deliberate kill); the
aggregates are stand-ins that the builder must find equal to what it recomputes. The receipt's R7 and resume blocks
are then checked by gate G2 on the synthetic GREEN world. Synthetic validation fixtures only.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from passage_eval.campaign import runner as RU

HERE = Path(__file__).resolve().parent


def _load(name, filename):
    key = f"passage_eval_tests_{name}"
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, HERE / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


W = _load("campaign_world", "campaign_world.py")
R = _load("campaign_runner", "test_campaign_runner.py")


def stand_in(procedure, **changes):
    aggregate = {
        "procedure": procedure,
        "canonical_equivalence": {"equal": True},
        "per_seed": {
            "2026": {
                "latency": {
                    "n_warm_requests": 120,
                    "meets_min_warm_requests": True,
                    "warmup": {"declared": True, "performed": True},
                }
            }
        },
        "restarts": {"rewarms": 1},
        "receipts": [{"batch_index": 0, "unit": "000000", "completion_sha256": "a" * 64}],
    }
    for path, value in changes.items():
        target = aggregate
        *head, last = path.split(".")
        for key in head:
            target = target[key]
        target[last] = value
    return aggregate


@pytest.fixture
def rehearsal(tmp_path, monkeypatch):
    """A rehearsed campaign: paused after one batch, resumed with a deliberate kill of batch 1's seg session."""
    from passage_eval.campaign import aggregate as aggregate_module
    from passage_eval.protocol import admission

    monkeypatch.setattr(RU, "verify_scored", R.fake_receipts)
    monkeypatch.setattr(admission, "_uncommitted_harness_paths", lambda root=None: [])
    campaign = W.campaign_world(tmp_path / "campaign")
    fake = R.FakeSessions()

    def killing(argv, timeout, log):
        fake(argv, timeout, log)
        if timeout == 0.5:
            raise TimeoutError

    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, stop_after_batches=1)
    RU.run_campaign(campaign.plan_path, campaign.sha, command=killing, deliberate_kills=[(1, "seg", 0.5)])
    overrides = {}
    monkeypatch.setattr(
        aggregate_module,
        "aggregate_campaign",
        lambda plan_path, plan_sha256, *, procedure, allow_test=False: stand_in(
            procedure, **overrides.get(procedure, {})
        ),
    )
    evidence = tmp_path / "evidence"
    paths = {}
    for name in ("astar", "reg", "seg"):
        relative = f"rehearsal/aggregate-{name}.json"
        W.write_json(evidence / relative, stand_in(name))
        paths[name] = relative
    campaign.evidence, campaign.aggregates, campaign.overrides = evidence, paths, overrides
    return campaign


def build(campaign, **kwargs):
    from passage_eval.campaign.rehearsal import build_rehearsal_receipt

    values = dict(evidence_dir=campaign.evidence, aggregates=campaign.aggregates)
    values.update(kwargs)
    return build_rehearsal_receipt(campaign.plan_path, campaign.sha, **values)


def test_a_qualifying_rehearsal_gets_a_receipt_from_its_real_outputs(rehearsal):
    receipt = build(rehearsal)
    plan = rehearsal.plan
    assert receipt["schema"] == "passage-native-chain-rehearsal/v1" and receipt["plan_sha256"] == rehearsal.sha
    assert (receipt["split"], receipt["target"]) == ("validation", "host")
    assert receipt["harness"] == {key: plan["harness"][key] for key in ("commit", "source_sha256")}
    assert receipt["candidates"]["seg"]["checkpoint_sha256"] == plan["procedures"]["seg"]["checkpoint_sha256"]
    assert receipt["budget"] == plan["budget"] and receipt["timing_profile"] == plan["timing"]["profile"]
    assert receipt["warmup"] == {"performed": True, "roster_sha256": plan["warmup"]["roster"]["roster_sha256"], "k": 2}
    assert receipt["rewarm"] == {"exercised": True, "n_rewarm_events": 3}
    assert receipt["deliberate_kill"] == {
        "exercised": True,
        "n_deliberate_kills": 1,
        "n_planned_relaunches": 1,
        "killed_attempts_scored": 0,
        "units": ["000001"],
    }
    assert receipt["resume"] == {"exercised": True, "n_batches_resumed": 1, "n_unplanned_relaunches": 0}
    for name, relative in rehearsal.aggregates.items():
        digest = hashlib.sha256((rehearsal.evidence / relative).read_bytes()).hexdigest()
        assert receipt["procedures"][name] == {"aggregate": {"path": relative, "sha256": digest}}
    assert receipt["latency"]["rows"]["seg/2026"] == {"n_warm_requests": 120, "meets_min_warm_requests": True}


def test_gate_g2_accepts_the_r7_and_resume_blocks_of_a_built_receipt(rehearsal):
    """The builder writes exactly the blocks G2 reads; on the GREEN world they keep G2 GREEN."""
    from passage_eval.protocol import admission

    built = build(rehearsal)
    green = W.copy_green(rehearsal.root / "green")
    path = green.evidence / green.record["evidence"]["native_chain_rehearsal"]["path"]
    receipt = json.loads(path.read_text())
    receipt.update(resume=built["resume"], deliberate_kill=built["deliberate_kill"])
    record = copy.deepcopy(green.record)
    W.write_json(path, receipt)
    record["evidence"]["native_chain_rehearsal"] = W.ref(green.evidence, path)
    W.save_record(green, record)
    report = admission.check_admission(green.record_path, green.evidence)
    assert report.gate("G2").status == "GREEN", report.gate("G2").reasons


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.overrides.update(seg={"canonical_equivalence.equal": False}), "differs from the aggregate"),
        (lambda c: _rewrite(c, "seg", {"canonical_equivalence.equal": False}), "not equal to the canonical scorer"),
        (
            lambda c: _rewrite(c, "reg", {"per_seed.2026.latency.meets_min_warm_requests": False}),
            "fewer warm requests",
        ),
        (
            lambda c: _rewrite(c, "reg", {"per_seed.2026.latency.warmup": {"declared": True, "performed": False}}),
            "warm-up was not declared and performed",
        ),
        (lambda c: [_rewrite(c, n, {"restarts.rewarms": 0}) for n in ("astar", "reg", "seg")], "no re-warm"),
        (lambda c: c.aggregates.pop("astar"), "one aggregate per procedure"),
        (lambda c: c.aggregates.update(seg="../outside.json"), "unreadable under the evidence directory"),
    ],
)
def test_the_receipt_is_refused_unless_every_aggregate_property_holds(rehearsal, change, message):
    from passage_eval.campaign.rehearsal import RehearsalNotQualified

    change(rehearsal)
    with pytest.raises(RehearsalNotQualified, match=message):
        build(rehearsal)


def _rewrite(campaign, name, changes):
    aggregate = stand_in(name, **changes)
    campaign.overrides[name] = changes
    W.write_json(campaign.evidence / campaign.aggregates[name], aggregate)


def _log(campaign):
    return Path(campaign.plan["output_dir"]) / "relaunch-log.jsonl"


def test_the_receipt_needs_a_deliberate_kill_a_resume_and_no_unplanned_relaunch(rehearsal):
    from passage_eval.campaign.rehearsal import RehearsalNotQualified

    output = Path(rehearsal.plan["output_dir"])
    log = _log(rehearsal)
    original = log.read_text()
    entry = json.loads(original)
    log.write_text(json.dumps(dict(entry, classification="killed_by_window")) + "\n")
    with pytest.raises(RehearsalNotQualified, match="no deliberately killed batch.*1 unplanned relaunch"):
        build(rehearsal)
    log.write_text(json.dumps(dict(entry, decision="split")) + "\n")
    with pytest.raises(RehearsalNotQualified, match="no deliberately killed batch was relaunched"):
        build(rehearsal)
    log.write_text(original + json.dumps({"kind": "operator", "unit": "000002"}) + "\n")
    with pytest.raises(RehearsalNotQualified, match="1 unplanned relaunch"):
        build(rehearsal)
    log.write_text(original)
    invocations = output / "invocations.jsonl"
    rows = [json.loads(line) for line in invocations.read_text().splitlines()]
    invocations.write_text("".join(json.dumps(dict(row, outcome="complete")) + "\n" for row in rows))
    with pytest.raises(RehearsalNotQualified, match="no resume after a pause"):
        build(rehearsal)


def test_an_archived_attempt_that_is_scored_refuses_the_receipt(rehearsal):
    from passage_eval.campaign.rehearsal import RehearsalNotQualified

    entry = json.loads(_log(rehearsal).read_text())
    archived = Path(rehearsal.plan["output_dir"]) / entry["archived_to"] / "completion.json"
    archived.write_text("{}")
    digest = hashlib.sha256(archived.read_bytes()).hexdigest()
    _rewrite(rehearsal, "seg", {"receipts": [{"batch_index": 1, "unit": "000001", "completion_sha256": digest}]})
    with pytest.raises(RehearsalNotQualified, match="1 archived attempt"):
        build(rehearsal)


def test_the_receipt_needs_a_clean_harness_tree_and_the_two_candidates(rehearsal, monkeypatch, tmp_path):
    from passage_eval.campaign.rehearsal import RehearsalNotQualified
    from passage_eval.protocol import admission

    monkeypatch.setattr(admission, "_uncommitted_harness_paths", lambda root=None: ["src/passage-eval/cli.py"])
    with pytest.raises(RehearsalNotQualified, match="harness tree is not clean"):
        build(rehearsal)
    monkeypatch.setattr(admission, "_uncommitted_harness_paths", lambda root=None: None)
    with pytest.raises(RehearsalNotQualified, match="not a git checkout"):
        build(rehearsal)
    monkeypatch.setattr(admission, "_uncommitted_harness_paths", lambda root=None: [])
    astar = W.campaign_world(tmp_path / "astar-only", procedures=("astar",))
    astar.evidence, astar.aggregates = rehearsal.evidence, {"astar": rehearsal.aggregates["astar"]}
    with pytest.raises(RehearsalNotQualified, match=r"does not run \['seg', 'reg'\]"):
        build(astar)


def test_the_rehearsal_receipt_command_writes_a_new_pinned_file(rehearsal, capsys):
    from passage_eval.campaign.__main__ import main

    args = ["rehearsal-receipt", "--plan", str(rehearsal.plan_path), "--plan-sha256", rehearsal.sha]
    args += ["--evidence-dir", str(rehearsal.evidence), "--output", "rehearsal/native-chain-rehearsal.json"]
    for name, relative in rehearsal.aggregates.items():
        args += ["--aggregate", f"{name}={relative}"]
    assert main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    written = rehearsal.evidence / "rehearsal" / "native-chain-rehearsal.json"
    assert printed == {
        "path": "rehearsal/native-chain-rehearsal.json",
        "sha256": hashlib.sha256(written.read_bytes()).hexdigest(),
    }
    assert json.loads(written.read_text())["deliberate_kill"]["exercised"] is True
    with pytest.raises(FileExistsError):
        main(args)
    with pytest.raises(SystemExit, match="one PROCEDURE=RELATIVE_PATH"):
        main([*args[:-2], "--aggregate", "seg"])
