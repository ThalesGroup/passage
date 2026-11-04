"""Campaign plan and runner: pinned plan, strict schema, batch bound, paired order, receipts, resume, relaunch.

Ported from the external preparation runner and extended to the sealed plan. Synthetic validation worlds only
(``campaign_world.py``); a split named ``test`` is a synthetic fixture. The end-to-end test runs real supervised
workers under bwrap on the synthetic planner of ``test_supervision_hardening.py``.
"""

import copy
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from passage_eval.campaign import EdgeTransportNotQualified
from passage_eval.campaign import plan as PL
from passage_eval.campaign import runner as RU
from passage_eval.io import pinned
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent


def _world():
    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


W = _world()


@pytest.fixture
def campaign(tmp_path):
    return W.campaign_world(tmp_path / "campaign")


def rewrite(campaign, change):
    plan = copy.deepcopy(campaign.plan)
    change(plan)
    W.write_plan(campaign, plan)
    return campaign.plan_path, campaign.sha


# ------------------------------------------------------------------------------------------------------- the plan


def test_a_plan_is_read_only_with_its_exact_pin(campaign):
    checked = PL.read_plan(campaign.plan_path, campaign.sha)
    assert checked.procedures == ("astar", "reg", "seg") and checked.admission is None
    assert len(checked.partition["batches"]) == 3 and checked.partition["n_expected"] == 5
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        PL.read_plan(campaign.plan_path, "0" * 64)


@pytest.mark.parametrize("seed", [1234, 3407])
def test_validation_plan_accepts_only_the_pinned_replica_checkpoint(campaign, seed):
    from passage_eval.protocol import replicas

    path = Path(__file__).resolve().parents[2] / "docs/benchmark/replica-candidates-256.json"
    digest = pinned.sha256_bytes(path.read_bytes())
    catalog = replicas.load_replica_catalog(path, digest)
    plan = copy.deepcopy(campaign.plan)
    plan["replica_catalog"] = {"path": str(path), "sha256": digest}
    plan["seeds"] = [seed]
    for name, entry in plan["procedures"].items():
        spec_path = Path(entry["spec"]["path"])
        spec = pinned.read_json(spec_path)
        spec["seeds"] = [seed]
        entry["spec"]["sha256"] = W.write_json(spec_path, spec)
        if name in ("seg", "reg"):
            entry["checkpoint_sha256"] = replicas.replica_identity(catalog, name, seed)["checkpoint_sha256"]
    W.write_plan(campaign, plan)
    assert PL.read_plan(campaign.plan_path, campaign.sha).plan["seeds"] == [seed]
    plan["procedures"]["seg"]["checkpoint_sha256"] = "0" * 64
    W.write_plan(campaign, plan)
    with pytest.raises(PL.PlanError, match="checkpoint differs"):
        PL.read_plan(campaign.plan_path, campaign.sha)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p.update(extra=1), "unknown keys"),
        (lambda p: p.pop("seeds"), "missing keys"),
        (lambda p: p["budget"].update(max_steps=3), "budget: unknown keys"),
        (lambda p: p["timing"]["rlimits"].pop("cpu_s"), "missing keys"),
        (lambda p: p["procedures"]["seg"].update(notes="x"), "unknown keys"),
        (lambda p: p.update(schema="passage-validation-batched-plan/v1"), "schema"),
        (lambda p: p.update(purpose="leaderboard"), "purpose"),
        (lambda p: p["timing"].update(profile="supervised_service_v1"), "re-warm timing profile"),
        (lambda p: p["timing"].update(boundary_id="passage-timing-boundary/v4-supervised-ipc-service"), "boundary"),
        (lambda p: p["budget"].update(wall_time_ms=0), "invalid budget"),
        (lambda p: p["order"].update(granularity="per_scene"), "per_batch"),
        (lambda p: p["order"].update(procedures=["seg"]), "every procedure"),
        (lambda p: p["aggregation"].update(censoring_rule_id="other/v1"), "censoring rule"),
        (lambda p: p["reference"].update(max_expansions=65536), "admission chain"),
        (lambda p: p["rosters"].update(source_roster_sha256="0" * 64), "another source"),
        (lambda p: p["warmup"].update(k=3), "warm-up k"),
        (lambda p: p["warmup"].update(salt="other"), "warm-up salt"),
        (lambda p: p["procedures"]["seg"].update(checkpoint_sha256="0" * 64), "final-candidates"),
        (lambda p: p["procedures"]["reg"].update(decoder="mypkg:Faster"), "frozen decoder"),
        (lambda p: p["procedures"]["seg"].update(decoder_allow_sha256=[]), "at least one pin"),
        (lambda p: p["procedures"]["reg"].update(split_procedure_allow_sha256=[]), "at least one pin"),
        (lambda p: p["procedures"]["astar"].update(environment_allow_sha256=[]), "at least one pin"),
        (lambda p: p["procedures"]["seg"].update(decoder_kwargs={"w": 1}), "keyword arguments"),
        (lambda p: p["harness"].update(source_sha256="0" * 64), "running harness source"),
        (lambda p: p["harness"].update(commit="main"), "40-hex"),
        (lambda p: p.update(seeds=[2026, 1234]), "seeds"),
        (lambda p: p["partition"].update(batch_size=3), "batch size"),
        (lambda p: p.update(infrastructure_relaunch_policy=""), "relaunch policy"),
    ],
)
def test_every_sealed_field_is_checked_and_an_unknown_key_refused(campaign, change, message):
    path, sha = rewrite(campaign, change)
    with pytest.raises(ValueError, match=message):
        PL.read_plan(path, sha)


def test_dynamic_admission_replaces_the_worst_case_refusal(campaign):
    """R7: the worst case is reported only; the plan must hold the observed batch three times and engage a request."""
    checked = PL.read_plan(campaign.plan_path, campaign.sha)
    bound = PL.batch_bounds(campaign.plan, checked.specs)
    assert set(bound["per_procedure"]) == {"astar", "reg", "seg"}
    assert bound["worst_case_wall_ms"] == sum(b["worst_case_wall_ms"] for b in bound["per_procedure"].values())
    # no cold pass in a campaign session: the worst case counts the warm pass only
    assert all(b["plan"]["cold_passes"] == 0 for b in bound["per_procedure"].values())
    sizing = PL.admission_sizing(campaign.plan, checked.specs)
    assert sizing["margin_ok"] and sizing["sanity_ok"] and sizing["safety_margin"] == 3.0
    lease = 3 * sizing["expected_batch_ms"]
    assert lease < bound["worst_case_wall_ms"]  # a lease far below the worst case is admitted ...

    def window(value):
        return lambda p: [p["batching"].update(lease_or_timeout_ms=value), p["timing"].update(startup_timeout_s=1.0)]

    path, sha = rewrite(campaign, window(lease))
    sized = PL.admission_sizing(campaign.plan, PL.read_plan(path, sha).specs)
    assert sized["required_lease_ms"] == lease and sized["admissible"]
    path, sha = rewrite(campaign, window(lease - 1))
    with pytest.raises(PL.PlanError, match="does not hold 3 times the observed batch"):  # ... the margin is not
        PL.read_plan(path, sha)


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda p: p["batching"].update(safety_margin=2.99), "safety_margin must be at least 3"),
        (lambda p: p["batching"].update(safety_margin=True), "safety_margin must be at least 3"),
        (lambda p: p["batching"]["observed_request_ms"].pop("seg"), "observed warm request time for every"),
        (lambda p: p["batching"]["observed_request_ms"].update(reg=0), "observed warm request time for every"),
        (lambda p: p["batching"].update(persistence_ms=-1), "persistence_ms must be nonnegative"),
        (lambda p: p["batching"].pop("persistence_ms"), "missing keys"),
        (lambda p: p["timing"].update(max_restarts=1), "max_restarts must be at least the batch size"),
        (lambda p: p["timing"].update(startup_timeout_s=10000.0), "share of a fresh window cannot engage one request"),
    ],
)
def test_the_admission_window_rules_of_a_plan_are_enforced(campaign, change, message):
    path, sha = rewrite(campaign, change)
    with pytest.raises(PL.PlanError, match=message):
        PL.read_plan(path, sha)


def test_the_sanity_floor_counts_a_full_start_and_one_request_per_procedure_share(campaign):
    checked = PL.read_plan(campaign.plan_path, campaign.sha)
    sizing = PL.admission_sizing(campaign.plan, checked.specs)
    seg, astar = sizing["per_procedure"]["seg"], sizing["per_procedure"]["astar"]
    b, t = campaign.plan["batching"], campaign.plan["timing"]
    start = t["startup_timeout_s"] * 1000 + 15000.0
    reserve = 20000 + t["deadline_grace_ms"] + 15000.0 + b["persistence_ms"]
    assert seg["request_reserve_ms"] == reserve
    step = b["observed_request_ms"]["seg"] + b["per_request_overhead_ms"]
    assert seg["sanity_floor_ms"] == b["fixed_overhead_ms"] + b["per_seed_overhead_ms"] + 2 * start + 2 * step + reserve
    assert seg["sanity_floor_ms"] - astar["sanity_floor_ms"] == start  # split mode starts two roles
    assert seg["window_share_ms"] == b["lease_or_timeout_ms"] / 3


def test_a_campaign_plan_runs_one_seed(tmp_path):
    campaign = W.campaign_world(tmp_path / "astar", procedures=("astar",))
    spec = Path(campaign.plan["procedures"]["astar"]["spec"]["path"])
    payload = json.loads(spec.read_text())
    payload["seeds"] = [2026, 1234]
    spec.write_text(json.dumps(payload))
    path, sha = rewrite(
        campaign,
        lambda p: [p.update(seeds=[2026, 1234]), p["procedures"]["astar"]["spec"].update(sha256=W._sha(spec))],
    )
    with pytest.raises(PL.PlanError, match="one seed"):
        PL.read_plan(path, sha)


def test_a_changed_spec_roster_chain_or_partition_child_is_refused(campaign):
    spec = Path(campaign.plan["procedures"]["seg"]["spec"]["path"])
    original = spec.read_bytes()
    spec.write_bytes(original + b" ")
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        PL.read_plan(campaign.plan_path, campaign.sha)
    payload = json.loads(original)
    payload["latency_warmup"] = 0
    spec.write_text(json.dumps(payload))
    path, sha = rewrite(campaign, lambda p: p["procedures"]["seg"]["spec"].update(sha256=W._sha(spec)))
    with pytest.raises(PL.PlanError, match="latency_warmup"):  # updating the pin does not make it admissible
        PL.read_plan(path, sha)
    spec.write_bytes(original)
    path, sha = rewrite(campaign, lambda p: p["procedures"]["seg"]["spec"].update(sha256=W._sha(spec)))
    child = campaign.partition_dir / campaign.manifest["batches"][1]["file"]
    child.write_bytes(child.read_bytes() + b" ")
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        PL.read_plan(path, sha)


@pytest.mark.parametrize("field", ["dataset_root", "partition"])
def test_the_output_cannot_overlap_a_private_input_tree(campaign, field):
    root = campaign.plan["dataset_root"] if field == "dataset_root" else campaign.plan["partition"]["dir"]
    path, sha = rewrite(campaign, lambda p: p.update(output_dir=str(Path(root) / "out")))
    with pytest.raises(PL.PlanError, match="separate"):
        PL.read_plan(path, sha)


def test_a_path_component_named_test_is_refused_outside_an_admitted_test_plan(campaign, tmp_path):
    (tmp_path / "test").mkdir()
    path, sha = rewrite(campaign, lambda p: p.update(output_dir=str(tmp_path / "test" / "out")))
    with pytest.raises(PL.PlanError, match="named test"):
        PL.read_plan(path, sha)


def test_the_test_split_needs_allow_test_and_an_all_green_record_before_any_roster_is_read(
    campaign, tmp_path, monkeypatch
):
    from passage_eval.protocol.admission import template_record

    reads = []
    original = PL._read_roster_file
    monkeypatch.setattr(PL, "_read_roster_file", lambda *a, **k: reads.append(a[1]) or original(*a, **k))
    path, sha = rewrite(campaign, lambda p: p.update(split="test"))
    with pytest.raises(TestSplitRefused, match="needs an admission record"):
        PL.read_plan(path, sha)
    green = W.green_world()
    record = {"record": str(green.record_path), "record_sha256": W._sha(green.record_path)}
    record["evidence_dir"] = str(green.evidence)
    path, sha = rewrite(campaign, lambda p: p.update(split="test", admission=record))
    with pytest.raises(TestSplitRefused, match="--allow-test-split"):
        PL.read_plan(path, sha)
    template = tmp_path / "record.json"
    template.write_text(json.dumps(template_record()))
    red = {"record": str(template), "record_sha256": W._sha(template), "evidence_dir": str(tmp_path)}
    path, sha = rewrite(campaign, lambda p: p.update(split="test", admission=red))
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        PL.read_plan(path, sha, allow_test=True)
    path, sha = rewrite(campaign, lambda p: p.update(split="test", admission=dict(record, record_sha256="0" * 64)))
    with pytest.raises(TestSplitRefused, match="differs from its pin"):
        PL.read_plan(path, sha, allow_test=True)
    path, sha = rewrite(campaign, lambda p: p.update(split=" Test "))
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        PL.read_plan(path, sha, allow_test=True)
    assert reads == []  # every refusal came before any roster file was read
    # An all-GREEN record admits only its own sealed rosters and decisions: this validation world is not one of
    # them, so the plan is still refused after the grant.
    path, sha = rewrite(campaign, lambda p: p.update(split="test", admission=record))
    with pytest.raises(PL.PlanError, match="split differs"):
        PL.read_plan(path, sha, allow_test=True)


def test_an_edge_target_is_refused_as_not_qualified(campaign, monkeypatch):
    path, sha = rewrite(campaign, lambda p: p.update(target="thor"))
    with pytest.raises(EdgeTransportNotQualified, match="edge transport not qualified"):
        PL.read_plan(path, sha)
    green = W.green_world()
    evidence = {"record": str(green.record_path), "record_sha256": W._sha(green.record_path)}
    evidence["evidence_dir"] = str(green.evidence)
    path, sha = rewrite(campaign, lambda p: p.update(target="thor", edge_transport_evidence=evidence))
    with pytest.raises(EdgeTransportNotQualified, match="gate G3"):  # GREEN G3 for host only: thor not sealed
        PL.read_plan(path, sha)
    from passage_eval.protocol import admission

    real = admission.check_admission

    def thor_green(*args, **kwargs):  # even a record whose G3 were GREEN for thor ...
        report = real(*args, **kwargs)
        return admission.AdmissionReport(
            report.record_name,
            report.record_sha256,
            report.record_core_sha256,
            report.evidence_dir,
            report.split,
            report.gates,
            report.file_pins,
            dict(report.admitted, targets=["host", "thor"]),
        )

    monkeypatch.setattr(admission, "check_admission", thor_green)
    PL.read_plan(path, sha)
    assert RU.run_admitted_edge_relay(path, sha, native_relay=lambda checked: checked.sha256) == sha
    with pytest.raises(EdgeTransportNotQualified, match="no remote transport"):  # ... runs nothing on a target
        RU.run_campaign(path, sha, command=lambda *a: pytest.fail("a command was started"))
    assert not Path(campaign.plan["output_dir"]).exists()


def test_an_edge_test_plan_cannot_dispatch_a_relay_before_its_g3_check_or_any_roster_read(campaign, monkeypatch):
    """The regular test route stays behind ``read_plan``; the validation relay cannot weaken this canary."""
    reads, invoked = [], []
    original = PL._read_roster_file
    monkeypatch.setattr(PL, "_read_roster_file", lambda *a, **k: reads.append(a[1]) or original(*a, **k))
    green = W.green_world()
    admission = {
        "record": str(green.record_path),
        "record_sha256": W._sha(green.record_path),
        "evidence_dir": str(green.evidence),
    }
    path, sha = rewrite(
        campaign,
        lambda p: p.update(split="test", target="thor", admission=admission, edge_transport_evidence=admission),
    )

    with pytest.raises(EdgeTransportNotQualified, match="gate G3"):
        RU.run_admitted_edge_relay(path, sha, allow_test=True, native_relay=lambda _: invoked.append(True))
    assert reads == [] and invoked == []


# --------------------------------------------------------------------------------------------------- commands


def test_the_commands_carry_every_sealed_limit_pin_and_the_warmup_roster(campaign):
    checked = PL.read_plan(campaign.plan_path, campaign.sha)
    item = checked.partition["batches"][0]
    unit = RU.root_unit(checked, item)
    run, score = RU.build_commands(checked, unit.roster, Path("/x/batch/seg"), "seg", deadline_monotonic=1234.5)
    plan = campaign.plan

    def value(flag):
        return run[run.index(flag) + 1]

    assert run[:6] == [plan["harness"]["python"], "-I", "-B", "-m", "passage_eval.supervision", "run"]
    assert value("--timing-profile") == "supervised_service_rewarm_v1" and value("--isolation") == "bwrap"
    assert "--campaign-session" in run and value("--session-deadline-monotonic") == "1234.5"
    assert value("--session-persistence-ms") == "2000.0"
    assert value("--deadline-grace-ms") == "10.0" and value("--hard-timeout-ms") == "60000.0"
    assert value("--startup-timeout-s") == "120.0" and value("--max-restarts") == "2"
    assert value("--rlimit-nofile") == "1024" and value("--rlimit-fsize-mb") == "1024" and "--rlimit-as-mb" not in run
    assert value("--warmup-roster-sha256") == plan["warmup"]["roster"]["roster_sha256"]
    assert value("--roster-sha256") == item["roster_sha256"] and value("--purpose") == "diagnostic"
    assert value("--roster") == str(campaign.partition_dir / item["file"])
    assert value("--split-decoder") == W.SEG_DECODER and "--require-split-procedure-pin" in run
    assert value("--decoder-allow-sha256") == "b" * 64 and value("--split-procedure-allow-sha256") == "c" * 64
    assert value("--environment-allow-sha256") == "a" * 64
    assert "--allow-test-split" not in run and "--allow-test-split" not in score
    assert score[5] == "score-roster" and score[score.index("--roster-sha256") + 1] == item["roster_sha256"]
    run, _ = RU.build_commands(checked, unit.roster, Path("/x/batch/astar"), "astar")
    assert "--planner-allow-sha256" in run and "--split-decoder" not in run
    assert "--session-deadline-monotonic" not in run  # no window: every request is admitted


# ----------------------------------------------------------------------------------------- run, resume, relaunch


def fake_receipts(procedure_dir, expected):
    return [
        {
            "path": f"scored/seed-{seed}/score_receipt.json",
            "sha256": hashlib.sha256(f"{procedure_dir}{seed}{expected['sample_ids']}".encode()).hexdigest(),
            "seed": seed,
            "series": "warm",
            "procedure_id": "b" * 64,
            "executed_artifact_sha256": [],
        }
        for seed in expected["seeds"]
    ]


@pytest.fixture
def fake_scoring(monkeypatch):
    monkeypatch.setattr(RU, "verify_scored", fake_receipts)


class FakeSessions:
    """Stands in for the supervisor and the scorer: writes the session receipt that ``script`` decides.

    ``script(procedure, unit, window, ids, attempt)`` returns the number of measured tasks, ``"invalid"``, or raises
    (``TimeoutError`` for a killed session, ``RU.TransportLost``, ``KeyboardInterrupt``, ``RuntimeError``).
    """

    def __init__(self, script=None):
        self.script = script or (lambda *a: None)
        self.calls = []  # (kind, procedure, unit, window, ids, timeout, deadline)

    def __call__(self, argv, timeout, log):
        log.write_text("")
        procedure_dir = log.parent
        procedure, window = procedure_dir.name, int(procedure_dir.parent.name.split("-")[1])
        attempt_dir = procedure_dir.parent.parent
        unit = attempt_dir.name.removeprefix("batch-")
        attempt = json.loads((attempt_dir / "intent.json").read_text())["attempt"]
        roster = json.loads(Path(argv[argv.index("--roster") + 1]).read_text())
        ids = [task["sample_id"] for task in roster["tasks"]]
        kind = argv[5]
        deadline = float(argv[argv.index("--session-deadline-monotonic") + 1]) if kind == "run" else None
        self.calls.append((kind, procedure, unit, window, ids, timeout, deadline))
        if kind != "run":
            return
        outcome = self.script(procedure, unit, window, ids, attempt)
        n = len(ids) if outcome is None else outcome
        run_dir = procedure_dir / "run"
        run_dir.mkdir()
        if n == "invalid":
            receipt = {"status": "invalid", "executed_sample_ids": [], "deferred_sample_ids": [], "run_artifact": None}
            receipt["invalid"] = {"reason": "initial_warmup_failed", "detail": "synthetic"}
        else:
            receipt = {
                "status": "complete" if n == len(ids) else "closed_by_admission",
                "executed_sample_ids": ids[:n],
                "deferred_sample_ids": ids[n:],
                "run_artifact": "seed-2026" if n else None,
                "closure": None if n == len(ids) else {"point": "before_measured_request"},
                "cold_start": {"worker_load_ms": {"predictor": 10.0}, "first_warmup_request": {"latency_ms": 5.0}},
            }
        W.write_json(run_dir / "session.json", {"schema": "passage-supervised-session/v1", "seed": 2026, **receipt})

    def runs(self):
        return [call for call in self.calls if call[0] == "run"]


def _output(campaign):
    return Path(campaign.plan["output_dir"])


def _log(campaign):
    path = _output(campaign) / "relaunch-log.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_a_run_writes_receipts_in_paired_order_and_a_restart_skips_every_completed_batch(campaign, fake_scoring):
    from passage_eval.protocol.warmup import paired_order

    fake = FakeSessions()
    result = RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    assert result["completed_batches"] == [0, 1, 2] and result["resumed_batches"] == [] and result["n_expected"] == 5
    assert result["verified"] is False and result["r7_relaunches"] == 0
    assert len(fake.calls) == 3 * 3 * 2  # run + score per procedure and batch
    output = _output(campaign)
    for index in range(3):
        batch = output / f"batch-{index:06d}"
        intent = json.loads((batch / "intent.json").read_text())
        completion = json.loads((batch / "completion.json").read_text())
        order = paired_order(2026, index, ["astar", "reg", "seg"])
        assert intent["paired_order"] == completion["paired_order"] == order and intent["attempt"] == 1
        assert completion["split"] == "validation" and completion["plan_sha256"] == campaign.sha
        assert completion["effect_key"] == intent["effect_key"] and len(intent["effect_key"]) == 32
        (window,) = completion["windows"]
        assert [row["procedure"] for row in window["sessions"]] == order
        runs = [call[1] for call in fake.runs() if call[2] == f"{index:06d}"]
        assert runs == order  # procedures run in the batch's paired order
    again = RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("a batch was rerun"))
    assert again["resumed_batches"] == [0, 1, 2] and again["completed_batches"] == []
    lines = (output / "invocations.jsonl").read_text().splitlines()
    assert [json.loads(line)["outcome"] for line in lines] == ["complete", "complete"]


def test_a_closed_session_carries_its_deferred_tasks_in_order_into_the_next_window(campaign, fake_scoring):
    """Dynamic admission: seg measures one of two tasks; the other runs first in its next window, alone."""
    fake = FakeSessions(
        lambda procedure, unit, window, ids, attempt: 1 if (procedure, unit, window) == ("seg", "000000", 1) else None
    )
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    output = _output(campaign)
    completion = json.loads((output / "batch-000000" / "completion.json").read_text())
    first, second = completion["windows"]
    seg = next(row for row in first["sessions"] if row["procedure"] == "seg")
    batch = [task["sample_id"] for task in campaign.world.admissible["tasks"][:2]]
    assert seg["status"] == "closed_by_admission" and seg["executed_sample_ids"] == batch[:1]
    assert seg["deferred_sample_ids"] == batch[1:]
    (carried,) = second["sessions"]
    assert carried["procedure"] == "seg" and carried["executed_sample_ids"] == batch[1:]
    roster_path = output / carried["roster"]["path"]
    continuation = json.loads(roster_path.read_text())
    assert [t["sample_id"] for t in continuation["tasks"]] == batch[1:]
    assert continuation["tasks"][0]["parent_rank"] == 1  # derived from the admissible roster, rank kept
    derivation = continuation["selection"]["derivation"]
    assert derivation["kind"] == "campaign_continuation" and derivation["window"] == 2
    assert derivation["parent_roster_sha256"] == campaign.world.admissible["roster_sha256"]
    assert continuation["roster_sha256"] == carried["roster"]["roster_sha256"]
    window_two = [call for call in fake.runs() if call[2] == "000000" and call[3] == 2]
    assert [(call[1], call[4]) for call in window_two] == [("seg", batch[1:])]
    # resume re-verifies every session, the continuation roster included
    RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("rerun"))
    roster_path.write_bytes(roster_path.read_bytes() + b" ")
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("rerun"))


def test_each_session_gets_an_equal_share_of_the_window_and_is_killed_at_the_window_end(
    campaign, fake_scoring, monkeypatch
):
    from types import SimpleNamespace

    now = [1000.0]
    monkeypatch.setattr(RU, "time", SimpleNamespace(monotonic=lambda: now[0]))
    lease = campaign.plan["batching"]["lease_or_timeout_ms"] / 1000.0
    fake = FakeSessions()

    def advancing(argv, timeout, log):
        fake(argv, timeout, log)
        now[0] += 100.0  # every command takes 100 s of the window

    RU.run_campaign(campaign.plan_path, campaign.sha, command=advancing)
    first = [call for call in fake.calls if call[2] == "000000"]
    runs = [call for call in first if call[0] == "run"]
    starts = [1000.0, 1100.0, 1200.0]
    end = 1000.0 + lease
    for (_, _, _, _, _, timeout, deadline), start, left in zip(runs, starts, (3, 2, 1)):
        assert timeout == pytest.approx(end - start)  # the kill bound is the end of the window
        assert deadline == pytest.approx(start + (end - start) / left)  # the admission deadline is an equal share
    scores = [call for call in first if call[0] == "score-roster"]
    assert all(call[5] == lease for call in scores)  # scoring runs after the window, with its own bound


def test_a_spent_window_leaves_the_remaining_procedures_to_the_next_window(campaign, fake_scoring, monkeypatch):
    from types import SimpleNamespace

    now = [0.0]
    monkeypatch.setattr(RU, "time", SimpleNamespace(monotonic=lambda: now[0]))
    lease = campaign.plan["batching"]["lease_or_timeout_ms"] / 1000.0
    fake = FakeSessions()

    def slow_first(argv, timeout, log):
        fake(argv, timeout, log)
        if argv[5] == "run" and len(fake.runs()) == 1:
            now[0] += lease  # the first session spends the whole window

    RU.run_campaign(campaign.plan_path, campaign.sha, command=slow_first)
    completion = json.loads((_output(campaign) / "batch-000000" / "completion.json").read_text())
    first, second = completion["windows"]
    assert len(first["sessions"]) == 1 and len(second["sessions"]) == 2
    assert all(
        row["roster"]["roster_sha256"] == campaign.manifest["batches"][0]["roster_sha256"] for row in second["sessions"]
    )
    assert not (_output(campaign) / "batch-000000" / "window-01" / second["sessions"][0]["procedure"]).exists()


def test_the_campaign_command_checks_a_plan_and_parses_run_options(campaign, capsys, monkeypatch):
    from passage_eval.campaign.__main__ import _kills, main

    assert main(["check-plan", "--plan", str(campaign.plan_path), "--plan-sha256", campaign.sha]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["dynamic_admission"]["admissible"] is True and summary["dynamic_admission"]["safety_margin"] == 3.0
    assert summary["worst_case_batch_wall_ms_information_only"] > 0
    assert summary["lot_memory"] == {**summary["lot_memory"], "resolution": 3, "batch_size": 2, "per_task_bytes": 189}
    assert _kills(["1:seg:2.5", "0:reg:1"]) == [(1, "seg", 2.5), (0, "reg", 1.0)]
    with pytest.raises(SystemExit, match="BATCH:PROCEDURE:SECONDS"):
        _kills(["seg:2"])
    seen = {}
    monkeypatch.setattr(RU, "run_campaign", lambda *a, **k: seen.update(k) or {"outcome": "paused"})
    args = ["run", "--plan", str(campaign.plan_path), "--plan-sha256", campaign.sha, "--stop-after-batches", "1"]
    assert main([*args, "--deliberate-kill", "1:seg:0.5"]) == 0
    assert seen["deliberate_kills"] == [(1, "seg", 0.5)] and seen["stop_after_batches"] == 1


def test_a_window_that_measures_no_task_stops_the_campaign(campaign, fake_scoring):
    fake = FakeSessions(lambda *a: 0)
    with pytest.raises(RU.CampaignStopped, match="measured no task"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    failure = json.loads((_output(campaign) / "batch-000000" / "failure.json").read_text())
    assert failure["classification"] == "harness_error"


@pytest.fixture
def lenient(monkeypatch):
    """Lift the 5% stop rule so that one small synthetic campaign can exercise relaunches, splits and failures."""
    original = RU._r7_policy
    monkeypatch.setattr(
        RU,
        "_r7_policy",
        lambda campaign: {**original(campaign), "relaunch_stop_fraction": 1.0},
    )


def test_a_killed_session_is_archived_hashed_never_read_and_relaunched_with_a_new_effect_key(
    campaign, fake_scoring, lenient
):
    def script(procedure, unit, window, ids, attempt):
        if (procedure, unit, attempt) == ("seg", "000001", 1):
            (Path(campaign.plan["output_dir"]) / "batch-000001" / "window-01" / "seg" / "partial.json").write_text("{}")
            raise TimeoutError("killed at the window bound")

    fake = FakeSessions(script)
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    output = _output(campaign)
    (entry,) = _log(campaign)
    assert (entry["kind"], entry["unit"], entry["attempt"]) == ("r7", "000001", 1)
    assert (entry["classification"], entry["decision"], entry["procedure"]) == ("killed_by_window", "relaunch", "seg")
    assert entry["green"] is True and entry["drain_proof"]["ok"] and entry["lease"]["ok"]
    archived = output / entry["archived_to"]
    manifest = json.loads((archived / "attempt-archive.json").read_text())
    assert manifest["scored"] is False and entry["archive_sha256"] == pinned.canonical_sha256(manifest)
    for relative, digest in manifest["files"].items():
        assert hashlib.sha256((archived / relative).read_bytes()).hexdigest() == digest
    assert "window-01/seg/partial.json" in manifest["files"] and not (archived / "completion.json").exists()
    first = json.loads((archived / "intent.json").read_text())
    second = json.loads((output / "batch-000001" / "intent.json").read_text())
    completion = json.loads((output / "batch-000001" / "completion.json").read_text())
    assert second["attempt"] == completion["attempt"] == 2 and second["effect_key"] != first["effect_key"]
    # relaunched identically: same roster, same paired order, every procedure again
    reruns = [call for call in fake.runs() if call[2] == "000001"]
    assert [call[1] for call in reruns][-3:] == first["paired_order"] == second["paired_order"]
    assert second["child_roster_sha256"] == first["child_roster_sha256"]


def test_three_ambiguous_attempts_split_the_batch_deterministically_in_two(campaign, fake_scoring, lenient):
    killed = []

    def script(procedure, unit, window, ids, attempt):
        if procedure == "reg" and unit == "000000":
            killed.append(attempt)
            raise TimeoutError

    fake = FakeSessions(script)
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    output = _output(campaign)
    assert killed == [1, 2, 3]
    assert [row["decision"] for row in _log(campaign)] == ["relaunch", "relaunch", "split"]
    split = json.loads((output / "split-log.jsonl").read_text())
    assert split["unit"] == "000000" and split["children"] == ["000000.0", "000000.1"]
    ids = [task["sample_id"] for task in campaign.world.admissible["tasks"][:2]]
    for suffix, sample_id in enumerate(ids):
        half = f"000000.{suffix}"
        completion = json.loads((output / f"batch-{half}" / "completion.json").read_text())
        assert completion["unit"] == half and completion["parent_ranks"] == [suffix, suffix + 1]
        roster = json.loads((output / "splits" / f"batch-{half}.json").read_text())
        assert [t["sample_id"] for t in roster["tasks"]] == [sample_id] and roster["tasks"][0]["parent_rank"] == suffix
        assert roster["selection"]["derivation"]["kind"] == "campaign_r7_split"
    assert not (output / "batch-000000").exists()
    status = RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("rerun"))
    assert status["resumed_batches"] == [0, 1, 2] and "000000.1" in status["resumed_units"]


def test_a_single_task_that_kills_its_batch_twice_with_green_infrastructure_is_a_candidate_failure(
    tmp_path, fake_scoring, lenient
):
    campaign = W.campaign_world(tmp_path / "single", batch_size=1)
    sample_id = campaign.world.admissible["tasks"][0]["sample_id"]

    def script(procedure, unit, window, ids, attempt):
        if (procedure, unit) == ("seg", "000000"):
            raise TimeoutError

    fake = FakeSessions(script)
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    output = _output(campaign)
    assert [row["decision"] for row in _log(campaign)] == ["relaunch", "impute_candidate_failure"]
    imputed = json.loads((output / "imputed" / "batch-000000--seg.json").read_text())
    assert imputed["sample_id"] == sample_id and imputed["status"] == "runtime_error"
    assert imputed["charged_latency_ms"] == campaign.plan["budget"]["wall_time_ms"]
    assert [row["classification"] for row in imputed["attempts"]] == ["killed_by_window", "killed_by_window"]
    completion = json.loads((output / "batch-000000" / "completion.json").read_text())
    assert completion["attempt"] == 3 and sorted(completion["procedures"]) == ["astar", "reg"]
    assert completion["imputed_failures"] == {"seg": imputed["sha256"]}
    # the other procedures still measured the task: the candidate failure never leaves the denominator
    assert sorted(row["procedure"] for row in completion["windows"][0]["sessions"]) == ["astar", "reg"]
    imputed_path = output / "imputed" / "batch-000000--seg.json"
    imputed_path.write_text(imputed_path.read_text().replace("runtime_error", "ok"))
    with pytest.raises(ValueError, match="imputed failure record changed"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("rerun"))


@pytest.mark.parametrize("cause", ["transport", "lease"])
def test_a_transport_or_lease_loss_is_never_imputed_to_the_candidate(tmp_path, fake_scoring, lenient, cause):
    campaign = W.campaign_world(tmp_path / "single", batch_size=1)
    infrastructure = RU.HostInfrastructure()
    # killed attempt 1 (lease checked at its window and after the kill), then attempt 2 finds its lease lost
    leases = iter([True, True, False, False] + [True] * 100)

    class Lossy(RU.HostInfrastructure):
        def verify_lease(self):
            return dict(infrastructure.verify_lease(), ok=next(leases) if cause == "lease" else True)

    def script(procedure, unit, window, ids, attempt):
        if unit == "000000" and procedure == "seg" and cause == "transport":
            raise RU.TransportLost("the relay lost its connection")
        if unit == "000000" and procedure == "seg" and cause == "lease" and attempt == 1:
            raise TimeoutError

    fake = FakeSessions(script)
    with pytest.raises(RU.CampaignStopped):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, infrastructure=Lossy())
    log = _log(campaign)
    assert not (_output(campaign) / "imputed").exists()
    if cause == "transport":
        assert [row["classification"] for row in log] == ["transport_or_lease_loss"] * 3
        assert [row["green"] for row in log] == [False] * 3
        assert log[-1]["decision"] == "stop_single_task_not_attributable"
        # a resume does not lift this stop; an operator relaunch does
        with pytest.raises(RU.CampaignStopped, match="operator relaunch-batch"):
            RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
        entry = RU.relaunch_batch(
            campaign.plan_path, campaign.sha, 0, operator="op", reason="relay fixed", reason_class="infrastructure"
        )
        assert entry["kind"] == "operator" and entry["archived_to"] is None
        RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
        assert (_output(campaign) / "batch-000000" / "completion.json").exists()
    else:
        assert [row["decision"] for row in log] == ["relaunch", "stop_lease_not_verified"]
        assert log[-1]["classification"] == "transport_or_lease_loss" and log[-1]["green"] is False


def test_an_invalid_session_is_ambiguous_and_never_charged_to_a_task(tmp_path, fake_scoring, lenient):
    """A candidate that cannot start follows the R7 rule: relaunched, then a candidate failure, never excluded."""
    campaign = W.campaign_world(tmp_path / "single", batch_size=1)
    fake = FakeSessions(lambda procedure, unit, *a: "invalid" if (procedure, unit) == ("reg", "000002") else None)
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    log = _log(campaign)
    assert [(row["classification"], row["decision"]) for row in log] == [
        ("invalid_session", "relaunch"),
        ("invalid_session", "impute_candidate_failure"),
    ]
    assert json.loads(log[0]["detail"]) == {"reason": "initial_warmup_failed", "detail": "synthetic"}
    assert (_output(campaign) / "imputed" / "batch-000002--reg.json").exists()


def test_more_than_five_percent_of_relaunched_batches_stops_the_campaign(campaign, fake_scoring):
    fake = FakeSessions(
        lambda procedure, unit, window, ids, attempt: (
            (_ for _ in ()).throw(TimeoutError()) if unit == "000001" else None
        )
    )
    with pytest.raises(RU.CampaignStopped, match="stop_relaunch_fraction"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    assert [row["decision"] for row in _log(campaign)] == ["stop_relaunch_fraction"]
    with pytest.raises(RU.CampaignStopped, match="1 of 3 batches were relaunched"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())


def test_a_drain_that_is_not_proven_stops_before_anything_is_archived(campaign, fake_scoring, lenient):
    class Undrained(RU.HostInfrastructure):
        def drain_proof(self, attempt_dir):
            return {"ok": False, "kind": "test", "attempt": attempt_dir.name}

    fake = FakeSessions(
        lambda procedure, unit, window, ids, attempt: (
            (_ for _ in ()).throw(TimeoutError()) if (unit, attempt) == ("000000", 1) else None
        )
    )
    with pytest.raises(RU.CampaignStopped, match="drain of the ambiguous attempt is not proven"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, infrastructure=Undrained())
    output = _output(campaign)
    assert (output / "batch-000000" / "failure.json").exists() and not (output / "relaunched").exists()
    assert _log(campaign) == []
    RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)  # drained now: the same attempt is decided again
    (entry,) = _log(campaign)
    assert (entry["classification"], entry["decision"], entry["attempt"]) == ("killed_by_window", "relaunch", 1)


def test_the_host_drain_proof_finds_a_process_that_still_names_the_attempt(tmp_path):
    attempt = tmp_path / "batch-000000"
    attempt.mkdir()
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", str(attempt / "window-01")])
    try:
        proof = RU.HostInfrastructure().drain_proof(attempt)
        assert proof["ok"] is False and proof["processes_naming_the_attempt"] == 1
    finally:
        process.kill()
        process.wait()
    assert RU.HostInfrastructure().drain_proof(attempt)["ok"] is True


def test_a_deliberate_kill_exercises_r7_apart_from_the_relaunch_fraction(campaign, fake_scoring, monkeypatch):
    def killing(argv, timeout, log):
        fake(argv, timeout, log)
        if timeout == 0.5:  # the deliberate kill's bound
            raise TimeoutError

    fake = FakeSessions()
    status = RU.run_campaign(campaign.plan_path, campaign.sha, command=killing, deliberate_kills=[(1, "seg", 0.5)])
    assert status["deliberate_kills"] == 1 and status["r7_relaunches"] == 0  # the 5% rule is not triggered
    (entry,) = _log(campaign)
    assert (entry["classification"], entry["decision"], entry["unit"]) == ("deliberate_kill", "relaunch", "000001")
    intent = json.loads((_output(campaign) / entry["archived_to"] / "intent.json").read_text())
    assert intent["deliberate_kill"] == {"seg": 0.5}
    assert json.loads((_output(campaign) / "batch-000001" / "intent.json").read_text())["deliberate_kill"] is None
    for kills, message in (([(9, "seg", 1.0)], "names no batch"), ([(0, "seg", 0)], "positive delay")):
        with pytest.raises(ValueError, match=message):
            RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, deliberate_kills=kills)


def test_a_deliberate_kill_is_refused_on_an_admitted_test_plan(admitted):
    campaign, _ = admitted
    with pytest.raises(ValueError, match="validation rehearsal only"):
        RU.run_campaign(campaign.plan_path, campaign.sha, allow_test=True, deliberate_kills=[(0, "seg", 1.0)])


def test_stop_after_batches_pauses_at_a_batch_boundary_and_resume_skips_completed_batches(campaign, fake_scoring):
    fake = FakeSessions()
    paused = RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, stop_after_batches=1)
    assert paused["outcome"] == "paused" and paused["completed_batches"] == [0]
    resumed = RU.run_campaign(campaign.plan_path, campaign.sha, command=fake)
    assert resumed["resumed_batches"] == [0] and resumed["completed_batches"] == [1, 2]
    assert _log(campaign) == []
    with pytest.raises(ValueError, match="positive integer"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=fake, stop_after_batches=0)


def test_an_interrupted_runner_leaves_an_ambiguous_attempt_that_resume_relaunches_under_r7(
    campaign, fake_scoring, lenient
):
    calls = []

    def interrupted(argv, timeout, log):
        calls.append(log)
        if len(calls) == 8:  # inside the second batch
            raise KeyboardInterrupt
        FakeSessions()(argv, timeout, log)

    with pytest.raises(KeyboardInterrupt):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=interrupted)
    output = _output(campaign)
    failure = json.loads((output / "batch-000001" / "failure.json").read_text())
    assert failure["classification"] == "interrupted"
    result = RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
    assert result["resumed_batches"] == [0] and result["completed_batches"] == [1, 2]
    (entry,) = _log(campaign)
    assert (entry["classification"], entry["decision"], entry["unit"]) == ("interrupted", "relaunch", "000001")
    assert json.loads((output / "batch-000001" / "intent.json").read_text())["attempt"] == 2


@pytest.mark.parametrize("failure", [RuntimeError("synthetic failure"), ValueError("refused receipt")])
def test_a_harness_error_stops_and_only_relaunch_batch_retries_it(campaign, fake_scoring, failure):
    starts = []

    def failing(argv, timeout, log):
        starts.append(log)
        if len(starts) == 3:  # second procedure of the first batch
            raise failure
        FakeSessions()(argv, timeout, log)

    with pytest.raises(type(failure)):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=failing)
    output = _output(campaign)
    first = output / "batch-000000"
    record = json.loads((first / "failure.json").read_text())
    assert record["classification"] == "harness_error" and record["type"] == type(failure).__name__
    with pytest.raises(RU.IncompleteBatch, match="relaunch-batch"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=failing)
    assert len(starts) == 3 and _log(campaign) == []
    with pytest.raises(ValueError, match="infrastructure"):
        RU.relaunch_batch(campaign.plan_path, campaign.sha, 0, operator="op", reason="r", reason_class="candidate")
    with pytest.raises(ValueError, match="operator"):
        RU.relaunch_batch(campaign.plan_path, campaign.sha, 0, operator=" ", reason="r", reason_class="infrastructure")
    with pytest.raises(ValueError, match="never started"):
        RU.relaunch_batch(campaign.plan_path, campaign.sha, 2, operator="op", reason="r", reason_class="infrastructure")
    with pytest.raises(ValueError, match="not a unit of batch"):
        RU.relaunch_batch(
            campaign.plan_path, campaign.sha, 0, operator="op", reason="r", reason_class="infrastructure", unit="000001"
        )
    entry = RU.relaunch_batch(
        campaign.plan_path, campaign.sha, 0, operator="operator-a", reason="host reboot", reason_class="infrastructure"
    )
    assert entry["attempt"] == 1 and entry["policy"] == campaign.plan["infrastructure_relaunch_policy"]
    assert entry["kind"] == "operator" and (output / entry["archived_to"] / "failure.json").exists()
    result = RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
    assert result["completed_batches"] == [0, 1, 2]
    assert json.loads((first / "intent.json").read_text())["attempt"] == 2
    with pytest.raises(ValueError, match="complete"):
        RU.relaunch_batch(campaign.plan_path, campaign.sha, 0, operator="op", reason="r", reason_class="infrastructure")


def test_a_second_runner_and_a_changed_completion_are_refused(campaign, fake_scoring):
    output = _output(campaign)
    output.mkdir(parents=True)
    with (output / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already running"):
            RU.run_campaign(campaign.plan_path, campaign.sha, command=lambda *a: pytest.fail("started"))
    RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
    completion = output / "batch-000000" / "completion.json"
    row = json.loads(completion.read_text())
    row["seeds"] = [3407]
    completion.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="changed or belongs to another plan"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
    row["seeds"] = [2026]
    row["windows"][0]["sessions"][0]["executed_sample_ids"] = []
    completion.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="scored session changed|every task once"):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())


def test_an_output_directory_of_another_plan_is_refused(campaign, fake_scoring):
    RU.run_campaign(campaign.plan_path, campaign.sha, command=FakeSessions())
    path, sha = rewrite(campaign, lambda p: p.update(campaign_id="another campaign"))
    with pytest.raises(ValueError, match="another campaign"):
        RU.run_campaign(path, sha, command=FakeSessions())


@pytest.mark.parametrize(
    "receipt, message",
    [
        (
            lambda ids: {"status": "complete", "executed_sample_ids": ["x"], "deferred_sample_ids": []},
            "measured prefix",
        ),
        (
            lambda ids: {"status": "closed_by_admission", "executed_sample_ids": [], "deferred_sample_ids": []},
            "measured prefix",
        ),
        (  # closed, yet nothing deferred
            lambda ids: {
                "status": "closed_by_admission",
                "executed_sample_ids": ids,
                "deferred_sample_ids": [],
                "run_artifact": "seed-2026",
            },
            "measured prefix",
        ),
        (  # complete, yet a task deferred
            lambda ids: {
                "status": "complete",
                "executed_sample_ids": ids[:1],
                "deferred_sample_ids": ids[1:],
                "run_artifact": "seed-2026",
            },
            "measured prefix",
        ),
        (  # a measured task but no run artifact
            lambda ids: {
                "status": "complete",
                "executed_sample_ids": ids,
                "deferred_sample_ids": [],
                "run_artifact": None,
            },
            "measured prefix",
        ),
        (lambda ids: {"status": "unknown"}, "invalid session receipt"),
    ],
)
def test_a_session_receipt_that_does_not_split_its_roster_is_a_harness_error(campaign, fake_scoring, receipt, message):
    def writing(argv, timeout, log):
        log.write_text("")
        if argv[5] == "run":
            ids = [t["sample_id"] for t in json.loads(Path(argv[argv.index("--roster") + 1]).read_text())["tasks"]]
            (log.parent / "run").mkdir()
            W.write_json(
                log.parent / "run" / "session.json",
                {"schema": "passage-supervised-session/v1", "seed": 2026, **receipt(ids)},
            )

    with pytest.raises(ValueError, match=message):
        RU.run_campaign(campaign.plan_path, campaign.sha, command=writing)
    assert json.loads((_output(campaign) / "batch-000000" / "failure.json").read_text())["classification"] == (
        "harness_error"
    )


# ------------------------------------------------------------------------------------------ bounded subprocesses


def test_a_real_subprocess_timeout_is_bounded_and_its_log_kept(tmp_path):
    log = tmp_path / "command.log"
    with pytest.raises(TimeoutError):
        RU.run_command([sys.executable, "-c", "import time; time.sleep(20)"], 1, log)
    assert log.exists()


def _assert_stopped(pid: int) -> None:
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            status = Path(f"/proc/{pid}/status").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return
        if "State:\tZ" in status:  # a reparented zombie does not execute
            return
        time.sleep(0.01)
    try:
        os.kill(pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    pytest.fail("child process survived group cleanup")


def test_a_timeout_stops_a_spawned_child_and_one_that_ignores_sigterm(tmp_path):
    marker = tmp_path / "child.pid"
    source = (
        "import pathlib,subprocess,sys,time; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)"
    )
    with pytest.raises(TimeoutError):
        RU.run_command([sys.executable, "-c", source, str(marker)], 1, tmp_path / "timeout.log")
    _assert_stopped(int(marker.read_text()))
    marker = tmp_path / "ignores.pid"
    child = (
        "import os,pathlib,signal,sys,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    leader = (
        "import subprocess,sys,time; subprocess.Popen([sys.executable,'-c',sys.argv[1],sys.argv[2]]); time.sleep(30)"
    )
    with pytest.raises(TimeoutError):
        RU.run_command([sys.executable, "-c", leader, child, str(marker)], 1, tmp_path / "ignore.log")
    _assert_stopped(int(marker.read_text()))


def test_a_successful_leader_with_a_live_child_is_refused(tmp_path):
    marker = tmp_path / "surviving.pid"
    leader = (
        "import pathlib,subprocess,sys; "
        "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
        "pathlib.Path(sys.argv[1]).write_text(str(p.pid))"
    )
    with pytest.raises(RuntimeError, match="child processes"):
        RU.run_command([sys.executable, "-c", leader, str(marker)], 5, tmp_path / "orphan.log")
    _assert_stopped(int(marker.read_text()))


def test_an_unexpected_wait_exception_cleans_the_process_group(tmp_path, monkeypatch):
    actual = subprocess.Popen
    created = []

    class FailingWait:
        def __init__(self, process):
            self.process, self.pid, self.calls = process, process.pid, 0

        def wait(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("injected wait failure")
            return self.process.wait(*args, **kwargs)

    def launch(*args, **kwargs):
        process = actual(*args, **kwargs)
        created.append(process)
        return FailingWait(process)

    monkeypatch.setattr(RU.subprocess, "Popen", launch)
    try:
        with pytest.raises(RuntimeError, match="injected wait failure"):
            RU.run_command([sys.executable, "-c", "import time; time.sleep(30)"], 5, tmp_path / "exception.log")
        assert created and created[0].poll() is not None
    finally:
        for process in created:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)


# ---------------------------------------------------------------------------------------- scored-artifact checks


def write_scored(campaign, monkeypatch, procedure="astar", executed=None):
    """The score receipt of one session of the first batch that ``verify_scored`` accepts (``executed``: the
    ``checkpoint_hashes`` of the run; by default the plan's bundle for Seg and Reg, none for A*)."""
    from passage_eval.supervision.session import SESSION_POLICY_ID

    checked = PL.read_plan(campaign.plan_path, campaign.sha)
    unit = RU.root_unit(checked, checked.partition["batches"][0])
    expected = RU._expected(checked, unit.roster, procedure, list(unit.sample_ids))
    procedure_dir = Path(campaign.root) / "batch" / procedure
    plan = campaign.plan
    t = plan["timing"]
    if executed is None:
        executed = {} if procedure == "astar" else {"bundle": plan["procedures"][procedure]["bundle_sha256"]}
    directory = procedure_dir / "scored" / "seed-2026"
    directory.mkdir(parents=True)
    manifest = {
        "seed": 2026,
        "series": "warm",
        "repeats": 1,
        "quality_repeat_index": 0,
        "procedure_id": "d" * 64,
        "dataset": {"split": "validation"},
        "bench_spec": expected["bench_specs"][2026],
        "supervision": {
            "roster": {"roster_sha256": unit.roster["roster_sha256"]},
            "purpose": {"name": "diagnostic"},
            "session": {"executed_sample_ids": expected["sample_ids"]},
        },
        "roster": expected["sample_ids"],
        "timing_profile": t["profile"],
        "timing_boundary": t["boundary_id"],
        "runtime_source_hash": plan["harness"]["source_sha256"],
        "checkpoint_hashes": dict(executed),
        "procedure": {
            "checkpoint_hashes": dict(executed),
            "supervision": {
                "session_policy": SESSION_POLICY_ID,
                "limits": {
                    "budget": plan["budget"],
                    "timing_profile": t["profile"],
                    "timing_boundary": t["boundary_id"],
                    "deadline_grace_ms": t["deadline_grace_ms"],
                    "hard_timeout_ms": t["hard_timeout_ms"],
                    "startup_timeout_s": t["startup_timeout_s"],
                    "max_restarts": t["max_restarts"],
                    "worker_rlimits": {
                        "core": 0,
                        "nofile": 1024,
                        "fsize_bytes": 1024 << 20,
                        "address_space_bytes": None,
                        "cpu_seconds": None,
                    },
                },
                "latency_protocol": {"warmup_source": expected["warmup_source"]},
            },
        },
    }
    W.write_json(directory / "run_manifest.json", manifest)
    for name in ("per_sample.npz", "metrics.json", "score_reference_inputs.json"):
        (directory / name).write_bytes(f"warm-{name}".encode())
    receipt = {
        "schema": "passage-offline-score-v1",
        "qualified_reference": True,
        "metric_contract": "passage-qualified-complete-v1",
        "scored_records": 2,
        "scored_repeats": 1,
        **{field: W._sha(directory / name) for field, name in RU.SCORE_FILES},
    }
    W.write_json(directory / "score_receipt.json", receipt)

    class Artifact:
        records = [{}, {}]

        def validate(self):
            pass

    import passage_eval.execution as execution

    monkeypatch.setattr(execution, "read_run_artifact", lambda _: Artifact())
    return procedure_dir, expected


@pytest.fixture
def scored(campaign, monkeypatch):
    return write_scored(campaign, monkeypatch)


def test_scored_receipts_are_bound_to_the_child_the_spec_the_limits_and_the_harness(scored):
    procedure_dir, expected = scored
    records = RU.verify_scored(procedure_dir, expected)
    assert [(r["seed"], r["series"]) for r in records] == [(2026, "warm")]  # warm requests only: no cold series
    assert all(r["executed_artifact_sha256"] == [] for r in records)


def _edit_manifest(procedure_dir, change, series="warm"):
    path = procedure_dir / "scored" / "seed-2026" / ("cold" if series == "cold" else "") / "run_manifest.json"
    path.parent.mkdir(exist_ok=True)
    manifest = json.loads(path.read_text())
    change(manifest)
    W.write_json(path, manifest)
    receipt_path = path.parent / "score_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["run_manifest_sha256"] = W._sha(path)
    W.write_json(receipt_path, receipt)


def _limits(m):
    return m["procedure"]["supervision"]["limits"]


MANIFEST_CASES = {
    "timing_profile": lambda m: m.update(timing_profile="supervised_service_v1"),
    "timing_boundary": lambda m: m.update(timing_boundary="passage-timing-boundary/v4-supervised-ipc-service"),
    "source_hash": lambda m: m.update(runtime_source_hash="0" * 64),
    "child_roster": lambda m: m["supervision"]["roster"].update(roster_sha256="e" * 64),
    "sample_ids": lambda m: m["roster"].reverse(),
    "split": lambda m: m["dataset"].update(split="test"),
    "warmup_source": lambda m: m["procedure"]["supervision"]["latency_protocol"].update(
        warmup_source="split:calibration"
    ),
    "bench_spec": lambda m: m["bench_spec"].update(latency_warmup=0),
    "purpose": lambda m: m["supervision"]["purpose"].update(name="verified-candidate-run"),
    "repeats": lambda m: m.update(repeats=2),
    "quality_repeat_index": lambda m: m.update(quality_repeat_index=1),
    "seed": lambda m: m.update(seed=1234),
    "limits_budget": lambda m: _limits(m)["budget"].update(wall_time_ms=1),
    "limits_profile": lambda m: _limits(m).update(timing_profile="supervised_service_v1"),
    "limits_boundary": lambda m: _limits(m).update(timing_boundary="other"),
    "limits_grace": lambda m: _limits(m).update(deadline_grace_ms=20.0),
    "limits_hard_timeout": lambda m: _limits(m).update(hard_timeout_ms=1.0),
    "limits_startup": lambda m: _limits(m).update(startup_timeout_s=1.0),
    "limits_restarts": lambda m: _limits(m).update(max_restarts=3),
    "limits_nofile": lambda m: _limits(m)["worker_rlimits"].update(nofile=64),
    "limits_fsize": lambda m: _limits(m)["worker_rlimits"].update(fsize_bytes=1),
    "limits_address_space": lambda m: _limits(m)["worker_rlimits"].update(address_space_bytes=1 << 30),
    "limits_cpu": lambda m: _limits(m)["worker_rlimits"].update(cpu_seconds=10),
    "limits_missing": lambda m: m["procedure"]["supervision"].pop("limits"),
    "series_cold": lambda m: m.update(series="cold"),
    "session_policy": lambda m: m["procedure"]["supervision"].pop("session_policy"),
    "session_prefix": lambda m: m["supervision"]["session"].update(executed_sample_ids=m["roster"][:1]),
    "session_missing": lambda m: m["supervision"].pop("session"),
}


@pytest.mark.parametrize("case", sorted(MANIFEST_CASES))
def test_a_scored_artifact_made_with_other_limits_or_bindings_is_refused(scored, case):
    procedure_dir, expected = scored
    _edit_manifest(procedure_dir, MANIFEST_CASES[case])
    with pytest.raises(ValueError, match="mismatch"):
        RU.verify_scored(procedure_dir, expected)


RECEIPT_CASES = {
    "schema": lambda r: r.update(schema="passage-offline-score-v0"),
    "qualified_reference": lambda r: r.update(qualified_reference=False),
    "metric_contract": lambda r: r.update(metric_contract="passage-partial-v1"),
    "scored_records": lambda r: r.update(scored_records=1),
    "scored_repeats": lambda r: r.update(scored_repeats=2),
}


@pytest.mark.parametrize("case", sorted(RECEIPT_CASES))
def test_a_score_receipt_without_the_qualified_canonical_contract_is_refused(scored, case):
    procedure_dir, expected = scored
    path = procedure_dir / "scored" / "seed-2026" / "score_receipt.json"
    receipt = json.loads(path.read_text())
    RECEIPT_CASES[case](receipt)
    W.write_json(path, receipt)
    with pytest.raises(ValueError, match="mismatch"):
        RU.verify_scored(procedure_dir, expected)


def test_a_duplicate_or_missing_series_and_a_missing_procedure_identity_are_refused(scored):
    procedure_dir, expected = scored
    warm = procedure_dir / "scored" / "seed-2026"
    shutil.copytree(warm, procedure_dir / "scored" / "seed-2026-copy")
    with pytest.raises(ValueError, match="expected 1 score receipts, found 2"):  # a second series of the seed
        RU.verify_scored(procedure_dir, expected)
    shutil.rmtree(procedure_dir / "scored" / "seed-2026-copy")
    _edit_manifest(procedure_dir, lambda m: m.update(procedure_id="d" * 63))
    with pytest.raises(ValueError, match="missing procedure identity"):
        RU.verify_scored(procedure_dir, expected)
    (warm / "score_receipt.json").unlink()
    with pytest.raises(ValueError, match="expected 1 score receipts, found 0"):
        RU.verify_scored(procedure_dir, expected)


@pytest.mark.parametrize("name", ["per_sample.npz", "metrics.json", "score_reference_inputs.json"])
def test_a_scored_file_changed_after_its_receipt_is_refused(scored, name):
    procedure_dir, expected = scored
    path = procedure_dir / "scored" / "seed-2026" / name
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="hash mismatch"):
        RU.verify_scored(procedure_dir, expected)


def test_the_executed_checkpoint_or_bundle_must_be_the_plans_frozen_one(campaign, monkeypatch):
    """The reviewer's case: a Seg plan declaring the frozen checkpoint while its candidate tree loads other weights."""
    plan = campaign.plan["procedures"]["seg"]
    procedure_dir, expected = write_scored(campaign, monkeypatch, "seg")
    assert expected["executed_artifacts"] == sorted({plan["checkpoint_sha256"], plan["bundle_sha256"]})
    records = RU.verify_scored(procedure_dir, expected)
    assert all(r["executed_artifact_sha256"] == [plan["bundle_sha256"]] for r in records)
    for executed, message in (
        ({"checkpoint": "9" * 64}, "are not the plan's frozen candidate artifacts"),
        ({"bundle": plan["bundle_sha256"], "checkpoint": "9" * 64}, "are not the plan's frozen candidate artifacts"),
        ({}, "are not the plan's frozen candidate artifacts"),
    ):
        shutil.rmtree(procedure_dir)
        procedure_dir, expected = write_scored(campaign, monkeypatch, "seg", executed=executed)
        with pytest.raises(ValueError, match=message):
            RU.verify_scored(procedure_dir, expected)
    shutil.rmtree(procedure_dir)
    procedure_dir, expected = write_scored(
        campaign, monkeypatch, "seg", executed={"checkpoint": plan["checkpoint_sha256"]}
    )
    assert RU.verify_scored(procedure_dir, expected)[0]["executed_artifact_sha256"] == [plan["checkpoint_sha256"]]
    _edit_manifest(procedure_dir, lambda m: m["procedure"].update(checkpoint_hashes={}))
    with pytest.raises(ValueError, match="no consistent checkpoint_hashes"):
        RU.verify_scored(procedure_dir, expected)
    shutil.rmtree(procedure_dir)
    procedure_dir, expected = write_scored(campaign, monkeypatch, "astar", executed={"checkpoint": "9" * 64})
    with pytest.raises(ValueError, match="are not the plan's frozen candidate artifacts"):  # a planner loads none
        RU.verify_scored(procedure_dir, expected)


def test_a_nonzero_exit_is_an_error_and_its_log_is_kept(tmp_path):
    log = tmp_path / "exit.log"
    with pytest.raises(RuntimeError, match="exited 3"):
        RU.run_command([sys.executable, "-c", "import sys; print('x'); sys.exit(3)"], 5, log)
    assert log.read_text().strip() == "x"


# ------------------------------------------------------------------------------------------ an admitted test plan


def test_an_admitted_test_plan_repeats_its_sealed_decisions_and_carries_the_admission_to_every_command(tmp_path):
    admitted = W.admitted_campaign(tmp_path / "admitted")
    checked = PL.read_plan(admitted.plan_path, admitted.sha, allow_test=True)
    assert checked.allow_test and checked.admission.admissible_roster_sha256 == checked.admissible["roster_sha256"]
    with pytest.raises(TestSplitRefused, match="--allow-test-split"):
        PL.read_plan(admitted.plan_path, admitted.sha)
    unit = RU.root_unit(checked, checked.partition["batches"][0])
    run, score = RU.build_commands(checked, unit.roster, tmp_path / "batch" / "seg", "seg")
    for argv in (run, score):
        assert argv[argv.index("--test-admission-record") + 1] == str(admitted.green.record_path)
        assert argv[argv.index("--test-admission-evidence-dir") + 1] == str(admitted.green.evidence)
        assert "--allow-test-split" in argv
    assert run[run.index("--purpose") + 1] == "verified-candidate-run"
    for change, field in (
        (lambda p: p["timing"].update(max_restarts=99), "max_restarts"),
        (lambda p: p.update(infrastructure_relaunch_policy="relaunch freely"), "infrastructure_relaunch_policy"),
        (lambda p: p.update(purpose="diagnostic"), "purpose"),
        (lambda p: p.update(protocol_record_sha256="0" * 64), "protocol_record_sha256"),
    ):
        plan = copy.deepcopy(admitted.plan)
        change(plan)
        sha = W.write_json(admitted.plan_path, plan)
        with pytest.raises(PL.PlanError, match=f"sealed admission record in .*{field}"):
            PL.read_plan(admitted.plan_path, sha, allow_test=True)


@pytest.fixture(scope="module")
def admitted(tmp_path_factory):
    """One admitted test plan over the shared GREEN world, checked once (tests below change copies only)."""
    campaign = W.admitted_campaign(tmp_path_factory.mktemp("admitted") / "plan")
    checked = PL.read_plan(campaign.plan_path, campaign.sha, allow_test=True)
    return campaign, checked


BINDING_CASES = {
    "harness": lambda p, c: p["harness"].update(commit="0" * 40),
    "admissible_roster": lambda p, c: p["rosters"]["admissible"].update(roster_sha256="0" * 64),
    "source_roster": lambda p, c: p["rosters"].update(source_roster_sha256="0" * 64),
    "chain": lambda p, c: c.update(chain_sha256="0" * 64),
    "budget": lambda p, c: p["budget"].update(max_expansions=65536),
    "timing_profile": lambda p, c: p["timing"].update(profile="supervised_service_v1"),
    "timing_boundary_id": lambda p, c: p["timing"].update(boundary_id="other"),
    "deadline_grace_ms": lambda p, c: p["timing"].update(deadline_grace_ms=20.0),
    "hard_timeout_ms": lambda p, c: p["timing"].update(hard_timeout_ms=1.0),
    "startup_timeout_s": lambda p, c: p["timing"].update(startup_timeout_s=1.0),
    "max_restarts": lambda p, c: p["timing"].update(max_restarts=99),
    "worker_rlimits": lambda p, c: p["timing"]["rlimits"].update(cpu_s=60),
    "reference": lambda p, c: p["reference"].update(max_expansions=65536),
    "warmup": lambda p, c: p["warmup"].update(salt="another"),
    "paired_order": lambda p, c: p["order"].update(schedule_seed=7),
    "batching": lambda p, c: p["batching"].update(fixed_overhead_ms=1),
    "seeds": lambda p, c: p.update(seeds=[1234]),
    "censoring_rule": lambda p, c: p["aggregation"].update(censoring_rule_id="other/v1"),
    "min_warm_requests": lambda p, c: p["aggregation"].update(min_warm_requests=99),
    "targets": lambda p, c: p.update(target="orin"),
    "infrastructure_relaunch_policy": lambda p, c: p.update(infrastructure_relaunch_policy="relaunch freely"),
    "r7_policy": lambda p, c: p["r7_policy"].update(max_ambiguous_attempts=1),
    "purpose": lambda p, c: p.update(purpose="diagnostic"),
    "protocol_record_sha256": lambda p, c: p.update(protocol_record_sha256="0" * 64),
    "candidates.seg.checkpoint": lambda p, c: p["procedures"]["seg"].update(checkpoint_sha256="0" * 64),
    "candidates.reg.predictor": lambda p, c: p["procedures"]["reg"].update(predictor="other.models:Predictor"),
    "candidates.seg.decoder": lambda p, c: p["procedures"]["seg"].update(
        decoder="passage_eval.baselines.final_operator_adapters:SegFinalDecoder"
    ),
    "candidates.reg.decoder_tree": lambda p, c: p["procedures"]["reg"].update(decoder_allow_sha256=["b" * 64]),
    "candidates.seg.bundle": lambda p, c: p["procedures"]["seg"].update(bundle_sha256="1" * 64),
    "astar.planner_tree": lambda p, c: p["procedures"]["astar"].update(planner_allow_sha256=["e" * 64]),
}


@pytest.mark.parametrize("field", sorted(BINDING_CASES))
def test_an_admitted_plan_is_bound_to_every_sealed_decision_and_candidate_identity(admitted, field):
    """Each binding of ``_check_against_admission``, refused alone on a plan that otherwise repeats the record."""
    _, checked = admitted
    PL._check_against_admission(copy.deepcopy(checked.plan), checked.admission, checked.chain, checked.specs)
    plan, chain = copy.deepcopy(checked.plan), copy.deepcopy(checked.chain)
    BINDING_CASES[field](plan, chain)
    with pytest.raises(PL.PlanError, match=rf"sealed admission record in \[.*'{field}'"):
        PL._check_against_admission(plan, checked.admission, chain, checked.specs)


@pytest.mark.parametrize(
    "change",
    [
        lambda b: b.update(safety_margin=4.0),
        lambda b: b["observed_request_ms"].update(seg=1.0),
        lambda b: b.update(persistence_ms=1.0),
    ],
)
def test_an_admitted_plan_repeats_the_sealed_admission_window(admitted, change):
    _, checked = admitted
    plan = copy.deepcopy(checked.plan)
    change(plan["batching"])
    with pytest.raises(PL.PlanError, match=r"sealed admission record in \['batching'\]"):
        PL._check_against_admission(plan, checked.admission, checked.chain, checked.specs)


def test_an_admitted_plan_runs_only_a_harness_planner_as_astar(admitted):
    _, checked = admitted
    specs = copy.deepcopy(checked.specs)
    specs["astar"]["inference"] = "hpkg.entry:Planner"
    with pytest.raises(PL.PlanError, match=r"'astar.planner'"):
        PL._check_against_admission(copy.deepcopy(checked.plan), checked.admission, checked.chain, specs)


def test_roster_s_admission_allows_only_edge_seg_reg_seed_2026(admitted, monkeypatch):
    """The narrow S authority cannot be widened by a plan's procedure list."""
    _, checked = admitted
    record = pinned.read_pinned_json(checked.admission.record_path, checked.admission.record_sha256)
    record["protocol"]["decisions"]["roster_selection"]["value"]["n"] = 1024
    # Unit scope probe: substitute the reader result, rather than mutating an authentic admitted artifact.
    monkeypatch.setattr(PL.pinned, "read_pinned_json", lambda path, digest: record)
    with pytest.raises(PL.PlanError, match="exactly the Seg and Reg"):
        PL._check_against_admission(copy.deepcopy(checked.plan), checked.admission, checked.chain, checked.specs)
    plan = copy.deepcopy(checked.plan)
    plan["procedures"].pop("astar")
    plan["order"]["procedures"] = ["seg", "reg"]
    plan["seeds"] = [1234]
    with pytest.raises(PL.PlanError, match="seed 2026 only"):
        PL._check_against_admission(plan, checked.admission, checked.chain, checked.specs)
    plan["seeds"] = [2026]
    with pytest.raises(PL.PlanError, match="Thor, Orin or Tiger"):
        PL._check_against_admission(plan, checked.admission, checked.chain, checked.specs)


def test_candidate_record_substitution_after_grant_check_is_refused(admitted, monkeypatch):
    _, checked = admitted
    grant = checked.admission
    record_path = Path(grant.record_path)
    original = record_path.read_bytes()
    substituted = pinned.strict_json_loads(original)
    substituted["candidates"]["seg"]["checkpoint_sha256"] = "0" * 64
    real_check = type(grant).ensure_unchanged

    def replace_after_check(self):
        real_check(self)
        record_path.write_bytes(pinned.canonical_json(substituted))

    monkeypatch.setattr(type(grant), "ensure_unchanged", replace_after_check)
    try:
        with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
            PL._check_procedures(checked.plan, checked.admissible, True, grant)
        with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
            PL._check_against_admission(checked.plan, grant, checked.chain, checked.specs)
    finally:
        record_path.write_bytes(original)


@pytest.mark.parametrize(
    "change, field",
    [
        (lambda p: p["procedures"]["seg"].update(predictor="some_other_package.models:AnotherPredictor"), "predictor"),
        (
            lambda p: p["procedures"]["seg"].update(
                decoder="passage_eval.baselines.final_operator_adapters:SegFinalDecoder"
            ),
            "decoder",
        ),
        (lambda p: p["procedures"]["seg"].update(bundle_sha256="e" * 64), "bundle"),
        (lambda p: p["procedures"]["seg"].update(decoder_allow_sha256=["9" * 64]), "decoder_tree"),
        (lambda p: p["procedures"]["reg"]["decoder_allow_sha256"].append("9" * 64), "decoder_tree"),
        (lambda p: p["procedures"]["astar"].update(planner_allow_sha256=["9" * 64]), "planner_tree"),
    ],
)
def test_read_plan_refuses_an_admitted_plan_with_other_candidate_identities(tmp_path, admitted, change, field):
    """The reviewer's probe: other predictor, decoder entry, bundle or tree pins under an all-GREEN grant."""
    campaign, _ = admitted
    plan = copy.deepcopy(campaign.plan)
    change(plan)
    path = tmp_path / "plan.json"
    sha = W.write_json(path, plan)
    with pytest.raises(PL.PlanError, match=rf"sealed admission record in .*{field}"):
        PL.read_plan(path, sha, allow_test=True)


def test_an_admitted_plan_names_the_bundle_of_its_own_target(admitted, tmp_path):
    _, checked = admitted
    record = json.loads(Path(checked.plan["admission"]["record"]).read_text())
    bindings = PL._candidate_bindings(checked.plan, record, checked.specs)
    assert bindings and all(bindings.values())
    plan = dict(checked.plan, target="thor")  # no thor bundle in the host-only record
    bindings = PL._candidate_bindings(plan, record, checked.specs)
    assert not bindings["candidates.seg.bundle"] and not bindings["candidates.reg.bundle"]


# ------------------------------------------------------------------------------------ control files of a plan


def test_split_procedures_run_only_the_frozen_training_seed(campaign):
    path, sha = rewrite(campaign, lambda p: p.update(seeds=[2026, 1234]))
    for name in ("astar", "seg", "reg"):  # every spec repeats the plan's seeds
        spec = Path(campaign.plan["procedures"][name]["spec"]["path"])
        payload = json.loads(spec.read_text())
        payload["seeds"] = [2026, 1234]
        spec.write_text(json.dumps(payload))
    path, sha = rewrite(
        campaign,
        lambda p: [
            p["procedures"][name]["spec"].update(sha256=W._sha(Path(p["procedures"][name]["spec"]["path"])))
            for name in ("astar", "seg", "reg")
        ],
    )
    with pytest.raises(PL.PlanError, match="freezes one checkpoint .training seed 2026.*need a frozen checkpoint"):
        PL.read_plan(path, sha)


def test_a_plan_reads_final_candidates_at_its_harness_commit(campaign, monkeypatch):
    from passage_eval.protocol import admission

    monkeypatch.setattr(admission, "_committed_file", lambda *a, **k: b"{}")
    with pytest.raises(PL.PlanError, match="differs from the file committed"):
        PL.read_plan(campaign.plan_path, campaign.sha)
    monkeypatch.setattr(admission, "_committed_file", lambda *a, **k: None)
    with pytest.raises(PL.PlanError, match="cannot be read at commit"):
        PL.read_plan(campaign.plan_path, campaign.sha)


def test_a_plan_pins_the_running_harness_commit(campaign, monkeypatch):
    from passage_eval import cli

    monkeypatch.setattr(cli, "_checkout_commit", lambda: "1" * 40)
    with pytest.raises(PL.PlanError, match="running harness commit differs"):
        PL.read_plan(campaign.plan_path, campaign.sha)


def _rewrite_chain(campaign, change):
    from passage_eval.io import pinned as P

    entry = campaign.plan["rosters"]["reference_admission_chain"]
    chain = json.loads(Path(entry["path"]).read_text())
    change(chain)
    chain["chain_sha256"] = P.canonical_sha256({k: v for k, v in chain.items() if k != "chain_sha256"})
    path = Path(campaign.root) / "edited-chain.json"
    file_sha = W.write_json(path, chain)
    return rewrite(
        campaign,
        lambda p: p["rosters"].update(
            reference_admission_chain={
                "path": str(path),
                "file_sha256": file_sha,
                "chain_sha256": chain["chain_sha256"],
            }
        ),
    )


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda c: c.update(source_file_hashes="not_verified"), "did not verify its source files"),
        (lambda c: c.update(inventory_sha256="0" * 64), "not derived by this admission chain"),
        (lambda c: c.update(policy_sha256="0" * 64), "not derived by this admission chain"),
        (lambda c: c["counts"].update(n_excluded=2, n_source=7), "counts differ"),
        (lambda c: c.update(admissible_roster_sha256="0" * 64), "another admissible roster"),
        (lambda c: c.update(schema="other/v1"), "record is required"),
    ],
)
def test_the_admission_chain_must_verify_and_derive_the_admissible_roster(campaign, change, message):
    path, sha = _rewrite_chain(campaign, change)
    with pytest.raises(PL.PlanError, match=message):
        PL.read_plan(path, sha)


def _rewrite_warmup(campaign, change):
    from passage_eval.io.roster import compute_roster_sha256

    roster = json.loads(Path(campaign.plan["warmup"]["roster"]["path"]).read_text())
    change(roster)
    roster["roster_sha256"] = compute_roster_sha256(roster)
    path = Path(campaign.root) / "edited-warmup.json"
    W.write_json(path, roster)
    return rewrite(campaign, lambda p: p["warmup"].update(roster=W.file_entry(path, roster)))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: r["selection"]["warmup"].update(evaluated_roster_sha256="0" * 64), "built disjoint"),
        (lambda r: r["selection"]["warmup"].update(disjointness_checked_by="sample_id"), "built disjoint"),
        (lambda r: r["dataset_root_identity"].update(sha256="0" * 64), "same dataset root"),
        (lambda r: r["dataset"].update(commit="2" * 40), "same dataset root"),
        (lambda r: r.update(schema="passage-roster-v1"), "passage-roster-v2"),
    ],
)
def test_the_warmup_roster_is_disjoint_from_and_rooted_like_the_admissible_roster(campaign, change, message):
    path, sha = _rewrite_warmup(campaign, change)
    with pytest.raises(PL.PlanError, match=message):
        PL.read_plan(path, sha)


def test_warmup_inputs_come_from_the_evaluated_cell(campaign, monkeypatch):
    from passage_eval.io import roster as roster_io

    real = roster_io.validate_roster

    def other_cell(payload, **kwargs):
        roster = real(payload, **kwargs)
        if roster["selection"].get("warmup"):
            roster["cell"] = dict(roster["cell"], cost_weight=1.0)
        return roster

    monkeypatch.setattr(roster_io, "validate_roster", other_cell)
    with pytest.raises(PL.PlanError, match="evaluated cell"):
        PL.read_plan(campaign.plan_path, campaign.sha)


def test_a_split_directory_or_source_file_reached_through_a_link_is_refused(campaign, tmp_path):
    original = Path(campaign.plan["dataset_root"])
    linked = tmp_path / "linked-root"
    (linked / "3x3").mkdir(parents=True)
    shutil.copy2(original / "PINNED.json", linked / "PINNED.json")
    (linked / "3x3" / "validation").symlink_to(original / "3x3" / "validation", target_is_directory=True)
    path, sha = rewrite(campaign, lambda p: p.update(dataset_root=str(linked)))
    with pytest.raises(PL.PlanError, match="validation must not be a symbolic link"):
        PL.read_plan(path, sha)
    (linked / "3x3" / "validation").unlink()
    (linked / "3x3" / "validation").mkdir()
    for shard in (original / "3x3" / "validation").iterdir():
        (linked / "3x3" / "validation" / shard.name).symlink_to(shard)
    with pytest.raises(PL.PlanError, match="must not be a symbolic link"):
        PL.read_plan(path, sha)
