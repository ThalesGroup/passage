"""The test admission gate ``passage-test-admission/v1``: each gate flips between RED and GREEN on its evidence only.

Every record and evidence file here is built in ``tmp_path`` from the synthetic world of ``campaign_world.py``; no
file of the released dataset is read. The edge evidence of G3 is written by these tests to exercise the verifier's
logic; it is never evidence of a real leased run.
"""

import base64
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from passage_eval.protocol import admission as A

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
#: The real clean-tree check (``conftest.py`` reports a clean checkout to the other tests).
REAL_UNCOMMITTED = A._uncommitted_harness_paths


def check(world, **kwargs):
    return A.check_admission(world.record_path, world.evidence, **kwargs)


def statuses(report):
    return {gate.gate: gate.status for gate in report.gates}


def codes(report, gate):
    return [reason["code"] for reason in report.gate(gate).reasons]


def all_green_except(report, *red):
    expected = {gate: ("RED" if gate in red else "GREEN") for gate in A.GATES}
    return statuses(report) == expected


@pytest.fixture
def world(tmp_path):
    copied = W.copy_green(tmp_path / "copy")
    assert check(copied).all_green
    return copied


def rewrite_evidence(world, relative, payload) -> dict:
    path = world.evidence / relative
    W.write_json(path, payload)
    return W.ref(world.evidence, path)


# ---------------------------------------------------------------------------------------------- template and CLI


def test_the_template_lists_every_author_decision_as_required_and_is_red_everywhere(tmp_path, capsys):
    from passage_eval.cli import main

    assert main(["admission-check", "--template"]) == 0
    template = json.loads(capsys.readouterr().out)
    assert template == A.template_record()
    decisions = template["protocol"]["decisions"]
    assert set(decisions) == set(A.DECISIONS) and len(decisions) >= 25
    assert all(
        entry == {"value": None, "status": "required", "note": A.DECISIONS[name]} for name, entry in decisions.items()
    )
    assert template["status"] == "draft" and template["approval"] is None
    path = tmp_path / "template.json"
    path.write_text(json.dumps(template))
    report = A.check_admission(path)
    assert statuses(report) == {gate: "RED" for gate in A.GATES}
    assert "record_not_sealed" in codes(report, "G1") and "approval_missing" in codes(report, "G1")
    assert codes(report, "G1").count("decision_not_sealed") == len(A.DECISIONS)
    assert main(["admission-check", str(path)]) == 1
    table = capsys.readouterr().out
    assert "RED: the test split stays refused (0/6 gates GREEN)" in table
    for gate, title in A.GATE_TITLES.items():
        assert f"{gate}    RED    {title}" in table


def test_the_green_world_passes_every_gate_and_the_cli_reports_it(capsys):
    from passage_eval.cli import main

    green = W.green_world()
    assert main(["admission-check", str(green.record_path), "--evidence-dir", str(green.evidence)]) == 0
    assert "GREEN: the test split may be opened (6/6 gates GREEN)" in capsys.readouterr().out
    assert main(["admission-check", str(green.record_path), "--json"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["verdict"] == "GREEN" and report["all_green"] is True and report["test_split"] == "may be opened"
    assert [gate["status"] for gate in report["gates"]] == ["GREEN"] * 6


@pytest.mark.parametrize(
    "content, code",
    [
        (None, "record_unreadable"),
        (b"{not json", "record_structure_invalid"),
        (b'{"schema": "passage-test-admission/v1", "schema": "x"}', "record_structure_invalid"),
        (b'{"schema": "other/v1"}', "record_structure_invalid"),
    ],
)
def test_an_unreadable_or_malformed_record_turns_every_gate_red_without_raising(tmp_path, content, code):
    path = tmp_path / "record.json"
    if content is not None:
        path.write_bytes(content)
    report = A.check_admission(path)
    assert not report.all_green and all(codes(report, gate) == [code] for gate in A.GATES)


def test_unknown_keys_bad_values_and_a_missing_evidence_dir_stay_red_and_never_raise(world, tmp_path):
    record = copy.deepcopy(world.record)
    record["extra"] = 1
    W.save_record(world, record, reapprove=False)
    assert codes(check(world), "G3") == ["record_structure_invalid"]
    record.pop("extra")
    record["protocol"]["decisions"]["budget"]["value"] = "x"
    record["protocol"]["decisions"]["paired_order"]["value"] = {"procedures": 3}
    W.save_record(world, record)
    report = check(world)
    assert "decision_invalid" in codes(report, "G1") and not report.all_green
    assert (
        A.check_admission(world.record_path, tmp_path / "absent").gate("G4").reasons[0]["code"]
        == "evidence_dir_missing"
    )


# --------------------------------------------------------------------------------------------------------- G1

#: The sealed batching decision of the synthetic world (dynamic admission window, R7).
BATCHING = {
    "batch_size": 4,
    "lease_or_timeout_ms": 3.6e9,
    "fixed_overhead_ms": 30000,
    "per_seed_overhead_ms": 1000,
    "per_request_overhead_ms": 50,
    "persistence_ms": 2000,
    "observed_request_ms": {"astar": 30.0, "reg": 1100.0, "seg": 1600.0},
    "safety_margin": 3.0,
}


def test_the_sealed_batching_accepts_lots_of_1024_tasks_with_a_restart_cap_at_least_as_large(world):
    from passage_eval.protocol import admission

    assert world.record["protocol"]["decisions"]["batching"]["value"] == BATCHING
    decisions = copy.deepcopy(world.record["protocol"]["decisions"])
    decisions["batching"]["value"] = dict(BATCHING, batch_size=1024)
    assert admission._decision_problem("batching", decisions["batching"]["value"], decisions) is None
    assert "at least the sealed batch size" in admission._decision_problem("max_restarts", 1023, decisions)
    assert admission._decision_problem("max_restarts", 1024, decisions) is None


def _set_decision(name, value=None, status=None):
    def change(world, record):
        entry = record["protocol"]["decisions"][name]
        if value is not None:
            entry["value"] = value
        if status is not None:
            entry["status"] = status

    return change


G1_CASES = {
    "record_not_sealed": lambda world, record: record.update(status="draft"),
    "decision_not_sealed": _set_decision("max_restarts", status="proposed_not_sealed"),
    "decision_has_unsealed_value": _set_decision("cold_start_policy", value="draft: decide later"),
    "decision_invalid": _set_decision("timing_boundary_id", value="passage-timing-boundary/v4-supervised-ipc-service"),
    "harness_commit_not_authorized": _set_decision("harness_commit_authorization", value={"commit": "0" * 40}),
    "approval_missing": None,
}


@pytest.mark.parametrize("code", sorted(G1_CASES))
def test_g1_is_red_for_each_unsealed_or_unapproved_record_and_green_again_once_sealed(world, code):
    original = copy.deepcopy(world.record)
    record = copy.deepcopy(world.record)
    if code == "approval_missing":
        record["approval"] = None
        W.save_record(world, record, reapprove=False)
    else:
        G1_CASES[code](world, record)
        W.save_record(world, record)
    report = check(world)
    assert code in codes(report, "G1")
    assert report.gate("G1").status == "RED" and not report.all_green
    W.save_record(world, original, reapprove=False)
    assert check(world).all_green


@pytest.mark.parametrize(
    "name, value",
    [
        ("dataset_revision", {"repo": "org/passage", "commit": "abc", "config": "3x3", "resolution": 3}),
        ("dataset_revision", {"repo": "org/passage", "commit": "1" * 40, "config": "256x256", "resolution": 3}),
        ("dataset_revision", {"repo": " ", "commit": "1" * 40, "config": "3x3", "resolution": 3}),
        ("dataset_revision", {"repo": "org/passage", "commit": "1" * 40, "config": "3x3"}),
        ("dataset_root_sha256", "0" * 63),
        ("evaluation_cell", {"cost_model": "elevation", "cost_weight": 62.4, "task_variant": "roads"}),
        ("evaluation_cell", {"cost_model": "elevation", "cost_weight": "62.4", "task_variant": "obstacles"}),
        ("evaluation_cell", {"cost_model": "elevation", "task_variant": "obstacles"}),
        ("population", {"n_rows": 0, "ids_sha256": "0" * 64}),
        ("population", {"n_rows": 11, "ids_sha256": None}),
        ("warmup", {"salt": W.WARM_SALT, "k": 2, "dataset_root_sha256": "0" * 64}),
        ("batching", dict(BATCHING, batch_size=1025)),
        ("batching", dict(BATCHING, safety_margin=2.5)),
        ("batching", dict(BATCHING, observed_request_ms={})),
        ("batching", dict(BATCHING, observed_request_ms={"seg": 0})),
        ("batching", dict(BATCHING, persistence_ms=-1)),
        ("batching", {k: v for k, v in BATCHING.items() if k != "safety_margin"}),
        ("max_restarts", 3),
    ],
)
def test_g1_validates_the_sealed_dataset_cell_population_and_warmup_root(world, name, value):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"][name]["value"] = value
    W.save_record(world, record)
    assert f"{name}: " in " ".join(
        r["detail"] for r in check(world).gate("G1").reasons if r["code"] == "decision_invalid"
    )


def test_g1_detects_a_record_changed_after_its_approval_and_an_approval_dated_in_the_future(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["seeds"]["value"] = [2026, 1234]
    W.save_record(world, record, reapprove=False)  # edited after approval
    report = check(world)
    assert codes(report, "G1") == ["approval_hash_mismatch"]
    W.save_record(world, record)  # approved again: the edit is now what the approver approved
    assert check(world).gate("G1").status == "GREEN"
    early = dt.datetime(2026, 10, 1, tzinfo=dt.timezone.utc)
    assert "approval_in_the_future" in codes(check(world, now=early), "G1")


def test_g1_refuses_a_sealed_document_that_still_carries_proposed_values(world):
    document = world.evidence / "sealed-protocol.md"
    document.write_text("budget: 120000 ms (proposed_not_sealed)\n")
    record = copy.deepcopy(world.record)
    record["protocol"]["document"] = W.ref(world.evidence, document)
    W.save_record(world, record)
    assert codes(check(world), "G1") == ["protocol_document_has_unsealed_values"]


# --------------------------------------------------------------------------------------------------------- G2


def _rehearsal(world):
    path = world.evidence / world.record["evidence"]["native_chain_rehearsal"]["path"]
    return json.loads(path.read_text())


def _swap_rehearsal(world, change):
    receipt = _rehearsal(world)
    change(receipt)
    record = copy.deepcopy(world.record)
    record["evidence"]["native_chain_rehearsal"] = rewrite_evidence(
        world, "rehearsal/native-chain-rehearsal.json", receipt
    )
    W.save_record(world, record)


G2_CASES = {
    "rehearsal_not_validation": lambda r: r.update(split="test"),
    "rehearsal_not_host": lambda r: r.update(target="thor"),
    "rehearsal_harness_differs": lambda r: r["harness"].update(commit="0" * 40),
    "rehearsal_candidate_differs": lambda r: r["candidates"]["seg"].update(checkpoint_sha256="0" * 64),
    "rehearsal_budget_differs": lambda r: r["budget"].update(wall_time_ms=180000),
    "rehearsal_timing_profile_differs": lambda r: r.update(timing_profile="supervised_service_v1"),
    "rehearsal_warmup_not_performed": lambda r: r["warmup"].update(performed=False),
    "rehearsal_rewarm_not_exercised": lambda r: r["rewarm"].update(n_rewarm_events=0),
    "rehearsal_resume_not_exercised": lambda r: r["resume"].update(n_unplanned_relaunches=1),
    "rehearsal_kill_not_exercised": lambda r: r.pop("deliberate_kill"),
    "rehearsal_procedures_differ": lambda r: r["procedures"].pop("astar"),
    "rehearsal_plan_missing": lambda r: r.pop("plan_sha256"),
    "rehearsal_evidence_not_real": lambda r: r.update(notes="loopback rehearsal, never run"),
    "rehearsal_has_unsealed_values": lambda r: r.update(notes="TODO: rerun with the sealed budget"),
}
#: Receipt changes that must turn G2 RED with an existing code (each check of the receipt is exercised).
G2_MORE_CASES = {
    "warmup_roster_digest": (lambda r: r["warmup"].update(roster_sha256="d" * 63), "rehearsal_warmup_not_performed"),
    "resume_without_resumed_batch": (
        lambda r: r["resume"].update(n_batches_resumed=0),
        "rehearsal_resume_not_exercised",
    ),
    "not_qualified": (lambda r: r.update(qualified=False), "rehearsal_evidence_not_real"),
    "resume_counts_only_the_legacy_field": (
        lambda r: r["resume"].update(n_automatic_relaunches=0, n_unplanned_relaunches=None),
        "rehearsal_resume_not_exercised",
    ),
    "kill_not_exercised": (lambda r: r["deliberate_kill"].update(exercised=False), "rehearsal_kill_not_exercised"),
    "no_deliberate_kill": (lambda r: r["deliberate_kill"].update(n_deliberate_kills=0), "rehearsal_kill_not_exercised"),
    "kill_not_relaunched": (
        lambda r: r["deliberate_kill"].update(n_deliberate_kills=2, n_planned_relaunches=1),
        "rehearsal_kill_not_exercised",
    ),
    "killed_attempt_scored": (
        lambda r: r["deliberate_kill"].update(killed_attempts_scored=1),
        "rehearsal_kill_not_exercised",
    ),
}


@pytest.mark.parametrize("code", sorted(G2_CASES))
def test_g2_is_red_for_each_incomplete_validation_rehearsal(world, code):
    _swap_rehearsal(world, G2_CASES[code])
    report = check(world)
    assert code in codes(report, "G2") and all_green_except(report, "G2")


@pytest.mark.parametrize("case", sorted(G2_MORE_CASES))
def test_g2_checks_every_field_of_the_rehearsal_receipt(world, case):
    change, code = G2_MORE_CASES[case]
    _swap_rehearsal(world, change)
    report = check(world)
    assert code in codes(report, "G2") and all_green_except(report, "G2")


def _aggregate(world, name):
    path = world.evidence / _rehearsal(world)["procedures"][name]["aggregate"]["path"]
    return json.loads(path.read_text())


def _swap_aggregate(world, name, aggregate, *, relative=None):
    """Write ``aggregate`` as the rehearsal aggregate of ``name`` and re-approve the record."""
    receipt = _rehearsal(world)
    receipt["procedures"][name]["aggregate"] = rewrite_evidence(
        world, relative or f"rehearsal/aggregate-{name}.json", aggregate
    )
    record = copy.deepcopy(world.record)
    record["evidence"]["native_chain_rehearsal"] = rewrite_evidence(
        world, "rehearsal/native-chain-rehearsal.json", receipt
    )
    W.save_record(world, record)


SEED = "2026"
AGGREGATE_CASES = {
    "schema": (lambda a: a.update(schema="passage-campaign-aggregate/v0"), "aggregate_schema_invalid"),
    "procedure": (lambda a: a.update(procedure="astar"), "aggregate_procedure_differs"),
    "target": (lambda a: a.update(target="thor"), "aggregate_not_host"),
    "split": (lambda a: a.update(split="test"), "aggregate_not_validation"),
    "harness": (lambda a: a["harness"].update(source_sha256="0" * 64), "aggregate_harness_differs"),
    "plan": (lambda a: a.update(plan_sha256="0" * 64), "aggregate_plan_differs"),
    "budget": (lambda a: a["budget"].update(wall_time_ms=1), "aggregate_budget_differs"),
    "timing_profile": (lambda a: a.update(timing_profile="supervised_service_v1"), "aggregate_timing_profile_differs"),
    "population": (lambda a: a["population"].update(n_scored=4), "aggregate_population_invalid"),
    "population_missing": (lambda a: a.pop("population"), "aggregate_population_invalid"),
    "rosters": (lambda a: a.update(admissible_roster_sha256=None), "aggregate_rosters_missing"),
    "chain": (lambda a: a.update(reference_admission_chain_sha256="x"), "aggregate_rosters_missing"),
    "receipts_empty": (lambda a: a.update(receipts=[]), "aggregate_receipts_missing"),
    "receipts_repeated": (lambda a: a["receipts"][1].update(batch_index=0), "aggregate_receipts_missing"),
    "receipts_repeated_unit": (
        lambda a: [row.update(batch_index=0, unit="000000.1") for row in a["receipts"][:2]],
        "aggregate_receipts_missing",
    ),
    "receipt_digest": (lambda a: a["receipts"][0].update(completion_sha256="0"), "aggregate_receipts_missing"),
    "top_equivalence": (
        lambda a: a["canonical_equivalence"].update(equal=False),
        "aggregate_not_equal_to_canonical_scorer",
    ),
    "seed_equivalence": (
        lambda a: a["per_seed"][SEED]["canonical_equivalence"].update(equal=False),
        "aggregate_not_equal_to_canonical_scorer",
    ),
    "seed_latency_rule": (
        lambda a: a["per_seed"][SEED]["latency"].update(rule="passage-censored-latency/v0"),
        "aggregate_latency_missing",
    ),
    "seed_latency_count": (
        lambda a: a["per_seed"][SEED]["latency"].update(n_warm_requests=99),
        "aggregate_latency_below_minimum",
    ),
    "seed_latency_flag": (
        lambda a: a["per_seed"][SEED]["latency"].update(meets_min_warm_requests=False),
        "aggregate_latency_below_minimum",
    ),
    "seeds": (lambda a: a.update(seeds=[2026, 1234]), "aggregate_seeds_differ"),
    "per_seed_missing": (lambda a: a.pop("per_seed"), "aggregate_seeds_differ"),
    "synthetic": (
        lambda a: a.update(limitations=["synthetic stub aggregate written by hand"]),
        "aggregate_evidence_not_real",
    ),
    "not_qualified": (lambda a: a["per_seed"][SEED].update(qualified=False), "aggregate_evidence_not_real"),
    "draft": (lambda a: a.update(limitations=["draft aggregate"]), "aggregate_has_unsealed_values"),
    "unsealed_status_elsewhere": (
        lambda a: a["per_seed"][SEED]["canonical_equivalence"].update(status="proposed_not_sealed"),
        "aggregate_has_unsealed_values",
    ),
}


def test_g2_accepts_one_completion_per_r7_unit_of_a_split_batch(world):
    """A batch split by R7 contributes one completion per half: same batch index, distinct units."""
    aggregate = _aggregate(world, "seg")
    first = aggregate["receipts"][0]
    aggregate["receipts"][:1] = [dict(first, unit="000000.0"), dict(first, unit="000000.1")]
    _swap_aggregate(world, "seg", aggregate)
    assert check(world).all_green


@pytest.mark.parametrize("case", sorted(AGGREGATE_CASES))
def test_g2_binds_each_aggregate_to_its_procedure_plan_limits_seeds_and_scorer(world, case):
    change, code = AGGREGATE_CASES[case]
    aggregate = _aggregate(world, "seg")
    change(aggregate)
    _swap_aggregate(world, "seg", aggregate)
    report = check(world)
    assert code in codes(report, "G2")
    assert [reason["detail"].split(":")[0] for reason in report.gate("G2").reasons if reason["code"] == code] == [
        "seg"
    ] * codes(report, "G2").count(code)
    assert all_green_except(report, "G2", "G5") if code in A._LATENCY_UNSHOWN else all_green_except(report, "G2")


def test_g2_accepts_the_installed_rule_status_label_only_in_a_latency_block_of_that_rule():
    from passage_eval.metrics import latency

    aggregate = W.rehearsal_aggregate(W.green_world().record, "seg")
    assert aggregate["per_seed"][SEED]["latency"]["status"] == latency.RULE_STATUS == "proposed_not_sealed"
    assert A.unsealed_markers(aggregate) == ["proposed_not_sealed"]
    assert A.unsealed_markers(A._without_installed_rule_status(aggregate)) == []
    aggregate["per_seed"][SEED]["latency"]["rule"] = "passage-censored-latency/v2"
    assert A.unsealed_markers(A._without_installed_rule_status(aggregate)) == ["proposed_not_sealed"]


def test_g2_refuses_one_aggregate_shared_by_several_procedures(world):
    """The reviewer's probe: one hand-written aggregate, of another target, budget and profile, for every procedure."""
    aggregate = _aggregate(world, "astar")
    aggregate.update(target="thor", timing_profile="supervised_service_v1")
    aggregate["budget"]["wall_time_ms"] = 1
    shared = rewrite_evidence(world, "rehearsal/shared-aggregate.json", aggregate)
    receipt = _rehearsal(world)
    for name in receipt["procedures"]:
        receipt["procedures"][name] = {"aggregate": shared}
    receipt["notes"] = "synthetic loopback simulated mock rehearsal, never run"
    record = copy.deepcopy(world.record)
    record["evidence"]["native_chain_rehearsal"] = rewrite_evidence(world, "rehearsal/receipt-probe.json", receipt)
    W.save_record(world, record)
    report = check(world)
    found = set(codes(report, "G2"))
    assert {
        "rehearsal_aggregate_shared",
        "rehearsal_evidence_not_real",
        "aggregate_procedure_differs",
        "aggregate_not_host",
        "aggregate_budget_differs",
        "aggregate_timing_profile_differs",
    } <= found
    assert not report.all_green
    # A byte copy under another path is still one aggregate for several procedures.
    world2 = W.copy_green(world.root.parent / "copy-2")
    copied = _aggregate(world2, "seg")
    _swap_aggregate(world2, "reg", copied, relative="rehearsal/aggregate-reg-copy.json")
    assert {"rehearsal_aggregate_shared", "aggregate_procedure_differs"} <= set(codes(check(world2), "G2"))


def test_g2_refuses_aggregates_of_different_rehearsal_campaigns(world):
    aggregate = _aggregate(world, "reg")
    aggregate["campaign_id"] = "another-rehearsal"
    _swap_aggregate(world, "reg", aggregate)
    report = check(world)
    assert codes(report, "G2") == ["rehearsal_aggregates_disagree"] and all_green_except(report, "G2")


def test_g2_and_g5_need_the_minimum_of_warm_requests_and_an_aggregation_equal_to_the_scorer(world):
    aggregate = _aggregate(world, "seg")
    aggregate["per_seed"][SEED]["latency"].update(n_warm_requests=99, meets_min_warm_requests=False)
    aggregate["per_seed"][SEED]["canonical_equivalence"]["equal"] = False
    aggregate["canonical_equivalence"]["equal"] = False
    _swap_aggregate(world, "seg", aggregate)
    report = check(world)
    assert {"aggregate_latency_below_minimum", "aggregate_not_equal_to_canonical_scorer"} <= set(codes(report, "G2"))
    assert codes(report, "G5") == ["rehearsal_latency_not_shown"] and all_green_except(report, "G2", "G5")


def test_g2_checks_every_sealed_seed_of_a_multi_seed_rehearsal(world):
    """A confirmatory rehearsal (three seeds): a later seed that disagrees with the scorer or lacks warm requests is
    RED even when the first seed is clean."""
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["seeds"]["value"] = [1234, 2026, 3407]
    record["evidence"]["native_chain_rehearsal"] = W.write_rehearsal(world.evidence, record)
    W.save_record(world, record)
    assert check(world).all_green
    aggregate = _aggregate(world, "reg")
    assert sorted(aggregate["per_seed"]) == ["1234", "2026", "3407"] and aggregate["latency"] is None
    aggregate["per_seed"]["3407"]["canonical_equivalence"]["equal"] = False
    aggregate["per_seed"]["2026"]["latency"]["n_warm_requests"] = 3
    _swap_aggregate(world, "reg", aggregate)
    report = check(world)
    details = {reason["detail"] for reason in report.gate("G2").reasons}
    assert "reg: seed 3407: metrics differ from the scorer" in details
    assert "reg: seed 2026: 3 warm request(s) < 100" in details
    assert all_green_except(report, "G2", "G5")
    aggregate["per_seed"].pop("3407")
    _swap_aggregate(world, "reg", aggregate)
    assert "aggregate_seeds_differ" in codes(check(world), "G2")


def test_g2_requires_the_running_harness_to_be_the_pinned_one(world, monkeypatch):
    monkeypatch.setattr(A, "_running_harness", lambda: ("1" * 40, "2" * 64))
    report = check(world)
    assert {"running_harness_source_differs", "running_harness_commit_differs"} <= set(codes(report, "G2"))
    monkeypatch.setattr(A, "_running_harness", lambda: (None, world.record["harness"]["source_sha256"]))
    assert "running_harness_commit_unknown" in codes(check(world), "G2")


def test_g2_refuses_a_harness_checkout_with_uncommitted_changes(world, monkeypatch):
    monkeypatch.setattr(A, "_uncommitted_harness_paths", lambda root=None: ["docs/benchmark/final-candidates.json"])
    report = check(world)
    assert codes(report, "G2") == ["running_harness_tree_dirty"] and all_green_except(report, "G2")
    with pytest.raises(PermissionError, match="running_harness_tree_dirty"):
        A.grant_test_admission(world.record_path, world.evidence)
    monkeypatch.setattr(A, "_uncommitted_harness_paths", lambda root=None: None)
    assert codes(check(world), "G2") == ["running_harness_tree_unverifiable"]


def _git(root, *args):
    import os
    import subprocess

    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@example.invalid", GIT_CONFIG_NOSYSTEM="1")
    env.update(GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@example.invalid", GIT_CONFIG_GLOBAL=os.devnull)
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True, env=env)


def test_the_clean_tree_check_sees_tracked_untracked_and_frozen_file_edits(tmp_path):
    root = tmp_path / "repo"
    (root / "src" / "pkg").mkdir(parents=True)
    (root / "docs" / "benchmark").mkdir(parents=True)
    (root / "src" / "pkg" / "module.py").write_text("x = 1\n")
    (root / "docs" / "benchmark" / "final-candidates.json").write_text("{}\n")
    (root / "notes.md").write_text("outside the harness paths\n")
    (root / ".gitignore").write_text("__pycache__/\n")
    _git(root, "init", "-q")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "initial")
    assert REAL_UNCOMMITTED(root) == []
    (root / "notes.md").write_text("an edit outside the harness paths is not a harness change\n")
    (root / "src" / "pkg" / "__pycache__").mkdir()
    (root / "src" / "pkg" / "__pycache__" / "module.pyc").write_bytes(b"ignored")
    assert REAL_UNCOMMITTED(root) == []
    (root / "src" / "pkg" / "module.py").write_text("x = 2\n")
    (root / "src" / "pkg" / "extra.py").write_text("y = 1\n")
    (root / "docs" / "benchmark" / "final-candidates.json").write_text('{"forged": true}\n')
    assert REAL_UNCOMMITTED(root) == [
        "docs/benchmark/final-candidates.json",
        "src/pkg/extra.py",
        "src/pkg/module.py",
    ]
    assert REAL_UNCOMMITTED(tmp_path / "not-a-repository") is None


# --------------------------------------------------------------------------------------------------------- G3


T0 = dt.datetime(2026, 11, 1, 10, 0, tzinfo=dt.timezone.utc)


def _iso(seconds):
    return (T0 + dt.timedelta(seconds=seconds)).isoformat()


def write_edge_evidence(
    world,
    target,
    *,
    machine=None,
    period=3.0,
    checks=range(0, 60, 3),
    changes=None,
    activity=None,
    log_changes=None,
    receipt_changes=None,
    n_checks=None,
):
    """Well-formed G3 evidence written by this test: exercises the verifier, proves nothing about a target.

    ``log_changes(index, entry)`` edits a lease check line, ``receipt_changes(kind, receipt)`` a fault-test receipt
    and ``changes(evidence)`` the evidence file; ``n_checks`` overrides the declared number of checks.
    """
    activity = activity or "job:" + hashlib.sha256(f"lease-{target}".encode()).hexdigest()[:16]
    resource = f"{target}-accelerator"
    directory = world.evidence / "edge" / target
    entries = [
        {
            "schema": A.LEASE_CHECK_SCHEMA,
            "checked_at": _iso(second),
            "activity_id": activity,
            "resource": resource,
            "holder_host": f"edge-{target}",
            "confirmed": True,
        }
        for second in checks
    ]
    for index, entry in enumerate(entries):
        if log_changes:
            log_changes(index, entry)
    lines = [json.dumps(entry, sort_keys=True) for entry in entries]
    log = directory / "lease-checks.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("\n".join(lines) + "\n")
    receipts = {}
    for kind, outcome in (
        ("transport_loss_test", "fail_closed_lease_retained"),
        ("cancellation_test", "drained_before_release"),
    ):
        fault = {
            "schema": A.FAULT_TEST_SCHEMA,
            "target": target,
            "kind": kind,
            "outcome": outcome,
            "drain_confirmed": True,
            "lease_retained_until_drain": True,
            "activity_id": activity,
        }
        if receipt_changes:
            receipt_changes(kind, fault)
        receipts[kind] = rewrite_evidence(world, f"edge/{target}/{kind}.json", fault)
    evidence = {
        "schema": A.EDGE_EVIDENCE_SCHEMA,
        "target": target,
        "harness": dict(world.record["harness"]),
        "target_identity": {
            "hostname": f"{target}-unit-01",
            "machine_id_sha256": hashlib.sha256((machine or target).encode()).hexdigest(),
            "ssh_host_key_sha256": "SHA256:" + base64.b64encode(hashlib.sha256(target.encode()).digest()).decode(),
            "agent_sha256": hashlib.sha256(b"remote agent").hexdigest(),
            "attested_at": _iso(1),
        },
        "lease": {
            "resource": resource,
            "activity_id": activity,
            "job_id": f"cavops-{target}-0001",
            "holder_host": f"edge-{target}",
            "effect_key": f"passage-256-{target}-batch-0",
            "acquired_at": _iso(0),
            "released_at": _iso(58),
        },
        "run": {"started_at": _iso(3), "finished_at": _iso(50), "plan_sha256": "e" * 64},
        "lease_verification_log": {
            "path": str(log.relative_to(world.evidence)),
            "sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
            "period_s": period,
            "n_checks": len(lines) if n_checks is None else n_checks,
        },
        "drain": {"cgroup_empty": True, "accelerator_idle": True, "before_release": True, "confirmed_at": _iso(54)},
        "transport_loss_test": {"receipt": receipts["transport_loss_test"], "outcome": "fail_closed_lease_retained"},
        "cancellation_test": {"receipt": receipts["cancellation_test"], "outcome": "drained_before_release"},
    }
    if changes:
        changes(evidence)
    return rewrite_evidence(world, f"edge/{target}/evidence.json", evidence)


def _with_targets(world, targets, edge_refs):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["targets"]["value"] = list(targets)
    record["candidates"] = W.candidates_block(world.evidence, targets=tuple(targets))
    for target, value in edge_refs.items():
        record["evidence"]["edge"][target] = value
    return W.save_record(world, record)


def test_g3_is_red_without_edge_evidence_and_green_only_with_complete_evidence(world):
    _with_targets(world, ["host", "thor"], {})
    report = check(world)
    assert codes(report, "G3") == ["edge_evidence_thor_missing"] and all_green_except(report, "G3")
    _with_targets(world, ["host", "thor"], {"thor": write_edge_evidence(world, "thor")})
    assert check(world).all_green
    _with_targets(world, ["host", "thor", "orin"], {"thor": write_edge_evidence(world, "thor")})
    assert codes(check(world), "G3") == ["edge_evidence_orin_missing"]


G3_CASES = {
    "synthetic_marker": (lambda e: e["target_identity"].update(hostname="thor-synthetic-01"), "edge_evidence_not_real"),
    "loopback_marker": (lambda e: e["lease"].update(resource="loopback-transport"), "edge_evidence_not_real"),
    "not_qualified": (lambda e: e.update(qualified=False), "edge_evidence_not_real"),
    "no_target_evidence": (lambda e: e.update(target_evidence="none"), "edge_evidence_not_real"),
    "activity_format": (lambda e: e["lease"].update(activity_id="job-1"), "edge_lease_invalid"),
    "holder_host": (lambda e: e["lease"].update(holder_host="workstation"), "edge_lease_invalid"),
    "drain_after_release": (lambda e: e["drain"].update(confirmed_at=_iso(59)), "edge_lease_order_invalid"),
    "drain_unconfirmed": (lambda e: e["drain"].update(accelerator_idle=False), "edge_drain_not_confirmed"),
    "identity": (lambda e: e["target_identity"].update(machine_id_sha256="abc"), "edge_identity_attestation_invalid"),
    "harness": (lambda e: e["harness"].update(source_sha256="0" * 64), "edge_harness_differs"),
    "fault_outcome": (lambda e: e["cancellation_test"].update(outcome="killed"), "edge_cancellation_test_invalid"),
    "target_mismatch": (lambda e: e.update(target="orin"), "edge_target_mismatch"),
}


@pytest.mark.parametrize("case", sorted(G3_CASES))
def test_g3_refuses_marked_incomplete_or_inconsistent_edge_evidence(world, case):
    change, code = G3_CASES[case]
    _with_targets(world, ["host", "thor"], {"thor": write_edge_evidence(world, "thor", changes=change)})
    report = check(world)
    assert code in codes(report, "G3") and all_green_except(report, "G3")


@pytest.mark.parametrize(
    "kwargs, code",
    [
        ({"checks": [0, 3, 6, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57]}, "edge_lease_not_continuously_confirmed"),
        ({"checks": range(6, 60, 3)}, "edge_lease_log_does_not_cover_run"),
        ({"checks": range(0, 51, 3)}, "edge_lease_log_does_not_cover_run"),
        ({"period": 0.0}, "edge_lease_log_invalid"),
    ],
)
def test_g3_needs_a_continuous_lease_verification_covering_the_run_and_the_drain(world, kwargs, code):
    _with_targets(world, ["host", "thor"], {"thor": write_edge_evidence(world, "thor", **kwargs)})
    assert code in codes(check(world), "G3")


def _only_cancellation(change):
    return lambda kind, receipt: change(receipt) if kind == "cancellation_test" else None


G3_MORE_CASES = {
    "receipt_marker": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(note="simulated drain"))},
        "edge_evidence_not_real",
    ),
    "receipt_unqualified": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(qualified=False))},
        "edge_evidence_not_real",
    ),
    "receipt_target": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(target="orin"))},
        "edge_cancellation_test_invalid",
    ),
    "receipt_kind": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(kind="transport_loss_test"))},
        "edge_cancellation_test_invalid",
    ),
    "receipt_outcome": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(outcome="killed"))},
        "edge_cancellation_test_invalid",
    ),
    "receipt_drain": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(drain_confirmed=False))},
        "edge_cancellation_test_invalid",
    ),
    "receipt_lease": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(lease_retained_until_drain=False))},
        "edge_cancellation_test_invalid",
    ),
    "receipt_activity": (
        {"receipt_changes": _only_cancellation(lambda r: r.update(activity_id="job-1"))},
        "edge_cancellation_test_invalid",
    ),
    "transport_receipt_schema": (
        {"receipt_changes": lambda kind, r: r.update(schema="x") if kind == "transport_loss_test" else None},
        "edge_transport_loss_test_invalid",
    ),
    "log_count": ({"n_checks": 3}, "edge_lease_log_invalid"),
    "log_activity": (
        {"log_changes": lambda i, e: e.update(activity_id="job:" + "0" * 16) if i == 4 else None},
        "edge_lease_not_continuously_confirmed",
    ),
    "log_resource": (
        {"log_changes": lambda i, e: e.update(resource="another-board") if i == 4 else None},
        "edge_lease_not_continuously_confirmed",
    ),
    "log_holder": (
        {"log_changes": lambda i, e: e.update(holder_host="edge-orin") if i == 4 else None},
        "edge_lease_not_continuously_confirmed",
    ),
    "log_unconfirmed": (
        {"log_changes": lambda i, e: e.update(confirmed=False) if i == 4 else None},
        "edge_lease_not_continuously_confirmed",
    ),
    "log_schema": (
        {"log_changes": lambda i, e: e.update(schema="x") if i == 4 else None},
        "edge_lease_not_continuously_confirmed",
    ),
    "log_order": (
        {"checks": [0, 3, 6, 12, 9, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48, 51, 54, 57]},
        "edge_lease_log_invalid",
    ),
    "log_timestamp": (
        {"log_changes": lambda i, e: e.update(checked_at="yesterday") if i == 4 else None},
        "edge_lease_log_invalid",
    ),
    "host_key_prefix": (
        {"changes": lambda e: e["target_identity"].update(ssh_host_key_sha256="MD5:" + "a" * 43)},
        "edge_identity_attestation_invalid",
    ),
    "agent_digest": (
        {"changes": lambda e: e["target_identity"].update(agent_sha256=None)},
        "edge_identity_attestation_invalid",
    ),
    "effect_key": ({"changes": lambda e: e["lease"].update(effect_key=" ")}, "edge_lease_invalid"),
    "job_id": ({"changes": lambda e: e["lease"].update(job_id="")}, "edge_lease_invalid"),
    "run_plan": ({"changes": lambda e: e["run"].update(plan_sha256="e" * 63)}, "edge_run_invalid"),
    "drain_before_release": (
        {"changes": lambda e: e["drain"].update(before_release=False)},
        "edge_drain_not_confirmed",
    ),
    "cgroup": ({"changes": lambda e: e["drain"].update(cgroup_empty=None)}, "edge_drain_not_confirmed"),
    "timestamp": (
        {"changes": lambda e: e["lease"].update(acquired_at="2026-11-01T10:00:00")},
        "edge_timestamps_invalid",
    ),
    "schema": (
        {"changes": lambda e: e.update(schema="passage-edge-lease-evidence/v0")},
        "edge_evidence_schema_invalid",
    ),
    "missing_receipt": (
        {"changes": lambda e: e.pop("transport_loss_test")},
        "edge_transport_loss_test_receipt_thor_missing",
    ),
}


@pytest.mark.parametrize("case", sorted(G3_MORE_CASES))
def test_g3_checks_every_receipt_log_and_lease_field(world, case):
    kwargs, code = G3_MORE_CASES[case]
    _with_targets(world, ["host", "thor"], {"thor": write_edge_evidence(world, "thor", **kwargs)})
    report = check(world)
    assert code in codes(report, "G3") and all_green_except(report, "G3")


def test_g3_refuses_two_targets_on_one_lease_activity(world):
    shared = "job:" + "1" * 16
    thor = write_edge_evidence(world, "thor", activity=shared)
    orin = write_edge_evidence(world, "orin", activity=shared)
    _with_targets(world, ["host", "thor", "orin"], {"thor": thor, "orin": orin})
    report = check(world)
    assert codes(report, "G3") == ["edge_lease_not_distinct"] and all_green_except(report, "G3")


def test_the_report_names_the_edge_targets_without_an_admitted_latency_claim(world):
    report = check(world)  # host only: G3 is vacuous and no edge latency claim is admitted
    assert report.all_green
    assert report.edge_latency_claims == {target: "not_admitted: not a sealed target" for target in A.EDGE_TARGETS}
    assert report.to_dict()["edge_latency_claims"] == report.edge_latency_claims
    assert "Edge latency claims admitted: none; not admitted: thor, orin, tiger" in A.format_report(report)
    _with_targets(world, ["host", "thor"], {"thor": write_edge_evidence(world, "thor")})
    report = check(world)
    assert report.all_green and report.edge_latency_claims == {
        "thor": "admitted",
        "orin": "not_admitted: not a sealed target",
        "tiger": "not_admitted: not a sealed target",
    }
    assert "Edge latency claims admitted: thor; not admitted: orin, tiger" in A.format_report(report)
    _with_targets(world, ["host", "thor", "orin"], {"thor": write_edge_evidence(world, "thor")})
    report = check(world)
    assert report.edge_latency_claims["thor"] == report.edge_latency_claims["orin"] == "not_admitted: gates RED"


def test_g3_refuses_a_marked_lease_log_line_and_targets_that_share_an_identity(world):
    reference = write_edge_evidence(world, "thor")
    evidence = json.loads((world.evidence / reference["path"]).read_text())
    log = world.evidence / evidence["lease_verification_log"]["path"]
    lines = log.read_text().splitlines()
    first = json.loads(lines[0])
    first["note"] = "replayed from a mock registry"
    lines[0] = json.dumps(first, sort_keys=True)
    log.write_text("\n".join(lines) + "\n")
    evidence["lease_verification_log"]["sha256"] = hashlib.sha256(log.read_bytes()).hexdigest()
    reference = rewrite_evidence(world, "edge/thor/evidence.json", evidence)
    _with_targets(world, ["host", "thor"], {"thor": reference})
    assert "edge_evidence_not_real" in codes(check(world), "G3")
    thor = write_edge_evidence(world, "thor", machine="same-board")
    orin = write_edge_evidence(world, "orin", machine="same-board")
    _with_targets(world, ["host", "thor", "orin"], {"thor": thor, "orin": orin})
    assert "edge_target_identity_not_distinct" in codes(check(world), "G3")


def test_g3_is_red_while_the_targets_are_not_sealed(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["targets"]["status"] = "proposed_not_sealed"
    W.save_record(world, record)
    report = check(world)
    assert "targets_not_sealed" in codes(report, "G3") and all_green_except(report, "G1", "G3")


# --------------------------------------------------------------------------------------------------------- G4


def test_g4_detects_an_evidence_file_changed_after_the_record(world):
    inventory = world.evidence / "reference-admission-inventory.json"
    inventory.write_bytes(inventory.read_bytes() + b" ")
    report = check(world)
    assert codes(report, "G4") == ["inventory_unverifiable"]
    # G5 checks the warm-up against the admissible roster that only a GREEN G4 establishes.
    assert codes(report, "G5") == ["warmup_not_disjoint_from_admissible_roster"] and all_green_except(
        report, "G4", "G5"
    )


def test_g4_rederives_the_chain_and_refuses_a_resealed_tampered_inventory(world):
    inventory = json.loads((world.evidence / "reference-admission-inventory.json").read_text())
    task = next(t for t in inventory["tasks"] if t["category"] == "admissible")
    task.update(category="excluded", reason="candidate_timeout", reasons=["candidate_timeout"], reference_id=None)
    inventory["inventory_sha256"] = hashlib.sha256(
        json.dumps(
            {k: v for k, v in inventory.items() if k != "inventory_sha256"}, sort_keys=True, separators=(",", ":")
        ).encode()
    ).hexdigest()
    record = copy.deepcopy(world.record)
    record["evidence"]["references"]["inventory"] = rewrite_evidence(
        world, "reference-admission-inventory.json", inventory
    )
    W.save_record(world, record)  # even re-approved, the inventory does not re-derive
    report = check(world)
    assert codes(report, "G4") == ["admission_chain_invalid"] and all_green_except(report, "G4", "G5")


@pytest.mark.parametrize(
    "decision, value, code",
    [
        ("reference_max_expansions", 65536, "reference_limit_differs"),
        ("reference_policy", "independent_analytic_reference", "reference_policy_differs"),
        ("reference_category_table_sha256", "0" * 64, "category_table_differs"),
    ],
)
def test_g4_binds_the_sealed_reference_decisions(world, decision, value, code):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"][decision]["value"] = value
    if decision == "reference_policy":
        record["protocol"]["decisions"]["reference_cohort_id"]["value"] = "cohort-a"
    W.save_record(world, record)
    assert code in codes(check(world), "G4")


def test_g4_does_not_treat_the_source_inventory_as_the_selected_roster_population(world):
    from passage_eval.io.roster import compute_roster_sha256

    source = json.loads((world.evidence / "source-roster.json").read_text())
    source["selection"]["population"]["n_rows"] = 12  # 11 of 12 rows: a sample, not the population
    source["roster_sha256"] = compute_roster_sha256(source)
    record = copy.deepcopy(world.record)
    record["evidence"]["references"]["source_roster"] = rewrite_evidence(world, "source-roster.json", source)
    W.save_record(world, record)
    # G4 pins the selected roster S; changing a source-inventory annotation
    # does not create the old "must be the full test population" failure.
    assert codes(check(world), "G4") == ["admission_chain_invalid"]


def test_g4_requires_the_recorded_chain_to_equal_the_rederived_chain(world):
    chain = json.loads((world.evidence / "reference-admission-chain.json").read_text())
    chain["order"] = "edited"
    record = copy.deepcopy(world.record)
    record["evidence"]["references"]["chain"] = rewrite_evidence(world, "reference-admission-chain.json", chain)
    W.save_record(world, record)
    report = check(world)
    assert codes(report, "G4") == ["admission_chain_differs"] and all_green_except(report, "G4")


def _reseal_roster(world, name, change):
    """Change an evidence roster, recompute its digest and point the record at it (re-approved)."""
    from passage_eval.io.roster import compute_roster_sha256

    roster = json.loads((world.evidence / name).read_text())
    change(roster)
    roster["roster_sha256"] = compute_roster_sha256(roster)
    reference = rewrite_evidence(world, name, roster)
    record = copy.deepcopy(world.record)
    for group in ("references", "warmup"):
        for key, value in record["evidence"][group].items():
            if value and value["path"] == name:
                record["evidence"][group][key] = reference
    W.save_record(world, record)
    return roster


G4_SOURCE_CASES = {
    "split": (lambda r: r["dataset"].update(split="validation"), "source_roster_split_differs"),
    "schema_v1": (lambda r: r.update(schema="passage-roster-v1"), "source_roster_not_v2"),
    "footprints_optional": (
        lambda r: r["geographic_blocks"].update(footprints_required=False),
        "source_roster_footprints_not_required",
    ),
    "revision": (lambda r: r["dataset"].update(commit="2" * 40), "source_roster_revision_differs"),
    "repo": (lambda r: r["dataset"].update(repo="another/passage"), "source_roster_revision_differs"),
    "root": (lambda r: r["dataset_root_identity"].update(sha256="0" * 64), "source_roster_dataset_root_differs"),
}


@pytest.mark.parametrize("case", sorted(G4_SOURCE_CASES))
def test_g4_checks_every_field_of_the_source_roster(world, case):
    change, code = G4_SOURCE_CASES[case]
    _reseal_roster(world, "source-roster.json", change)
    report = check(world)
    assert code in codes(report, "G4") and statuses(report)["G1"] == "GREEN"


@pytest.mark.parametrize(
    "decision, change, code",
    [
        ("dataset_revision", lambda v: v.update(commit="2" * 40), "source_roster_revision_differs"),
        ("dataset_revision", lambda v: v.update(commit="2" * 40), "admissible_roster_revision_differs"),
        ("evaluation_cell", lambda v: v.update(cost_weight=1.0), "source_roster_cell_differs"),
        ("evaluation_cell", lambda v: v.update(task_variant="free"), "admissible_roster_cell_differs"),
        ("population", lambda v: v.update(n_rows=40000), "source_roster_population_differs"),
        ("population", lambda v: v.update(ids_sha256="0" * 64), "source_roster_population_differs"),
    ],
)
def test_g4_binds_the_rosters_to_the_sealed_revision_cell_and_population(world, decision, change, code):
    """The reviewer's probe: an all-GREEN record admitted a roster of any revision, cell or subset."""
    record = copy.deepcopy(world.record)
    change(record["protocol"]["decisions"][decision]["value"])
    W.save_record(world, record)
    report = check(world)
    assert code in codes(report, "G4") and statuses(report)["G1"] == "GREEN" and not report.all_green


def test_g4_binds_the_sealed_dataset_root(world):
    record = copy.deepcopy(world.record)
    for name in ("dataset_root_sha256",):
        record["protocol"]["decisions"][name]["value"] = "0" * 64
    record["protocol"]["decisions"]["warmup"]["value"]["dataset_root_sha256"] = "0" * 64
    W.save_record(world, record)
    report = check(world)
    assert {"source_roster_dataset_root_differs", "admissible_roster_dataset_root_differs"} <= set(codes(report, "G4"))
    assert {"warmup_dataset_root_differs", "warmup_roster_dataset_root_differs"} <= set(codes(report, "G5"))
    assert statuses(report)["G1"] == "GREEN"


def test_g4_refuses_incomplete_block_tables_of_the_source_and_admissible_rosters(world, monkeypatch):
    from passage_eval.io import roster as roster_io

    real = roster_io.roster_block_table
    monkeypatch.setattr(roster_io, "roster_block_table", lambda roster: dict(real(roster), complete=False))
    report = check(world)
    assert {"source_roster_blocks_incomplete", "admissible_blocks_unsealed"} <= set(codes(report, "G4"))


def test_g4_binds_the_reference_cohort_and_refuses_an_unsealed_category_table(world, tmp_path):
    from passage_eval.protocol.reference_admission import PROPOSED_CATEGORY_TABLE

    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["reference_cohort_id"]["value"] = "cohort-a"
    W.save_record(world, record)
    assert "reference_cohort_differs" in codes(check(world), "G4")
    proposed = W.build_green(tmp_path / "proposed", category_table=PROPOSED_CATEGORY_TABLE)
    report = A.check_admission(proposed.record_path, proposed.evidence)
    assert codes(report, "G4") == ["category_table_not_sealed"] and all_green_except(report, "G4")


# --------------------------------------------------------------------------------------------------------- G5


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda d: d["warmup"]["value"].update(salt="another-salt"), "warmup_salt_differs"),
        (lambda d: d["warmup"]["value"].update(k=3), "warmup_size_differs"),
        (lambda d: d["warmup"]["value"].update(dataset_root_sha256="0" * 64), "warmup_dataset_root_differs"),
        (
            lambda d: d["censoring_rule"]["value"].update(id="passage-censored-latency/v2"),
            "censoring_rule_not_installed",
        ),
    ],
)
def test_g5_binds_the_sealed_warmup_and_censoring_rule(world, change, code):
    record = copy.deepcopy(world.record)
    change(record["protocol"]["decisions"])
    W.save_record(world, record)
    report = check(world)
    assert code in codes(report, "G5") and statuses(report)["G4"] == "GREEN"


def test_g5_refuses_a_warmup_roster_built_against_another_evaluated_roster(world):
    from passage_eval.io.roster import compute_roster_sha256

    roster = json.loads((world.evidence / "warmup-roster.json").read_text())
    roster["selection"]["warmup"]["evaluated_roster_sha256"] = "f" * 64
    roster["roster_sha256"] = compute_roster_sha256(roster)
    record = copy.deepcopy(world.record)
    record["evidence"]["warmup"]["roster"] = rewrite_evidence(world, "warmup-roster.json", roster)
    W.save_record(world, record)
    report = check(world)
    assert codes(report, "G5") == ["warmup_not_disjoint_from_admissible_roster"] and all_green_except(report, "G5")


G5_WARMUP_CASES = {
    "test_split": (lambda r: r["dataset"].update(split="test"), "warmup_roster_not_validation_v2"),
    "schema_v1": (lambda r: r.update(schema="passage-roster-v1"), "warmup_roster_not_validation_v2"),
    "disjoint_by_ids": (
        lambda r: r["selection"]["warmup"].update(disjointness_checked_by="sample_id"),
        "warmup_disjointness_by_content_missing",
    ),
    "revision": (lambda r: r["dataset"].update(commit="2" * 40), "warmup_roster_revision_differs"),
    "root": (lambda r: r["dataset_root_identity"].update(sha256="0" * 64), "warmup_dataset_root_not_admissible_root"),
}


@pytest.mark.parametrize("case", sorted(G5_WARMUP_CASES))
def test_g5_checks_every_field_of_the_warmup_roster(world, case):
    change, code = G5_WARMUP_CASES[case]
    _reseal_roster(world, "warmup-roster.json", change)
    report = check(world)
    assert code in codes(report, "G5") and all_green_except(report, "G5")


def _rewrite_selection(world, change, *, reseal_roster=True):
    from passage_eval.io import pinned

    selection = json.loads((world.evidence / "warmup-selection.json").read_text())
    change(selection)
    if reseal_roster:
        selection["selection_sha256"] = pinned.canonical_sha256(
            {k: v for k, v in selection.items() if k != "selection_sha256"}
        )
        _reseal_roster(
            world,
            "warmup-roster.json",
            lambda r: r["selection"]["warmup"].update(selection_sha256=selection["selection_sha256"]),
        )
    record = copy.deepcopy(world.record)
    record["evidence"]["warmup"]["selection"] = rewrite_evidence(world, "warmup-selection.json", selection)
    W.save_record(world, record)


@pytest.mark.parametrize(
    "change, reseal, code",
    [
        (lambda s: s.update(schema="passage-warmup-selection/v0"), False, "warmup_selection_invalid"),
        (lambda s: s.update(salt="edited"), False, "warmup_selection_differs"),
        (lambda s: s["selected_ids"].reverse(), True, "warmup_selection_differs"),
    ],
)
def test_g5_checks_the_selection_record_of_the_warmup_roster(world, change, reseal, code):
    _rewrite_selection(world, change, reseal_roster=reseal)
    report = check(world)
    assert codes(report, "G5") == [code] and all_green_except(report, "G5")


# --------------------------------------------------------------------------------------------------------- G6


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda c: c["seg"].update(checkpoint_sha256="0" * 64), "checkpoint_differs_from_final_candidates"),
        (lambda c: c["reg"].update(decoder="mypkg.decoders:Faster"), "decoder_entry_not_frozen"),
        (lambda c: c["seg"].update(decoder_tree_sha256="0" * 64), "decoder_tree_differs"),
        (lambda c: c["reg"].update(final_decoders_sha256="0" * 64), "final_decoders_pin_differs"),
        (lambda c: c["seg"]["targets"]["host"].update(bundle_sha256=None), "bundle_digest_missing"),
        (lambda c: c["seg"]["targets"].pop("host"), "target_artifacts_missing"),
    ],
)
def test_g6_verifies_candidate_identities_and_artifact_digests(world, change, code):
    record = copy.deepcopy(world.record)
    change(record["candidates"])
    W.save_record(world, record)
    report = check(world)
    assert code in codes(report, "G6")
    assert all(statuses(report)[gate] == "GREEN" for gate in ("G1", "G3", "G4", "G5"))


def test_g6_needs_edge_engine_and_container_digests_and_binds_a_bundle_file(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["targets"]["value"] = ["host", "tiger"]
    record["candidates"] = W.candidates_block(world.evidence, targets=("host", "tiger"))
    record["candidates"]["reg"]["targets"]["tiger"]["engine_sha256"] = None
    W.save_record(world, record)
    assert "engine_digest_missing" in codes(check(world), "G6")
    record = copy.deepcopy(W.green_world().record)  # back to the host-only record
    bundle = {"lineage": {"checkpoint_sha256": "0" * 64}}
    reference = rewrite_evidence(world, "bundles/seg-host.json", bundle)
    record["candidates"]["seg"]["targets"]["host"].update(bundle=reference, bundle_sha256=reference["sha256"])
    W.save_record(world, record)
    assert codes(check(world), "G6") == ["bundle_checkpoint_differs"]
    bundle["lineage"]["checkpoint_sha256"] = record["candidates"]["seg"]["checkpoint_sha256"]
    reference = rewrite_evidence(world, "bundles/seg-host.json", bundle)
    record["candidates"]["seg"]["targets"]["host"].update(bundle=reference, bundle_sha256=reference["sha256"])
    W.save_record(world, record)
    assert check(world).all_green


def test_g6_compares_the_installed_decoders_with_the_frozen_candidate_file(world, tmp_path):
    frozen = W.final_candidates()
    frozen["reference_decoders"]["module_sha256"] = "0" * 64
    path = tmp_path / "final-candidates.json"
    path.write_text(json.dumps(frozen))
    report = check(world, final_candidates=path)
    assert {"final_candidates_decoder_pin_differs", "final_candidates_not_committed"} <= set(codes(report, "G6"))
    assert codes(check(world, final_candidates=tmp_path / "absent.json"), "G6")[0] == "final_candidates_unavailable"


def _set_targets(world, change):
    record = copy.deepcopy(world.record)
    change(record["candidates"])
    W.save_record(world, record)
    return check(world)


def test_g6_requires_a_bundle_file_whose_digest_is_the_declared_one(world):
    report = _set_targets(world, lambda c: c["seg"]["targets"]["host"].update(bundle=None))
    assert codes(report, "G6") == ["bundle_file_missing"] and all_green_except(report, "G6")
    record = copy.deepcopy(W.green_world().record)
    record["candidates"]["reg"]["targets"]["host"]["bundle_sha256"] = "1" * 64
    W.save_record(world, record)
    assert codes(check(world), "G6") == ["bundle_digest_differs"]


def test_g6_refuses_a_malformed_container_digest(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["targets"]["value"] = ["host", "tiger"]
    record["candidates"] = W.candidates_block(world.evidence, targets=("host", "tiger"))
    record["candidates"]["seg"]["targets"]["tiger"]["container_digest"] = "sha256:" + "A" * 64
    W.save_record(world, record)
    assert "container_digest_missing" in codes(check(world), "G6")


def test_g6_reads_final_candidates_as_committed_at_the_record_commit(world, tmp_path, monkeypatch):
    """The reviewer's probe: forged checkpoints in a working-tree final-candidates.json are refused."""
    committed = A._default_final_candidates().read_bytes()
    copy_path = tmp_path / "final-candidates.json"
    copy_path.write_bytes(committed)
    assert check(world, final_candidates=copy_path).all_green
    forged = json.loads(committed)
    forged["candidates"]["seg"]["checkpoint_256"]["sha256"] = "f" * 64
    copy_path.write_text(json.dumps(forged))
    record = copy.deepcopy(world.record)
    record["candidates"]["seg"]["checkpoint_sha256"] = "f" * 64
    W.save_record(world, record)
    report = check(world, final_candidates=copy_path)
    assert "final_candidates_not_committed" in codes(report, "G6") and not report.all_green
    monkeypatch.setattr(A, "_committed_file", lambda *args, **kwargs: None)
    assert "final_candidates_unverifiable" in codes(check(world), "G6")


def test_g6_holds_the_frozen_decoder_digest_itself(world, tmp_path, monkeypatch):
    """A changed final_decoders.py is refused even when the record and final-candidates.json name its new digest."""
    assert A.FROZEN_FINAL_DECODERS_SHA256 == "2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4"
    edited = tmp_path / "final_decoders.py"
    edited.write_bytes(A._installed_final_decoders().read_bytes() + b"# edited\n")
    digest = hashlib.sha256(edited.read_bytes()).hexdigest()
    monkeypatch.setattr(A, "_installed_final_decoders", lambda: edited)
    frozen = W.final_candidates()
    frozen["reference_decoders"]["module_sha256"] = digest
    path = tmp_path / "final-candidates.json"
    path.write_text(json.dumps(frozen))
    record = copy.deepcopy(world.record)
    for name in A.CANDIDATES:
        record["candidates"][name]["final_decoders_sha256"] = digest
    W.save_record(world, record)
    found = set(codes(check(world, final_candidates=path), "G6"))
    assert {"final_decoders_changed", "final_candidates_decoder_pin_differs", "final_candidates_not_committed"} <= found


def test_committed_final_candidates_refuses_a_working_tree_edit_and_an_unknown_commit(tmp_path):
    from passage_eval import cli

    commit = cli._checkout_commit()
    content, digest = A.committed_final_candidates(commit)
    assert digest == hashlib.sha256(A._default_final_candidates().read_bytes()).hexdigest()
    assert content["reference_decoders"]["module_sha256"] == A.FROZEN_FINAL_DECODERS_SHA256
    edited = tmp_path / "final-candidates.json"
    edited.write_bytes(A._default_final_candidates().read_bytes() + b" ")
    with pytest.raises(ValueError, match="differs from the file committed"):
        A.committed_final_candidates(commit, edited)
    with pytest.raises(ValueError, match="cannot be read at commit"):
        A.committed_final_candidates("0" * 40)
    with pytest.raises(ValueError, match="cannot be read at commit"):
        A.committed_final_candidates("--output=/dev/null")
    with pytest.raises(ValueError, match="unavailable"):
        A.committed_final_candidates(commit, tmp_path / "absent.json")


def test_a_grant_rehashes_the_frozen_files_before_each_use(world, tmp_path):
    from passage_eval.protocol.guards import TestSplitRefused, refuse_test_split

    path = tmp_path / "final-candidates.json"
    path.write_bytes(A._default_final_candidates().read_bytes())
    grant = A.grant_test_admission(world.record_path, world.evidence, final_candidates=path)
    assert dict(grant.report.external_pins)[str(path.resolve())] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert str(A._installed_final_decoders()) in dict(grant.report.external_pins)
    assert refuse_test_split("test", allow_test=True, admission=grant) == "test"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(TestSplitRefused, match="SHA-256 mismatch for final-candidates.json"):
        refuse_test_split("test", allow_test=True, admission=grant)


def test_a_grant_admits_the_sealed_roster_and_true_children_only(world):
    from passage_eval.io import roster as roster_io

    grant = A.grant_test_admission(world.record_path, world.evidence)
    admissible = json.loads((world.evidence / "admissible-roster.json").read_text())
    assert grant.admits_roster(admissible)
    ids = [task["sample_id"] for task in admissible["tasks"]]
    child = roster_io.derive_roster(admissible, ids[:4], rule="test-child")
    assert grant.admits_roster(child)
    forged = copy.deepcopy(child)  # claims the admissible parent but analyses a task in another block
    blocks = sorted({task["parent_geographic_block"] for task in admissible["tasks"]})
    assert len(blocks) > 1
    task = forged["tasks"][0]
    task["parent_geographic_block"] = next(block for block in blocks if block != task["parent_geographic_block"])
    forged["roster_sha256"] = roster_io.compute_roster_sha256(forged)
    assert not grant.admits_roster(forged)
    unrelated = copy.deepcopy(child)
    unrelated["selection"]["derivation"]["parent_roster_sha256"] = "0" * 64
    unrelated["roster_sha256"] = roster_io.compute_roster_sha256(unrelated)
    assert not grant.admits_roster(unrelated) and not grant.admits_roster({"tasks": []})


def test_marker_scans_skip_digests_but_not_words():
    assert A.non_evidence_markers({"key": "SHA256:" + "stubFAKEmock" * 4}) == []
    assert A.non_evidence_markers({"key": "fa" * 32}) == []
    assert A.non_evidence_markers({"note": "a loopback transport"}) == ["loopback"]
    assert A.non_evidence_markers(["Synthetic lease"]) == ["synthetic"]
    assert A.unsealed_markers({"v": "Draft value", "w": "drafted"}) == ["draft"]
