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

"""Gate G4 for blocks and exclusion caps, and the phased test preparation grant.

Every record and evidence file is built in ``tmp_path`` from the synthetic world of ``campaign_world.py``; a split
named ``test`` there is a synthetic fixture, never the released test split.
"""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from passage_eval.protocol import admission as A
from passage_eval.protocol import guards
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


def check(world):
    return A.check_admission(world.record_path, world.evidence)


def codes(report, gate):
    return [reason["code"] for reason in report.gate(gate).reasons]


def red_gates(report):
    return [gate.gate for gate in report.gates if gate.status != "GREEN"]


@pytest.fixture
def world(tmp_path):
    copied = W.copy_green(tmp_path / "copy")
    assert check(copied).all_green
    return copied


def edit_decision(world, name, change):
    record = copy.deepcopy(world.record)
    value = record["protocol"]["decisions"][name]["value"]
    record["protocol"]["decisions"][name]["value"] = change(value) if callable(change) else change
    W.save_record(world, record)
    return check(world)


# ------------------------------------------------------------------------------------------------------------ G4


@pytest.mark.parametrize(
    "change, code",
    [
        (lambda v: dict(v, rule="main_tile_cell"), "source_roster_block_rule_differs"),
        (lambda v: dict(v, block_degrees=10), "source_roster_block_rule_differs"),
        (lambda v: dict(v, missing_footprint="unplaced_block"), "source_roster_unplaced_block_differs"),
    ],
)
def test_g4_binds_the_source_roster_to_the_sealed_block_analysis(world, change, code):
    report = edit_decision(world, "block_analysis", change)
    assert code in codes(report, "G4") and red_gates(report) == ["G4"]


def test_g4_binds_the_policy_to_the_sealed_exclusion_caps(world, tmp_path):
    report = edit_decision(world, "reference_exclusion_caps", lambda v: dict(v, max_block_fraction=0.06))
    assert codes(report, "G4") == ["exclusion_caps_differ"] and red_gates(report) == ["G4"]
    uncapped = W.build_green(tmp_path / "uncapped", exclusion_caps=None)
    report = A.check_admission(uncapped.record_path, uncapped.evidence)
    assert codes(report, "G4") == ["exclusion_caps_missing"] and red_gates(report) == ["G4"]
    from passage_eval.protocol.reference_admission import PROPOSED_EXCLUSION_CAPS

    proposed = W.build_green(tmp_path / "proposed", exclusion_caps=PROPOSED_EXCLUSION_CAPS)
    report = A.check_admission(proposed.record_path, proposed.evidence)
    assert codes(report, "G4") == ["exclusion_caps_not_sealed"] and red_gates(report) == ["G4"]


@pytest.mark.parametrize(
    "name, value",
    [
        ("block_analysis", None),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, rule="capital")),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, block_degrees=7)),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, missing_footprint="exclude")),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, min_blocks=1)),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, max_block_share=0)),
        ("block_analysis", dict(W.SEALED_BLOCK_ANALYSIS, max_unplaced_fraction=1)),
        ("block_analysis", {**W.SEALED_BLOCK_ANALYSIS, "extra": 1}),
        (
            "reference_exclusion_caps",
            {"max_total_fraction": 1.0, "max_block_fraction": 0.05, "min_block_exclusions": 2},
        ),
        (
            "reference_exclusion_caps",
            {"max_total_fraction": 0.1, "max_block_fraction": 0.05, "min_block_exclusions": 0},
        ),
        ("reference_exclusion_caps", {"max_total_fraction": 0.1}),
        ("confirmatory_analysis", None),
        ("confirmatory_analysis", {**W.SEALED_CONFIRMATORY_ANALYSIS, "extra": 1}),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, a="reg", b="seg")),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, target="moon")),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, seeds=[2026])),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, seeds=[2026, 2026])),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, alpha=1.0)),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, draws=0)),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, permutations=True)),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, bootstrap_seed=-1)),
        ("confirmatory_analysis", dict(W.SEALED_CONFIRMATORY_ANALYSIS, latency_bootstrap_seed=None)),
    ],
)
def test_g1_validates_the_block_analysis_and_exclusion_cap_decisions(world, name, value):
    report = edit_decision(world, name, value)
    assert any(r["code"] == "decision_invalid" and r["detail"].startswith(name) for r in report.gate("G1").reasons)
    assert A.decision_problem(name, value, world.record["protocol"]["decisions"]) is not None


def test_the_sealed_confirmatory_analysis_is_a_required_decision_of_the_record(world):
    assert "confirmatory_analysis" in A.DECISIONS
    assert A.template_record()["protocol"]["decisions"]["confirmatory_analysis"]["status"] == "required"
    assert A.decision_problem("confirmatory_analysis", W.SEALED_CONFIRMATORY_ANALYSIS, {}) is None
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["confirmatory_analysis"]["status"] = "proposed_not_sealed"
    W.save_record(world, record)
    assert "decision_not_sealed" in codes(check(world), "G1")


def test_the_preparation_scope_has_a_sealed_s_roster_selection_decision():
    """Preparation may only create the named S selection, never an arbitrary test roster."""
    selection = {
        "rule": "sha256(utf-8(salt + id)) hexdigest, ascending lexicographic",
        "salt": "passage-public-test-v1/r0256/roster-v1/",
        "n": 1024,
    }
    assert "roster_selection" in A.DECISIONS
    assert A.decision_problem("roster_selection", selection, {}) is None


def test_the_unplaced_block_admits_a_task_without_footprint_within_the_sealed_cap(tmp_path):
    analysis = dict(W.SEALED_BLOCK_ANALYSIS, missing_footprint="unplaced_block", max_unplaced_fraction=0.1)
    green = W.build_green(tmp_path / "unplaced", unplaced_block=True, block_analysis=analysis)
    report = A.check_admission(green.record_path, green.evidence)
    assert report.all_green
    adequacy = report.admitted["block_adequacy"]
    assert adequacy["n_unplaced"] == 1 and adequacy["n_tasks"] == 12 and adequacy["meets_criterion"] is False
    assert adequacy["meets_min_blocks"] is False  # 4 blocks < 30: the block analysis can only be descriptive
    analysis = report.admitted["block_analysis"]
    assert analysis["status"] == "descriptive_only" and analysis["confirmatory_block_analysis_possible"] is False
    assert analysis["n_unplaced"] == 1 and analysis["max_unplaced_fraction"] == 0.1


def test_unplaced_tasks_above_the_cap_keep_g4_red(tmp_path):
    """The stricter S protocol refuses G4 above its sealed 0.5 % unplaced cap."""
    analysis = dict(W.SEALED_BLOCK_ANALYSIS, missing_footprint="unplaced_block", max_unplaced_fraction=0.005)
    green = W.build_green(tmp_path / "unplaced", unplaced_block=True, block_analysis=analysis)
    report = A.check_admission(green.record_path, green.evidence)
    assert not report.all_green and "unplaced_cap_exceeded" in codes(report, "G4")
    halted = report.admitted["block_analysis"]
    assert halted["status"] == "halted_unplaced_cap" and halted["confirmatory_block_analysis_possible"] is False
    assert halted["n_unplaced"] == 1 and halted["n_tasks"] == 12 and halted["max_unplaced_fraction"] == 0.005
    assert report.admitted["block_adequacy"]["unplaced_within_cap"] is False


def test_a_block_analysis_without_a_numeric_unplaced_cap_is_red_in_g4(world):
    report = edit_decision(world, "block_analysis", lambda v: dict(v, max_unplaced_fraction="half"))
    assert "block_analysis_invalid" in codes(report, "G4")
    assert any(r["code"] == "decision_invalid" for r in report.gate("G1").reasons)
    assert report.admitted["block_analysis"]["status"] == "invalid"


def test_the_grant_exposes_the_inventory_reference_ids_and_policy():
    grant = W.green_grant()
    world = W.green_world().world
    expected = {t["sample_id"]: t["reference_id"] for t in world.inventory["tasks"] if t["category"] == "admissible"}
    assert grant.reference_ids() == expected and len(expected) == 11
    assert grant.reference_policy() == {
        "reference_policy": "released_labels",
        "reference_cohort_id": None,
        "reference_max_expansions": 9,
    }


# ----------------------------------------------------------------------------------------------- preparation


@pytest.fixture
def roster_phase(tmp_path):
    return W.preparation_world(tmp_path / "roster-phase")


def test_a_record_waiting_for_its_g4_g5_artefacts_allows_only_the_preparation_readers(roster_phase):
    report = check(roster_phase)
    assert red_gates(report) == ["G1", "G4", "G5"]
    assert codes(report, "G1") == ["decision_not_sealed"]
    assert set(codes(report, "G4")) == {f"{name}_missing" for name in A._REFERENCE_KEYS}
    assert codes(report, "G5") == ["warmup_roster_missing", "warmup_selection_missing"]
    with pytest.raises(PermissionError, match="not all GREEN"):
        A.grant_test_admission(roster_phase.record_path, roster_phase.evidence)
    grant = W.preparation_grant(roster_phase)
    assert isinstance(grant, A.PreparationGrant) and grant.readers == guards.PREPARATION_READERS
    assert grant.admits_roster(W.green_world().world.admissible) is False
    for reader in guards.PREPARATION_READERS:
        assert guards.refuse_test_split("test", allow_test=True, admission=grant, reader=reader) == "test"


def test_a_preparation_grant_never_authorises_a_run_a_candidate_or_scoring(roster_phase):
    grant = W.preparation_grant(roster_phase)
    for reader in (None, "run_supervised", "score_roster", "campaign"):
        with pytest.raises(TestSplitRefused, match="authorises only"):
            guards.refuse_test_split("test", allow_test=True, admission=grant, reader=reader)
    roster = W.green_world().world.admissible
    with pytest.raises(TestSplitRefused, match="authorises only"):
        guards.refuse_test_roster(roster, allow_test=True, admission=grant, require_admitted_roster=True)
    with pytest.raises(TestSplitRefused, match="not the admission's sealed admissible roster"):
        guards.refuse_test_roster(
            roster, allow_test=True, admission=grant, require_admitted_roster=True, reader="build_roster"
        )
    with pytest.raises(TestSplitRefused, match="needs allow_test"):
        guards.refuse_test_split("test", allow_test=False, admission=grant, reader="build_roster")


def test_preparation_refuses_a_fixture_sized_or_other_non_s_roster_selection(roster_phase):
    record = copy.deepcopy(roster_phase.record)
    record["protocol"]["decisions"]["roster_selection"]["value"]["n"] = 12
    W.save_record(roster_phase, record)
    W.write_preparation_approval(roster_phase)
    with pytest.raises(PermissionError, match="roster S.*1024"):
        W.preparation_grant(roster_phase)


def test_preparation_grant_refuses_a_non_s_roster_before_opening_the_dataset(roster_phase, tmp_path):
    grant = W.preparation_grant(roster_phase)
    with pytest.raises(TestSplitRefused, match="roster S.*1024"):
        W.roster_fixtures().BUILD.build_roster(
            tmp_path / "must-not-be-opened",
            split="test",
            n_expected=12,
            salt=roster_phase.record["protocol"]["decisions"]["roster_selection"]["value"]["salt"],
            allow_test=True,
            admission=grant,
        )


def test_preparation_grant_refuses_a_different_cell_before_opening_the_dataset(roster_phase, tmp_path):
    grant = W.preparation_grant(roster_phase)
    with pytest.raises(TestSplitRefused, match="sealed evaluation cell"):
        W.roster_fixtures().BUILD.build_roster(
            tmp_path / "must-not-be-opened",
            split="test",
            n_expected=1024,
            salt=roster_phase.record["protocol"]["decisions"]["roster_selection"]["value"]["salt"],
            cost_model="slope_uphill",
            allow_test=True,
            admission=grant,
        )


@pytest.mark.parametrize(
    "module_name, args",
    [
        (
            "passage_eval.protocol.block_census",
            ["--roster", "must-not-load.json", "--roster-sha256", "0" * 64, "--allow-test-split"],
        ),
        (
            "passage_eval.protocol.reference_admission",
            [
                "--roster",
                "must-not-load.json",
                "--roster-sha256",
                "0" * 64,
                "--dataset-root",
                "must-not-open",
                "--reference-policy",
                "released_labels",
                "--reference-max-expansions",
                "9",
                "--output-dir",
                "must-not-create",
                "--allow-test-split",
            ],
        ),
    ],
)
def test_preparation_clis_validate_the_grant_before_loading_a_roster(monkeypatch, module_name, args):
    module = __import__(module_name, fromlist=["main"])
    monkeypatch.setattr(module.roster_io, "load_roster", lambda *_args, **_kwargs: pytest.fail("roster was loaded"))
    with pytest.raises(TestSplitRefused, match="verified test admission record"):
        module.main(args)


def test_the_approval_must_name_the_reader(tmp_path):
    world = W.preparation_world(tmp_path / "census-only", readers=["block_census"])
    grant = W.preparation_grant(world)
    assert guards.refuse_test_split("test", allow_test=True, admission=grant, reader="block_census") == "test"
    with pytest.raises(TestSplitRefused, match="does not name 'build_roster'"):
        guards.refuse_test_split("test", allow_test=True, admission=grant, reader="build_roster")


@pytest.mark.parametrize(
    "edit, problem",
    [
        (lambda r: r["evidence"].update(native_chain_rehearsal=None), "G2: rehearsal_receipt_missing"),
        (lambda r: r["protocol"]["decisions"]["seeds"].update(status="required"), "G1: decision_not_sealed"),
        (lambda r: r["protocol"]["decisions"]["population"].update(status="proposed_not_sealed"), "G1"),
        (lambda r: r.update(status="draft"), "G1: record_not_sealed"),
    ],
)
def test_any_red_reason_other_than_the_artefacts_refuses_the_preparation(roster_phase, edit, problem):
    record = copy.deepcopy(roster_phase.record)
    edit(record)
    W.save_record(roster_phase, record)
    W.write_preparation_approval(roster_phase)
    with pytest.raises(PermissionError, match=f"preparation .* is refused .*{problem}"):
        W.preparation_grant(roster_phase)
    assert A.preparation_problems(check(roster_phase), roster_phase.record)


def test_a_present_but_broken_artefact_refuses_the_preparation(tmp_path):
    world = W.preparation_world(tmp_path / "warmup-phase", phase="warmup")
    assert red_gates(check(world)) == ["G5"]
    record = copy.deepcopy(world.record)
    record["evidence"]["references"]["chain"] = {"path": "reference-admission-chain.json", "sha256": "0" * 64}
    W.save_record(world, record)
    W.write_preparation_approval(world)
    with pytest.raises(PermissionError, match="G4: chain_unverifiable"):
        W.preparation_grant(world)


@pytest.mark.parametrize(
    "changes, problem",
    [
        ({"scope": "run"}, "scope must be"),
        ({"readers": ["build_roster", "run_supervised"]}, "readers must be"),
        ({"readers": []}, "readers must be"),
        ({"admission_record_sha256": "0" * 64}, "names another admission record"),
        ({"approver": " "}, "approver must be named"),
        ({"statement": None}, "statement"),
        ({"extra": 1}, "approval keys"),
        ({"schema": "other"}, "is not a passage-test-preparation-approval/v1"),
    ],
)
def test_an_invalid_preparation_approval_is_refused(roster_phase, changes, problem):
    W.write_preparation_approval(roster_phase, **changes)
    with pytest.raises(PermissionError, match=problem):
        W.preparation_grant(roster_phase)


@pytest.mark.parametrize(
    "timing, problem",
    [
        ({"hours_ago": -1.0}, "dated in the future"),
        ({"hours_ago": 2.0, "hours_left": -1.0}, "has expired"),
        ({"hours_ago": 1.0, "hours_left": 72.0}, "at most 3 days"),
    ],
)
def test_the_preparation_is_time_boxed(roster_phase, timing, problem):
    W.write_preparation_approval(roster_phase, **timing)
    with pytest.raises(PermissionError, match=problem):
        W.preparation_grant(roster_phase)


def test_a_preparation_grant_is_bound_by_hash_and_expires(roster_phase, monkeypatch):
    import datetime as dt

    grant = W.preparation_grant(roster_phase)
    grant.ensure_unchanged()
    later = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=25)
    monkeypatch.setattr(A, "_now", lambda: later)
    with pytest.raises(TestSplitRefused, match="expired"):
        guards.refuse_test_split("test", allow_test=True, admission=grant, reader="build_roster")
    monkeypatch.undo()
    approval = json.loads(roster_phase.approval_path.read_text())
    roster_phase.approval_path.write_text(json.dumps(dict(approval, statement="edited")))
    with pytest.raises(TestSplitRefused, match="sha256|SHA-256|differs"):
        guards.refuse_test_split("test", allow_test=True, admission=grant, reader="build_roster")
    roster_phase.approval_path.write_text(json.dumps(approval, indent=1, sort_keys=True))
    fresh = W.preparation_grant(roster_phase)
    record = json.loads(roster_phase.record_path.read_text())
    roster_phase.record_path.write_text(json.dumps(dict(record, notes="edited after the approval")))
    with pytest.raises(TestSplitRefused, match="changed after the preparation was approved"):
        guards.refuse_test_split("test", allow_test=True, admission=fresh, reader="build_roster")


def test_a_preparation_grant_is_not_forgeable():
    with pytest.raises(TypeError, match="only by grant_test_preparation"):
        A.PreparationGrant(object(), Path("r"), Path("e"), Path("a"), "0" * 64, None, ("build_roster",), None)


# ------------------------------------------------------------------- the preparation readers on a synthetic test


def test_a_preparation_grant_cannot_use_the_synthetic_fixture_as_a_test_roster(roster_phase):
    grant = W.preparation_grant(roster_phase)
    base = W.green_world().world
    TR = base.TR
    with pytest.raises(TestSplitRefused, match="roster S"):
        TR.BUILD.build_roster(
            base.dataset,
            benchmark_id=TR.BENCHMARK,
            resolution=3,
            n_expected=11,
            salt=W.BLOCK_SALT,
            split="test",
            allow_test=True,
            admission=grant,
        )


def test_the_commands_take_a_preparation_approval(roster_phase, tmp_path):
    from passage_eval.protocol import block_census as C

    base = W.green_world().world
    path = tmp_path / "source.json"
    path.write_text(json.dumps(base.source))
    args = ["--roster", str(path), "--roster-sha256", base.source["roster_sha256"], "--output", str(tmp_path / "c")]
    record = ["--allow-test-split", "--test-admission-record", str(roster_phase.record_path)]
    record += ["--test-admission-evidence-dir", str(roster_phase.evidence)]
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        C.main([*args, *record])
    with pytest.raises(TestSplitRefused, match="roster S"):
        C.main([*args, *record, "--test-preparation-approval", str(roster_phase.approval_path)])
    with pytest.raises(TestSplitRefused, match="only together with --allow-test-split"):
        guards.test_split_allowed(False, None, preparation_approval=roster_phase.approval_path)
