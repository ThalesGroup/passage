"""``passage-reference-admission/v1``: policy, inventory, exclusions, halts and the admissible roster.

The dataset is a synthetic pinned validation snapshot of eight 3x3 rows built here (row builder from
``test_roster.py``). Flat crops make the analytic optimum the diagonal, of cost 2 * 30 * sqrt(2). Row 2
stores a legal detour with its exact cost (a suboptimal label); row 5 stores the diagonal with a rounded cost
(a cost disagreement). A second snapshot adds labels that the execution loader refuses: a disconnected route
(row 3), a route through the obstacle at (0, 2) (row 6), a negative cost (row 4) and a missing cost (row 7).
Nothing here reads a released split.
"""

import copy
import functools
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from passage_eval.io import roster as R
from passage_eval.protocol import reference_admission as A
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("roster_fixtures_for_admission_tests", HERE / "test_roster.py")
TR = importlib.util.module_from_spec(spec)
spec.loader.exec_module(TR)

DIAGONAL = [[0, 0], [1, 1], [2, 2]]
DETOUR = [[0, 0], [0, 1], [1, 2], [2, 2]]
EXACT = 2 * 30 * math.sqrt(2)
DETOUR_COST = 30 + 30 * math.sqrt(2) + 30
SUBOPTIMAL, DISAGREEING = "0000000002", "0000000005"
BENCHMARK = "synthetic-validation-v1"
# Labels that scene_from_row refuses: (route, stored obstacle-variant cost) by sample id.
DEFECTS = {
    "0000000003": ([[0, 0], [2, 2]], EXACT),  # disconnected
    "0000000004": (DIAGONAL, -1.0),  # nonpositive cost
    "0000000006": ([[0, 0], [0, 1], [0, 2], [1, 2], [2, 2]], EXACT),  # crosses the obstacle at (0, 2)
    "0000000007": (DIAGONAL, None),  # missing cost
}


def _row(i, defects=None):
    row = TR._row(i)
    sample_id = f"{i:010d}"
    route = DETOUR if sample_id == SUBOPTIMAL else DIAGONAL
    cost = DETOUR_COST if sample_id == SUBOPTIMAL else 84.85 if sample_id == DISAGREEING else EXACT
    metadata = copy.deepcopy(row["metadata"])
    metadata["split"] = "validation"
    metadata["paths"] = {"astar_elevation": {"obstacles": {"cost": cost}, "free": {"cost": cost}}}
    row.update(metadata=metadata, path_astar_elevation_obstacles=route, path_astar_elevation_free=route)
    if defects and sample_id in defects:
        route, cost = defects[sample_id]
        row["path_astar_elevation_obstacles"] = route
        metadata["paths"]["astar_elevation"]["obstacles"]["cost"] = cost
    return row


def write_snapshot(root, defects=None):
    shard_dir = root / "3x3" / "validation"
    shard_dir.mkdir(parents=True)
    listed = []
    for number, members in enumerate((range(0, 4), range(4, 8))):
        path = shard_dir / f"data-3-{number:04d}.parquet"
        rows = [_row(i, defects) for i in members]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
        data = path.read_bytes()
        listed.append(
            {"path": f"3x3/validation/{path.name}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    receipt = {"schema": R.POINTER_SCHEMA, "repo_id": "test/passage", "revision": "1" * 40, "tag": "v", "files": listed}
    (root / "PINNED.json").write_text(json.dumps(receipt, sort_keys=True))
    return root


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = write_snapshot(tmp_path_factory.mktemp("admission") / "dataset")
    roster, _ = TR.BUILD.build_roster(root, benchmark_id=BENCHMARK, resolution=3, split="validation", n_expected=8)
    return root, roster


@pytest.fixture(scope="module")
def defective(tmp_path_factory):
    root = write_snapshot(tmp_path_factory.mktemp("label-defects") / "dataset", DEFECTS)
    roster, _ = TR.BUILD.build_roster(root, benchmark_id=BENCHMARK, resolution=3, split="validation", n_expected=8)
    return root, roster


def loader(root):
    return functools.partial(A.load_admission_scenes, dataset_path=root)


def released(roster, **overrides):
    values = dict(reference_policy="released_labels", reference_max_expansions=9)
    values.update(overrides)
    return A.admission_policy(roster, **values)


def independent(roster, **overrides):
    values = dict(reference_cohort_id="synthetic-c", exclusion_caps=A.PROPOSED_EXCLUSION_CAPS)
    values.update(overrides)
    return released(roster, reference_policy="independent_analytic_reference", **values)


@pytest.fixture(scope="module")
def inventory(world):
    root, roster = world
    policy = released(roster)
    return policy, A.build_reference_inventory(roster, loader(root), policy)


# ------------------------------------------------------------------------------------------------------ policy


def test_the_policy_has_no_default_and_binds_the_roster_cohort_limit_and_table(world):
    _, roster = world
    with pytest.raises(TypeError):
        A.admission_policy(roster, reference_max_expansions=9)  # no default reference policy
    with pytest.raises(ValueError, match="no default"):
        A.admission_policy(roster, reference_policy="stored_labels", reference_max_expansions=9)
    policy = released(roster)
    assert policy["source_roster_sha256"] == roster["roster_sha256"] and policy["reference_cohort_id"] is None
    assert policy["category_table_status"] == "proposed_not_sealed" and policy["candidate_outputs_used"] is False
    assert policy["minimum_reference_max_expansions"] == 9
    body = {key: value for key, value in policy.items() if key != "policy_sha256"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert policy["policy_sha256"] == hashlib.sha256(canonical).hexdigest()
    assert A.validate_policy(policy) == policy
    assert policy["exclusion_caps"] is None and policy["execution_loader_labels"] == "validated"
    assert policy["second_reference"] == {
        "implementation": "passage-dijkstra8/v1",
        "coverage": "every task whose A* search ended with an exact optimum or no path",
        "rtol": 1e-7,
        "atol": 1e-6,
    }
    assert policy["label_tolerance"] == {"rtol": 1e-7, "atol": 1e-6}
    analytic = independent(roster, reference_cohort_id="c1")
    assert analytic["policy_sha256"] != policy["policy_sha256"]
    assert analytic["exclusion_caps"] == A.PROPOSED_EXCLUSION_CAPS and analytic["execution_loader_labels"] == "tolerant"
    assert A.PROPOSED_EXCLUSION_CAPS["status"] == "proposed_not_sealed"
    assert (A.PROPOSED_EXCLUSION_CAPS["max_total_fraction"], A.PROPOSED_EXCLUSION_CAPS["max_block_fraction"]) == (
        0.005,
        0.05,
    )
    for field, value in (
        ("execution_loader_labels", "validated"),
        ("second_reference", None),
        ("label_tolerance", {"rtol": 1e-6, "atol": 1e-6}),
        ("exclusion_caps", None),
    ):
        forged = dict(analytic, **{field: value})
        forged["policy_sha256"] = A._digest(forged, "policy_sha256")
        with pytest.raises(ValueError, match="inconsistent"):
            A.validate_policy(forged)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"exclusion_caps": None}, "needs exclusion caps"),
        ({"exclusion_caps": {**A.PROPOSED_EXCLUSION_CAPS, "max_total_fraction": 1.0}}, "max_total_fraction"),
        ({"exclusion_caps": {**A.PROPOSED_EXCLUSION_CAPS, "min_block_exclusions": 0}}, "min_block_exclusions"),
        ({"exclusion_caps": {**A.PROPOSED_EXCLUSION_CAPS, "extra": 1}}, "exactly schema, status"),
        ({"exclusion_caps": {**A.PROPOSED_EXCLUSION_CAPS, "status": " "}}, "sealing status"),
        (
            {
                "category_table": {
                    **A.PROPOSED_CATEGORY_TABLE,
                    "reasons": {
                        **A.PROPOSED_CATEGORY_TABLE["reasons"],
                        "independent_reference_budget_exhausted": "excluded",
                    },
                }
            },
            "exhausted search halts",
        ),
    ],
)
def test_the_independent_policy_needs_caps_and_halts_on_an_exhausted_search(world, overrides, match):
    with pytest.raises(ValueError, match=match):
        independent(world[1], **overrides)


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"reference_max_expansions": 8}, "at least N\\*N = 9"),
        ({"reference_max_expansions": True}, "at least N\\*N"),
        ({"reference_max_expansions": 9.5}, "at least N\\*N"),
        ({"reference_policy": "independent_analytic_reference"}, "explicit reference_cohort_id"),
        ({"reference_policy": "independent_analytic_reference", "reference_cohort_id": " "}, "reference_cohort_id"),
        ({"reference_cohort_id": "c1"}, "does not use a reference cohort"),
    ],
)
def test_an_unsafe_limit_or_a_mismatched_cohort_is_refused(world, overrides, match):
    with pytest.raises(ValueError, match=match):
        released(world[1], **overrides)


def _table(**changes):
    table = copy.deepcopy(A.PROPOSED_CATEGORY_TABLE)
    table["reasons"].update(changes)
    return table


@pytest.mark.parametrize(
    "table, match",
    [
        ({**A.PROPOSED_CATEGORY_TABLE, "schema": "other"}, "category table is required"),
        (_table(candidate_timeout="excluded"), "exactly the qualification reasons"),
        (_table(stored_path_suboptimal="admissible"), "only be excluded or halt"),
        ({**A.PROPOSED_CATEGORY_TABLE, "reasons": {"stored_path_illegal": "excluded"}}, "exactly the qualification"),
        ({**A.PROPOSED_CATEGORY_TABLE, "extra": 1}, "exactly schema, status, reasons"),
        ({**A.PROPOSED_CATEGORY_TABLE, "status": ""}, "sealing status"),
    ],
)
def test_the_category_table_maps_exactly_the_reference_reasons_to_excluded_or_halt(world, table, match):
    with pytest.raises(ValueError, match=match):
        released(world[1], category_table=table)


def test_a_supplied_table_and_its_status_are_part_of_the_hashed_policy(world):
    sealed = dict(_table(stored_cost_disagreement="halt"), status="sealed by the author")
    policy = released(world[1], category_table=sealed)
    assert policy["category_table"] == sealed and policy["category_table_status"] == "sealed by the author"
    assert policy["policy_sha256"] != released(world[1])["policy_sha256"]
    edited = copy.deepcopy(policy)
    edited["category_table"]["reasons"]["stored_cost_disagreement"] = "excluded"
    with pytest.raises(ValueError, match="policy_sha256"):
        A.validate_policy(edited)


# --------------------------------------------------------------------------------------------------- inventory


def test_non_qualifiable_references_are_excluded_with_their_reason_and_counted_in_the_source(inventory, world):
    from passage.audit import qualified_reference_for_scene

    root, roster = world
    policy, result = inventory
    assert result["outcome"] == "complete" and result["candidate_outputs_read"] is False
    assert result["counts"] == {"n_source": 8, "n_admissible": 6, "n_excluded": 2, "n_halt": 0}
    assert [task["sample_id"] for task in result["tasks"]] == [task["sample_id"] for task in roster["tasks"]]
    excluded = {item["sample_id"]: item["reason"] for item in result["exclusions"]}
    assert excluded == {SUBOPTIMAL: "stored_path_suboptimal", DISAGREEING: "stored_cost_disagreement"}
    assert result["excluded_reason_counts"] == {"stored_cost_disagreement": 1, "stored_path_suboptimal": 1}
    assert result["policy"] == policy and result["policy_sha256"] == policy["policy_sha256"]
    scenes = {scene.sample_id: scene for scene in R.load_roster_scenes(roster, root)}
    for task in result["tasks"]:
        expected = qualified_reference_for_scene(scenes[task["sample_id"]], max_expansions=9)
        assert task["status"] == "exact" and task["rank"] == next(
            t["rank"] for t in roster["tasks"] if t["sample_id"] == task["sample_id"]
        )
        if task["category"] == "admissible":
            assert task["reference_id"] == expected["reference_id"] and task["qualified"] is True
        else:
            assert task["reference_id"] is None and task["qualified"] is False
    assert result["inventory_sha256"] == A.validate_inventory(result, roster)["inventory_sha256"]


def test_the_admissible_roster_keeps_source_ranks_and_blocks_and_never_substitutes(inventory, world):
    _, roster = world
    policy, result = inventory
    admissible = A.admissible_roster(roster, result)
    source_rank = {task["sample_id"]: task["rank"] for task in roster["tasks"]}
    source_block = {task["sample_id"]: task["geographic_block"] for task in roster["tasks"]}
    ids = [task["sample_id"] for task in admissible["tasks"]]
    assert ids == [t["sample_id"] for t in roster["tasks"] if t["sample_id"] not in {SUBOPTIMAL, DISAGREEING}]
    assert admissible["selection"]["n_expected"] == 6 and len(ids) == 6
    assert all(task["parent_rank"] == source_rank[task["sample_id"]] for task in admissible["tasks"])
    assert R.roster_block_table(admissible)["blocks"] == {sample_id: source_block[sample_id] for sample_id in ids}
    derivation = admissible["selection"]["derivation"]
    assert derivation["inventory_sha256"] == result["inventory_sha256"]
    assert derivation["policy_sha256"] == policy["policy_sha256"]
    assert derivation["parent_roster_sha256"] == roster["roster_sha256"]
    assert (derivation["n_source"], derivation["n_excluded"], derivation["n_admissible"]) == (8, 2, 6)
    chain = A.admission_chain(roster, policy, result, admissible)
    assert chain["admissible_roster_sha256"] == admissible["roster_sha256"]
    assert chain["inventory_sha256"] == result["inventory_sha256"] and chain["policy_sha256"] == policy["policy_sha256"]
    assert chain["counts"] == result["counts"] and chain["category_table_status"] == "proposed_not_sealed"


def test_the_chain_refuses_another_admissible_roster_policy_or_edited_inventory(inventory, world):
    _, roster = world
    policy, result = inventory
    admissible = A.admissible_roster(roster, result)
    smaller = R.derive_roster(roster, [t["sample_id"] for t in admissible["tasks"]][:-1], rule=A.ADMISSIBLE_RULE)
    with pytest.raises(ValueError, match="not the one derived"):
        A.admission_chain(roster, policy, result, smaller)
    with pytest.raises(ValueError, match="another admission policy"):
        A.admission_chain(roster, released(roster, reference_max_expansions=10), result, admissible)
    edited = copy.deepcopy(result)
    next(t for t in edited["tasks"] if t["sample_id"] == SUBOPTIMAL)["category"] = "admissible"
    edited["inventory_sha256"] = A._digest(edited, "inventory_sha256")
    with pytest.raises(ValueError, match="does not follow"):
        A.validate_inventory(edited, roster)
    recounted = copy.deepcopy(result)
    recounted["counts"]["n_excluded"] += 1
    recounted["inventory_sha256"] = A._digest(recounted, "inventory_sha256")
    with pytest.raises(ValueError, match="counts or outcome"):
        A.validate_inventory(recounted, roster)
    with pytest.raises(ValueError, match="inventory_sha256"):
        A.validate_inventory(dict(result, outcome="halt"), roster)
    other = R.derive_roster(roster, [roster["tasks"][0]["sample_id"]], rule="other source")
    with pytest.raises(ValueError, match="another source roster"):
        A.build_reference_inventory(other, lambda r: [], policy)


def test_the_independent_policy_admits_analytic_references_and_keeps_label_findings(world):
    root, roster = world
    policy = independent(roster)
    result = A.build_reference_inventory(roster, loader(root), policy)
    assert result["counts"] == {"n_source": 8, "n_admissible": 8, "n_excluded": 0, "n_halt": 0}
    findings = {task["sample_id"]: task["source_label_reasons"] for task in result["tasks"]}
    assert findings[SUBOPTIMAL] == ["stored_path_suboptimal"] and findings[DISAGREEING] == ["stored_cost_disagreement"]
    published = result["label_findings"]
    assert published["tolerance"] == {"rtol": 1e-7, "atol": 1e-6} and published["n_tasks_with_findings"] == 2
    assert published["reason_counts"] == {"stored_cost_disagreement": 1, "stored_path_suboptimal": 1}
    assert {task["sample_id"] for task in published["tasks"]} == {SUBOPTIMAL, DISAGREEING}
    assert result["exclusion_caps_check"]["within_caps"] is True and result["exclusion_caps_check"]["n_excluded"] == 0
    assert all(task["second_reference"]["status"] == "agrees" for task in result["tasks"])
    assert A.admissible_roster(roster, result)["selection"]["n_expected"] == 8


def _halting(world, transform):
    root, roster = world

    def load(payload):
        return transform(R.load_roster_scenes(payload, root))

    return A.build_reference_inventory(roster, load, released(roster))


def test_input_inconsistencies_halt_with_a_reason_per_task(world):
    from dataclasses import replace

    first, second = (task["sample_id"] for task in world[1]["tasks"][:2])

    def changed(scenes):
        return [replace(scenes[0], elevation=scenes[0].elevation + 1.0), *scenes[1:]]

    def invalid(scenes):  # an obstacle grid that disagrees with the input tensor is not even a valid request
        obstacles = scenes[0].obstacles.copy()
        obstacles[1, 0] = True
        return [replace(scenes[0], obstacles=obstacles), *scenes[1:]]

    cases = {
        "input_identity_mismatch": changed,
        "invalid_request": invalid,
        "duplicate_sample_id": lambda scenes: [*scenes, scenes[0]],
        "scene_missing": lambda scenes: scenes[1:],
    }
    for reason, transform in cases.items():
        result = _halting(world, transform)
        assert result["outcome"] == "halt" and result["counts"]["n_halt"] == 1, reason
        assert A.validate_inventory(result, world[1]) == result, reason
        expected = "input_identity_mismatch" if reason == "invalid_request" else reason
        assert result["halts"][0]["sample_id"] == first and result["halts"][0]["reason"] == expected
        with pytest.raises(A.ReferenceAdmissionHalt, match="halted"):
            A.admissible_roster(world[1], result)
    foreign = _halting(world, lambda scenes: [*scenes, replace(scenes[1], sample_id="foreign")])
    assert foreign["outcome"] == "halt" and foreign["foreign_scene_ids"] == ["foreign"]
    assert foreign["counts"]["n_halt"] == 0 and second not in foreign["foreign_scene_ids"]

    def unreadable(scenes):
        raise ValueError("unreadable shard")

    with pytest.raises(A.ReferenceAdmissionHalt, match="scene loading failed.*unreadable shard"):
        _halting(world, unreadable)


def test_undeterminable_references_unknown_reasons_and_qualifier_errors_halt(world, monkeypatch):
    import passage.audit as audit

    original = audit.qualified_reference_for_scene
    target = world[1]["tasks"][0]["sample_id"]
    outcomes = {
        "budget": (
            {"qualified": False, "qualification_reasons": ["reference_budget_exhausted"]},
            "reference_budget_exhausted",
        ),
        "unknown": ({"qualified": False, "qualification_reasons": ["new_reason"]}, "new_reason"),
        "unknown after a known halt": (
            {"qualified": False, "qualification_reasons": ["reference_budget_exhausted", "new_reason"]},
            "reference_budget_exhausted",
        ),
        "a reason named like a halt of this module": (
            {"qualified": False, "qualification_reasons": ["scene_missing"]},
            "inconsistent_qualification_record",
        ),
        "empty": ({"qualified": False, "qualification_reasons": []}, "inconsistent_qualification_record"),
        "contradiction": ({"qualified": True, "qualification_reasons": ["x"]}, "inconsistent_qualification_record"),
        "error": (None, "qualification_error"),
    }
    for name, (record, reason) in outcomes.items():

        def fake(scene, *, max_expansions, _record=record):
            if scene.sample_id != target:
                return original(scene, max_expansions=max_expansions)
            if _record is None:
                raise ValueError("reference endpoints must be distinct")
            return dict(_record, reference_status="budget_exhausted")

        monkeypatch.setattr(audit, "qualified_reference_for_scene", fake)
        result = A.build_reference_inventory(world[1], loader(world[0]), released(world[1]))
        assert result["outcome"] == "halt" and result["halts"][0]["sample_id"] == target, name
        assert result["halts"][0]["reason"] == reason, name
        assert A.validate_inventory(result, world[1]) == result, name
        if name == "unknown":
            assert result["halts"][0]["reasons"] == ["new_reason", "unknown_reason"]
        if name == "unknown after a known halt":
            assert result["halts"][0]["reasons"] == ["reference_budget_exhausted", "new_reason", "unknown_reason"]
        if name in {"contradiction", "a reason named like a halt of this module"}:
            assert result["halts"][0]["reasons"] == ["inconsistent_qualification_record"]
            assert result["tasks"][0]["qualifier_reasons"] == record["qualification_reasons"]
    monkeypatch.setattr(audit, "qualified_reference_for_scene", original)


def test_the_category_table_decides_between_exclusion_and_halt(world):
    root, roster = world
    policy = released(roster, category_table=_table(stored_path_suboptimal="halt"))
    result = A.build_reference_inventory(roster, loader(root), policy)
    assert result["outcome"] == "halt" and [item["sample_id"] for item in result["halts"]] == [SUBOPTIMAL]
    assert [item["sample_id"] for item in result["exclusions"]] == [DISAGREEING]


def test_unreadable_inputs_and_unexpected_loader_items_halt(world):
    first = world[1]["tasks"][0]["sample_id"]

    def unreadable(scenes):
        return [A.AdmissionScene(first, None, "ValueError: corrupt tensor"), *scenes[1:]]

    result = _halting(world, unreadable)
    assert result["outcome"] == "halt" and result["halts"][0]["reason"] == "scene_unreadable"
    assert result["tasks"][0]["execution_loader_error"] == "ValueError: corrupt tensor"
    assert A.validate_inventory(result, world[1]) == result
    with pytest.raises(A.ReferenceAdmissionHalt, match="not a scene"):
        _halting(world, lambda scenes: [*scenes[1:], {"sample_id": first}])


# ----------------------------------------------------------------------------------------- forged inventories


def _reseal(inventory):
    inventory["inventory_sha256"] = A._digest(inventory, "inventory_sha256")
    return inventory


def _recount(inventory):
    counts = Counter(task["category"] for task in inventory["tasks"])
    inventory["counts"] = {
        "n_source": len(inventory["tasks"]),
        "n_admissible": counts["admissible"],
        "n_excluded": counts["excluded"],
        "n_halt": counts["halt"],
    }
    return inventory


def test_a_candidate_outcome_cannot_become_an_exclusion_even_in_a_resealed_inventory(inventory, world):
    _, roster = world
    policy, result = inventory
    forged = copy.deepcopy(result)
    victim = next(task for task in forged["tasks"] if task["category"] == "admissible")
    victim.update(
        category="excluded",
        reason="candidate_timeout",
        reasons=["candidate_timeout"],
        reference_id=None,
        qualified=False,
    )
    stale = _reseal(_recount(copy.deepcopy(forged)))  # counts updated, exclusion tables left stale
    fresh = copy.deepcopy(stale)  # the human-facing tables recomputed too
    fresh["exclusions"] = A._listing(fresh["tasks"], "excluded")
    fresh["excluded_reason_counts"] = A._excluded_reason_counts(fresh["tasks"])
    for candidate in (stale, _reseal(fresh)):
        with pytest.raises(ValueError, match="candidate_timeout.*does not follow"):
            A.validate_inventory(candidate, roster)
        with pytest.raises(ValueError, match="does not follow"):
            A.admissible_roster(roster, candidate)
    admissible = A.admissible_roster(roster, result)
    with pytest.raises(ValueError, match="does not follow"):
        A.admission_chain(roster, policy, fresh, admissible)


def _forged(result, change):
    forged = copy.deepcopy(result)
    change(forged)
    return _reseal(forged)


def _task(inventory, sample_id):
    return next(task for task in inventory["tasks"] if task["sample_id"] == sample_id)


def test_validation_rederives_each_category_and_every_table_from_the_task_records(inventory, world):
    _, roster = world
    _, result = inventory
    admitted = next(task["sample_id"] for task in result["tasks"] if task["category"] == "admissible")

    def swap(forged):  # categories exchanged, counts unchanged
        _task(forged, SUBOPTIMAL)["category"], _task(forged, admitted)["category"] = "admissible", "excluded"

    def relabel(forged):  # an allowed reason, but the category table maps it to halt
        _task(forged, SUBOPTIMAL).update(reason="reference_budget_exhausted", reasons=["reference_budget_exhausted"])

    def no_reference(forged):
        _task(forged, admitted)["reference_id"] = None

    def halt_without_cause(forged):
        _task(forged, admitted).update(category="halt", reason="scene_missing", reasons=[], reference_id=None)
        forged["halts"] = A._listing(forged["tasks"], "halt")
        _recount(forged)["outcome"] = "halt"

    cases = {
        "does not follow": [swap, relabel, no_reference, halt_without_cause],
        "differs from the source roster": [
            lambda forged: _task(forged, admitted).update(rank=99),
            lambda forged: _task(forged, admitted).update(input_identity_sha256="0" * 64),
            lambda forged: _task(forged, admitted).update(geographic_block="geo5:0:0"),
        ],
        "exclusion, halt or excluded-reason tables": [
            lambda forged: forged["exclusions"].pop(),
            lambda forged: forged["excluded_reason_counts"].update(stored_path_suboptimal=2),
            lambda forged: forged["halts"].append(dict(forged["exclusions"][0])),
        ],
        "does not belong": [
            lambda forged: forged.update(foreign_scene_ids=[admitted]),
            lambda forged: forged.update(candidate_outputs_read=True),
            lambda forged: forged.update(source_block_table_sha256="0" * 64),
        ],
    }
    for match, changes in cases.items():
        for change in changes:
            with pytest.raises(ValueError, match=match):
                A.validate_inventory(_forged(result, change), roster)
    assert A.validate_inventory(result, roster) == result


def test_a_policy_whose_minimum_was_lowered_is_refused_for_this_roster(inventory, world):
    root, roster = world
    policy, result = inventory
    weak = copy.deepcopy(policy)
    weak.update(reference_max_expansions=2, minimum_reference_max_expansions=1)
    weak["policy_sha256"] = A._digest(weak, "policy_sha256")
    assert A.validate_policy(weak) == weak  # self-consistent, but not for a 3x3 roster
    with pytest.raises(ValueError, match="another source roster"):
        A.build_reference_inventory(roster, loader(root), weak)
    forged = _forged(result, lambda inventory: inventory.update(policy=weak, policy_sha256=weak["policy_sha256"]))
    with pytest.raises(ValueError, match="another source roster"):
        A.validate_inventory(forged, roster)


# ------------------------------------------------------------------------------------------------ label defects


def test_label_defects_are_per_task_exclusions_under_the_released_label_policy(defective):
    from passage.audit import qualified_reference_for_scene

    root, roster = defective
    policy = released(roster)
    result = A.build_reference_inventory(roster, loader(root), policy)
    assert result["outcome"] == "complete"
    assert result["counts"] == {"n_source": 8, "n_admissible": 2, "n_excluded": 6, "n_halt": 0}
    reasons = {task["sample_id"]: (task["reason"], task["reasons"]) for task in result["tasks"]}
    for illegal in ("0000000003", "0000000006"):
        assert reasons[illegal][0] == "stored_path_illegal"
        assert "stored_path_optimum_unassessable" in reasons[illegal][1]
    assert reasons["0000000004"] == ("stored_cost_missing_or_invalid", ["stored_cost_missing_or_invalid"])
    assert reasons["0000000007"] == ("stored_cost_missing_or_invalid", ["stored_cost_missing_or_invalid"])
    assert reasons[SUBOPTIMAL][0] == "stored_path_suboptimal" and reasons[DISAGREEING][0] == "stored_cost_disagreement"
    loader_errors = {sample_id: _task(result, sample_id).get("execution_loader_error") for sample_id in reasons}
    assert "released oracle is not a valid path" in loader_errors["0000000003"]
    assert "released oracle is not a valid path" in loader_errors["0000000006"]
    assert "released oracle cost must be positive and finite" in loader_errors["0000000004"]
    assert "PASSAGE sample 0000000007" in loader_errors["0000000007"]
    assert {sample_id for sample_id, error in loader_errors.items() if error} == set(DEFECTS)
    assert {item["sample_id"] for item in result["exclusions"]} == {*DEFECTS, SUBOPTIMAL, DISAGREEING}
    assert A.validate_inventory(result, roster) == result
    with pytest.raises(ValueError, match="PASSAGE sample 000000000[3467]"):  # the execution loader refuses the source
        R.load_roster_scenes(roster, root)
    admissible = A.admissible_roster(roster, result)
    scenes = R.load_roster_scenes(admissible, root)  # ...but loads the admissible roster: no defective row is in it
    assert [scene.sample_id for scene in scenes] == ["0000000000", "0000000001"]
    for scene, task in zip(scenes, (_task(result, "0000000000"), _task(result, "0000000001"))):
        assert task["reference_id"] == qualified_reference_for_scene(scene, max_expansions=9)["reference_id"]
    assert A.admission_chain(roster, policy, result, admissible)["counts"]["n_excluded"] == 6


def test_the_independent_policy_admits_rows_with_defective_published_labels_as_findings(defective):
    """Judge R1/R6: a defective published label is a dataset finding; the tolerant execution loader reads its row."""
    from passage_eval import cli

    root, roster = defective
    policy = independent(roster)
    result = A.build_reference_inventory(roster, loader(root), policy)
    assert result["outcome"] == "complete" and result["halts"] == []
    assert result["counts"] == {"n_source": 8, "n_admissible": 8, "n_excluded": 0, "n_halt": 0}
    for sample_id in DEFECTS:
        task = _task(result, sample_id)
        assert task["status"] == "exact" and task["reference_id"] and task["execution_loader_error"]
        assert task["source_label_reasons"]  # the label finding is kept, it does not decide the category
    assert _task(result, "0000000003")["source_label_reasons"][0] == "stored_path_illegal"
    findings = {task["sample_id"]: task["reasons"] for task in result["label_findings"]["tasks"]}
    assert set(findings) == {*DEFECTS, SUBOPTIMAL, DISAGREEING}
    assert all("execution_loader_refuses_label" in findings[sample_id] for sample_id in DEFECTS)
    assert "execution_loader_refuses_label" not in findings[SUBOPTIMAL]
    assert A.validate_inventory(result, roster) == result
    admissible = A.admissible_roster(roster, result)
    with pytest.raises(ValueError, match="PASSAGE sample 000000000[3467]"):  # the validating loader still refuses
        R.load_roster_scenes(admissible, root)
    scenes = R.load_roster_scenes(admissible, root, tolerant_labels=True)
    assert [scene.sample_id for scene in scenes] == [task["sample_id"] for task in roster["tasks"]]
    spec = cli.BenchSpec(
        method_name="m",
        family="f",
        architecture="a",
        inference="x:Y",
        resolution=3,
        reference_policy="independent_analytic_reference",
        reference_cohort_id="synthetic-c",
        reference_max_expansions=9,
    )
    qualification = cli._qualification(scenes, spec)  # what the supervisor and score-roster compute
    assert A.check_reference_ids(qualification, A.inventory_reference_ids(result), context="t") == 8
    with pytest.raises(TypeError):
        R.load_roster_scenes(admissible, root, tolerant_labels="yes")


def test_the_released_label_policy_still_halts_a_qualified_row_the_validating_loader_refuses(defective, monkeypatch):
    from passage import audit

    root, roster = defective
    original = audit.qualified_reference_for_scene

    def qualifying(scene, **kwargs):  # pretend the released labels qualify, to reach the loader check
        record = original(scene, **kwargs)
        return dict(record, qualified=True, qualification_reasons=[])

    monkeypatch.setattr(audit, "qualified_reference_for_scene", qualifying)
    result = A.build_reference_inventory(roster, loader(root), released(roster))
    assert {item["sample_id"]: item["reason"] for item in result["halts"]} == {
        sample_id: "execution_loader_refuses_label" for sample_id in DEFECTS
    }
    assert result["outcome"] == "halt" and result["label_findings"]["n_tasks_with_findings"] == 0


def test_the_command_writes_per_task_label_findings_instead_of_stopping(defective, tmp_path):
    root, roster = defective
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))
    out = tmp_path / "released"
    assert A.main(arguments(root, path, roster["roster_sha256"], out)) == 0
    inventory = json.loads((out / "reference-admission-inventory.json").read_text())
    assert inventory["excluded_reason_counts"] == {
        "stored_cost_disagreement": 1,
        "stored_cost_missing_or_invalid": 2,
        "stored_path_illegal": 2,
        "stored_path_suboptimal": 1,
    }
    analytic = arguments(root, path, roster["roster_sha256"], tmp_path / "independent")
    analytic[analytic.index("released_labels")] = "independent_analytic_reference"
    with pytest.raises(ValueError, match="needs exclusion caps"):
        A.main([*analytic, "--reference-cohort-id", "synthetic-c"])
    caps = tmp_path / "caps.json"
    caps.write_text(json.dumps(A.PROPOSED_EXCLUSION_CAPS))
    assert A.main([*analytic, "--reference-cohort-id", "synthetic-c", "--exclusion-caps", str(caps)]) == 0
    written = json.loads((tmp_path / "independent" / "reference-admission-inventory.json").read_text())
    assert written["halts"] == [] and written["counts"]["n_admissible"] == 8
    assert sorted(task["sample_id"] for task in written["label_findings"]["tasks"]) == sorted(
        [*DEFECTS, SUBOPTIMAL, DISAGREEING]
    )
    assert (tmp_path / "independent" / "admissible-roster.json").exists()


# ------------------------------------------------------------------------------------------------ test split


def test_the_library_entry_points_refuse_a_test_roster_before_any_scene_is_loaded(tmp_path):
    synthetic_test = TR.write_dataset(tmp_path / "synthetic-test")  # a synthetic fixture whose split is named test
    roster, _ = TR.build(synthetic_test)
    calls = []

    def recording(payload):
        calls.append(payload["dataset"]["split"])
        return []

    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        released(roster)
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        released(roster, allow_test="yes")  # only the literal True lifts the guard
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        A.build_reference_inventory(roster, recording, {})
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        A.build_reference_inventory(roster, recording, {}, allow_test=1)
    with pytest.raises(TestSplitRefused, match="admission grant"):  # allow_test alone is not an admission
        released(roster, allow_test=True)
    assert calls == []
    # An all-GREEN synthetic admission grant (campaign_world.py) lifts the guard of the library functions.
    admitted = released(roster, allow_test=True, admission=TR.green_admission())
    assert admitted["dataset"]["split"] == "test"


# --------------------------------------------------------------------------------------------------- command line


def arguments(root, roster_path, sha, out, *extra):
    return [
        "--roster",
        str(roster_path),
        "--roster-sha256",
        sha,
        "--dataset-root",
        str(root),
        "--reference-policy",
        "released_labels",
        "--reference-max-expansions",
        "9",
        "--output-dir",
        str(out),
        *extra,
    ]


@pytest.fixture
def roster_file(world, tmp_path):
    path = tmp_path / "source-roster.json"
    path.write_text(json.dumps(world[1]))
    return path


def test_the_command_writes_the_chain_and_the_admissible_roster(world, roster_file, tmp_path, capsys):
    root, roster = world
    out = tmp_path / "admission"
    assert A.main(arguments(root, roster_file, roster["roster_sha256"], out)) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["counts"]["n_excluded"] == 2 and summary["category_table_status"] == "proposed_not_sealed"
    policy = json.loads((out / "reference-admission-policy.json").read_text())
    inventory = json.loads((out / "reference-admission-inventory.json").read_text())
    admissible = R.load_roster(out / "admissible-roster.json")  # also checks the .sha256 sidecar
    chain = json.loads((out / "reference-admission-chain.json").read_text())
    assert A.admission_chain(roster, policy, inventory, admissible) == chain
    assert summary["admissible_roster_sha256"] == admissible["roster_sha256"]
    with pytest.raises(FileExistsError):
        A.main(arguments(root, roster_file, roster["roster_sha256"], out))


def test_the_command_halts_without_an_admissible_roster(world, roster_file, tmp_path):
    root, roster = world
    table = tmp_path / "table.json"
    table.write_text(json.dumps(dict(_table(stored_path_suboptimal="halt"), status="sealed for this test")))
    out = tmp_path / "admission"
    assert A.main(arguments(root, roster_file, roster["roster_sha256"], out, "--category-table", str(table))) == 2
    assert {p.name for p in out.iterdir()} == {"reference-admission-policy.json", "reference-admission-inventory.json"}


def test_the_command_refuses_a_test_roster_before_reading_any_scene(tmp_path, monkeypatch):
    synthetic_test = TR.write_dataset(tmp_path / "synthetic-test")  # a synthetic fixture whose split is named test
    roster, _ = TR.build(synthetic_test)
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))

    def never(*args, **kwargs):
        raise AssertionError("a scene was loaded")

    monkeypatch.setattr(R, "load_roster_scenes", never)
    monkeypatch.setattr(R, "open_roster_rows", never)
    monkeypatch.setattr(A, "load_admission_scenes", never)
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        A.main(arguments(synthetic_test, path, roster["roster_sha256"], tmp_path / "o"))
    record = tmp_path / "admission.json"
    record.write_text("{}")
    sha = roster["roster_sha256"]
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        A.main(
            arguments(
                synthetic_test, path, sha, tmp_path / "o", "--allow-test-split", "--test-admission-record", str(record)
            )
        )
    with pytest.raises(TestSplitRefused, match="only together with --allow-test-split"):
        A.main(arguments(synthetic_test, path, sha, tmp_path / "o", "--test-admission-record", str(record)))
    with pytest.raises(TestSplitRefused, match="needs a verified test admission record"):
        A.main(arguments(synthetic_test, path, sha, tmp_path / "o", "--allow-test-split"))
    assert not (tmp_path / "o").exists()


def test_the_module_entry_point_runs_in_its_own_process(world, roster_file, tmp_path):
    root, roster = world
    out = tmp_path / "by-process"
    done = subprocess.run(
        [sys.executable, "-I", "-m", "passage_eval.protocol.reference_admission"]
        + arguments(root, roster_file, roster["roster_sha256"], out),
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout)["counts"] == {"n_source": 8, "n_admissible": 6, "n_excluded": 2, "n_halt": 0}
    without_policy = arguments(root, roster_file, roster["roster_sha256"], tmp_path / "x")
    index = without_policy.index("--reference-policy")
    del without_policy[index : index + 2]
    missing = subprocess.run(
        [sys.executable, "-I", "-m", "passage_eval.protocol.reference_admission", *without_policy],
        capture_output=True,
        text=True,
    )
    assert missing.returncode == 2 and "--reference-policy" in missing.stderr and not (tmp_path / "x").exists()


# ----------------------------------------------------------------------------- no admissible task, unverified files


def test_the_command_exits_2_without_a_traceback_when_every_task_is_excluded(defective, tmp_path, capsys):
    root, roster = defective
    probe = A.build_reference_inventory(roster, loader(root), released(roster))
    excluded = [item["sample_id"] for item in probe["exclusions"]]
    assert excluded and len(excluded) < len(roster["tasks"])
    only_excluded = R.derive_roster(roster, excluded, rule="synthetic: the tasks the released labels exclude")
    path = tmp_path / "only-excluded.json"
    path.write_text(json.dumps(only_excluded))
    out = tmp_path / "admission"
    assert A.main(arguments(root, path, only_excluded["roster_sha256"], out)) == 2
    captured = capsys.readouterr()
    assert "no source task has a qualified reference" in captured.err and "Traceback" not in captured.err
    summary = json.loads(captured.out)
    assert summary["outcome"] == "complete" and summary["counts"]["n_admissible"] == 0
    assert {p.name for p in out.iterdir()} == {"reference-admission-policy.json", "reference-admission-inventory.json"}


def test_skipped_file_hashes_are_recorded_and_refused_for_any_sealed_use(world, roster_file, tmp_path, capsys):
    root, roster = world
    policy = released(roster)
    unverified = A.build_reference_inventory(
        roster, functools.partial(A.load_admission_scenes, dataset_path=root, verify_file_hashes=False), policy
    )
    assert unverified["source_file_hashes"] == "skipped" and unverified["outcome"] == "complete"
    assert A.validate_inventory(unverified, roster) == unverified  # consistent, but not sealable
    with pytest.raises(A.ReferenceAdmissionHalt, match="without re-hashing the source files"):
        A.admissible_roster(roster, unverified)
    verified = A.build_reference_inventory(roster, loader(root), policy)
    assert verified["source_file_hashes"] == "verified"
    assert unverified["inventory_sha256"] != verified["inventory_sha256"]
    forged = copy.deepcopy(unverified)
    forged["source_file_hashes"] = "verified"  # without resealing, the digest objects
    with pytest.raises(ValueError, match="inventory_sha256"):
        A.validate_inventory(forged, roster)
    unknown = copy.deepcopy(unverified)
    unknown["source_file_hashes"] = "partly"
    unknown["inventory_sha256"] = A._digest(unknown, "inventory_sha256")
    with pytest.raises(ValueError, match="does not belong"):
        A.validate_inventory(unknown, roster)
    admissible = A.admissible_roster(roster, verified)
    assert admissible["selection"]["derivation"]["source_file_hashes"] == "verified"
    chain = A.admission_chain(roster, policy, verified, admissible)
    assert chain["source_file_hashes"] == "verified"
    with pytest.raises(A.ReferenceAdmissionHalt, match="diagnostic only"):
        A.admission_chain(roster, policy, unverified, admissible)
    out = tmp_path / "skipped"
    assert A.main(arguments(root, roster_file, roster["roster_sha256"], out, "--skip-file-hashes")) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out)["source_file_hashes"] == "skipped" and "Traceback" not in captured.err
    assert {p.name for p in out.iterdir()} == {"reference-admission-policy.json", "reference-admission-inventory.json"}


# ------------------------------------------------------------------- judge R1/R6: second implementation and caps


def _forging_qualifier(monkeypatch, change):
    """Wrap the A* qualifier so that ``change(record)`` edits its outcome before the inventory sees it."""
    from passage import audit

    original = audit.qualified_reference_for_scene

    def forged(scene, **kwargs):
        record = original(scene, **kwargs)
        return change(scene, dict(record)) or record

    monkeypatch.setattr(audit, "qualified_reference_for_scene", forged)


def test_a_reference_cost_the_second_implementation_does_not_confirm_halts(world, monkeypatch):
    root, roster = world

    def inflate(scene, record):
        if scene.sample_id == "0000000000":
            return dict(record, optimal_cost=record["optimal_cost"] * (1 + 1e-6))  # beyond rtol 1e-7, atol 1e-6
        return None

    _forging_qualifier(monkeypatch, inflate)
    result = A.build_reference_inventory(roster, loader(root), independent(roster))
    task = _task(result, "0000000000")
    assert task["category"] == "halt" and task["reason"] == "second_reference_disagreement"
    assert task["second_reference"]["status"] == "disagrees" and task["reference_id"] is None
    assert task["second_reference"]["abs_difference"] > 0
    assert result["outcome"] == "halt" and A.validate_inventory(result, roster) == result
    with pytest.raises(A.ReferenceAdmissionHalt, match="halted"):
        A.admissible_roster(roster, result)
    # within the graph tolerance the implementations agree
    monkeypatch.undo()

    def nudge(scene, record):
        return (
            dict(record, optimal_cost=record["optimal_cost"] * (1 + 1e-9)) if scene.sample_id == "0000000000" else None
        )

    _forging_qualifier(monkeypatch, nudge)
    agreed = A.build_reference_inventory(roster, loader(root), independent(roster))
    assert _task(agreed, "0000000000")["category"] == "admissible"


def test_a_no_path_that_the_second_implementation_contradicts_halts(world, monkeypatch):
    root, roster = world

    def no_path(scene, record):
        if scene.sample_id != "0000000001":
            return None
        return dict(
            record,
            reference_status="no_path",
            optimal_cost=None,
            qualified=False,
            qualification_reasons=["reference_no_path"],
        )

    _forging_qualifier(monkeypatch, no_path)
    result = A.build_reference_inventory(roster, loader(root), released(roster))
    task = _task(result, "0000000001")
    assert task["reason"] == "second_reference_disagreement" and task["second_reference"]["cost"] == pytest.approx(
        EXACT
    )


def test_excluding_a_task_whose_published_label_is_legal_halts(world, monkeypatch):
    root, roster = world

    def no_path(scene, record):
        if scene.sample_id != "0000000001":
            return None
        return dict(
            record,
            reference_status="no_path",
            optimal_cost=None,
            qualified=False,
            qualification_reasons=["reference_no_path"],
        )

    _forging_qualifier(monkeypatch, no_path)
    # Only the label rule is under test here: the second implementation is made to agree with the forged no-path.
    monkeypatch.setattr(A, "second_reference_check", lambda scene, record, **k: {"status": "agrees"})
    result = A.build_reference_inventory(roster, loader(root), released(roster))
    task = _task(result, "0000000001")
    assert task["category"] == "halt" and task["reason"] == "no_path_with_legal_label"
    assert task["stored_label_legal"] is True and result["outcome"] == "halt"


def test_exclusions_beyond_the_total_or_block_caps_halt_the_preparation(world):
    root, roster = world  # two of eight tasks are excluded (rows 2 and 5), both in the block of rows 0-5
    table = R.roster_block_table(roster)
    block = table["blocks"][SUBOPTIMAL]
    assert table["blocks"][DISAGREEING] == block
    size = sum(value == block for value in table["blocks"].values())

    def run(**caps):
        values = {**A.PROPOSED_EXCLUSION_CAPS, "status": "sealed for this test", **caps}
        return A.build_reference_inventory(roster, loader(root), released(roster, exclusion_caps=values))

    total = run(max_total_fraction=0.2, max_block_fraction=0.9)
    assert total["outcome"] == "halt" and [b["scope"] for b in total["exclusion_caps_check"]["breaches"]] == ["total"]
    assert total["exclusion_caps_check"]["excluded_fraction"] == 0.25 and total["counts"]["n_halt"] == 0
    with pytest.raises(A.ReferenceAdmissionHalt, match="1 exclusion cap breach"):
        A.admissible_roster(roster, total)
    per_block = run(max_total_fraction=0.3, max_block_fraction=0.05)
    assert per_block["exclusion_caps_check"]["breaches"] == [
        {"scope": "block", "block": block, "n_excluded": 2, "n_tasks": size}
    ]
    assert per_block["exclusion_caps_check"]["excluded_by_block"] == {block: {"n_excluded": 2, "n_tasks": size}}
    assert per_block["outcome"] == "halt"
    # at least two exclusions in the block are needed for a block breach, and exactly the cap is allowed
    assert run(max_total_fraction=0.3, max_block_fraction=0.05, min_block_exclusions=3)["outcome"] == "complete"
    assert run(max_total_fraction=0.25, max_block_fraction=2 / size)["outcome"] == "complete"
    assert run(max_total_fraction=0.249, max_block_fraction=2 / size)["outcome"] == "halt"
    within = run(max_total_fraction=0.3, max_block_fraction=0.5)
    assert within["outcome"] == "complete" and within["exclusion_caps_check"]["within_caps"] is True
    assert A.validate_inventory(within, roster) == within


@pytest.mark.parametrize(
    "edit",
    ["caps_check", "findings", "second_reference", "legal_no_path"],
)
def test_validation_recomputes_the_caps_findings_and_second_reference(inventory, world, edit):
    _, roster = world
    _, result = inventory
    forged = copy.deepcopy(result)
    if edit == "caps_check":
        forged["exclusion_caps_check"] = {"breaches": []}
    elif edit == "findings":
        forged["label_findings"]["n_tasks_with_findings"] = 5
    elif edit == "second_reference":
        _task(forged, "0000000000")["second_reference"] = {"status": "not_compared"}
    else:
        task = _task(forged, SUBOPTIMAL)
        task.update(reasons=["reference_no_path"], reason="reference_no_path", stored_label_legal=True)
        forged["exclusions"] = A._listing(forged["tasks"], "excluded")
        forged["excluded_reason_counts"] = A._excluded_reason_counts(forged["tasks"])
    forged["inventory_sha256"] = A._digest(forged, "inventory_sha256")
    with pytest.raises(ValueError):
        A.validate_inventory(forged, roster)


def test_reference_ids_of_a_run_must_be_the_inventory_ids(inventory):
    _, result = inventory
    expected = A.inventory_reference_ids(result)
    assert set(expected) == {t["sample_id"] for t in result["tasks"] if t["category"] == "admissible"}
    run = {sample_id: {"reference_id": value} for sample_id, value in expected.items()}
    assert A.check_reference_ids(run, expected, context="t") == len(expected)
    other = dict(run, **{next(iter(run)): {"reference_id": "f" * 64}})
    with pytest.raises(A.ReferenceIdentityMismatch, match="1 reference_id"):
        A.check_reference_ids(other, expected, context="t")
    with pytest.raises(A.ReferenceIdentityMismatch, match="1 task\\(s\\) outside"):
        A.check_reference_ids({**run, SUBOPTIMAL: {"reference_id": "a" * 64}}, expected, context="t")
    with pytest.raises(A.ReferenceIdentityMismatch, match="no admissible reference"):
        A.check_reference_ids(run, {}, context="t")


def test_the_supervisor_loads_defective_labels_only_under_the_independent_policy(defective, tmp_path):
    from passage_eval import cli
    from passage_eval.supervision.supervisor import RosterInput, _load_roster_scenes

    root, roster = defective
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(roster))
    pinned_roster = RosterInput(path=str(path), sha256=roster["roster_sha256"], dataset_root=str(root))
    common = dict(method_name="m", family="f", architecture="a", inference="x:Y", resolution=3, split="validation")
    released_spec = cli.BenchSpec(**common)
    with pytest.raises(ValueError, match="PASSAGE sample 000000000[3467]"):
        _load_roster_scenes(pinned_roster, released_spec)
    analytic = cli.BenchSpec(**common, reference_policy="independent_analytic_reference", reference_cohort_id="c")
    _, scenes = _load_roster_scenes(pinned_roster, analytic)
    assert [scene.sample_id for scene in scenes] == [task["sample_id"] for task in roster["tasks"]]
