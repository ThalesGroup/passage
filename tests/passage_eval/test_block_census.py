"""Block rules A and B, the sealed unplaced block and the label-free block census.

Synthetic shards of ``test_roster.py`` (their split is named ``test`` as a synthetic fixture; the builder accepts it
with the synthetic GREEN grant). Expected blocks are recomputed here from the tile coordinates: rows 0-5 use the
tiles (0,0)...(5,0), so rule A merges them through row 4, whose tiles (4,0) and (5,0) fall in the 5-degree cells
(18,36) and (19,36); rule B keeps row 5 alone in (19,36). Rows 9 and 10 use (0,179) and (0,180), across the
antimeridian: rule A merges the cells (18,71) and (18,0); rule B keeps the main tile's (18,71). Row 11 has no used
tile and the main tile (0,0). Rule A: {geo5:08:22: 3, geo5:18:00: 2, geo5:18:36: 6} and row 11 unplaced; rule B:
{cell5:08:22: 3, cell5:18:36: 6, cell5:18:71: 2, cell5:19:36: 1}, rows 4 and 10 crossing a cell boundary.
"""

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from passage_eval.io import roster as R
from passage_eval.protocol import block_census as C
from passage_eval.protocol import geographic_blocks as G
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TR = _load("roster_fixtures_for_census_tests", HERE / "test_roster.py")


def cell(lat, lon, degrees=5):
    return (lat + 90) // degrees, (lon + 180) // degrees


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return TR.write_dataset(tmp_path_factory.mktemp("census"))


def build(dataset, **kwargs):
    kwargs.setdefault("require_footprints", True)
    return TR.build(dataset, n_expected=12, **kwargs)[0]


@pytest.fixture(scope="module")
def unplaced(dataset):
    """All twelve rows under rule A with the sealed unplaced block: row 11 is unplaced."""
    return build(dataset, unplaced_block=True)


@pytest.fixture(scope="module")
def main_tile(dataset):
    return build(dataset, block_rule="main_tile_cell")


# ------------------------------------------------------------------------------------------------ rule B


def test_rule_b_is_the_cell_of_the_main_tile_without_union(main_tile):
    blocks = {int(task["sample_id"]): task["geographic_block"] for task in main_tile["tasks"]}
    assert blocks[4] == "cell5:{:02d}:{:02d}".format(*cell(4, 0)) == "cell5:18:36"
    assert blocks[5] == "cell5:{:02d}:{:02d}".format(*cell(5, 0)) == "cell5:19:36"
    assert blocks[11] == "cell5:18:36"  # main tile (0, 0) although no used tile is recorded
    summary = main_tile["geographic_blocks"]
    assert summary["rule"] == "main_tile_cell" and summary["function"] == G.block_function_identity("main_tile_cell")
    assert summary["function"]["callable"] == "main_tile_blocks" and summary["footprints_required"] is True
    assert summary["crossing_task_count"] == 2  # row 4: (18,36)-(19,36); row 10: (18,71)-(18,0)
    assert summary["block_sizes"] == {"cell5:08:22": 3, "cell5:18:36": 6, "cell5:18:71": 2, "cell5:19:36": 1}
    table = R.roster_block_table(main_tile)
    assert table["complete"] and table["rule"] == "main_tile_cell" and table["n_unplaced"] == 0
    assert R.block_table_sealed(table) is True


def test_a_tampered_rule_b_roster_is_refused(main_tile):
    moved = copy.deepcopy(main_tile)
    moved["tasks"][0]["source_tile"] = {"lat": 40, "lon": 40}
    with pytest.raises(ValueError, match="geographic_block differs|differs from the task footprints"):
        R.validate_roster(TR.rehash(moved))
    relabelled = copy.deepcopy(main_tile)
    relabelled["geographic_blocks"]["rule"] = "source_window_union"
    with pytest.raises(ValueError, match="Invalid roster"):
        R.validate_roster(TR.rehash(relabelled))
    unknown = copy.deepcopy(main_tile)
    unknown["geographic_blocks"]["rule"] = "nearest_capital"
    with pytest.raises(ValueError, match="geographic_blocks.rule must be one of"):
        R.validate_roster(TR.rehash(unknown))


def test_a_task_without_main_tile_is_unplaced_under_rule_b(dataset, main_tile):
    roster = copy.deepcopy(main_tile)
    roster["tasks"][0]["source_tile"] = None
    roster["tasks"][0]["geographic_block"] = None
    with pytest.raises(ValueError, match="block analysis is planned"):
        R.validate_roster(TR.rehash(roster))


# ------------------------------------------------------------------------------------------- the unplaced block


def test_the_unplaced_block_is_sealed_in_the_roster_and_completes_the_table(unplaced):
    summary = unplaced["geographic_blocks"]
    assert summary["footprints_required"] is False and summary["n_tasks_without_footprint"] == 1
    assert summary["unplaced_block"] == R.unplaced_block_record(["0000000011"])
    assert summary["unplaced_block"]["block_id"] == G.UNPLACED_BLOCK == "unplaced"
    task = next(t for t in unplaced["tasks"] if t["sample_id"] == "0000000011")
    assert task["geographic_block"] is None  # the task record keeps no invented block
    table = R.roster_block_table(unplaced)
    assert table["blocks"]["0000000011"] == "unplaced" and table["complete"] is True
    assert table["n_unplaced"] == 1 and table["unplaced_block_declared"] is True
    assert R.validate_block_table(table) == table
    assert R.block_summary(unplaced)["n_unplaced"] == 1


@pytest.mark.parametrize("edit", ["drop_task", "extra_task", "required", "count", "v1"])
def test_a_tampered_unplaced_block_is_refused(unplaced, edit):
    roster = copy.deepcopy(unplaced)
    record = roster["geographic_blocks"]["unplaced_block"]
    if edit == "drop_task":
        record["task_ids"] = []
    elif edit == "extra_task":
        record["task_ids"].append("0000000000")
    elif edit == "required":
        roster["geographic_blocks"]["footprints_required"] = True
    elif edit == "count":
        record["n_tasks"] = 2
    else:
        roster["schema"] = "passage-roster-v1"
    with pytest.raises(ValueError, match="Invalid roster"):
        R.validate_roster(TR.rehash(roster))
    if edit in ("drop_task", "count"):
        return
    with pytest.raises(ValueError):
        R.geographic_block_record(
            [], n_tasks=1, block_degrees=5, footprints_required=True, source="s", unplaced_ids=["x"]
        )


def test_a_block_table_must_count_its_unplaced_tasks(unplaced):
    table = R.roster_block_table(unplaced)
    table.pop("block_table_sha256")
    for change in ({"n_unplaced": 0}, {"unplaced_block_declared": False}):
        forged = dict(table, **change)
        forged["block_table_sha256"] = R.sha256_hex(R.canonical_json(forged))
        with pytest.raises(ValueError, match="inconsistent"):
            R.validate_block_table(forged)


def test_derived_rosters_and_partitions_keep_the_unplaced_block(unplaced, tmp_path):
    child = R.derive_roster(unplaced, ["0000000005", "0000000011"], rule="subset")
    by_id = {task["sample_id"]: task for task in child["tasks"]}
    assert by_id["0000000011"]["parent_geographic_block"] == "unplaced"
    assert child["geographic_blocks"]["unplaced_block"]["task_ids"] == ["0000000011"]
    assert R.roster_block_table(child)["blocks"] == {"0000000005": "geo5:18:36", "0000000011": "unplaced"}
    R.verify_child_blocks(unplaced, child)
    placed = copy.deepcopy(child)
    # a placed task never inherits the unplaced block
    next(t for t in placed["tasks"] if t["sample_id"] == "0000000005")["parent_geographic_block"] = "unplaced"
    with pytest.raises(ValueError, match="Invalid roster"):
        R.validate_roster(TR.rehash(placed))
    undeclared = copy.deepcopy(child)
    del undeclared["geographic_blocks"]["unplaced_block"]  # an unplaced inheritance needs the declared block
    undeclared["geographic_blocks"]["parent_blocks"]["block_sizes"].pop("unplaced")
    with pytest.raises(ValueError, match="Invalid roster"):
        R.validate_roster(TR.rehash(undeclared))


def test_a_campaign_partition_keeps_the_unplaced_block(tmp_path):
    from passage_eval.campaign import partition as P
    from passage_eval.io import pinned

    root = TR.write_dataset(tmp_path / "validation", split="validation")
    roster = TR.BUILD.build_roster(
        root, benchmark_id="v", resolution=3, split="validation", n_expected=12, unplaced_block=True
    )[0]
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(roster))
    manifest = P.partition(path, pinned.sha256_bytes(path.read_bytes()), roster["roster_sha256"], tmp_path / "parts", 5)
    blocks = {
        row["sample_id"]: row["parent_geographic_block"]
        for item in manifest["batches"]
        for row in item["parent_mapping"]
    }
    assert blocks["0000000011"] == "unplaced" and None not in blocks.values()
    without = copy.deepcopy(roster)
    del without["geographic_blocks"]["unplaced_block"]
    path.write_text(json.dumps(TR.rehash(without)))
    with pytest.raises(ValueError, match="without a geographic block"):
        P.read_parent(path, pinned.sha256_bytes(path.read_bytes()), TR.rehash(without)["roster_sha256"])


def test_the_scorer_assigns_the_sealed_unplaced_block(metre_scene, tmp_path, unplaced):
    from dataclasses import replace

    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import run_requests, score_run_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult

    table = R.roster_block_table(unplaced)
    ids = [task["sample_id"] for task in unplaced["tasks"] if task["sample_id"] != "0000000011"][:3] + ["0000000011"]
    scenes = [replace(metre_scene, sample_id=sample_id) for sample_id in ids]

    class Direct:
        def plan(self, request):
            return PlanResult(metre_scene.oracle_path)

    manifest = {"supervision": {"roster": {"roster_sha256": unplaced["roster_sha256"]}}}
    run = run_requests(Direct(), [PlanRequest.from_scene(s) for s in scenes], output_dir=tmp_path, manifest=manifest)
    references = {s.sample_id: qualified_reference_for_scene(s) for s in scenes}
    metrics, arrays = score_run_artifact(run, scenes, reference_store=references, geographic_blocks=table)
    assert arrays["cluster_id"].tolist() == [table["blocks"][sample_id] for sample_id in ids]
    assert arrays["cluster_id"].tolist()[-1] == "unplaced" and "unplaced" not in arrays["cluster_id"].tolist()[:3]
    assert set(arrays["cluster_provenance"].tolist()) == {"sealed_roster_geographic_block"}
    assert metrics["geographic_blocks"]["status"] == "sealed_roster_blocks"


# ------------------------------------------------------------------------------------------------- the census


def criterion(**changes):
    return {**C.PROPOSED_CRITERION, "status": "sealed for this test", **changes}


def test_the_proposed_criterion_is_the_judges_and_labelled_unsealed():
    assert C.PROPOSED_CRITERION["status"] == "proposed_not_sealed"
    assert (C.PROPOSED_CRITERION["min_blocks"], C.PROPOSED_CRITERION["max_block_share"]) == (30, 0.10)
    assert C.PROPOSED_CRITERION["max_unplaced_fraction"] == 0.005
    assert (C.PROPOSED_CRITERION["block_degrees"], C.PROPOSED_CRITERION["sensitivity_degrees"]) == (5, 10)


def test_the_census_reports_both_rules_and_falls_back_in_order(unplaced):
    grant = TR.green_admission()
    census = C.block_census(unplaced, allow_test=True, admission=grant)
    assert census["labels_read"] is False and census["criterion_status"] == "proposed_not_sealed"
    a, b = census["rules"]["rule_a"]["adequacy"], census["rules"]["rule_b"]["adequacy"]
    # rule A: rows 0-5 and 11 is unplaced; the block of rows 0-5 holds 6 of 12 tasks
    assert a["largest_block_tasks"] == 6 and a["largest_block_share"] == 0.5 and a["n_unplaced"] == 1
    assert census["rules"]["rule_a"]["unplaced_task_ids"] == ["0000000011"]
    # rule B: row 5 leaves the block of rows 0-4; row 11 is placed by its main tile (0, 0)
    assert b["n_unplaced"] == 0 and b["largest_block_tasks"] == 6  # rows 0-4 and 11 share (18, 36)
    assert census["decision"] == "halt_unplaced_cap"  # 1 of 12 unplaced exceeds 0.5 %
    assert census["confirmatory_block_analysis_possible"] is False
    loose = dict(max_unplaced_fraction=0.1)
    assert C.block_census(unplaced, criterion=criterion(**loose), allow_test=True, admission=grant)["decision"] == (
        "descriptive_only"
    )
    rule_a = C.block_census(
        unplaced, criterion=criterion(min_blocks=3, max_block_share=0.5, **loose), allow_test=True, admission=grant
    )
    assert rule_a["decision"] == "rule_a" and rule_a["confirmatory_block_analysis_possible"] is True
    rule_b = C.block_census(
        unplaced,
        criterion=criterion(min_blocks=3, max_block_share=0.5, max_unplaced_fraction=0.0),
        allow_test=True,
        admission=grant,
    )
    assert rule_b["decision"] == "halt_unplaced_cap"  # rule A has an unplaced task beyond a zero cap: no fallback
    body = {key: value for key, value in rule_a.items() if key != "census_sha256"}
    assert rule_a["census_sha256"] == R.sha256_hex(R.canonical_json(body))


def test_rule_b_is_retained_only_when_rule_a_fails(main_tile):
    # Without row 11 (11 tasks): rule A has 3 blocks, the largest 6/11; rule B has 4, the largest 5/11.
    roster = R.derive_roster(
        main_tile, [t["sample_id"] for t in main_tile["tasks"] if t["sample_id"] != "0000000011"], rule="without row 11"
    )
    grant = TR.green_admission()
    census = C.block_census(
        roster, criterion=criterion(min_blocks=4, max_block_share=0.5), allow_test=True, admission=grant
    )
    a, b = census["rules"]["rule_a"]["adequacy"], census["rules"]["rule_b"]["adequacy"]
    assert (a["number_of_blocks"], a["largest_block_tasks"], a["meets_criterion"]) == (3, 6, False)
    assert (b["number_of_blocks"], b["largest_block_tasks"], b["meets_criterion"]) == (4, 5, True)
    assert census["decision"] == "rule_b" and census["confirmatory_block_analysis_possible"] is True
    # rule A is retained first whenever it meets the criterion
    first = C.block_census(
        roster, criterion=criterion(min_blocks=3, max_block_share=0.6), allow_test=True, admission=grant
    )
    assert first["decision"] == "rule_a"
    neither = C.block_census(
        roster, criterion=criterion(min_blocks=5, max_block_share=0.6), allow_test=True, admission=grant
    )
    assert neither["decision"] == "descriptive_only" and neither["confirmatory_block_analysis_possible"] is False


def test_the_sensitivities_drop_crossing_tasks_and_use_coarser_cells(main_tile):
    census = C.block_census(main_tile, allow_test=True, admission=TR.green_admission())
    without = census["rules"]["rule_b_without_crossing_tasks"]
    assert without["decisive"] is False and without["excluded_crossing_tasks"] == 2  # rows 4 and 10
    adequacy = without["adequacy"]
    assert adequacy["n_tasks"] == 10 and adequacy["largest_block_tasks"] == 5 and adequacy["number_of_blocks"] == 4
    coarse = census["rules"]["rule_b_sensitivity_degrees"]
    assert coarse["block_degrees"] == 10 and coarse["decisive"] is False
    assert coarse["adequacy"]["block_size_distribution"]["largest_blocks"] == {
        "cell10:09:18": 7,
        "cell10:04:11": 3,
        "cell10:09:35": 2,
    }
    with pytest.raises(ValueError, match="only rule B"):
        C.rule_assignment(main_tile["tasks"], rule="source_window_union", block_degrees=5, exclude_crossing=True)


@pytest.mark.parametrize(
    "change, match",
    [
        ({"min_blocks": 1}, "at least 2"),
        ({"max_block_share": 0}, "\\(0, 1\\]"),
        ({"max_unplaced_fraction": 1}, "\\[0, 1\\)"),
        ({"block_degrees": 7}, "divide"),
        ({"extra": 1}, "exactly the keys"),
        ({"schema": "other"}, "criterion is required"),
        ({"status": ""}, "sealing status"),
    ],
)
def test_an_invalid_criterion_is_refused(change, match):
    with pytest.raises(ValueError, match=match):
        C.validate_criterion({**C.PROPOSED_CRITERION, **change})


def test_adequacy_counts_the_unplaced_block_and_refuses_incomplete_assignments():
    blocks = {f"t{i}": f"b{i % 3}" for i in range(9)} | {"u": "unplaced"}
    result = C.adequacy(blocks, min_blocks=4, max_block_share=0.3, max_unplaced_fraction=0.1)
    assert result["number_of_blocks"] == 4 and result["n_unplaced"] == 1 and result["meets_criterion"] is True
    assert C.adequacy(blocks, min_blocks=5, max_block_share=0.3, max_unplaced_fraction=0.1)["meets_criterion"] is False
    assert C.adequacy(blocks, min_blocks=4, max_block_share=0.29, max_unplaced_fraction=0.1)["meets_criterion"] is False
    assert C.adequacy(blocks, min_blocks=4, max_block_share=0.3, max_unplaced_fraction=0.09)["meets_criterion"] is False
    incomplete = C.adequacy({**blocks, "x": None}, min_blocks=4, max_block_share=0.4, max_unplaced_fraction=0.1)
    assert incomplete["complete"] is False and incomplete["meets_criterion"] is False


def test_the_census_reads_no_dataset_file_and_guards_the_test_split(unplaced, tmp_path):
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        C.block_census(unplaced)
    with pytest.raises(TestSplitRefused, match="admission grant"):
        C.block_census(unplaced, allow_test=True)
    path = tmp_path / "roster.json"
    path.write_text(json.dumps(unplaced))
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        C.main(["--roster", str(path), "--roster-sha256", unplaced["roster_sha256"]])
    record = tmp_path / "red.json"
    record.write_text("{}")
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        C.main(
            [
                "--roster",
                str(path),
                "--roster-sha256",
                unplaced["roster_sha256"],
                "--allow-test-split",
                "--test-admission-record",
                str(record),
            ]
        )


def test_the_census_command_writes_a_new_file(dataset, tmp_path, capsys):
    validation_root = TR.write_dataset(tmp_path / "validation", split="validation")
    roster = TR.BUILD.build_roster(
        validation_root, benchmark_id="v", resolution=3, split="validation", n_expected=12, unplaced_block=True
    )[0]
    path, out = tmp_path / "roster.json", tmp_path / "census.json"
    path.write_text(json.dumps(roster))
    assert C.main(["--roster", str(path), "--roster-sha256", roster["roster_sha256"], "--output", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["decision"] == "halt_unplaced_cap" and written["roster_sha256"] == roster["roster_sha256"]
    assert json.loads(capsys.readouterr().out)["census_sha256"] == written["census_sha256"]
    with pytest.raises(FileExistsError):
        C.main(["--roster", str(path), "--roster-sha256", roster["roster_sha256"], "--output", str(out)])
