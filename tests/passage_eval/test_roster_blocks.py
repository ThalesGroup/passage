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

"""Roster v2 blocks: footprint requirement, bound block rule, derived rosters and sealed block tables.

Synthetic shards from ``test_roster.py`` (split label ``test`` there is a synthetic fixture, never the
released test split). Expected blocks are recomputed here from the tile coordinates.
"""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from passage_eval.io import roster as R
from passage_eval.protocol import geographic_blocks as G

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TR = _load("roster_fixtures_for_block_tests", HERE / "test_roster.py")
BLOCK_SALT = "synthetic-test-v1/r0003/roster-v1/blocks-0/"  # first six rows: 10, 4, 5, 1, 3, 6 (all placed)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return TR.write_dataset(tmp_path_factory.mktemp("blocks"))


@pytest.fixture(scope="module")
def full(dataset):
    """All twelve rows; row 11 has no footprint, so block analysis cannot be planned for it."""
    return TR.build(dataset, n_expected=12)[0]


@pytest.fixture(scope="module")
def placed(dataset):
    """Six rows that all have a footprint, built with the default (block analysis planned)."""
    return TR.BUILD.build_roster(  # synthetic fixture whose split is named test
        dataset,
        benchmark_id=TR.BENCHMARK,
        resolution=3,
        n_expected=6,
        salt=BLOCK_SALT,
        allow_test=True,
        admission=TR.green_admission(),
    )[0]


def rehash(roster):
    return TR.rehash(roster)


# ------------------------------------------------------------------------------------------ the block rule binding


def test_the_block_function_identity_is_the_schema_and_the_source_digest():
    identity = G.block_function_identity()
    source = Path(G.__file__).read_bytes()
    assert identity == {
        "schema": "passage-source-window-geographic-blocks/v1",
        "module": "passage_eval.protocol.geographic_blocks",
        "callable": "source_window_blocks",
        "source_sha256": hashlib.sha256(source).hexdigest(),
    }


def test_the_builder_writes_v2_with_the_bound_rule_and_requires_footprints_by_default(dataset, placed):
    with pytest.raises(ValueError, match="no usable source footprint.*0000000011"):
        TR.BUILD.build_roster(  # row 11 is selected
            dataset,
            benchmark_id=TR.BENCHMARK,
            resolution=3,
            n_expected=6,
            allow_test=True,
            admission=TR.green_admission(),
        )
    assert placed["schema"] == "passage-roster-v2"
    blocks = placed["geographic_blocks"]
    assert blocks["function"] == G.block_function_identity() and blocks["footprints_required"] is True
    assert all(task["source_windows"] for task in placed["tasks"])
    assert R.validate_roster(placed)["roster_sha256"] == placed["roster_sha256"]


def test_a_roster_without_planned_block_analysis_records_it(full):
    assert full["schema"] == "passage-roster-v2" and full["geographic_blocks"]["footprints_required"] is False
    assert full["geographic_blocks"]["n_tasks_without_footprint"] == 1


def _unplace(roster, index=0):
    roster = copy.deepcopy(roster)
    task = roster["tasks"][index]
    task.update(source_windows=[], footprint_status="missing", geographic_block=None)
    records = [{"sample_id": t["sample_id"], "source_windows": t["source_windows"]} for t in roster["tasks"]]
    records = [record for record in records if record["source_windows"]]
    blocks, assignments = R.geographic_block_record(
        records,
        n_tasks=len(roster["tasks"]),
        block_degrees=5,
        footprints_required=False,
        source=roster["geographic_blocks"]["source"],
    )
    for item in roster["tasks"]:
        if item["source_windows"]:
            item["geographic_block"] = assignments[item["sample_id"]]
    roster["geographic_blocks"] = blocks
    return roster


def test_a_task_without_footprint_is_refused_when_block_analysis_is_planned(placed):
    tolerated = rehash(_unplace(placed))
    assert R.validate_roster(tolerated)  # the same roster without planned block analysis is valid
    required = copy.deepcopy(tolerated)
    required["geographic_blocks"]["footprints_required"] = True
    with pytest.raises(ValueError, match="block analysis is planned but 1 task"):
        R.validate_roster(rehash(required))
    with pytest.raises(ValueError, match="block analysis is planned"):
        R.geographic_block_record([], n_tasks=1, block_degrees=5, footprints_required=True, source="s")


@pytest.mark.parametrize("field", ["source_sha256", "schema", "module", "drop", "flag", "null_record"])
def test_a_v2_roster_whose_block_rule_or_flag_was_changed_is_refused(placed, field):
    roster = copy.deepcopy(placed)
    blocks = roster["geographic_blocks"]
    if field == "drop":
        del blocks["function"]
    elif field == "flag":
        blocks["footprints_required"] = "yes"
    elif field == "null_record":
        roster["geographic_blocks"] = None
    else:
        blocks["function"][field] = "0" * 64 if field == "source_sha256" else "other"
    with pytest.raises(ValueError, match="Invalid roster"):
        R.validate_roster(rehash(roster))


def test_an_edit_of_the_block_rule_source_invalidates_v2_rosters(placed, monkeypatch):
    original = G.block_function_identity()
    monkeypatch.setattr(R, "block_function_identity", lambda rule=None: dict(original, source_sha256="f" * 64))
    with pytest.raises(ValueError, match="block rule changed after this roster was sealed"):
        R.validate_roster(placed)


def _as_v1(roster):
    roster = copy.deepcopy(roster)
    roster["schema"] = "passage-roster-v1"
    for key in ("function", "footprints_required"):
        roster["geographic_blocks"].pop(key)
    return rehash(roster)


def test_v1_rosters_stay_readable_with_their_tolerant_footprint_rule(full):
    v1 = _as_v1(full)
    assert R.validate_roster(v1)["schema"] == "passage-roster-v1"
    assert any(not task["source_windows"] for task in v1["tasks"])
    table = R.roster_block_table(v1)
    assert table["function"] is None and not table["complete"] and table["n_tasks_without_block"] == 1


def test_a_v2_roster_without_any_footprint_records_zero_blocks(full):
    roster = copy.deepcopy(full)
    for task in roster["tasks"]:
        task.update(source_windows=[], footprint_status="missing", geographic_block=None)
    roster["geographic_blocks"], _ = R.geographic_block_record(
        [], n_tasks=len(roster["tasks"]), block_degrees=5, footprints_required=False, source="s"
    )
    assert R.validate_roster(rehash(roster))["geographic_blocks"]["number_of_blocks"] == 0
    roster["geographic_blocks"]["number_of_blocks"] = 1
    with pytest.raises(ValueError, match="zero blocks"):
        R.validate_roster(rehash(roster))


# ------------------------------------------------------------------------------------------------- block tables


def test_the_block_table_and_summary_describe_the_roster_blocks(full, placed):
    table = R.roster_block_table(full)
    blocks = {int(sample_id): block for sample_id, block in table["blocks"].items()}
    # Rows 0-5 chain through cells (18, 36)-(19, 36); rows 6-8 sit in the south west; 9-10 span the antimeridian.
    assert {blocks[row] for row in range(6)} == {"geo5:18:36"}
    assert len({blocks[row] for row in (6, 7, 8)}) == 1 and blocks[6] != blocks[0]
    assert blocks[9] == blocks[10] == "geo5:18:00" and blocks[11] is None
    assert table["number_of_blocks"] == 3 and table["complete"] is False and table["assignment"] == "roster_blocks"
    assert R.validate_block_table(table) == table
    summary = R.block_summary(placed)  # rows 10, 4, 5, 1, 3, 6
    assert summary["number_of_blocks"] == 3 and summary["number_of_tasks"] == 6 and summary["complete"]
    assert summary["block_sizes"] == {"geo5:08:22": 1, "geo5:18:00": 1, "geo5:18:36": 4}
    assert summary["singleton_blocks"] == 2 and summary["min_tasks_per_block"] == 1
    assert summary["max_tasks_per_block"] == 4 and summary["median_tasks_per_block"] == 1
    assert summary["function"] == G.block_function_identity() and summary["statistical_independence_proven"] is False


@pytest.mark.parametrize("edit", ["block", "complete", "digest", "count", "empty_block"])
def test_a_tampered_block_table_is_refused(full, edit):
    table = R.roster_block_table(full)
    if edit == "block":
        table["blocks"]["0000000000"] = "geo5:00:00"
    elif edit == "complete":
        table["complete"] = True
    elif edit == "digest":
        table["block_table_sha256"] = "0" * 64
    elif edit == "count":
        table["number_of_blocks"] = 4
    elif edit == "empty_block":
        table["blocks"]["0000000000"] = ""
    with pytest.raises(ValueError):
        R.validate_block_table(table)


# ---------------------------------------------------------------------------------------------- derived rosters


def test_a_derived_roster_keeps_the_parent_ranks_and_blocks_where_local_blocks_differ(full):
    # Row 5 alone sits in cell (19, 36); in the parent, row 4 joins it to the block of rows 0-3 (18, 36).
    child = R.derive_roster(full, ["0000000006", "0000000005", "0000000011"], rule="test subset", derivation={"k": 1})
    by_id = {task["sample_id"]: task for task in child["tasks"]}
    parent_rank = {task["sample_id"]: task["rank"] for task in full["tasks"]}
    assert [task["sample_id"] for task in child["tasks"]] == sorted(by_id, key=parent_rank.__getitem__)
    assert [task["rank"] for task in child["tasks"]] == [0, 1, 2]
    assert all(task["parent_rank"] == parent_rank[sample_id] for sample_id, task in by_id.items())
    assert by_id["0000000005"]["geographic_block"] == "geo5:19:36"  # recomputed on the subset
    assert by_id["0000000005"]["parent_geographic_block"] == "geo5:18:36"  # kept from the parent
    assert by_id["0000000011"]["parent_geographic_block"] is None
    derivation = child["selection"]["derivation"]
    assert derivation["parent_roster_sha256"] == full["roster_sha256"] and derivation["k"] == 1
    assert child["selection"]["n_expected"] == child["selection"]["n_selected"] == 3
    table = R.roster_block_table(child)
    assert table["assignment"] == "inherited_parent_blocks"
    assert table["assignment_roster_sha256"] == full["roster_sha256"]
    assert table["blocks"]["0000000005"] == "geo5:18:36"
    check = R.verify_child_blocks(full, child)
    assert check["blocks_kept"] is True and check["n_child_tasks"] == 3
    grandchild = R.derive_roster(child, ["0000000005"], rule="smaller")
    assert R.roster_block_table(grandchild)["blocks"] == {"0000000005": "geo5:18:36"}
    assert R.roster_block_table(grandchild)["assignment_roster_sha256"] == full["roster_sha256"]
    R.verify_child_blocks(child, grandchild)


def test_a_child_that_only_recomputed_local_blocks_fails_the_parent_block_check(full):
    child = R.derive_roster(full, ["0000000005", "0000000006"], rule="subset")
    local = copy.deepcopy(child)
    for task in local["tasks"]:
        del task["parent_rank"], task["parent_geographic_block"]
    del local["selection"]["derivation"], local["geographic_blocks"]["parent_blocks"]
    local = rehash(local)
    R.validate_roster(local)  # a valid roster in itself, like a partition child with child-local blocks
    with pytest.raises(ValueError, match="analysed in block 'geo5:19:36', the parent's is 'geo5:18:36'"):
        R.verify_child_blocks(full, local)


def test_derived_roster_fields_are_validated(full, placed):
    child = R.derive_roster(full, ["0000000004", "0000000005", "0000000006"], rule="subset")
    swapped = copy.deepcopy(child)
    swapped["tasks"][0]["parent_geographic_block"] = "geo5:00:00"
    with pytest.raises(ValueError, match="parent_blocks differs"):
        R.validate_roster(rehash(swapped))
    partial = copy.deepcopy(child)
    del partial["tasks"][0]["parent_rank"]
    with pytest.raises(ValueError, match="needs parent_rank and parent_geographic_block"):
        R.validate_roster(rehash(partial))
    order = copy.deepcopy(child)
    order["tasks"][0]["parent_rank"], order["tasks"][1]["parent_rank"] = (
        order["tasks"][1]["parent_rank"],
        order["tasks"][0]["parent_rank"],
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        R.validate_roster(rehash(order))
    # Rows 4 and 5 share a local block; giving them different parent blocks contradicts the parent rule.
    split = R.derive_roster(full, ["0000000004", "0000000005"], rule="subset")
    split["tasks"][1]["parent_geographic_block"] = "geo5:99:99"
    with pytest.raises(ValueError, match="another parent block"):
        R.validate_roster(rehash(split))
    with pytest.raises(ValueError, match="outside the parent"):
        R.derive_roster(full, ["missing"], rule="subset")
    with pytest.raises(ValueError, match="unique"):
        R.derive_roster(full, ["0000000004", "0000000004"], rule="subset")
    with pytest.raises(ValueError, match="override"):
        R.derive_roster(full, ["0000000004"], rule="subset", derivation={"parent_roster_sha256": "0" * 64})
    with pytest.raises(ValueError, match="another parent roster_sha256"):
        R.verify_child_blocks(placed, child)


def test_a_v1_parent_gives_a_v1_child_with_inherited_blocks(full):
    v1 = _as_v1(full)
    child = R.derive_roster(v1, ["0000000005", "0000000011"], rule="subset")
    assert child["schema"] == "passage-roster-v1"
    assert R.roster_block_table(child)["blocks"] == {"0000000005": "geo5:18:36", "0000000011": None}
    R.verify_child_blocks(v1, child)


# ------------------------------------------------------------------------------------ scorer and comparison


def table_for(ids, blocks, roster_sha256="a" * 64):
    table = {
        "schema": R.BLOCK_TABLE_SCHEMA,
        "roster_schema": R.SCHEMA_V2,
        "roster_sha256": roster_sha256,
        "assignment": "roster_blocks",
        "assignment_roster_sha256": roster_sha256,
        "function": G.block_function_identity(),
        "block_degrees": 5,
        "footprints_required": all(block is not None for block in blocks),
        "n_tasks": len(ids),
        "n_tasks_without_block": sum(block is None for block in blocks),
        "number_of_blocks": len({block for block in blocks if block is not None}),
        "complete": all(block is not None for block in blocks),
        "blocks": dict(zip(ids, blocks)),
    }
    table["block_table_sha256"] = R.sha256_hex(R.canonical_json(table))
    return table


def _roster_manifest(roster_sha256="a" * 64, **extra):
    """A synthetic in-process manifest that records the roster it was made with, as a supervised roster run does."""
    return dict(extra, supervision={"roster": {"roster_sha256": roster_sha256}})


def _v1_table(ids, blocks, roster_sha256="a" * 64):
    table = table_for(ids, blocks, roster_sha256)
    table.pop("block_table_sha256")
    table.update(roster_schema=R.SCHEMA_V1, function=None, footprints_required=None)
    table["block_table_sha256"] = R.sha256_hex(R.canonical_json(table))
    return table


def _scenes(metre_scene, names):
    from dataclasses import replace

    return [replace(metre_scene, sample_id=name) for name in names]


def test_the_scorer_takes_cluster_ids_only_from_the_sealed_block_table(metre_scene, tmp_path):
    from dataclasses import replace

    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import read_run_artifact, run_requests, score_run_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult

    scenes = [replace(s, metadata={"cluster_id": "declared"}) for s in _scenes(metre_scene, ["a", "b", "c"])]

    class Direct:
        def plan(self, request):
            return PlanResult(metre_scene.oracle_path)

    requests = [PlanRequest.from_scene(s) for s in scenes]
    run = run_requests(Direct(), requests, output_dir=tmp_path / "run", manifest=_roster_manifest())
    artifact = read_run_artifact(tmp_path / "run")
    references = {s.sample_id: qualified_reference_for_scene(s) for s in scenes}
    _, declared = score_run_artifact(run, scenes, reference_store=references)
    assert declared["cluster_id"].tolist() == ["declared"] * 3  # unchanged without a table
    table = table_for(["a", "b", "c", "extra"], ["geo5:01:01", "geo5:01:01", "geo5:02:02", "geo5:03:03"])
    metrics, arrays = score_run_artifact(artifact, scenes, reference_store=references, geographic_blocks=table)
    assert arrays["cluster_id"].tolist() == ["geo5:01:01", "geo5:01:01", "geo5:02:02"]
    assert set(arrays["cluster_provenance"].tolist()) == {"sealed_roster_geographic_block"}
    record = metrics["geographic_blocks"]
    assert record["block_table_sha256"] == table["block_table_sha256"] and record["number_of_blocks"] == 2
    assert record["sealed"] is True and record["run_roster_sha256"] == "a" * 64
    # the table must belong to the roster the run records
    with pytest.raises(ValueError, match="belongs to roster"):
        score_run_artifact(
            artifact,
            scenes,
            reference_store=references,
            geographic_blocks=dict(table_for(["a", "b", "c"], ["x"] * 3, "b" * 64)),
        )
    unbound = run_requests(Direct(), requests, output_dir=tmp_path / "unbound")
    with pytest.raises(ValueError, match="this run records none"):
        score_run_artifact(unbound, scenes, reference_store=references, geographic_blocks=table)
    incomplete = table_for(["a", "b", "c"], ["geo5:01:01", None, "geo5:02:02"])
    metrics, arrays = score_run_artifact(artifact, scenes, reference_store=references, geographic_blocks=incomplete)
    assert arrays["cluster_id"].tolist() == ["", "", ""] and set(arrays["cluster_provenance"].tolist()) == {"unknown"}
    assert metrics["geographic_blocks"]["status"] == "unknown_tasks_without_footprint"
    with pytest.raises(ValueError, match="does not list 1 run task"):
        score_run_artifact(
            artifact, scenes, reference_store=references, geographic_blocks=table_for(["a", "b"], ["x", "y"])
        )
    tampered = copy.deepcopy(table)
    tampered["blocks"]["a"], tampered["blocks"]["c"] = "geo5:02:02", "geo5:01:01"  # same counts, other blocks
    with pytest.raises(ValueError, match="block_table_sha256"):
        score_run_artifact(artifact, scenes, reference_store=references, geographic_blocks=tampered)


def _scored(tmp_path, scenes, table=None, roster_sha256="a" * 64):
    import hashlib as _hashlib

    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import run_requests, score_run_artifact, write_scored_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult

    references = {s.sample_id: qualified_reference_for_scene(s) for s in scenes}

    class Direct:
        def plan(self, request):
            return PlanResult([(0, 0), (1, 1), (2, 2)])

    manifest = {
        "data_identity": {"source": "fixture", "revision": "frozen", "split": "validation"},
        "reference_identity": _hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest(),
        "seed": 1234,
    }
    if roster_sha256 is not None:
        manifest = _roster_manifest(roster_sha256, **manifest)
    run = run_requests(Direct(), [PlanRequest.from_scene(s) for s in scenes], output_dir=tmp_path, manifest=manifest)
    metrics, arrays = score_run_artifact(run, scenes, reference_store=references, geographic_blocks=table)
    write_scored_artifact(run, metrics, arrays)
    return tmp_path


def test_the_comparison_uses_an_explicit_or_scored_sealed_block_table(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = _scenes(metre_scene, ["a", "b", "c", "d"])
    table = table_for(["a", "b", "c", "d"], ["n1", "n1", "n2", "n3"])
    plain = [_scored(tmp_path / name, scenes) for name in ("p1", "p2")]
    unknown = compare_artifacts(plain, bootstrap_resamples=10)
    assert (
        unknown["uncertainty"]["block_provenance"] == "unknown"
        and unknown["uncertainty"]["sealed_roster_blocks"] is None
    )
    explicit = compare_artifacts(plain, bootstrap_resamples=10, geographic_blocks=table)
    assert explicit["uncertainty"]["block_provenance"] == "sealed_roster_block_table"
    assert explicit["comparisons"][0]["all_task"]["cost_regret"]["median"]["n_blocks"] == 3
    sealed = [_scored(tmp_path / name, scenes, table) for name in ("s1", "s2")]
    automatic = compare_artifacts(sealed, bootstrap_resamples=10)
    assert automatic["uncertainty"]["block_provenance"] == "scored_sealed_roster_blocks"
    assert automatic["uncertainty"]["sealed_roster_blocks"]["block_table_sha256"] == table["block_table_sha256"]
    other = table_for(["a", "b", "c", "d"], ["n1", "n2", "n2", "n3"])
    with pytest.raises(ValueError, match="differ from the supplied roster block table"):
        compare_artifacts(sealed, bootstrap_resamples=10, geographic_blocks=other)
    with pytest.raises(ValueError, match="some scores carry roster blocks and others do not"):
        compare_artifacts([sealed[0], plain[0]], bootstrap_resamples=10)
    # an earlier score of the same roster, without stored blocks, compares with the explicit table
    assert (
        compare_artifacts([sealed[0], plain[0]], bootstrap_resamples=10, geographic_blocks=table)["uncertainty"][
            "block_provenance"
        ]
        == "sealed_roster_block_table"
    )
    mixed = _scored(tmp_path / "s3", scenes, other)
    with pytest.raises(ValueError, match="different roster block tables"):
        compare_artifacts([sealed[0], mixed], bootstrap_resamples=10)
    with pytest.raises(ValueError, match="does not list 1 compared task"):
        compare_artifacts(
            plain, bootstrap_resamples=10, geographic_blocks=table_for(["a", "b", "c"], ["n1", "n1", "n2"])
        )
    partial = table_for(["a", "b", "c", "d"], ["n1", None, "n2", "n3"])
    result = compare_artifacts(plain, bootstrap_resamples=10, geographic_blocks=partial)
    assert result["uncertainty"]["status"] == "unknown_geographic_blocks"
    assert result["uncertainty"]["sealed_roster_blocks"]["status"] == "unknown_tasks_without_footprint"


def test_an_explicit_block_table_is_bound_to_the_roster_of_every_compared_run(metre_scene, tmp_path):
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = _scenes(metre_scene, ["a", "b", "c", "d"])
    table = table_for(["a", "b", "c", "d"], ["n1", "n1", "n2", "n3"])
    plain = [_scored(tmp_path / name, scenes) for name in ("p1", "p2")]
    foreign = table_for(["a", "b", "c", "d"], ["n1", "n1", "n2", "n3"], roster_sha256="b" * 64)
    with pytest.raises(ValueError, match="compared run 0: the block table belongs to roster bbbb"):
        compare_artifacts(plain, bootstrap_resamples=10, geographic_blocks=foreign)
    other_roster = _scored(tmp_path / "o1", scenes, roster_sha256="c" * 64)
    with pytest.raises(ValueError, match="compared run 1: the block table belongs to roster aaaa"):
        compare_artifacts([plain[0], other_roster], bootstrap_resamples=10, geographic_blocks=table)
    unbound = [_scored(tmp_path / name, scenes, roster_sha256=None) for name in ("u1", "u2")]
    with pytest.raises(ValueError, match="this run records none"):
        compare_artifacts(unbound, bootstrap_resamples=10, geographic_blocks=table)
    assert compare_artifacts(unbound, bootstrap_resamples=10)["uncertainty"]["block_provenance"] == "unknown"


def test_v1_roster_blocks_are_used_but_never_labelled_sealed(metre_scene, tmp_path, full):
    from passage_eval.execution import read_run_artifact, score_run_artifact
    from passage_eval.metrics.comparison import compare_artifacts

    scenes = _scenes(metre_scene, ["a", "b", "c", "d"])
    v1 = _v1_table(["a", "b", "c", "d"], ["n1", "n1", "n2", "n3"])
    assert R.block_table_sealed(v1) is False and R.block_table_sealed(R.roster_block_table(full)) is True
    scored = [_scored(tmp_path / name, scenes, v1) for name in ("v1a", "v1b")]
    with np.load(scored[0] / "per_sample.npz", allow_pickle=False) as arrays:
        assert set(arrays["cluster_provenance"].tolist()) == {"unsealed_roster_geographic_block"}
    record = json.loads((scored[0] / "metrics.json").read_text())["geographic_blocks"]
    assert record["sealed"] is False and record["status"] == "unsealed_roster_blocks"
    assert record["roster_schema"] == R.SCHEMA_V1
    automatic = compare_artifacts(scored, bootstrap_resamples=10)
    assert automatic["uncertainty"]["block_provenance"] == "scored_unsealed_roster_blocks"
    assert automatic["uncertainty"]["status"] == "declared_blocks"
    explicit = compare_artifacts(scored, bootstrap_resamples=10, geographic_blocks=v1)
    assert explicit["uncertainty"]["block_provenance"] == "unsealed_roster_block_table"
    assert explicit["uncertainty"]["sealed_roster_blocks"]["sealed"] is False
    # a v2 table whose bound block rule is not the installed one is refused, as its roster would be
    stale = table_for(["a", "b", "c", "d"], ["n1", "n1", "n2", "n3"])
    stale.pop("block_table_sha256")
    stale["function"] = dict(stale["function"], source_sha256="0" * 64)
    stale["block_table_sha256"] = R.sha256_hex(R.canonical_json(stale))
    artifact = read_run_artifact(scored[0])
    with pytest.raises(ValueError, match="another block rule"):
        score_run_artifact(artifact, scenes, reference_store=_references(scenes), geographic_blocks=stale)


def _references(scenes):
    from passage.audit import qualified_reference_for_scene

    return {s.sample_id: qualified_reference_for_scene(s) for s in scenes}
