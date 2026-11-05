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

"""Warm-up selection, warm-up roster and paired procedure order, on synthetic validation snapshots.

The snapshot comes from ``test_rehearsal_roster.write_validation_snapshot`` (twelve synthetic rows labelled
``validation``). A second validation snapshot holds the same rows under other ids and metadata indexes: each of its
rows has the content of an evaluated row but another ordinary-metadata digest, as a validation copy of an evaluated
row would. Expected rankings are recomputed here with hashlib.
"""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from passage_eval.io import roster as R
from passage_eval.protocol import warmup as W
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REHEARSAL = _load("rehearsal_fixtures_for_warmup_tests", HERE / "test_rehearsal_roster.py")
TR = REHEARSAL.TR
SCRIPT = _load("build_warmup_roster_under_test", HERE.parents[1] / "scripts" / "build_warmup_roster.py")
SALT = "synthetic-warmup/r0003/v1/"
IDS = [f"{i:010d}" for i in range(12)]


def ranked(salt, ids):
    return sorted(ids, key=lambda i: hashlib.sha256((salt + i).encode("utf-8")).hexdigest())


def identity(content, metadata="validation"):
    """A synthetic input identity record: every content field derives from ``content``."""
    record = {field: f"{field}:{content}" for field in R.IDENTITY_FIELDS}
    record["ordinary_metadata_hash"] = f"ordinary metadata of a {metadata} row"
    return record


# ------------------------------------------------------------------------------------------------- selection


def test_the_selection_is_the_salted_sha256_ranking_minus_evaluated_ids_and_inputs():
    evaluated = ranked(SALT, IDS)[:2]
    identities = {i: identity(i) for i in IDS}
    twin = ranked(SALT, IDS)[2]  # another id with the content of an evaluated input from another split
    evaluated_twin = identity(twin, metadata="test")
    assert R.identity_digest(evaluated_twin) != R.identity_digest(identities[twin])  # only the metadata differs
    assert R.content_identity_digest(evaluated_twin) == R.content_identity_digest(identities[twin])
    record = W.select_warmup_scenes(
        IDS,
        SALT,
        4,
        split="validation",
        evaluated_ids=evaluated + ["not-in-the-population"],
        identities=identities,
        evaluated_identities=[evaluated_twin],
    )
    expected = [i for i in ranked(SALT, IDS) if i not in evaluated and i != twin][:4]
    assert record["selected_ids"] == expected
    assert record["skipped_evaluated_ids"] == evaluated and record["skipped_evaluated_inputs"] == [twin]
    assert record["selected_input_identity_sha256"] == [R.identity_digest(identities[i]) for i in expected]
    assert record["selected_input_content_sha256"] == [R.content_identity_digest(identities[i]) for i in expected]
    assert record["salt"] == SALT and record["k"] == 4 and record["split"] == "validation"
    assert record["disjointness_checked_by"] == "sample_id_and_input_content_identity"
    assert record["content_identity_fields"] == [f for f in R.IDENTITY_FIELDS if f != "ordinary_metadata_hash"]
    assert record["decision_status"] == "proposed_not_sealed"
    for item in record["selected"]:
        assert item["selection_key"] == hashlib.sha256((SALT + item["sample_id"]).encode()).hexdigest()
    body = {key: value for key, value in record.items() if key != "selection_sha256"}
    assert record["selection_sha256"] == R.sha256_hex(R.canonical_json(body))
    again = W.select_warmup_scenes(
        list(reversed(IDS)),
        SALT,
        4,
        split="val",
        evaluated_ids=set(evaluated) | {"not-in-the-population"},
        identities=identities,
        evaluated_identities=(evaluated_twin,),
    )
    assert again == record  # independent of input order and of the split alias


def test_the_default_count_is_the_proposed_eight_and_ids_alone_are_recorded_as_such():
    record = W.select_warmup_scenes(IDS, SALT, split="validation", evaluated_ids=[])
    assert W.PROPOSED_WARMUP_COUNT == 8 and record["k"] == 8 and record["selected_ids"] == ranked(SALT, IDS)[:8]
    assert record["disjointness_checked_by"] == "sample_id_only" and record["selected_input_identity_sha256"] is None


@pytest.mark.parametrize(
    "kwargs, error, match",
    [
        ({"split": "test"}, TestSplitRefused, "refuses the test split"),
        ({"split": "calibration"}, ValueError, "validation split only"),
        ({"ids": IDS + IDS[:1]}, ValueError, "unique"),
        ({"salt": ""}, ValueError, "salt"),
        ({"k": 0}, ValueError, "positive"),
        ({"k": True}, ValueError, "positive"),
        ({"k": 13}, ValueError, "only 12 of 12"),
        ({"evaluated_ids": "0000000001"}, ValueError, "collection"),
        ({"identities": {IDS[0]: "x"}, "evaluated_identities": []}, ValueError, "cover every"),
        ({"identities": {i: identity(i) for i in IDS}}, ValueError, "evaluated identities"),
        ({"evaluated_identities": ["x"]}, ValueError, "candidates' identities"),
        ({"identities": {i: i for i in IDS}, "evaluated_identities": []}, ValueError, "record of identity fields"),
        (
            {"identities": {i: {"graph_id": i} for i in IDS}, "evaluated_identities": []},
            ValueError,
            "lacks the content fields",
        ),
        (
            {"identities": {i: identity(i) for i in IDS}, "evaluated_identities": identity("x")},
            ValueError,
            "collection of input identity records",
        ),
    ],
)
def test_invalid_selections_are_refused(kwargs, error, match):
    values = dict(ids=IDS, salt=SALT, k=4, split="validation", evaluated_ids=[])
    values.update(kwargs)
    ids, salt, k = values.pop("ids"), values.pop("salt"), values.pop("k")
    with pytest.raises(error, match=match):
        W.select_warmup_scenes(ids, salt, k, **values)


# ---------------------------------------------------------------------------------------------- warm-up roster


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = REHEARSAL.write_validation_snapshot(tmp_path_factory.mktemp("warmup") / "validation")
    evaluated, _ = TR.BUILD.build_roster(
        root, benchmark_id="synthetic-evaluated-v1", resolution=3, split="validation", n_expected=6, salt="eval/"
    )
    return SimpleNamespace(root=root, evaluated=evaluated)


def pool_builder(world, calls):
    def build(n):
        calls.append(n)
        return TR.BUILD.build_roster(
            world.root,
            benchmark_id="synthetic-warmup-v1",
            resolution=3,
            split="validation",
            n_expected=n,
            salt=SALT,
            require_footprints=False,
        )[0]

    return build


def test_the_warmup_roster_is_a_validated_roster_of_the_disjoint_selection(world):
    calls = []
    roster, selection = W.build_warmup_roster(
        pool_builder(world, calls), salt=SALT, evaluated_roster=world.evaluated, benchmark_id="synthetic-warmup-v1", k=4
    )
    evaluated_ids = {task["sample_id"] for task in world.evaluated["tasks"]}
    expected = [i for i in ranked(SALT, IDS) if i not in evaluated_ids][:4]
    assert [task["sample_id"] for task in roster["tasks"]] == expected == selection["selected_ids"]
    assert calls[0] == 4 and calls[-1] > 4  # the pool grew past the evaluated ids it met
    assert R.validate_roster(roster)["roster_sha256"] == roster["roster_sha256"]
    assert roster["schema"] == "passage-roster-v2" and roster["dataset"]["split"] == "validation"
    assert roster["selection"]["salt"] == SALT and roster["selection"]["n_expected"] == 4
    warm = roster["selection"]["warmup"]
    assert (
        warm["selection_sha256"] == selection["selection_sha256"] and warm["decision_status"] == "proposed_not_sealed"
    )
    assert warm["evaluated_roster_sha256"] == world.evaluated["roster_sha256"]
    assert roster["selection"]["population"]["n_rows"] == 12
    assert selection["selected_input_identity_sha256"] == [task["input_identity_sha256"] for task in roster["tasks"]]
    assert selection["selected_input_content_sha256"] == [
        R.content_identity_digest(task["input_identity"]) for task in roster["tasks"]
    ]
    assert roster["dataset_root_identity"] == world.evaluated["dataset_root_identity"]
    scenes = R.load_roster_scenes(roster, world.root)
    assert [scene.sample_id for scene in scenes] == expected


def test_the_supervisor_accepts_the_warmup_roster_as_disjoint_inputs(world, tmp_path):
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.supervision.supervisor import RosterInput, WarmupInput, _load_warmup_requests

    roster, _ = W.build_warmup_roster(
        pool_builder(world, []), salt=SALT, evaluated_roster=world.evaluated, benchmark_id="synthetic-warmup-v1", k=4
    )
    path = tmp_path / "warmup.json"
    path.write_text(json.dumps(roster))
    evaluated = [PlanRequest.from_scene(scene) for scene in R.load_roster_scenes(world.evaluated, world.root)]
    source = WarmupInput(roster=RosterInput(str(path), roster["roster_sha256"], str(world.root)))
    requests, record = _load_warmup_requests(SimpleNamespace(budget={}, split="validation"), source, 4, None, evaluated)
    assert [request.sample_id for request in requests] == [task["sample_id"] for task in roster["tasks"]]
    assert record["disjoint_from_roster"] is True and record["source"] == f"roster:{roster['roster_sha256']}"
    with pytest.raises(ValueError, match="not disjoint"):
        _load_warmup_requests(
            SimpleNamespace(budget={}, split="validation"), source, 4, None, [*evaluated, *requests[:1]]
        )


def twin_id(sample_id):
    return f"v{int(sample_id):09d}"


def write_twin_snapshot(root):
    """Validation rows with the content of ``write_validation_snapshot``'s rows, another id and another index."""
    shard_dir = root / "3x3" / "validation"
    shard_dir.mkdir(parents=True)
    listed = []
    for number, members in enumerate(TR.FILES):
        rows = []
        for i in members:
            row = REHEARSAL._validation_row(i)
            row["id"] = twin_id(row["id"])
            row["metadata"] = {**row["metadata"], "index": 100 + i}
            rows.append(row)
        path = shard_dir / f"data-3-{number:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
        data = path.read_bytes()
        listed.append(
            {"path": f"3x3/validation/{path.name}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        )
    receipt = {"schema": R.POINTER_SCHEMA, "repo_id": "test/passage", "revision": "1" * 40, "tag": "v", "files": listed}
    (root / "PINNED.json").write_text(json.dumps(receipt, sort_keys=True))
    return root


def twin_pool(root):
    def build(n):
        return TR.BUILD.build_roster(
            root,
            benchmark_id="synthetic-warmup-v1",
            resolution=3,
            split="validation",
            n_expected=n,
            salt=SALT,
            require_footprints=False,
        )[0]

    return build


def test_a_twin_of_an_evaluated_input_with_other_metadata_is_never_a_warmup_input(world, tmp_path):
    twins = write_twin_snapshot(tmp_path / "twins")
    roster, selection = W.build_warmup_roster(
        twin_pool(twins), salt=SALT, evaluated_roster=world.evaluated, benchmark_id="synthetic-warmup-v1", k=4
    )
    evaluated = {task["sample_id"]: task["input_identity"] for task in world.evaluated["tasks"]}
    twin_of_evaluated = {twin_id(sample_id) for sample_id in evaluated}
    expected = [i for i in ranked(SALT, [twin_id(i) for i in IDS]) if i not in twin_of_evaluated][:4]
    assert selection["selected_ids"] == expected == [task["sample_id"] for task in roster["tasks"]]
    assert selection["skipped_evaluated_inputs"] and set(selection["skipped_evaluated_inputs"]) <= twin_of_evaluated
    assert selection["skipped_evaluated_ids"] == []
    assert roster["selection"]["warmup"]["disjointness_checked_by"] == "sample_id_and_input_content_identity"
    by_twin = {twin_id(sample_id): record for sample_id, record in evaluated.items()}
    for task in twin_pool(twins)(12)["tasks"]:  # the twins differ from the evaluated inputs in metadata only
        if task["sample_id"] in by_twin:
            original = by_twin[task["sample_id"]]
            assert [f for f in R.IDENTITY_FIELDS if original[f] != task["input_identity"][f]] == [
                "ordinary_metadata_hash"
            ]


def test_the_supervisor_refuses_warmup_inputs_whose_content_equals_a_roster_input(world, tmp_path):
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.supervision.supervisor import RosterInput, WarmupInput, _load_warmup_requests

    twins = write_twin_snapshot(tmp_path / "twins")
    roster, _ = W.build_warmup_roster(
        twin_pool(twins), salt=SALT, evaluated_roster=world.evaluated, benchmark_id="synthetic-warmup-v1", k=4
    )
    path = tmp_path / "warmup.json"
    path.write_text(json.dumps(roster))
    source = WarmupInput(roster=RosterInput(str(path), roster["roster_sha256"], str(twins)))
    spec = SimpleNamespace(budget={}, split="validation")
    evaluated = [PlanRequest.from_scene(scene) for scene in R.load_roster_scenes(world.evaluated, world.root)]
    requests, record = _load_warmup_requests(spec, source, 4, None, evaluated)
    assert record["disjointness_checked_by"] == "sample_id_and_input_content_identity"
    every_row, _ = TR.BUILD.build_roster(
        world.root, benchmark_id="all", resolution=3, split="validation", n_expected=12, require_footprints=False
    )
    originals = [PlanRequest.from_scene(scene) for scene in R.load_roster_scenes(every_row, world.root)]
    assert not {request.sample_id for request in requests} & {request.sample_id for request in originals}
    with pytest.raises(ValueError, match="0 shared ids, 4 shared inputs"):
        _load_warmup_requests(spec, source, 4, None, originals)


def test_the_warmup_pool_must_match_the_salt_cell_split_and_size(world):
    good = pool_builder(world, [])
    for build, match in (
        (lambda n: {**good(n), "roster_sha256": "0" * 64}, "roster_sha256"),
        (lambda n: good(n + 1), "this salt and size"),
        (
            lambda n: TR.BUILD.build_roster(
                world.root, benchmark_id="x", resolution=3, split="validation", n_expected=n, salt="other/"
            )[0],
            "this salt and size",
        ),
    ):
        with pytest.raises(ValueError, match=match):
            W.build_warmup_roster(build, salt=SALT, evaluated_roster=world.evaluated, benchmark_id="w", k=4)
    with pytest.raises(ValueError, match="fewer than 7"):
        W.build_warmup_roster(good, salt=SALT, evaluated_roster=world.evaluated, benchmark_id="w", k=7)
    with pytest.raises(ValueError, match="salt"):
        W.build_warmup_roster(good, salt="", evaluated_roster=world.evaluated, benchmark_id="w", k=4)


# ------------------------------------------------------------------------------------------------- the script


def script_arguments(world, evaluated_path, out, *extra):
    return [
        "--dataset-root",
        str(world.root),
        "--salt",
        SALT,
        "--evaluated-roster",
        str(evaluated_path),
        "--evaluated-roster-sha256",
        world.evaluated["roster_sha256"],
        "--k",
        "4",
        "--out-dir",
        str(out),
        *extra,
    ]


@pytest.fixture
def evaluated_path(world, tmp_path):
    path = tmp_path / "evaluated.json"
    path.write_text(json.dumps(world.evaluated))
    return path


def test_the_script_writes_the_roster_its_sidecar_and_a_receipt(world, evaluated_path, tmp_path, capsys):
    out = tmp_path / "out"
    assert SCRIPT.main(script_arguments(world, evaluated_path, out)) == 0
    printed = json.loads(capsys.readouterr().out)
    stem = out / "passage-validation-warmup-v1-r0003-warmup-roster-v2"
    roster = R.load_roster(stem.with_suffix(".json"))  # checks the sidecar too
    receipt = json.loads(Path(f"{stem}.receipt.json").read_text())
    assert printed["roster_sha256"] == roster["roster_sha256"] == receipt["roster_sha256"]
    assert receipt["selection"]["selected_ids"] == [task["sample_id"] for task in roster["tasks"]]
    assert receipt["decision_status"] == "proposed_not_sealed" and receipt["shared_root"] is False
    assert not any(name.startswith("path_") for name in receipt["label_policy"]["columns_requested"])
    with pytest.raises(FileExistsError):
        SCRIPT.main(script_arguments(world, evaluated_path, out))
    with pytest.raises(ValueError, match="expected pin"):
        SCRIPT.main(script_arguments(world, evaluated_path, out, "--evaluated-roster-sha256", "0" * 64))
    with pytest.raises(SystemExit, match="validation rosters"):
        SCRIPT.main(script_arguments(world, evaluated_path, out, "--benchmark-id", "passage-test-warmup"))
    with pytest.raises(SystemExit):
        SCRIPT.main([a for a in script_arguments(world, evaluated_path, out) if a not in {"--salt", SALT}])


def test_a_root_with_other_splits_needs_the_shared_root_flag_and_its_other_files_are_never_read(
    world, evaluated_path, tmp_path
):
    import shutil

    shared = tmp_path / "shared"
    shutil.copytree(world.root, shared)
    shard = shared / "3x3" / "test" / "data-3-0000.parquet"
    shard.parent.mkdir(parents=True)
    shard.write_bytes(b"synthetic placeholder, not parquet")  # any read or hash check of it would fail
    receipt = json.loads((shared / "PINNED.json").read_text())
    receipt["files"].append({"path": "3x3/test/data-3-0000.parquet", "size": 1, "sha256": "0" * 64})
    receipt.pop("subset")
    (shared / "PINNED.json").write_text(json.dumps(receipt, sort_keys=True))
    # The evaluated roster pins the original root; rebuild it on the shared root so both pin the same PINNED.json.
    evaluated, _ = TR.BUILD.build_roster(
        shared, benchmark_id="synthetic-evaluated-v1", resolution=3, split="validation", n_expected=6, salt="eval/"
    )
    path = tmp_path / "evaluated-shared.json"
    path.write_text(json.dumps(evaluated))
    arguments = [
        "--dataset-root",
        str(shared),
        "--salt",
        SALT,
        "--evaluated-roster",
        str(path),
        "--evaluated-roster-sha256",
        evaluated["roster_sha256"],
        "--k",
        "4",
        "--out-dir",
        str(tmp_path / "out"),
    ]
    with pytest.raises(ValueError, match="allow-shared-root"):
        SCRIPT.main(arguments)
    assert SCRIPT.main([*arguments, "--allow-shared-root"]) == 0
    roster = R.load_roster(tmp_path / "out" / "passage-validation-warmup-v1-r0003-warmup-roster-v2.json")
    assert roster["dataset_root_identity"] == evaluated["dataset_root_identity"]
    assert all(task["source_file"].startswith("3x3/validation/") for task in roster["tasks"])
    assert all(record["path"].startswith("3x3/validation/") for record in roster["source_files"])


# ------------------------------------------------------------------------------------------------ paired order


def expected_order(seed, batch, procedures):
    def key(procedure):
        payload = json.dumps(
            ["passage-paired-order/v1", seed, batch, procedure], sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    return sorted(procedures, key=lambda procedure: (key(procedure), procedure))


def test_the_paired_order_is_a_deterministic_sha256_order_independent_of_input_order():
    procedures = ["astar", "seg-2026", "reg-2026"]
    orders = [W.paired_order(2026, batch, procedures) for batch in range(16)]
    assert orders == [expected_order(2026, batch, procedures) for batch in range(16)]
    assert all(W.paired_order(2026, batch, list(reversed(procedures))) == orders[batch] for batch in range(16))
    assert len({tuple(order) for order in orders}) > 1  # the order changes across batches
    assert W.paired_order("2026", 0, procedures) == expected_order("2026", 0, procedures)
    schedule = W.paired_schedule(2026, procedures, 16)
    assert schedule["orders"] == orders and schedule["procedures"] == sorted(procedures)
    assert schedule["granularity"] == "per_batch" and schedule["decision_status"] == "proposed_not_sealed"
    body = {key: value for key, value in schedule.items() if key != "schedule_sha256"}
    assert schedule["schedule_sha256"] == R.sha256_hex(R.canonical_json(body))
    assert W.paired_schedule(2027, procedures, 16)["schedule_sha256"] != schedule["schedule_sha256"]


@pytest.mark.parametrize(
    "seed, batch, procedures",
    [
        (None, 0, ["a", "b"]),
        (True, 0, ["a", "b"]),
        ("", 0, ["a", "b"]),
        (1, -1, ["a", "b"]),
        (1, 1.0, ["a", "b"]),
        (1, 0, []),
        (1, 0, ["a", "a"]),
        (1, 0, "ab"),
        (1, 0, ["a", ""]),
    ],
)
def test_invalid_order_inputs_are_refused(seed, batch, procedures):
    with pytest.raises(ValueError):
        W.paired_order(seed, batch, procedures)


def test_the_supervisor_refuses_a_test_split_or_test_roster_as_warmup_source(tmp_path, monkeypatch):
    from passage_eval import cli
    from passage_eval.supervision.__main__ import main as supervision_main
    from passage_eval.supervision.supervisor import RosterInput, WarmupInput, _load_warmup_requests

    def never(*args, **kwargs):
        raise AssertionError("a warm-up row was read")

    monkeypatch.setattr(cli, "_load_scenes", never)
    monkeypatch.setattr(R, "load_roster_scenes", never)
    for name in ("test", "Test", " test "):
        with pytest.raises(TestSplitRefused, match="warm-up refuses the test split"):
            WarmupInput(split=name)
        with pytest.raises(TestSplitRefused):
            WarmupInput(split=name, allow_test="yes")
    with pytest.raises(TestSplitRefused, match="admission grant"):  # allow_test alone is not an admission
        WarmupInput(split="test", allow_test=True)
    allowed = WarmupInput(split="test", allow_test=True, admission=TR.green_admission())  # all-GREEN synthetic grant
    object.__setattr__(allowed, "allow_test", False)  # a source changed after construction is refused at loading
    with pytest.raises(TestSplitRefused, match="warm-up refuses the test split"):
        _load_warmup_requests(SimpleNamespace(budget={}, split="validation"), allowed, 4, None, [])
    synthetic_test = TR.write_dataset(tmp_path / "synthetic-test")  # a synthetic fixture whose split is named test
    roster, _ = TR.build(synthetic_test)
    path = tmp_path / "test-roster.json"
    path.write_text(json.dumps(roster))
    source = WarmupInput(roster=RosterInput(str(path), roster["roster_sha256"], str(synthetic_test)))
    with pytest.raises(TestSplitRefused, match="warm-up refuses the test split"):
        _load_warmup_requests(SimpleNamespace(budget={}, split="validation"), source, 4, None, [])
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"method_name": "m", "family": "f", "architecture": "a", "inference": "x:Y"}))
    base = ["run", "--spec", str(spec), "--output", str(tmp_path / "run"), "--worker-python", sys.executable]
    base += ["--isolation", "process", "--candidate-path", str(tmp_path), "--warmup-split", "test"]
    with pytest.raises(TestSplitRefused, match="warm-up refuses the test split"):
        supervision_main(base)
    record = tmp_path / "admission.json"
    record.write_text("{}")
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        supervision_main([*base, "--allow-test-warmup", "--test-admission-record", str(record)])
    with pytest.raises(TestSplitRefused, match="--allow-test-warmup needs a verified test admission record"):
        supervision_main([*base, "--allow-test-warmup"])
    assert not (tmp_path / "run").exists()
