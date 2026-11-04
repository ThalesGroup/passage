"""The validation rehearsal wrapper: split guard, labelling, and a label-free build on a synthetic snapshot.

Rows come from ``test_roster._row`` with only the split label changed to ``validation``;
nothing here reads the real dataset.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from passage_eval.io import roster as R


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


HERE = Path(__file__).resolve().parent
TR = _load("roster_fixtures_for_rehearsal_tests", HERE / "test_roster.py")  # synthetic row builder only
WRAP = _load("build_rehearsal_roster_under_test", HERE.parents[1] / "scripts" / "build_rehearsal_roster.py")


def _validation_row(i):
    row = TR._row(i)
    row["metadata"] = {**row["metadata"], "split": "validation"}
    return row


def write_validation_snapshot(root, *, split="validation"):
    shard_dir = root / "3x3" / "validation"
    shard_dir.mkdir(parents=True)
    listed = []
    for number, members in enumerate(TR.FILES):
        path = shard_dir / f"data-3-{number:04d}.parquet"
        pq.write_table(
            pa.Table.from_pylist([_validation_row(i) for i in members]), path, row_group_size=2, compression="none"
        )
        data = path.read_bytes()
        listed.append(
            {
                "path": f"3x3/validation/{path.name}",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "hub_blob_id": "b" * 40,
            }
        )
    receipt = {
        "schema": R.POINTER_SCHEMA,
        "repo_id": "test/passage",
        "revision": "1" * 40,
        "tag": "vtest",
        "files": listed,
        "subset": {"config": "3x3", "split": split},
    }
    (root / "PINNED.json").write_text(json.dumps(receipt, sort_keys=True))
    return root


def test_build_labels_the_roster_as_a_validation_rehearsal(tmp_path, capsys):
    root = write_validation_snapshot(tmp_path / "val")
    out = tmp_path / "out"
    arguments = ["--dataset-root", str(root), "--out-dir", str(out), "--resolution", "3", "--n", "6"]
    with pytest.raises(ValueError, match="no usable source footprint"):  # block analysis is planned by default
        WRAP.main(arguments)
    assert WRAP.main([*arguments, "--allow-missing-footprints"]) == 0  # row 11 is selected and has no footprint
    stem = out / "passage-validation-rehearsal-v1-r0003-roster-v2"
    roster = R.load_roster(stem.with_suffix(".json"))
    assert roster["benchmark_id"] == "passage-validation-rehearsal-v1"
    assert roster["dataset"]["split"] == "validation"
    assert "not the public test benchmark" in roster["dataset"]["visibility"] and roster["dataset"]["purpose"]
    assert roster["selection"]["salt"] == "passage-validation-rehearsal-v1/r0003/roster-v1/"
    assert all(t["source_file"].startswith("3x3/validation/") for t in roster["tasks"])
    ids = [f"{i:010d}" for i in range(12)]
    expected = sorted(ids, key=lambda i: hashlib.sha256((roster["selection"]["salt"] + i).encode()).hexdigest())[:6]
    assert [t["sample_id"] for t in roster["tasks"]] == expected
    assert stem.with_suffix(".sha256").read_text().split() == [roster["roster_sha256"]]
    receipt = json.loads(Path(f"{stem}.receipt.json").read_text())
    assert receipt["rehearsal"]["is_public_test_benchmark"] is False and receipt["rehearsal"]["split"] == "validation"
    assert "development_overlap_caveat" in receipt["rehearsal"]
    assert receipt["label_policy"]["columns_requested"] and not any(
        name.startswith("path_") for name in receipt["label_policy"]["columns_requested"]
    )
    scenes = R.load_roster_scenes(roster, root)
    assert [s.sample_id for s in scenes] == expected


def test_the_test_split_and_test_benchmark_ids_are_refused(tmp_path):
    root = write_validation_snapshot(tmp_path / "val", split="test")
    with pytest.raises(ValueError, match="must declare subset"):
        WRAP.check_validation_root(root, config="3x3")
    with pytest.raises(SystemExit, match="validation rehearsal rosters only"):
        WRAP.main(["--dataset-root", str(root), "--benchmark-id", "passage-public-test-v1"])


def test_a_root_that_also_holds_the_test_split_is_refused(tmp_path):
    root = write_validation_snapshot(tmp_path / "val")
    (root / "3x3" / "test").mkdir()
    with pytest.raises(ValueError, match="validation split only"):
        WRAP.check_validation_root(root, config="3x3")


def test_a_receipt_listing_foreign_files_is_refused(tmp_path):
    root = write_validation_snapshot(tmp_path / "val")
    receipt = json.loads((root / "PINNED.json").read_text())
    receipt["files"].append({"path": "3x3/test/data-3-0000.parquet", "size": 1, "sha256": "0" * 64})
    (root / "PINNED.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="outside"):
        WRAP.check_validation_root(root, config="3x3")
