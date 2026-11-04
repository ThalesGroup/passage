"""Roster build, validation and fail-closed loading on tiny synthetic Parquet shards.

The fixtures mimic the released schema (id, tensor, struct metadata with tile,
markers, calibration, solvers and paths, ``path_*`` oracle columns). Expected
values are recomputed here from first principles, not through the module under
test.
"""

import copy
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from passage.utils import encode_tensor_blob
from passage_eval.io import roster as R
from passage_eval.protocol.contracts import GraphSpec
from passage_eval.protocol.geographic_blocks import source_window_blocks

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_public_test_roster.py"
BENCHMARK = "synthetic-test-v1"
SALT = f"{BENCHMARK}/r0003/roster-v1/"
MARKER = "UNREAD-RECEIPT-MARKER-0123456789"


def _builder():
    spec = importlib.util.spec_from_file_location("build_public_test_roster_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILD = _builder()

# Row i -> source tiles (lat, lon) used by its crop. Rows 0-5 chain through the
# 5-degree cells (18,36)-(19,36); rows 6-8 sit elsewhere; row 10 spans the
# antimeridian; row 11 has no recorded footprint.
TILES = {
    0: [(0, 0)],
    1: [(1, 1)],
    2: [(2, 2)],
    3: [(3, 3)],
    4: [(4, 0), (5, 0)],
    5: [(5, 0)],
    6: [(-50, -70)],
    7: [(-49, -69)],
    8: [(-48, -68)],
    9: [(0, 179)],
    10: [(0, 179), (0, 180)],
    11: [],
}
FILES = [range(0, 5), range(5, 9), range(9, 12)]


def _tensor(i):
    tensor = np.zeros((3, 3, 3), dtype=np.float32)
    tensor[:, :, 0] = np.float32(0.1 + 0.001 * i)
    tensor[:, :, 1] = -1
    tensor[0, 0, 1] = 0
    tensor[2, 2, 1] = 1
    tensor[0, 2, 2] = 1
    return tensor


def _row(i, *, poisoned=False, split="test"):
    tiles = TILES[i]
    windows = [{"lat": lat, "lon": lon, "name": f"tile-{lat}-{lon}"} for lat, lon in tiles]
    route = [[0, 0], [1, 1], [2, 2]]
    return {
        "id": f"{i:010d}",
        "tensor": encode_tensor_blob(_tensor(i)),
        "metadata": {
            "resolution": 3,
            "split": split,
            "index": i,
            "calibration": {"min": -100.0, "max": 100.0},
            "markers": {"start_point": {"i": 0, "j": 0}, "end_point": {"i": 2, "j": 2}},
            "tile": {
                "source": {"lat": tiles[0][0] if tiles else 0, "lon": tiles[0][1] if tiles else 0, "name": "s"},
                "used": windows,
                "overlapping": windows,
            },
            "solvers": {"astar_elevation": {"cost_model": "elevation", "weight": 62.4, "diagonal_movement": "always"}},
            "paths": {
                "astar_elevation": {
                    "obstacles": {"cost": -1.0 if poisoned else 84.85},
                    "free": {"cost": -1.0 if poisoned else 84.85},
                }
            },
        },
        "metadata_json": "not json" if poisoned else "{}",
        "scene_receipt": MARKER if not poisoned else "poisoned",
        "path_astar_elevation_obstacles": [[99, 99]] if poisoned else route,
        "path_astar_elevation_free": [[99, 99]] if poisoned else route,
    }


def write_dataset(root, *, poisoned=False, files=FILES, extra_rows=None, split="test"):
    """Write synthetic 3x3/<split> shards (uncompressed, 2-row groups) plus a PINNED.json receipt.

    The default split name is ``test`` for these synthetic fixtures only; the builder refuses that name unless
    ``build_roster`` gets ``allow_test=True`` and an all-GREEN admission grant (see ``build``).
    """
    shard_dir = root / "3x3" / split
    shard_dir.mkdir(parents=True)
    listed = []
    for number, members in enumerate(files):
        rows = [_row(i, poisoned=poisoned, split=split) for i in members]
        if number == 0 and extra_rows:
            rows += extra_rows
        path = shard_dir / f"data-3-{number:04d}.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
        data = path.read_bytes()
        listed.append(
            {
                "path": f"3x3/{split}/{path.name}",
                "size": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "hub_blob_id": "b" * 40,
            }
        )
    write_receipt(root, listed)
    return root


def write_receipt(root, listed):
    receipt = {
        "schema": R.POINTER_SCHEMA,
        "repo_id": "test/passage",
        "revision": "1" * 40,
        "tag": "vtest",
        "files": listed,
    }
    (root / "PINNED.json").write_text(json.dumps(receipt, sort_keys=True))


def build(root, **kwargs):
    kwargs.setdefault("benchmark_id", BENCHMARK)
    kwargs.setdefault("resolution", 3)
    kwargs.setdefault("n_expected", 6)
    # Row 11 has no footprint and is selected: these fixtures exercise a roster without planned block analysis.
    # The builder's default (footprints required) is tested in test_roster_blocks.py.
    kwargs.setdefault("require_footprints", False)
    # The synthetic fixture's split is named test; the builder refuses that name without an all-GREEN admission.
    if kwargs.get("split", "test") == "test":
        kwargs.setdefault("allow_test", True)
        kwargs.setdefault("admission", green_admission())
    return BUILD.build_roster(root, **kwargs)


def green_admission():
    """The all-GREEN synthetic test admission grant of ``campaign_world.py`` (synthetic fixtures named test only)."""
    import sys

    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parent / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name].green_grant()


def rehash(roster):
    roster = copy.deepcopy(roster)
    roster["roster_sha256"] = R.compute_roster_sha256(roster)
    return roster


def retarget(roster, root):
    """Point a roster at another tree's file sizes: only the input identity can then object."""
    roster = copy.deepcopy(roster)
    for record in roster["source_files"]:
        record["size"] = (root / record["path"]).stat().st_size
    return rehash(roster)


def expected_order(n):
    ids = [f"{i:010d}" for i in range(12)]
    return sorted(ids, key=lambda i: hashlib.sha256((SALT + i).encode("utf-8")).hexdigest())[:n]


def grid_hash(grid):
    grid = np.ascontiguousarray(grid)
    return hashlib.sha256(grid.dtype.str.encode() + repr(grid.shape).encode() + grid.tobytes()).hexdigest()


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    return write_dataset(tmp_path_factory.mktemp("clean"))


@pytest.fixture(scope="module")
def built(dataset):
    return build(dataset)


@pytest.fixture
def written(tmp_path, dataset, built):
    roster, info = built
    paths = BUILD.write_outputs(
        roster, info, tmp_path / "rosters", argv=["test"], dataset_root=dataset, verify_hashes=True
    )
    return roster, paths


def save(tmp_path, roster, name="r.json"):
    path = tmp_path / name
    path.write_text(json.dumps(roster))
    return path


def copy_dataset(dataset, tmp_path):
    target = tmp_path / "copy"
    shutil.copytree(dataset, target)
    return target


# --------------------------------------------------------------------------- build


def test_tasks_follow_the_independent_hash_ranking(built):
    roster, _ = built
    assert [t["sample_id"] for t in roster["tasks"]] == expected_order(6)
    assert [t["rank"] for t in roster["tasks"]] == list(range(6))
    for task in roster["tasks"]:
        assert task["selection_key"] == hashlib.sha256((SALT + task["sample_id"]).encode("utf-8")).hexdigest()
    assert roster["selection"]["salt"] == SALT and roster["selection"]["n_selected"] == 6
    assert roster["selection"]["population"]["n_rows"] == 12


def test_source_file_and_row_index_locate_the_listed_row(built, dataset):
    roster, _ = built
    for task in roster["tasks"]:
        table = pq.ParquetFile(dataset / task["source_file"]).read(columns=["id"])
        assert table.column("id")[task["row_index"]].as_py() == task["sample_id"]
    assert [r["path"] for r in roster["source_files"]] == sorted(r["path"] for r in roster["source_files"])
    assert sum(r["n_rows"] for r in roster["source_files"]) == 12


def test_input_identity_equals_independent_hashes_of_the_ordinary_inputs(built):
    roster, _ = built
    for task in roster["tasks"]:
        i = int(task["sample_id"])
        tensor = _tensor(i)
        elevation = tensor[:, :, 0].astype(np.float64) * 200.0 + (-100.0)
        metadata = {
            "split": "test",
            "resolution": 3,
            "index": i,
            "calibration": {"min": -100.0, "max": 100.0},
            "task_variant": "obstacles",
        }
        identity = task["input_identity"]
        assert identity["elevation_hash"] == grid_hash(elevation)
        assert identity["obstacles_hash"] == grid_hash(tensor[:, :, 2].astype(bool))
        assert identity["input_tensor_hash"] == grid_hash(tensor)
        assert (
            identity["ordinary_metadata_hash"]
            == hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        )
        assert identity["start"] == [0, 0] and identity["goal"] == [2, 2] and identity["shape"] == [3, 3]
        assert identity["cost_model"] == "elevation" and identity["cost_weight"] == 62.4
        assert identity["graph_id"] == GraphSpec(cost_model="elevation", cost_weight=62.4).identity()
        assert set(identity) == set(R.IDENTITY_FIELDS) and "budget" not in identity
        assert (
            task["input_identity_sha256"]
            == hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        )


def test_roster_sha256_is_the_canonical_json_digest_and_sidecar_receipt_agree(written):
    roster, paths = written
    body = {k: v for k, v in json.loads(paths["roster"].read_text()).items() if k != "roster_sha256"}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    assert digest == roster["roster_sha256"]
    assert paths["sidecar"].read_text().split() == [digest]
    receipt = json.loads(paths["receipt"].read_text())
    assert receipt["roster"]["roster_sha256"] == digest
    assert receipt["roster"]["roster_file_sha256"] == hashlib.sha256(paths["roster"].read_bytes()).hexdigest()
    assert receipt["inputs"]["pinned_receipt_sha256"] == roster["dataset_root_identity"]["sha256"]
    assert receipt["code"]["build_script"]["sha256"] == hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    assert receipt["command"]["cpu_affinity"] and receipt["date_utc"].endswith("Z")
    assert "oracle" not in paths["roster"].read_text().lower() and "path_astar" not in paths["roster"].read_text()


def test_build_requests_no_label_column_and_ignores_poisoned_labels(tmp_path, built, monkeypatch):
    requested = []
    for name in ("read", "read_row_group"):
        original = getattr(pq.ParquetFile, name)

        def spy(self, *args, _original=original, **kwargs):
            requested.append(list(kwargs.get("columns") or args[-1:]))
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(pq.ParquetFile, name, spy)
    poisoned_root = write_dataset(tmp_path / "poisoned", poisoned=True)
    roster, _ = build(poisoned_root)
    flat = {column for call in requested for column in call if isinstance(column, str)}
    assert flat and flat <= BUILD.ALLOWED_COLUMNS
    assert not any(
        c.startswith("path_") or c in {"metadata", "metadata.paths", "metadata_json", "scene_receipt"} for c in flat
    )
    clean, _ = built
    assert roster["tasks"] == clean["tasks"]
    with pytest.raises(PermissionError):
        BUILD.read_columns(
            pq.ParquetFile(poisoned_root / roster["source_files"][0]["path"]), ["path_astar_elevation_obstacles"]
        )
    with pytest.raises(PermissionError):
        BUILD.read_columns(pq.ParquetFile(poisoned_root / roster["source_files"][0]["path"]), ["metadata"])
    with pytest.raises(ValueError, match="oracle"):  # the scorer-side loader does read and validate labels
        R.load_roster_scenes(roster, poisoned_root)


def test_geographic_blocks_merge_crossing_tasks_and_flag_missing_footprints(tmp_path):
    root = write_dataset(tmp_path / "all")
    roster, info = build(root, n_expected=12)
    by_id = {t["sample_id"]: t for t in roster["tasks"]}

    def block(i):
        return by_id[f"{i:010d}"]["geographic_block"]

    assert len({block(i) for i in range(6)}) == 1 and block(0) == "geo5:18:36"
    assert len({block(i) for i in (6, 7, 8)}) == 1 and block(6) != block(0)
    assert block(9) == block(10) and block(9) != block(0)
    assert block(11) is None and by_id["0000000011"]["footprint_status"] == "missing"
    summary = roster["geographic_blocks"]
    assert summary["number_of_blocks"] == 3 and summary["crossing_task_count"] == 2
    assert summary["number_of_tasks"] == 11 and summary["n_tasks_without_footprint"] == 1
    assert summary["statistical_independence_proven"] is False and "task_blocks" not in summary
    direct = source_window_blocks(
        [
            {"sample_id": t["sample_id"], "source_windows": t["source_windows"]}
            for t in roster["tasks"]
            if t["source_windows"]
        ],
        block_degrees=5,
    )
    assert {t["sample_id"]: t["geographic_block"] for t in roster["tasks"] if t["source_windows"]} == direct[
        "task_blocks"
    ]
    assert info["block_statistics"]["number_of_blocks"] == 3 and info["block_statistics"]["max_tasks_per_block"] == 6
    assert info["footprint_status"] == {"ok": 11, "missing": 1}


def test_free_variant_roster_loads_with_its_own_label_column(tmp_path):
    root = write_dataset(tmp_path / "free")
    roster, _ = build(root, task_variant="free")
    assert roster["cell"]["task_variant"] == "free"
    scenes = R.load_roster_scenes(roster, root)
    assert all(not scene.obstacles.any() and not scene.input_tensor[:, :, 2].any() for scene in scenes)


BUILD_FAILURES = {
    "duplicate_id": "Duplicate row id",
    "too_small": "smaller than n_expected",
    "weight": "differs from the cell weight",
    "size": "size differs",
    "bytes": "sha256 differs",
    "missing_file": "missing or not a regular file",
    "no_receipt": "PINNED.json",
}


@pytest.mark.parametrize("failure", sorted(BUILD_FAILURES))
def test_build_fails_closed(tmp_path, failure):
    extra = [_row(0)] if failure == "duplicate_id" else None
    root = write_dataset(tmp_path / "d", extra_rows=extra)
    first = sorted((root / "3x3" / "test").glob("*.parquet"))[0]
    kwargs = {}
    if failure == "duplicate_id":  # the receipt must describe the final files so only the duplicate objects
        listed = [
            {
                "path": f"3x3/test/{p.name}",
                "size": p.stat().st_size,
                "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            }
            for p in sorted((root / "3x3" / "test").glob("*.parquet"))
        ]
        write_receipt(root, listed)
    elif failure == "too_small":
        kwargs["n_expected"] = 13
    elif failure == "weight":
        kwargs["cost_weight"] = 1.0
    elif failure == "size":
        with first.open("ab") as handle:
            handle.write(b"x")
    elif failure == "bytes":
        data = bytearray(first.read_bytes())
        data[data.index(MARKER.encode())] ^= 1
        first.write_bytes(bytes(data))
    elif failure == "missing_file":
        first.unlink()
    elif failure == "no_receipt":
        (root / "PINNED.json").unlink()
    with pytest.raises((ValueError, OSError), match=BUILD_FAILURES[failure]):
        build(root, **kwargs)


def test_the_builder_refuses_the_test_split_before_reading_any_file(dataset, tmp_path, monkeypatch):
    from passage_eval.protocol.guards import TestSplitRefused

    def never(*args, **kwargs):
        raise AssertionError("a dataset file was read")

    monkeypatch.setattr(BUILD, "load_pinned_receipt", never)
    args = ["--dataset-root", str(dataset), "--out-dir", str(tmp_path / "o"), "--resolution", "3", "--n", "6"]
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        BUILD.main(args)  # the default split is test
    record = tmp_path / "admission.json"
    record.write_text("{}")
    with pytest.raises(TestSplitRefused, match="not all GREEN"):
        BUILD.main([*args, "--allow-test-split", "--test-admission-record", str(record)])
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        BUILD.build_roster(dataset, benchmark_id=BENCHMARK, resolution=3, n_expected=6)
    with pytest.raises(TestSplitRefused, match="admission grant"):  # allow_test alone is not an admission
        BUILD.build_roster(dataset, benchmark_id=BENCHMARK, resolution=3, n_expected=6, allow_test=True)
    assert not (tmp_path / "o").exists()


def test_outputs_are_not_overwritten_and_cli_writes_three_files(tmp_path, capsys):
    out = tmp_path / "out"
    validation = write_dataset(tmp_path / "validation", split="validation")
    args = [
        "--dataset-root",
        str(validation),
        "--out-dir",
        str(out),
        "--benchmark-id",
        BENCHMARK,
        "--resolution",
        "3",
        "--split",
        "validation",
        "--n",
        "6",
        "--allow-missing-footprints",  # row 11 is selected and has no footprint
    ]
    assert BUILD.main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        f"{BENCHMARK}-r0003-roster-v2.json",
        f"{BENCHMARK}-r0003-roster-v2.receipt.json",
        f"{BENCHMARK}-r0003-roster-v2.sha256",
    ]
    assert json.loads(capsys.readouterr().out)["roster_sha256"] == R.load_roster(out / names[0])["roster_sha256"]
    with pytest.raises(FileExistsError):
        BUILD.main(args)
    assert BUILD.main(args + ["--overwrite"]) == 0


# ------------------------------------------------------------------------ load_roster


def test_load_roster_round_trips_and_pins(written):
    roster, paths = written
    loaded = R.load_roster(paths["roster"])
    assert loaded == json.loads(json.dumps(roster))
    assert R.load_roster(paths["roster"], expected_sha256=roster["roster_sha256"]) == loaded
    with pytest.raises(ValueError, match="expected pin"):
        R.load_roster(paths["roster"], expected_sha256="0" * 64)


def _mutate(roster, kind):
    roster = copy.deepcopy(roster)
    tasks = roster["tasks"]
    if kind == "sample_id":
        tasks[2]["sample_id"] = "0000000099"
    elif kind == "identity_hash":
        tasks[1]["input_identity"]["elevation_hash"] = "0" * 64
    elif kind == "swap_order":
        tasks[0], tasks[1] = tasks[1], tasks[0]
    elif kind == "drop_task":
        tasks.pop()
    elif kind == "salt":
        roster["selection"]["salt"] += "x"
    elif kind == "schema":
        roster["schema"] = "passage-roster-v0"
    elif kind == "digest":
        roster["roster_sha256"] = "0" * 64
    elif kind == "extra_key":
        roster["unreviewed"] = 1
    elif kind == "commit":
        roster["dataset"]["commit"] = "2" * 40
    elif kind == "row_index":
        tasks[0]["row_index"] += 1
    return roster


@pytest.mark.parametrize(
    "kind",
    [
        "sample_id",
        "identity_hash",
        "swap_order",
        "drop_task",
        "salt",
        "schema",
        "digest",
        "extra_key",
        "commit",
        "row_index",
    ],
)
def test_tampered_roster_is_rejected(tmp_path, built, kind):
    roster, _ = built
    with pytest.raises(ValueError):
        R.load_roster(save(tmp_path, _mutate(roster, kind)))


@pytest.mark.parametrize(
    "kind", ["sample_id", "identity_hash", "swap_order", "drop_task", "salt", "schema", "extra_key", "row_index"]
)
def test_tamper_with_recomputed_digest_is_still_rejected(tmp_path, built, kind):
    roster, _ = built
    with pytest.raises(ValueError):
        R.load_roster(save(tmp_path, rehash(_mutate(roster, kind))))


@pytest.mark.parametrize(
    "kind", ["duplicate_task", "path_escape", "block", "n_expected", "missing_footprint_with_block", "rank"]
)
def test_structural_and_footprint_attacks_with_valid_digest_are_rejected(tmp_path, built, kind):
    roster = copy.deepcopy(built[0])
    tasks = roster["tasks"]
    if kind == "duplicate_task":
        tasks[3] = copy.deepcopy(tasks[2])
    elif kind == "path_escape":
        roster["source_files"][0]["path"] = "../escape.parquet"
        for task in tasks:
            task["source_file"] = (
                task["source_file"]
                if task["source_file"] != built[0]["source_files"][0]["path"]
                else "../escape.parquet"
            )
    elif kind == "block":
        tasks[0]["geographic_block"] = "geo5:00:00"
    elif kind == "n_expected":
        roster["selection"]["n_expected"] = 7
    elif kind == "missing_footprint_with_block":
        tasks[0]["source_windows"] = []
        tasks[0]["footprint_status"] = "missing"
    elif kind == "rank":
        tasks[1]["rank"] = 5
    with pytest.raises(ValueError):
        R.load_roster(save(tmp_path, rehash(roster)))


def test_nan_duplicate_keys_and_bad_json_are_rejected(tmp_path, written):
    _, paths = written
    text = paths["roster"].read_text()
    for name, bad in (
        ("nan", text.replace('"cost_weight": 62.4', '"cost_weight": NaN', 1)),
        ("dup", text.replace('"schema":', '"schema": "x", "schema":', 1)),
        ("junk", text[:-20]),
    ):
        path = tmp_path / f"{name}.json"
        path.write_text(bad)
        with pytest.raises(ValueError):
            R.load_roster(path)


def test_sidecar_mismatch_is_rejected(tmp_path, written):
    roster, paths = written
    copy_path = tmp_path / "copy.json"
    shutil.copy(paths["roster"], copy_path)
    copy_path.with_suffix(".sha256").write_text("f" * 64 + "\n")
    with pytest.raises(ValueError, match="sidecar"):
        R.load_roster(copy_path)


# ------------------------------------------------------------------ load_roster_scenes


def test_scenes_are_the_listed_rows_in_roster_order(built, dataset):
    roster, _ = built
    scenes = R.load_roster_scenes(roster, dataset)
    assert [s.sample_id for s in scenes] == expected_order(6)
    for scene in scenes:
        i = int(scene.sample_id)
        np.testing.assert_array_equal(scene.input_tensor, _tensor(i))
        assert scene.start == (0, 0) and scene.goal == (2, 2) and scene.cost_weight == 62.4
        assert scene.oracle_cost == pytest.approx(84.85)  # labels are present: redact before any worker


def test_loader_revalidates_a_mutated_in_memory_roster(built, dataset):
    roster = copy.deepcopy(built[0])
    roster["benchmark_id"] += "-edited"
    with pytest.raises(ValueError, match="roster_sha256"):
        R.load_roster_scenes(roster, dataset)


def test_changed_input_is_detected_by_hash_and_by_identity(tmp_path, built, dataset):
    roster, _ = built
    target = copy_dataset(dataset, tmp_path)
    victim = roster["tasks"][2]
    path = target / victim["source_file"]
    rows = pq.ParquetFile(path).read().to_pylist()
    rows[victim["row_index"]]["tensor"] = encode_tensor_blob(_tensor(11))  # valid input, same file size
    pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
    assert (
        path.stat().st_size
        == roster["source_files"][[r["path"] for r in roster["source_files"]].index(victim["source_file"])]["size"]
    )
    with pytest.raises(ValueError, match="sha256 differs"):  # default: the pinned bytes no longer match
        R.load_roster_scenes(roster, target)
    with pytest.raises(ValueError, match=rf"{victim['sample_id']}: input identity differs .*elevation_hash"):
        R.load_roster_scenes(roster, target, verify_file_hashes=False)  # the identity alone still objects


def test_same_size_corruption_of_an_unread_column_is_found_only_by_rehashing(tmp_path, built, dataset):
    roster, _ = built
    target = copy_dataset(dataset, tmp_path)
    path = target / roster["source_files"][0]["path"]
    data = bytearray(path.read_bytes())
    data[data.index(MARKER.encode())] ^= 1  # an unread column: same size, rows still decode
    path.write_bytes(bytes(data))
    assert [s.sample_id for s in R.load_roster_scenes(roster, target, verify_file_hashes=False)] == expected_order(6)
    with pytest.raises(ValueError, match="sha256 differs"):
        R.load_roster_scenes(roster, target)


LOAD_FAILURES = {
    "id_changed": "holds id",
    "short_file": "the roster records",
    "missing_file": "missing or not a regular file",
    "no_receipt": "PINNED.json",
    "receipt_edited": "receipt sha256 differs",
    "symlink": "missing or not a regular file",
}


@pytest.mark.parametrize("failure", sorted(LOAD_FAILURES))
def test_loader_fails_closed_on_dataset_changes(tmp_path, built, dataset, failure):
    roster, _ = built
    target = copy_dataset(dataset, tmp_path)
    victim = roster["tasks"][1]
    path = target / victim["source_file"]
    options = {}
    if failure in {"id_changed", "short_file"}:
        rows = pq.ParquetFile(path).read().to_pylist()
        if failure == "id_changed":
            rows[victim["row_index"]]["id"] = "0000000098"
        else:
            rows = rows[:-1]
        pq.write_table(pa.Table.from_pylist(rows), path, row_group_size=2, compression="none")
        # Retarget sizes and skip byte hashing so that only the row-level checks can object.
        roster = retarget(roster, target)
        options = {"require_root_receipt": False, "verify_file_hashes": False}
    elif failure == "missing_file":
        path.unlink()
    elif failure == "no_receipt":
        (target / "PINNED.json").unlink()
    elif failure == "receipt_edited":
        (target / "PINNED.json").write_text((target / "PINNED.json").read_text() + " ")
    elif failure == "symlink":
        moved = target / "elsewhere.parquet"
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(ValueError, match=LOAD_FAILURES[failure]):
        R.load_roster_scenes(roster, target, **options)


def test_missing_receipt_can_be_waived_only_explicitly(tmp_path, built, dataset):
    roster, _ = built
    target = copy_dataset(dataset, tmp_path)
    (target / "PINNED.json").unlink()
    with pytest.raises(ValueError, match="PINNED.json"):
        R.load_roster_scenes(roster, target)
    assert len(R.load_roster_scenes(roster, target, require_root_receipt=False)) == 6


def test_rows_swapped_inside_a_valid_roster_are_caught_by_the_row_id_check(built, dataset):
    roster = copy.deepcopy(built[0])
    by_file = {}
    for task in roster["tasks"]:
        by_file.setdefault(task["source_file"], []).append(task)
    first, second = next(group[:2] for group in by_file.values() if len(group) >= 2)
    first["row_index"], second["row_index"] = second["row_index"], first["row_index"]
    roster = rehash(roster)
    assert R.validate_roster(roster)["roster_sha256"] == roster["roster_sha256"]  # structurally and digest valid
    with pytest.raises(ValueError, match="holds id"):
        R.load_roster_scenes(roster, dataset)
