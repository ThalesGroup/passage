#!/usr/bin/env python
"""Build the label-free task roster of one PASSAGE public-test leaderboard cell.

The roster fixes which rows of a pinned Hub snapshot form the cell. It is built
from three kinds of data only: the ``id`` column (to rank), the ordinary inputs
of the selected rows (``tensor`` and the metadata fields ``resolution``,
``split``, ``index``, ``calibration``, ``markers``, ``solvers`` configuration) and
the ``metadata.tile`` footprint. It never reads a ``path_*`` column, the
``metadata.paths`` costs, ``metadata_json`` or ``scene_receipt``; every column
request goes through ``read_columns``, which refuses anything outside an
allowlist and records what was requested in the receipt. No candidate and no
reference solver is run.

Selection rule: rank every row id of the split by
``sha256((salt + id).encode("utf-8")).hexdigest()`` ascending and keep the first
``n``. The default salt is ``<benchmark_id>/r<resolution:04d>/roster-v1/``.

The test split is refused (``passage_eval.protocol.guards``) before any file is
read, unless ``--allow-test-split`` comes with a ``--test-admission-record``
whose six ``passage-test-admission/v1`` gates are GREEN (check it first with
``passage-eval admission-check``); other splits need no admission
(``build_rehearsal_roster.py`` uses the builder on validation).

Example (refused unless the admission record checks all GREEN)::

    nice -n 10 uv run --frozen python scripts/build_public_test_roster.py \\
        --dataset-root /path/to/dataset --allow-test-split --test-admission-record admission.json
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow
import pyarrow.parquet as pq

from passage.dataset import decode_scene_inputs
from passage_eval.io import roster as roster_module
from passage_eval.io.roster import (
    GEOGRAPHIC_BLOCK_DEGREES,
    IDENTITY_FIELDS,
    PINNED_RECEIPT_NAME,
    POINTER_SCHEMA,
    SCHEMA,
    compute_roster_sha256,
    geographic_block_record,
    identity_digest,
    inputs_request,
    request_input_identity,
    selection_key,
    sha256_hex,
    validate_roster,
)
from passage_eval.metrics.cost import canonical_cost_model
from passage_eval.protocol.geographic_blocks import (
    BLOCK_RULES,
    DEFAULT_BLOCK_RULE,
    source_window_blocks,
    task_footprint,
)
from passage_eval.protocol.guards import add_test_admission_arguments, refuse_test_split, test_split_allowed

RECEIPT_SCHEMA = "passage-roster-receipt-v1"
REPOSITORY = Path(__file__).resolve().parents[1]
# The only columns the build may request. Everything label-bearing is absent.
ALLOWED_COLUMNS = frozenset(
    {
        "id",
        "tensor",
        "metadata.resolution",
        "metadata.split",
        "metadata.index",
        "metadata.calibration",
        "metadata.markers",
        "metadata.tile",
        "metadata.solvers",
    }
)
INPUT_COLUMNS = [
    "id",
    "tensor",
    "metadata.resolution",
    "metadata.split",
    "metadata.index",
    "metadata.calibration",
    "metadata.markers",
    "metadata.tile",
    "metadata.solvers",
]
NOT_READ = ["path_*", "metadata.paths", "metadata_json", "scene_receipt", "preview images"]
COLUMNS_REQUESTED: set[str] = set()


def read_columns(parquet: pq.ParquetFile, columns: list[str], row_group: int | None = None):
    """The single entry point for table reads; enforces the label-free allowlist."""
    unexpected = sorted(set(columns) - ALLOWED_COLUMNS)
    if unexpected:
        raise PermissionError(f"Roster build may not read columns {unexpected}")
    COLUMNS_REQUESTED.update(columns)
    return parquet.read(columns=columns) if row_group is None else parquet.read_row_group(row_group, columns=columns)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_pinned_receipt(root: Path, config: str, split: str) -> tuple[dict[str, Any], str, list[dict[str, Any]]]:
    data = (root / PINNED_RECEIPT_NAME).read_bytes()
    receipt = json.loads(data.decode("utf-8"))
    if receipt.get("schema") != POINTER_SCHEMA:
        raise ValueError(f"{PINNED_RECEIPT_NAME} schema must be {POINTER_SCHEMA}")
    revision = receipt.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set("0123456789abcdef"):
        raise ValueError("The pinned receipt needs a 40-hex lowercase revision")
    prefix = f"{config}/{split}/"
    files = sorted((f for f in receipt.get("files", []) if f["path"].startswith(prefix)), key=lambda f: f["path"])
    if not files:
        raise ValueError(f"The pinned receipt lists no files under {prefix}")
    return receipt, sha256_hex(data), files


def verified_files(root: Path, files: list[dict[str, Any]], *, verify_hashes: bool) -> list[dict[str, Any]]:
    """Check size (and sha256) of every listed file against the pinned receipt."""
    paths = [root / f["path"] for f in files]
    for path, record in zip(paths, files):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Pinned file is missing or not a regular file: {record['path']}")
        if path.stat().st_size != record["size"]:
            raise ValueError(f"Pinned file size differs: {record['path']}")
    if verify_hashes:
        with ThreadPoolExecutor(min(12, len(paths))) as pool:
            digests = list(pool.map(file_sha256, paths))
        for digest, record in zip(digests, files):
            if digest != record["sha256"]:
                raise ValueError(f"Pinned file sha256 differs: {record['path']}")
    return files


def footprint(tile: Any) -> tuple[list[dict[str, int]], str]:
    """Integer one-degree source tiles from ``metadata.tile.used``; no labels."""
    used = tile.get("used") if isinstance(tile, dict) else None
    if not used:
        return [], "missing"
    cells = set()
    for window in used:
        lat, lon = (window.get("lat"), window.get("lon")) if isinstance(window, dict) else (None, None)
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (lat, lon)):
            return [], "malformed"
        cells.add((lat, lon))
    windows = [{"lat": lat, "lon": lon} for lat, lon in sorted(cells)]
    try:
        source_window_blocks([{"sample_id": "probe", "source_windows": windows}])
    except (TypeError, ValueError):
        return [], "malformed"
    return windows, "ok"


def released_solver_weight(metadata: dict[str, Any], cost_model: str) -> float:
    """The cell's released objective weight: solver configuration, never a label."""
    matches = [cfg for cfg in metadata["solvers"].values() if canonical_cost_model(cfg["cost_model"]) == cost_model]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one released solver for {cost_model}")
    weight = float(matches[0]["weight"])
    if not np.isfinite(weight) or weight < 0 or matches[0]["diagonal_movement"] != "always":
        raise ValueError("invalid released weight or connectivity")
    return weight


def block_statistics(blocks: dict[str, Any] | None) -> dict[str, Any] | None:
    if not blocks or not blocks.get("block_sizes"):
        return None
    sizes = sorted(blocks["block_sizes"].values())
    unplaced = blocks.get("unplaced_block")
    return {
        "rule": blocks.get("rule", DEFAULT_BLOCK_RULE),
        "n_unplaced": unplaced["n_tasks"] if unplaced else 0,
        "number_of_blocks": blocks["number_of_blocks"],
        "number_of_tasks": blocks["number_of_tasks"],
        "min_tasks_per_block": sizes[0],
        "median_tasks_per_block": statistics.median(sizes),
        "mean_tasks_per_block": round(statistics.fmean(sizes), 4),
        "max_tasks_per_block": sizes[-1],
        "singleton_blocks": sum(1 for s in sizes if s == 1),
        "crossing_task_count": blocks["crossing_task_count"],
        "grid_cell_count": blocks["grid_cell_count"],
        "size_histogram": {str(size): count for size, count in sorted(Counter(sizes).items())},
        "largest_blocks": dict(sorted(blocks["block_sizes"].items(), key=lambda kv: (-kv[1], kv[0]))[:10]),
        "descriptive_only": True,
    }


def build_roster(
    dataset_root: str | Path,
    *,
    benchmark_id: str = "passage-public-test-v1",
    resolution: int = 256,
    split: str = "test",
    cost_model: str = "elevation",
    cost_weight: float = 62.4,
    task_variant: str = "obstacles",
    n_expected: int = 1024,
    salt: str | None = None,
    verify_hashes: bool = True,
    require_footprints: bool = True,
    unplaced_block: bool = False,
    block_rule: str = DEFAULT_BLOCK_RULE,
    allow_test: bool = False,
    admission: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(roster, build_info)``; the roster has no dates, paths or labels.

    ``require_footprints`` (default on: block analysis is planned) refuses a selection in which any task lacks the
    source footprint its block rule needs, instead of leaving that task outside every block. Pass ``False`` only for
    a roster that will not be analysed by geographic blocks; the roster records the choice. ``unplaced_block``
    (block analysis planned with the sealed unplaced block of ``io.roster``) puts every task without footprint in
    the one ``unplaced`` block instead of stopping; the roster lists them and records ``footprints_required`` false.
    ``block_rule`` is ``source_window_union`` (rule A, the default: used tiles merged by transitive union) or
    ``main_tile_cell`` (rule B: the 5-degree cell of the main source tile, ``metadata.tile.source``, no union).

    The test split raises ``TestSplitRefused`` before any file is read unless ``allow_test`` is true and
    ``admission`` is an all-GREEN test admission grant, or a test preparation grant that names this reader
    (``guards.test_split_allowed``).
    """
    refuse_test_split(
        split, allow_test=allow_test, admission=admission, context="the public roster builder", reader="build_roster"
    )
    if block_rule not in BLOCK_RULES:
        raise ValueError(f"block_rule must be one of {list(BLOCK_RULES)}")
    if not isinstance(unplaced_block, bool):
        raise TypeError("unplaced_block must be a boolean")
    root = Path(dataset_root)
    cost_model = canonical_cost_model(cost_model)
    config = f"{resolution}x{resolution}"
    salt = salt if salt is not None else f"{benchmark_id}/r{resolution:04d}/roster-v1/"
    # A phased preparation approval has a strictly narrower scope than an
    # all-GREEN admission: it may build only the sealed 1024-task S roster.
    # Check that declaration before opening the receipt or any parquet file.
    from passage_eval.protocol.admission import PreparationGrant, ROSTER_SELECTION_RULE

    if isinstance(admission, PreparationGrant):
        admission.require_roster_scope(
            rule=ROSTER_SELECTION_RULE,
            salt=salt,
            n=n_expected,
            cell={"cost_model": cost_model, "cost_weight": cost_weight, "task_variant": task_variant},
            block_rule=block_rule,
        )
    pinned, pinned_sha, listed = load_pinned_receipt(root, config, split)
    if isinstance(admission, PreparationGrant):
        admission.require_dataset_scope(
            revision={
                "repo": pinned["repo_id"],
                "commit": pinned["revision"],
                "config": config,
                "resolution": resolution,
            },
            root_sha256=pinned_sha,
        )
    verified_files(root, listed, verify_hashes=verify_hashes)

    handles, source_files, layouts, ids, locations = {}, [], {}, [], []
    for record in listed:
        parquet = pq.ParquetFile(root / record["path"])
        handles[record["path"]] = parquet
        layouts[record["path"]] = [
            parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)
        ]
        column = read_columns(parquet, ["id"]).column("id").to_pylist()
        if len(column) != parquet.metadata.num_rows:
            raise ValueError(f"{record['path']}: id count differs from the footer row count")
        source_files.append(
            {
                "path": record["path"],
                "size": record["size"],
                "n_rows": len(column),
                "lfs_sha256": record["sha256"],
                "hub_blob_id": record.get("hub_blob_id"),
            }
        )
        ids.extend(column)
        locations.extend((record["path"], index) for index in range(len(column)))
    if any(not isinstance(i, str) or not i.strip() for i in ids):
        raise ValueError("Every row needs a nonempty string id")
    if len(set(ids)) != len(ids):
        duplicate = next(i for i, c in Counter(ids).items() if c > 1)
        raise ValueError(f"Duplicate row id {duplicate!r} in the population")
    keys = [selection_key(salt, i) for i in ids]
    if len(set(keys)) != len(keys):
        raise ValueError("Selection keys collide; choose another salt")
    if len(ids) < n_expected:
        raise ValueError(f"Population of {len(ids)} rows is smaller than n_expected={n_expected}")
    order = sorted(range(len(ids)), key=keys.__getitem__)[:n_expected]

    # Group the selected rows by row group so each group is read once.
    wanted: dict[tuple[str, int], list[tuple[int, int]]] = {}
    for rank, position in enumerate(order):
        name, index = locations[position]
        offset, group = index, 0
        for group, size in enumerate(layouts[name]):
            if offset < size:
                break
            offset -= size
        wanted.setdefault((name, group), []).append((rank, offset))
    tasks: list[dict[str, Any] | None] = [None] * n_expected
    for (name, group), members in sorted(wanted.items()):
        table = read_columns(handles[name], INPUT_COLUMNS, row_group=group)
        for rank, offset in members:
            position = order[rank]
            row = table.slice(offset, 1).to_pylist()[0]
            if row["id"] != ids[position]:
                raise ValueError(f"Row {locations[position]} holds id {row['id']!r}, expected {ids[position]!r}")
            inputs = decode_scene_inputs(row, resolution=resolution, split=split)
            weight = released_solver_weight(inputs["metadata"], cost_model)
            if not np.isclose(weight, cost_weight, rtol=1e-6, atol=0):
                raise ValueError(f"{row['id']}: released weight {weight} differs from the cell weight {cost_weight}")
            request = inputs_request(inputs, cost_model=cost_model, cost_weight=weight, task_variant=task_variant)
            identity = request_input_identity(request)
            tile = inputs["metadata"].get("tile")
            windows, status = footprint(tile)
            source = tile.get("source") if isinstance(tile, dict) else None
            source_tile = (
                {"lat": source["lat"], "lon": source["lon"]}
                if isinstance(source, dict) and all(isinstance(source.get(k), int) for k in ("lat", "lon"))
                else None
            )
            tasks[rank] = {
                "rank": rank,
                "sample_id": row["id"],
                "selection_key": keys[position],
                "source_file": locations[position][0],
                "row_index": locations[position][1],
                "metadata_index": inputs["metadata"].get("index"),
                "input_identity": identity,
                "input_identity_sha256": identity_digest(identity),
                "source_tile": source_tile,
                "source_windows": windows,
                "footprint_status": status,
                "geographic_block": None,
            }
    assert all(task is not None for task in tasks)
    footprints = {t["sample_id"]: task_footprint(t, block_rule) for t in tasks}
    unplaced = [sample_id for sample_id, footprint in footprints.items() if footprint is None]
    if require_footprints and not unplaced_block and unplaced:
        raise ValueError(
            f"{len(unplaced)} selected task(s) have no usable source footprint, e.g. {unplaced[:5]}; block analysis "
            "is planned, so the build stops (unplaced_block=True / --unplaced-block seals them in one unplaced "
            "block; require_footprints=False / --allow-missing-footprints builds a roster without block analysis)"
        )
    records = [footprint for footprint in footprints.values() if footprint is not None]
    blocks, assignments = geographic_block_record(
        records,
        n_tasks=len(tasks),
        block_degrees=GEOGRAPHIC_BLOCK_DEGREES,
        footprints_required=require_footprints and not unplaced_block,
        source=(
            "metadata.tile.used (integer one-degree source tiles of each task)"
            if block_rule == DEFAULT_BLOCK_RULE
            else "metadata.tile.source (the integer one-degree main source tile of each task)"
        ),
        rule=block_rule,
        unplaced_ids=unplaced if unplaced_block else None,
    )
    for task in tasks:
        if footprints[task["sample_id"]] is not None:
            task["geographic_block"] = assignments[task["sample_id"]]
    id_digest = hashlib.sha256()
    for i in ids:
        id_digest.update(i.encode("utf-8") + b"\n")
    roster = {
        "schema": SCHEMA,
        "benchmark_id": benchmark_id,
        "dataset": {
            "repo": pinned["repo_id"],
            "commit": pinned["revision"],
            "tag": pinned.get("tag"),
            "config": config,
            "split": split,
            "resolution": resolution,
            "visibility": "public split of the pinned Hub commit",
        },
        "cell": {"cost_model": cost_model, "cost_weight": cost_weight, "task_variant": task_variant},
        "selection": {
            "rule": (
                "Rank every row id of the split by sha256((salt + id).encode('utf-8')).hexdigest() ascending and keep "
                "the first n_expected; label-free; a duplicate id or a key collision is an error."
            ),
            "salt": salt,
            "key": "sha256(utf-8(salt + id)) hexdigest, ascending lexicographic",
            "n_expected": n_expected,
            "n_selected": len(tasks),
            "population": {
                "n_rows": len(ids),
                "ids_sha256": id_digest.hexdigest(),
                "id_order": "source files sorted by path, rows in file order, ids joined by newline",
            },
        },
        "dataset_root_identity": {"file": PINNED_RECEIPT_NAME, "sha256": pinned_sha, "schema": POINTER_SCHEMA},
        "input_identity_spec": {
            "fields": list(IDENTITY_FIELDS),
            "function": "passage_eval.execution.runner.request_identity restricted to these fields (no budget)",
            "digest": "sha256 of the canonical JSON (sorted keys, compact separators) of the identity object",
            "grid_hash": "sha256(dtype.str + repr(shape) + C-contiguous bytes)",
            "arrays": {
                "elevation": "float64 (H,W): tensor channel 0 * (calibration.max - calibration.min) + calibration.min",
                "obstacles": "bool (H,W): tensor channel 2",
                "input_tensor": "float32 (H,W,3) as stored",
            },
            "start_goal": "(row, column) from metadata.markers",
            "ordinary_metadata": (
                "passage_eval.protocol.contracts.ordinary_metadata of the decoded metadata with "
                "task_variant added; keeps split, resolution, index, calibration min/max, task_variant"
            ),
            "graph_id": "sha256 of GraphSpec(cost_model, cost_weight) with the released 8-connected destination-only graph",
        },
        "source_files": source_files,
        "geographic_blocks": blocks,
        "tasks": tasks,
    }
    roster["roster_sha256"] = compute_roster_sha256(roster)
    validate_roster(roster)
    info = {
        "pinned_receipt": pinned,
        "pinned_receipt_sha256": pinned_sha,
        "n_row_groups_read": len(wanted),
        "block_statistics": block_statistics(blocks),
        "footprint_status": dict(Counter(t["footprint_status"] for t in tasks)),
        "tasks_per_file": dict(sorted(Counter(t["source_file"] for t in tasks).items())),
    }
    for parquet in handles.values():
        parquet.close()
    return roster, info


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=REPOSITORY, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_outputs(
    roster: dict[str, Any],
    info: dict[str, Any],
    out_dir: Path,
    *,
    argv: list[str],
    dataset_root: Path,
    verify_hashes: bool,
    overwrite: bool = False,
) -> dict[str, Path]:
    stem = (
        f"{roster['benchmark_id']}-r{roster['dataset']['resolution']:04d}-{roster['schema'].removeprefix('passage-')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "roster": out_dir / f"{stem}.json",
        "sidecar": out_dir / f"{stem}.sha256",
        "receipt": out_dir / f"{stem}.receipt.json",
    }
    if not overwrite and any(path.exists() for path in paths.values()):
        raise FileExistsError(f"Refusing to overwrite an existing roster output in {out_dir}; pass --overwrite")
    body = json.dumps(roster, indent=1, sort_keys=True, allow_nan=False) + "\n"
    paths["roster"].write_text(body, encoding="utf-8")
    paths["sidecar"].write_text(roster["roster_sha256"] + "\n", encoding="ascii")
    code = {
        name: {"path": str(path), "sha256": file_sha256(path)}
        for name, path in {
            "build_script": Path(__file__).resolve(),
            "roster_module": Path(roster_module.__file__).resolve(),
            "geographic_blocks": Path(source_window_blocks.__code__.co_filename).resolve(),
            "request_identity_module": Path(roster_module.request_identity.__code__.co_filename).resolve(),
            "scene_decoder": Path(decode_scene_inputs.__code__.co_filename).resolve(),
        }.items()
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "benchmark_id": roster["benchmark_id"],
        "date_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "roster": {
            "path": str(paths["roster"]),
            "roster_sha256": roster["roster_sha256"],
            "roster_file_sha256": file_sha256(paths["roster"]),
            "sidecar": str(paths["sidecar"]),
            "roster_sha256_covers": "canonical JSON (sorted keys, compact separators) of every top-level field except roster_sha256",
            "n_tasks": len(roster["tasks"]),
            "n_population": roster["selection"]["population"]["n_rows"],
        },
        "inputs": {
            "dataset_root": str(dataset_root),
            "pinned_receipt": str(dataset_root / PINNED_RECEIPT_NAME),
            "pinned_receipt_sha256": info["pinned_receipt_sha256"],
            "dataset_commit": roster["dataset"]["commit"],
            "n_source_files": len(roster["source_files"]),
            "file_verification": (
                "size and sha256 of every source file recomputed during this build and equal to the pinned receipt"
                if verify_hashes
                else "size only; sha256 not recomputed (--skip-file-hashes)"
            ),
        },
        "command": {
            "argv": argv,
            "cwd": os.getcwd(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "nice": os.nice(0),
        },
        "libraries": {"pyarrow": pyarrow.__version__, "numpy": np.__version__},
        "code": code,
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "note": "the working tree may hold uncommitted files; hashes above identify the code that ran",
        },
        "label_policy": {
            "columns_requested": sorted(COLUMNS_REQUESTED),
            "not_read": NOT_READ,
            "enforced_by": "read_columns allowlist in the build script",
            "reference_solver_run": False,
            "candidate_run": False,
        },
        "build": {
            "row_groups_read": info["n_row_groups_read"],
            "footprint_status": info["footprint_status"],
            "tasks_per_file": info["tasks_per_file"],
        },
        "geographic_blocks": info["block_statistics"],
        "limits": [
            "Blocks follow the roster's block rule (geographic_blocks.rule); independence between blocks is not proven.",
            "Reading metadata.solvers retrieves solver configuration (weight, connectivity), not route costs.",
        ],
    }
    paths["receipt"].write_text(json.dumps(receipt, indent=1, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dataset-root", required=True, type=Path, help="pinned snapshot root holding PINNED.json")
    parser.add_argument("--out-dir", type=Path, default=REPOSITORY / "outputs" / "rosters")
    parser.add_argument("--benchmark-id", default="passage-public-test-v1")
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cost-model", default="elevation")
    parser.add_argument("--cost-weight", type=float, default=62.4)
    parser.add_argument("--task-variant", default="obstacles", choices=["obstacles", "free"])
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--salt", default=None)
    parser.add_argument("--skip-file-hashes", action="store_true")
    parser.add_argument(
        "--allow-missing-footprints",
        action="store_true",
        help="build a roster without planned block analysis: tasks without a footprint stay outside every block",
    )
    parser.add_argument(
        "--unplaced-block",
        action="store_true",
        help="plan block analysis with one sealed 'unplaced' block for the tasks without footprint",
    )
    parser.add_argument("--block-rule", choices=BLOCK_RULES, default=DEFAULT_BLOCK_RULE)
    parser.add_argument("--overwrite", action="store_true")
    add_test_admission_arguments(parser, what="building a roster", preparation=True)
    args = parser.parse_args(argv)
    if args.unplaced_block and args.allow_missing_footprints:
        parser.error("--unplaced-block and --allow-missing-footprints are exclusive")
    admission = test_split_allowed(
        args.allow_test_split,
        args.test_admission_record,
        evidence_dir=args.test_admission_evidence_dir,
        preparation_approval=args.test_preparation_approval,
    )
    COLUMNS_REQUESTED.clear()
    roster, info = build_roster(
        args.dataset_root,
        benchmark_id=args.benchmark_id,
        resolution=args.resolution,
        split=args.split,
        cost_model=args.cost_model,
        cost_weight=args.cost_weight,
        task_variant=args.task_variant,
        n_expected=args.n,
        salt=args.salt,
        verify_hashes=not args.skip_file_hashes,
        require_footprints=not args.allow_missing_footprints,
        unplaced_block=args.unplaced_block,
        block_rule=args.block_rule,
        allow_test=admission is not None,
        admission=admission,
    )
    paths = write_outputs(
        roster,
        info,
        args.out_dir,
        argv=[sys.argv[0], *(argv if argv is not None else sys.argv[1:])],
        dataset_root=args.dataset_root.resolve(),
        verify_hashes=not args.skip_file_hashes,
        overwrite=args.overwrite,
    )
    print(
        json.dumps(
            {
                "roster_sha256": roster["roster_sha256"],
                "paths": {k: str(v) for k, v in paths.items()},
                "geographic_blocks": info["block_statistics"],
                "footprint_status": info["footprint_status"],
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
