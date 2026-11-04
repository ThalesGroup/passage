# MIT License
#
# Copyright (c) 2026 THALES

"""Split a pinned admissible roster into contiguous batch rosters without reading any scene.

Every child is ``io.roster.derive_roster`` of the parent over a contiguous range of parent ranks: no fresh sampling,
every task keeps its ``parent_rank`` and its analysis block (``parent_geographic_block``, the block the parent uses,
itself inherited from the source roster for an admissible roster). A child's locally recomputed block may differ;
global statistics use the parent blocks recorded in the partition manifest. A parent whose block table is
incomplete (a task without block) is refused: block analysis needs every block.

The test split is refused before the parent is opened as a scene source, unless ``allow_test`` is true and an
all-GREEN admission grant admits the parent (``protocol.guards``). Hashes prove byte correspondence only.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from passage_eval.campaign import PARTITION_SCHEMA
from passage_eval.io import pinned
from passage_eval.io import roster as roster_io
from passage_eval.protocol.guards import refuse_test_roster

#: Sanity bound on a parent roster: the whole 256 test population (about 40,000 rows, inventoried before the run)
#: must fit with room for the exact count.
MAX_TASKS = 100_000
#: Largest batch (lot): 1,024 tasks. A session holds every scene of its batch in memory (``lot_memory_bytes``).
MAX_BATCH = 1024
#: Bytes of the ordinary inputs of one task of resolution ``N``: float64 elevation, boolean obstacles and the float32
#: ``(N, N, 3)`` input tensor (``PlanRequest`` copies each array).
ORDINARY_BYTES_PER_CELL = 8 + 1 + 12
#: Copies of the ordinary inputs a supervisor holds at its peak: the label-free scenes and the per-seed requests.
SUPERVISOR_COPIES = 2
_MANIFEST_KEYS = {
    "schema",
    "split",
    "dataset_commit",
    "parent_roster_sha256",
    "parent_file_sha256",
    "parent_block_table_sha256",
    "n_expected",
    "batch_size",
    "batches",
    "admission_record_sha256",
    "verified",
    "note",
}
_BATCH_KEYS = {"index", "start", "stop", "file", "file_sha256", "roster_sha256", "parent_mapping"}
NOTE = "Use the parent geographic blocks of parent_mapping for global inference; child-local blocks can differ."


def lot_memory_bytes(resolution: int, batch_size: int) -> dict[str, Any]:
    """Estimated memory that the ordinary inputs of one lot occupy in a supervisor (labels and scorer excluded)."""
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (resolution, batch_size)):
        raise ValueError("resolution and batch_size must be positive integers")
    per_task = ORDINARY_BYTES_PER_CELL * resolution * resolution
    return {
        "resolution": resolution,
        "batch_size": batch_size,
        "per_task_bytes": per_task,
        "copies": SUPERVISOR_COPIES,
        "supervisor_peak_bytes": SUPERVISOR_COPIES * per_task * batch_size,
        "note": "estimate of the label-free scenes and requests a session holds; the scorer loads the lot again",
    }


def _batch_size(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= MAX_BATCH:
        raise ValueError(f"batch_size must be an integer from 1 to {MAX_BATCH}")
    return value


def read_parent(
    path: str | Path,
    file_sha256: str,
    roster_sha256: str,
    *,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """Read one pinned roster file (metadata only), validate it and apply the test-split guard."""
    if not pinned.is_hex64(roster_sha256):
        raise ValueError("a lowercase 64-character roster_sha256 pin is required")
    roster = pinned.read_pinned_json(path, file_sha256)
    roster = roster_io.validate_roster(roster, expected_sha256=roster_sha256)
    refuse_test_roster(
        roster,
        allow_test=allow_test,
        admission=admission,
        context="the campaign partition",
        require_admitted_roster=True,
    )
    if not 0 < len(roster["tasks"]) <= MAX_TASKS:
        raise ValueError(f"the parent roster must hold 1 to {MAX_TASKS} tasks")
    table = roster_io.roster_block_table(roster)
    if not table["complete"]:
        raise ValueError(
            f"the parent roster has {table['n_tasks_without_block']} task(s) without a geographic block; "
            "a campaign keeps the parent blocks and refuses a null block"
        )
    return roster


def derive_batch(parent: Mapping[str, Any], start: int, stop: int, index: int) -> tuple[dict[str, Any], list[dict]]:
    """The child roster of parent ranks ``[start, stop)`` and its mapping to the parent ranks and blocks."""
    parent = roster_io.validate_roster(parent)
    if (
        any(isinstance(v, bool) or not isinstance(v, int) for v in (start, stop))
        or not 0 <= start < stop <= len(parent["tasks"])
        or stop - start > MAX_BATCH
    ):
        raise ValueError("invalid batch range")
    table = roster_io.roster_block_table(parent)
    ids = [task["sample_id"] for task in parent["tasks"][start:stop]]
    child = roster_io.derive_roster(
        parent,
        ids,
        rule=f"Contiguous parent ranks [{start}, {stop}) of the pinned parent roster; no fresh sampling.",
        derivation={"kind": "campaign_partition", "batch_index": index, "start": start, "stop": stop},
    )
    mapping = []
    for offset, sample_id in enumerate(ids):
        block = table["blocks"][sample_id]
        if block is None:
            raise ValueError(f"{sample_id} has no parent geographic block")
        mapping.append({"parent_index": start + offset, "sample_id": sample_id, "parent_geographic_block": block})
    return child, mapping


def _batches(parent: Mapping[str, Any], batch_size: int):
    n = len(parent["tasks"])
    for index, start in enumerate(range(0, n, batch_size)):
        stop = min(start + batch_size, n)
        child, mapping = derive_batch(parent, start, stop, index)
        yield index, start, stop, child, mapping


def partition(
    parent_path: str | Path,
    file_sha256: str,
    roster_sha256: str,
    out_dir: str | Path,
    batch_size: int,
    *,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """Write the child rosters and ``partition-manifest.json`` (with a ``.sha256`` sidecar) to a new directory."""
    batch_size = _batch_size(batch_size)
    parent = read_parent(parent_path, file_sha256, roster_sha256, allow_test=allow_test, admission=admission)
    out_dir = Path(out_dir)
    out_dir.mkdir(mode=0o700, parents=False, exist_ok=False)
    items = []
    for index, start, stop, child, mapping in _batches(parent, batch_size):
        name = f"batch-{index:06d}.json"
        raw = roster_io.canonical_json(child)
        with (out_dir / name).open("xb") as target:
            target.write(raw)
        items.append(
            {
                "index": index,
                "start": start,
                "stop": stop,
                "file": name,
                "file_sha256": pinned.sha256_bytes(raw),
                "roster_sha256": child["roster_sha256"],
                "parent_mapping": mapping,
            }
        )
    manifest = {
        "schema": PARTITION_SCHEMA,
        "split": parent["dataset"]["split"],
        "dataset_commit": parent["dataset"]["commit"],
        "parent_roster_sha256": roster_sha256,
        "parent_file_sha256": file_sha256,
        "parent_block_table_sha256": roster_io.roster_block_table(parent)["block_table_sha256"],
        "n_expected": len(parent["tasks"]),
        "batch_size": batch_size,
        "batches": items,
        "admission_record_sha256": admission.record_sha256 if admission is not None else None,
        "verified": False,
        "note": NOTE,
    }
    raw = roster_io.canonical_json(manifest)
    with (out_dir / "partition-manifest.json").open("xb") as target:
        target.write(raw)
    with (out_dir / "partition-manifest.json.sha256").open("xb") as target:
        target.write((pinned.sha256_bytes(raw) + "\n").encode("ascii"))
    return manifest


def verify_partition(
    parent_path: str | Path,
    file_sha256: str,
    roster_sha256: str,
    out_dir: str | Path,
    manifest_sha256: str,
    *,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """Re-derive every child and compare bytes, ranges, digests and parent blocks; return the manifest.

    Reads only the parent roster, the manifest and the child roster files; no scene file is opened.
    """
    parent = read_parent(parent_path, file_sha256, roster_sha256, allow_test=allow_test, admission=admission)
    out_dir = Path(out_dir)
    manifest = pinned.read_pinned_json(out_dir / "partition-manifest.json", manifest_sha256)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != _MANIFEST_KEYS
        or manifest["schema"] != PARTITION_SCHEMA
        or manifest["split"] != parent["dataset"]["split"]
        or manifest["dataset_commit"] != parent["dataset"]["commit"]
        or manifest["parent_roster_sha256"] != roster_sha256
        or manifest["parent_file_sha256"] != file_sha256
        or manifest["parent_block_table_sha256"] != roster_io.roster_block_table(parent)["block_table_sha256"]
        or manifest["n_expected"] != len(parent["tasks"])
        or manifest["verified"] is not False
        or manifest["note"] != NOTE
        or not isinstance(manifest["batches"], list)
        or not manifest["batches"]
    ):
        raise ValueError("invalid partition manifest identity or policy")
    expected_record = admission.record_sha256 if admission is not None else None
    if manifest["admission_record_sha256"] != expected_record:
        raise ValueError("the partition was made under another test admission record")
    batch_size = _batch_size(manifest["batch_size"])
    expected = list(_batches(parent, batch_size))
    if len(expected) != len(manifest["batches"]):
        raise ValueError("the partition does not cover the full parent roster")
    for (index, start, stop, child, mapping), item in zip(expected, manifest["batches"]):
        if not isinstance(item, dict) or set(item) != _BATCH_KEYS:
            raise ValueError("partition batch record invalid")
        if (item["index"], item["start"], item["stop"], item["file"]) != (
            index,
            start,
            stop,
            f"batch-{index:06d}.json",
        ):
            raise ValueError("partition ranges or batch identity invalid")
        raw = pinned.read_pinned(out_dir / item["file"], item["file_sha256"])
        if raw != roster_io.canonical_json(child) or item["roster_sha256"] != child["roster_sha256"]:
            raise ValueError("child bytes do not match the derived child roster")
        if item["parent_mapping"] != mapping:
            raise ValueError("child parent mapping (ranks or geographic blocks) does not match the parent")
    return manifest


def load_child(out_dir: str | Path, item: Mapping[str, Any]) -> dict[str, Any]:
    """A verified child roster of a manifest batch record."""
    roster = pinned.read_pinned_json(Path(out_dir) / item["file"], item["file_sha256"])
    return roster_io.validate_roster(roster, expected_sha256=item["roster_sha256"])
