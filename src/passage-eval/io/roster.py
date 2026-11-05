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

"""Pinned task rosters: validation and fail-closed loading of the listed rows.

A roster names the exact rows of one benchmark cell: the pinned dataset commit
and files, the label-free selection rule, and for every task its source file,
row index, ordinary-input identity and geographic footprint. It never contains
oracle routes, costs or other labels.

Two schemas are readable. ``passage-roster-v1`` tolerates tasks without a
footprint and leaves the block rule's version outside the digest.
``passage-roster-v2`` (written by the current builders) always carries a
``geographic_blocks`` record that binds the block rule's identity
(``block_function_identity``: schema, callable and source SHA-256) under
``roster_sha256``, names the rule (``rule``: ``source_window_union``, the
default when absent, or ``main_tile_cell``) and declares ``footprints_required``.
When that flag is true, a task without the footprint its rule needs makes the
roster invalid: block analysis is planned and cannot proceed with an unplaced
task. A v2 roster may instead declare ``unplaced_block``: every task without
that footprint is listed there, in roster order, and analysed in the one sealed
block ``geographic_blocks.UNPLACED_BLOCK``; no singleton block is invented.

A derived roster (``derive_roster``) keeps a subset of a parent's tasks in parent
order without fresh sampling. Each task records its ``parent_rank`` and its
``parent_geographic_block``; ``roster_block_table`` returns those inherited
blocks, so a child is analysed with the parent's blocks rather than with blocks
recomputed on the subset.

``load_roster`` validates the file and recomputes ``roster_sha256``.
``load_roster_scenes`` reads exactly the listed rows, in roster order, through
the same ``scene_from_row`` used by ``load_val_scenes`` and rejects any task whose
ordinary inputs no longer hash to the recorded identity. ``open_roster_rows``
gives the same verified rows undecoded, for readers that decode them otherwise
(the reference admission inventory).

Scorer-side warning: ``scene_from_row`` decodes the released oracle route and
cost of each row, so the returned ``Scene`` objects carry labels
(``oracle_path``, ``oracle_cost`` and ``metadata['paths']``). The supervisor must
redact them (for example by passing only ``PlanRequest.from_scene(scene)``,
which drops the oracle) before any worker or candidate sees a scene. Nothing
here makes a roster a statistical sample or its blocks independent.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import math
import statistics
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from numbers import Integral
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from passage_eval.execution.runner import request_identity
from passage_eval.io.dataset import scene_from_row, scene_from_row_unvalidated_label
from passage_eval.protocol.contracts import PlanRequest
from passage_eval.protocol.geographic_blocks import (
    BLOCK_RULES,
    BLOCKS_SCHEMA,
    DEFAULT_BLOCK_RULE,
    UNPLACED_BLOCK,
    block_function_identity,
    compute_blocks,
    source_window_blocks,
    task_footprint,
)
from passage_eval.protocol.planner import Scene

SCHEMA_V1 = "passage-roster-v1"
SCHEMA_V2 = "passage-roster-v2"
SCHEMA = SCHEMA_V2  # written by the current builders
READABLE_SCHEMAS = (SCHEMA_V1, SCHEMA_V2)
BLOCK_TABLE_SCHEMA = "passage-roster-block-table/v1"
POINTER_SCHEMA = "passage-pinned-hub-snapshot-v1"
PINNED_RECEIPT_NAME = "PINNED.json"
GEOGRAPHIC_BLOCK_DEGREES = 5
# Fields of ``request_identity`` that depend only on the ordinary task inputs
# and the cell's objective. The budget is a protocol setting, not a task input.
IDENTITY_FIELDS = (
    "graph_id",
    "cost_model",
    "cost_weight",
    "start",
    "goal",
    "shape",
    "elevation_hash",
    "obstacles_hash",
    "input_tensor_hash",
    "ordinary_metadata_hash",
    "task_variant",
)
# The input identity without ``ordinary_metadata_hash``. The ordinary metadata carries the split name and the row
# index, so a row copied into another split keeps its terrain, obstacles, input tensor, endpoints and objective but
# changes its input identity. Checks of disjointness between splits (warm-up inputs) compare this content identity.
CONTENT_IDENTITY_FIELDS = tuple(name for name in IDENTITY_FIELDS if name != "ordinary_metadata_hash")
_TOP_LEVEL = (
    "schema",
    "benchmark_id",
    "dataset",
    "cell",
    "selection",
    "dataset_root_identity",
    "input_identity_spec",
    "source_files",
    "geographic_blocks",
    "tasks",
    "roster_sha256",
)
_HEX64 = frozenset("0123456789abcdef")


def canonical_json(value: Any) -> bytes:
    """Sorted keys, compact separators, ASCII, no NaN: the hashed byte form."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True).encode("ascii")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def selection_key(salt: str, sample_id: str) -> str:
    """Label-free ranking key: sha256 of the UTF-8 bytes of ``salt + sample_id``."""
    return hashlib.sha256((salt + sample_id).encode("utf-8")).hexdigest()


def compute_roster_sha256(roster: Mapping[str, Any]) -> str:
    """Hash every top-level field except ``roster_sha256`` itself."""
    return sha256_hex(canonical_json({key: value for key, value in roster.items() if key != "roster_sha256"}))


def identity_digest(identity: Mapping[str, Any]) -> str:
    return sha256_hex(canonical_json(dict(identity)))


def content_identity_digest(identity: Mapping[str, Any]) -> str:
    """Digest of the ``CONTENT_IDENTITY_FIELDS`` of an input identity (ordinary metadata left out)."""
    missing = [name for name in CONTENT_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(f"an input identity lacks the content fields {missing}")
    return identity_digest({name: identity[name] for name in CONTENT_IDENTITY_FIELDS})


def request_input_identity(request: PlanRequest) -> dict[str, Any]:
    """The ``request_identity`` fields that identify the task inputs (no budget)."""
    full = request_identity(request)
    return {name: full[name] for name in IDENTITY_FIELDS}


def inputs_request(inputs: Mapping[str, Any], *, cost_model: str, cost_weight: float, task_variant: str) -> PlanRequest:
    """Build the ordinary request from ``decode_scene_inputs`` output, label-free.

    Mirrors ``scene_from_row`` (variant handling, metadata) without reading any
    oracle column, so a roster can be built without touching labels.
    """
    if task_variant not in {"obstacles", "free"}:
        raise ValueError("task_variant must be obstacles or free")
    metadata = dict(inputs["metadata"])
    metadata["task_variant"] = task_variant
    tensor = np.array(inputs["input_tensor"], copy=True)
    obstacles = np.array(inputs["obstacles"], copy=True)
    if task_variant == "free":
        obstacles[:] = False
        tensor[:, :, 2] = 0
    return PlanRequest(
        inputs["sample_id"],
        inputs["elevation"],
        obstacles,
        tuple(inputs["start"]),
        tuple(inputs["goal"]),
        cost_model,
        float(cost_weight),
        metadata=metadata,
        input_tensor=tensor,
    )


def scene_input_identity(scene: Scene) -> dict[str, Any]:
    return request_input_identity(PlanRequest.from_scene(scene))


def _reject_constant(name: str):
    raise ValueError(f"Roster JSON contains the non-finite constant {name}")


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Roster JSON repeats the key {key!r}")
        result[key] = value
    return result


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX64


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _safe_relative(path: Any) -> str:
    if not isinstance(path, str) or not path or "\\" in path:
        raise ValueError("Roster file paths must be nonempty POSIX strings")
    pure = PurePosixPath(path)
    if pure.is_absolute() or ".." in pure.parts or pure.as_posix() != path or pure.suffix != ".parquet":
        raise ValueError(f"Unsafe roster file path {path!r}")
    return path


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"Invalid roster: {message}")


def _valid_degrees(degrees: Any) -> bool:
    return type(degrees) is int and degrees >= 1 and not 180 % degrees and not 360 % degrees


def block_rule(roster: Mapping[str, Any]) -> str:
    """The block rule a roster declares (``source_window_union`` when its block record names none)."""
    blocks = roster.get("geographic_blocks")
    rule = blocks.get("rule", DEFAULT_BLOCK_RULE) if isinstance(blocks, Mapping) else DEFAULT_BLOCK_RULE
    _require(rule in BLOCK_RULES, f"geographic_blocks.rule must be one of {list(BLOCK_RULES)}")
    return rule


def unplaced_block_record(task_ids: Sequence[str]) -> dict[str, Any]:
    """The sealed ``unplaced_block`` record of a v2 roster: its tasks without footprint, in roster order."""
    return {
        "block_id": UNPLACED_BLOCK,
        "n_tasks": len(task_ids),
        "task_ids": list(task_ids),
        "rule": (
            "every task whose footprint for the block rule is absent or undecodable is analysed in this one block; "
            "no singleton block is invented"
        ),
    }


def _declares_unplaced(roster: Mapping[str, Any]) -> bool:
    blocks = roster.get("geographic_blocks")
    return isinstance(blocks, Mapping) and blocks.get("unplaced_block") is not None


def _validate_derivation(roster: Mapping[str, Any]) -> None:
    """Check the parent fields of a derived roster: all or none, ordered, and coarser than the local blocks."""
    tasks, selection, blocks = roster["tasks"], roster["selection"], roster["geographic_blocks"]
    flags = [("parent_rank" in task, "parent_geographic_block" in task) for task in tasks]
    derivation = selection.get("derivation")
    if not any(any(flag) for flag in flags):
        _require(derivation is None, "selection.derivation needs parent_rank and parent_geographic_block on tasks")
        _require(not isinstance(blocks, dict) or "parent_blocks" not in blocks, "parent_blocks without a derivation")
        return
    _require(all(all(flag) for flag in flags), "a derived roster needs parent_rank and parent_geographic_block")
    _require(
        isinstance(derivation, dict) and _is_hex64(derivation.get("parent_roster_sha256")),
        "selection.derivation must name the parent roster_sha256",
    )
    ranks = [task["parent_rank"] for task in tasks]
    _require(
        all(_is_int(rank) and rank >= 0 for rank in ranks) and all(a < b for a, b in zip(ranks, ranks[1:])),
        "parent_rank must be strictly increasing nonnegative integers (parent order is kept)",
    )
    rule, unplaced_declared = block_rule(roster), _declares_unplaced(roster)
    local_to_parent: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for task in tasks:
        inherited = task["parent_geographic_block"]
        if task_footprint(task, rule) is None:
            _require(
                inherited is None or (unplaced_declared and inherited == UNPLACED_BLOCK),
                f"{task['sample_id']} inherits a block without a footprint",
            )
            if inherited is not None:
                counts[inherited] += 1
            continue
        _require(
            isinstance(inherited, str) and inherited and inherited != UNPLACED_BLOCK,
            f"{task['sample_id']} needs its parent_geographic_block",
        )
        # A block of the subset can only be part of one parent block: the parent joins at least as much.
        _require(
            local_to_parent.setdefault(task["geographic_block"], inherited) == inherited,
            f"{task['sample_id']} shares a local block with a task of another parent block",
        )
        counts[inherited] += 1
    if counts:
        expected = {
            "parent_roster_sha256": derivation["parent_roster_sha256"],
            "source_roster_sha256": (blocks.get("parent_blocks") or {}).get("source_roster_sha256"),
            "number_of_blocks": len(counts),
            "block_sizes": dict(sorted(counts.items())),
        }
        _require(_is_hex64(expected["source_roster_sha256"]), "parent_blocks.source_roster_sha256 is required")
        _require(blocks.get("parent_blocks") == expected, "geographic_blocks.parent_blocks differs from the tasks")
    else:
        _require(not isinstance(blocks, dict) or "parent_blocks" not in blocks, "parent_blocks without footprints")


def geographic_block_record(
    records: Sequence[Mapping[str, Any]],
    *,
    n_tasks: int,
    block_degrees: int,
    footprints_required: bool,
    source: str,
    rule: str = DEFAULT_BLOCK_RULE,
    unplaced_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """The ``passage-roster-v2`` block record and the task assignments for ``records`` (``task_footprint`` inputs).

    Raises ``ValueError`` when ``footprints_required`` and a task has no footprint. ``unplaced_ids`` (the tasks
    without footprint, in roster order) declares the sealed unplaced block; it needs ``footprints_required=False``.
    """
    if not isinstance(footprints_required, bool):
        raise TypeError("footprints_required must be a boolean")
    if rule not in BLOCK_RULES:
        raise ValueError(f"unknown block rule {rule!r}; expected one of {list(BLOCK_RULES)}")
    if footprints_required and len(records) != n_tasks:
        raise ValueError(
            f"block analysis is planned but {n_tasks - len(records)} task(s) have no footprint; "
            "pass footprints_required=False only for a roster without block analysis or with a sealed unplaced block"
        )
    if unplaced_ids is not None:
        if footprints_required:
            raise ValueError("a sealed unplaced block replaces the footprint requirement: footprints_required=False")
        if len(unplaced_ids) != n_tasks - len(records) or len(set(unplaced_ids)) != len(unplaced_ids):
            raise ValueError("the unplaced block must list exactly the tasks without footprint")
    if records:
        computed = compute_blocks(records, rule=rule, block_degrees=block_degrees)
        assignments = computed.pop("task_blocks")
        computed.pop("crossing_task_ids", None)
    else:
        if not _valid_degrees(block_degrees):
            raise ValueError("block_degrees must be a positive divisor of both 180 and 360")
        computed = {
            "schema": BLOCKS_SCHEMA,
            "rule": rule,
            "block_degrees": block_degrees,
            "number_of_tasks": 0,
            "number_of_blocks": 0,
            "grid_cell_count": 0,
            "crossing_task_count": 0,
            "block_sizes": {},
            "statistical_independence_proven": False,
        }
        assignments = {}
    computed.update(
        source=source,
        n_tasks_without_footprint=n_tasks - len(records),
        function=block_function_identity(rule),
        footprints_required=footprints_required,
    )
    if unplaced_ids is not None:
        computed["unplaced_block"] = unplaced_block_record(unplaced_ids)
    return computed, assignments


def validate_roster(roster: Any, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Check structure, ordering, identities, blocks and the recomputed digest.

    Raises ``ValueError`` on the first problem; never repairs anything. The
    returned object is a deep copy, so later edits do not alias the caller's data.
    """
    _require(isinstance(roster, Mapping), "a JSON object is required")
    roster = copy.deepcopy(dict(roster))
    _require(roster.get("schema") in READABLE_SCHEMAS, f"schema must be one of {list(READABLE_SCHEMAS)}")
    _require(set(roster) == set(_TOP_LEVEL), f"top-level keys must be exactly {sorted(_TOP_LEVEL)}")
    _require(isinstance(roster["benchmark_id"], str) and roster["benchmark_id"], "benchmark_id is required")
    dataset, cell, selection = roster["dataset"], roster["cell"], roster["selection"]
    _require(
        all(isinstance(part, dict) for part in (dataset, cell, selection)), "dataset, cell and selection are objects"
    )
    _require(isinstance(dataset.get("repo"), str) and dataset["repo"], "dataset.repo is required")
    _require(
        isinstance(dataset.get("commit"), str) and len(dataset["commit"]) == 40 and set(dataset["commit"]) <= _HEX64,
        "dataset.commit must be a 40-hex lowercase commit",
    )
    _require(_is_int(dataset.get("resolution")) and dataset["resolution"] > 0, "dataset.resolution must be positive")
    _require(
        dataset.get("config") == f"{dataset['resolution']}x{dataset['resolution']}", "dataset.config must be <N>x<N>"
    )
    _require(isinstance(dataset.get("split"), str) and dataset["split"], "dataset.split is required")
    _require(
        isinstance(cell.get("cost_model"), str)
        and cell.get("task_variant") in {"obstacles", "free"}
        and isinstance(cell.get("cost_weight"), (int, float))
        and not isinstance(cell["cost_weight"], bool)
        and math.isfinite(cell["cost_weight"]),
        "cell needs cost_model, finite cost_weight and task_variant",
    )
    salt, expected = selection.get("salt"), selection.get("n_expected")
    _require(isinstance(salt, str) and salt, "selection.salt is required")
    _require(_is_int(expected) and expected > 0, "selection.n_expected must be a positive integer")
    root = roster["dataset_root_identity"]
    _require(
        isinstance(root, dict) and root.get("file") == PINNED_RECEIPT_NAME and _is_hex64(root.get("sha256")),
        "dataset_root_identity needs file PINNED.json and a 64-hex sha256",
    )

    files = roster["source_files"]
    _require(isinstance(files, list) and files, "source_files must be a nonempty list")
    by_path: dict[str, dict[str, Any]] = {}
    for record in files:
        _require(isinstance(record, dict), "source file records are objects")
        path = _safe_relative(record.get("path"))
        _require(path not in by_path, f"duplicate source file {path}")
        _require(
            _is_int(record.get("size"))
            and record["size"] > 0
            and _is_int(record.get("n_rows"))
            and record["n_rows"] > 0
            and _is_hex64(record.get("lfs_sha256")),
            f"incomplete source file record {path}",
        )
        by_path[path] = record
    _require([r["path"] for r in files] == sorted(by_path), "source_files must be sorted by path")

    tasks = roster["tasks"]
    _require(
        isinstance(tasks, list) and len(tasks) == expected == selection.get("n_selected"),
        "tasks, selection.n_expected and selection.n_selected must agree",
    )
    spec = roster["input_identity_spec"]
    _require(
        isinstance(spec, dict) and list(spec.get("fields", [])) == list(IDENTITY_FIELDS),
        "input_identity_spec.fields differs from this module",
    )
    seen_ids: set[str] = set()
    seen_rows: set[tuple[str, int]] = set()
    previous_key = ""
    records = []
    unplaced_ids = []
    rule = block_rule(roster)
    _require(roster["schema"] == SCHEMA_V2 or rule == DEFAULT_BLOCK_RULE, "passage-roster-v1 knows only the union rule")
    for position, task in enumerate(tasks):
        _require(isinstance(task, dict), "tasks are objects")
        label = f"task {position}"
        task_id = task.get("sample_id")
        _require(isinstance(task_id, str) and task_id.strip(), f"{label} needs a sample_id")
        _require(task["sample_id"] not in seen_ids, f"duplicate sample_id {task_id}")
        seen_ids.add(task_id)
        _require(task.get("rank") == position and _is_int(task.get("rank")), f"{task_id} rank must equal its position")
        key = task.get("selection_key")
        _require(key == selection_key(salt, task_id), f"{task_id} selection_key differs from the declared rule")
        _require(key > previous_key, f"{task_id} breaks the ascending selection order")
        previous_key = key
        source = task.get("source_file")
        _require(source in by_path, f"{task_id} names a file outside source_files")
        row = task.get("row_index")
        _require(_is_int(row) and 0 <= row < by_path[source]["n_rows"], f"{task_id} row_index is outside its file")
        _require((source, row) not in seen_rows, f"{task_id} repeats a source row")
        seen_rows.add((source, int(row)))
        identity = task.get("input_identity")
        _require(
            isinstance(identity, dict) and set(identity) == set(IDENTITY_FIELDS), f"{task_id} input_identity fields"
        )
        _require(
            identity["cost_model"] == cell["cost_model"]
            and identity["task_variant"] == cell["task_variant"]
            and identity["cost_weight"] == cell["cost_weight"]
            and identity["shape"] == [dataset["resolution"]] * 2,
            f"{task_id} identity disagrees with the cell",
        )
        _require(
            task.get("input_identity_sha256") == identity_digest(identity), f"{task_id} input_identity_sha256 differs"
        )
        windows, block = task.get("source_windows"), task.get("geographic_block")
        _require(isinstance(windows, list), f"{task_id} source_windows must be a list")
        status = task.get("footprint_status")
        _require(
            status == "ok" if windows else status in {"missing", "malformed"},
            f"{task_id} footprint_status disagrees with its windows",
        )
        footprint = task_footprint(task, rule)
        if footprint is not None:
            records.append(footprint)
        else:
            unplaced_ids.append(task_id)
            _require(block is None, f"{task_id} has a block without a footprint")
    blocks = roster["geographic_blocks"]
    if roster["schema"] == SCHEMA_V2:
        _require(isinstance(blocks, dict), "geographic_blocks must be an object in passage-roster-v2")
        _require(blocks.get("schema") == BLOCKS_SCHEMA, f"geographic_blocks.schema must be {BLOCKS_SCHEMA!r}")
        _require(
            blocks.get("function") == block_function_identity(rule),
            "geographic_blocks.function differs from the installed block rule: the block rule "
            "changed after this roster was sealed; rebuild the roster with the current rule",
        )
        required = blocks.get("footprints_required")
        _require(isinstance(required, bool), "geographic_blocks.footprints_required must be a boolean")
        if required:
            _require(
                not unplaced_ids,
                f"block analysis is planned but {len(unplaced_ids)} task(s) have no footprint, e.g. {unplaced_ids[:3]}",
            )
        if blocks.get("unplaced_block") is not None:
            _require(required is False, "an unplaced block needs footprints_required false")
            _require(
                blocks["unplaced_block"] == unplaced_block_record(unplaced_ids),
                "geographic_blocks.unplaced_block must list exactly the tasks without footprint, in roster order",
            )
    else:
        _require(not isinstance(blocks, dict) or "unplaced_block" not in blocks, "an unplaced block needs v2")
    if records:
        _require(isinstance(blocks, dict), "geographic_blocks summary is missing")
        degrees = blocks.get("block_degrees")
        try:
            recomputed = compute_blocks(records, rule=rule, block_degrees=degrees)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid roster: footprints do not form blocks: {exc}") from exc
        for task in tasks:
            if task["sample_id"] in recomputed["task_blocks"]:
                _require(
                    task.get("geographic_block") == recomputed["task_blocks"][task["sample_id"]],
                    f"{task['sample_id']} geographic_block differs from its footprint",
                )
        for name in ("block_degrees", "number_of_blocks", "grid_cell_count", "crossing_task_count", "block_sizes"):
            _require(blocks.get(name) == recomputed[name], f"geographic_blocks.{name} differs from the task footprints")
        _require(
            blocks.get("number_of_tasks") == len(records)
            and blocks.get("n_tasks_without_footprint") == len(tasks) - len(records),
            "geographic_blocks task counts differ",
        )
    elif roster["schema"] == SCHEMA_V2:
        degrees = blocks.get("block_degrees")
        _require(_valid_degrees(degrees), "geographic_blocks.block_degrees must divide 180 and 360")
        _require(
            all(blocks.get(name) == 0 for name in ("number_of_blocks", "grid_cell_count", "crossing_task_count"))
            and blocks.get("block_sizes") == {}
            and blocks.get("number_of_tasks") == 0
            and blocks.get("n_tasks_without_footprint") == len(tasks),
            "geographic_blocks must record zero blocks when no task has a footprint",
        )
    else:
        _require(blocks is None, "geographic_blocks must be null when no task has a footprint")
    _validate_derivation(roster)

    digest = roster["roster_sha256"]
    _require(_is_hex64(digest), "roster_sha256 must be 64 lowercase hex characters")
    _require(digest == compute_roster_sha256(roster), "roster_sha256 differs from the recomputed digest")
    if expected_sha256 is not None:
        _require(digest == expected_sha256, "roster_sha256 differs from the expected pin")
    return roster


def load_roster(path: str | Path, *, expected_sha256: str | None = None) -> dict[str, Any]:
    """Parse and validate a roster file; recompute and check ``roster_sha256``.

    A ``.sha256`` sidecar next to the file (same stem) is checked when present.
    Pass ``expected_sha256`` to pin the roster to a digest recorded elsewhere.
    """
    path = Path(path)
    try:
        roster = json.loads(
            path.read_bytes().decode("utf-8"), object_pairs_hook=_no_duplicate_keys, parse_constant=_reject_constant
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Roster {path} is not valid JSON: {exc}") from exc
    roster = validate_roster(roster, expected_sha256=expected_sha256)
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists():
        pinned = sidecar.read_text(encoding="ascii").split()
        if len(pinned) != 1 or pinned[0] != roster["roster_sha256"]:
            raise ValueError(f"Roster sidecar {sidecar} does not hold the recomputed roster_sha256")
    return roster


def roster_block_table(roster: Mapping[str, Any]) -> dict[str, Any]:
    """The block of every task to use for analysis, with its provenance and a digest.

    For a derived roster these are the inherited ``parent_geographic_block`` values; otherwise the roster's own
    ``geographic_block`` values. A task without a footprint maps to ``UNPLACED_BLOCK`` when the roster declares
    the sealed unplaced block, otherwise to ``None``, which makes the table incomplete.
    """
    roster = validate_roster(roster)
    tasks = roster["tasks"]
    derived = "parent_geographic_block" in tasks[0]
    key = "parent_geographic_block" if derived else "geographic_block"
    blocks = {task["sample_id"]: task[key] for task in tasks}
    summary = roster["geographic_blocks"] or {}
    unplaced = summary.get("unplaced_block")
    if unplaced is not None and not derived:
        blocks.update(dict.fromkeys(unplaced["task_ids"], UNPLACED_BLOCK))
    if derived:
        source = (summary.get("parent_blocks") or {}).get("source_roster_sha256") or roster["selection"]["derivation"][
            "parent_roster_sha256"
        ]
    else:
        source = roster["roster_sha256"]
    present = [block for block in blocks.values() if block is not None]
    table = {
        "schema": BLOCK_TABLE_SCHEMA,
        "roster_schema": roster["schema"],
        "roster_sha256": roster["roster_sha256"],
        "assignment": "inherited_parent_blocks" if derived else "roster_blocks",
        "assignment_roster_sha256": source,
        "function": summary.get("function"),
        "block_degrees": summary.get("block_degrees"),
        "footprints_required": summary.get("footprints_required"),
        "n_tasks": len(blocks),
        "n_tasks_without_block": len(blocks) - len(present),
        "number_of_blocks": len(set(present)),
        "complete": len(present) == len(blocks),
        "blocks": blocks,
    }
    if roster["schema"] == SCHEMA_V2:
        # Only v2 tables name the rule and the unplaced block, so v1 tables keep their earlier digest.
        table.update(
            rule=block_rule(roster),
            unplaced_block_declared=unplaced is not None,
            n_unplaced=sum(block == UNPLACED_BLOCK for block in blocks.values()),
        )
    table["block_table_sha256"] = sha256_hex(canonical_json(table))
    return table


def block_table_sealed(table: Mapping[str, Any]) -> bool:
    """Whether a block table comes from a sealed ``passage-roster-v2`` roster with the installed block rule.

    A ``passage-roster-v1`` table is not sealed: its block rule's version is outside the roster digest, and a v1
    child (for example one that recomputed local blocks without derivation fields) may use other blocks than its
    parent. A v2 table that binds another block rule than the installed one is refused, as ``validate_roster``
    refuses its roster.
    """
    if table.get("roster_schema") != SCHEMA_V2:
        return False
    rule = table.get("rule", DEFAULT_BLOCK_RULE)
    if rule not in BLOCK_RULES or table.get("function") != block_function_identity(rule):
        raise ValueError(
            "the v2 block table binds another block rule than the installed geographic_blocks.py; rebuild the roster"
        )
    return True


#: Where a run manifest records the roster it was made with: the supervisor's procedure and record, and the
#: roster scorer's staging record.
_RUN_ROSTER_PATHS = (("procedure", "supervision", "roster"), ("supervision", "roster"), ("scoring_stage", "roster"))


def run_roster_sha256(manifest: Mapping[str, Any]) -> str | None:
    """The ``roster_sha256`` a run manifest records (``None`` for a run made without a roster).

    Every place that records it must agree; a disagreement is an error.
    """
    values = set()
    for path in _RUN_ROSTER_PATHS:
        node: Any = manifest
        for key in path:
            node = node.get(key) if isinstance(node, Mapping) else None
        if isinstance(node, Mapping) and node.get("roster_sha256") is not None:
            values.add(node["roster_sha256"])
    if len(values) > 1:
        raise ValueError("the run manifest records different roster digests")
    return next(iter(values), None)


def check_block_table_binds_run(table: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    """Refuse a block table that does not belong to the roster the run was made with; return that roster digest."""
    recorded = run_roster_sha256(manifest)
    if recorded is None:
        raise ValueError(
            "a roster block table applies only to a run that records its roster (roster_sha256); this run records none"
        )
    if recorded != table.get("roster_sha256"):
        raise ValueError(
            f"the block table belongs to roster {str(table.get('roster_sha256'))[:12]}..., "
            f"the run was made with roster {recorded[:12]}..."
        )
    return recorded


def validate_block_table(table: Any) -> dict[str, Any]:
    """Check a ``roster_block_table`` record's structure and recompute its digest."""
    if not isinstance(table, Mapping) or table.get("schema") != BLOCK_TABLE_SCHEMA:
        raise ValueError(f"a {BLOCK_TABLE_SCHEMA} record is required")
    table = copy.deepcopy(dict(table))
    blocks = table.get("blocks")
    if (
        not isinstance(blocks, dict)
        or not blocks
        or any(not isinstance(key, str) or not key for key in blocks)
        or any(value is not None and (not isinstance(value, str) or not value) for value in blocks.values())
    ):
        raise ValueError("block table blocks must map task ids to nonempty block ids or null")
    present = [value for value in blocks.values() if value is not None]
    if (
        table.get("complete") is not (len(present) == len(blocks))
        or table.get("n_tasks") != len(blocks)
        or table.get("n_tasks_without_block") != len(blocks) - len(present)
        or table.get("number_of_blocks") != len(set(present))
        or not _is_hex64(table.get("roster_sha256"))
        or table.get("n_unplaced", 0) != sum(value == UNPLACED_BLOCK for value in present)
        or (table.get("n_unplaced", 0) and table.get("unplaced_block_declared") is not True)
    ):
        raise ValueError("block table counts or roster identity are inconsistent")
    digest = table.pop("block_table_sha256", None)
    if digest != sha256_hex(canonical_json(table)):
        raise ValueError("block_table_sha256 differs from the recomputed digest")
    table["block_table_sha256"] = digest
    return table


def block_summary(roster: Mapping[str, Any]) -> dict[str, Any]:
    """Counts and sizes of the analysis blocks of a roster, for the sealing record (descriptive only)."""
    table = roster_block_table(roster)
    sizes = Counter(block for block in table["blocks"].values() if block is not None)
    ordered = sorted(sizes.values())
    return {
        "schema": "passage-roster-block-summary/v1",
        "roster_sha256": table["roster_sha256"],
        "block_table_sha256": table["block_table_sha256"],
        "assignment": table["assignment"],
        "assignment_roster_sha256": table["assignment_roster_sha256"],
        "function": table["function"],
        "rule": table.get("rule", DEFAULT_BLOCK_RULE),
        "block_degrees": table["block_degrees"],
        "footprints_required": table["footprints_required"],
        "complete": table["complete"],
        "number_of_tasks": table["n_tasks"],
        "n_tasks_without_block": table["n_tasks_without_block"],
        "n_unplaced": table.get("n_unplaced", 0),
        "number_of_blocks": len(sizes),
        "block_sizes": dict(sorted(sizes.items())),
        "min_tasks_per_block": ordered[0] if ordered else None,
        "median_tasks_per_block": statistics.median(ordered) if ordered else None,
        "max_tasks_per_block": ordered[-1] if ordered else None,
        "singleton_blocks": sum(1 for size in ordered if size == 1),
        "size_histogram": {str(size): count for size, count in sorted(Counter(ordered).items())},
        "statistical_independence_proven": False,
    }


def derive_roster(
    parent: Mapping[str, Any],
    sample_ids: Iterable[str],
    *,
    rule: str,
    derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A roster of the named parent tasks, in parent order, that keeps the parent's ranks and blocks.

    No task is added, replaced or re-sampled. Every kept task records ``parent_rank`` and the block it has for
    analysis in the parent (``parent_geographic_block``, from ``roster_block_table``); its ``geographic_block``
    is recomputed on the subset, as validation requires. ``selection.derivation`` holds ``derivation`` plus the
    parent digest. The child keeps the parent's schema and ``footprints_required``.
    """
    parent = validate_roster(parent)
    if not isinstance(rule, str) or not rule.strip():
        raise ValueError("a derived roster needs a nonempty rule")
    wanted = list(sample_ids)
    if len(set(wanted)) != len(wanted):
        raise ValueError("derived roster task ids must be unique")
    position = {task["sample_id"]: index for index, task in enumerate(parent["tasks"])}
    unknown = [sample_id for sample_id in wanted if sample_id not in position]
    if unknown:
        raise ValueError(f"derived roster names {len(unknown)} task(s) outside the parent, e.g. {unknown[:3]}")
    if not wanted:
        raise ValueError("a derived roster needs at least one task")
    inherited = roster_block_table(parent)
    extra = dict(derivation or {})
    if {"parent_roster_sha256", "parent_n_tasks", "parent_derivation"} & extra.keys():
        raise ValueError("derivation may not override the parent identity fields")
    block_rule_name = block_rule(parent)
    tasks = []
    for rank, index in enumerate(sorted(position[sample_id] for sample_id in wanted)):
        task = copy.deepcopy(parent["tasks"][index])
        task["rank"] = rank
        task["parent_rank"] = index
        # The parent's analysis block: its own or inherited block, the sealed unplaced block, or None.
        task["parent_geographic_block"] = inherited["blocks"][task["sample_id"]]
        task["geographic_block"] = None
        tasks.append(task)
    child = {key: copy.deepcopy(value) for key, value in parent.items() if key not in {"tasks", "roster_sha256"}}
    selection = child["selection"]
    previous = selection.pop("derivation", None)
    selection.update(n_expected=len(tasks), n_selected=len(tasks), rule=rule)
    selection["derivation"] = {
        **extra,
        "parent_roster_sha256": parent["roster_sha256"],
        "parent_n_tasks": len(parent["tasks"]),
        "parent_derivation": previous,
    }
    footprints = {t["sample_id"]: task_footprint(t, block_rule_name) for t in tasks}
    records = [footprint for footprint in footprints.values() if footprint is not None]
    parent_blocks = parent["geographic_blocks"] or {}
    if parent["schema"] == SCHEMA_V2:
        blocks, assignments = geographic_block_record(
            records,
            n_tasks=len(tasks),
            block_degrees=parent_blocks["block_degrees"],
            footprints_required=parent_blocks["footprints_required"],
            source=parent_blocks.get("source", "parent roster source windows"),
            rule=block_rule_name,
            unplaced_ids=[sid for sid, footprint in footprints.items() if footprint is None]
            if parent_blocks.get("unplaced_block") is not None
            else None,
        )
    elif records:
        blocks = source_window_blocks(records, block_degrees=parent_blocks["block_degrees"])
        assignments = blocks.pop("task_blocks")
        blocks.update(
            source=parent_blocks.get("source", "parent roster source windows"),
            n_tasks_without_footprint=len(tasks) - len(records),
        )
    else:
        blocks, assignments = None, {}
    counts = Counter(task["parent_geographic_block"] for task in tasks if task["parent_geographic_block"] is not None)
    if counts:
        blocks["parent_blocks"] = {
            "parent_roster_sha256": parent["roster_sha256"],
            "source_roster_sha256": inherited["assignment_roster_sha256"],
            "number_of_blocks": len(counts),
            "block_sizes": dict(sorted(counts.items())),
        }
    for task in tasks:
        if footprints[task["sample_id"]] is not None:
            task["geographic_block"] = assignments[task["sample_id"]]
    child["geographic_blocks"] = blocks
    child["tasks"] = tasks
    child["roster_sha256"] = compute_roster_sha256(child)
    return validate_roster(child, expected_sha256=child["roster_sha256"])


def verify_child_blocks(parent: Mapping[str, Any], child: Mapping[str, Any]) -> dict[str, Any]:
    """Check that ``child`` is an ordered subset of ``parent`` analysed with the parent's blocks.

    Raises ``ValueError`` when a child task is absent from the parent, has another input identity, is out of
    parent order, records another parent rank, or would be analysed in another block than the parent's. A child
    that only recomputed blocks on its subset (no ``parent_geographic_block``) fails whenever those differ.
    """
    parent, child = validate_roster(parent), validate_roster(child)
    parent_table, child_table = roster_block_table(parent), roster_block_table(child)
    by_id = {task["sample_id"]: (index, task) for index, task in enumerate(parent["tasks"])}
    derivation = child["selection"].get("derivation")
    if derivation is not None and derivation.get("parent_roster_sha256") != parent["roster_sha256"]:
        raise ValueError("the child names another parent roster_sha256")
    problems, previous = [], -1
    for task in child["tasks"]:
        sample_id = task["sample_id"]
        if sample_id not in by_id:
            problems.append(f"{sample_id}: not in the parent")
            continue
        index, original = by_id[sample_id]
        if original["input_identity_sha256"] != task["input_identity_sha256"]:
            problems.append(f"{sample_id}: input identity differs from the parent")
        if index <= previous:
            problems.append(f"{sample_id}: out of parent order")
        previous = index
        if "parent_rank" in task and task["parent_rank"] != index:
            problems.append(f"{sample_id}: parent_rank {task['parent_rank']} differs from the parent rank {index}")
        if child_table["blocks"][sample_id] != parent_table["blocks"][sample_id]:
            problems.append(
                f"{sample_id}: analysed in block {child_table['blocks'][sample_id]!r}, "
                f"the parent's is {parent_table['blocks'][sample_id]!r}"
            )
    if problems:
        raise ValueError(f"the child roster does not keep the parent's tasks and blocks: {problems[:5]}")
    return {
        "schema": "passage-roster-child-block-check/v1",
        "parent_roster_sha256": parent["roster_sha256"],
        "child_roster_sha256": child["roster_sha256"],
        "n_child_tasks": len(child["tasks"]),
        "parent_block_table_sha256": parent_table["block_table_sha256"],
        "child_block_table_sha256": child_table["block_table_sha256"],
        "blocks_kept": True,
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_root(roster: Mapping[str, Any], root: Path, *, require_receipt: bool, verify_hashes: bool) -> None:
    used = sorted({task["source_file"] for task in roster["tasks"]})
    records = {record["path"]: record for record in roster["source_files"]}
    if require_receipt:
        receipt_path = root / PINNED_RECEIPT_NAME
        try:
            data = receipt_path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Dataset root has no readable {PINNED_RECEIPT_NAME}: {exc}") from exc
        if sha256_hex(data) != roster["dataset_root_identity"]["sha256"]:
            raise ValueError("Dataset root receipt sha256 differs from the roster's dataset_root_identity")
        receipt = json.loads(data.decode("utf-8"))
        if receipt.get("schema") != POINTER_SCHEMA or receipt.get("revision") != roster["dataset"]["commit"]:
            raise ValueError("Dataset root receipt names a different schema or commit than the roster")
        listed = {entry["path"]: entry for entry in receipt.get("files", [])}
        for name in used:
            entry = listed.get(name)
            if (
                entry is None
                or entry.get("sha256") != records[name]["lfs_sha256"]
                or entry.get("size") != records[name]["size"]
            ):
                raise ValueError(f"Dataset root receipt does not list {name} with the roster's size and sha256")
    for name in used:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Roster source file is missing or not a regular file: {name}")
        if path.stat().st_size != records[name]["size"]:
            raise ValueError(f"Roster source file size differs: {name}")
        if verify_hashes and _file_sha256(path) != records[name]["lfs_sha256"]:
            raise ValueError(f"Roster source file sha256 differs: {name}")


def _label_column(cost_model: str, task_variant: str) -> str:
    from passage_eval.metrics.cost import canonical_cost_model

    model = canonical_cost_model(cost_model)
    return f"path_astar_{'slope' if model == 'slope_uphill' else model}_{task_variant}"


def load_roster_scenes(
    roster: Mapping[str, Any],
    dataset_path: str | Path,
    *,
    require_root_receipt: bool = True,
    verify_file_hashes: bool = True,
    tolerant_labels: bool = False,
) -> list[Scene]:
    """Read exactly the roster rows, in roster order, and verify every input.

    The roster is validated again (a mutated in-memory copy fails on its digest).
    Each row must exist at its recorded file and index, carry the recorded id and
    decode through ``scene_from_row`` to inputs whose identity equals the stored
    one. Any mismatch, missing row or duplicate raises ``ValueError``; no task is
    skipped or replaced.

    ``require_root_receipt`` demands ``<dataset_path>/PINNED.json`` whose sha256
    equals the roster's ``dataset_root_identity`` and which lists the used files
    with the roster's size and sha256. ``verify_file_hashes`` (default on)
    rehashes every used file against the roster; the input identity covers only
    the ordinary inputs, so this is what protects the released labels read here
    (about 3.4 GB for the full 256 test cell). Disable it only for a trusted,
    already verified tree.

    ``tolerant_labels`` is for the independent analytic reference policy only, where the stored route and cost
    are dataset findings and never the reference: a row whose stored label ``scene_from_row`` refuses is decoded
    with that label kept as read (``scene_from_row_unvalidated_label``), with every input check unchanged, so a
    defective published label neither halts nor excludes its task. The reference admission inventory decodes such
    rows the same way. Never score such a scene under the released-label policy.

    The returned scenes contain released labels: redact before any worker.
    """
    if not isinstance(tolerant_labels, bool):
        raise TypeError("tolerant_labels must be a boolean")
    roster, rows = open_roster_rows(
        roster, dataset_path, require_root_receipt=require_root_receipt, verify_file_hashes=verify_file_hashes
    )
    dataset, cell = roster["dataset"], roster["cell"]
    options = {
        "resolution": dataset["resolution"],
        "cost_model": cell["cost_model"],
        "split": dataset["split"],
        "cost_weight": cell["cost_weight"],
        "task_variant": cell["task_variant"],
    }
    scenes: list[Scene] = []
    with rows as pairs:
        for task, row in pairs:
            task_id = task["sample_id"]
            try:
                scene = scene_from_row(row, **options)
            except ValueError:
                if not tolerant_labels:
                    raise
                scene = scene_from_row_unvalidated_label(row, **options)  # input defects still raise here
            if scene.sample_id != task_id:
                raise ValueError(f"{task_id}: decoded scene has id {scene.sample_id!r}")
            identity = scene_input_identity(scene)
            if identity != task["input_identity"]:
                differing = sorted(
                    field for field in IDENTITY_FIELDS if identity.get(field) != task["input_identity"].get(field)
                )
                raise ValueError(f"{task_id}: input identity differs from the roster in {differing}")
            scenes.append(scene)
    return scenes


def open_roster_rows(
    roster: Mapping[str, Any],
    dataset_path: str | Path,
    *,
    require_root_receipt: bool = True,
    verify_file_hashes: bool = True,
) -> tuple[dict[str, Any], contextlib.closing[Iterator[tuple[dict[str, Any], dict[str, Any]]]]]:
    """Validate the roster and its dataset root, then return it with its raw rows, undecoded.

    The second value is a closing iterator of ``(task, row)`` in roster order; use it in a ``with`` block. Each
    row is read at its recorded file and index, with the cell's label column, and must carry the task's id. The
    root and file checks are those of ``load_roster_scenes`` and run before this returns. Decoding the row and
    checking its input identity are left to the caller.
    """
    roster = validate_roster(roster)
    root = Path(dataset_path)
    _check_root(roster, root, require_receipt=require_root_receipt, verify_hashes=verify_file_hashes)
    return roster, contextlib.closing(_roster_rows(roster, root))


def _roster_rows(roster: Mapping[str, Any], root: Path) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    import pyarrow.parquet as pq

    cell = roster["cell"]
    columns = ["id", "tensor", "metadata", _label_column(cell["cost_model"], cell["task_variant"])]
    rows_per_file = {record["path"]: record["n_rows"] for record in roster["source_files"]}
    handles: dict[str, Any] = {}
    layouts: dict[str, list[int]] = {}
    seen: set[str] = set()
    try:
        for task in roster["tasks"]:
            task_id, name, row_index = task["sample_id"], task["source_file"], task["row_index"]
            if task_id in seen:
                raise ValueError(f"Duplicate roster sample {task_id}")
            seen.add(task_id)
            if name not in handles:
                parquet = pq.ParquetFile(root / name)
                if parquet.metadata.num_rows != rows_per_file[name]:
                    raise ValueError(
                        f"{name} has {parquet.metadata.num_rows} rows; the roster records {rows_per_file[name]}"
                    )
                missing = [column for column in columns if column not in parquet.schema_arrow.names]
                if missing:
                    raise ValueError(f"{name} lacks columns {missing}")
                handles[name] = parquet
                layouts[name] = [parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)]
            offset = row_index
            for group, size in enumerate(layouts[name]):
                if offset < size:
                    break
                offset -= size
            else:
                raise ValueError(f"{task_id}: row {row_index} is outside {name}")
            row = handles[name].read_row_group(group, columns=columns).slice(offset, 1).to_pylist()[0]
            if row.get("id") != task_id:
                raise ValueError(f"{task_id}: {name} row {row_index} holds id {row.get('id')!r}")
            yield task, row
    finally:
        for parquet in handles.values():
            parquet.close()
