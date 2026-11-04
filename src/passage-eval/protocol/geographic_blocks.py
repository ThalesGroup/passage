"""Build paired geographic blocks from authenticated source-tile footprints.

The caller must authenticate records and decide whether these blocks support the
intended statistical inference. Sharing a block is not proof that different
blocks are independent or exchangeable.

Two rules are implemented (``BLOCK_RULES``):

* ``source_window_union`` (rule A, ``source_window_blocks``): the grid cells of
  every source tile a task uses, merged by transitive union across tasks;
* ``main_tile_cell`` (rule B, ``main_tile_blocks``): the grid cell of the task's
  main source tile (``metadata.tile.source``), without any union. A task whose used
  tiles span several cells is reported (``crossing_task_ids``) but keeps the cell
  of its main tile.

``task_footprint`` returns the input a rule needs from a roster task, or ``None``
when that footprint is absent or undecodable; such a task is never given an
invented singleton block. A roster may instead declare the sealed ``UNPLACED_BLOCK``
for these tasks (``io.roster``).

``block_function_identity`` names the version of a rule: the record schema, the
callable and the SHA-256 of this source file. A sealed roster stores that identity,
so any edit to this file changes the identity and invalidates the rosters sealed
before it.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping
from numbers import Integral
from pathlib import Path
from typing import Any

BLOCKS_SCHEMA = "passage-source-window-geographic-blocks/v1"
SOURCE_WINDOW_UNION = "source_window_union"
MAIN_TILE_CELL = "main_tile_cell"
BLOCK_RULES = (SOURCE_WINDOW_UNION, MAIN_TILE_CELL)
DEFAULT_BLOCK_RULE = SOURCE_WINDOW_UNION
#: The one block of every task whose footprint is absent or undecodable, when a roster declares it.
UNPLACED_BLOCK = "unplaced"
_CALLABLES = {SOURCE_WINDOW_UNION: "source_window_blocks", MAIN_TILE_CELL: "main_tile_blocks"}


def block_function_identity(rule: str = DEFAULT_BLOCK_RULE) -> dict[str, str]:
    """The schema, location and source digest of the function that implements ``rule``."""
    if rule not in _CALLABLES:
        raise ValueError(f"unknown block rule {rule!r}; expected one of {list(BLOCK_RULES)}")
    return {
        "schema": BLOCKS_SCHEMA,
        "module": __name__,
        "callable": _CALLABLES[rule],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def _check_degrees(block_degrees: Any) -> None:
    if type(block_degrees) is not int or block_degrees < 1 or 180 % block_degrees or 360 % block_degrees:
        raise ValueError("block_degrees must be a positive divisor of both 180 and 360")


def _tile(window: Any) -> tuple[int, int]:
    """Integer one-degree tile coordinates, longitude normalized across the antimeridian."""
    if not isinstance(window, Mapping):
        raise TypeError("Source windows must be mappings")
    lat, lon = window.get("lat"), window.get("lon")
    if (
        isinstance(lat, bool)
        or isinstance(lon, bool)
        or not isinstance(lat, Integral)
        or not isinstance(lon, Integral)
        or not -90 <= lat < 90
        or not -180 <= lon <= 180
    ):
        raise ValueError("Source tile latitude and longitude must be valid integers")
    return int(lat), (int(lon) + 180) % 360 - 180


def _cell(tile: tuple[int, int], block_degrees: int) -> tuple[int, int]:
    lat, lon = tile
    return (lat + 90) // block_degrees, (lon + 180) // block_degrees


def _task_id(record: Any, ids: set[str]) -> str:
    if not isinstance(record, Mapping):
        raise TypeError("Geographic task records must be mappings")
    task = record.get("sample_id")
    if not isinstance(task, str) or not task.strip() or task in ids:
        raise ValueError("A unique nonempty sample_id is required for every task")
    return task


def source_window_blocks(records: Iterable[Mapping[str, Any]], *, block_degrees: int = 5) -> dict[str, Any]:
    """Rule A: merge frozen geographic grid cells whenever a task uses more than one.

    Source windows use integer one-degree tile coordinates, with longitude
    normalized across the antimeridian. Components receive the lexicographically
    smallest grid cell as a stable ID, independent of input order. A missing,
    malformed or duplicate task footprint is an error, never a singleton block.
    """
    _check_degrees(block_degrees)
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    items = []
    ids: set[str] = set()
    crossings = 0

    def find(cell):
        parent.setdefault(cell, cell)
        while parent[cell] != cell:
            parent[cell] = parent[parent[cell]]
            cell = parent[cell]
        return cell

    def union(left, right):
        x, y = find(left), find(right)
        if x != y:
            lower, upper = sorted((x, y))
            parent[upper] = lower

    for record in records:
        task = _task_id(record, ids)
        windows = record.get("source_windows")
        if not isinstance(windows, list) or not windows:
            raise ValueError("Every task requires known source windows")
        cells = sorted({_cell(_tile(window), block_degrees) for window in windows})
        for cell in cells:
            find(cell)
            union(cells[0], cell)
        crossings += int(len(cells) > 1)
        ids.add(task)
        items.append((task, cells[0]))
    if not items:
        raise ValueError("A nonempty task roster is required")
    task_blocks = {task: f"geo{block_degrees}:{find(cell)[0]:02d}:{find(cell)[1]:02d}" for task, cell in items}
    counts = Counter(task_blocks.values())
    return {
        "schema": BLOCKS_SCHEMA,
        "rule": SOURCE_WINDOW_UNION,
        "block_degrees": block_degrees,
        "task_blocks": task_blocks,
        "number_of_tasks": len(items),
        "number_of_blocks": len(counts),
        "grid_cell_count": len(parent),
        "crossing_task_count": crossings,
        "block_sizes": dict(sorted(counts.items())),
        "statistical_independence_proven": False,
    }


def main_tile_blocks(records: Iterable[Mapping[str, Any]], *, block_degrees: int = 5) -> dict[str, Any]:
    """Rule B: the block of a task is the grid cell of its main source tile, without union.

    Each record needs ``source_tile`` (``{"lat", "lon"}`` integer one-degree coordinates). Its optional
    ``source_windows`` only count the tasks whose used tiles span more than one cell (``crossing_task_ids``); they
    never change a block. Block IDs (``cell<degrees>:<row>:<column>``) differ from rule A's (``geo<degrees>:...``).
    """
    _check_degrees(block_degrees)
    ids: set[str] = set()
    task_blocks: dict[str, str] = {}
    crossing: list[str] = []
    for record in records:
        task = _task_id(record, ids)
        tile = record.get("source_tile")
        if tile is None:
            raise ValueError("Every task requires its main source tile")
        row, column = _cell(_tile(tile), block_degrees)
        windows = record.get("source_windows") or []
        if not isinstance(windows, list):
            raise ValueError("source_windows must be a list")
        if len({_cell(_tile(window), block_degrees) for window in windows} | {(row, column)}) > 1:
            crossing.append(task)
        ids.add(task)
        task_blocks[task] = f"cell{block_degrees}:{row:02d}:{column:02d}"
    if not task_blocks:
        raise ValueError("A nonempty task roster is required")
    counts = Counter(task_blocks.values())
    return {
        "schema": BLOCKS_SCHEMA,
        "rule": MAIN_TILE_CELL,
        "block_degrees": block_degrees,
        "task_blocks": task_blocks,
        "number_of_tasks": len(task_blocks),
        "number_of_blocks": len(counts),
        "grid_cell_count": len(counts),
        "crossing_task_count": len(crossing),
        "crossing_task_ids": crossing,
        "block_sizes": dict(sorted(counts.items())),
        "statistical_independence_proven": False,
    }


def compute_blocks(records: Iterable[Mapping[str, Any]], *, rule: str, block_degrees: int) -> dict[str, Any]:
    """Apply the named block rule to block inputs from ``task_footprint``."""
    if rule == SOURCE_WINDOW_UNION:
        return source_window_blocks(records, block_degrees=block_degrees)
    if rule == MAIN_TILE_CELL:
        return main_tile_blocks(records, block_degrees=block_degrees)
    raise ValueError(f"unknown block rule {rule!r}; expected one of {list(BLOCK_RULES)}")


def task_footprint(task: Mapping[str, Any], rule: str) -> dict[str, Any] | None:
    """The block input of a roster task under ``rule``, or ``None`` when the footprint the rule needs is absent.

    Rule A needs the task's nonempty ``source_windows``. Rule B needs a decodable ``source_tile``; a missing or
    malformed main tile leaves the task without footprint (it is never repaired from the used tiles).
    """
    if rule == SOURCE_WINDOW_UNION:
        windows = task.get("source_windows")
        return {"sample_id": task["sample_id"], "source_windows": windows} if windows else None
    if rule == MAIN_TILE_CELL:
        tile = task.get("source_tile")
        try:
            _tile(tile)
        except (TypeError, ValueError):
            return None
        return {
            "sample_id": task["sample_id"],
            "source_tile": {"lat": tile["lat"], "lon": tile["lon"]},
            "source_windows": list(task.get("source_windows") or []),
        }
    raise ValueError(f"unknown block rule {rule!r}; expected one of {list(BLOCK_RULES)}")
