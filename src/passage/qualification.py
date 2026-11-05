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

"""Projected, resumable core-population qualification and a frozen audit roster.

Selection uses ordinary metadata only, before tensor/reference checks. The
diagnostic strata deliberately have unequal inclusion probabilities: audit
results are not population estimates. SQLite retains exact IDs and geographic
footprints without materializing tensors, images, or all intersecting pairs.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from datetime import date, datetime
import fcntl
import hashlib
import heapq
from itertools import combinations
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import sqlite3
import stat
import tempfile
import time
from typing import Any

from passage.dataset import DatasetSource, POPULATION_SPLITS, SPLITS, normalize_cell

RESOLUTIONS = (64, 128, 256, 512, 1024, 2048, 4096)
RELIEF_EDGES = (0, 200, 500, 1000, 3000, math.inf)
OCCUPANCY_EDGES = (0, 0.05, 0.15, 0.3, 0.5, 1)
PROTOCOL_VERSION = "passage-metadata-qualification-v1"
DEFAULT_SEED = 20261010
CHECKPOINT_SECONDS = 10.0
_ORDINARY = ("index", "resolution", "split", "calibration", "tile", "crop", "markers", "obstacles", "provenance")
_REQUIRED = (
    "resolution",
    "split",
    "calibration.min",
    "calibration.max",
    "tile.source.name",
    "crop.min_elevation",
    "crop.max_elevation",
    "crop.top_left.lat",
    "crop.top_left.lon",
    "crop.bottom_right.lat",
    "crop.bottom_right.lon",
    "obstacles.ratio_actual",
)
_STAT_KEYS = ("device", "inode", "size_bytes", "mtime_ns", "ctime_ns", "resolved_path")


def _json_default(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value {type(value).__name__}")


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default, allow_nan=False)


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _safe(value: Any) -> Any:
    """Preserve malformed/nonfinite observations in JSON without accepting them."""
    if isinstance(value, float) and not math.isfinite(value):
        return {"nonfinite": str(value)}
    if isinstance(value, dict):
        return {key: _safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(item) for item in value]
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"binary_hex": bytes(value).hex()}
    return value


def _write_json(path: Path, value: Any) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True, default=_json_default, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _get(metadata: Any, path: str) -> Any:
    value = metadata
    for key in path.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _labels(edges: tuple[float, ...], last_closed: bool = False) -> list[str]:
    def fmt(number: float) -> str:
        return "inf" if math.isinf(number) else f"{number:g}"

    return [
        f"[{fmt(low)},{fmt(high)}" + ("]" if last_closed and i == len(edges) - 2 else ")")
        for i, (low, high) in enumerate(zip(edges, edges[1:]))
    ]


def _bin(value: Any, edges: tuple[float, ...], last_closed: bool = False) -> str:
    if value is None:
        return "missing"
    number = _number(value)
    if number is None:
        return "invalid"
    if number < edges[0] or number > edges[-1]:
        return "out_of_range"
    for i, (low, high) in enumerate(zip(edges, edges[1:])):
        if low <= number < high or last_closed and i == len(edges) - 2 and number == high:
            return _labels(edges, last_closed)[i]
    return "out_of_range"


def _stats() -> dict[str, Any]:
    return {
        "rows": 0,
        "missing": {},
        "invalid": {},
        "resolution_mismatch": 0,
        "split_mismatch": 0,
        "relief": {"bins": dict.fromkeys(_labels(RELIEF_EDGES), 0), "missing": 0, "invalid": 0, "out_of_range": 0},
        "occupancy": {
            "bins": dict.fromkeys(_labels(OCCUPANCY_EDGES, True), 0),
            "missing": 0,
            "invalid": 0,
            "out_of_range": 0,
        },
        "calibration": {"invalid_rows": 0, "pairs": {}},
        "footprints": {},
        "odd_strata": {},
        "duplicate_id_rows": 0,
        "duplicate_id_values": 0,
        "source_tiles": {},
        "support_tiles": {},
    }


def _increment(mapping: dict[str, Any], key: str, n: int = 1) -> None:
    mapping[key] = mapping.get(key, 0) + n


def _merge(a: dict[str, Any], b: dict[str, Any]) -> None:
    for key, value in b.items():
        if isinstance(value, dict):
            _merge(a.setdefault(key, {}), value)
        else:
            a[key] = a.get(key, 0) + value


def _footprint(metadata: Any) -> tuple[str, tuple[float, float, float, float] | None]:
    raw = [
        _get(metadata, key)
        for key in ("crop.top_left.lon", "crop.bottom_right.lon", "crop.bottom_right.lat", "crop.top_left.lat")
    ]
    if any(value is None for value in raw):
        return "missing", None
    values = [_number(value) for value in raw]
    if any(value is None for value in values):
        return "invalid", None
    west, east, south, north = values
    if west > east:
        return "antimeridian_or_reversed_longitude", None
    if not (-180 <= west < east <= 180 and -90 <= south < north <= 90):
        return "out_of_range_or_degenerate", None
    return "valid", (west, east, south, north)


def _tile_name(tile: Any) -> str | None:
    if isinstance(tile, dict):
        name = tile.get("name")
        if isinstance(name, str) and name.strip():
            return name
        lat, lon = _number(tile.get("lat")), _number(tile.get("lon"))
        if lat is not None and lon is not None:
            return f"lat={lat:g},lon={lon:g}"
    return None


def _describe(
    row: dict[str, Any], resolution: int, split: str, seed: int, locator: dict[str, Any], stats: dict[str, Any]
) -> dict[str, Any]:
    metadata, sample_id = row.get("metadata"), row.get("id")
    stats["rows"] += 1
    valid_id = isinstance(sample_id, str) and bool(sample_id.strip())
    if not valid_id:
        _increment(stats["missing"] if sample_id is None or sample_id == "" else stats["invalid"], "id")
    if not isinstance(metadata, dict):
        _increment(stats["missing"] if metadata is None else stats["invalid"], "metadata")
    for path in _REQUIRED:
        if _get(metadata, path) is None:
            _increment(stats["missing"], path)
    if _get(metadata, "resolution") != resolution:
        stats["resolution_mismatch"] += 1
    declared_split = _get(metadata, "split")
    if not isinstance(declared_split, str) or SPLITS.get(declared_split) != split:
        stats["split_mismatch"] += 1
        if declared_split is not None and not isinstance(declared_split, str):
            _increment(stats["invalid"], "split")
    low, high = (_number(_get(metadata, f"calibration.{key}")) for key in ("min", "max"))
    if low is None or high is None or high <= low:
        stats["calibration"]["invalid_rows"] += 1
    else:
        _increment(stats["calibration"]["pairs"], _canonical([low, high]))
    crop_low, crop_high = (_number(_get(metadata, f"crop.{key}_elevation")) for key in ("min", "max"))
    raw_low, raw_high = (_get(metadata, f"crop.{key}_elevation") for key in ("min", "max"))
    relief = crop_high - crop_low if crop_low is not None and crop_high is not None else None
    relief_bin = (
        _bin(relief, RELIEF_EDGES)
        if relief is not None
        else "missing"
        if raw_low is None or raw_high is None
        else "invalid"
    )
    occupancy = _get(metadata, "obstacles.ratio_actual")
    occupancy_bin = _bin(occupancy, OCCUPANCY_EDGES, True)
    for name, label in (("relief", relief_bin), ("occupancy", occupancy_bin)):
        _increment(stats[name]["bins"] if label.startswith("[") else stats[name], label)
    stratum = f"relief={relief_bin};occupancy={occupancy_bin}"
    _increment(stats["odd_strata"], stratum)
    footprint_status, bbox = _footprint(metadata)
    _increment(stats["footprints"], footprint_status)
    source_tile = _tile_name(_get(metadata, "tile.source"))
    support = defaultdict(set)
    if source_tile:
        support["source"].add(source_tile)
    for kind in ("used", "overlapping"):
        tiles = _get(metadata, f"tile.{kind}")
        if isinstance(tiles, list):
            for tile in tiles:
                name = _tile_name(tile)
                if name:
                    support[kind].add(name)
        else:
            _increment(stats["missing"] if tiles is None else stats["invalid"], f"tile.{kind}")
    ordinary = {name: metadata[name] for name in _ORDINARY if isinstance(metadata, dict) and name in metadata}
    ordinary_hash = _hash(_safe(ordinary))
    rank = _hash(
        {
            "seed": seed,
            "resolution": resolution,
            "split": split,
            "id": _safe(sample_id),
            "ordinary_metadata_sha256": ordinary_hash,
            "locator": locator,
        }
    )
    return {
        "sample_id": sample_id if valid_id else None,
        "source_tile": source_tile,
        "stratum": stratum,
        "rank": rank,
        "ordinary_metadata_sha256": ordinary_hash,
        "bbox": bbox,
        "relief": relief,
        "occupancy": _number(occupancy),
        "calibration_min": low,
        "calibration_max": high,
        "support": support,
        "eligible": valid_id and isinstance(metadata, dict),
    }


def _rtree_name(resolution: int, split: str) -> str:
    return f"geo_{resolution}_{split}"


def _database(path: Path, resolutions: tuple[int, ...]) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA cache_size=-65536")
    db.executescript("""
        CREATE TABLE IF NOT EXISTS shards (
          path TEXT PRIMARY KEY, resolution INTEGER, split TEXT, stats TEXT, schema_record TEXT);
        CREATE TABLE IF NOT EXISTS scenes (
          pk INTEGER PRIMARY KEY, resolution INTEGER, split TEXT, shard TEXT, row_index INTEGER,
          row_group INTEGER, row_in_group INTEGER, sample_id TEXT, source_tile TEXT, stratum TEXT,
          rank TEXT, ordinary_hash TEXT, relief REAL, occupancy REAL, calibration_min REAL, calibration_max REAL,
          west REAL, east REAL, south REAL, north REAL);
        CREATE INDEX IF NOT EXISTS scene_cell ON scenes(resolution, split);
        CREATE TABLE IF NOT EXISTS id_counts (
          resolution INTEGER, split TEXT, sample_id TEXT, occurrences INTEGER,
          PRIMARY KEY(resolution,split,sample_id)) WITHOUT ROWID;
        CREATE TABLE IF NOT EXISTS tiles (tile_id INTEGER PRIMARY KEY, name TEXT UNIQUE);
        CREATE TABLE IF NOT EXISTS scene_tiles (
          scene_pk INTEGER, tile_id INTEGER, kind TEXT, PRIMARY KEY(scene_pk,tile_id,kind)) WITHOUT ROWID;
        CREATE INDEX IF NOT EXISTS tiles_incidence ON scene_tiles(tile_id,kind,scene_pk);
        CREATE TABLE IF NOT EXISTS candidates (scene_pk INTEGER PRIMARY KEY);
        CREATE TABLE IF NOT EXISTS geography (resolution INTEGER PRIMARY KEY, result TEXT);
        CREATE TABLE IF NOT EXISTS cell_summaries (cell TEXT PRIMARY KEY, result TEXT);
        CREATE TABLE IF NOT EXISTS cell_aggregates (cell TEXT PRIMARY KEY, result TEXT);
    """)
    for resolution in resolutions:
        for split in POPULATION_SPLITS:
            db.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS {_rtree_name(resolution, split)} "
                "USING rtree(pk,west,east,south,north)"
            )
    db.commit()
    return db


def _source_stat(source: DatasetSource, path: str | Path) -> dict[str, Any]:
    root = source.root.resolve()
    shard = Path(path)
    if not shard.is_absolute():
        shard = root / shard
    resolved = shard.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Source shard resolves outside snapshot root: {shard}")
    observed = shard.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"Source shard is not a regular file: {shard}")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "size_bytes": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
        "resolved_path": str(resolved),
    }


def _assert_unchanged(source: DatasetSource, record: dict[str, Any]) -> None:
    observed = _source_stat(source, record["path"])
    if any(record.get(key) != value for key, value in observed.items()):
        raise ValueError(f"Source shard changed from frozen manifest: {record['path']}")


def _authentication_binding(source: DatasetSource, path: str | Path | None) -> tuple[dict | None, dict]:
    if path is None:
        return None, {}
    manifest_path = Path(path).resolve()
    contents = manifest_path.read_bytes()
    manifest = json.loads(contents)
    if (
        manifest.get("schema_version") != 1
        or manifest.get("repo_type") != "dataset"
        or manifest.get("root") != str(source.root.resolve())
        or manifest.get("repo") != source.repo
        or not isinstance(source.revision, str)
        or manifest.get("revision") != source.revision.lower()
    ):
        raise ValueError("Authentication manifest identity differs from census source")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Authentication manifest files must be a nonempty list")
    expected = {}
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("Invalid authentication manifest object")
        name = entry.get("path")
        if not isinstance(name, str):
            raise ValueError("Invalid authentication manifest path")
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or ".." in relative.parts
            or "\\" in name
            or str(relative) != name
            or name in expected
        ):
            raise ValueError(f"Unsafe or duplicate authentication manifest path: {name!r}")
        algorithm, digest, size = (entry.get(key) for key in ("algorithm", "expected_hash", "size"))
        length = 64 if algorithm == "sha256" else 40 if algorithm == "git-blob-sha1" else None
        if (
            length is None
            or not isinstance(digest, str)
            or not re.fullmatch(rf"[0-9a-f]{{{length}}}", digest)
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise ValueError(f"Invalid authentication manifest object hash/size: {name}")
        expected[name] = entry
    return {
        "path": str(manifest_path),
        "manifest_file_sha256": hashlib.sha256(contents).hexdigest(),
        "manifest_sha256": hashlib.sha256((_canonical(manifest) + "\n").encode()).hexdigest(),
        "status": "expected_objects_bound; authentication completion is separate",
        "content_verified": False,
    }, expected


def _manifest(
    source: DatasetSource,
    resolutions: tuple[int, ...],
    expected_objects: dict | None = None,
    schema_registry: dict | None = None,
) -> dict[str, Any]:
    import pyarrow.parquet as pq

    cells = {}
    for resolution in resolutions:
        for split in POPULATION_SPLITS:
            key = f"{resolution}/{split}"
            try:
                shards = source._shards(resolution, split, False)
            except FileNotFoundError:
                cells[key] = {"status": "missing", "shards": []}
                continue
            records = []
            for shard in shards:
                relative_path = str(shard.relative_to(source.root))
                record = {"path": relative_path, **_source_stat(source, relative_path)}
                if expected_objects is not None:
                    expected = expected_objects.get(record["path"])
                    if expected is None:
                        raise ValueError(
                            f"Extra local core shard absent from authentication manifest: {record['path']}"
                        )
                    if record["size_bytes"] != expected["size"]:
                        raise ValueError(f"Local shard size differs from authentication manifest: {record['path']}")
                    record["expected_object"] = expected
                try:
                    with pq.ParquetFile(shard) as parquet:
                        schema = parquet.schema_arrow
                        schema_text = str(schema)
                        schema_id = _hash(schema_text)
                        if schema_registry is not None:
                            schema_registry.setdefault(schema_id, schema_text)
                        record.update(
                            rows=parquet.metadata.num_rows,
                            columns=schema.names,
                            schema_sha256=schema_id,
                            row_groups=[
                                parquet.metadata.row_group(i).num_rows for i in range(parquet.metadata.num_row_groups)
                            ],
                            missing_required_columns=sorted({"id", "metadata", "tensor"} - set(schema.names)),
                        )
                        import pyarrow as pa

                        invalid = []
                        for name, predicates in (
                            ("id", (pa.types.is_string, pa.types.is_large_string)),
                            ("metadata", (pa.types.is_struct,)),
                            ("tensor", (pa.types.is_binary, pa.types.is_large_binary)),
                        ):
                            if name in schema.names and not any(p(schema.field(name).type) for p in predicates):
                                invalid.append(name)
                        record["invalid_column_types"] = invalid
                except (OSError, ValueError) as exc:
                    record.update(error=f"{type(exc).__name__}: {exc}")
                _assert_unchanged(source, record)
                records.append(record)
            cells[key] = {"status": "available", "shards": records}
    if expected_objects is not None:
        prefixes = {f"{resolution}x{resolution}/{split}/" for resolution in resolutions for split in POPULATION_SPLITS}
        required = {
            name
            for name in expected_objects
            if name.endswith(".parquet") and any(name.startswith(prefix) for prefix in prefixes)
        }
        observed = {r["path"] for cell in cells.values() for r in cell["shards"]}
        if observed != required:
            raise ValueError(
                f"Local core source differs from authentication manifest; missing={sorted(required - observed)}, "
                f"extra={sorted(observed - required)}"
            )
    return cells


def _revalidate_source(source: DatasetSource, protocol: dict[str, Any]) -> None:
    for resolution in protocol["resolutions"]:
        for split in POPULATION_SPLITS:
            records = protocol["manifest"][f"{resolution}/{split}"]["shards"]
            expected = {record["path"] for record in records}
            try:
                current = {str(path.relative_to(source.root)) for path in source._shards(resolution, split, False)}
            except FileNotFoundError:
                current = set()
            if current != expected:
                raise ValueError(f"Source shard set changed at {resolution}/{split}")
            for record in records:
                _assert_unchanged(source, record)
    authentication = protocol.get("authentication_manifest")
    if (
        authentication
        and hashlib.sha256(Path(authentication["path"]).read_bytes()).hexdigest()
        != authentication["manifest_file_sha256"]
    ):
        raise ValueError("Authentication manifest changed from frozen protocol")


def read_roster_row(source: DatasetSource, entry: dict[str, Any], columns: list[str] | None = None) -> dict[str, Any]:
    """Read a stored locator in bounded batches, including high-resolution tensors.

    Parquet row groups can contain 20,000 scenes. Reading a whole row group of
    tensors would exceed memory bounds; skipping bounded batches may cost I/O,
    but never decodes preceding blobs or retains the complete group.
    """
    import pyarrow.parquet as pq

    if source.root is None:
        raise ValueError("Roster locators require a local pinned snapshot")
    resolution, split = normalize_cell(entry["resolution"], entry["split"])
    locator = entry["locator"]
    root = source.root.resolve()
    shard = (root / locator["shard"]).resolve()
    if not shard.is_relative_to(root):
        raise ValueError("Roster shard escapes source root")
    expected_parent = root / f"{resolution}x{resolution}" / split
    if shard.parent != expected_parent:
        raise ValueError("Roster shard differs from its resolution/split")
    before = _source_stat(source, locator["shard"])
    if entry.get("source_stat") is not None and before != entry["source_stat"]:
        raise ValueError("Roster source shard changed from its frozen fingerprint")
    projected = columns if columns is not None else ["id", "tensor", "metadata"]
    batch_size = source.batch_size_for(resolution, projected, requested=32)
    skip = locator["row_in_group"]
    with pq.ParquetFile(shard) as parquet:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=projected, row_groups=[locator["row_group"]]):
            if skip >= batch.num_rows:
                skip -= batch.num_rows
                continue
            selected = batch.slice(skip, 1).to_pylist()[0]
            if "id" in selected and selected["id"] != entry["sample_id"]:
                raise ValueError("Roster locator ID differs from frozen source")
            if _source_stat(source, locator["shard"]) != before:
                raise ValueError("Roster source shard changed during selected row read")
            return selected
    raise IndexError("Roster row locator outside its Parquet group")


def _candidate_pool(db: sqlite3.Connection, resolution: int, split: str) -> dict[tuple[str, str], list]:
    pools = defaultdict(list)
    for pk, tile, stratum, rank in db.execute(
        "SELECT s.pk,s.source_tile,s.stratum,s.rank FROM candidates c JOIN scenes s ON s.pk=c.scene_pk "
        "WHERE s.resolution=? AND s.split=?",
        (resolution, split),
    ):
        heapq.heappush(pools[(tile or "<missing>", stratum)], (-int(rank, 16), pk))
    return pools


def _scan_shard(
    db: sqlite3.Connection,
    source: DatasetSource,
    resolution: int,
    split: str,
    record: dict[str, Any],
    seed: int,
    batch_size: int,
    pools: dict,
    quota: int,
) -> None:
    import pyarrow.parquet as pq

    stats = _stats()
    additions, removals = set(), set()
    tile_cache = dict(db.execute("SELECT name,tile_id FROM tiles"))
    shard = source.root / record["path"]
    _assert_unchanged(source, record)
    with db:
        with pq.ParquetFile(shard) as parquet:
            import pyarrow as pa

            schema = parquet.schema_arrow
            projected = ["id"] if "id" in schema.names else []
            if "metadata" in schema.names:
                metadata_type = schema.field("metadata").type
                if pa.types.is_struct(metadata_type):
                    fields = {field.name for field in metadata_type}
                    ordinary_fields = [f"metadata.{name}" for name in _ORDINARY if name in fields]
                    projected.extend(ordinary_fields or ["metadata"])
                else:
                    projected.append("metadata")
            row_index = 0
            for group, count in enumerate(record["row_groups"]):
                offset = 0
                batches: Iterable = (
                    parquet.iter_batches(batch_size=batch_size, columns=projected, row_groups=[group])
                    if projected
                    else ([{}] * min(batch_size, count - i) for i in range(0, count, batch_size))
                )
                for batch in batches:
                    rows = batch.to_pylist() if hasattr(batch, "to_pylist") else batch
                    for row in rows:
                        locator = {
                            "shard": record["path"],
                            "row_index": row_index,
                            "row_group": group,
                            "row_in_group": offset,
                        }
                        item = _describe(row, resolution, split, seed, locator, stats)
                        bbox = item["bbox"] or (None,) * 4
                        cursor = db.execute(
                            "INSERT INTO scenes(resolution,split,shard,row_index,row_group,row_in_group,sample_id,"
                            "source_tile,stratum,rank,ordinary_hash,relief,occupancy,calibration_min,calibration_max,"
                            "west,east,south,north) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (
                                resolution,
                                split,
                                record["path"],
                                row_index,
                                group,
                                offset,
                                item["sample_id"],
                                item["source_tile"],
                                item["stratum"],
                                item["rank"],
                                item["ordinary_metadata_sha256"],
                                item["relief"],
                                item["occupancy"],
                                item["calibration_min"],
                                item["calibration_max"],
                                *bbox,
                            ),
                        )
                        pk = cursor.lastrowid
                        if item["sample_id"] is not None:
                            inserted = db.execute(
                                "INSERT OR IGNORE INTO id_counts VALUES(?,?,?,1)",
                                (resolution, split, item["sample_id"]),
                            ).rowcount
                            if not inserted:
                                previous = db.execute(
                                    "SELECT occurrences FROM id_counts WHERE resolution=? AND split=? AND sample_id=?",
                                    (resolution, split, item["sample_id"]),
                                ).fetchone()[0]
                                db.execute(
                                    "UPDATE id_counts SET occurrences=occurrences+1 WHERE resolution=? "
                                    "AND split=? AND sample_id=?",
                                    (resolution, split, item["sample_id"]),
                                )
                                stats["duplicate_id_rows"] += 1
                                stats["duplicate_id_values"] += previous == 1
                        if item["source_tile"]:
                            _increment(stats["source_tiles"], item["source_tile"])
                        for tile in set().union(*item["support"].values()):
                            _increment(stats["support_tiles"], tile)
                        if item["bbox"]:
                            db.execute(f"INSERT INTO {_rtree_name(resolution, split)} VALUES(?,?,?,?,?)", (pk, *bbox))
                        for kind, tiles in item["support"].items():
                            for tile in tiles:
                                if tile not in tile_cache:
                                    db.execute("INSERT OR IGNORE INTO tiles(name) VALUES(?)", (tile,))
                                    tile_cache[tile] = db.execute(
                                        "SELECT tile_id FROM tiles WHERE name=?", (tile,)
                                    ).fetchone()[0]
                                db.execute("INSERT INTO scene_tiles VALUES(?,?,?)", (pk, tile_cache[tile], kind))
                        if item["eligible"]:
                            key = (item["source_tile"] or "<missing>", item["stratum"])
                            candidate = (-int(item["rank"], 16), pk)
                            if len(pools[key]) < quota:
                                heapq.heappush(pools[key], candidate)
                                additions.add(pk)
                            elif candidate > pools[key][0]:
                                _, old_pk = heapq.heapreplace(pools[key], candidate)
                                if old_pk in additions:
                                    additions.remove(old_pk)
                                else:
                                    removals.add(old_pk)
                                additions.add(pk)
                        offset += 1
                        row_index += 1
        _assert_unchanged(source, record)
        db.executemany("DELETE FROM candidates WHERE scene_pk=?", ((pk,) for pk in removals))
        db.executemany("INSERT INTO candidates VALUES(?)", ((pk,) for pk in additions))
        db.execute(
            "INSERT INTO shards VALUES(?,?,?,?,?)",
            (record["path"], resolution, split, _canonical(stats), _canonical(record)),
        )
        cell = f"{resolution}/{split}"
        previous = db.execute("SELECT result FROM cell_aggregates WHERE cell=?", (cell,)).fetchone()
        aggregate = json.loads(previous[0]) if previous else _stats()
        _merge(aggregate, stats)
        db.execute("INSERT OR REPLACE INTO cell_aggregates VALUES(?,?)", (cell, _canonical(aggregate)))


def _summary(
    db: sqlite3.Connection,
    protocol: dict[str, Any],
    *,
    status: str = "in_progress",
    refresh: str | None = None,
    protocol_sha256: str | None = None,
) -> dict[str, Any]:
    result = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256 or _hash(protocol),
        "source": protocol["source"],
        "status": status,
        "cells": {},
        "geography": {},
        "scope": "all core rows in frozen local shard manifest; projected id/metadata only",
        "content_verified": False,
        "source_verification": "local device/inode/ctime/size/mtime/schema/row-footer identity; "
        "expected objects bound when supplied; byte authentication is a separate receipt",
    }
    cached = dict(db.execute("SELECT cell,result FROM cell_summaries"))
    shard_counts = {
        f"{resolution}/{split}": count
        for resolution, split, count in db.execute(
            "SELECT resolution,split,COUNT(*) FROM shards GROUP BY resolution,split"
        )
    }
    for key, cell in protocol["manifest"].items():
        if key in cached and key != refresh:
            previous = json.loads(cached[key])
            if previous["scanned_shards"] == shard_counts.get(key, 0):
                result["cells"][key] = previous
                continue
        resolution, split = key.split("/")
        aggregate = db.execute("SELECT result FROM cell_aggregates WHERE cell=?", (key,)).fetchone()
        stats = json.loads(aggregate[0]) if aggregate else _stats()
        scanned_count = shard_counts.get(key, 0)
        stats.update(
            status="complete" if scanned_count == len(cell["shards"]) else "in_progress",
            expected_footer_rows=sum(r.get("rows", 0) for r in cell["shards"]),
            scanned_shards=scanned_count,
            total_shards=len(cell["shards"]),
            schema_consistent=len({r.get("schema_sha256") for r in cell["shards"]}) <= 1,
            schemas=[
                {
                    name: r.get(name)
                    for name in ("path", "schema_sha256", "missing_required_columns", "invalid_column_types", "error")
                }
                for r in cell["shards"]
            ],
        )
        if cell["status"] == "missing":
            stats["status"] = "missing"
        elif any(r.get("error") for r in cell["shards"]):
            stats["status"] = "unreadable"
        elif any(r.get("missing_required_columns") or r.get("invalid_column_types") for r in cell["shards"]):
            stats["status"] = "invalid_schema"
        elif not stats["schema_consistent"]:
            stats["status"] = "inconsistent_schema"
        result["cells"][key] = stats
        db.execute("INSERT OR REPLACE INTO cell_summaries VALUES(?,?)", (key, _canonical(stats)))
    db.commit()
    result["population_scope_complete"] = all(
        cell["status"] not in ("missing", "unreadable", "in_progress") and cell["rows"] == cell["expected_footer_rows"]
        for cell in result["cells"].values()
    )
    for resolution, value in db.execute("SELECT resolution,result FROM geography"):
        result["geography"][str(resolution)] = json.loads(value)
    return result


def _select_roster(
    db: sqlite3.Connection, source: DatasetSource, protocol: dict[str, Any], protocol_sha256: str | None = None
) -> dict[str, Any]:
    scenes = []
    coverage = {}
    records = {record["path"]: record for cell in protocol["manifest"].values() for record in cell["shards"]}
    for resolution in protocol["resolutions"]:
        for split in POPULATION_SPLITS:
            quota = protocol["audit_counts"][split]
            candidates = [
                dict(
                    zip(
                        (
                            "pk",
                            "sample_id",
                            "source_tile",
                            "stratum",
                            "rank",
                            "ordinary_metadata_sha256",
                            "shard",
                            "row_index",
                            "row_group",
                            "row_in_group",
                        ),
                        row,
                    )
                )
                for row in db.execute(
                    "SELECT s.pk,s.sample_id,s.source_tile,s.stratum,s.rank,s.ordinary_hash,s.shard,s.row_index,"
                    "s.row_group,s.row_in_group FROM candidates c JOIN scenes s ON s.pk=c.scene_pk "
                    "WHERE s.resolution=? AND s.split=? ORDER BY s.rank,s.pk",
                    (resolution, split),
                )
            ]
            selected, used_tiles, used_strata = [], set(), Counter()
            while candidates and len(selected) < quota:
                # Distinct source tiles first; then least represented occupied
                # fixed stratum; metadata hash is the deterministic tiebreaker.
                entry = min(
                    candidates,
                    key=lambda e: (
                        e["source_tile"] in used_tiles if e["source_tile"] else True,
                        used_strata[e["stratum"]],
                        e["rank"],
                        e["pk"],
                    ),
                )
                candidates.remove(entry)
                selected.append(entry)
                if entry["source_tile"]:
                    used_tiles.add(entry["source_tile"])
                used_strata[entry["stratum"]] += 1
            coverage[f"{resolution}/{split}"] = {
                "requested": quota,
                "selected": len(selected),
                "distinct_source_tiles": len(used_tiles),
                "selected_strata": dict(used_strata),
            }
            for entry in selected:
                locator = {name: entry.pop(name) for name in ("shard", "row_index", "row_group", "row_in_group")}
                entry.pop("pk")
                entry.update(resolution=resolution, split=split, locator=locator)
                entry["source_stat"] = {name: records[locator["shard"]][name] for name in _STAT_KEYS}
                row = read_roster_row(source, entry, ["id", "metadata"])
                entry["metadata"] = _safe(row["metadata"])
                scenes.append(entry)
    body = {
        "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": protocol_sha256 or _hash(protocol),
        "source": protocol["source"],
        "seed": protocol["seed"],
        "population_estimation": False,
        "selection": protocol["selection"],
        "interpretation": "Diagnostic stratified selection has unequal inclusion probabilities; "
        "unweighted audit rates cannot estimate population error rates.",
        "frozen_before_tensor_or_reference_audit": True,
        "coverage": coverage,
        "scenes": scenes,
    }
    return {**body, "roster_sha256": _hash(body)}


def _geography(db: sqlite3.Connection, resolution: int, *, witness_limit: int = 6) -> dict[str, Any]:
    result = {
        "status": "complete",
        "scope": "all valid nonwrapping recorded crop rectangles at this resolution",
        "source_cell_windows": "unknown",
        "independence_claim": False,
        "interpretation": "Shared tiles and geographic rectangle intersection are separate diagnostics. "
        "No source-cell windows are recorded, so strict cell-support independence is unknown.",
        "split_pairs": {},
    }
    for left, right in combinations(POPULATION_SPLITS, 2):
        pair = {"positive_overlap_rows": {}, "boundary_contact_rows": {}, "witnesses": []}
        # Tile names denote recorded source/used/overlapping support, not a proof
        # that source cell windows overlap.
        for kind, label in (("source", "shared_source_tiles"), (None, "shared_support_tiles")):
            condition = " AND st.kind='source'" if kind else ""
            support_query = (
                "SELECT DISTINCT st.tile_id FROM scene_tiles st JOIN scenes s ON s.pk=st.scene_pk "
                "WHERE s.resolution=? AND s.split=?" + condition
            )
            names = [
                row[0]
                for row in db.execute(
                    "SELECT t.name FROM tiles t WHERE t.tile_id IN ("
                    + support_query
                    + " INTERSECT "
                    + support_query
                    + ") "
                    "ORDER BY t.name",
                    (resolution, left, resolution, right),
                )
            ]
            pair[label] = len(names)
            pair[label + "_names"] = names
        for split, other in ((left, right), (right, left)):
            rtree = _rtree_name(resolution, other)
            broad = "r.west<=a.east AND r.east>=a.west AND r.south<=a.north AND r.north>=a.south"
            # RTree coordinates have outward-rounded float32 bounds. The exact
            # double-precision predicates below prevent false overlaps/contact.
            strict = "b.west<a.east AND b.east>a.west AND b.south<a.north AND b.north>a.south"
            closed = "b.west<=a.east AND b.east>=a.west AND b.south<=a.north AND b.north>=a.south"
            boundary = closed + " AND (b.west=a.east OR b.east=a.west OR b.south=a.north OR b.north=a.south)"
            for name, predicate in (("positive_overlap_rows", strict), ("boundary_contact_rows", boundary)):
                exists = (
                    f"EXISTS(SELECT 1 FROM {rtree} r JOIN scenes b ON b.pk=r.pk WHERE {broad} AND {predicate} LIMIT 1)"
                )
                count = db.execute(
                    "SELECT COUNT(*) FROM scenes a WHERE a.resolution=? AND a.split=? "
                    f"AND a.west IS NOT NULL AND {exists}",
                    (resolution, split),
                ).fetchone()[0]
                pair[name][split] = count
                if split == left and count:
                    for pk, sample_id, shard, row_index in db.execute(
                        "SELECT a.pk,a.sample_id,a.shard,a.row_index FROM scenes a WHERE a.resolution=? "
                        f"AND a.split=? AND a.west IS NOT NULL AND {exists} LIMIT ?",
                        (resolution, split, witness_limit),
                    ):
                        other_id, other_shard, other_row = db.execute(
                            f"SELECT b.sample_id,b.shard,b.row_index FROM scenes a JOIN {rtree} r ON {broad} "
                            f"JOIN scenes b ON b.pk=r.pk WHERE a.pk=? AND {predicate} LIMIT 1",
                            (pk,),
                        ).fetchone()
                        pair["witnesses"].append(
                            {
                                "kind": name,
                                "split": split,
                                "sample_id": sample_id,
                                "shard": shard,
                                "row_index": row_index,
                                "other_split": other,
                                "other_sample_id": other_id,
                                "other_shard": other_shard,
                                "other_row_index": other_row,
                            }
                        )
        result["split_pairs"][f"{left}/{right}"] = pair
    return result


def run_metadata_census(
    source: DatasetSource,
    output_dir: str | Path,
    *,
    resolutions: tuple[int, ...] = RESOLUTIONS,
    seed: int = DEFAULT_SEED,
    batch_size: int = 2048,
    spatial: bool = True,
    progress: Callable[[dict[str, Any]], None] | None = None,
    authentication_manifest: str | Path | None = None,
) -> dict[str, Any]:
    """Scan the frozen local core population and produce a deterministic roster.

    Each shard commits its rows and exact aggregates atomically. JSON summaries
    checkpoint every ten seconds and at cell completion; progress callbacks
    report every committed shard. An interruption resumes at the next shard; changed source
    size/mtime/schema/footers or changed protocol settings are rejected. A full
    byte-content verification against the named remote commit is not claimed.
    """
    if source.root is None:
        raise ValueError("Full resumable census requires a local pinned snapshot")
    root, output = source.root.resolve(), Path(output_dir).resolve()
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError("Qualification output must be outside the source root without overlap")
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot root is not a directory: {root}")
    output.mkdir(parents=True, exist_ok=True)
    for name in (
        ".qualification.lock",
        "protocol.json",
        "summary.json",
        "audit_roster.json",
        "census.sqlite",
        "census.sqlite-wal",
        "census.sqlite-shm",
        "census.sqlite-journal",
    ):
        if (output / name).is_symlink():
            raise ValueError(f"Qualification output artifact must not be a symlink: {name}")
    with (output / ".qualification.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Qualification output already has an active writer") from exc
        return _run_metadata_census(
            source,
            output,
            resolutions=resolutions,
            seed=seed,
            batch_size=batch_size,
            spatial=spatial,
            progress=progress,
            authentication_manifest=authentication_manifest,
        )


def _run_metadata_census(
    source: DatasetSource,
    output_dir: str | Path,
    *,
    resolutions: tuple[int, ...],
    seed: int,
    batch_size: int,
    spatial: bool,
    progress: Callable[[dict[str, Any]], None] | None,
    authentication_manifest: str | Path | None,
) -> dict[str, Any]:
    resolutions = tuple(resolutions)
    if not resolutions or len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must be nonempty and unique")
    for resolution in resolutions:
        normalize_cell(resolution, "train")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    source.batch_size_for(resolutions[0], ["id", "metadata"], batch_size)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    authentication, expected_objects = _authentication_binding(source, authentication_manifest)
    schemas = {}
    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "source": source.identity(),
        "resolutions": list(resolutions),
        "qualification_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "splits": list(POPULATION_SPLITS),
        "population": "core",
        "projected_columns": ["id", "metadata"],
        "physical_metadata_columns": [f"metadata.{name}" for name in _ORDINARY],
        "seed": seed,
        "relief_bins": [0, 200, 500, 1000, 3000, "inf"],
        "occupancy_bins": list(OCCUPANCY_EDGES),
        "bin_convention": "left closed, right open; occupancy final bin includes 1; missing/invalid/outside separate",
        "audit_counts": {"validation": 16, "train": 4, "calibration": 4, "test": 4},
        "selection": "Distinct source tiles first, then least represented occupied fixed relief/occupancy "
        "stratum; ascending SHA256 of seed/cell/id/ordinary-metadata/locator breaks ties. "
        "Bounded per tile/stratum candidate pool retains the split quota.",
        "reference_audit": {
            "max_expansions": 100000,
            "variants": [
                "elevation/free",
                "elevation/obstacles",
                "slope/free",
                "slope/obstacles",
                "energy/free",
                "energy/obstacles",
            ],
            "retain_failures_and_censoring": True,
        },
        "spatial_audit": spatial,
        "coordinate_tolerance": 0,
        "spatial_witness_limit": 6,
        "authentication_manifest": authentication,
        "schemas": schemas,
        "manifest": _manifest(source, resolutions, expected_objects if authentication else None, schemas),
    }
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        if json.loads(protocol_path.read_text()) != protocol:
            raise ValueError("Frozen protocol/source manifest differs; choose a new output directory")
    else:
        _write_json(protocol_path, protocol)
    protocol_id = _hash(protocol)
    db = _database(output / "census.sqlite", resolutions)
    failed_shard = None
    try:
        summary = _summary(db, protocol, protocol_sha256=protocol_id)
        _write_json(output / "summary.json", summary)
        last_checkpoint = time.monotonic()
        for resolution in resolutions:
            for split in POPULATION_SPLITS:
                pools = _candidate_pool(db, resolution, split)
                for record in protocol["manifest"][f"{resolution}/{split}"]["shards"]:
                    if (
                        record.get("error")
                        or db.execute("SELECT 1 FROM shards WHERE path=?", (record["path"],)).fetchone()
                    ):
                        continue
                    try:
                        _scan_shard(
                            db,
                            source,
                            resolution,
                            split,
                            record,
                            seed,
                            batch_size,
                            pools,
                            protocol["audit_counts"][split],
                        )
                    except Exception:
                        failed_shard = record["path"]
                        raise
                    if time.monotonic() - last_checkpoint >= CHECKPOINT_SECONDS:
                        summary = _summary(db, protocol, protocol_sha256=protocol_id)
                        _write_json(output / "summary.json", summary)
                        last_checkpoint = time.monotonic()
                    if progress:
                        rows = db.execute(
                            "SELECT json_extract(result,'$.rows') FROM cell_aggregates WHERE cell=?",
                            (f"{resolution}/{split}",),
                        ).fetchone()[0]
                        progress(
                            {
                                "event": "shard_complete",
                                "resolution": resolution,
                                "split": split,
                                "shard": record["path"],
                                "rows": rows,
                            }
                        )
                summary = _summary(db, protocol, protocol_sha256=protocol_id)
                _write_json(output / "summary.json", summary)
                last_checkpoint = time.monotonic()
        roster_path = output / "audit_roster.json"
        if roster_path.exists():
            roster = json.loads(roster_path.read_text())
            if roster["protocol_sha256"] != protocol_id or roster["roster_sha256"] != _hash(
                {k: v for k, v in roster.items() if k != "roster_sha256"}
            ):
                raise ValueError("Frozen roster hash does not match its protocol/content")
        else:
            roster = _select_roster(db, source, protocol, protocol_sha256=protocol_id)
            _write_json(roster_path, roster)
        if progress:
            progress(
                {"event": "roster_frozen", "selected": len(roster["scenes"]), "roster_sha256": roster["roster_sha256"]}
            )
        if spatial:
            for resolution in resolutions:
                if db.execute("SELECT 1 FROM geography WHERE resolution=?", (resolution,)).fetchone():
                    continue
                geography = _geography(db, resolution, witness_limit=protocol["spatial_witness_limit"])
                with db:
                    db.execute("INSERT INTO geography VALUES(?,?)", (resolution, _canonical(geography)))
                summary = _summary(db, protocol, protocol_sha256=protocol_id)
                summary["roster"] = {"selected": len(roster["scenes"]), "roster_sha256": roster["roster_sha256"]}
                _write_json(output / "summary.json", summary)
                if progress:
                    progress({"event": "geography_complete", "resolution": resolution})
        _revalidate_source(source, protocol)
        summary = _summary(db, protocol, status="complete", protocol_sha256=protocol_id)
        summary["roster"] = {
            "selected": len(roster["scenes"]),
            "roster_sha256": roster["roster_sha256"],
            "population_estimation": False,
        }
        if not spatial:
            summary["geography"] = {
                str(resolution): {"status": "not_requested", "source_cell_windows": "unknown"}
                for resolution in resolutions
            }
        _write_json(output / "summary.json", summary)
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return summary
    except Exception as exc:
        summary = _summary(db, protocol, protocol_sha256=protocol_id)
        failure = {"error_type": type(exc).__name__, "error": str(exc)}
        if failed_shard:
            failure["shard"] = failed_shard
        summary.update(status="failed", failure=failure)
        _write_json(output / "summary.json", summary)
        raise
    finally:
        db.close()
