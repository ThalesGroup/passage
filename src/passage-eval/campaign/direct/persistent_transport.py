"""Local prototype: bounded input-only fragments and fail-closed per-request journal.

No network, device operation, data loading, or scoring occurs at import. Not a launcher.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import tarfile
import struct
import zipfile
from pathlib import Path

MAX_FRAGMENT = 128 * 1024 * 1024
_KEYS = {"sample_ids", "terrain", "obstacles", "starts", "goals"}
_BINARY_KEYS = {"sample_ids", "terrain", "obstacles", "starts", "goals"}


def require_session_drained(proof, *, target, candidate, job):
    """Refuse a candidate transition unless that exact session's cleanup is evidenced."""
    if (
        target not in ("orin", "tiger", "thor")
        or candidate not in ("seg", "reg")
        or not isinstance(job, str)
        or not re.fullmatch(r"job:[0-9a-f]{16}", job)
    ):
        raise ValueError("session identity differs")
    expected_scope = f"passage-edge-{target}-{job[4:]}.scope"
    flags = (
        "synchronization_complete",
        "runtime_close_complete",
        "unit_inactive",
        "cgroup_empty",
        "session_gpu_handles_observed",
    )
    if (
        not isinstance(proof, dict)
        or proof.get("target") != target
        or proof.get("candidate") != candidate
        or proof.get("scope") != expected_scope
        or any(proof.get(key) is not True for key in flags)
        or proof.get("session_gpu_handles") != []
    ):
        raise RuntimeError("session drain has not been established")


def _raw(path):
    path = Path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    with os.fdopen(fd, "rb") as stream:
        raw = stream.read(MAX_FRAGMENT + 1)
    if len(raw) > MAX_FRAGMENT:
        raise ValueError("input fragment exceeds bounded size")
    return raw


def _check(raw):
    def reject_constant(_):
        raise ValueError("input-only fragment contains a nonfinite number")

    def unique_object(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("input-only fragment has duplicate keys")
            obj[key] = value
        return obj

    value = json.loads(raw, parse_constant=reject_constant, object_pairs_hook=unique_object)
    if not isinstance(value, dict) or set(value) != _KEYS:
        raise ValueError("input-only fragment fields differ")
    arrays = [value[k] for k in _KEYS]
    ids = value["sample_ids"]
    if (
        not isinstance(ids, list)
        or not ids
        or any(not isinstance(x, str) or not x for x in ids)
        or len(ids) != len(set(ids))
        or any(not isinstance(a, list) or len(a) != len(ids) for a in arrays)
    ):
        raise ValueError("input-only fragment arrays differ")

    def numeric_tree(item):
        if isinstance(item, list):
            return all(numeric_tree(child) for child in item)
        return type(item) in (int, float, bool) and (type(item) is not float or math.isfinite(item))

    if any(not numeric_tree(value[name]) for name in _KEYS - {"sample_ids"}):
        raise ValueError("input-only fragment contains nonnumeric nested values")
    return ids


def _file_info(path):
    raw = _raw(path)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "ids": _check(raw)}


def build_manifest(target, procedure, paths, warmup, *, expected_ids, warmup_count=16):
    if target not in ("orin", "tiger", "thor") or procedure not in ("seg", "reg"):
        raise ValueError("target or procedure differs")
    fragments = [_file_info(path) for path in paths]
    warm = _file_info(warmup)
    ids = [item for fragment in fragments for item in fragment["ids"]]
    if not ids or ids != expected_ids or len(set(ids)) != len(ids):
        raise ValueError("fragment order or IDs differ")
    if len(warm["ids"]) != warmup_count or set(ids) & set(warm["ids"]):
        raise ValueError("warmup IDs overlap or count differs")
    return {
        "schema": "passage-persistent-inputs/v1",
        "target": target,
        "procedure": procedure,
        "warmup": warm,
        "fragments": fragments,
    }


def stream_inputs(output, manifest, paths, warmup):
    if len(paths) != len(manifest["fragments"]):
        raise ValueError("fragment count differs")
    with tarfile.open(fileobj=output, mode="w|") as archive:
        raw = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        member = tarfile.TarInfo("manifest.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
        entries = [("warmup.json", warmup, manifest["warmup"])]
        entries += [
            (f"inputs/{index:05d}.json", path, info)
            for index, (path, info) in enumerate(zip(paths, manifest["fragments"]))
        ]
        for name, path, info in entries:
            raw = _raw(path)
            if (
                len(raw) != info["bytes"]
                or hashlib.sha256(raw).hexdigest() != info["sha256"]
                or _check(raw) != info["ids"]
            ):
                raise ValueError(f"fragment changed before transfer: {name}")
            member = tarfile.TarInfo(name)
            member.size = len(raw)
            archive.addfile(member, io.BytesIO(raw))


def _write_all(stream, data):
    view = memoryview(data)
    while view:
        count = stream.write(view)
        if count is None or count <= 0:
            raise OSError("stream did not accept frame bytes")
        view = view[count:]


def write_frame(stream, kind, sequence, path, expected_sha256):
    """Send one bounded input file; the receiver validates before starting its timer."""
    if kind not in ("warmup", "fragment") or type(sequence) is not int or sequence < 0:
        raise ValueError("frame identity differs")
    raw = _raw(path)
    ids = _check(raw)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise ValueError("fragment changed before transfer")
    header = json.dumps(
        {"kind": kind, "sequence": sequence, "sha256": actual, "ids": ids}, sort_keys=True, separators=(",", ":")
    ).encode()
    if len(header) > 65536:
        raise ValueError("frame header exceeds bound")
    _write_all(stream, struct.pack("!4sIQ", b"PSF1", len(header), len(raw)))
    _write_all(stream, header)
    _write_all(stream, raw)
    stream.flush()


def _read_exact(stream, size):
    parts = []
    while size:
        part = stream.read(size)
        if not part:
            raise ValueError("truncated input frame")
        parts.append(part)
        size -= len(part)
    return b"".join(parts)


def read_frame(stream):
    """Receive a bounded verified fragment, without scoring or network timing."""
    magic, header_len, data_len = struct.unpack("!4sIQ", _read_exact(stream, 16))
    if magic != b"PSF1" or not 0 < header_len <= 65536 or not 0 < data_len <= MAX_FRAGMENT:
        raise ValueError("frame bounds or magic differ")
    header = json.loads(_read_exact(stream, header_len))
    if (
        not isinstance(header, dict)
        or set(header) != {"kind", "sequence", "sha256", "ids"}
        or header["kind"] not in ("warmup", "fragment")
        or type(header["sequence"]) is not int
        or header["sequence"] < 0
    ):
        raise ValueError("frame header differs")
    raw = _read_exact(stream, data_len)
    if hashlib.sha256(raw).hexdigest() != header["sha256"]:
        raise ValueError("frame payload digest differs")
    if _check(raw) != header["ids"]:
        raise ValueError("frame sample IDs differ")
    return header, raw


def _array_payload(raw):
    """Decode bounded, uncompressed primitive arrays without pickle or extra members."""
    import numpy as np

    names = {name + ".npy" for name in _BINARY_KEYS}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (
            len(entries) != len(names)
            or {item.filename for item in entries} != names
            or any(item.compress_type != zipfile.ZIP_STORED for item in entries)
            or sum(item.file_size for item in entries) > MAX_FRAGMENT
        ):
            raise ValueError("binary input-only fields or bounds differ")
    with np.load(io.BytesIO(raw), allow_pickle=False, max_header_size=10000) as data:
        arrays = {key: data[key] for key in _BINARY_KEYS}
    ids, terrain, obstacles = arrays["sample_ids"], arrays["terrain"], arrays["obstacles"]
    starts, goals = arrays["starts"], arrays["goals"]
    if (
        ids.ndim != 1
        or ids.dtype.kind != "U"
        or not 0 < len(ids) <= 256
        or any(not item for item in ids.tolist())
        or len(set(ids.tolist())) != len(ids)
        or terrain.dtype != np.float64
        or terrain.ndim != 3
        or terrain.shape[0] != len(ids)
        or not terrain.shape[1]
        or not terrain.shape[2]
        or not np.isfinite(terrain).all()
        or obstacles.dtype != np.bool_
        or obstacles.shape != terrain.shape
        or starts.dtype != np.int64
        or goals.dtype != np.int64
        or starts.shape != (len(ids), 2)
        or goals.shape != (len(ids), 2)
    ):
        raise ValueError("binary input-only dtype, shape or values differ")
    for points in (starts, goals):
        if (
            np.any(points < 0)
            or np.any(points[:, 0] >= terrain.shape[1])
            or np.any(points[:, 1] >= terrain.shape[2])
            or np.any(obstacles[np.arange(len(ids)), points[:, 0], points[:, 1]])
        ):
            raise ValueError("binary input-only endpoint invalid or blocked")
    return arrays


def write_array_frame(stream, kind, sequence, path, expected_sha256):
    if kind not in ("warmup", "fragment") or type(sequence) is not int or sequence < 0:
        raise ValueError("frame identity differs")
    raw = _raw(path)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("fragment changed before transfer")
    arrays = _array_payload(raw)
    header = json.dumps(
        {"kind": kind, "sequence": sequence, "sha256": expected_sha256, "ids": arrays["sample_ids"].tolist()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if len(header) > 65536:
        raise ValueError("frame header exceeds bound")
    _write_all(stream, struct.pack("!4sIQ", b"PSF2", len(header), len(raw)))
    _write_all(stream, header)
    _write_all(stream, raw)
    stream.flush()


def read_array_frame(stream):
    magic, header_len, data_len = struct.unpack("!4sIQ", _read_exact(stream, 16))
    if magic != b"PSF2" or not 0 < header_len <= 65536 or not 0 < data_len <= MAX_FRAGMENT:
        raise ValueError("frame bounds or magic differ")
    header = json.loads(_read_exact(stream, header_len))
    if (
        not isinstance(header, dict)
        or set(header) != {"kind", "sequence", "sha256", "ids"}
        or header["kind"] not in ("warmup", "fragment")
        or type(header["sequence"]) is not int
        or header["sequence"] < 0
    ):
        raise ValueError("frame header differs")
    raw = _read_exact(stream, data_len)
    if hashlib.sha256(raw).hexdigest() != header["sha256"]:
        raise ValueError("frame payload digest differs")
    arrays = _array_payload(raw)
    if arrays["sample_ids"].tolist() != header["ids"]:
        raise ValueError("frame sample IDs differ")
    return header, arrays


def as_raw_payload(arrays):
    """Convert IDs and coordinates to plain Python values; leave raw arrays unchanged."""
    return {
        **arrays,
        "sample_ids": arrays["sample_ids"].tolist(),
        "starts": arrays["starts"].tolist(),
        "goals": arrays["goals"].tolist(),
    }


def write_end(stream):
    """Mark the input stream complete without relying on EOF."""
    _write_all(stream, struct.pack("!4sIQ", b"PSE1", 0, 0))
    stream.flush()


def read_task_index(stream, expected_sha256, *, max_tasks=1024):
    """Read a bounded input-only JSONL index row by row, verifying its pin."""
    magic, header_len, size = struct.unpack("!4sIQ", _read_exact(stream, 16))
    if (
        magic != b"PSI1"
        or header_len != 0
        or not 0 < size <= 8 * 1024 * 1024
        or type(max_tasks) is not int
        or max_tasks < 1
        or not isinstance(expected_sha256, str)
        or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
    ):
        raise ValueError("task index bounds or identity differ")
    digest = hashlib.sha256()
    rows = []
    seen = set()
    remaining = size
    while remaining:
        line = stream.readline(min(remaining, 4097))
        if not line or len(line) > 4096 or not line.endswith(b"\n"):
            raise ValueError("task index line truncated or exceeds bound")
        remaining -= len(line)
        digest.update(line)

        def unique_object(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("task index duplicate key")
                result[key] = value
            return result

        row = json.loads(line, object_pairs_hook=unique_object)
        if (
            not isinstance(row, dict)
            or set(row) != {"sample_id", "raw_input_sha256"}
            or not isinstance(row["sample_id"], str)
            or not row["sample_id"]
            or row["sample_id"] in seen
            or not isinstance(row["raw_input_sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", row["raw_input_sha256"])
        ):
            raise ValueError("task index row differs")
        seen.add(row["sample_id"])
        rows.append(row)
        if len(rows) > max_tasks:
            raise ValueError("task index count exceeds authorization")
    if not rows or digest.hexdigest() != expected_sha256:
        raise ValueError("task index digest differs")
    return rows


def write_task_index(stream, path, expected_sha256, *, max_tasks=1024):
    """Write a verified JSONL index before warmup; no reference data permitted."""
    raw = _raw(path)
    if len(raw) > 8 * 1024 * 1024:
        raise ValueError("task index exceeds bound")
    framed = struct.pack("!4sIQ", b"PSI1", 0, len(raw))
    read_task_index(io.BytesIO(framed + raw), expected_sha256, max_tasks=max_tasks)
    _write_all(stream, framed)
    _write_all(stream, raw)
    stream.flush()


def requests_from_arrays(arrays, budget):
    """Materialize ordinary PlanRequests before the direct measurement starts."""
    from passage_eval.protocol.contracts import PlanRequest

    payload = as_raw_payload(arrays)
    return [
        PlanRequest(
            sample,
            payload["terrain"][i],
            payload["obstacles"][i],
            tuple(payload["starts"][i]),
            tuple(payload["goals"][i]),
            "elevation",
            62.4,
            budget=budget,
        )
        for i, sample in enumerate(payload["sample_ids"])
    ]


def read_warmup(stream, budget, expected_warmup):
    """Read the one warmup scene; the runner repeats it 16 times."""
    from passage_eval.campaign.direct.raw_identity import raw_digest

    header, arrays = read_array_frame(stream)
    requests = requests_from_arrays(arrays, budget)
    if (
        header["kind"] != "warmup"
        or header["sequence"] != 0
        or len(requests) != 1
        or len(expected_warmup) != 1
        or set(expected_warmup[0]) != {"sample_id", "raw_input_sha256"}
        or requests[0].sample_id != expected_warmup[0]["sample_id"]
        or raw_digest(requests[0]) != expected_warmup[0]["raw_input_sha256"]
    ):
        raise ValueError("warmup identity or framing differs")
    return requests


def blocks_from_stream(stream, ordered_tasks, budget):
    """Yield checked fragments once, then require an explicit end marker."""
    from passage_eval.campaign.direct.raw_identity import raw_digest

    if (
        not ordered_tasks
        or any(not isinstance(t, dict) or set(t) != {"sample_id", "raw_input_sha256"} for t in ordered_tasks)
        or len({t["sample_id"] for t in ordered_tasks}) != len(ordered_tasks)
    ):
        raise ValueError("ordered task index differs")
    index, sequence = 0, 0
    while index < len(ordered_tasks):
        header, arrays = read_array_frame(stream)
        requests = requests_from_arrays(arrays, budget)
        if header["kind"] != "fragment" or header["sequence"] != sequence or index + len(requests) > len(ordered_tasks):
            raise ValueError("fragment sequence differs")
        for request, expected in zip(requests, ordered_tasks[index : index + len(requests)], strict=True):
            if request.sample_id != expected["sample_id"] or raw_digest(request) != expected["raw_input_sha256"]:
                raise ValueError("fragment input identity differs")
        index += len(requests)
        sequence += 1
        yield requests
    if struct.unpack("!4sIQ", _read_exact(stream, 16)) != (b"PSE1", 0, 0):
        raise ValueError("input stream end marker differs")
