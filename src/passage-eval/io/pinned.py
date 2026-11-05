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

"""Pinned control files: strict JSON, bounded reads, hash pins and exclusive writes.

The campaign tools and the test admission verifier read small control files (plans, receipts, rosters, evidence)
whose bytes are pinned by SHA-256. Every reader here refuses a symbolic link, a non-regular file, a file larger
than the bound, a file that changes while it is read, duplicate JSON keys and the non-finite constants ``NaN`` and
``Infinity``. A hash proves byte correspondence only: it does not authenticate who wrote a file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

MAX_CONTROL_BYTES = 128 * 1024 * 1024
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
HEX40 = re.compile(r"[0-9a-f]{40}\Z")


class PinnedFileError(ValueError):
    """A pinned control file is missing, unsafe, malformed or differs from its pin."""


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and HEX64.fullmatch(value) is not None


def is_hex40(value: Any) -> bool:
    return isinstance(value, str) and HEX40.fullmatch(value) is not None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: Any) -> bytes:
    """Sorted keys, compact separators, ASCII, no NaN: the hashed byte form (as ``io.roster.canonical_json``)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=True).encode("ascii")


def canonical_sha256(value: Any) -> str:
    return sha256_bytes(canonical_json(value))


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PinnedFileError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str):
    raise PinnedFileError(f"non-finite JSON constant {name}")


def strict_json_loads(raw: bytes, *, name: str = "control file") -> Any:
    """Parse UTF-8 JSON, refusing duplicate keys and ``NaN``/``Infinity``."""
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
    except PinnedFileError as exc:
        raise PinnedFileError(f"{name}: {exc}") from exc
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PinnedFileError(f"{name} is not valid JSON: {exc}") from exc


def read_bounded(path: str | Path, *, max_bytes: int = MAX_CONTROL_BYTES) -> bytes:
    """Read one regular file without following a final symbolic link; refuse a file that changes meanwhile."""
    path = Path(path)
    try:
        fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError as exc:
        raise PinnedFileError(f"cannot open {path.name}: {exc.strerror or exc}") from exc
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PinnedFileError(f"{path.name} is not a regular file")
        if before.st_size > max_bytes:
            raise PinnedFileError(f"{path.name} exceeds {max_bytes} bytes")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            raw = stream.read(max_bytes + 1)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    fingerprint = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    if len(raw) > max_bytes or fingerprint != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
        raise PinnedFileError(f"{path.name} changed while it was read or exceeded the size bound")
    return raw


def read_pinned(path: str | Path, expected_sha256: str, *, max_bytes: int = MAX_CONTROL_BYTES) -> bytes:
    """The bytes of ``path`` if their SHA-256 equals ``expected_sha256`` (lowercase hex)."""
    if not is_hex64(expected_sha256):
        raise PinnedFileError("a lowercase 64-character SHA-256 pin is required")
    raw = read_bounded(path, max_bytes=max_bytes)
    if sha256_bytes(raw) != expected_sha256:
        raise PinnedFileError(f"SHA-256 mismatch for {Path(path).name}")
    return raw


def read_pinned_json(path: str | Path, expected_sha256: str, *, max_bytes: int = MAX_CONTROL_BYTES) -> Any:
    return strict_json_loads(read_pinned(path, expected_sha256, max_bytes=max_bytes), name=Path(path).name)


def read_json(path: str | Path, *, max_bytes: int = MAX_CONTROL_BYTES) -> Any:
    return strict_json_loads(read_bounded(path, max_bytes=max_bytes), name=Path(path).name)


def safe_relative(value: Any) -> PurePosixPath:
    """A relative POSIX path without empty, ``.`` or ``..`` components and without a backslash."""
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise PinnedFileError(f"a relative path is required, not {value!r}")
    parts = PurePosixPath(value).parts
    if not parts or any(part in ("", ".", "..") for part in parts):
        raise PinnedFileError(f"unsafe relative path {value!r}")
    return PurePosixPath(*parts)


def resolve_inside(root: str | Path, relative: Any) -> Path:
    """``root / relative`` as an existing regular file reached without any symbolic link component."""
    root = Path(root)
    current = root
    for part in safe_relative(relative).parts:
        current = current / part
        if current.is_symlink():
            raise PinnedFileError(f"symbolic links are not accepted in {relative!r}")
    if not current.is_file():
        raise PinnedFileError(f"{relative!r} is not a regular file under the declared root")
    if not current.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise PinnedFileError(f"{relative!r} escapes the declared root")
    return current


def file_ref(value: Any) -> tuple[str, str]:
    """``(path, sha256)`` of a ``{"path": ..., "sha256": ...}`` reference; anything else is an error."""
    if not isinstance(value, Mapping) or set(value) != {"path", "sha256"}:
        raise PinnedFileError("a file reference is exactly {path, sha256}")
    if not is_hex64(value["sha256"]):
        raise PinnedFileError("a file reference needs a lowercase 64-character sha256")
    return str(safe_relative(value["path"])), value["sha256"]


def write_json_exclusive(path: str | Path, payload: Any) -> str:
    """Write ``payload`` as indented JSON to a new file (never overwrite), fsync it and its directory; return its SHA-256."""
    path = Path(path)
    raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    partial = path.with_name(path.name + ".partial")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    if path.exists():
        partial.unlink()
        raise FileExistsError(path)
    os.replace(partial, path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    return sha256_bytes(raw)
