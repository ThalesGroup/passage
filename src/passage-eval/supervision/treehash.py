# MIT License
#
# Copyright (c) 2026 THALES

"""Deterministic tree hashes, stat signatures and read-only staging of declared files.

Every byte that is bound into a candidate sandbox must be covered by an identity. This
module provides the two primitives for that:

``hash_tree``
    A deterministic digest of a file or directory tree, ``passage-tree-sha256/v1``. The
    digest covers, for each entry in sorted relative-path order: the relative path, the
    kind, the executable bit (not the full mode: staged copies are made read-only, and
    group/other bits are not content) and the SHA-256 of every regular file; symbolic links
    are recorded by target and never followed; unreadable files are recorded, not skipped.
    ``__pycache__`` directories can be excluded (candidate source trees): a stale or forged
    bytecode file must never stand in for the source that was hashed.

``StageArea``
    A private, content-addressed, read-only copy of declared candidate files and
    directories. Hashing happens *while copying*, over the very bytes that are written, so
    nothing is hashed that is not then used. The copy is the only thing the sandbox binds,
    at the original path. Symbolic links and special files are refused.

For large runtime trees that cannot be copied (a virtual environment, ``/usr``) the digest
is computed in place. A stat signature (path, kind, mode, size, mtime, ctime and link
target of every entry) allows an optional on-disk cache of the digest and a cheap
change check; ``ctime`` cannot be set from user space, so a content change that keeps the
size and the modification time still changes the signature. The cache is an optimisation
only: every record states whether its digest was computed or taken from the cache.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import errno
import hashlib
import json
import os
import shutil
import stat
import tempfile
import weakref
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

TREE_SCHEMA = "passage-tree-sha256/v1"
CACHE_SCHEMA = "passage-tree-hash-cache/v1"
#: Directory names that are not part of a candidate source tree's identity.
CANDIDATE_EXCLUDES = ("__pycache__",)
_CHUNK = 1 << 20
_MAX_LISTED = 64  # symlinks / unreadable entries listed in a record (the counts are exact)


class TreeError(ValueError):
    """A tree cannot be hashed or staged under the requested policy."""


class _Entry:
    """One tree entry (a slotted class: a /usr walk creates several hundred thousand of them)."""

    __slots__ = ("rel", "kind", "mode", "size", "mtime_ns", "ctime_ns", "target", "abspath")

    def __init__(
        self, rel: str, kind: str, mode: int, size: int, mtime_ns: int, ctime_ns: int, target: str | None, abspath: str
    ):
        self.rel, self.kind, self.mode, self.size = rel, kind, mode, size
        self.mtime_ns, self.ctime_ns, self.target, self.abspath = mtime_ns, ctime_ns, target, abspath


@dataclass
class TreeRecord:
    """The identity of one bound tree, as written to the manifest."""

    role: str  # candidate_staged | candidate_in_place | environment | decoder_entry
    kind: str  # file | dir
    path: str  # original absolute path; the sandbox binds the bytes hashed here at this path
    sha256: str
    files: int
    directories: int
    bytes: int
    symlinks: int = 0
    unreadable: int = 0
    excluded_dirs: tuple[str, ...] = ()
    hash_source: str = "computed"  # computed | stat_signature_cache
    signature: str | None = None
    external_symlinks: list[dict[str, str]] = field(default_factory=list)
    unreadable_listed: list[dict[str, str]] = field(default_factory=list)
    symlinks_listed: list[dict[str, str]] = field(default_factory=list)
    flagged: int = 0  # entries whose name matched the caller's ``flag`` rule (e.g. onnxruntime*)
    flagged_listed: list[str] = field(default_factory=list)
    staged_path: str | None = None  # private; not written to the manifest
    staged_signature: str | None = None  # stat signature of the staged copy once sealed (private)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": TREE_SCHEMA,
            "role": self.role,
            "kind": self.kind,
            "path": self.path,
            "sha256": self.sha256,
            "files": self.files,
            "directories": self.directories,
            "bytes": self.bytes,
            "symlinks": self.symlinks,
            "unreadable_files": self.unreadable,
            "excluded_dirs": list(self.excluded_dirs),
            "hash_source": self.hash_source,
            "stat_signature": self.signature,
            "staged": self.staged_path is not None,
            "symlinks_listed": self.symlinks_listed,
            "external_symlinks": self.external_symlinks,
            "unreadable_listed": self.unreadable_listed,
            "flagged_entries": self.flagged,
            "flagged_listed": self.flagged_listed,
        }


def is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# walking
# ---------------------------------------------------------------------------


def _walk(root: Path, exclude_dirs: Sequence[str]) -> list[_Entry]:
    """Every entry below ``root`` (``root`` itself excluded), sorted by relative path."""
    entries: list[_Entry] = []
    stack = [(str(root), "")]
    while stack:
        directory, prefix = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                children = list(iterator)
        except OSError as error:
            raise TreeError(f"cannot list {directory}: {error.strerror or error}") from None
        for child in children:
            rel = f"{prefix}{child.name}"
            try:
                info = child.stat(follow_symlinks=False)
            except OSError as error:
                raise TreeError(f"cannot stat {child.path}: {error.strerror or error}") from None
            mode = info.st_mode
            if stat.S_ISDIR(mode):
                if child.name in exclude_dirs:
                    continue
                entries.append(_Entry(rel, "D", mode, 0, info.st_mtime_ns, info.st_ctime_ns, None, child.path))
                stack.append((child.path, rel + "/"))
            elif stat.S_ISREG(mode):
                entries.append(
                    _Entry(rel, "F", mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, None, child.path)
                )
            elif stat.S_ISLNK(mode):
                entries.append(
                    _Entry(rel, "L", mode, 0, info.st_mtime_ns, info.st_ctime_ns, os.readlink(child.path), child.path)
                )
            else:
                entries.append(_Entry(rel, "S", mode, 0, info.st_mtime_ns, info.st_ctime_ns, None, child.path))
    entries.sort(key=_rel_key)  # code-point order equals UTF-8 byte order for every valid name
    return entries


def _rel_key(entry: "_Entry") -> str:
    return entry.rel


def _sha256_path(path: str) -> str:
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise TreeError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, _CHUNK)
            if not block:
                return digest.hexdigest()
            digest.update(block)
    finally:
        os.close(descriptor)


def _hash_batch(paths: Sequence[str]) -> list[tuple[str, str]]:
    """``(path, sha256 | 'U:<errno name>')`` for each path."""
    out = []
    for path in paths:
        try:
            out.append((path, _sha256_path(path)))
        except OSError as error:
            out.append((path, "U:" + errno.errorcode.get(error.errno or 0, "EIO")))
    return out


def _hash_files(paths: Sequence[str], workers: int | None) -> dict[str, str]:
    """SHA-256 of many files, in parallel batches (hashlib releases the GIL for large blocks)."""
    if not paths:
        return {}
    count = workers or min(32, (os.cpu_count() or 4) * 2)
    batches = [paths[i : i + 256] for i in range(0, len(paths), 256)]
    result: dict[str, str] = {}
    if count <= 1 or len(batches) == 1:
        for batch in batches:
            result.update(_hash_batch(batch))
        return result
    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        for rows in pool.map(_hash_batch, batches):
            result.update(rows)
    return result


def _exec_flag(mode: int) -> str:
    return "x" if mode & 0o111 else "-"


def stat_signature(entries: Iterable[_Entry]) -> str:
    digest = hashlib.sha256(b"passage-tree-signature/v1\n")
    lines = []
    for entry in entries:
        if entry.kind == "D":  # a directory's times change with its listing, which the entries cover
            lines.append(f"{entry.rel}\0D\0{entry.mode:o}\n")
        else:
            lines.append(
                f"{entry.rel}\0{entry.kind}\0{entry.mode:o}\0{entry.size}\0{entry.mtime_ns}\0"
                f"{entry.ctime_ns}\0{entry.target or ''}\n"
            )
        if len(lines) >= 4096:
            digest.update("".join(lines).encode("utf-8", "surrogateescape"))
            lines.clear()
    digest.update("".join(lines).encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


def _digest(rows: Iterable[tuple[_Entry, str | None]]) -> str:
    digest = hashlib.sha256(TREE_SCHEMA.encode() + b"\n")
    for entry, content in rows:
        if entry.kind == "D":
            line = f"D\0{entry.rel}\n"
        elif entry.kind == "F":
            line = f"F\0{entry.rel}\0{_exec_flag(entry.mode)}\0{content}\n"
        elif entry.kind == "L":
            line = f"L\0{entry.rel}\0{entry.target}\n"
        elif entry.kind == "U":
            line = f"U\0{entry.rel}\0{content}\n"
        else:
            line = f"S\0{entry.rel}\0{stat.S_IFMT(entry.mode):o}\n"
        digest.update(line.encode("utf-8", "surrogateescape"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# cache
# ---------------------------------------------------------------------------


class TreeHashCache:
    """Optional on-disk cache of tree digests keyed by path and stat signature."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def _file(self, path: str) -> Path:
        return self.directory / (hashlib.sha256(path.encode("utf-8", "surrogateescape")).hexdigest() + ".json")

    def load(self, path: str, signature: str) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._file(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != CACHE_SCHEMA
            or payload.get("path") != path
            or payload.get("signature") != signature
            or not is_hex64(payload.get("sha256"))
        ):
            return None
        return payload

    def store(self, path: str, signature: str, record: TreeRecord) -> None:
        payload = {
            "schema": CACHE_SCHEMA,
            "path": path,
            "signature": signature,
            "sha256": record.sha256,
            "files": record.files,
            "directories": record.directories,
            "bytes": record.bytes,
            "symlinks": record.symlinks,
            "unreadable": record.unreadable,
            "external_symlinks": record.external_symlinks,
            "unreadable_listed": record.unreadable_listed,
            "symlinks_listed": record.symlinks_listed,
        }
        with contextlib.suppress(OSError):
            self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
            temporary = self.directory / f".tmp-{os.getpid()}-{hashlib.sha256(path.encode()).hexdigest()[:12]}"
            temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            os.replace(temporary, self._file(path))


# ---------------------------------------------------------------------------
# hashing a tree in place
# ---------------------------------------------------------------------------


def flagged_entries(entries: Iterable[_Entry], flag: Callable[[str], bool] | None) -> list[str]:
    """Relative paths of the entries whose own name matches ``flag`` (a rule on the lowercased base name)."""
    if flag is None:
        return []
    return [entry.rel for entry in entries if flag(entry.rel.rsplit("/", 1)[-1].lower())]


def _record_from_entries(
    root: Path,
    role: str,
    kind: str,
    rows: list[tuple[_Entry, str | None]],
    entries: list[_Entry],
    exclude_dirs: Sequence[str],
    source: str,
    signature: str | None,
    flag: Callable[[str], bool] | None = None,
) -> TreeRecord:
    symlinks = [entry for entry in entries if entry.kind == "L"]
    unreadable = [entry for entry in entries if entry.kind == "U"]
    flagged = flagged_entries(entries, flag)
    external = []
    for entry in symlinks:
        target = Path(os.path.normpath(os.path.join(os.path.dirname(entry.abspath), entry.target or "")))
        if not (target == root or root in target.parents):
            external.append({"path": entry.rel, "target": entry.target or ""})
    return TreeRecord(
        role=role,
        kind=kind,
        path=str(root),
        sha256=_digest(rows),
        files=sum(1 for e in entries if e.kind in "FU"),
        directories=sum(1 for e in entries if e.kind == "D"),
        bytes=sum(e.size for e in entries if e.kind == "F"),
        symlinks=len(symlinks),
        unreadable=len(unreadable),
        excluded_dirs=tuple(exclude_dirs),
        hash_source=source,
        signature=signature,
        external_symlinks=external[:_MAX_LISTED],
        unreadable_listed=[{"path": e.rel} for e in unreadable[:_MAX_LISTED]],
        symlinks_listed=[{"path": e.rel, "target": e.target or ""} for e in symlinks[:_MAX_LISTED]],
        flagged=len(flagged),
        flagged_listed=flagged[:_MAX_LISTED],
    )


def hash_tree(
    path: str | Path,
    *,
    role: str = "environment",
    symlinks: str = "record",
    strict: bool = False,
    exclude_dirs: Sequence[str] = (),
    cache: TreeHashCache | None = None,
    workers: int | None = None,
    flag: Callable[[str], bool] | None = None,
) -> TreeRecord:
    """Digest of a file or directory tree read in place.

    ``symlinks`` is ``'record'`` (default) or ``'refuse'``. With ``strict`` an unreadable file is an
    error instead of a recorded entry. With ``cache`` the digest is reused when the stat signature of
    the whole tree is unchanged since it was stored; the record says which happened. ``flag`` is a rule on
    the lowercased base name of every entry; matches are listed in the record (supervisor-side scan of the
    very tree that was walked for the digest, cached or not).
    """
    if symlinks not in {"record", "refuse"}:
        raise ValueError("symlinks must be 'record' or 'refuse'")
    root = Path(os.path.abspath(path))
    info = os.lstat(root)
    if stat.S_ISLNK(info.st_mode):
        raise TreeError(f"tree root is a symbolic link: {root}")
    if stat.S_ISREG(info.st_mode):
        entry = _Entry(root.name, "F", info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, None, str(root))
        entries = [entry]
        kind = "file"
    elif stat.S_ISDIR(info.st_mode):
        entries = _walk(root, exclude_dirs)
        kind = "dir"
    else:
        raise TreeError(f"not a file or directory: {root}")
    if symlinks == "refuse" and any(e.kind == "L" for e in entries):
        raise TreeError(f"symbolic link refused in {root}: {next(e.rel for e in entries if e.kind == 'L')}")
    if strict and any(e.kind == "S" for e in entries):
        raise TreeError(f"special file refused in {root}: {next(e.rel for e in entries if e.kind == 'S')}")
    signature = stat_signature(entries)
    if cache is not None:
        hit = cache.load(str(root), signature)
        if hit is not None:
            external = hit.get("external_symlinks", [])
            hit_flagged = flagged_entries(entries, flag)
            return TreeRecord(
                role=role,
                kind=kind,
                path=str(root),
                sha256=hit["sha256"],
                files=hit["files"],
                directories=hit["directories"],
                bytes=hit["bytes"],
                symlinks=hit["symlinks"],
                unreadable=hit["unreadable"],
                excluded_dirs=tuple(exclude_dirs),
                hash_source="stat_signature_cache",
                signature=signature,
                external_symlinks=external,
                unreadable_listed=hit.get("unreadable_listed", []),
                symlinks_listed=hit.get("symlinks_listed", []),
                flagged=len(hit_flagged),
                flagged_listed=hit_flagged[:_MAX_LISTED],
            )
    hashed = _hash_files([e.abspath for e in entries if e.kind == "F"], workers)
    rows: list[tuple[_Entry, str | None]] = []
    final: list[_Entry] = []
    for entry in entries:
        if entry.kind == "F":
            content = hashed[entry.abspath]
            if content.startswith("U:"):
                if strict:
                    raise TreeError(f"unreadable file in {root}: {entry.rel} ({content[2:]})")
                entry = _Entry(
                    entry.rel, "U", entry.mode, entry.size, entry.mtime_ns, entry.ctime_ns, None, entry.abspath
                )
            rows.append((entry, content))
        else:
            rows.append((entry, None))
        final.append(entry)
    record = _record_from_entries(root, role, kind, rows, final, exclude_dirs, "computed", signature, flag)
    if cache is not None:
        cache.store(str(root), signature, record)
    return record


def current_signature(path: str | Path, *, exclude_dirs: Sequence[str] = ()) -> str:
    """The stat signature of a tree now (cheap: no file is read)."""
    root = Path(os.path.abspath(path))
    info = os.lstat(root)
    if stat.S_ISREG(info.st_mode):
        return stat_signature(
            [_Entry(root.name, "F", info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, None, str(root))]
        )
    return stat_signature(_walk(root, exclude_dirs))


# ---------------------------------------------------------------------------
# staging
# ---------------------------------------------------------------------------


def _make_removable(path: Path) -> None:
    for current, directories, files in os.walk(path):
        for name in directories:
            with contextlib.suppress(OSError):
                os.chmod(os.path.join(current, name), 0o700)
    with contextlib.suppress(OSError):
        os.chmod(path, 0o700)


def remove_tree(path: str | Path) -> None:
    """``rmtree`` for read-only staged trees."""
    path = Path(path)
    if path.exists():
        _make_removable(path)
        shutil.rmtree(path, ignore_errors=True)


def _seal(path: Path) -> None:
    """Make a staged tree read-only: directories r-x, files already r-- or r-x."""
    if path.is_dir():
        for current, directories, _ in os.walk(path, topdown=False):
            os.chmod(current, 0o500)


class StageArea:
    """A private directory that holds content-addressed, read-only copies of declared inputs.

    Layout: ``<root>/<tree sha256>/<name>``. The same content is stored once. ``close()`` removes it.
    """

    def __init__(self, base: str | Path | None = None, flag: Callable[[str], bool] | None = None):
        self.root = Path(tempfile.mkdtemp(prefix="passage-stage-", dir=None if base is None else str(base)))
        os.chmod(self.root, 0o700)
        self._counter = 0
        self._flag = flag
        self.trees: dict[tuple[str, str], TreeRecord] = {}
        # also removed at interpreter exit or when the area is garbage collected, so a crash leaves no copy behind
        self._finalizer = weakref.finalize(self, remove_tree, self.root)

    def stage(
        self, source: str | Path, *, role: str = "candidate_staged", exclude_dirs: Sequence[str] = CANDIDATE_EXCLUDES
    ) -> TreeRecord:
        """Copy ``source`` (a regular file or a directory without symlinks) and return the identity of the copy."""
        original = Path(os.path.abspath(source))
        info = os.lstat(original)
        if stat.S_ISLNK(info.st_mode):
            raise TreeError(f"declared path is a symbolic link (declare the resolved path): {original}")
        self._counter += 1
        incoming = self.root / f".incoming-{self._counter}"
        incoming.mkdir(mode=0o700)
        try:
            if stat.S_ISREG(info.st_mode):
                kind, name = "file", original.name
                rows, entries = self._copy_file(original, incoming / name, original.name)
            elif stat.S_ISDIR(info.st_mode):
                kind, name = "dir", original.name or "root"
                rows, entries = self._copy_tree(original, incoming / name, exclude_dirs)
            else:
                raise TreeError(f"not a regular file or directory: {original}")
            sha = _digest(rows)
            destination = self.root / sha
            if destination.exists():
                remove_tree(incoming)
                staged_name = next(iter(os.listdir(destination)))
                if self._staged_digest(destination / staged_name) != sha:  # content-addressed: the name is a claim
                    raise TreeError(f"existing staged copy {sha} no longer holds the bytes it is named after")
            else:
                destination.mkdir(mode=0o700)
                os.rename(incoming / name, destination / name)  # a directory is renamed while it is still writable
                remove_tree(incoming)
                _seal(destination / name)
                os.chmod(destination, 0o500)
                staged_name = name
        except BaseException:
            remove_tree(incoming)
            raise
        record = _record_from_entries(original, role, kind, rows, entries, exclude_dirs, "computed", None, self._flag)
        record.staged_path = str(self.root / sha / staged_name)
        record.staged_signature = current_signature(record.staged_path)  # sealed: any later touch moves ctime
        self.trees[(str(original), role)] = record
        return record

    @staticmethod
    def _staged_digest(path: Path) -> str:
        """Digest of a staged copy as it is now, with nothing excluded (a staged copy holds no ``__pycache__``)."""
        return hash_tree(path, role="candidate_staged", symlinks="refuse", strict=True).sha256

    def _copy_file(self, source: Path, destination: Path, rel: str):
        info = os.lstat(source)
        digest = self._copy_bytes(source, destination, executable=bool(info.st_mode & 0o111))
        entry = _Entry(rel, "F", info.st_mode, info.st_size, info.st_mtime_ns, info.st_ctime_ns, None, str(destination))
        return [(entry, digest)], [entry]

    def _copy_tree(self, source: Path, destination: Path, exclude_dirs: Sequence[str]):
        entries = _walk(source, exclude_dirs)
        for entry in entries:
            if entry.kind == "L":
                raise TreeError(f"symbolic link refused in declared tree {source}: {entry.rel}")
            if entry.kind == "S":
                raise TreeError(f"special file refused in declared tree {source}: {entry.rel}")
        destination.mkdir(mode=0o700)
        rows, staged_entries = [], []
        directories = []
        for entry in entries:
            target = destination / entry.rel
            if entry.kind == "D":
                target.mkdir(mode=0o700)
                directories.append(target)
                rows.append((entry, None))
                staged_entries.append(entry)
            else:
                digest = self._copy_bytes(Path(entry.abspath), target, executable=bool(entry.mode & 0o111))
                rows.append((entry, digest))
                staged_entries.append(entry)
        return rows, staged_entries

    @staticmethod
    def _copy_bytes(source: Path, destination: Path, *, executable: bool) -> str:
        """Copy and hash the same bytes; the source is opened without following links."""
        descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            if not stat.S_ISREG(os.fstat(descriptor).st_mode):
                raise TreeError(f"not a regular file: {source}")
            out = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                digest = hashlib.sha256()
                while True:
                    block = os.read(descriptor, _CHUNK)
                    if not block:
                        break
                    digest.update(block)
                    view = memoryview(block)
                    while view:
                        view = view[os.write(out, view) :]
            finally:
                os.close(out)
        finally:
            os.close(descriptor)
        os.chmod(destination, 0o500 if executable else 0o400)
        return digest.hexdigest()

    def verify(self, record: TreeRecord) -> None:
        """Re-check the staged copy by content and by stat signature; any difference is an error.

        The content check alone passes a copy that was modified and then restored. The signature
        (path, kind, mode, size, modification time, change time, link target of every entry) does not: the
        change time cannot be set from user space, so writing a file, a mode change or a rename leaves a trace
        even when the bytes, the mode and the modification time are put back. Nothing is excluded here: the
        staged copy never holds a ``__pycache__``, so one that appears is a difference.
        """
        if record.staged_path is None:
            raise TreeError("record is not a staged tree")
        again = hash_tree(record.staged_path, role=record.role, symlinks="refuse", strict=True)
        if again.sha256 != record.sha256:
            raise TreeError(f"staged copy of {record.path} changed after it was hashed")
        if record.staged_signature is not None and current_signature(record.staged_path) != record.staged_signature:
            raise TreeError(f"staged copy of {record.path} was touched after it was sealed (stat signature differs)")

    def close(self) -> None:
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ---------------------------------------------------------------------------
# files an import could pick that a source hash does not see
# ---------------------------------------------------------------------------


def unmatched_non_source_files(
    root: str | Path,
    reference: str | Path,
    *,
    source_suffixes: Sequence[str] = (".py",),
    exclude_dirs: Sequence[str] = ("__pycache__",),
) -> dict[str, Any]:
    """Files below ``root`` that are not source files and that ``reference`` does not hold identically.

    The evaluator source fingerprint (``cli._source_hash``) covers ``*.py`` only. An import also finds extension
    modules (``.so``, tried before ``.py`` in the same directory), sourceless bytecode (``.pyc``), ``.pth`` files and
    packages that are not there on the supervisor's side. Every such file in the worker's copy must exist at the same
    relative path in the supervisor's own copy with the same SHA-256; symbolic links and special files never match.
    Only the files found below ``root`` are read, so a supervisor-side tree with large data directories costs nothing
    when the worker's copy is a plain installed package.
    """
    root, reference = Path(os.path.abspath(root)), Path(os.path.abspath(reference))
    unmatched: list[str] = []
    checked = 0
    for current, directories, files in os.walk(root):
        base = Path(current)
        directories[:] = sorted(d for d in directories if d not in exclude_dirs)
        for name in sorted([*files, *[d for d in directories if (base / d).is_symlink()]]):
            path = base / name
            rel = path.relative_to(root).as_posix()
            if name.endswith(tuple(source_suffixes)):
                continue
            checked += 1
            info = os.lstat(path)
            counterpart = reference / rel
            try:
                matches = (
                    stat.S_ISREG(info.st_mode)
                    and os.path.isfile(counterpart)
                    and not os.path.islink(counterpart)
                    and _sha256_path(str(path)) == _sha256_path(str(counterpart))
                )
            except OSError:
                matches = False
            if not matches:
                unmatched.append(rel)
    return {"checked": checked, "unmatched_count": len(unmatched), "unmatched": unmatched[:_MAX_LISTED]}
