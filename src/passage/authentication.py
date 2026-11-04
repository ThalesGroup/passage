"""Read-only authentication of an entire local snapshot against pinned HF objects.

This authenticates source identity, not dataset semantics. An external output
directory holds a pinned manifest, resumable per-file observations, and a receipt.
Interrupted or incomplete work never produces ``content_verified=True``.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
from threading import Event, Lock
import time
from typing import Any

CHUNK_BYTES = 8 * 1024 * 1024
PROGRESS_SECONDS = 10.0
MAX_WORKERS = 8


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        stream.write(_json_bytes(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _field(value: Any, name: str) -> Any:
    return value.get(name) if isinstance(value, dict) else getattr(value, name, None)


def _manifest(root: Path, repo: str, revision: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    info = HfApi().repo_info(repo_id=repo, repo_type="dataset", revision=revision, files_metadata=True)
    if info.sha.lower() != revision:
        raise ValueError("Hugging Face returned a different revision than the pinned commit")
    entries = []
    seen = set()
    for sibling in info.siblings:
        name = sibling.rfilename
        parts = PurePosixPath(name)
        if not name or parts.is_absolute() or ".." in parts.parts or "\\" in name or str(parts) != name or name in seen:
            raise ValueError(f"Unsafe or duplicate manifest path: {name!r}")
        seen.add(name)
        lfs = sibling.lfs
        algorithm = "sha256" if lfs is not None else "git-blob-sha1"
        digest = _field(lfs, "sha256") if lfs is not None else sibling.blob_id
        size = _field(lfs, "size") if lfs is not None else sibling.size
        length = 64 if lfs is not None else 40
        if not isinstance(digest, str) or not re.fullmatch(rf"[0-9a-fA-F]{{{length}}}", digest):
            raise ValueError(f"Missing or invalid pinned object hash: {name}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"Missing or invalid pinned object size: {name}")
        entries.append({"path": name, "size": size, "algorithm": algorithm, "expected_hash": digest.lower()})
    if not entries:
        raise ValueError("Pinned manifest contains no files")
    return {
        "schema_version": 1,
        "root": str(root),
        "repo": repo,
        "revision": revision,
        "repo_type": "dataset",
        "files": sorted(entries, key=lambda item: item["path"]),
    }


def _file_stat(path: Path, root: Path) -> dict[str, int]:
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Source file resolves outside snapshot root: {path}")
    observed = path.stat()
    if not stat.S_ISREG(observed.st_mode):
        raise ValueError(f"Source is not a regular file: {path}")
    return {
        "device": observed.st_dev,
        "inode": observed.st_ino,
        "size": observed.st_size,
        "mtime_ns": observed.st_mtime_ns,
        "ctime_ns": observed.st_ctime_ns,
    }


class _ReadCounter:
    def __init__(self) -> None:
        self.lock = Lock()
        self.value = 0

    def add(self, count: int) -> None:
        with self.lock:
            self.value += count

    def read(self) -> int:
        with self.lock:
            return self.value


def _digest_file(path: Path, algorithm: str, size: int, counter: _ReadCounter, stop: Event) -> str:
    digest = hashlib.sha256() if algorithm == "sha256" else hashlib.sha1()
    if algorithm == "git-blob-sha1":
        digest.update(f"blob {size}\0".encode("ascii"))
    with path.open("rb") as stream:
        while not stop.is_set():
            block = stream.read(CHUNK_BYTES)
            if not block:
                return digest.hexdigest()
            digest.update(block)
            counter.add(len(block))
    raise InterruptedError("Authentication interrupted")


def _authenticate_file(root: Path, entry: dict[str, Any], counter: _ReadCounter, stop: Event) -> dict[str, Any]:
    path = root / entry["path"]
    result = {"path": entry["path"], "algorithm": entry["algorithm"], "expected_hash": entry["expected_hash"]}
    try:
        before = _file_stat(path, root)
        result["stat"] = before
        if before["size"] != entry["size"]:
            return {**result, "status": "mismatch", "reason": "size", "expected_size": entry["size"]}
        observed = _digest_file(path, entry["algorithm"], entry["size"], counter, stop)
        after = _file_stat(path, root)
        result.update(observed_hash=observed, stat_after=after)
        if before != after:
            return {**result, "status": "changed", "reason": "stat_changed_during_hash"}
        return {**result, "status": "verified" if observed == entry["expected_hash"] else "mismatch"}
    except FileNotFoundError as error:
        return {**result, "status": "missing", "reason": str(error)}
    except InterruptedError:
        return {**result, "status": "interrupted"}
    except (OSError, ValueError) as error:
        return {**result, "status": "error", "reason": str(error)}


def _read_progress(path: Path, manifest_id: str) -> dict[str, dict[str, Any]]:
    records = {}
    if not path.exists():
        return records
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.endswith(b"\n"):
                # A crash can leave a torn final append. Remove only that tail.
                stream.truncate(offset)
                break
            try:
                row = json.loads(line)
            except (ValueError, UnicodeDecodeError) as error:
                raise ValueError("Corrupt authentication progress record") from error
            if row.get("manifest_sha256") != manifest_id:
                raise ValueError("Progress records belong to a different manifest")
            records[row["path"]] = row
    return records


def authenticate_snapshot(
    root: str | Path, repo: str, revision: str, output: str | Path, workers: int = 4
) -> dict[str, Any]:
    """Authenticate all files listed at an immutable dataset commit, without downloads.

    ``output`` must be outside ``root``. Workers are bounded to 1–8, with at most
    one 8 MiB block buffered per worker. Matching successful records can resume
    without rereading unchanged files; device/inode/size/mtime/ctime must agree.
    Newly hashed matching files are reread before completion to detect content
    changes hidden by timestamp granularity. The source must remain quiescent;
    these observations do not establish an atomic snapshot against writers.
    Per-file progress is flushed on completion, summaries every ten seconds.
    Missing, changed, mismatching, or unreadable files return a partial receipt.
    Network/manifest errors raise; KeyboardInterrupt writes a partial receipt
    and then propagates. One writer may own the output directory at a time.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be an integer between 1 and {MAX_WORKERS}")
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be an immutable 40-hex commit SHA")
    if not isinstance(repo, str) or not repo.strip():
        raise ValueError("repo must be a nonempty dataset repository ID")
    revision = revision.lower()
    root, output = Path(root).resolve(), Path(output).resolve()
    if output.is_relative_to(root):
        raise ValueError("Authentication output must be outside the source root")
    if not root.is_dir():
        raise FileNotFoundError(f"Snapshot root is not a directory: {root}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".authentication.lock").open("a") as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Authentication output already has an active writer") from error
        return _run(root, repo, revision, output, workers)


def _run(root: Path, repo: str, revision: str, output: Path, workers: int) -> dict[str, Any]:
    manifest = _manifest(root, repo, revision)
    manifest_id = hashlib.sha256(_json_bytes(manifest)).hexdigest()
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Existing authentication manifest differs; choose a new output directory")
    else:
        _atomic_json(manifest_path, manifest)
    prior = _read_progress(output / "progress.jsonl", manifest_id)
    results: dict[str, dict[str, Any]] = {}
    pending = []
    resumed = 0
    for entry in manifest["files"]:
        record = prior.get(entry["path"])
        try:
            valid = (
                record is not None
                and record.get("status") == "verified"
                and record.get("content_rechecked") is True
                and record.get("observed_hash") == entry["expected_hash"]
                and record.get("algorithm") == entry["algorithm"]
                and record.get("stat") == _file_stat(root / entry["path"], root)
            )
        except (OSError, ValueError):
            valid = False
        if valid:
            results[entry["path"]] = record
            resumed += 1
        else:
            pending.append(entry)
    counter, stop = _ReadCounter(), Event()
    started = datetime.now(timezone.utc).isoformat()
    start_clock = time.monotonic()
    required_bytes = sum(entry["size"] for entry in manifest["files"])

    def receipt(final: bool = False) -> dict[str, Any]:
        counts = Counter(record["status"] for record in results.values())
        complete = final and counts["verified"] == len(manifest["files"])
        return {
            "schema_version": 1,
            "status": "complete" if complete else "partial",
            "content_verified": complete,
            "root": str(root),
            "repo": repo,
            "revision": revision,
            "manifest_sha256": manifest_id,
            "required_files": len(manifest["files"]),
            "required_bytes": required_bytes,
            "verified_files": counts["verified"],
            "verified_bytes": sum(
                record["stat"]["size"] for record in results.values() if record["status"] == "verified"
            ),
            "mismatched_files": counts["mismatch"],
            "missing_files": counts["missing"],
            "changed_files": counts["changed"],
            "error_files": counts["error"],
            "interrupted_files": counts["interrupted"],
            "pending_files": len(manifest["files"]) - len(results),
            "resumed_files": resumed,
            "bytes_read_this_run": counter.read(),
            "workers": workers,
            "started_at": started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(time.monotonic() - start_clock, 3),
            "failures": [record for record in results.values() if record["status"] != "verified"],
            "scope": "all pinned repository files; content identity only",
            "resume_condition": "unchanged device/inode/size/mtime_ns/ctime_ns; no source writer",
        }

    with (output / "progress.jsonl").open("ab") as journal:

        def record_result(result: dict[str, Any]) -> None:
            result = {**result, "manifest_sha256": manifest_id}
            results[result["path"]] = result
            journal.write(_json_bytes(result))
            journal.flush()

        def report(final: bool = False) -> dict[str, Any]:
            value = receipt(final)
            os.fsync(journal.fileno())
            _atomic_json(output / "receipt.json", value)
            print(
                json.dumps(
                    {
                        key: value[key]
                        for key in (
                            "status",
                            "required_files",
                            "verified_files",
                            "pending_files",
                            "bytes_read_this_run",
                            "elapsed_seconds",
                        )
                    }
                ),
                flush=True,
            )
            return value

        report()
        finished = False
        try:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="source-auth") as pool:
                entries = iter(pending)
                futures = {}

                def submit_next() -> None:
                    entry = next(entries, None)
                    if entry is not None:
                        futures[pool.submit(_authenticate_file, root, entry, counter, stop)] = entry

                for _ in range(workers):
                    submit_next()
                last_report = time.monotonic()
                try:
                    while futures:
                        done, _ = wait(futures, timeout=1.0, return_when=FIRST_COMPLETED)
                        for future in done:
                            futures.pop(future)
                            record_result(future.result())
                            submit_next()
                        if time.monotonic() - last_report >= PROGRESS_SECONDS:
                            report()
                            last_report = time.monotonic()
                except BaseException:
                    stop.set()
                    for future in futures:
                        future.cancel()
                    raise
            # Stat identity alone misses same-size writes within a timestamp tick.
            # Only completed content rechecks may be reused by a later run.
            for record in list(results.values()):
                if record["status"] != "verified":
                    continue
                reason = "stat_changed_before_receipt"
                try:
                    path = root / record["path"]
                    unchanged = record["stat"] == _file_stat(path, root)
                    if unchanged and not record.get("content_rechecked", False):
                        observed = _digest_file(path, record["algorithm"], record["stat"]["size"], counter, stop)
                        unchanged = observed == record["expected_hash"]
                        if not unchanged:
                            reason = "content_changed_before_receipt"
                        else:
                            unchanged = record["stat"] == _file_stat(path, root)
                        if unchanged:
                            record_result({**record, "content_rechecked": True})
                except InterruptedError:
                    record_result({**record, "status": "interrupted"})
                    continue
                except (OSError, ValueError) as error:
                    record_result({**record, "status": "error", "reason": str(error)})
                    continue
                if not unchanged:
                    record_result({**record, "status": "changed", "reason": reason})
            # A later content recheck may invalidate an earlier observation.
            for record in list(results.values()):
                if record["status"] != "verified":
                    continue
                try:
                    unchanged = record["stat"] == _file_stat(root / record["path"], root)
                except (OSError, ValueError):
                    unchanged = False
                if not unchanged:
                    record_result({**record, "status": "changed", "reason": "stat_changed_before_receipt"})
            finished = True
        finally:
            final_receipt = report(final=finished)
        return final_receipt
