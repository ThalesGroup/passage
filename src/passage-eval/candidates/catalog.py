"""Atomic operator-local registration; identities cannot silently change bytes."""

import fcntl
import json
import os
from pathlib import Path
import tempfile

from passage_eval.protocol.candidate_package import CandidatePackage, read_json


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def register_candidate(package: CandidatePackage, catalog: str | Path) -> dict:
    package.verify()
    catalog = Path(catalog)
    catalog.parent.mkdir(parents=True, exist_ok=True)
    with catalog.with_suffix(catalog.suffix + ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = read_json(catalog) if catalog.exists() else {"schema": "passage-candidate-catalog/v1", "candidates": {}}
        if set(data) != {"schema", "candidates"} or data["schema"] != "passage-candidate-catalog/v1":
            raise ValueError("invalid candidate catalog")
        previous = data["candidates"].get(package.identity)
        if previous:
            if previous["manifest_sha256"] != package.digest:
                raise ValueError("candidate version is already registered with different bytes")
            return {"identity": package.identity, "manifest_sha256": package.digest, "added": False}
        data["candidates"][package.identity] = {
            "manifest_sha256": package.digest,
            "manifest_path": str(package.path),
            "family": package.document["family"],
        }
        atomic_json(catalog, data)
        return {"identity": package.identity, "manifest_sha256": package.digest, "added": True}
