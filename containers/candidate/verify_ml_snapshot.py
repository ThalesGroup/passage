"""Verify the exact pre-install site-packages snapshot inside a new image."""
from hashlib import file_digest
import json
from pathlib import Path

INPUTS = Path("/opt/passage-image-inputs")
ROOT = Path("/usr/local/lib/python3.12/site-packages")
lock = json.loads((INPUTS / "runtime.snapshot.lock.json").read_text())
manifest_path = INPUTS / lock["snapshot_files"]
with manifest_path.open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == lock["snapshot_files_sha256"]
manifest = json.loads(manifest_path.read_text())
actual = {path.relative_to(ROOT).as_posix() for path in ROOT.rglob("*") if path.is_file()}
assert actual == set(manifest), "installed snapshot file set differs"
for relative, record in manifest.items():
    path = ROOT / relative
    assert not path.is_symlink() and path.resolve().is_relative_to(ROOT)
    assert path.stat().st_size == record["bytes"], relative
    with path.open("rb") as stream:
        assert file_digest(stream, "sha256").hexdigest() == record["sha256"], relative
wheel = INPUTS / lock["core_wheel"]["path"]
with wheel.open("rb") as stream:
    assert file_digest(stream, "sha256").hexdigest() == lock["core_wheel"]["sha256"]
print(json.dumps({"verified_snapshot_files": len(manifest), "snapshot_bytes": lock["snapshot_bytes"]}))
