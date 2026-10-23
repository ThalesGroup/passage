"""Offline source-authentication contracts using tiny real files."""

import hashlib
import importlib
import json
from types import SimpleNamespace

import pytest


REVISION = "a" * 40


def auth():
    return importlib.import_module("passage.authentication")


def snapshot(tmp_path, monkeypatch, files):
    root = tmp_path / "source"
    root.mkdir()
    siblings = []
    for name, data, lfs in files:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        siblings.append(
            SimpleNamespace(
                rfilename=name,
                size=len(data),
                blob_id=blob,
                lfs=SimpleNamespace(sha256=hashlib.sha256(data).hexdigest(), size=len(data)) if lfs else None,
            )
        )
    import huggingface_hub

    monkeypatch.setattr(
        huggingface_hub.HfApi, "repo_info", lambda self, **kwargs: SimpleNamespace(sha=REVISION, siblings=siblings)
    )
    return root, tmp_path / "evidence", siblings


def test_lfs_and_git_blob_known_hashes_cover_all_source_files(tmp_path, monkeypatch):
    root, output, _ = snapshot(
        tmp_path, monkeypatch, [("README.md", b"hello\n", False), ("train/data.parquet", b"abc", True)]
    )
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output, workers=2)
    assert result["status"] == "complete"
    assert result["content_verified"] is True
    assert result["verified_files"] == result["required_files"] == 2
    manifest = json.loads((output / "manifest.json").read_text())
    entries = {entry["path"]: entry for entry in manifest["files"]}
    assert entries["README.md"]["expected_hash"] == "ce013625030ba8dba906f756967f9e9ca394464a"
    assert entries["train/data.parquet"]["expected_hash"] == hashlib.sha256(b"abc").hexdigest()
    assert result == json.loads((output / "receipt.json").read_text())
    assert (root / "train/data.parquet").read_bytes() == b"abc"


def test_valid_resume_reads_no_source_bytes_and_ignores_truncated_tail(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    with (output / "progress.jsonl").open("ab") as stream:
        stream.write(b"{truncated")
    monkeypatch.setattr(auth(), "_digest_file", lambda *args, **kwargs: pytest.fail("unchanged file rehashed"))
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    assert result["status"] == "complete"
    assert result["resumed_files"] == 1
    assert result["bytes_read_this_run"] == 0


def test_modified_file_invalidates_resume_and_preserves_mismatch(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    (root / "one").write_bytes(b"xyz")
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    assert result["status"] == "partial"
    assert result["content_verified"] is False
    assert result["mismatched_files"] == 1
    assert result["resumed_files"] == 0
    assert (root / "one").read_bytes() == b"xyz"


def test_missing_and_size_mismatched_files_prevent_complete_receipt(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("missing", b"abc", True), ("short", b"abc", True)])
    (root / "missing").unlink()
    (root / "short").write_bytes(b"x")
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    assert result["status"] == "partial"
    assert result["missing_files"] == 1
    assert result["mismatched_files"] == 1
    assert result["verified_files"] == 0


def test_modification_during_hash_is_not_verified(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    original = auth()._digest_file

    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (root / "one").write_bytes(b"xyz")
        return result

    monkeypatch.setattr(auth(), "_digest_file", mutate)
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    assert result["status"] == "partial"
    assert result["changed_files"] == 1


def test_pinned_manifest_change_never_reuses_previous_records(tmp_path, monkeypatch):
    root, output, siblings = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    siblings[0].lfs.sha256 = "0" * 64
    with pytest.raises(ValueError, match="manifest"):
        auth().authenticate_snapshot(root, "owner/data", REVISION, output)


def test_file_changed_after_its_worker_finishes_is_rejected(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("first", b"abc", True), ("second", b"def", True)])
    original = auth()._digest_file

    def mutate_earlier(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == "second":
            (root / "first").write_bytes(b"xyz")
        return result

    monkeypatch.setattr(auth(), "_digest_file", mutate_earlier)
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output, workers=1)
    assert result["status"] == "partial"
    assert result["changed_files"] == 1
    assert result["verified_files"] == 1


def test_keyboard_interrupt_propagates_with_partial_receipt(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(auth(), "_digest_file", interrupt)
    with pytest.raises(KeyboardInterrupt):
        auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["status"] == "partial"
    assert receipt["content_verified"] is False
    assert receipt["pending_files"] == 1


def test_outside_source_symlink_is_rejected_without_reading_target(tmp_path, monkeypatch):
    root, output, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    external = tmp_path / "external"
    external.write_bytes(b"abc")
    (root / "one").unlink()
    (root / "one").symlink_to(external)
    result = auth().authenticate_snapshot(root, "owner/data", REVISION, output)
    assert result["status"] == "partial"
    assert result["error_files"] == 1
    assert result["bytes_read_this_run"] == 0


@pytest.mark.parametrize("workers", [0, -1, 9, True, 1.5])
def test_worker_bound_validated_before_network(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        auth().authenticate_snapshot(tmp_path, "owner/data", REVISION, tmp_path / "out", workers=workers)


def test_output_cannot_mutate_source_and_revision_must_be_immutable(tmp_path, monkeypatch):
    root, _, _ = snapshot(tmp_path, monkeypatch, [("one", b"abc", True)])
    with pytest.raises(ValueError, match="outside"):
        auth().authenticate_snapshot(root, "owner/data", REVISION, root / "evidence")
    with pytest.raises(ValueError, match="revision"):
        auth().authenticate_snapshot(root, "owner/data", "main", tmp_path / "out")
