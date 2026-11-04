"""Check the extracted Tiger tree against its pinned rootfs export (outside timed inference)."""

import os
import stat
import subprocess
import tarfile
from pathlib import Path
from passage_eval.campaign.direct.bwrap_lineage import walk_tar, apply, sha


def compare_tree(root, archive, expected_sha):
    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ValueError("root directory differs")
    with open(archive, "rb") as f:
        if sha(f) != expected_sha:
            raise ValueError("rootfs archive digest differs")
    expected = {}
    with subprocess.Popen(["zstd", "-q", "-d", "-c", "--", str(archive)], stdout=subprocess.PIPE) as p:
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as tar:
                apply(walk_tar(tar), expected)
        finally:
            p.stdout.close()
        if p.wait() != 0:
            raise ValueError("archive decompression failed")
    for name in list(expected):
        parent = Path(name).parent
        while str(parent) != ".":
            expected.setdefault(parent.as_posix(), ("dir",))
            parent = parent.parent
    actual = {}
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        for name in dirs + files:
            path = Path(dirpath) / name
            key = path.relative_to(root).as_posix()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                actual[key] = ("symlink", os.readlink(path))
            elif stat.S_ISDIR(info.st_mode):
                actual[key] = ("dir",)
            elif stat.S_ISREG(info.st_mode):
                with path.open("rb") as f:
                    digest = sha(f)
                actual[key] = ("file", info.st_size, digest, bool(info.st_mode & 0o111))
            else:
                raise ValueError("special filesystem node: " + key)
    if actual != expected:
        differing = [key for key in sorted(actual.keys() | expected.keys()) if actual.get(key) != expected.get(key)]
        raise ValueError(f"extracted rootfs differs: {len(differing)} paths; {differing[:20]}")
    return {
        "rootfs_archive_sha256": expected_sha,
        "rootfs_path": str(root),
        "file_count": len(actual),
        "scope": "point-in-time path types, bytes, symlinks and executable bits; not mode/ownership or transient modifications",
    }
