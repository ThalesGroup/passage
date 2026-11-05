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

"""Compare an OCI image's applied layer contents with an exported rootfs.

The proof describes file types, file bytes, symlink targets and executable bits.
It does not attest container publication, runtime use, device placement, or permissions
that the rootless extractor intentionally changes. Inputs must be trusted OCI archives.
"""

from __future__ import annotations
import hashlib
import json
import posixpath
import re
import subprocess
import tarfile


def sha(stream):
    h = hashlib.sha256()
    while data := stream.read(1024 * 1024):
        h.update(data)
    return h.hexdigest()


def path_name(value):
    if not isinstance(value, str) or value.startswith("/"):
        raise ValueError("absolute archive path")
    value = posixpath.normpath(value)
    if value == ".." or value.startswith("../") or "\x00" in value:
        raise ValueError("escaping archive path")
    return value


def walk_tar(archive, *, layers=False):
    records = []
    for member in archive:
        name = path_name(member.name)
        if name == ".":
            continue
        base = posixpath.basename(name)
        if layers and base.startswith(".wh."):
            records.append((name, ("whiteout",)))
            continue
        if member.isdir():
            entry = ("dir",)
        elif member.issym():
            entry = ("symlink", member.linkname)
        elif member.islnk():
            entry = ("hardlink", path_name(member.linkname))
        elif member.isfile():
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("missing file content")
            entry = ("file", member.size, sha(stream), bool(member.mode & 0o111))
        elif member.ischr() or member.isblk() or member.isfifo():
            entry = ("special", member.type.decode("latin1"), member.devmajor, member.devminor)
        else:
            raise ValueError("unsupported tar member type")
        records.append((name, entry))
    return records


def apply(records, tree, *, whiteouts=False):
    if whiteouts:
        for name, entry in records:
            if entry[0] != "whiteout":
                continue
            parent = posixpath.dirname(name)
            base = posixpath.basename(name)
            if base == ".wh..wh..opq":
                prefix = parent + "/" if parent else ""
                for key in list(tree):
                    if key.startswith(prefix) and key != parent:
                        del tree[key]
            else:
                target = posixpath.join(parent, base[4:])
                for key in list(tree):
                    if key == target or key.startswith(target + "/"):
                        del tree[key]
    pending = []
    for name, entry in records:
        if entry[0] == "whiteout":
            continue
        if entry[0] == "hardlink":
            pending.append((name, entry[1]))
            continue
        if name in tree and tree[name][0] == "dir" and entry[0] != "dir":
            for key in list(tree):
                if key.startswith(name + "/"):
                    del tree[key]
        tree[name] = entry
    while pending:
        unresolved = []
        progressed = False
        for name, target in pending:
            ref = tree.get(target)
            if ref and ref[0] == "file":
                tree[name] = ref
                progressed = True
            else:
                unresolved.append((name, target))
        if not progressed:
            raise ValueError("unresolved hardlink")
        pending = unresolved


def blob(tar, descriptor):
    digest = descriptor.get("digest", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        raise ValueError("descriptor digest")
    member = tar.getmember("blobs/sha256/" + digest[7:])
    if not member.isfile() or member.size != descriptor.get("size"):
        raise ValueError("descriptor size")
    stream = tar.extractfile(member)
    if sha(stream) != digest[7:]:
        raise ValueError("descriptor hash")
    return tar.extractfile(member)


def compare(oci_path, manifest_sha256, rootfs_archive, *, allowed_differences=()):
    if not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256):
        raise ValueError("manifest pin")
    tree = {}
    with tarfile.open(oci_path, "r:") as tar:
        index = json.load(tar.extractfile("index.json"))
        manifests = index["manifests"]
        if len(manifests) != 1 or manifests[0].get("digest") != "sha256:" + manifest_sha256:
            raise ValueError("image manifest mismatch")
        manifest = json.load(blob(tar, manifests[0]))
        config = json.load(blob(tar, manifest["config"]))
        if (config.get("os"), config.get("architecture")) != ("linux", "amd64"):
            raise ValueError("platform mismatch")
        for descriptor in manifest["layers"]:
            with tarfile.open(fileobj=blob(tar, descriptor), mode="r|*") as layer:
                apply(walk_tar(layer, layers=True), tree, whiteouts=True)
    result = {}
    with subprocess.Popen(["zstd", "-q", "-d", "-c", "--", str(rootfs_archive)], stdout=subprocess.PIPE) as p:
        try:
            with tarfile.open(fileobj=p.stdout, mode="r|") as archive:
                apply(walk_tar(archive), result)
        finally:
            p.stdout.close()
        if p.wait() != 0:
            raise ValueError("rootfs decompression failed")
    # A tar need not carry explicit parent directory records.
    for mapping in (tree, result):
        for name in list(mapping):
            parent = posixpath.dirname(name)
            while parent and parent != ".":
                mapping.setdefault(parent, ("dir",))
                parent = posixpath.dirname(parent)
    differing = [key for key in sorted(tree.keys() | result.keys()) if tree.get(key) != result.get(key)]
    if set(differing) != set(allowed_differences):
        raise ValueError(f"OCI and rootfs differ: {len(differing)} paths; {differing[:30]}")
    exceptions = {key: {"oci": tree.get(key), "export": result.get(key)} for key in differing}
    return {
        "exceptions": exceptions,
        "manifest_digest": "sha256:" + manifest_sha256,
        "config_digest": manifest["config"]["digest"],
        "layers": len(manifest["layers"]),
        "paths": len(tree),
        "export_paths": len(result),
        "rootfs_archive_sha256": sha(open(rootfs_archive, "rb")),
        "scope": "verified path types, file bytes, symlinks and executable bits except recorded paths; not OCI runtime or mounted tree",
    }
