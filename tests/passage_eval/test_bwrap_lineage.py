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

import io
import tarfile
import hashlib
import json
import subprocess
import pytest
from passage_eval.campaign.direct.bwrap_lineage import compare


def packed(files):
    out = io.BytesIO()
    with tarfile.open(fileobj=out, mode="w") as tar:
        for name, data in files:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if name == "app" else 0o644
            tar.addfile(info, io.BytesIO(data))
    return out.getvalue()


def image(tmp_path, root):
    layer1 = packed([("app", b"old"), ("removed", b"x")])
    layer2 = packed([("app", b"new"), (".wh.removed", b"")])
    config = json.dumps({"architecture": "amd64", "os": "linux"}).encode()

    def desc(data):
        return {"digest": "sha256:" + hashlib.sha256(data).hexdigest(), "size": len(data)}

    manifest = json.dumps({"config": desc(config), "layers": [desc(layer1), desc(layer2)]}).encode()
    md = desc(manifest)
    index = json.dumps({"manifests": [md]}).encode()
    files = [("index.json", index)] + [
        ("blobs/sha256/" + desc(blob)["digest"][7:], blob) for blob in [config, layer1, layer2, manifest]
    ]
    oci = tmp_path / "oci.tar"
    oci.write_bytes(packed(files))
    exported = tmp_path / "root.zst"
    exported.write_bytes(
        subprocess.run(["zstd", "-q", "-c"], input=packed(root), capture_output=True, check=True).stdout
    )
    return oci, md["digest"][7:], exported


def test_layer_whiteout_and_replacement(tmp_path):
    oci, digest, root = image(tmp_path, [("app", b"new")])
    result = compare(oci, digest, root)
    assert result["paths"] == 1 and result["layers"] == 2


def test_rootfs_substitution_rejected(tmp_path):
    oci, digest, root = image(tmp_path, [("app", b"evil")])
    with pytest.raises(ValueError, match="OCI and rootfs differ"):
        compare(oci, digest, root)


def test_manifest_substitution_rejected(tmp_path):
    oci, digest, root = image(tmp_path, [("app", b"new")])
    with pytest.raises(ValueError, match="manifest mismatch"):
        compare(oci, "a" * 64, root)


def test_extracted_tree_rejects_changed_file(tmp_path):
    from passage_eval.campaign.direct.bwrap_tree import compare_tree

    _, _, archive = image(tmp_path, [("app", b"new")])
    root = tmp_path / "extracted"
    root.mkdir()
    (root / "app").write_bytes(b"new")
    (root / "app").chmod(0o755)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert compare_tree(root, archive, digest)["rootfs_path"] == str(root)
    (root / "app").write_bytes(b"different")
    with pytest.raises(ValueError, match="extracted rootfs differs"):
        compare_tree(root, archive, digest)
