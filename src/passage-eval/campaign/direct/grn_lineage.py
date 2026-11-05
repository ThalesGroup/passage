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

"""Verify the frozen Seg GRN rewrite and recorded TensorRT build lineage.

This is a graph/build identity check, not a claim about runtime arithmetic,
numerical parity, route quality, precision admission or worker file loading.
"""

import hashlib
import types
import json
import tarfile
import uuid
from pathlib import Path
from passage_eval.campaign.edge.target import _sha256_regular_file

SOURCE = "c06beba975a84d52a9d99d9dc556ca65e70906be94d7a654ca21e8c7f711156b"
REWRITTEN = "d695f28bbf82c6675b27fb36b866d7f9773906f377adfc499e2ebd60367083f0"
SCRIPT = "f76343f4cb31b48366333fadc3c00f600cef02e8031e6fb8590697b0a9e7f16f"
INPUT_ARCHIVE = "403db0144c70c0f9ac1b4d78ac157bfe765d29e6e00e7d88358c42d93daa22c6"
RESULTS = {
    "orin": (
        "4cd783b8b937a1b4d98efec68b3bd66a999d5e298ff115f321154691e6e88b0b",
        "700062ee3f98eb3188173665dcbe812fcd8c7419cdc61431065532bd8c07883b",
    ),
    "thor": (
        "b0def296457db184597b2393f0943df199d2cddf7ff765405f7fd85cdca73e6b",
        "990f57e43a231e636ddbb5ea2bc46e9b808b8c0941fee3300e536b8b5aada4fd",
    ),
}


# Each case is selected by checkpoint and target; values come from the pinned
# source export and the actual build archives, not from the caller's manifest.
_CASES = {
    ("eee055e084aaaec2aa4befb9db9654882d98fe629bb85e70d24a7bf5575eb79b", target): {
        "source": SOURCE,
        "rewritten": REWRITTEN,
        "input_archive": INPUT_ARCHIVE,
        "result_archive": result,
        "engine": engine,
    }
    for target, (result, engine) in RESULTS.items()
}
_CASES.update(
    {
        ("70a4bcd60a3b484b804b4a540a6b23fde3f12ac0113d96919eb94e171e304768", "orin"): {
            "source": "08db1e50bac53131b450601067ca62dccc8541a6023935d08f25205d0349e39d",
            "rewritten": "4f23412bb824998d0cfc8148e590975cbfa3c9d3d2e00e541fb3d463ff5238ff",
            "input_archive": "fac3319e2d04124dd0017e4b29810ca7a01ca768339b7192f15687207fd41f05",
            "result_archive": "db3e04569065c745874ff4ed76e2ad70ed5940946bb2b244e522811b9dd8e0f1",
            "engine": "771abca6fefc392da4ad8cdb8954b2123830c82edf1078d43b0ef8682b3dbe98",
        },
        ("70a4bcd60a3b484b804b4a540a6b23fde3f12ac0113d96919eb94e171e304768", "thor"): {
            "source": "08db1e50bac53131b450601067ca62dccc8541a6023935d08f25205d0349e39d",
            "rewritten": "4f23412bb824998d0cfc8148e590975cbfa3c9d3d2e00e541fb3d463ff5238ff",
            "input_archive": "fac3319e2d04124dd0017e4b29810ca7a01ca768339b7192f15687207fd41f05",
            "result_archive": "517eab63bafbbee7fb662ee09f79bdad833f5166a50cf4bd34a20272bd79e5a2",
            "engine": "9526a154858ca03b7ad7ae82c407c71637042ac798b300b16145685a1483148b",
        },
        ("10ad13f4b678da156714284fcd1788a998652405a9126d2a20b3a20a62f7f1d7", "orin"): {
            "source": "07cf0112c2b5d91738a3cf4c62b0991c6600df97809b15bcc775a6b71042417e",
            "rewritten": "e52fa8dad8b71d55bb93238b3a4c1a79c4a884e6ab0e105f469683b04035c52d",
            "input_archive": "4865dbf4492d8bf8b6a79b83a69fecb0b8bd4a592dc316779a4e08bb93f9f122",
            "result_archive": "f37cb80ddefc8d334aa88d6d9ba7b47ee0918d83bc69473231c459847861ac5e",
            "engine": "3153b8f3a887440a917fbb5e3699bfc57c8beb57fdf4c48241b55208ce8fe1bf",
        },
        ("10ad13f4b678da156714284fcd1788a998652405a9126d2a20b3a20a62f7f1d7", "thor"): {
            "source": "07cf0112c2b5d91738a3cf4c62b0991c6600df97809b15bcc775a6b71042417e",
            "rewritten": "e52fa8dad8b71d55bb93238b3a4c1a79c4a884e6ab0e105f469683b04035c52d",
            "input_archive": "4865dbf4492d8bf8b6a79b83a69fecb0b8bd4a592dc316779a4e08bb93f9f122",
            "result_archive": "fb07618bf2ce53a50fa8f922187aec0fc98522720bf3a050c899df81ece800b3",
            "engine": "c54a2aa4d7f83ee035dae8d5816d12d2f085216abae7c99d5d5a7c15187becf1",
        },
    }
)


def _select_case(bundle, compiled, target):
    checkpoint = (bundle.manifest.get("lineage") or {}).get("checkpoint_sha256")
    case = _CASES.get((checkpoint, target))
    if case is None or compiled.get("engine_sha256") != case["engine"]:
        raise ValueError("GRN checkpoint, target or engine is not a pinned case")
    return case


def _member(archive, name):
    matches = [item for item in archive.getmembers() if item.name == name]
    if len(matches) != 1 or not matches[0].isfile():
        raise ValueError("GRN archive missing unique regular member: " + name)
    return archive.extractfile(matches[0]).read()


def _digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_grn_lineage(bundle, compiled, target, evidence):
    """Reproduce the pinned graph rewrite and verify both pinned build archives."""
    if (
        target not in RESULTS
        or not isinstance(evidence, dict)
        or set(evidence) != {"rewrite_script", "input_archive", "result_archive", "reproduction_root"}
    ):
        raise ValueError("GRN lineage requires target and three pinned evidence files")
    case = _select_case(bundle, compiled, target)
    source, rewritten = case["source"], case["rewritten"]
    input_archive, result_archive, engine_hash = case["input_archive"], case["result_archive"], case["engine"]
    paths = {name: Path(evidence[name]) for name in ("rewrite_script", "input_archive", "result_archive")}
    reproduction_root = Path(evidence["reproduction_root"]).resolve(strict=True)
    if not reproduction_root.is_dir():
        raise ValueError("GRN reproduction root must be an existing durable directory")
    expected = {"rewrite_script": SCRIPT, "input_archive": input_archive, "result_archive": result_archive}
    for name, path in paths.items():
        if _sha256_regular_file(path) != expected[name]:
            raise ValueError("GRN evidence hash mismatch: " + name)
    if _sha256_regular_file(bundle.model_path) != source or bundle.manifest["artifacts"].get("model.onnx") != source:
        raise ValueError("GRN original bundle model differs")
    transforms = [
        x for x in compiled.get("transforms", []) if isinstance(x, dict) and x.get("kind") == "onnx_grn_rewrite"
    ]
    if len(transforms) != 1:
        raise ValueError("GRN compiled transform missing or ambiguous")
    transform = transforms[0]
    if (
        transform.get("source_onnx_sha256") != source
        or transform.get("rewritten_onnx_sha256") != rewritten
        or transform.get("script_sha256") != SCRIPT
        or type(transform.get("pinned_node_count")) is not int
        or transform["pinned_node_count"] != 72
        or compiled.get("engine_sha256") != engine_hash
    ):
        raise ValueError("GRN compiled transform or engine differs")
    build = compiled.get("build_evidence", {})
    if (
        build.get("input_archive_sha256") != input_archive
        or build.get("result_archive_sha256") != result_archive
        or build.get("input_onnx_sha256") != rewritten
    ):
        raise ValueError("GRN compiled build evidence differs")
    with tarfile.open(paths["input_archive"], "r:") as archive:
        model = _member(archive, "seg.onnx")
        layer_spec = _member(archive, "grn-fp32-layers.txt").decode("utf-8").strip()
        script = _member(archive, "target-script.sh").decode("utf-8")
        package = json.loads(_member(archive, "package-manifest.json"))
    if (
        _digest(model) != rewritten
        or package.get("files", {}).get("seg.onnx") != rewritten
        or package["files"].get("grn-fp32-layers.txt") != _digest((layer_spec + "\n").encode())
    ):
        raise ValueError("GRN build input differs")
    if not all(
        flag in script
        for flag in (
            "--fp16",
            "--noTF32",
            "--precisionConstraints=obey",
            "--layerPrecisions=",
            "--layerOutputTypes=",
            "--skipInference",
        )
    ):
        raise ValueError("GRN build flags differ")
    with tarfile.open(paths["result_archive"], "r:") as archive:
        engine = _member(archive, "results/seg.engine")
        log = _member(archive, "results/seg-build.log")
        checks = dict(
            line.split("  ", 1)[::-1] for line in _member(archive, "results/SHA256SUMS").decode("ascii").splitlines()
        )
    if (
        _digest(engine) != engine_hash
        or checks.get("seg.engine") != engine_hash
        or checks.get("seg-build.log") != _digest(log)
        or build.get("build_log_sha256") != _digest(log)
        or b"&&&& PASSED TensorRT.trtexec" not in log
    ):
        raise ValueError("GRN build result differs")
    script_bytes = paths["rewrite_script"].read_bytes()
    if _digest(script_bytes) != SCRIPT:
        raise ValueError("GRN rewrite script changed")
    module = types.ModuleType("_pinned_grn_rewrite")
    module.__file__ = str(paths["rewrite_script"])
    exec(compile(script_bytes, module.__file__, "exec"), module.__dict__)
    # Create a unique durable directory and retain every reproduction.
    result_dir = reproduction_root / ("grn-reproduction-" + uuid.uuid4().hex)
    result_dir.mkdir(mode=0o700, exist_ok=False)
    result = result_dir / "rewritten.onnx"
    nodes = module.rewrite(bundle.model_path, result)
    if (
        len(nodes) != 72
        or ",".join(n + ":fp32" for n in nodes) != layer_spec
        or _sha256_regular_file(result) != rewritten
    ):
        raise ValueError("GRN rewrite reproduction differs")
    if any(_sha256_regular_file(path) != expected[name] for name, path in paths.items()):
        raise ValueError("GRN evidence changed during verification")
    if _sha256_regular_file(bundle.model_path) != source:
        raise ValueError("GRN source changed during verification")
    return {
        "source_onnx_sha256": source,
        "rewritten_onnx_sha256": rewritten,
        "rewrite_script_sha256": SCRIPT,
        "input_archive_sha256": input_archive,
        "result_archive_sha256": result_archive,
        "engine_sha256": engine_hash,
        "pinned_node_count": 72,
        "reproduced_file": str(result),
        "scope": "graph and recorded build lineage only; arithmetic precision and quality not established",
    }
