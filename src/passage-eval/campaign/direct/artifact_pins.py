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

"""Prototype checks only; the worker's actual loaded files still need binding."""

from pathlib import Path
from passage_eval.campaign.edge.target import _sha256_regular_file
from passage_eval.io import pinned


def verify_files(paths, expected):
    names = {"bundle", "checkpoint", "engine", "decoder"}
    if set(paths) != names or set(expected) != names:
        raise ValueError("bundle, checkpoint, engine and decoder pins are required")
    actual = {}
    for name in sorted(names):
        actual[name] = _sha256_regular_file(paths[name])
        if actual[name] != expected[name]:
            raise ValueError(f"{name} digest differs")
    data = pinned.strict_json_loads(pinned.read_bounded(Path(paths["bundle"])), name="bundle")
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("lineage"), dict)
        or data["lineage"].get("checkpoint_sha256") != actual["checkpoint"]
    ):
        raise ValueError("bundle checkpoint lineage differs")
    if _sha256_regular_file(paths["bundle"]) != actual["bundle"]:
        raise ValueError("bundle changed while checking lineage")
    return actual
