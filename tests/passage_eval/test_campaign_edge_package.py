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

"""The edge package is a sealed input-only boundary."""

from __future__ import annotations

import json

import pytest

from tests.passage_eval import edge_world as world


def test_package_contains_only_declared_inputs_and_rejects_labels(tmp_path):
    from passage_eval.campaign.edge import PackageError
    from passage_eval.campaign.edge.package import build_input_package, verify_package

    package = world.package(tmp_path / "package")
    view = verify_package(package.root, package.sha256)
    assert view.manifest["procedure"] == "seg"
    assert set(view.inputs) == {
        "sample_ids",
        "terrain",
        "obstacles",
        "starts",
        "goals",
        "input_tensor",
    }
    with pytest.raises(PackageError, match="unknown"):
        build_input_package(
            tmp_path / "labelled",
            target="thor",
            plan_sha256=world.PLAN_SHA256,
            procedure="seg",
            inputs={**world.inputs(), "ground_truth": ["private"]},
        )


def test_tampered_package_is_rejected(tmp_path):
    from passage_eval.campaign.edge import PackageError
    from passage_eval.campaign.edge.package import verify_package

    package = world.package(tmp_path / "package")
    payload = json.loads((package.root / "package.json").read_text())
    payload["procedure"] = "reg"
    (package.root / "package.json").write_text(json.dumps(payload))
    with pytest.raises(PackageError, match="SHA-256 mismatch"):
        verify_package(package.root, package.sha256)
