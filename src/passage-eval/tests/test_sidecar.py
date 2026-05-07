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

"""Tests for the reproducibility-sidecar manifest (App. J Rule 8)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from passage_eval import (
    DEFAULT_SEEDS,
    ReproducibilitySidecar,
    compute_calibration_decisions_hash,
)


def test_calibration_decisions_hash_is_deterministic():
    a = {"threshold": 0.5, "step": 1, "schedule": [1, 2, 3]}
    b = {"step": 1, "schedule": [1, 2, 3], "threshold": 0.5}  # different order, same content
    assert compute_calibration_decisions_hash(a) == compute_calibration_decisions_hash(b)


def test_calibration_decisions_hash_changes_with_value():
    a = {"threshold": 0.5}
    b = {"threshold": 0.6}
    assert compute_calibration_decisions_hash(a) != compute_calibration_decisions_hash(b)


def test_sidecar_round_trip(tmp_path: Path):
    decisions_hash = compute_calibration_decisions_hash({"threshold": 0.5, "step": 1})
    sc = ReproducibilitySidecar(
        method_name="dense_regression_finalist",
        family="dense_regression",
        dataset_version="v2.0.0",
        dataset_content_hash="sha256:deadbeef" + "0" * 56,
        operating_point="256x256/elevation",
        seeds=list(DEFAULT_SEEDS),
        split_seed=0,
        calibration_decisions_hash=decisions_hash,
        hardware_stack="R1",
        batch_size=1,
        dtype="fp16",
        determinism_flag="algorithmically_deterministic",
        commit_sha="abc123",
        notes="example",
    )
    p = sc.write(tmp_path / "row.json")
    assert p.exists()
    loaded = ReproducibilitySidecar.read(p)
    assert loaded.method_name == sc.method_name
    assert loaded.calibration_decisions_hash == decisions_hash
    assert loaded.seeds == list(DEFAULT_SEEDS)
    # JSON is canonical (sorted keys)
    raw = json.loads(p.read_text())
    assert "calibration_decisions_hash" in raw


def test_sidecar_rejects_invalid_determinism_flag():
    with pytest.raises(ValueError):
        ReproducibilitySidecar(
            method_name="m",
            family="dense_regression",
            dataset_version="v2.0.0",
            dataset_content_hash="sha256:0",
            operating_point="256x256/elevation",
            determinism_flag="ambiguous",
        )


def test_sidecar_rejects_invalid_dtype():
    with pytest.raises(ValueError):
        ReproducibilitySidecar(
            method_name="m",
            family="dense_regression",
            dataset_version="v2.0.0",
            dataset_content_hash="sha256:0",
            operating_point="256x256/elevation",
            dtype="float64",
        )


def test_sidecar_extra_round_trips(tmp_path: Path):
    sc = ReproducibilitySidecar(
        method_name="m",
        family="dense_regression",
        dataset_version="v2.0.0",
        dataset_content_hash="sha256:0",
        operating_point="256x256/elevation",
        extra={"calibration_split_index": "0.02-slice"},
    )
    sc.write(tmp_path / "row.json")
    loaded = ReproducibilitySidecar.read(tmp_path / "row.json")
    assert loaded.extra["calibration_split_index"] == "0.02-slice"
