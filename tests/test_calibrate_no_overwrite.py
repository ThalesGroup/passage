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

import json
from pathlib import Path

import pytest

import passage.calibrate as cal


@pytest.mark.unit
@pytest.mark.calibrate
def test_run_calibrate_does_not_overwrite_repo_outputs(tmp_path, monkeypatch):
    # Create a sentinel calibrate.json in the repository outputs/calibrate
    repo_outputs = Path("outputs") / "calibrate"
    repo_outputs.mkdir(parents=True, exist_ok=True)
    sentinel = repo_outputs / "calibrate.json"
    sentinel.write_text(json.dumps({"marker": "keep"}))

    # Prepare tiles dir (empty)
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # Run calibrate using the repository default outputs path; under pytest it should redirect
    cal.run_calibrate(str(repo_outputs), str(tiles_dir), resume=False, num_cpus=0)

    # Ensure the sentinel file still contains the same content (not overwritten)
    assert sentinel.exists()
    data = json.loads(sentinel.read_text())
    assert data.get("marker") == "keep"
