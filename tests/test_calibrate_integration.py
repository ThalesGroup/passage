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

import pytest

import passage.calibrate as cal


@pytest.mark.integration
@pytest.mark.calibrate
def test_run_calibrate_with_existing_resume(tmp_path, monkeypatch):
    outputs_dir = tmp_path / "outputs"
    # Place calibrate.json at outputs root (as expected by run_calibrate)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    existing = {
        "global_min": 0.0,
        "global_max": 100.0,
        "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0.0, "max": 100.0, "lat": 0, "lon": 0}],
    }
    with open(outputs_dir / "calibrate.json", "w") as f:
        json.dump(existing, f)

    # Create tiles dir with a matching DSM file name (content not important for resume path)
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()
    (tiles_dir / "N000E000_DSM.tif").write_text("")

    # Run calibrate in sequential mode; should read existing stats and write outputs
    cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=True, num_cpus=0)

    # Expect outputs/calibrate.json to exist and map image to be created
    assert (outputs_dir / "calibrate.json").exists()
    assert (outputs_dir / "max_elevation_map.png").exists()


@pytest.mark.unit
@pytest.mark.calibrate
def test_run_calibrate_no_tiles(tmp_path):
    outputs_dir = tmp_path / "outputs"
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # No DSM tiles present -> should early return and not create outputs/calibrate.json
    cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)
    assert not (outputs_dir / "calibrate.json").exists()
