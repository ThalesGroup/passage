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

import passage.generate as gen


def test_num_samples_mapping_per_resolution(monkeypatch, tmp_path):
    # Create a minimal config with per-resolution num_samples mapping
    cfg = {"resolutions": [64, 128], "num_samples": {"64": 3, "128": 5}}
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(json.dumps(cfg))

    outputs_dir = tmp_path / "outputs"

    # Create a dummy calibration file
    calib_dir = tmp_path / "outputs" / "calibrate"
    calib_dir.mkdir(parents=True)
    calib = {"global_min": 0.0, "global_max": 100.0, "tile_stats": [{"min": 0.0, "max": 100.0}]}
    (calib_dir / "calibrate.json").write_text(json.dumps(calib))

    # Monkeypatch generate_sample to avoid heavy computation; return success
    def fake_generate_sample(args):
        index = args[0]
        return (index, True, "ok", 0.001)

    monkeypatch.setattr(gen, "generate_sample", fake_generate_sample)

    # Run generation sequentially (num_cpus=0) and force regeneration
    gen.run_generate(
        str(outputs_dir),
        str(cfg_path),
        str(tmp_path / "tiles"),
        calibration_filepath=str(calib_dir / "calibrate.json"),
        num_cpus=0,
        disable_images=True,
        force=True,
    )

    # Check per-resolution summaries
    for res, expected in [(64, 3), (128, 5)]:
        summary_path = outputs_dir / str(res) / "summary.json"
        assert summary_path.exists(), f"summary.json for {res} missing"
        data = json.loads(summary_path.read_text())
        assert data.get("requested_samples") == expected
        assert data.get("num_samples") == expected
        assert data.get("generated_samples") == expected
