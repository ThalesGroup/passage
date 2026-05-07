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

import builtins
import importlib
import sys

import passage


def test_no_images_flag_passed_to_run_generate(monkeypatch):
    captured = {}

    def fake_run_generate(
        outputs_dirpath,
        config_filepath,
        tiles_dirpath,
        calibration_filepath="outputs/calibrate/calibrate.json",
        resolution=None,
        num_samples=None,
        num_cpus=None,
        disable_images=False,
        force=False,
    ):
        captured["disable_images"] = disable_images

    monkeypatch.setattr(passage, "run_generate", fake_run_generate)

    test_argv = ["passage", "generate", "--no-images"]
    monkeypatch.setattr(sys, "argv", test_argv)

    passage.main()

    assert "disable_images" in captured
    assert captured["disable_images"] is True


def test_generate_import_does_not_require_cv2(monkeypatch):
    original_import = builtins.__import__

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "cv2":
            raise ImportError("cv2 intentionally unavailable for this test")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", blocked_import)
    sys.modules.pop("passage.generate", None)
    sys.modules.pop("passage.visualization", None)

    module = importlib.import_module("passage.generate")

    assert module is not None
    assert hasattr(module, "run_generate")
