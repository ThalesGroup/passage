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

from pathlib import Path

from passage.export import _execute_notebook_with_retries


def test_prepare_retries_notebook(monkeypatch, tmp_path):
    calls = {"count": 0}

    def fake_executor(input_nb: Path, output_nb: Path, parameters: dict, cwd=None):
        calls["count"] += 1
        assert parameters.get("RESOLUTION") == 256
        return calls["count"] >= 3  # succeed on third attempt

    monkeypatch.setattr("passage.export._execute_notebook_with_papermill", fake_executor)
    monkeypatch.setattr("time.sleep", lambda s: None)

    ok = _execute_notebook_with_retries(
        tmp_path / "in.ipynb",
        tmp_path / "out.ipynb",
        parameters={"RESOLUTION": 256},
        cwd=tmp_path,
        max_retries=5,
        retry_delay=0,
    )

    assert ok is True
    assert calls["count"] >= 3
