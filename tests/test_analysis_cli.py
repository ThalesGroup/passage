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

"""CLI boundaries for read-only release audits and notebook preparation."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock

import pytest


def test_audit_writes_receipts_without_generation_or_upload(tmp_path, monkeypatch):
    import passage

    source_module = ModuleType("passage.dataset")
    source_module.DatasetSource = Mock(return_value="source")
    audit_module = ModuleType("passage.audit")
    audit_module.audit_source = Mock(return_value={"status": "sampled", "audited_rows": 2})
    monkeypatch.setitem(sys.modules, "passage.dataset", source_module)
    monkeypatch.setitem(sys.modules, "passage.audit", audit_module)
    output = tmp_path / "audit.json"
    passage.main(
        [
            "audit",
            "--dataset-path",
            str(tmp_path),
            "--resolution",
            "64",
            "128",
            "--max-samples",
            "2",
            "--max-expansions",
            "500",
            "--output",
            str(output),
        ]
    )
    assert [call.kwargs["resolution"] for call in audit_module.audit_source.call_args_list] == [64, 128]
    assert all(call.kwargs["limit"] == 2 for call in audit_module.audit_source.call_args_list)
    assert len(json.loads(output.read_text())["audits"]) == 2


def test_analyze_execution_is_explicit_and_source_root_is_not_destination(tmp_path, monkeypatch):
    import passage

    module = ModuleType("passage.notebook_export")
    module.prepare_snapshot_notebooks = Mock(return_value={"status": "prepared"})
    monkeypatch.setitem(sys.modules, "passage.notebook_export", module)
    passage.main(
        [
            "analyze",
            "--dataset-path",
            str(tmp_path / "source"),
            "--staging-root",
            str(tmp_path / "stage"),
            "--resolution",
            "64",
        ]
    )
    kwargs = module.prepare_snapshot_notebooks.call_args.kwargs
    assert kwargs["execute"] is False
    assert kwargs["resolutions"] == [64]
    assert kwargs["cell_timeout"] > 0 and kwargs["job_timeout"] > 0


@pytest.mark.parametrize(
    "command,option",
    [
        ("audit", "--max-samples"),
        ("audit", "--max-expansions"),
        ("analyze", "--cell-timeout"),
        ("analyze", "--job-timeout"),
    ],
)
def test_analysis_limits_must_be_positive(command, option, tmp_path):
    import passage

    args = [command, "--dataset-path", str(tmp_path), option, "0"]
    if command == "analyze":
        args += ["--staging-root", str(tmp_path / "stage")]
    with pytest.raises(SystemExit):
        passage.main(args)
