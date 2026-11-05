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

"""Guard ordering: no test files or real datasets are needed."""

import pytest
from passage_eval.campaign.direct import admitted_input


def test_refusal_happens_before_reading_roster_or_dataset(monkeypatch, tmp_path):
    def denied(*args):
        raise PermissionError("gates are not GREEN")

    def forbidden(*args, **kwargs):
        pytest.fail("read before admission")

    monkeypatch.setattr(admitted_input, "grant_test_admission", denied)
    monkeypatch.setattr(admitted_input.roster_io, "load_roster", forbidden)
    monkeypatch.setattr(admitted_input.roster_io, "_check_root", forbidden)
    with pytest.raises(PermissionError, match="not GREEN"):
        next(admitted_input.admitted_requests("record", "evidence", "roster", tmp_path, None))


def test_refuses_wrong_scope_before_opening_dataset(monkeypatch, tmp_path):
    class Grant:
        from types import SimpleNamespace

        report = SimpleNamespace(admitted={"profile": "descriptive_native_2026"})
        admissible_roster_sha256 = "a" * 64

        def ensure_unchanged(self):
            pass

    monkeypatch.setattr(admitted_input, "grant_test_admission", lambda *args: Grant())
    monkeypatch.setattr(
        admitted_input.roster_io, "load_roster", lambda *args, **kw: {"dataset": {"split": "test"}, "tasks": []}
    )
    monkeypatch.setattr(admitted_input, "refuse_test_roster", lambda *args, **kw: None)
    monkeypatch.setattr(admitted_input.roster_io, "_check_root", lambda *args, **kw: pytest.fail("dataset opened"))
    with pytest.raises(ValueError, match="1024"):
        next(admitted_input.admitted_requests("record", "evidence", "roster", tmp_path, None))
