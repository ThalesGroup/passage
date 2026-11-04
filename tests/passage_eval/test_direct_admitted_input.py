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
