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
