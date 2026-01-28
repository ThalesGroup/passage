import subprocess


from passage.export import _execute_notebook_with_papermill


def test_uv_run_success(monkeypatch, tmp_path):
    in_nb = tmp_path / "in.ipynb"
    out_nb = tmp_path / "out.ipynb"
    in_nb.write_text("{}")

    calls = {"cmd": None}

    def fake_run(cmd, cwd=None, check=False):
        calls["cmd"] = cmd

        class R:
            returncode = 0

        return R()

    monkeypatch.setattr(subprocess, "run", fake_run)

    ok = _execute_notebook_with_papermill(in_nb, out_nb, parameters={"RESOLUTION": 256}, cwd=tmp_path)
    assert ok is True
    assert calls["cmd"] is not None
    assert "papermill" in calls["cmd"]


def test_uv_run_fails_fallback_to_pm(monkeypatch, tmp_path):
    in_nb = tmp_path / "in.ipynb"
    out_nb = tmp_path / "out.ipynb"
    in_nb.write_text("{}")

    # Simulate uv run failing
    def fake_run(cmd, cwd=None, check=False):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)

    # Simulate papermill being available and executing successfully
    monkeypatch.setattr("passage.export.PAPERMILL_AVAILABLE", True)

    def fake_exec(input_nb, output_nb, parameters, kernel_name=None):
        return None

    monkeypatch.setattr("passage.export.pm.execute_notebook", fake_exec)

    ok = _execute_notebook_with_papermill(in_nb, out_nb, parameters={"MAX_SAMPLES": 0}, cwd=tmp_path)
    assert ok is True


def test_uv_and_pm_fail(monkeypatch, tmp_path):
    in_nb = tmp_path / "in.ipynb"
    out_nb = tmp_path / "out.ipynb"
    in_nb.write_text("{}")

    def fake_run(cmd, cwd=None, check=False):
        raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr("passage.export.PAPERMILL_AVAILABLE", False)

    ok = _execute_notebook_with_papermill(in_nb, out_nb, parameters={}, cwd=tmp_path)
    assert ok is False
