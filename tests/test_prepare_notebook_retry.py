import json


from passage.prepare import prepare_export


def test_prepare_retries_notebook(tmp_path, monkeypatch):
    # Setup minimal dataset and notebooks
    dataset_root = tmp_path / "outputs" / "dataset"
    notebooks_root = tmp_path / "notebooks"
    export_root = tmp_path / "outputs" / "export"

    # Create a single resolution directory
    res_dir = dataset_root / "256"
    res_dir.mkdir(parents=True)

    # Create a simple dataset summary so prepare continues
    (res_dir / "summary.json").write_text(json.dumps({}))

    # Create notebooks dir and a fake dataset.ipynb
    notebooks_root.mkdir(parents=True)
    (notebooks_root / "dataset.ipynb").write_text("{}")

    # Track calls to the papermill executor and simulate two failures then success
    calls = {"count": 0}

    def fake_execute(input_nb, output_nb, parameters, cwd=None):
        calls["count"] += 1
        # Ensure DATASET_DIR was passed through to the notebook parameters
        assert parameters.get("DATASET_DIR") == str(dataset_root)
        # fail twice, succeed on third
        return calls["count"] >= 3

    monkeypatch.setattr("passage.prepare._execute_notebook_with_papermill", fake_execute)
    monkeypatch.setattr("time.sleep", lambda s: None)

    ok = prepare_export(
        export_dirpath=str(export_root),
        dataset_dirpath=str(dataset_root),
        calibrate_dirpath=str(tmp_path / "outputs" / "calibrate"),
        notebooks_dirpath=str(notebooks_root),
        resolutions=[256],
        execute_notebooks=True,
        max_samples_notebook=0,
        max_notebook_retries=5,
        notebook_retry_delay=0,
    )

    assert ok is True
    # Ensure executor was called at least 3 times
    assert calls["count"] >= 3
    # Ensure dataset.ipynb was copied or produced in the export
    assert (export_root / "256" / "dataset.ipynb").exists()
