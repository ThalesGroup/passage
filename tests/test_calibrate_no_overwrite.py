import json
from pathlib import Path

import pytest

import passage.calibrate as cal


@pytest.mark.unit
@pytest.mark.calibrate
def test_run_calibrate_does_not_overwrite_repo_outputs(tmp_path, monkeypatch):
    # Create a sentinel calibrate.json in the repository outputs/calibrate
    repo_outputs = Path("outputs") / "calibrate"
    repo_outputs.mkdir(parents=True, exist_ok=True)
    sentinel = repo_outputs / "calibrate.json"
    sentinel.write_text(json.dumps({"marker": "keep"}))

    # Prepare tiles dir (empty)
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir()

    # Run calibrate using the repository default outputs path; under pytest it should redirect
    cal.run_calibrate(str(repo_outputs), str(tiles_dir), resume=False, num_cpus=0)

    # Ensure the sentinel file still contains the same content (not overwritten)
    assert sentinel.exists()
    data = json.loads(sentinel.read_text())
    assert data.get("marker") == "keep"
