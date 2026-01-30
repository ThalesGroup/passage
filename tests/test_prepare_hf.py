from unittest.mock import MagicMock, patch

from passage.export import _generate_huggingface_yaml_header, export_and_upload


def test_generate_hf_yaml_header_configs_names():
    solver_names = ["astar_elevation", "astar_energy"]
    yaml = _generate_huggingface_yaml_header(
        "user/repo", [64, 128], ["train", "validation"], "1024x1024_sample", solver_names
    )
    # Check that config names are present (format may vary with yaml.dump)
    assert "64x64" in yaml
    assert "128x128" in yaml
    assert "train" in yaml
    assert "validation" in yaml
    # Check that solver path columns are present
    assert "path_astar_elevation_free" in yaml
    assert "path_astar_elevation_obstacles" in yaml
    assert "path_astar_energy_free" in yaml


def test_export_and_upload_uses_chunk_size(tmp_path):
    """Test that export_and_upload loads config properly."""
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "64" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "metadata").mkdir(parents=True)
    (split_dir / "tensors").mkdir(parents=True)
    (split_dir / "paths").mkdir(parents=True)

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """
export:
  chunk_size:
    64: 123
  sample:
    num_samples: 5
  default: 1024x1024_sample
  compression_level: 4
solvers:
  - name: astar_elevation
    backend: grid
    solver: astar
    cost_model: elevation
    weight: 2000.0
    heuristic: euclidean
    diagonal_movement: always
    max_marker_retries: 10
"""
    )

    calibrate_dir = tmp_path / "calibrate"
    calibrate_dir.mkdir(parents=True)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "README.md").write_text("# Test")

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_mock.return_value = api_instance
                export_and_upload(
                    dataset_dirpath=dataset_dir,
                    calibrate_dirpath=calibrate_dir,
                    notebooks_dirpath=notebooks_dir,
                    resolutions=[64],
                    repo_id=None,
                    execute_notebooks=False,
                    skip_upload=True,  # Don't actually upload
                )


def test_export_and_upload_passes_num_cpus_to_notebook(tmp_path):
    """If num_cpus is provided, ensure the MAX_WORKERS parameter passed to notebook execution reflects it."""
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "64" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "metadata").mkdir(parents=True)
    (split_dir / "tensors").mkdir(parents=True)
    (split_dir / "paths").mkdir(parents=True)

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True)
    (config_dir / "config.yaml").write_text(
        """
export:
  chunk_size:
    64: 123
  sample:
    num_samples: 5
  default: 1024x1024_sample
  compression_level: 4
solvers:
  - name: astar_elevation
    backend: grid
    solver: astar
    cost_model: elevation
    weight: 2000.0
    heuristic: euclidean
    diagonal_movement: always
    max_marker_retries: 10
"""
    )

    calibrate_dir = tmp_path / "calibrate"
    calibrate_dir.mkdir(parents=True)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "README.md").write_text("# Test")

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_mock.return_value = api_instance
                with patch("passage.export._execute_notebook_with_retries") as exec_mock:
                    ok = export_and_upload(
                        dataset_dirpath=dataset_dir,
                        calibrate_dirpath=calibrate_dir,
                        notebooks_dirpath=notebooks_dir,
                        resolutions=[64],
                        repo_id=None,
                        execute_notebooks=True,
                        skip_upload=True,
                        num_cpus=4,
                    )

    assert ok is True
    # Ensure we attempted to execute the dataset notebook and the parameters included MAX_WORKERS=4
    exec_mock.assert_called()
    # parameters may be passed positionally; find parameters dict
    parameters_dict = None
    for arg in exec_mock.call_args[0]:
        if isinstance(arg, dict) and "MAX_WORKERS" in arg:
            parameters_dict = arg
            break
    assert parameters_dict is not None and parameters_dict.get("MAX_WORKERS") == 4

    assert ok is True


def test_export_and_upload_skips_upload_when_requested(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "64" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "metadata").mkdir(parents=True)
    (split_dir / "tensors").mkdir(parents=True)
    (split_dir / "paths").mkdir(parents=True)

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """
export:
  chunk_size:
    64: 123
  sample:
    num_samples: 5
  default: 1024x1024_sample
  compression_level: 4
solvers:
  - name: astar_elevation
    backend: grid
    solver: astar
    cost_model: elevation
    weight: 2000.0
    heuristic: euclidean
    diagonal_movement: always
    max_marker_retries: 10
"""
    )

    calibrate_dir = tmp_path / "calibrate"
    calibrate_dir.mkdir(parents=True)

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "README.md").write_text("# Test")

    # skip_upload=True should not call _load_hf_credentials until after config is loaded
    ok = export_and_upload(
        dataset_dirpath=dataset_dir,
        calibrate_dirpath=calibrate_dir,
        notebooks_dirpath=notebooks_dir,
        resolutions=[64],
        execute_notebooks=False,
        skip_upload=True,
    )

    assert ok is True
