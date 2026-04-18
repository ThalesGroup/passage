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
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml
from datasets import Features, Value

from passage.export import (
    _generate_huggingface_yaml_header,
    _process_meta_for_prepare,
    _reset_hf_upload_cache,
    export_and_upload,
)
from passage.utils import save_tensor


def _setup_minimal_export_tree(tmp_path: Path) -> tuple[Path, Path, Path]:
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
  default: 1024x1024_samples
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
    (calibrate_dir / "calibrate.json").write_text(
        """
{
  "global_min": 0.0,
  "global_max": 1.0,
  "tile_stats": []
}
""".strip()
    )

    assets_dir = tmp_path / "assets"
    assets_dir.mkdir(parents=True)
    (assets_dir / "README.md").write_text("# Test")

    return dataset_dir, notebooks_dir, calibrate_dir


def _write_dummy_metadata(dataset_dir: Path, resolution: int, split: str = "train") -> None:
    metadata_dir = dataset_dir / str(resolution) / split / "metadata" / "N000" / "E000"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "0000000000.yaml").write_text("index: 0\n")


def _write_many_dummy_metadata(dataset_dir: Path, resolution: int, count: int, split: str = "train") -> None:
    metadata_dir = dataset_dir / str(resolution) / split / "metadata" / "N000" / "E000"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        (metadata_dir / f"{idx:010d}.yaml").write_text(f"index: {idx}\n")


def _mock_prepare_row(*_args, **_kwargs) -> dict:
    return {"id": "0000000000", "tensor": b"tensor", "metadata": "{}"}


def _mock_features(*_args, **_kwargs) -> Features:
    return Features(
        {
            "id": Value("string"),
            "tensor": Value("binary"),
            "metadata": Value("string"),
        }
    )


def _read_readme_front_matter(readme_path: Path) -> dict:
    readme_text = readme_path.read_text(encoding="utf-8")
    _, front_matter, _ = readme_text.split("---", 2)
    return yaml.safe_load(front_matter)


def test_generate_hf_yaml_header_configs_names():
    solver_names = ["astar_elevation", "astar_energy"]
    yaml = _generate_huggingface_yaml_header(
        "user/repo",
        ["64x64_samples", "64x64", "128x128_samples", "128x128"],
        ["train", "validation"],
        "1024x1024_samples",
        solver_names,
    )
    # Check that config names are present (format may vary with yaml.dump)
    assert "64x64" in yaml
    assert "128x128" in yaml
    assert "train" in yaml
    assert "validation" in yaml
    # Check that solver path and per-solver image columns are present
    assert "image_astar_elevation" in yaml
    assert "path_astar_elevation_free" in yaml
    assert "path_astar_elevation_obstacles" in yaml
    assert "path_astar_energy_free" in yaml
    assert "considered_area" in yaml


def test_export_and_upload_uses_chunk_size(tmp_path):
    """Test that export_and_upload loads config properly."""
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    _write_dummy_metadata(dataset_dir, 64)

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_instance.list_repo_files.return_value = []
                api_mock.return_value = api_instance
                with patch("passage.export._process_meta_for_prepare", side_effect=_mock_prepare_row):
                    with patch("passage.export._features", side_effect=_mock_features):
                        export_and_upload(
                            dataset_dirpath=dataset_dir,
                            calibrate_dirpath=calibrate_dir,
                            notebooks_dirpath=notebooks_dir,
                            resolutions=[64],
                            repo_id=None,
                            execute_notebooks=False,
                            skip_upload=True,  # Don't actually upload
                            num_cpus=0,
                        )


def test_export_and_upload_limits_sample_metadata_scan_to_sample_size(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    _write_many_dummy_metadata(dataset_dir, 64, count=10)

    processed_ids: list[str] = []

    def counting_prepare(*args, **kwargs):
        metadata_filepath = Path(args[-1] if args else kwargs["metadata_filepath"])
        processed_ids.append(metadata_filepath.stem)
        return _mock_prepare_row()

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export._process_meta_for_prepare", side_effect=counting_prepare):
            with patch("passage.export._features", side_effect=_mock_features):
                ok = export_and_upload(
                    dataset_dirpath=dataset_dir,
                    calibrate_dirpath=calibrate_dir,
                    notebooks_dirpath=notebooks_dir,
                    resolutions=[64],
                    repo_id="alice/repo",
                    execute_notebooks=False,
                    skip_upload=True,
                    num_cpus=0,
                )

    assert ok is True
    # The sample export should process only the configured 5 rows, then the full export
    # should process all 10 rows, for 15 total metadata conversions instead of 20.
    assert len(processed_ids) == 15


def test_export_and_upload_passes_num_cpus_to_notebook(tmp_path):
    """If num_cpus is provided, ensure the MAX_WORKERS parameter passed to notebook execution reflects it."""
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)

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


def test_export_and_upload_uses_explicit_calibration_filepath(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    custom_calibration = tmp_path / "custom" / "my-calibrate.json"
    custom_calibration.parent.mkdir(parents=True)
    custom_calibration.write_text(
        """
{
  "global_min": 10.0,
  "global_max": 20.0,
  "tile_stats": []
}
""".strip()
    )

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_mock.return_value = api_instance
                with patch("passage.export._execute_notebook_with_retries") as exec_mock:
                    ok = export_and_upload(
                        dataset_dirpath=dataset_dir,
                        calibrate_dirpath=calibrate_dir,
                        calibration_filepath=custom_calibration,
                        notebooks_dirpath=notebooks_dir,
                        resolutions=[64],
                        repo_id=None,
                        execute_notebooks=True,
                        skip_upload=True,
                    )

    assert ok is True
    copied_calibration = tmp_path / "export" / "calibration" / custom_calibration.name
    assert copied_calibration.exists()
    parameters_dict = None
    for arg in exec_mock.call_args[0]:
        if isinstance(arg, dict) and "CALIBRATE_FILEPATH" in arg:
            parameters_dict = arg
            break
    assert parameters_dict is not None
    assert parameters_dict.get("CALIBRATE_FILEPATH") == str(custom_calibration.resolve())

    assert ok is True


def test_process_meta_for_prepare_generates_missing_preview_images(tmp_path):
    split_dir = tmp_path / "train"
    metadata_dir = split_dir / "metadata" / "N000" / "E000"
    tensor_dir = split_dir / "tensors" / "N000" / "E000"
    path_free_dir = split_dir / "paths" / "astar_elevation" / "free" / "N000" / "E000"
    path_obstacles_dir = split_dir / "paths" / "astar_elevation" / "obstacles" / "N000" / "E000"
    image_dir = split_dir / "images" / "astar_elevation" / "N000" / "E000"
    metadata_dir.mkdir(parents=True)
    tensor_dir.mkdir(parents=True)
    path_free_dir.mkdir(parents=True)
    path_obstacles_dir.mkdir(parents=True)

    metadata_path = metadata_dir / "0000000000.yaml"
    metadata_path.write_text("index: 0\nobstacles:\n  enabled: true\n")
    save_tensor(np.zeros((64, 64, 3), dtype=np.float32), tensor_dir / "0000000000.npy.zst")
    (path_free_dir / "0000000000.csv").write_text("0,0\n1,1\n")
    (path_obstacles_dir / "0000000000.csv").write_text("0,1\n1,2\n")

    row = _process_meta_for_prepare(64, True, ["astar_elevation"], str(metadata_path))

    assert "image_astar_elevation" in row
    assert row["image_astar_elevation"]["bytes"]
    assert (image_dir / "0000000000.png").exists()


def test_process_meta_for_prepare_missing_tensor_still_fails(tmp_path):
    split_dir = tmp_path / "train"
    metadata_dir = split_dir / "metadata" / "N000" / "E000"
    metadata_dir.mkdir(parents=True)

    metadata_path = metadata_dir / "0000000000.yaml"
    metadata_path.write_text("index: 0\n")

    with pytest.raises(FileNotFoundError, match="Missing required tensor for export"):
        _process_meta_for_prepare(64, True, ["astar_elevation"], str(metadata_path))


def test_export_and_upload_skips_upload_when_requested(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)

    with patch("passage.export.HfApi") as api_mock:
        with patch("passage.export._reset_hf_upload_cache") as reset_mock:
            ok = export_and_upload(
                dataset_dirpath=dataset_dir,
                calibrate_dirpath=calibrate_dir,
                notebooks_dirpath=notebooks_dir,
                resolutions=[64],
                repo_id="alice/repo",
                execute_notebooks=False,
                skip_upload=True,
            )

    assert ok is True
    api_mock.assert_not_called()
    reset_mock.assert_not_called()


def test_reset_hf_upload_cache_removes_cache_dir(tmp_path):
    export_dir = tmp_path / "export"
    cache_dir = export_dir / ".cache" / "huggingface" / "upload"
    cache_dir.mkdir(parents=True)
    (cache_dir / "dummy.txt").write_text("x")

    _reset_hf_upload_cache(export_dir)

    assert not (export_dir / ".cache" / "huggingface").exists()


def test_export_and_upload_resets_cache_before_upload(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    export_dir = calibrate_dir.parent / "export"
    _write_dummy_metadata(dataset_dir, 64)

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_instance.list_repo_files.return_value = []
                api_mock.return_value = api_instance
                with patch("passage.export._reset_hf_upload_cache") as reset_mock:
                    with patch("passage.export._process_meta_for_prepare", side_effect=_mock_prepare_row):
                        with patch("passage.export._features", side_effect=_mock_features):
                            ok = export_and_upload(
                                dataset_dirpath=dataset_dir,
                                calibrate_dirpath=calibrate_dir,
                                notebooks_dirpath=notebooks_dir,
                                resolutions=[64],
                                execute_notebooks=False,
                                skip_upload=False,
                                num_cpus=0,
                            )

    assert ok is True
    reset_mock.assert_called_once_with(export_dir)
    api_instance.upload_large_folder.assert_called_once()
    upload_kwargs = api_instance.upload_large_folder.call_args.kwargs
    assert upload_kwargs["allow_patterns"] == [
        "README.md",
        "assets/**",
        "calibration/**",
        "notebooks/**",
        "64x64_samples/**",
        "64x64/**",
    ]


def test_export_and_upload_merges_remote_configs_into_readme(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    export_dir = calibrate_dir.parent / "export"
    _write_dummy_metadata(dataset_dir, 64)

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_instance.list_repo_files.return_value = [
                    "128x128/train/data-00000.parquet",
                    "128x128_samples/train/data-00000.parquet",
                ]
                api_mock.return_value = api_instance
                with patch("passage.export._process_meta_for_prepare", side_effect=_mock_prepare_row):
                    with patch("passage.export._features", side_effect=_mock_features):
                        ok = export_and_upload(
                            dataset_dirpath=dataset_dir,
                            calibrate_dirpath=calibrate_dir,
                            notebooks_dirpath=notebooks_dir,
                            resolutions=[64],
                            execute_notebooks=False,
                            skip_upload=False,
                            num_cpus=0,
                        )

    assert ok is True
    front_matter = _read_readme_front_matter(export_dir / "README.md")
    config_entries = {entry["config_name"]: entry for entry in front_matter["configs"]}
    assert set(config_entries) == {"64x64", "64x64_samples", "128x128", "128x128_samples"}
    assert config_entries["64x64_samples"]["default"] is True
    assert "default" not in config_entries["128x128_samples"]


def test_export_and_upload_cleans_only_selected_incremental_resolution(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    export_dir = calibrate_dir.parent / "export"
    _write_dummy_metadata(dataset_dir, 64)

    stale_full = export_dir / "64x64" / "stale.txt"
    stale_sample = export_dir / "64x64_samples" / "stale.txt"
    untouched_other = export_dir / "128x128" / "keep.txt"
    stale_full.parent.mkdir(parents=True, exist_ok=True)
    stale_sample.parent.mkdir(parents=True, exist_ok=True)
    untouched_other.parent.mkdir(parents=True, exist_ok=True)
    stale_full.write_text("stale-full")
    stale_sample.write_text("stale-sample")
    untouched_other.write_text("keep-me")

    with patch("passage.export._process_meta_for_prepare", side_effect=_mock_prepare_row):
        with patch("passage.export._features", side_effect=_mock_features):
            ok = export_and_upload(
                dataset_dirpath=dataset_dir,
                calibrate_dirpath=calibrate_dir,
                notebooks_dirpath=notebooks_dir,
                resolutions=[64],
                repo_id="alice/repo",
                execute_notebooks=False,
                skip_upload=True,
                num_cpus=0,
            )

    assert ok is True
    assert not stale_full.exists()
    assert not stale_sample.exists()
    assert (export_dir / "64x64" / "train" / "data-00000.parquet").exists()
    assert (export_dir / "64x64_samples" / "train" / "data-00000.parquet").exists()
    assert untouched_other.exists()


def test_export_and_upload_full_export_keeps_unrestricted_upload(tmp_path):
    dataset_dir, notebooks_dir, calibrate_dir = _setup_minimal_export_tree(tmp_path)
    _write_dummy_metadata(dataset_dir, 64)

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi") as api_mock:
                api_instance = MagicMock()
                api_instance.list_repo_files.return_value = []
                api_mock.return_value = api_instance
                with patch("passage.export._process_meta_for_prepare", side_effect=_mock_prepare_row):
                    with patch("passage.export._features", side_effect=_mock_features):
                        ok = export_and_upload(
                            dataset_dirpath=dataset_dir,
                            calibrate_dirpath=calibrate_dir,
                            notebooks_dirpath=notebooks_dir,
                            resolutions=None,
                            execute_notebooks=False,
                            skip_upload=False,
                            num_cpus=0,
                        )

    assert ok is True
    assert api_instance.upload_large_folder.call_args.kwargs["allow_patterns"] is None
