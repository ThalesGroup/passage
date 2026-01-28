from unittest.mock import MagicMock, patch

from passage.export import _generate_huggingface_yaml_header, export_and_upload


def test_generate_hf_yaml_header_configs_names():
    yaml = _generate_huggingface_yaml_header("user/repo", [64, 128], ["train", "validation"])
    assert 'config_name: "64x64"' in yaml
    assert 'path: "64x64/train"' in yaml
    assert 'config_name: "128x128"' in yaml
    assert 'path: "128x128/validation"' in yaml


def test_export_and_upload_uses_chunk_size(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "64" / "train"
    split_dir.mkdir(parents=True)

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "config.yaml").write_text(
        """
export:
  chunk_size:
    64: 123
"""
    )

    with patch("passage.export._load_hf_credentials", return_value=("me", "repo", "token")):
        with patch("passage.export.login", return_value=None):
            with patch("passage.export.HfApi", return_value=MagicMock()) as api_ctor:
                with patch("passage.export._load_rows_for_split", return_value=[{"id": "1", "tensor": b"t"}]):
                    with patch("passage.export._rows_to_dataset", return_value=object()):
                        with patch("passage.export._push_resolution") as mock_push:
                            ok = export_and_upload(
                                dataset_dirpath=dataset_dir,
                                calibrate_dirpath=tmp_path / "calibrate",
                                notebooks_dirpath=notebooks_dir,
                                resolutions=None,
                                repo_id=None,
                                execute_notebooks=False,
                            )

    assert ok is True
    api_ctor.return_value.create_repo.assert_called_once()
    mock_push.assert_called_once()
    _, kwargs = mock_push.call_args
    assert kwargs["resolution"] == 64
    assert kwargs["chunk_size"] == 123


def test_export_and_upload_skips_upload_when_requested(tmp_path):
    dataset_dir = tmp_path / "dataset"
    split_dir = dataset_dir / "64" / "train"
    split_dir.mkdir(parents=True)

    notebooks_dir = tmp_path / "notebooks"
    notebooks_dir.mkdir()
    (tmp_path / "config").mkdir()

    # If credentials are loaded when skip_upload is True, fail the test
    with patch("passage.export._load_hf_credentials", side_effect=AssertionError("should not load creds")):
        ok = export_and_upload(
            dataset_dirpath=dataset_dir,
            calibrate_dirpath=tmp_path / "calibrate",
            notebooks_dirpath=notebooks_dir,
            resolutions=None,
            execute_notebooks=False,
            skip_upload=True,
        )

    assert ok is True
