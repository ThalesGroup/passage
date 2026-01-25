from unittest.mock import patch

from passage.prepare import _generate_huggingface_yaml_header, upload_to_huggingface


def test_generate_hf_yaml_header_configs_names():
    yaml = _generate_huggingface_yaml_header("user/repo", [64, 128], ["train", "validation"])
    assert 'config_name: "64x64"' in yaml
    assert 'path: "64/train/data-*.parquet"' in yaml
    assert 'config_name: "128x128"' in yaml
    assert 'path: "128/validation/data-*.parquet"' in yaml


def test_upload_skips_parquet_and_pushes_metadata(tmp_path):
    """Ensure upload_to_huggingface uploads non-parquet files first and delegates
    parquet uploads to dataset metadata push (no double upload of parquets)."""
    called = {}

    class DummyApi:
        def create_repo(self, *args, **kwargs):
            called["create_repo"] = True

        def list_repo_files(self, *args, **kwargs):
            return []

        def upload_folder(self, *args, **kwargs):
            called["upload_folder_args"] = kwargs

        def upload_file(self, *args, **kwargs):
            called["upload_file"] = True

    def fake_load_credentials(*args, **kwargs):
        return ("me", "repo", "token")

    # Patch _load_hf_credentials and HfApi in the module to use DummyApi
    with patch("passage.prepare._load_hf_credentials", fake_load_credentials):
        with patch("passage.prepare.HfApi", return_value=DummyApi()):
            with patch("passage.prepare._push_dataset_metadata_to_hub", return_value=True) as mock_push:
                with patch("passage.prepare.login", return_value=None):
                    ok = upload_to_huggingface(
                        export_dirpath=str(tmp_path),
                        env_path=None,
                        secret_path=None,
                        repo_id="me/repo",
                        private=False,
                        create_repo=True,
                    )
                assert ok is True
                # upload_folder should have been called with ignore_patterns containing '*.parquet'
                assert "upload_folder_args" in called
                ignore_patterns = called["upload_folder_args"].get("ignore_patterns", [])
                assert any(pat == "*.parquet" for pat in ignore_patterns)
                mock_push.assert_called_once()
