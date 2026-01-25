"""Tests for CLI argument parsing."""

import sys
from unittest.mock import patch

import pytest

# =============================================================================
# Unit tests for CLI argument parsing
# =============================================================================


class TestDownloadCLI:
    """Tests for download subcommand CLI."""

    @pytest.mark.unit
    def test_download_defaults(self):
        """Test download subcommand default arguments."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "download"]):
            with patch("passage.run_download") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "outputs/download"
                assert kwargs.get("resume") is True
                assert kwargs.get("num_cpus") is None

    @pytest.mark.unit
    def test_download_custom_args(self):
        """Test download subcommand with custom arguments."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "download",
                "--outputs-dirpath",
                "/custom/path",
                "--no-resume",
                "--num-cpus",
                "4",
            ],
        ):
            with patch("passage.run_download") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "/custom/path"
                assert kwargs.get("resume") is False
                assert kwargs.get("num_cpus") == 4


class TestCalibrateCLI:
    """Tests for calibrate subcommand CLI."""

    @pytest.mark.unit
    def test_calibrate_defaults(self):
        """Test calibrate subcommand default arguments."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "calibrate"]):
            with patch("passage.run_calibrate") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "outputs/calibrate"
                assert args[1] == "outputs/download/tiles"
                assert kwargs.get("resume") is True
                assert kwargs.get("num_cpus") is None

    @pytest.mark.unit
    def test_calibrate_custom_args(self):
        """Test calibrate subcommand with custom arguments."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "calibrate",
                "--outputs-dirpath",
                "/custom/calibrate",
                "--tiles-dirpath",
                "/custom/tiles",
                "--no-resume",
                "--num-cpus",
                "8",
            ],
        ):
            with patch("passage.run_calibrate") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "/custom/calibrate"
                assert args[1] == "/custom/tiles"
                assert kwargs.get("resume") is False
                assert kwargs.get("overwrite") is True
                assert kwargs.get("num_cpus") == 8


class TestGenerateCLI:
    """Tests for generate subcommand CLI."""

    @pytest.mark.unit
    def test_generate_defaults(self):
        """Test generate subcommand default arguments."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "generate"]):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "outputs/dataset"
                assert args[1] == "config/config.yaml"
                assert args[2] == "outputs/download/tiles"
                assert kwargs.get("resolution") is None
                assert kwargs.get("num_samples") is None
                assert kwargs.get("num_cpus") is None
                assert kwargs.get("disable_images") is False
                assert kwargs.get("force") is False

    @pytest.mark.unit
    def test_generate_single_resolution(self):
        """Test generate subcommand with single resolution."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "generate",
                "--resolution",
                "256",
            ],
        ):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs.get("resolution") == [256]

    @pytest.mark.unit
    def test_generate_multiple_resolutions(self):
        """Test generate subcommand with multiple resolutions."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "generate",
                "--resolution",
                "64",
                "128",
                "256",
            ],
        ):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs.get("resolution") == [64, 128, 256]

    @pytest.mark.unit
    def test_generate_no_images_flag(self):
        """Test generate subcommand with --no-images flag."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "generate",
                "--no-images",
            ],
        ):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs.get("disable_images") is True

    @pytest.mark.unit
    def test_generate_force_flag(self):
        """Test generate subcommand with --force flag."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "generate",
                "--force",
            ],
        ):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                _, kwargs = mock_run.call_args
                assert kwargs.get("force") is True

    @pytest.mark.unit
    def test_generate_all_custom_args(self):
        """Test generate subcommand with all custom arguments."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "generate",
                "--outputs-dirpath",
                "/custom/dataset",
                "--config",
                "/custom/config.yaml",
                "--tiles-dirpath",
                "/custom/tiles",
                "--resolution",
                "512",
                "--num-samples",
                "100",
                "--num-cpus",
                "4",
                "--no-images",
                "--force",
            ],
        ):
            with patch("passage.run_generate") as mock_run:
                main()

                mock_run.assert_called_once()
                args, kwargs = mock_run.call_args
                assert args[0] == "/custom/dataset"
                assert args[1] == "/custom/config.yaml"
                assert args[2] == "/custom/tiles"
                assert kwargs.get("resolution") == [512]
                assert kwargs.get("num_samples") == 100
                assert kwargs.get("num_cpus") == 4
                assert kwargs.get("disable_images") is True
                assert kwargs.get("force") is True


class TestMainCLI:
    """Tests for main CLI entry point."""

    @pytest.mark.unit
    def test_main_no_command(self, capsys):
        """Test main with no command shows greeting."""
        from passage import main

        with patch.object(sys, "argv", ["passage"]):
            main()

        # Should log the hello message (captured by loguru, not capsys)
        # Just verify it doesn't crash

    @pytest.mark.unit
    def test_main_help(self, capsys):
        """Test main --help exits cleanly."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

            assert exc_info.value.code == 0

    @pytest.mark.unit
    def test_export_logs_env_credentials_message(self):
        """When --repo-id is omitted, the CLI should inform the user that it will read
        HF credentials from .env and .env.secret (or custom paths)."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--no-exec-notebooks", "--no-upload"]):
            with patch("passage.logger") as mock_logger:
                with patch("passage.export_and_upload") as mock_export:
                    main()

                    # Ensure export was called
                    mock_export.assert_called_once()

                    # Confirm an info log was emitted mentioning HF credentials
                    info_calls = [c for c in mock_logger.info.call_args_list]
                    assert any("Using Hugging Face credentials" in str(c.args[0]) for c in info_calls)

    @pytest.mark.unit
    def test_export_uses_env_repo_id(self):
        """When --repo-id is not provided, the export subcommand should leave repo_id as None so
        downstream code can read username/repo from .env."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--no-exec-notebooks", "--no-upload"]):
            with patch("passage.export_and_upload") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("repo_id") is None

    @pytest.mark.unit
    def test_export_passes_explicit_repo_id(self):
        """If --repo-id is provided it should be passed through unchanged."""
        from passage import main

        with patch.object(
            sys, "argv", ["passage", "export", "--repo-id", "alice/repo", "--no-exec-notebooks", "--no-upload"]
        ):
            with patch("passage.export_and_upload") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("repo_id") == "alice/repo"
