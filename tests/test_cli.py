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
                assert kwargs.get("calibration_filepath") == "outputs/calibrate/calibrate.json"
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
                "--calibration-filepath",
                "/custom/calibrate.json",
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
                assert kwargs.get("calibration_filepath") == "/custom/calibrate.json"
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
    def test_upload_logs_token_message(self):
        """When upload runs, the CLI should log the token source and default target repo."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "upload"]):
            with patch("passage.logger") as mock_logger:
                with patch("passage.upload_export") as mock_upload:
                    main()

                    mock_upload.assert_called_once()

                    info_calls = [c for c in mock_logger.info.call_args_list]
                    assert any("Using Hugging Face token" in str(c.args[0]) for c in info_calls)
                    assert any("thalesgroup/passage" in str(c.args[0]) for c in info_calls)

    @pytest.mark.unit
    def test_export_leaves_repo_id_none_for_default(self):
        """When --repo-id is omitted, the export subcommand should let the implementation apply its default."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export"]):
            with patch("passage.prepare_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("repo_id") is None

    @pytest.mark.unit
    def test_export_passes_explicit_repo_id(self):
        """If --repo-id is provided to export it should be passed through unchanged."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--repo-id", "alice/repo"]):
            with patch("passage.prepare_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("repo_id") == "alice/repo"

    @pytest.mark.unit
    def test_export_passes_anonymous_flag(self):
        """If --anonymous is provided to export it should be passed through unchanged."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--anonymous"]):
            with patch("passage.prepare_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("anonymous") is True

    @pytest.mark.unit
    def test_compress_passes_num_cpus_cli(self):
        """The compress subcommand should accept --num-cpus and pass it through."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "compress", "--num-cpus", "4"]):
            with patch("passage.compress_release") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("num_cpus") == 4

    @pytest.mark.unit
    def test_export_passes_config_path(self):
        """The export subcommand should pass the explicit config path through."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--config", "/custom/config.yaml"]):
            with patch("passage.prepare_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("config_filepath") == "/custom/config.yaml"

    @pytest.mark.unit
    def test_upload_passes_calibration_filepath(self):
        """The upload subcommand should pass the explicit calibration JSON path through."""
        from passage import main

        with patch.object(
            sys,
            "argv",
            [
                "passage",
                "upload",
                "--calibration-filepath",
                "/custom/calibrate.json",
            ],
        ):
            with patch("passage.upload_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("calibration_filepath") == "/custom/calibrate.json"

    @pytest.mark.unit
    def test_export_passes_explicit_resolutions(self):
        """The export subcommand should pass explicit resolution filters through."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "export", "--resolution", "64", "128"]):
            with patch("passage.prepare_export") as mock_export:
                main()

                mock_export.assert_called_once()
                _, kwargs = mock_export.call_args
                assert kwargs.get("resolutions") == [64, 128]

    @pytest.mark.unit
    def test_upload_passes_explicit_repo_id(self):
        """The upload subcommand should pass the explicit repo id through unchanged."""
        from passage import main

        with patch.object(sys, "argv", ["passage", "upload", "--repo-id", "alice/repo"]):
            with patch("passage.upload_export") as mock_upload:
                main()

                mock_upload.assert_called_once()
                _, kwargs = mock_upload.call_args
                assert kwargs.get("repo_id") == "alice/repo"
