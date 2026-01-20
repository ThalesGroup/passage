"""Tests for export module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


class TestExportModule:
    """Tests for export module functionality."""

    def test_import_export_module(self):
        """Test that export module can be imported."""
        from passage.export import (
            export_calibration,
            export_notebooks,
            export_resolution,
            run_export,
        )

        # Module should be importable regardless of HF availability
        assert callable(run_export)
        assert callable(export_calibration)
        assert callable(export_resolution)
        assert callable(export_notebooks)

    def test_generate_calibration_card(self):
        """Test dataset card generation for calibration data."""
        from passage.export import _generate_dataset_card

        calibrate_data = {
            "global_min": -100.0,
            "global_max": 5000.0,
            "tile_stats": [{"lat": 45, "lon": 10}] * 100,
        }

        card = _generate_dataset_card(
            repo_id="user/passage-calibration",
            resolution=None,
            calibrate_data=calibrate_data,
            summary_data=None,
            is_calibration=True,
        )

        assert "PASSAGE Calibration Data" in card
        assert "-100.0" in card
        assert "5000.0" in card
        assert "100 tiles" in card

    def test_generate_resolution_card(self):
        """Test dataset card generation for resolution dataset."""
        from passage.export import _generate_dataset_card

        calibrate_data = {"global_min": -100.0, "global_max": 5000.0}
        summary_data = {"num_samples": 1000}

        card = _generate_dataset_card(
            repo_id="user/passage-256",
            resolution=256,
            calibrate_data=calibrate_data,
            summary_data=summary_data,
            is_calibration=False,
        )

        assert "256x256" in card
        assert "1000" in card
        assert "-100.0" in card
        assert "5000.0" in card

    def test_check_hf_available_when_not_installed(self):
        """Test HF availability check when not installed."""
        from passage.export import HF_AVAILABLE, _check_hf_available

        # This will fail if HF is not installed (expected in test env)
        result = _check_hf_available()
        assert result == HF_AVAILABLE

    @patch("passage.export.HF_AVAILABLE", False)
    def test_export_fails_without_hf(self):
        """Test that export fails gracefully without huggingface_hub."""
        from passage.export import export_calibration, export_notebooks, export_resolution

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create minimal calibration data
            calibrate_dir = Path(tmpdir) / "calibrate"
            calibrate_dir.mkdir()
            (calibrate_dir / "calibrate.json").write_text('{"global_min": 0, "global_max": 100, "tile_stats": []}')

            result = export_calibration(calibrate_dir, "user/test")
            assert result is False

            result = export_resolution(tmpdir, 256, "user/test")
            assert result is False

            result = export_notebooks(tmpdir, "user/test")
            assert result is False


class TestExportCLI:
    """Tests for export CLI."""

    def test_export_parser_exists(self):
        """Test that export subcommand is registered."""
        from unittest.mock import patch

        # Parse with export command
        from passage import main

        with patch("sys.argv", ["passage", "export", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with 0
            assert exc_info.value.code == 0

    def test_export_requires_repo_prefix(self):
        """Test that export requires repo_prefix argument."""
        from unittest.mock import patch

        from passage import main

        with patch("sys.argv", ["passage", "export"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # Missing required argument exits with error
            assert exc_info.value.code != 0


class TestPrepareExport:
    """Tests for prepare_export functionality."""

    def test_prepare_export_creates_structure(self):
        """Test that prepare_export creates the correct directory structure."""
        from passage.export import prepare_export

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # Create minimal dataset structure
            dataset_dir = tmpdir / "dataset"
            res_dir = dataset_dir / "256"
            res_dir.mkdir(parents=True)
            (res_dir / "tensors").mkdir()
            (res_dir / "metadata").mkdir()
            (res_dir / "summary.json").write_text('{"num_samples": 10}')

            # Create calibration data
            calibrate_dir = tmpdir / "calibrate"
            calibrate_dir.mkdir()
            (calibrate_dir / "calibrate.json").write_text('{"global_min": 0, "global_max": 100, "tile_stats": []}')

            # Create notebooks dir
            notebooks_dir = tmpdir / "notebooks"
            notebooks_dir.mkdir()
            (notebooks_dir / "test.ipynb").write_text(
                '{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
            )

            # Create export
            export_dir = tmpdir / "export"
            result = prepare_export(
                export_dirpath=export_dir,
                dataset_dirpath=dataset_dir,
                calibrate_dirpath=calibrate_dir,
                notebooks_dirpath=notebooks_dir,
                resolutions=[256],
                execute_notebooks=False,  # Skip notebook execution in test
            )

            assert result is True
            assert (export_dir / "README.md").exists()
            assert (export_dir / "calibration" / "README.md").exists()
            assert (export_dir / "calibration" / "calibrate.json").exists()
            assert (export_dir / "docs" / "notebooks" / "test.ipynb").exists()
            assert (export_dir / "256" / "README.md").exists()
            # Check symlink
            data_link = export_dir / "256" / "data"
            assert data_link.is_symlink()

    def test_prepare_export_cli_flag(self):
        """Test that --prepare flag is recognized."""
        from unittest.mock import patch

        from passage import main

        with patch("sys.argv", ["passage", "export", "--prepare", "--help"]):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # --help exits with 0
            assert exc_info.value.code == 0
