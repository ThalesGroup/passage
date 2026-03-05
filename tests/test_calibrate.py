"""Tests for the calibrate module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import passage.calibrate as cal


# =============================================================================
# Unit tests for _process_tile_worker
# =============================================================================


class TestProcessTileWorker:
    """Tests for _process_tile_worker function."""

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate.rasterio.open")
    def test_process_tile_worker_valid_data(self, mock_rasterio_open):
        """Test processing a tile with valid data."""
        # Create mock data
        mock_data = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = mock_data
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = mock_src

        tile_file = Path("N045E010_DSM.tif")
        filename, min_val, max_val, err = cal._process_tile_worker(tile_file)

        assert filename == "N045E010_DSM.tif"
        assert min_val == 100.0
        assert max_val == 400.0
        assert err is None

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate.rasterio.open")
    def test_process_tile_worker_no_valid_data(self, mock_rasterio_open):
        """Test processing a tile with only no-data values."""
        # Create mock data with all no-data values (< -1000)
        mock_data = np.array([[-9999.0, -9999.0], [-9999.0, -9999.0]], dtype=np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = mock_data
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)
        mock_rasterio_open.return_value = mock_src

        tile_file = Path("N000E000_DSM.tif")
        filename, min_val, max_val, err = cal._process_tile_worker(tile_file)

        assert filename == "N000E000_DSM.tif"
        assert min_val is None
        assert max_val is None
        assert err == "no_valid_data"

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate.rasterio.open")
    def test_process_tile_worker_exception(self, mock_rasterio_open):
        """Test handling of exceptions during tile processing."""
        mock_rasterio_open.side_effect = Exception("File not found")

        tile_file = Path("missing_DSM.tif")
        filename, min_val, max_val, err = cal._process_tile_worker(tile_file)

        assert filename == "missing_DSM.tif"
        assert min_val is None
        assert max_val is None
        assert "File not found" in err


# =============================================================================
# Unit tests for run_calibrate
# =============================================================================


class TestRunCalibrate:
    """Tests for run_calibrate function."""

    @pytest.mark.unit
    @pytest.mark.calibrate
    def test_run_calibrate_no_tiles(self, tmp_path):
        """Test run_calibrate with no tiles found."""
        outputs_dir = tmp_path / "calibrate"
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)

        # Should not create calibrate.json if no tiles found
        calibrate_json = outputs_dir / "calibrate.json"
        assert not calibrate_json.exists()

    @pytest.mark.unit
    @pytest.mark.calibrate
    def test_run_calibrate_missing_tiles_dir(self, tmp_path):
        """Test run_calibrate with missing tiles directory."""
        outputs_dir = tmp_path / "calibrate"
        tiles_dir = tmp_path / "nonexistent_tiles"

        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)

        # Should not create any output
        assert not outputs_dir.exists() or not (outputs_dir / "calibrate.json").exists()

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate._process_tile_worker")
    def test_run_calibrate_basic(self, mock_worker, tmp_path):
        """Test basic calibrate run with mocked worker."""
        # Setup mock
        mock_worker.return_value = ("N045E010_DSM.tif", 100.0, 2000.0, None)

        outputs_dir = tmp_path / "calibrate"
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create a dummy tile file
        dummy_tile = tiles_dir / "N045E010_DSM.tif"
        dummy_tile.touch()

        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)

        # Should create calibrate.json
        calibrate_json = outputs_dir / "calibrate.json"
        assert calibrate_json.exists()

        with open(calibrate_json) as f:
            results = json.load(f)

        assert results["global_min"] == 100.0
        assert results["global_max"] == 2000.0
        assert results["tiles_processed"] == 1

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate._process_tile_worker")
    def test_run_calibrate_resume(self, mock_worker, tmp_path):
        """Test resuming calibration from existing JSON."""
        outputs_dir = tmp_path / "calibrate"
        outputs_dir.mkdir()
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create existing calibrate.json
        existing_data = {
            "global_min": 0.0,
            "global_max": 1000.0,
            "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0.0, "max": 1000.0, "lat": 0, "lon": 0}],
        }
        calibrate_json = outputs_dir / "calibrate.json"
        with open(calibrate_json, "w") as f:
            json.dump(existing_data, f)

        # Create tile files - one already processed, one new
        (tiles_dir / "N000E000_DSM.tif").touch()
        (tiles_dir / "N045E010_DSM.tif").touch()

        # Mock returns for new tile only
        mock_worker.return_value = ("N045E010_DSM.tif", 500.0, 2000.0, None)

        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=True, num_cpus=0)

        with open(calibrate_json) as f:
            results = json.load(f)

        # Should have merged results
        assert results["global_min"] == 0.0  # From existing
        assert results["global_max"] == 2000.0  # From new tile
        assert len(results["tile_stats"]) == 2

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate._process_tile_worker")
    def test_run_calibrate_failed_tiles(self, mock_worker, tmp_path):
        """Test handling of failed tiles."""
        # First call succeeds, second fails
        mock_worker.side_effect = [
            ("N000E000_DSM.tif", 100.0, 500.0, None),
            ("N045E010_DSM.tif", None, None, "read error"),
        ]

        outputs_dir = tmp_path / "calibrate"
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create dummy tile files
        (tiles_dir / "N000E000_DSM.tif").touch()
        (tiles_dir / "N045E010_DSM.tif").touch()

        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)

        calibrate_json = outputs_dir / "calibrate.json"
        with open(calibrate_json) as f:
            results = json.load(f)

        assert results["tiles_processed"] == 1
        assert results["tiles_failed"] == 1
        assert "N045E010_DSM.tif" in results["failed_tiles"]

    @pytest.mark.unit
    @pytest.mark.calibrate
    def test_run_calibrate_do_not_override_existing(self, tmp_path):
        """If calibrate.json exists and resume=False, do not overwrite it unless overwrite=True."""
        outputs_dir = tmp_path / "calibrate"
        outputs_dir.mkdir()
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create existing calibrate.json
        existing_data = {
            "global_min": -100.0,
            "global_max": 100.0,
            "tiles_processed": 0,
            "tile_stats": [],
        }
        calibrate_json = outputs_dir / "calibrate.json"
        with open(calibrate_json, "w") as f:
            json.dump(existing_data, f)

        # Create a dummy tile file that would normally be processed
        (tiles_dir / "N045E010_DSM.tif").touch()

        # Run calibrate with resume=False (default) and without overwrite
        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, num_cpus=0)

        # Check file was not overwritten
        with open(calibrate_json) as f:
            results = json.load(f)
        assert results["global_min"] == -100.0
        assert results["global_max"] == 100.0

    @pytest.mark.unit
    @pytest.mark.calibrate
    @patch("passage.calibrate._process_tile_worker")
    def test_run_calibrate_overwrite_when_requested(self, mock_worker, tmp_path):
        """If overwrite=True, existing calibrate.json is recomputed and overwritten."""
        mock_worker.return_value = ("N045E010_DSM.tif", 100.0, 2000.0, None)

        outputs_dir = tmp_path / "calibrate"
        outputs_dir.mkdir()
        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create existing calibrate.json
        existing_data = {
            "global_min": -100.0,
            "global_max": 100.0,
            "tiles_processed": 0,
            "tile_stats": [],
        }
        calibrate_json = outputs_dir / "calibrate.json"
        with open(calibrate_json, "w") as f:
            json.dump(existing_data, f)

        # Create a dummy tile file that will be processed
        (tiles_dir / "N045E010_DSM.tif").touch()

        # Run calibrate with overwrite=True
        cal.run_calibrate(str(outputs_dir), str(tiles_dir), resume=False, overwrite=True, num_cpus=0)

        with open(calibrate_json) as f:
            results = json.load(f)

        assert results["global_min"] == 100.0
        assert results["global_max"] == 2000.0
        assert results["tiles_processed"] == 1
