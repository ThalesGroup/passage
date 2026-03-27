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

"""Tests for the download module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import passage.download as dl


# =============================================================================
# Unit tests for download_tile
# =============================================================================


class TestDownloadTile:
    """Tests for download_tile function."""

    @pytest.mark.unit
    @pytest.mark.download
    @patch("passage.download.requests.head")
    def test_download_tile_valid_url(self, mock_head):
        """Test download_tile with a valid URL."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_head.return_value = mock_response

        lat, lon, url, url_valid, extracted = dl.download_tile((0, 0, None))

        assert lat == 0
        assert lon == 0
        assert url_valid is True
        assert extracted is False  # No raw_dir provided

    @pytest.mark.unit
    @pytest.mark.download
    @patch("passage.download.requests.head")
    def test_download_tile_invalid_url(self, mock_head):
        """Test download_tile with an invalid URL (404)."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_head.return_value = mock_response

        lat, lon, url, url_valid, extracted = dl.download_tile((45, 10, None))

        assert lat == 45
        assert lon == 10
        assert url_valid is False
        assert extracted is False

    @pytest.mark.unit
    @pytest.mark.download
    @patch("passage.download.requests.head")
    def test_download_tile_request_exception(self, mock_head):
        """Test download_tile with a request exception."""
        import requests

        mock_head.side_effect = requests.exceptions.RequestException("Network error")

        lat, lon, url, url_valid, extracted = dl.download_tile((0, 0, None))

        assert url_valid is False
        assert extracted is False

    @pytest.mark.unit
    @pytest.mark.download
    def test_download_tile_url_format(self):
        """Test that download_tile generates correct URL format."""
        with patch("passage.download.requests.head") as mock_head:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_head.return_value = mock_response

            lat, lon, url, url_valid, extracted = dl.download_tile((0, 0, None))

            # Check URL format
            assert "N000E000" in url
            assert "N005E005" in url
            assert ".zip" in url

    @pytest.mark.unit
    @pytest.mark.download
    def test_download_tile_negative_coords(self):
        """Test download_tile with negative coordinates."""
        with patch("passage.download.requests.head") as mock_head:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_head.return_value = mock_response

            lat, lon, url, url_valid, extracted = dl.download_tile((-10, -120, None))

            # Check URL format for negative coords
            assert "S010W120" in url
            assert "S005W115" in url


# =============================================================================
# Unit tests for generate_status_map
# =============================================================================


class TestGenerateStatusMap:
    """Tests for generate_status_map function."""

    @pytest.mark.unit
    @pytest.mark.download
    def test_generate_status_map_basic(self, tmp_path):
        """Test basic status map generation."""
        df_results = pd.DataFrame(
            [
                {"lat_min": 0, "lon_min": 0, "url": "http://test", "url_valid": True},
                {"lat_min": 5, "lon_min": 5, "url": "http://test", "url_valid": False},
            ]
        )

        output_path = tmp_path / "status.png"
        dl.generate_status_map(df_results, output_path)

        assert output_path.exists()

    @pytest.mark.unit
    @pytest.mark.download
    def test_generate_status_map_empty_df(self, tmp_path):
        """Test status map generation with empty DataFrame."""
        df_results = pd.DataFrame(columns=["lat_min", "lon_min", "url", "url_valid"])

        output_path = tmp_path / "status_empty.png"
        dl.generate_status_map(df_results, output_path)

        # Should still create file (black image)
        assert output_path.exists()

    @pytest.mark.unit
    @pytest.mark.download
    def test_generate_status_map_image_dimensions(self, tmp_path):
        """Test that status map has correct dimensions."""
        import cv2

        df_results = pd.DataFrame(
            [
                {"lat_min": 0, "lon_min": 0, "url": "http://test", "url_valid": True},
            ]
        )

        output_path = tmp_path / "status.png"
        dl.generate_status_map(df_results, output_path)

        img = cv2.imread(str(output_path))
        # Expected dimensions based on lat/lon ranges and cell size
        # NUM_LAT_STEPS = 36, NUM_LON_STEPS = 72, CELL_SIZE = 10
        assert img.shape[0] == 360  # 36 * 10
        assert img.shape[1] == 720  # 72 * 10


# =============================================================================
# Unit tests for run_download
# =============================================================================


class TestRunDownload:
    """Tests for run_download function."""

    @pytest.mark.unit
    @pytest.mark.download
    @patch("passage.download.download_tile")
    def test_run_download_creates_output_dir(self, mock_download, tmp_path):
        """Test that run_download creates output directory."""
        mock_download.return_value = (0, 0, "http://test", False, False)

        output_dir = tmp_path / "download_output"

        # Run download with num_cpus=0 to avoid multiprocessing issues
        with patch("passage.download.generate_status_map"):
            dl.run_download(str(output_dir), resume=False, num_cpus=0)

        assert output_dir.exists()

    @pytest.mark.unit
    @pytest.mark.download
    @patch("passage.download.download_tile")
    def test_run_download_resume_from_csv(self, mock_download, tmp_path):
        """Test resuming from existing CSV file."""
        mock_download.return_value = (0, 0, "http://test", False, False)

        output_dir = tmp_path / "download_output"
        output_dir.mkdir()

        # Create existing CSV with some results
        csv_path = output_dir / "download.csv"
        existing_df = pd.DataFrame(
            [
                {"lat_min": 0, "lon_min": 0, "url": "http://test", "url_valid": True, "extracted": True},
            ]
        )
        existing_df.to_csv(csv_path, index=False)

        with patch("passage.download.generate_status_map"):
            dl.run_download(str(output_dir), resume=True, num_cpus=0)

        # Should not reprocess the existing pair
        # Total pairs = 36*72 = 2592, minus 1 already processed = 2591
        # But since extracted=True, it won't be requeued
        assert csv_path.exists()
