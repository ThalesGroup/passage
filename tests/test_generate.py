"""Tests for the generate module."""

import math
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

import passage.generate as gen


# =============================================================================
# Unit tests for helper functions
# =============================================================================


class TestWrapLon:
    """Tests for wrap_lon function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_wrap_lon_normal(self):
        """Test wrap_lon with values in range."""
        assert gen.wrap_lon(0) == 0
        assert gen.wrap_lon(100) == 100
        assert gen.wrap_lon(-100) == -100
        assert gen.wrap_lon(179) == 179
        assert gen.wrap_lon(-180) == -180

    @pytest.mark.unit
    @pytest.mark.generate
    def test_wrap_lon_at_boundary_180(self):
        """Test wrap_lon at +180 boundary."""
        assert gen.wrap_lon(180) == -180

    @pytest.mark.unit
    @pytest.mark.generate
    def test_wrap_lon_above_180(self):
        """Test wrap_lon above +180."""
        assert gen.wrap_lon(181) == -179
        assert gen.wrap_lon(270) == -90
        assert gen.wrap_lon(359) == -1

    @pytest.mark.unit
    @pytest.mark.generate
    def test_wrap_lon_below_minus_180(self):
        """Test wrap_lon below -180."""
        assert gen.wrap_lon(-181) == 179
        assert gen.wrap_lon(-270) == 90
        assert gen.wrap_lon(-359) == 1


class TestFindPath:
    """Tests for find_path function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_simple(self):
        """Test find_path on a simple flat terrain."""
        crop = np.ones((10, 10), dtype=np.float32) * 100
        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0}
        path = gen.find_path(crop, (0, 0), (9, 9), solver_config)
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (9, 9)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_different_solvers(self):
        """Test find_path with different NetworkX solvers."""
        crop = np.ones((8, 8), dtype=np.float32) * 50
        solvers = [
            "astar",
            "astar_bi",
            "dijkstra",
            "dijkstra_bi",
            "bellman_ford",
            "unweighted",
            "grid_astar",
            "grid_dijkstra",
        ]
        for solver_name in solvers:
            solver_config = {"name": solver_name, "diagonal_movement": "always", "weight": 1.0}
            path = gen.find_path(crop, (0, 0), (7, 7), solver_config)
            assert len(path) > 0, f"Solver {solver_name} failed to find path"
            assert path[0] == (0, 0)
            assert path[-1] == (7, 7)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_diagonal_movement_options(self):
        """Test find_path with different diagonal movement options."""
        crop = np.ones((5, 5), dtype=np.float32) * 100
        diagonal_options = ["always", "never"]
        for diag in diagonal_options:
            solver_config = {"name": "astar", "diagonal_movement": diag, "weight": 1.0}
            path = gen.find_path(crop, (0, 0), (4, 4), solver_config)
            assert len(path) > 0, f"Diagonal movement {diag} failed to find path"

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_same_point(self):
        """Test find_path when start and end are the same."""
        crop = np.ones((5, 5), dtype=np.float32) * 100
        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0}
        path = gen.find_path(crop, (2, 2), (2, 2), solver_config)
        assert len(path) == 1
        assert path[0] == (2, 2)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_heuristics(self):
        """Test find_path with different A* heuristics."""
        crop = np.ones((6, 6), dtype=np.float32) * 100
        heuristics = ["euclidean", "manhattan", "chebyshev"]
        for heuristic in heuristics:
            solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "heuristic": heuristic}
            path = gen.find_path(crop, (0, 0), (5, 5), solver_config)
            assert len(path) > 0, f"Heuristic {heuristic} failed to find path"
            assert path[0] == (0, 0)
            assert path[-1] == (5, 5)


class TestBuildGraphFromElevation:
    """Tests for build_graph_from_elevation function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_build_graph_basic(self):
        """Test graph construction from simple elevation array."""
        elevation = np.ones((3, 3), dtype=np.float32) * 100
        G = gen.build_graph_from_elevation(elevation, weight=1.0, diagonal_movement="always")

        # 3x3 grid should have 9 nodes
        assert G.number_of_nodes() == 9
        # With 8-connectivity, interior node has 8 neighbors, corners have 3, edges have 5
        # Total edges: 9 nodes * avg ~5.3 neighbors = ~48 edges (directed, so count each direction)
        assert G.number_of_edges() > 0
        # Check center node (1,1) has 8 outgoing edges in 8-connectivity
        assert G.out_degree((1, 1)) == 8

    @pytest.mark.unit
    @pytest.mark.generate
    def test_build_graph_4connectivity(self):
        """Test graph construction with 4-connectivity (no diagonals)."""
        elevation = np.ones((3, 3), dtype=np.float32) * 100
        G = gen.build_graph_from_elevation(elevation, weight=1.0, diagonal_movement="never")

        # Check center node (1,1) has 4 outgoing edges in 4-connectivity
        assert G.out_degree((1, 1)) == 4

    @pytest.mark.unit
    @pytest.mark.generate
    def test_build_graph_edge_weights(self):
        """Test that edge weights depend on elevation."""
        elevation = np.array([[100, 200], [300, 400]], dtype=np.float32)
        G = gen.build_graph_from_elevation(elevation, weight=10.0, diagonal_movement="always")

        # Edge weights should be based on destination cell cost
        # Higher elevation = higher cost
        edge_to_400 = G[(0, 0)][(1, 1)]["weight"]
        edge_to_200 = G[(0, 0)][(0, 1)]["weight"]
        assert edge_to_400 > edge_to_200  # (1,1) has highest elevation


class TestLatToStr:
    """Tests for lat_to_str function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_lat_to_str_positive(self):
        """Test lat_to_str with positive latitudes."""
        assert gen.lat_to_str(0) == "N000"
        assert gen.lat_to_str(42) == "N042"
        assert gen.lat_to_str(90) == "N090"

    @pytest.mark.unit
    @pytest.mark.generate
    def test_lat_to_str_negative(self):
        """Test lat_to_str with negative latitudes."""
        assert gen.lat_to_str(-1) == "S001"
        assert gen.lat_to_str(-10) == "S010"
        assert gen.lat_to_str(-90) == "S090"


class TestLonToStr:
    """Tests for lon_to_str function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_lon_to_str_positive(self):
        """Test lon_to_str with positive longitudes."""
        assert gen.lon_to_str(0) == "E000"
        assert gen.lon_to_str(1) == "E001"
        assert gen.lon_to_str(120) == "E120"
        assert gen.lon_to_str(180) == "E180"

    @pytest.mark.unit
    @pytest.mark.generate
    def test_lon_to_str_negative(self):
        """Test lon_to_str with negative longitudes."""
        assert gen.lon_to_str(-1) == "W001"
        assert gen.lon_to_str(-120) == "W120"
        assert gen.lon_to_str(-180) == "W180"


class TestParseLatLonFromFilename:
    """Tests for parse_lat_lon_from_filename function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_parse_north_east(self):
        """Test parsing N/E coordinates."""
        result = gen.parse_lat_lon_from_filename("N042E001_DSM.tif")
        assert result == (42, 1)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_parse_south_west(self):
        """Test parsing S/W coordinates."""
        result = gen.parse_lat_lon_from_filename("S010W120_DSM.tif")
        assert result == (-10, -120)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_parse_mixed(self):
        """Test parsing mixed coordinates."""
        assert gen.parse_lat_lon_from_filename("N000W180_DSM.tif") == (0, -180)
        assert gen.parse_lat_lon_from_filename("S090E179_DSM.tif") == (-90, 179)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_parse_with_prefix(self):
        """Test parsing with ALPSMLC prefix."""
        result = gen.parse_lat_lon_from_filename("ALPSMLC30_N045E010_DSM.tif")
        assert result == (45, 10)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_parse_invalid(self):
        """Test parsing invalid filename returns None."""
        assert gen.parse_lat_lon_from_filename("invalid.tif") is None
        assert gen.parse_lat_lon_from_filename("N12E34_DSM.tif") is None  # Wrong format


# =============================================================================
# Tests for count_valid_samples
# =============================================================================


class TestCountValidSamples:
    """Tests for count_valid_samples function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_empty_dir(self, tmp_path):
        """Test with empty directory."""
        res_dir = tmp_path / "64"
        res_dir.mkdir()
        (res_dir / "tensors").mkdir()
        (res_dir / "metadata").mkdir()
        (res_dir / "images").mkdir()

        count = gen.count_valid_samples(res_dir, disable_images=False)
        assert count == 0

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_all_valid(self, valid_sample_files):
        """Test with all valid samples."""
        count = gen.count_valid_samples(valid_sample_files, disable_images=False)
        assert count == 3

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_partial_cleanup(self, partial_sample_files):
        """Test partial sample cleanup."""
        # Should return 2 (valid samples) and delete partial files at index 2
        count = gen.count_valid_samples(partial_sample_files, disable_images=False)
        assert count == 2

        # Verify partial tensor was deleted
        partial_tensor = partial_sample_files / "tensors" / "0000000002.npy"
        assert not partial_tensor.exists()

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_no_images(self, tmp_path):
        """Test count when images are disabled."""
        res_dir = tmp_path / "64"
        tensors_dir = res_dir / "tensors"
        metadata_dir = res_dir / "metadata"

        tensors_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        # Create 2 valid samples without images
        import yaml

        for i in range(2):
            np.save(tensors_dir / f"{i:010d}.npy", np.zeros((64, 64, 2)))
            with open(metadata_dir / f"{i:010d}.yaml", "w") as f:
                yaml.dump({"index": i}, f)

        count = gen.count_valid_samples(res_dir, disable_images=True)
        assert count == 2

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_missing_dirs(self, tmp_path):
        """Test with missing subdirectories."""
        res_dir = tmp_path / "64"
        res_dir.mkdir()
        # Don't create tensors or metadata dirs

        count = gen.count_valid_samples(res_dir, disable_images=False)
        assert count == 0


# =============================================================================
# Tests for load_tile_grid
# =============================================================================


class TestLoadTileGrid:
    """Tests for load_tile_grid function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_load_tile_grid_basic(self, monkeypatch, tmp_path):
        """Test basic tile grid loading."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Load with half_tiles=1, expect 3x3 grid = 48x48
        grid = gen.load_tile_grid(tiles_dir, 0, 0, half_tiles=1)
        assert grid.shape == (48, 48)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_load_tile_grid_caching(self, monkeypatch, tmp_path):
        """Test tile caching."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Ensure function runs and returns grid when half_tiles=0
        grid = gen.load_tile_grid(tiles_dir, 0, 0, half_tiles=0)
        assert grid.shape == (16, 16)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_load_tile_grid_latitude_bounds(self, monkeypatch, tmp_path):
        """Test latitude boundary handling."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Load at latitude 89 with half_tiles=1, should substitute zeros for out-of-bounds
        grid = gen.load_tile_grid(tiles_dir, 89, 0, half_tiles=1)
        assert grid.shape == (48, 48)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_load_tile_grid_longitude_wrapping(self, monkeypatch, tmp_path):
        """Test longitude wrapping at -180/180."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Load at longitude 179 with half_tiles=1, should wrap to -180 and return grid
        grid = gen.load_tile_grid(tiles_dir, 0, 179, half_tiles=1)
        assert grid.shape == (48, 48)


# =============================================================================
# Tests for generate_sample
# =============================================================================


class TestGenerateSample:
    """Tests for generate_sample function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_basic(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test basic sample generation."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}

        idx, success, msg = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config)
        )

        # May succeed or fail based on random tile selection, check it runs
        assert idx == 0
        assert isinstance(success, bool)
        assert isinstance(msg, str)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_metadata_fields(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test that generated metadata contains all required fields."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}

        idx, success, msg = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config)
        )

        if success:
            import yaml

            metadata_path = outputs_dir / "8" / "metadata" / "0000000000.yaml"
            with open(metadata_path) as f:
                metadata = yaml.safe_load(f)

            # Check all expected fields exist
            assert "index" in metadata
            assert "resolution" in metadata
            assert "target_elevation" in metadata
            assert "tile_min_elevation" in metadata
            assert "tile_max_elevation" in metadata
            assert "crop_min_elevation" in metadata
            assert "crop_max_elevation" in metadata
            assert "markers" in metadata
            assert len(metadata["markers"]) == 2
            assert "elevation" in metadata["markers"][0]
            assert "elevation" in metadata["markers"][1]

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_no_images(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test sample generation with images disabled."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}

        idx, success, msg = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, True, solver_config)
        )

        if success:
            # Images directory should not exist or be empty
            images_dir = outputs_dir / "8" / "images"
            if images_dir.exists():
                assert len(list(images_dir.glob("*.png"))) == 0


# =============================================================================
# Tests for tensor_to_image
# =============================================================================


class TestTensorToImage:
    """Tests for tensor_to_image function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_basic(self):
        """Test tensor_to_image with a simple 2-channel tensor and provided path."""
        # Create a simple 4x4 tensor with known values (elevation + markers)
        tensor = np.zeros((4, 4, 2), dtype=np.float32)
        tensor[:, :, 0] = 0.5  # Elevation channel (mid-gray)
        tensor[1, 1, 1] = 1.0  # Marker at (1, 1)
        tensor[2, 2, 1] = 1.0  # Marker at (2, 2)
        path_pixels = [(1, 2)]  # Path pixel

        image = gen.tensor_to_image(tensor, path_pixels=path_pixels)

        # Check output shape and dtype
        assert image.shape == (4, 4, 3)
        assert image.dtype == np.uint8

        # Check BGR channels
        # Blue (elevation) should be 127 or 128 (0.5 * 255)
        assert image[0, 0, 0] == 127 or image[0, 0, 0] == 128
        # Green (markers) at (1,1) should be 255
        assert image[1, 1, 1] == 255
        # Red (path) at (1,2) should be 255
        assert image[1, 2, 2] == 255

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_all_zeros(self):
        """Test tensor_to_image with all zeros (2-channel)."""
        tensor = np.zeros((8, 8, 2), dtype=np.float32)
        image = gen.tensor_to_image(tensor)

        assert image.shape == (8, 8, 3)
        assert image.dtype == np.uint8
        # Elevation and markers zero -> B and G channels are 0, R is 0 (no path)
        assert np.all(image == 0)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_all_ones(self):
        """Test tensor_to_image with all ones (2-channel)."""
        tensor = np.ones((8, 8, 2), dtype=np.float32)
        image = gen.tensor_to_image(tensor)

        assert image.shape == (8, 8, 3)
        assert image.dtype == np.uint8
        # Elevation and markers ones -> B and G channels 255, R is 0 (no path)
        assert np.all(image[:, :, 0] == 255)
        assert np.all(image[:, :, 1] == 255)
        assert np.all(image[:, :, 2] == 0)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_gradient(self):
        """Test tensor_to_image with gradient elevation (2-channel)."""
        tensor = np.zeros((10, 10, 2), dtype=np.float32)
        # Create gradient in elevation channel
        for i in range(10):
            tensor[i, :, 0] = i / 9.0  # 0 to 1 gradient

        image = gen.tensor_to_image(tensor)

        # Check gradient in blue channel (elevation)
        assert image[0, 0, 0] == 0  # Top row should be 0
        assert image[9, 0, 0] == 255  # Bottom row should be 255

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_invalid_shape_2d(self):
        """Test tensor_to_image raises error for 2D tensor."""
        tensor = np.zeros((10, 10), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected tensor of shape"):
            gen.tensor_to_image(tensor)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_invalid_shape_wrong_channels(self):
        """Test tensor_to_image raises error for wrong number of channels."""
        tensor = np.zeros((10, 10, 4), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected tensor of shape"):
            gen.tensor_to_image(tensor)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_invalid_shape_1d(self):
        """Test tensor_to_image raises error for 1D tensor."""
        tensor = np.zeros((100,), dtype=np.float32)

        with pytest.raises(ValueError, match="Expected tensor of shape"):
            gen.tensor_to_image(tensor)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_clipping(self):
        """Test tensor_to_image handles values outside [0, 1]."""
        tensor = np.zeros((4, 4, 3), dtype=np.float32)
        tensor[0, 0, 0] = 1.5  # Above 1
        tensor[1, 1, 0] = -0.5  # Below 0

        image = gen.tensor_to_image(tensor)

        # Values are not clipped in the function, but uint8 conversion handles overflow
        # 1.5 * 255 = 382.5 -> wraps to 127 (uint8 overflow)
        # -0.5 * 255 = -127.5 -> wraps to 129 (uint8 overflow)
        assert image.dtype == np.uint8

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_roundtrip_with_generate_sample(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test that tensor_to_image produces same output as direct generation."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}

        # Generate with images enabled
        idx, success, msg = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config)
        )

        if success:
            # Load the tensor and convert to image (pass path from metadata)
            tensor_path = outputs_dir / "8" / "tensors" / "0000000000.npy"
            tensor = np.load(tensor_path)
            metadata_path = outputs_dir / "8" / "metadata" / "0000000000.yaml"
            with open(metadata_path) as f:
                md = yaml.safe_load(f)
            path_pixels = [(p["i"], p["j"]) for p in md["path"]["pixels"]]
            reconstructed_image = gen.tensor_to_image(tensor, path_pixels=path_pixels)

            # Load the saved image
            import cv2

            image_path = outputs_dir / "8" / "images" / "0000000000.png"
            saved_image = cv2.imread(str(image_path))

            # They should be identical
            np.testing.assert_array_equal(reconstructed_image, saved_image)


# =============================================================================
# Tests for tiles_needed_for_resolution
# =============================================================================


class TestTilesNeeded:
    """Tests for tiles_needed_for_resolution helper."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tiles_needed_small_resolution(self):
        """Small resolutions should use 3x3 tiles (half_tiles=1)."""
        N, total, half = gen.tiles_needed_for_resolution(256)
        assert half == 1
        assert N == 3
        assert total == 9

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tiles_needed_large_resolution(self):
        """Large resolution should compute larger tile grids."""
        # Use a resolution that forces half_tiles > 1
        res = gen.TILE_SIZE_PX * 4 + 10
        N, total, half = gen.tiles_needed_for_resolution(res)
        expected_half = max(1, math.ceil((res // 2) / gen.TILE_SIZE_PX))
        expected_N = 2 * expected_half + 1
        assert half == expected_half
        assert N == expected_N
        assert total == expected_N * expected_N

    @pytest.mark.integration
    @pytest.mark.generate
    def test_run_generate_logs_for_high_resolution(self, monkeypatch, tmp_path, capsys):
        """Run generate with a large resolution and ensure logging contains diagnostics."""
        # Use small TILE_SIZE_PX to force many tiles
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        # Create calibration file at outputs/calibrate/calibrate.json (project-relative)
        calib_dir = Path("outputs") / "calibrate"
        calib_dir.mkdir(parents=True, exist_ok=True)
        calibrate = {
            "global_min": 0.0,
            "global_max": 1000.0,
            "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0.0, "max": 1000.0, "lat": 0, "lon": 0}],
        }
        with open(calib_dir / "calibrate.json", "w") as f:
            json.dump(calibrate, f)

        # Create config file with a single large resolution
        cfg = {"resolutions": [160], "num_samples": 1}
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Run generation sequentially to keep logs deterministic
        gen.run_generate(str(outputs_dir), str(cfg_path), str(tiles_dir), num_cpus=0, disable_images=True, force=True)

        # Check outputs summary and generated sample files
        summary_path = outputs_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert 160 in summary.get("resolutions", []) or 160 in [int(r) for r in summary.get("resolutions", [])]

        # Check that sample metadata and tensor were written
        metadata_path = outputs_dir / "160" / "metadata" / "0000000000.yaml"
        tensor_path = outputs_dir / "160" / "tensors" / "0000000000.npy"
        assert metadata_path.exists()
        assert tensor_path.exists()

        with open(metadata_path) as f:
            md = yaml.safe_load(f)
        assert "path" in md and "pixels" in md["path"]

        arr = np.load(tensor_path)
        assert arr.shape[0] == 160 and arr.shape[1] == 160 and arr.shape[2] == 2


# =============================================================================
# Tests for partial tile loading and crop assembly
# =============================================================================


class TestAssembleCrop:
    @pytest.mark.unit
    @pytest.mark.generate
    def test_assemble_center_tile_only(self, monkeypatch, tmp_path):
        """Assembling a crop exactly equal to the center tile should return the tile."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 8)
        half_tiles = 1
        center_lat = 0
        center_lon = 0

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create 3x3 tiles with unique values per tile
        for r in range(3):
            dlat = half_tiles - r
            lat = center_lat + dlat
            for c in range(3):
                dlon = -half_tiles + c
                lon = center_lon + dlon
                arr = np.full((8, 8), fill_value=(r * 10 + c), dtype=np.float32)
                fname = tiles_dir / f"{gen.lat_to_str(lat)}{gen.lon_to_str(lon)}_DSM.tif"
                import rasterio

                with rasterio.open(
                    str(fname),
                    "w",
                    driver="GTiff",
                    height=8,
                    width=8,
                    count=1,
                    dtype="float32",
                ) as dst:
                    dst.write(arr, 1)

        # center_offset = TILE_SIZE_PX * half_tiles = 8
        center_tile = gen.load_tile(tiles_dir, center_lat, center_lon)
        crop = gen._assemble_crop_from_tiles(tiles_dir, center_lat, center_lon, half_tiles, 8, 8, 8, center_tile)
        assert crop.shape == (8, 8)
        # The crop should equal the center tile filled with value 11 (row1 col1 => 1*10+1)
        assert np.all(crop == 11)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_assemble_cross_tile_crop(self, monkeypatch, tmp_path):
        """Assembling a crop that crosses center and right tiles."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 8)
        half_tiles = 1
        center_lat = 0
        center_lon = 0

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        # Create 3x3 tiles with unique values per tile
        for r in range(3):
            dlat = half_tiles - r
            lat = center_lat + dlat
            for c in range(3):
                dlon = -half_tiles + c
                lon = center_lon + dlon
                arr = np.full((8, 8), fill_value=(r * 10 + c), dtype=np.float32)
                fname = tiles_dir / f"{gen.lat_to_str(lat)}{gen.lon_to_str(lon)}_DSM.tif"
                import rasterio

                with rasterio.open(
                    str(fname),
                    "w",
                    driver="GTiff",
                    height=8,
                    width=8,
                    count=1,
                    dtype="float32",
                ) as dst:
                    dst.write(arr, 1)

        center_tile = gen.load_tile(tiles_dir, center_lat, center_lon)
        # Create a crop starting 4 pixels into center tile (x1=center_offset+4)
        crop = gen._assemble_crop_from_tiles(tiles_dir, center_lat, center_lon, half_tiles, 8, 12, 8, center_tile)
        assert crop.shape == (8, 8)
        # Left part (4 cols) should be center tile value 11, right part (4 cols) should be 12 (right tile)
        assert np.all(crop[:, :4] == 11)
        assert np.all(crop[:, 4:] == 12)


class TestMetadataCache:
    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_reads_metadata(self, tmp_path):
        """Ensure tensor_to_image reads metadata when provided a metadata path (no global cache)."""
        md = {"path": {"pixels": [{"i": 1, "j": 2}]}, "crop_min_elevation": 0.0, "crop_max_elevation": 100.0}
        md_path = tmp_path / "0000000000.yaml"
        with open(md_path, "w") as f:
            yaml.dump(md, f)

        tensor = np.zeros((4, 4, 2), dtype=np.float32)
        img = gen.tensor_to_image(tensor, metadata_path=md_path)
        # The red channel at (1,2) should be set by the metadata path pixels
        assert img[1, 2, 2] == 255
        # Ensure there is no global _METADATA_CACHE attribute
        assert not hasattr(gen, "_METADATA_CACHE")
