"""Tests for the generate module."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

import passage.generate as gen
from passage.utils import TENSOR_SUFFIX, save_tensor

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
            "grid:astar",
            "grid:dijkstra",
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

    @pytest.mark.unit
    @pytest.mark.generate
    def test_find_path_with_prebuilt_graph(self):
        """Test find_path when a prebuilt NetworkX graph is supplied."""
        crop = np.ones((8, 8), dtype=np.float32) * 50
        solver_config = {"name": "networkx:astar", "diagonal_movement": "always", "weight": 1.0}
        G = gen.build_graph_from_elevation(crop, weight=1.0, diagonal_movement="always")
        path = gen.find_path(crop, (0, 0), (7, 7), solver_config, prebuilt_graph=G)
        assert len(path) > 0
        assert path[0] == (0, 0)
        assert path[-1] == (7, 7)


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
        lat_dir = gen.lat_to_str(0)
        lon_dir = gen.lon_to_str(0)
        partial_tensor = partial_sample_files / "tensors" / lat_dir / lon_dir / f"0000000002{TENSOR_SUFFIX}"
        assert not partial_tensor.exists()

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_no_images(self, tmp_path):
        """Test count when images are disabled."""
        res_dir = tmp_path / "64"
        lat_dir = gen.lat_to_str(0)
        lon_dir = gen.lon_to_str(0)
        tensors_dir = res_dir / "tensors" / lat_dir / lon_dir
        metadata_dir = res_dir / "metadata" / lat_dir / lon_dir

        tensors_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        # Create 2 valid samples without images
        import yaml

        for i in range(2):
            save_tensor(np.zeros((64, 64, 3), dtype=np.float32), tensors_dir / f"{i:010d}{TENSOR_SUFFIX}")
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

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_split_layout(self, tmp_path):
        """Test split-aware counting across multiple splits, with partial cleanup."""
        res_dir = tmp_path / "256"
        lat_dir = gen.lat_to_str(0)
        lon_dir = gen.lon_to_str(0)

        # Assign indices to splits (0,1,2,3)
        split_assignments = {0: "train", 1: "validation", 2: "train", 3: "validation"}

        # Create valid files for indices 0 and 1, make index 2 partial (missing metadata), index 3 valid
        import yaml

        # index 0 -> train (valid)
        tensors_dir = res_dir / "train" / "tensors" / lat_dir / lon_dir
        metadata_dir = res_dir / "train" / "metadata" / lat_dir / lon_dir
        tensors_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        save_tensor(np.zeros((256, 256, 3), dtype=np.float32), tensors_dir / f"{0:010d}{TENSOR_SUFFIX}")
        with open(metadata_dir / f"{0:010d}.yaml", "w") as f:
            yaml.dump({"index": 0}, f)

        # index 1 -> validation (valid)
        tensors_dir = res_dir / "validation" / "tensors" / lat_dir / lon_dir
        metadata_dir = res_dir / "validation" / "metadata" / lat_dir / lon_dir
        tensors_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)
        save_tensor(np.zeros((256, 256, 3), dtype=np.float32), tensors_dir / f"{1:010d}{TENSOR_SUFFIX}")
        with open(metadata_dir / f"{1:010d}.yaml", "w") as f:
            yaml.dump({"index": 1}, f)

        # index 2 -> train (partial: tensor only)
        tensors_dir = res_dir / "train" / "tensors" / lat_dir / lon_dir
        tensors_dir.mkdir(parents=True, exist_ok=True)
        save_tensor(np.zeros((256, 256, 3), dtype=np.float32), tensors_dir / f"{2:010d}{TENSOR_SUFFIX}")

        # index 3 -> validation (valid but after partial index so should not be counted)
        tensors_dir = res_dir / "validation" / "tensors" / lat_dir / lon_dir
        metadata_dir = res_dir / "validation" / "metadata" / lat_dir / lon_dir
        tensors_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        save_tensor(np.zeros((256, 256, 3), dtype=np.float32), tensors_dir / f"{3:010d}{TENSOR_SUFFIX}")
        with open(metadata_dir / f"{3:010d}.yaml", "w") as f:
            yaml.dump({"index": 3}, f)

        # Count should stop at index 2 (first incomplete) and delete partial tensor at index 2
        count = gen.count_valid_samples(res_dir, disable_images=True, split_assignments=split_assignments)
        assert count == 2
        partial_tensor = res_dir / "train" / "tensors" / lat_dir / lon_dir / f"{2:010d}{TENSOR_SUFFIX}"
        assert not partial_tensor.exists()


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
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        idx, success, msg, duration = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config, obstacle_settings)
        )
        assert isinstance(duration, float)

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
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        idx, success, msg, duration = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config, obstacle_settings)
        )
        assert isinstance(duration, float)

        if success:
            import yaml

            # Prefer split-aware path (res/8/train/metadata) then fall back to legacy res/8/metadata
            metadata_matches = list((outputs_dir / "8" / "train" / "metadata").rglob("0000000000.yaml"))
            if not metadata_matches:
                metadata_matches = list((outputs_dir / "8" / "metadata").rglob("0000000000.yaml"))
            assert metadata_matches
            metadata_path = metadata_matches[0]
            with open(metadata_path) as f:
                metadata = yaml.safe_load(f)

            # Check all expected fields exist
            assert "index" in metadata
            assert "resolution" in metadata
            assert "target_elevation" in metadata
            assert "tile" in metadata
            assert "stats" in metadata["tile"]
            assert "min_elevation" in metadata["tile"]["stats"]
            assert "max_elevation" in metadata["tile"]["stats"]
            assert "crop" in metadata
            assert "min_elevation" in metadata["crop"]
            assert "max_elevation" in metadata["crop"]
            assert "markers" in metadata
            assert "start_point" in metadata["markers"]
            assert "end_point" in metadata["markers"]
            assert "elevation" in metadata["markers"]["start_point"]
            assert "elevation" in metadata["markers"]["end_point"]
            assert "paths" in metadata
            assert "without_obstacles" in metadata["paths"]
            assert "obstacles" in metadata

            # Timing diagnostics should be present
            assert "timing" in metadata
            for key in [
                "tile_loading_seconds",
                "marker_search_seconds",
                "marker_find_path_without_seconds",
                "marker_find_path_with_seconds",
                "graph_prebuild_seconds",
                "obstacle_generation_seconds",
                "cost_compute_seconds",
                "tensor_save_seconds",
                "metadata_write_seconds",
                "image_conversion_seconds",
                "image_write_seconds",
            ]:
                assert key in metadata["timing"]

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_paths_files(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test that pixel-index path files are written for free and obstacles."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": True}})

        idx, success, msg, duration = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config, obstacle_settings)
        )
        assert isinstance(duration, float)

        if success:
            all_paths = list((outputs_dir / "8").rglob("0000000000.csv"))
            free_matches = [p for p in all_paths if "paths" in p.parts and "free" in p.parts]
            obs_matches = [p for p in all_paths if "paths" in p.parts and "obstacles" in p.parts]

            assert free_matches
            assert obs_matches

            import csv

            with open(free_matches[0], newline="") as f:
                free_list = list(csv.reader(f))
            assert isinstance(free_list, list)
            assert all(len(pair) == 2 for pair in free_list)
            assert all(0 <= int(pair[0]) < 8 and 0 <= int(pair[1]) < 8 for pair in free_list)

            with open(obs_matches[0], newline="") as f:
                obs_list = list(csv.reader(f))
            assert isinstance(obs_list, list)
            assert all(len(pair) == 2 for pair in obs_list)
            assert all(0 <= int(pair[0]) < 8 and 0 <= int(pair[1]) < 8 for pair in obs_list)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_uses_split_subdirs(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Ensure generate_sample writes tensors, metadata, images and paths under the provided split subdir."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solver_config = {"name": "astar", "diagonal_movement": "always", "weight": 1.0, "max_marker_retries": 10}
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": True}})

        # Pass explicit split name different from default 'train'
        idx, success, msg, duration = gen.generate_sample(
            (
                0,
                tiles_dir,
                outputs_dir,
                8,
                mock_tile_stats_df,
                0.0,
                2000.0,
                False,
                solver_config,
                obstacle_settings,
                "validation",
            )
        )
        assert isinstance(duration, float)

        if success:
            base = outputs_dir / "8" / "validation"
            # Tensor
            tensor_matches = list((base / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
            assert tensor_matches, "Tensor file not saved under split-aware tensors dir"
            # Metadata
            metadata_matches = list((base / "metadata").rglob("0000000000.yaml"))
            assert metadata_matches, "Metadata file not saved under split-aware metadata dir"
            # Images
            images_matches = list((base / "images").rglob("0000000000.png"))
            assert images_matches, "Image file not saved under split-aware images dir"
            # Paths
            free_matches = list((base / "paths" / "free").rglob("0000000000.csv"))
            obs_matches = list((base / "paths" / "obstacles").rglob("0000000000.csv"))
            assert free_matches and obs_matches, "Path files not saved under split-aware paths dirs"

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
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        idx, success, msg, duration = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, True, solver_config, obstacle_settings)
        )
        assert isinstance(duration, float)

        if success:
            # Images directory should not exist or be empty
            images_dir = outputs_dir / "8" / "images"
            if images_dir.exists():
                assert len(list(images_dir.rglob("*.png"))) == 0


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
        tensor[:, :, 1] = -1.0
        tensor[1, 1, 1] = 0.0  # start_point at (1, 1)
        tensor[2, 2, 1] = 1.0  # end_point at (2, 2)
        path_pixels = [(1, 2)]  # Path pixel

        image = gen.tensor_to_image(tensor, path_pixels=path_pixels)

        # Check output shape and dtype
        assert image.shape == (4, 4, 3)
        assert image.dtype == np.uint8

        # Basic checks: elevation colormap applied (non-empty), markers and path drawn, no red
        assert image[0, 0].sum() > 0  # elevation colormap produced a non-black pixel
        # Start marker at (1,1) should be visible (green channel is filled)
        assert image[1, 1, 1] == 255
        # Path at (1,2) should be yellow (B low, G high, R high)
        assert (image[1, 2, 1] >= 200) and (image[1, 2, 2] >= 200) and (image[1, 2, 0] < 100)
        # Red channel now used for the yellow path overlay
        assert image[1, 2, 2] >= 200

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_two_paths(self):
        """Test that tensor_to_image draws both primary and secondary paths and that sizes scale."""
        # Small resolution (64) -> path thickness ~1
        tensor64 = np.zeros((64, 64, 3), dtype=np.float32)
        tensor64[:, :, 0] = 0.5
        tensor64[:, :, 1] = -1.0
        tensor64[32, 32, 1] = 0.0
        path64 = [(32, 30), (32, 31), (32, 32), (32, 33), (32, 34)]
        img64 = gen.tensor_to_image(tensor64, path_pixels=path64, alt_path_pixels=None)
        # Primary path is now yellow (B low, G high, R high)
        primary64 = np.sum((img64[:, :, 1] >= 200) & (img64[:, :, 2] >= 200) & (img64[:, :, 0] < 100))

        # Larger resolution (256) -> path thickness should increase
        tensor256 = np.zeros((256, 256, 3), dtype=np.float32)
        tensor256[:, :, 0] = 0.5
        tensor256[:, :, 1] = -1.0
        tensor256[128, 128, 1] = 0.0
        # Scale coordinates roughly by 4
        path256 = [(128, 120), (128, 121), (128, 122), (128, 123), (128, 124)]
        img256 = gen.tensor_to_image(tensor256, path_pixels=path256, alt_path_pixels=None)
        primary256 = np.sum((img256[:, :, 1] >= 200) & (img256[:, :, 2] >= 200) & (img256[:, :, 0] < 100))

        assert primary64 > 0, "No primary (yellow) pixels found for small resolution path"
        assert primary256 > primary64, "Primary path did not scale up for higher resolution"

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_alt_path_overrides_primary(self):
        """Alternate (with-obstacles) path should be green and override primary when overlapping."""
        tensor = np.zeros((8, 8, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.5
        tensor[:, :, 1] = -1.0
        # Place both primary and alternate path at the same pixel
        path_pixels = [(3, 3)]
        alt_path_pixels = [(3, 3)]
        img = gen.tensor_to_image(tensor, path_pixels=path_pixels, alt_path_pixels=alt_path_pixels)
        # The pixel should be green (alt path drawn last on top of primary)
        assert (img[3, 3, 1] == 255) and (img[3, 3, 2] < 100) and (img[3, 3, 0] < 100)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_marker_plus_shape_and_no_white(self):
        """Ensure marker is drawn as a plus-sign (3x3 at base) and no white outline exists."""
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.5
        tensor[:, :, 1] = -1.0
        # Place start marker at center and end marker elsewhere
        tensor[32, 32, 1] = 0.0
        tensor[40, 40, 1] = 1.0
        img = gen.tensor_to_image(tensor, path_pixels=None)

        # Start marker (cyan): check center and cardinal neighbors have high B and G
        sx, sy = 32, 32
        neighbors = [(sx, sy), (sx - 1, sy), (sx + 1, sy), (sx, sy - 1), (sx, sy + 1)]
        for y, x in neighbors:
            assert (img[y, x, 0] >= 200) and (img[y, x, 1] >= 200) and (img[y, x, 2] < 100)

        # End marker (blue): check central pixel has dominant B channel
        ex, ey = 40, 40
        assert (img[ex, ey, 0] >= 200) and (img[ex, ey, 0] > img[ex, ey, 1]) and (img[ex, ey, 0] > img[ex, ey, 2])

        # Check diagonal neighbors near start are not cyan/green-dominant (plus shape only)
        diagonals = [(sx - 1, sy - 1), (sx - 1, sy + 1), (sx + 1, sy - 1), (sx + 1, sy + 1)]
        for y, x in diagonals:
            assert not ((img[y, x, 0] >= 200) and (img[y, x, 1] >= 200))

        # Ensure no white (255,255,255) in a small neighborhood around start marker
        for y in range(sx - 2, sx + 3):
            for x in range(sy - 2, sy + 3):
                assert not (img[y, x, 0] == 255 and img[y, x, 1] == 255 and img[y, x, 2] == 255)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_with_obstacles_channel(self, tmp_path):
        """Test tensor_to_image with obstacle channel and metadata paths."""
        tensor = np.zeros((4, 4, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.25
        tensor[:, :, 1] = -1.0
        tensor[0, 0, 1] = 0.0
        tensor[1, 1, 2] = 1.0  # obstacle pixel

        # Provide metadata example (not needed here since we pass path_pixels explicitly)
        # md = {...}  # unused in this test
        # Provide path pixels explicitly instead of passing metadata_path
        path_pixels = [(2, 2)]

        image = gen.tensor_to_image(tensor, path_pixels=path_pixels)
        # obstacle should show in red channel (255)
        assert image[1, 1, 2] == 255
        # path pixel should be yellow at (2,2)
        assert (image[2, 2, 1] >= 200) and (image[2, 2, 2] >= 200) and (image[2, 2, 0] < 100)

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_all_zeros(self):
        """Test tensor_to_image with all zeros (2-channel)."""
        tensor = np.zeros((8, 8, 2), dtype=np.float32)
        tensor[:, :, 1] = -1.0
        image = gen.tensor_to_image(tensor)

        assert image.shape == (8, 8, 3)
        assert image.dtype == np.uint8
        # Elevation zeros map through the terrain colormap -> image should not be all zeros
        assert not np.all(image == 0)
        # With no obstacles there should be no pure-red pixels (obstacle overlay)
        assert not np.any((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255))

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_all_ones(self):
        """Test tensor_to_image with all ones (2-channel)."""
        tensor = np.ones((8, 8, 2), dtype=np.float32)
        image = gen.tensor_to_image(tensor)

        assert image.shape == (8, 8, 3)
        assert image.dtype == np.uint8
        # Elevation ones map through the terrain colormap -> not all black and no obstacles
        assert not np.all(image == 0)
        # With no obstacles there should be no pure-red pixels (obstacle overlay)
        assert not np.any((image[:, :, 0] == 0) & (image[:, :, 1] == 0) & (image[:, :, 2] == 255))

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_gradient(self):
        """Test tensor_to_image with gradient elevation (2-channel)."""
        tensor = np.zeros((10, 10, 2), dtype=np.float32)
        # Create gradient in elevation channel
        for i in range(10):
            tensor[i, :, 0] = i / 9.0  # 0 to 1 gradient

        image = gen.tensor_to_image(tensor)

        # Gradient should produce different colors at top and bottom
        assert not np.array_equal(image[0, 0], image[9, 0])  # Top and bottom should differ

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
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        # Generate with images enabled
        idx, success, msg, duration = gen.generate_sample(
            (0, tiles_dir, outputs_dir, 8, mock_tile_stats_df, 0.0, 2000.0, False, solver_config, obstacle_settings)
        )
        assert isinstance(duration, float)

        if success:
            # Load the tensor and convert to image (pass path from metadata)
            # Prefer split-aware paths (res/8/train/...) then fall back to legacy res/8/... locations
            tensor_matches = list((outputs_dir / "8" / "train" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
            if not tensor_matches:
                tensor_matches = list((outputs_dir / "8" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
            assert tensor_matches
            tensor_path = tensor_matches[0]
            from passage.utils import load_tensor

            tensor = load_tensor(tensor_path, (8, 8, 3), "float32")
            metadata_matches = list((outputs_dir / "8" / "train" / "metadata").rglob("0000000000.yaml"))
            if not metadata_matches:
                metadata_matches = list((outputs_dir / "8" / "metadata").rglob("0000000000.yaml"))
            assert metadata_matches
            metadata_path = metadata_matches[0]
            with open(metadata_path) as f:
                md = yaml.safe_load(f)
            path_section = md.get("paths", {}).get("without_obstacles") or md.get("paths", {}).get("with_obstacles")
            if path_section is None:
                path_section = md.get("path", {})
            path_pixels = [(p["i"], p["j"]) for p in path_section.get("pixels", [])]
            reconstructed_image = gen.tensor_to_image(tensor, path_pixels=path_pixels)

            # Load the saved image
            import cv2

            # Prefer split-aware images dir, then fall back to legacy top-level images
            image_matches = list((outputs_dir / "8" / "train" / "images").rglob("0000000000.png"))
            if not image_matches:
                image_matches = list((outputs_dir / "8" / "images").rglob("0000000000.png"))
            assert image_matches
            image_path = image_matches[0]
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
        # Prefer split-aware locations (res/160/train/...) then fall back to legacy res/160/...
        metadata_matches = list((outputs_dir / "160" / "train" / "metadata").rglob("0000000000.yaml"))
        if not metadata_matches:
            metadata_matches = list((outputs_dir / "160" / "metadata").rglob("0000000000.yaml"))
        tensor_matches = list((outputs_dir / "160" / "train" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
        if not tensor_matches:
            tensor_matches = list((outputs_dir / "160" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
        assert metadata_matches
        assert tensor_matches
        metadata_path = metadata_matches[0]
        tensor_path = tensor_matches[0]
        assert metadata_path.exists()
        assert tensor_path.exists()

        with open(metadata_path) as f:
            md = yaml.safe_load(f)
        assert "paths" in md and "without_obstacles" in md["paths"]

        from passage.utils import load_tensor

        arr = load_tensor(tensor_path, (160, 160, 3), "float32")
        assert arr.shape[0] == 160 and arr.shape[1] == 160 and arr.shape[2] == 3


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
    def test_tensor_to_image_reads_metadata_equivalent(self, tmp_path):
        """Ensure tensor_to_image behavior when metadata is provided externally (path pixels)."""
        md = {
            "paths": {"without_obstacles": {"pixels": [{"i": 1, "j": 2}]}, "with_obstacles": {"pixels": []}},
            "obstacles": {"enabled": False},
            "crop": {"min_elevation": 0.0, "max_elevation": 100.0},
        }
        # Instead of passing metadata_path to tensor_to_image, callers should extract path pixels
        path_section = md.get("paths", {}).get("without_obstacles") or md.get("paths", {}).get("with_obstacles")
        path_pixels = [(p["i"], p["j"]) for p in path_section.get("pixels", [])]

        tensor = np.zeros((4, 4, 2), dtype=np.float32)
        img = gen.tensor_to_image(tensor, path_pixels=path_pixels)
        # The path pixel at (1,2) should be yellow
        assert (img[1, 2, 1] >= 200) and (img[1, 2, 2] >= 200) and (img[1, 2, 0] < 100)
