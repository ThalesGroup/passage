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

"""Tests for the generate module."""

import json
import math

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


class TestResolveGenerationPoolSize:
    """Tests for generation worker-count selection."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_uses_visible_cpus_for_small_resolution(self, monkeypatch):
        monkeypatch.setattr(gen.os, "cpu_count", lambda: 120)

        assert gen._resolve_generation_pool_size(None, 256, 1000) == 120

    @pytest.mark.unit
    @pytest.mark.generate
    def test_caps_workers_to_remaining_samples(self, monkeypatch):
        monkeypatch.setattr(gen.os, "cpu_count", lambda: 120)

        assert gen._resolve_generation_pool_size(None, 256, 10) == 10

    @pytest.mark.unit
    @pytest.mark.generate
    def test_auto_caps_default_workers_for_4096(self, monkeypatch):
        monkeypatch.setattr(gen.os, "cpu_count", lambda: 120)

        assert gen._resolve_generation_pool_size(None, 4096, 10) == 8

    @pytest.mark.unit
    @pytest.mark.generate
    def test_preserves_explicit_worker_choice_up_to_task_count(self, monkeypatch):
        monkeypatch.setattr(gen.os, "cpu_count", lambda: 120)

        assert gen._resolve_generation_pool_size(4, 4096, 10) == 4

    @pytest.mark.unit
    @pytest.mark.generate
    def test_zero_workers_selects_sequential_mode(self):
        assert gen._resolve_generation_pool_size(0, 4096, 10) == 0


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

    @pytest.mark.unit
    @pytest.mark.generate
    def test_count_valid_samples_allows_missing_images_outside_preview_subset(self, tmp_path):
        """Rows outside the deterministic preview subset should not require images."""
        res_dir = tmp_path / "256"
        lat_dir = gen.lat_to_str(0)
        lon_dir = gen.lon_to_str(0)
        split_assignments = {0: "train", 1: "train", 2: "train"}
        preview_image_indices_by_split = {"train": {0}}
        solver_names = ["astar_elevation", "astar_energy"]

        import cv2
        import yaml

        tensors_dir = res_dir / "train" / "tensors" / lat_dir / lon_dir
        metadata_dir = res_dir / "train" / "metadata" / lat_dir / lon_dir
        tensors_dir.mkdir(parents=True)
        metadata_dir.mkdir(parents=True)

        for index in range(3):
            save_tensor(np.zeros((256, 256, 3), dtype=np.float32), tensors_dir / f"{index:010d}{TENSOR_SUFFIX}")
            with open(metadata_dir / f"{index:010d}.yaml", "w") as f:
                yaml.dump({"index": index}, f)

        for solver_name in solver_names:
            solver_dir = res_dir / "train" / "images" / solver_name / lat_dir / lon_dir
            solver_dir.mkdir(parents=True)
            cv2.imwrite(str(solver_dir / "0000000000.png"), np.zeros((16, 16, 3), dtype=np.uint8))

        count = gen.count_valid_samples(
            res_dir,
            disable_images=False,
            split_assignments=split_assignments,
            preview_image_indices_by_split=preview_image_indices_by_split,
            solver_names=solver_names,
        )
        assert count == 3


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
    def test_generate_sample_is_seed_reproducible(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """The same generation seed should yield the same tensor and deterministic metadata fields."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        fixed_tile = np.arange(16 * 16, dtype=np.float32).reshape(16, 16)

        def fake_load_tile(*_args, **_kwargs):
            return fixed_tile.copy()

        def fake_assemble_crop_from_tiles(*_args, **_kwargs):
            crop = fixed_tile[:8, :8].copy()
            loaded_tiles = [(0, 0, 1, 1)]
            if _kwargs.get("return_loaded_tiles"):
                return crop, loaded_tiles
            return crop

        def fake_find_path(_crop, start, end, _solver_config, prebuilt_graph=None, blocked_mask=None):
            _ = prebuilt_graph
            _ = blocked_mask
            return [tuple(map(int, start)), tuple(map(int, end))]

        monkeypatch.setattr(gen, "load_tile", fake_load_tile)
        monkeypatch.setattr(gen, "_assemble_crop_from_tiles", fake_assemble_crop_from_tiles)
        monkeypatch.setattr(gen, "find_path", fake_find_path)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_a = tmp_path / "outputs_a"
        outputs_b = tmp_path / "outputs_b"
        outputs_a.mkdir()
        outputs_b.mkdir()

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        common_args = (
            0,
            tiles_dir,
            None,
            8,
            mock_tile_stats_df,
            0.0,
            2000.0,
            True,
            solvers_configs,
            obstacle_settings,
            "train",
            1234,
        )

        idx1, success1, msg1, duration1 = gen.generate_sample(
            (common_args[0], common_args[1], outputs_a, *common_args[3:])
        )
        idx2, success2, msg2, duration2 = gen.generate_sample(
            (common_args[0], common_args[1], outputs_b, *common_args[3:])
        )

        assert idx1 == idx2 == 0
        assert success1 and success2, (msg1, msg2)
        assert isinstance(duration1, float)
        assert isinstance(duration2, float)

        tensor_a = next((outputs_a / "8" / "train" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
        tensor_b = next((outputs_b / "8" / "train" / "tensors").rglob(f"0000000000{TENSOR_SUFFIX}"))
        assert tensor_a.read_bytes() == tensor_b.read_bytes()

        metadata_a = yaml.safe_load(next((outputs_a / "8" / "train" / "metadata").rglob("0000000000.yaml")).read_text())
        metadata_b = yaml.safe_load(next((outputs_b / "8" / "train" / "metadata").rglob("0000000000.yaml")).read_text())

        expected_seed = gen._derive_sample_seed(1234, 8, 0, "train")
        assert metadata_a["provenance"] == {"generation_seed": 1234, "sample_seed": expected_seed}
        assert metadata_a["provenance"] == metadata_b["provenance"]
        assert metadata_a["tile"] == metadata_b["tile"]
        assert metadata_a["crop"] == metadata_b["crop"]
        assert metadata_a["markers"] == metadata_b["markers"]

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_basic(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """Test basic sample generation."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

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
                solvers_configs,
                obstacle_settings,
                "train",
            )
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

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

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
                solvers_configs,
                obstacle_settings,
                "train",
            )
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
            assert "tile" in metadata
            assert "stats" in metadata["tile"]
            assert "min_elevation" in metadata["tile"]["stats"]
            assert "max_elevation" in metadata["tile"]["stats"]
            assert "crop" in metadata
            assert "min_elevation" in metadata["crop"]
            assert "max_elevation" in metadata["crop"]
            assert "target_elevation" in metadata["crop"]
            assert "markers" in metadata
            assert "start_point" in metadata["markers"]
            assert "end_point" in metadata["markers"]
            assert "elevation" in metadata["markers"]["start_point"]
            assert "elevation" in metadata["markers"]["end_point"]
            assert "paths" in metadata
            assert "solvers" in metadata
            assert "obstacles" in metadata
            assert metadata["provenance"]["generation_seed"] == 0
            assert isinstance(metadata["provenance"]["sample_seed"], int)

            # Timing diagnostics should be present
            assert "timing" in metadata
            for key in [
                "tile_loading_seconds",
                "marker_search_seconds",
                "obstacle_generation_seconds",
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

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": True}})

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
                solvers_configs,
                obstacle_settings,
                "train",
            )
        )
        assert isinstance(duration, float)

        if success:
            # Paths are now under paths/<solver_name>/free and paths/<solver_name>/obstacles
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

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
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
                solvers_configs,
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
            # Per-solver images (image_<solver_name>.png)
            solver_images_matches = list((base / "images" / "test_solver").rglob("0000000000.png"))
            assert solver_images_matches, "Per-solver image not saved under images/<solver_name> dir"
            # Paths (now under paths/<solver_name>/free and paths/<solver_name>/obstacles)
            free_matches = list((base / "paths").rglob("free/**/0000000000.csv"))
            obs_matches = list((base / "paths").rglob("obstacles/**/0000000000.csv"))
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

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        idx, success, msg, duration = gen.generate_sample(
            (
                0,
                tiles_dir,
                outputs_dir,
                8,
                mock_tile_stats_df,
                0.0,
                2000.0,
                True,
                solvers_configs,
                obstacle_settings,
                "train",
            )
        )
        assert isinstance(duration, float)

        if success:
            # Images directory should not exist or be empty
            images_dir = outputs_dir / "8" / "images"
            if images_dir.exists():
                assert len(list(images_dir.rglob("*.png"))) == 0

    @pytest.mark.unit
    @pytest.mark.generate
    def test_generate_sample_skips_preview_images_when_not_selected(self, monkeypatch, tmp_path, mock_tile_stats_df):
        """A non-preview row should skip image generation even when global images are enabled."""
        monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

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
                solvers_configs,
                obstacle_settings,
                "train",
                1234,
                False,
            )
        )
        assert isinstance(duration, float)

        if success:
            images_dir = outputs_dir / "8" / "train" / "images"
            assert not images_dir.exists() or len(list(images_dir.rglob("*.png"))) == 0


# =============================================================================
# Tests for tensor_to_image
# =============================================================================


class TestTensorToImage:
    """Tests for tensor_to_image function."""

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_basic(self):
        """Test tensor_to_image with a simple 2-channel tensor and provided path."""
        tensor = np.zeros((64, 64, 2), dtype=np.float32)
        tensor[:, :, 0] = 0.5
        tensor[:, :, 1] = -1.0
        tensor[10, 10, 1] = 0.0
        tensor[30, 30, 1] = 1.0
        path_pixels = [(10, col) for col in range(10, 31)] + [(row, 30) for row in range(11, 31)]

        image = gen.tensor_to_image(tensor, path_pixels=path_pixels)

        assert image.shape == (64, 64, 3)
        assert image.dtype == np.uint8
        assert image[0, 0].sum() > 0
        np.testing.assert_array_equal(image[10, 10], [255, 255, 255])
        np.testing.assert_array_equal(image[30, 30], [255, 255, 255])
        np.testing.assert_array_equal(image[10, 20], [172, 103, 23])  # blue, BGR
        np.testing.assert_array_equal(image[20, 30], [172, 103, 23])
        np.testing.assert_array_equal(image[9, 20], [255, 255, 255])  # route halo
        np.testing.assert_array_equal(image[20, 20], image[0, 0])  # ordered L, no closing diagonal

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_two_paths(self):
        """Test that tensor_to_image draws both primary and secondary paths and that sizes scale."""
        tensor64 = np.zeros((64, 64, 3), dtype=np.float32)
        tensor64[:, :, 0] = 0.5
        tensor64[:, :, 1] = -1.0
        path64 = [(20, col) for col in range(15, 46)]
        alt64 = [(40, col) for col in range(15, 46)]
        img64 = gen.tensor_to_image(tensor64, path_pixels=path64, alt_path_pixels=alt64)
        primary64 = np.sum(np.all(img64[:, 30] == [172, 103, 23], axis=1))
        alternate64 = np.sum(np.all(img64[:, 30] == [24, 100, 213], axis=1))

        # Larger resolution (256) -> path thickness should increase
        tensor256 = np.zeros((256, 256, 3), dtype=np.float32)
        tensor256[:, :, 0] = 0.5
        tensor256[:, :, 1] = -1.0
        path256 = [(80, col) for col in range(60, 181)]
        alt256 = [(160, col) for col in range(60, 181)]
        img256 = gen.tensor_to_image(tensor256, path_pixels=path256, alt_path_pixels=alt256)
        primary256 = np.sum(np.all(img256[:, 120] == [172, 103, 23], axis=1))
        alternate256 = np.sum(np.all(img256[:, 120] == [24, 100, 213], axis=1))

        assert primary64 > 0 and alternate64 > 0
        assert primary256 > primary64, "Primary route thickness did not scale up"
        assert alternate256 > alternate64, "Alternate route thickness did not scale up"

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_alt_path_overrides_primary(self):
        """The orange alternate route is drawn above the blue primary route."""
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.5
        tensor[:, :, 1] = -1.0
        path_pixels = [(30, col) for col in range(15, 46)]
        alt_path_pixels = list(path_pixels)
        img = gen.tensor_to_image(tensor, path_pixels=path_pixels, alt_path_pixels=alt_path_pixels)
        np.testing.assert_array_equal(img[30, 30], [24, 100, 213])
        np.testing.assert_array_equal(img[29, 30], [255, 255, 255])

    @pytest.mark.unit
    @pytest.mark.generate
    def test_marker_circle_and_star_are_white_with_dark_edges(self):
        """Start circles and goal stars remain distinct and visible above routes."""
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.5
        tensor[:, :, 1] = -1.0
        tensor[16, 16, 1] = 0.0
        tensor[40, 40, 1] = 1.0
        path = [(cell, cell) for cell in range(16, 41)]
        img = gen.tensor_to_image(tensor, path_pixels=path)

        np.testing.assert_array_equal(img[16, 16], [255, 255, 255])
        np.testing.assert_array_equal(img[15, 15], [255, 255, 255])  # filled circle includes diagonal
        np.testing.assert_array_equal(img[16, 19], [43, 35, 24])  # dark circle boundary
        np.testing.assert_array_equal(img[40, 40], [255, 255, 255])
        np.testing.assert_array_equal(img[36, 40], [43, 35, 24])  # star's upward point
        np.testing.assert_array_equal(img[12, 16], img[0, 0])  # circle does not have that long point

    @pytest.mark.unit
    @pytest.mark.generate
    def test_tensor_to_image_with_obstacles_channel(self):
        """Obstacles retain charcoal fill and hatching independently of routes."""
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        tensor[:, :, 0] = 0.25
        tensor[:, :, 1] = -1.0
        tensor[5, 5, 1] = 0.0
        tensor[25:38, 25:38, 2] = 1.0
        path_pixels = [(50, col) for col in range(15, 46)]

        image = gen.tensor_to_image(tensor, path_pixels=path_pixels)
        np.testing.assert_array_equal(image[32, 32], [58, 56, 48])
        np.testing.assert_array_equal(image[30, 29], [165, 170, 163])  # hatch within mask
        np.testing.assert_array_equal(image[50, 30], [172, 103, 23])
        np.testing.assert_array_equal(image[5, 5], [255, 255, 255])
        np.testing.assert_array_equal(image[32, 45], image[0, 0])  # obstacle mask stays local

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

        solvers_configs = [
            {
                "name": "test_solver",
                "backend": "grid",
                "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
            }
        ]
        obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

        # Generate with images enabled
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
                solvers_configs,
                obstacle_settings,
                "train",
            )
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
            # New format: paths is keyed by solver name, with "free" and "obstacles" sub-keys
            first_solver = list(md.get("paths", {}).keys())[0] if md.get("paths") else None
            if first_solver:
                # Read path from CSV file instead of metadata (metadata no longer stores full pixel list)
                path_csv_matches = list(
                    (outputs_dir / "8" / "train" / "paths" / first_solver / "free").rglob("0000000000.csv")
                )
                if path_csv_matches:
                    import csv

                    with open(path_csv_matches[0], newline="") as f:
                        path_pixels = [(int(row[0]), int(row[1])) for row in csv.reader(f)]
                else:
                    path_pixels = []
            else:
                path_pixels = []
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

        # This synthetic calibration belongs only to this test.
        calib_dir = tmp_path / "calibrate"
        calib_dir.mkdir(parents=True, exist_ok=True)
        calibrate = {
            "global_min": 0.0,
            "global_max": 1000.0,
            "tile_stats": [{"filename": "N000E000_DSM.tif", "min": 0.0, "max": 1000.0, "lat": 0, "lon": 0}],
        }
        with open(calib_dir / "calibrate.json", "w") as f:
            json.dump(calibrate, f)

        # Create config file with a single large resolution and solvers list
        cfg = {
            "resolutions": [160],
            "num_samples": 1,
            "generation": {"seed": 314159},
            "solvers": [
                {
                    "name": "test_solver",
                    "backend": "grid",
                    "common": {"diagonal_movement": "always", "max_marker_retries": 10},
                    "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
                }
            ],
        }
        cfg_path = tmp_path / "config.yaml"
        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f)

        tiles_dir = tmp_path / "tiles"
        tiles_dir.mkdir()

        outputs_dir = tmp_path / "outputs"
        outputs_dir.mkdir()

        # Run generation sequentially to keep logs deterministic
        gen.run_generate(
            str(outputs_dir),
            str(cfg_path),
            str(tiles_dir),
            calibration_filepath=str(calib_dir / "calibrate.json"),
            num_cpus=0,
            disable_images=True,
            force=True,
        )

        # Check outputs summary and generated sample files
        summary_path = outputs_dir / "summary.json"
        assert summary_path.exists()
        with open(summary_path) as f:
            summary = json.load(f)
        assert 160 in summary.get("resolutions", []) or 160 in [int(r) for r in summary.get("resolutions", [])]
        assert summary["generation"]["seed"] == 314159

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
        # New format: paths is keyed by solver name
        assert "paths" in md and "solvers" in md
        assert md["provenance"]["generation_seed"] == 314159
        first_solver_paths = next(iter(md["paths"].values()))
        assert "considered_area" in first_solver_paths["free"]
        assert "considered_area" in first_solver_paths["obstacles"]

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
    def test_tensor_to_image_reads_metadata_equivalent(self):
        """Ensure tensor_to_image behavior when metadata is provided externally (path pixels)."""
        md = {
            "paths": {
                "without_obstacles": {
                    "pixels": [{"i": 15, "j": col} for col in range(20, 36)]
                    + [{"i": row, "j": 35} for row in range(16, 31)]
                },
                "with_obstacles": {"pixels": []},
            },
            "obstacles": {"enabled": False},
            "crop": {"min_elevation": 0.0, "max_elevation": 100.0},
        }
        # Instead of passing metadata_path to tensor_to_image, callers should extract path pixels
        path_section = md.get("paths", {}).get("without_obstacles") or md.get("paths", {}).get("with_obstacles")
        path_pixels = [(p["i"], p["j"]) for p in path_section.get("pixels", [])]

        tensor = np.zeros((64, 64, 2), dtype=np.float32)
        tensor[:, :, 1] = -1
        img = gen.tensor_to_image(tensor, path_pixels=path_pixels)
        np.testing.assert_array_equal(img[15, 25], [172, 103, 23])
        np.testing.assert_array_equal(img[25, 35], [172, 103, 23])
        np.testing.assert_array_equal(img[25, 15], img[0, 0])  # i/j fields were not exchanged
        np.testing.assert_array_equal(img[22, 27], img[0, 0])  # no synthetic closing diagonal
