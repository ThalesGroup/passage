from pathlib import Path

import pandas as pd

import passage.generate as gen


def test_load_tile_grid_half_tiles(monkeypatch, tmp_path):
    # Use small TILE_SIZE_PX for quick test
    monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

    tiles_dir = Path(tmp_path)

    # With half_tiles=1, grid should be 3*16 = 48
    grid = gen.load_tile_grid(tiles_dir, 0, 0, half_tiles=1)
    assert grid.shape == (48, 48)

    # With half_tiles=2, grid should be 5*16 = 80
    grid2 = gen.load_tile_grid(tiles_dir, 0, 0, half_tiles=2)
    assert grid2.shape == (80, 80)


def test_generate_sample_large_resolution(monkeypatch, tmp_path):
    # Use small tile size so crop sizes are manageable
    monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

    tiles_dir = Path(tmp_path)
    outputs_dir = Path(tmp_path) / "out"
    outputs_dir.mkdir()

    # Create tile_stats DataFrame with one tile
    tile_stats_df = pd.DataFrame([{"filename": "test.tif", "min": 0.0, "max": 100.0, "lat": 0, "lon": 0}])

    # Use resolution > TILE_SIZE_PX to force loading more tiles
    resolution = 32

    solvers_configs = [
        {
            "name": "test_solver",
            "backend": "grid",
            "common": {"diagonal_movement": "always", "max_marker_retries": 10},
            "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
        }
    ]
    obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

    # Call generate_sample with DataFrame and global min/max
    idx, success, msg, duration = gen.generate_sample(
        (
            0,
            tiles_dir,
            outputs_dir,
            resolution,
            tile_stats_df,
            0.0,
            100.0,
            False,
            solvers_configs,
            obstacle_settings,
            "train",
        )
    )
    assert isinstance(duration, float)

    assert not (msg.startswith("Error") or msg.startswith("Resolution"))


def test_generate_sample_skip_images(monkeypatch, tmp_path):
    # Use small tile size so crop sizes are manageable
    monkeypatch.setattr(gen, "TILE_SIZE_PX", 16)

    tiles_dir = tmp_path
    outputs_dir = Path(tmp_path) / "out2"
    outputs_dir.mkdir()

    # Create tile_stats DataFrame with one tile
    tile_stats_df = pd.DataFrame([{"filename": "test.tif", "min": 0.0, "max": 100.0, "lat": 0, "lon": 0}])

    # Use resolution > TILE_SIZE_PX to force loading more tiles
    resolution = 32

    solvers_configs = [
        {
            "name": "test_solver",
            "backend": "grid",
            "common": {"diagonal_movement": "always", "max_marker_retries": 10},
            "grid": {"solver": "astar", "cost_model": "elevation", "weight": 1.0},
        }
    ]
    obstacle_settings = gen._resolve_obstacle_config({"obstacles": {"enabled": False}})

    # Call generate_sample with disable_images=True
    idx, success, msg, duration = gen.generate_sample(
        (
            0,
            tiles_dir,
            outputs_dir,
            resolution,
            tile_stats_df,
            0.0,
            100.0,
            True,
            solvers_configs,
            obstacle_settings,
            "train",
        )
    )
    assert isinstance(duration, float)

    tensors = list((outputs_dir / str(resolution)).rglob("*.npy.zst"))
    images = list((outputs_dir / str(resolution)).rglob("*.png")) if (outputs_dir / str(resolution)).exists() else []

    assert len(tensors) == 1
    assert len(images) == 0
