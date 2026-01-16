"""Shared pytest fixtures for PASSAGE tests."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml


@pytest.fixture
def mock_tile_stats_df():
    """Create a mock tile_stats DataFrame for testing."""
    return pd.DataFrame(
        [
            {"filename": "N000E000_DSM.tif", "min": 0.0, "max": 100.0, "lat": 0, "lon": 0},
            {"filename": "N001E000_DSM.tif", "min": 50.0, "max": 150.0, "lat": 1, "lon": 0},
            {"filename": "N000E001_DSM.tif", "min": 100.0, "max": 200.0, "lat": 0, "lon": 1},
            {"filename": "N045E010_DSM.tif", "min": 500.0, "max": 2000.0, "lat": 45, "lon": 10},
        ]
    )


@pytest.fixture
def mock_calibration_data():
    """Create mock calibration data for testing."""
    return {
        "global_min": 0.0,
        "global_max": 2000.0,
        "tile_stats": [
            {"filename": "N000E000_DSM.tif", "min": 0.0, "max": 100.0, "lat": 0, "lon": 0},
            {"filename": "N001E000_DSM.tif", "min": 50.0, "max": 150.0, "lat": 1, "lon": 0},
            {"filename": "N000E001_DSM.tif", "min": 100.0, "max": 200.0, "lat": 0, "lon": 1},
            {"filename": "N045E010_DSM.tif", "min": 500.0, "max": 2000.0, "lat": 45, "lon": 10},
        ],
    }


@pytest.fixture
def calibrate_json_file(tmp_path, mock_calibration_data):
    """Create a calibrate.json file in tmp_path."""
    calibrate_dir = tmp_path / "calibrate"
    calibrate_dir.mkdir(parents=True, exist_ok=True)
    calibrate_path = calibrate_dir / "calibrate.json"
    with open(calibrate_path, "w") as f:
        json.dump(mock_calibration_data, f)
    return calibrate_path


@pytest.fixture
def config_yaml_file(tmp_path):
    """Create a config.yaml file in tmp_path."""
    config_dir = tmp_path / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / "config.yaml"
    config_data = {
        "resolutions": [64, 128],
        "num_samples": 5,
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


@pytest.fixture
def mock_tiles_dir(tmp_path):
    """Create a mock tiles directory (empty, for testing when tiles aren't found)."""
    tiles_dir = tmp_path / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    return tiles_dir


@pytest.fixture
def valid_sample_files(tmp_path):
    """Create valid sample files (tensor, metadata, image) for testing resume logic."""
    res_dir = tmp_path / "dataset" / "64"
    tensors_dir = res_dir / "tensors"
    metadata_dir = res_dir / "metadata"
    images_dir = res_dir / "images"

    tensors_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create 3 valid samples (indices 0, 1, 2)
    for i in range(3):
        # Create tensor
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        np.save(tensors_dir / f"{i:010d}.npy", tensor)

        # Create metadata
        metadata = {
            "index": i,
            "resolution": 64,
            "target_elevation": 100.0 + i * 10,
        }
        with open(metadata_dir / f"{i:010d}.yaml", "w") as f:
            yaml.dump(metadata, f)

        # Create image (simple 64x64 PNG)
        import cv2

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{i:010d}.png"), img)

    return res_dir


@pytest.fixture
def partial_sample_files(tmp_path):
    """Create partial sample files (incomplete sample) for testing resume cleanup."""
    res_dir = tmp_path / "dataset" / "64"
    tensors_dir = res_dir / "tensors"
    metadata_dir = res_dir / "metadata"
    images_dir = res_dir / "images"

    tensors_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create 2 valid samples (indices 0, 1)
    for i in range(2):
        tensor = np.zeros((64, 64, 3), dtype=np.float32)
        np.save(tensors_dir / f"{i:010d}.npy", tensor)
        metadata = {"index": i, "resolution": 64}
        with open(metadata_dir / f"{i:010d}.yaml", "w") as f:
            yaml.dump(metadata, f)
        import cv2

        img = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.imwrite(str(images_dir / f"{i:010d}.png"), img)

    # Create partial sample at index 2 (tensor only, no metadata or image)
    tensor = np.zeros((64, 64, 3), dtype=np.float32)
    np.save(tensors_dir / f"{2:010d}.npy", tensor)

    return res_dir
