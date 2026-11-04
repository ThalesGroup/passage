"""Geographic exclusion must never silently disappear when enabled."""

import json

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

import passage.generate as generate


@pytest.fixture(autouse=True)
def clear_mask(monkeypatch):
    monkeypatch.setattr(generate, "_HOLD_OUT_MASK", None)
    monkeypatch.setattr(generate, "_HOLD_OUT_MASK_PATH", None)


def test_missing_and_malformed_masks_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.load_hold_out_mask(tmp_path / "missing.png")
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.load_hold_out_mask(broken)
    wrong_shape = tmp_path / "wrong.png"
    Image.fromarray(np.zeros((2, 3), dtype=np.uint8)).save(wrong_shape)
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.load_hold_out_mask(wrong_shape)
    inconsistent = tmp_path / "inconsistent.png"
    color = np.zeros((181, 361, 3), dtype=np.uint8)
    color[90, 180, 1] = 255
    Image.fromarray(color).save(inconsistent)
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.load_hold_out_mask(inconsistent)


def test_missing_after_prior_load_does_not_use_stale_cache(tmp_path):
    path = tmp_path / "mask.png"
    Image.fromarray(np.zeros((181, 361), dtype=np.uint8)).save(path)
    generate.load_hold_out_mask(path)
    path.write_bytes(b"broken")
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.load_hold_out_mask(path)


def test_enabled_holdout_fails_before_generation_when_mask_missing(tmp_path, monkeypatch):
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"hold_out": {"enabled": True, "mask_path": str(tmp_path / "missing.png")}}))
    calibration = tmp_path / "calibration.json"
    calibration.write_text(
        json.dumps({"global_min": 0, "global_max": 1, "tile_stats": [{"lat": 0, "lon": 0, "min": 0, "max": 1}]})
    )
    monkeypatch.setattr(generate, "generate_sample", lambda *_: pytest.fail("generation began"))
    with pytest.raises(ValueError, match="hold-out mask"):
        generate.run_generate(str(tmp_path / "out"), str(config), str(tmp_path / "tiles"), str(calibration), num_cpus=0)
    assert not (tmp_path / "out").exists()


def test_worker_initialization_preserves_holdout_filter_without_parent_state():
    mask = np.zeros((181, 361), dtype=np.uint8)
    mask[90, 180] = 1
    generate._initialize_hold_out_worker(mask)
    frame = pd.DataFrame({"lat": [0, 1], "lon": [0, 0]})
    assert generate.filter_tiles_by_hold_out(frame, "train")["lat"].tolist() == [1]
    assert generate.filter_tiles_by_hold_out(frame, "test")["lat"].tolist() == [0]


def test_spawned_worker_uses_initialized_holdout_mask():
    import multiprocessing

    mask = np.zeros((181, 361), dtype=np.uint8)
    mask[90, 180] = 1
    frame = pd.DataFrame({"lat": [0, 1], "lon": [0, 0]})
    with multiprocessing.get_context("spawn").Pool(
        1, initializer=generate._initialize_hold_out_worker, initargs=(mask,)
    ) as pool:
        filtered = pool.apply(generate.filter_tiles_by_hold_out, (frame, "train"))
    assert filtered["lat"].tolist() == [1]
