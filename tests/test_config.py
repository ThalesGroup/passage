from pathlib import Path

import yaml


def test_config_has_obstacle_section():
    config_path = Path(__file__).resolve().parents[1] / "config" / "config.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    obstacles = cfg.get("obstacles", {})
    ratio_max = obstacles.get("ratio_max", None)
    size_min_ratio = obstacles.get("size_min_ratio", None)
    size_max_ratio = obstacles.get("size_max_ratio", None)
    generation = obstacles.get("generation", {})

    assert ratio_max is not None, "config.yaml should define obstacles.ratio_max"
    ratio_max = float(ratio_max)
    assert 0.0 <= ratio_max <= 1.0, "obstacles.ratio_max must be between 0 and 1"

    assert size_min_ratio is not None, "config.yaml should define obstacles.size_min_ratio"
    assert size_max_ratio is not None, "config.yaml should define obstacles.size_max_ratio"
    size_min_ratio = float(size_min_ratio)
    size_max_ratio = float(size_max_ratio)
    assert 0.0 < size_min_ratio <= size_max_ratio <= 1.0, "obstacle size ratios must be within (0, 1]"

    # generation params
    assert isinstance(generation, dict), "obstacles.generation must be a mapping"
    candidates_per_mask = generation.get("candidates_per_mask", None)
    max_attempts = generation.get("max_attempts", None)

    assert candidates_per_mask is not None and int(candidates_per_mask) > 0, (
        "generation.candidates_per_mask must be > 0"
    )
    assert max_attempts is not None and int(max_attempts) > 0, "generation.max_attempts must be > 0"

    # ensure obstacle penalty present and positive
    obstacle_penalty = obstacles.get("obstacle_penalty", None)
    assert obstacle_penalty is not None and float(obstacle_penalty) > 0.0, "obstacle_penalty must be a positive number"

    # defaults should be reasonably sized (per user request)
    assert size_min_ratio == 0.20 and size_max_ratio == 0.50, "Default size ratios should be 0.20 and 0.50"
