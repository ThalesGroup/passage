import pytest

from passage.export import _features


@pytest.mark.unit
def test_features_builds_dynamic_path_columns():
    """Test that _features creates dynamic path columns for each solver."""
    solver_names = ["astar_elevation", "astar_energy"]

    features = _features(sample=True, solver_names=solver_names)

    # Check basic columns exist
    assert "id" in features
    assert "tensor" in features
    assert "image" in features
    assert "metadata" in features

    # Check dynamic path columns exist for each solver
    for solver in solver_names:
        assert f"path_{solver}_free" in features
        assert f"path_{solver}_obstacles" in features


@pytest.mark.unit
def test_features_no_image_when_not_sample():
    """Test that _features excludes image column when sample=False."""
    solver_names = ["astar_elevation"]

    features = _features(sample=False, solver_names=solver_names)

    assert "id" in features
    assert "tensor" in features
    assert "image" not in features
    assert "metadata" in features
