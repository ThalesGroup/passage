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
    assert "metadata" in features

    # Check dynamic path columns exist for each solver and per-solver sample images
    for solver in solver_names:
        assert f"image_{solver}" in features
        assert f"path_{solver}_free" in features
        assert f"path_{solver}_obstacles" in features
        assert "considered_area" in str(features["metadata"]["paths"][solver]["free"])
        assert "considered_area" in str(features["metadata"]["paths"][solver]["obstacles"])


@pytest.mark.unit
def test_features_no_image_when_not_sample():
    """Test that _features excludes image column when sample=False."""
    solver_names = ["astar_elevation"]

    features = _features(sample=False, solver_names=solver_names)

    assert "id" in features
    assert "tensor" in features
    assert "image" not in features
    assert f"image_{solver_names[0]}" not in features
    assert "metadata" in features
