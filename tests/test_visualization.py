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

import numpy as np
import pytest

from passage.visualization import tensor_to_image


@pytest.mark.parametrize("channels", [2, 3])
def test_tensor_image_retains_hwc_shape_dtype_and_bgr_order(channels):
    tensor = np.zeros((40, 60, channels), dtype=np.float32)
    tensor[..., 1] = -1
    image = tensor_to_image(tensor, path_pixels=[(12, 10), (12, 40)])
    assert image.shape == (40, 60, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[12, 25], [172, 103, 23])  # blue route, BGR


def test_route_uses_vertex_order_without_closing_polyline():
    tensor = np.zeros((60, 60, 3), dtype=np.float32)
    tensor[..., 1] = -1
    plain = tensor_to_image(tensor)
    image = tensor_to_image(tensor, path_pixels=np.array([(10, 10), (10, 45), (45, 45)]))
    np.testing.assert_array_equal(image[10, 25], [172, 103, 23])
    np.testing.assert_array_equal(image[25, 45], [172, 103, 23])
    np.testing.assert_array_equal(image[25, 25], plain[25, 25])


@pytest.mark.parametrize("path", [[(-1, 2)], [(40, 2)], [(1.2, 2)], [(float("nan"), 2)], [1, 2, 3]])
def test_bad_coordinates_raise_instead_of_silent_clipping(path):
    with pytest.raises(ValueError, match="path_pixels"):
        tensor_to_image(np.zeros((40, 60, 2)), path_pixels=path)


def test_obstacles_have_charcoal_style_and_markers_are_white():
    tensor = np.zeros((64, 64, 3), dtype=np.float32)
    tensor[..., 1] = -1
    tensor[20:45, 20:45, 2] = 1
    tensor[10, 10, 1] = 0
    tensor[50, 50, 1] = 1
    image = tensor_to_image(tensor)
    np.testing.assert_array_equal(image[32, 32], [58, 56, 48])
    np.testing.assert_array_equal(image[10, 10], [255, 255, 255])
    np.testing.assert_array_equal(image[50, 50], [255, 255, 255])
