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

"""Raster previews compatible with OpenCV's HWC uint8 BGR convention."""

import cv2
import numpy as np
from matplotlib.colors import to_rgb

from passage.plotting import (
    MARKER_EDGE,
    OBSTACLE_COLOR,
    ROUTE_COLORS,
    elevation_normalization,
    terrain_colormap,
    validate_coordinates,
)


def _bgr(hex_color: str) -> tuple[int, int, int]:
    return tuple(round(channel * 255) for channel in reversed(to_rgb(hex_color)))


def tensor_to_image(
    tensor: np.ndarray,
    path_pixels: list[tuple[int, int]] | None = None,
    alt_path_pixels: list[tuple[int, int]] | None = None,
    elev_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Convert HWC elevation/markers[/obstacles] to an HWC uint8 BGR preview.

    Both route arguments contain ordered integer ``(row, column)`` vertices;
    the first is blue and the alternative orange, with white halos. Coordinates
    must be finite integers within the raster. Invalid geometry raises instead
    of being clipped. This is a display validation, not a route legality check.

    Elevation uses the shared sage–sand scale; ``elev_range`` is expressed in
    the same units as channel zero. Unlabelled raster previews may use normalized
    elevation. For labelled metre plots, use ``passage.plotting.decode_scene``.
    Obstacles are charcoal with light hatching. White circles/stars mark start
    and goal, above all paths. The call signature and BGR output are preserved.
    """
    tensor = np.asarray(tensor)
    if tensor.ndim != 3 or tensor.shape[2] not in (2, 3) or not all(tensor.shape[:2]):
        raise ValueError(f"Expected tensor of shape (H, W, 2) or (H, W, 3), got {tensor.shape}")
    if not np.isfinite(tensor).all():
        raise ValueError("tensor must contain finite values")
    elevation, markers = tensor[..., 0], tensor[..., 1]
    shape = elevation.shape
    paths = [
        validate_coordinates(points if points is not None else [], shape, name)
        for points, name in ((path_pixels, "path_pixels"), (alt_path_pixels, "alt_path_pixels"))
    ]
    norm = elevation_normalization(elevation, elev_range)
    img = (terrain_colormap()(norm(elevation))[..., :3] * 255).astype(np.uint8)[..., ::-1].copy()
    obstacles = tensor[..., 2] > 0.5 if tensor.shape[2] == 3 else np.zeros(shape, dtype=bool)
    img[obstacles] = _bgr(OBSTACLE_COLOR)
    scale = max(shape) / 64
    hatch_spacing = max(8, round(scale * 5))
    # Hatch coordinates are raster cells; the mask retains its exact footprint.
    rows, cols = np.nonzero(obstacles)
    hatch = (rows + cols) % hatch_spacing == 3
    img[rows[hatch], cols[hatch]] = _bgr("#a3aaa5")
    thickness = max(1, round(scale * 0.5))
    for path, color in zip(paths, ROUTE_COLORS):
        if not len(path):
            continue
        xy = np.ascontiguousarray(path[:, ::-1]).reshape(-1, 1, 2)
        if len(path) == 1:
            center = tuple(int(n) for n in xy[0, 0])
            cv2.circle(img, center, thickness + 1, (255, 255, 255), -1)
            cv2.circle(img, center, max(1, thickness // 2), _bgr(color), -1)
        else:
            cv2.polylines(img, [xy], False, (255, 255, 255), thickness + 2)
            cv2.polylines(img, [xy], False, _bgr(color), thickness)
    has_marker_encoding = np.any(markers > 0.5) or np.any(markers < -0.5)
    start_mask = np.isclose(markers, 0) if has_marker_encoding else np.zeros(shape, dtype=bool)
    radius = max(2, round(scale * 1.3))
    for row, col in np.argwhere(start_mask):
        center = (int(col), int(row))
        cv2.circle(img, center, radius + 1, _bgr(MARKER_EDGE), -1)
        cv2.circle(img, center, radius, (255, 255, 255), -1)
    for row, col in np.argwhere(markers > 0.5):
        angle = np.arange(10) * np.pi / 5 - np.pi / 2
        radii = np.where(np.arange(10) % 2 == 0, radius * 2, radius * 0.85)
        xy = np.column_stack((col + np.cos(angle) * radii, row + np.sin(angle) * radii)).round().astype(np.int32)
        cv2.fillPoly(img, [xy], (255, 255, 255))
        cv2.polylines(img, [xy], True, _bgr(MARKER_EDGE), 1)
    return img
