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

"""Visualization helpers for PASSAGE."""

import cv2
import numpy as np
from loguru import logger


def tensor_to_image(
    tensor: np.ndarray,
    path_pixels: list[tuple[int, int]] | None = None,
    alt_path_pixels: list[tuple[int, int]] | None = None,
    elev_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Convert a tensor to a BGR image for visualization.

    Supports either:
        - 2-channel tensors (elevation, markers) with the path provided via
            `path_pixels` (callers should provide extracted path pixels explicitly).
        - 3-channel tensors (elevation, markers, obstacles) where the third channel is
            treated as obstacles; path pixels should still be provided via `path_pixels`.

    This function now supports drawing two paths at once for comparison:
        - `path_pixels`: primary path (drawn in yellow by default)
        - `alt_path_pixels`: secondary path (drawn in green by default)

    Color mapping and overlays:
        - Elevation -> rendered with matplotlib 'terrain' colormap (base image)
        - Markers -> highlighted (start_point lime, end_point cyan)
        - Primary path -> yellow overlay (drawn as polyline/contours)
        - Secondary path -> green overlay (drawn as polyline/contours and drawn last on top)
        - Obstacles -> red overlay (drawn before paths)

    Args:
        tensor: float32 array of shape (H, W, 2) or (H, W, 3).
        path_pixels: Optional list of (i, j) tuples indicating primary path pixels.
        alt_path_pixels: Optional list of (i, j) tuples indicating secondary path pixels.
        elev_range: Optional (min, max) tuple to use for elevation normalization. If None, tensor min/max are used.

    Returns:
        BGR image as uint8 array of shape (H, W, 3).
    """
    if tensor.ndim != 3 or tensor.shape[2] not in (2, 3):
        raise ValueError(f"Expected tensor of shape (H, W, 2) or (H, W, 3), got {tensor.shape}")

    elevation_channel = tensor[:, :, 0]
    marker_channel = tensor[:, :, 1]

    # Debug: log elevation min/max before normalization
    elev_raw_min = float(np.min(elevation_channel))
    elev_raw_max = float(np.max(elevation_channel))
    logger.debug(f"tensor_to_image: elevation BEFORE normalization: min={elev_raw_min:.4f}, max={elev_raw_max:.4f}")

    # Third channel is assumed to be obstacles by default (tensor shape 3)
    if tensor.shape[2] == 3:
        obstacles_channel = tensor[:, :, 2]
    else:
        obstacles_channel = np.zeros_like(elevation_channel, dtype=np.float32)

    # Build path channels from provided path lists (callers must provide them when relevant)
    path_channel = np.zeros_like(elevation_channel, dtype=np.float32)
    alt_path_channel = np.zeros_like(elevation_channel, dtype=np.float32)
    if path_pixels:
        for pi, pj in path_pixels:
            if 0 <= int(pi) < path_channel.shape[0] and 0 <= int(pj) < path_channel.shape[1]:
                path_channel[int(pi), int(pj)] = 1.0
    if alt_path_pixels:
        for pi, pj in alt_path_pixels:
            if 0 <= int(pi) < alt_path_channel.shape[0] and 0 <= int(pj) < alt_path_channel.shape[1]:
                alt_path_channel[int(pi), int(pj)] = 1.0

    # Normalize elevation using provided elev_range tuple (min, max) in same units as elevation_channel
    if (
        elev_range is not None
        and isinstance(elev_range, tuple)
        and len(elev_range) == 2
        and elev_range[1] > elev_range[0]
    ):
        emn, emx = float(elev_range[0]), float(elev_range[1])
        norm = (elevation_channel - emn) / (emx - emn)
        logger.debug(f"tensor_to_image: using provided elevation range: [{emn:.1f}, {emx:.1f}]")
    else:
        e_min, e_max = elev_raw_min, elev_raw_max
        if e_max > e_min:
            norm = (elevation_channel - e_min) / (e_max - e_min)
        else:
            # If the elevation channel is constant and already in [0, 1], use it directly
            if 0.0 <= e_min <= 1.0:
                norm = elevation_channel.astype(np.float32)
            else:
                norm = np.zeros_like(elevation_channel, dtype=np.float32)

    norm = np.clip(norm, 0.0, 1.0)

    # Debug: log elevation min/max after normalization
    norm_min = float(np.min(norm))
    norm_max = float(np.max(norm))
    logger.debug(f"tensor_to_image: elevation AFTER normalization: min={norm_min:.4f}, max={norm_max:.4f}")

    # Use matplotlib 'terrain' colormap for elevation visualization (RGB)
    # If caller provided elev_range tuple, rescale using those bounds (they must be in same units as elevation_channel)
    try:
        import matplotlib as mpl

        if (
            elev_range is not None
            and isinstance(elev_range, tuple)
            and len(elev_range) == 2
            and elev_range[1] > elev_range[0]
        ):
            emn, emx = float(elev_range[0]), float(elev_range[1])
            norm_local = (elevation_channel - emn) / (emx - emn)
            norm_local = np.clip(norm_local, 0.0, 1.0)
        else:
            norm_local = norm

        # Use non-deprecated colormap access
        terrain_rgb = (mpl.colormaps.get_cmap("terrain")(norm_local)[:, :, :3] * 255).astype(np.uint8)
        # Convert RGB -> BGR for OpenCV
        img = terrain_rgb[:, :, ::-1].copy()
    except Exception:
        # Fallback: grayscale elevation in Blue channel
        elev_scaled = (norm * 255).astype(np.uint8)
        img = cv2.merge([elev_scaled, elev_scaled, elev_scaled])

    # Compute obstacle mask and log counts (assignment happens after path/markers to preserve overlay priority)
    obst_mask = obstacles_channel > 0.5
    has_end = bool(np.any(marker_channel > 0.5))
    has_background = bool(np.any(marker_channel < -0.5))
    start_mask = (
        np.isclose(marker_channel, 0.0) if (has_end or has_background) else np.zeros_like(marker_channel, dtype=bool)
    )
    end_mask = marker_channel > 0.5
    logger.debug(
        "tensor_to_image: obst_count=%d, path_count=%d, start_count=%d, end_count=%d",
        int(np.sum(obst_mask)),
        int(np.sum(path_channel > 0.5)),
        int(np.sum(start_mask)),
        int(np.sum(end_mask)),
    )

    # Determine resolution and scale factors (base reference = 64)
    H, W = elevation_channel.shape
    res = max(H, W)
    base = 64
    scale = float(res) / float(base) if base > 0 else 1.0

    # Path thickness scales with resolution but made thinner: 1px at 64 -> round(scale * 0.5)
    path_thickness = max(1, int(round(scale * 0.5)))

    # Marker plus arm length unchanged; marker thickness reduced for thinner markers
    marker_arm = max(1, int(round(scale)))
    marker_thickness = max(1, int(round(max(1, scale * 0.25))))

    # Colors (BGR)
    obst_color = (0, 0, 255)  # red
    start_color = (255, 255, 0)  # cyan
    end_color = (255, 0, 0)  # blue
    # Primary path colors: polyline and single-pixel points are yellow (path without obstacles)
    path_poly_color = (0, 255, 255)  # yellow (B low, G high, R high)
    path_point_color = path_poly_color  # yellow for single-pixel path points
    # Alternate/with-obstacles path uses green and is drawn last on top
    alt_poly_color = (0, 255, 0)  # green

    # 1) Draw obstacles on top of elevation
    if np.any(obst_mask):
        img[obst_mask] = obst_color

    # 2) Draw start and end markers (plus-signs) without white surrounding
    start_positions = np.argwhere(start_mask)
    end_positions = np.argwhere(end_mask)
    for mi, mj in start_positions:
        cx, cy = int(mj), int(mi)  # (x, y)
        cv2.line(img, (cx - marker_arm, cy), (cx + marker_arm, cy), color=start_color, thickness=marker_thickness)
        cv2.line(img, (cx, cy - marker_arm), (cx, cy + marker_arm), color=start_color, thickness=marker_thickness)
    for mi, mj in end_positions:
        cx, cy = int(mj), int(mi)  # (x, y)
        cv2.line(img, (cx - marker_arm, cy), (cx + marker_arm, cy), color=end_color, thickness=marker_thickness)
        cv2.line(img, (cx, cy - marker_arm), (cx, cy + marker_arm), color=end_color, thickness=marker_thickness)

    # 3) Draw path without obstacles (polyline yellow; single point green)
    path_mask = (path_channel > 0.5).astype(np.uint8) * 255
    if np.any(path_mask):
        try:
            contours, _ = cv2.findContours(path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue
                if len(cnt) >= 2:
                    cv2.polylines(img, [cnt], isClosed=False, color=path_poly_color, thickness=path_thickness)
                else:
                    x, y = int(cnt[0][0][0]), int(cnt[0][0][1])
                    cv2.circle(img, (x, y), radius=max(1, path_thickness // 2), color=path_point_color, thickness=-1)
        except Exception:
            ys, xs = np.nonzero(path_channel > 0.5)
            for y, x in zip(ys, xs):
                cv2.circle(
                    img, (int(x), int(y)), radius=max(1, path_thickness // 2), color=path_point_color, thickness=-1
                )

    # 4) Draw path with obstacles (alt path) on top
    alt_mask = (alt_path_channel > 0.5).astype(np.uint8) * 255
    if np.any(alt_mask):
        try:
            contours, _ = cv2.findContours(alt_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue
                if len(cnt) >= 2:
                    cv2.polylines(img, [cnt], isClosed=False, color=alt_poly_color, thickness=path_thickness)
                else:
                    x, y = int(cnt[0][0][0]), int(cnt[0][0][1])
                    cv2.circle(img, (x, y), radius=max(1, path_thickness // 2), color=alt_poly_color, thickness=-1)
        except Exception:
            ys, xs = np.nonzero(alt_path_channel > 0.5)
            for y, x in zip(ys, xs):
                cv2.circle(
                    img, (int(x), int(y)), radius=max(1, path_thickness // 2), color=alt_poly_color, thickness=-1
                )

    return img
