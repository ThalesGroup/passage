"""Visualization helpers for PASSAGE."""

import cv2
import numpy as np
from loguru import logger


def tensor_to_image(
    tensor: np.ndarray,
    path_pixels: list[tuple[int, int]] | None = None,
    elev_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """
    Convert a tensor to a BGR image for visualization.

    Supports either:
        - 2-channel tensors (elevation, markers) with the path provided via
            `path_pixels` (callers should provide extracted path pixels explicitly).
        - 3-channel tensors (elevation, markers, obstacles) where the third channel is
            treated as obstacles; path pixels should still be provided via `path_pixels`.

    Color mapping and overlays:
        - Elevation -> rendered with matplotlib 'terrain' colormap (base image)
        - Markers -> highlighted (first marker lime, second cyan)
        - Path -> green overlay (drawn as polyline/contours)
        - Obstacles -> red overlay (alpha blended on top of terrain)

    Args:
        tensor: float32 array of shape (H, W, 2) or (H, W, 3).
        path_pixels: Optional list of (i, j) tuples indicating path pixels.
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

    # Build path channel from provided `path_pixels` (callers must provide it when relevant)
    path_channel = np.zeros_like(elevation_channel, dtype=np.float32)
    if path_pixels:
        for pi, pj in path_pixels:
            if 0 <= int(pi) < path_channel.shape[0] and 0 <= int(pj) < path_channel.shape[1]:
                path_channel[int(pi), int(pj)] = 1.0

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
    logger.debug(
        f"tensor_to_image: obst_count={int(np.sum(obst_mask))}, path_count={int(np.sum(path_channel > 0.5))}, marker_count={int(np.sum(marker_channel > 0.5))}"
    )

    # Draw path (green) from path_channel or explicit pixels
    path_mask = (path_channel > 0.5).astype(np.uint8) * 255
    if np.any(path_mask):
        try:
            contours, _ = cv2.findContours(path_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:
                if cnt is None or len(cnt) == 0:
                    continue
                if len(cnt) >= 2:
                    cv2.polylines(img, [cnt], isClosed=False, color=(0, 255, 0), thickness=2)
                else:
                    x, y = int(cnt[0][0][0]), int(cnt[0][0][1])
                    cv2.circle(img, (x, y), radius=1, color=(0, 255, 0), thickness=-1)
        except Exception:
            # Last-resort: draw individual path pixels
            ys, xs = np.nonzero(path_channel > 0.5)
            for y, x in zip(ys, xs):
                cv2.circle(img, (int(x), int(y)), radius=1, color=(0, 255, 0), thickness=-1)

    # Draw markers: first marker lime, second marker cyan (if present)
    marker_positions = np.argwhere(marker_channel > 0.5)
    for idx, (mi, mj) in enumerate(marker_positions):
        center = (int(mj), int(mi))  # (x, y)
        color = (0, 255, 0) if idx == 0 else (255, 255, 0)  # BGR: lime, cyan-like
        cv2.circle(img, center, radius=6, color=color, thickness=-1)
        cv2.circle(img, center, radius=7, color=(255, 255, 255), thickness=1)

    # Re-apply obstacles on top so they are visually dominant
    if np.any(obst_mask):
        img[obst_mask] = (0, 0, 255)

    return img
