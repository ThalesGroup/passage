"""Shared, calibrated plan-view plots of PASSAGE terrain and ordered routes.

Coordinates are always integer ``[row, column]`` cells. Renderers display the
original raster without smoothing and preserve the supplied route order. They
validate geometry for display, but do not solve routes or establish optimality.
Matplotlib figures belong to the caller, who should close them after saving.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from matplotlib import patheffects, pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb

TERRAIN_COLORS = ("#758f83", "#b7c6a2", "#d8d5ac", "#d6b58f", "#f3e8d5")
ROUTE_COLORS = ("#1767ac", "#d56418", "#a32b8e")
OBSTACLE_COLOR = "#30383a"
MARKER_EDGE = "#18232b"
COSTS = ("elevation", "slope", "energy")
COST_LABELS = {"elevation": "Elevation", "slope": "Slope (uphill)", "energy": "Energy"}
REGIMES = ("free", "obstacles")
MAX_DISPLAY_CELLS = 512


def terrain_colormap() -> LinearSegmentedColormap:
    """Return the sage–sand elevation palette used by figures and BGR previews."""
    return LinearSegmentedColormap.from_list("passage_sage_sand", TERRAIN_COLORS)


def validate_coordinates(points: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    """Validate ordered integer cells without truncating or clipping coordinates."""
    try:
        array = np.asarray(points, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer [row, column] cells") from exc
    if array.size == 0 and array.shape in ((0,), (0, 2)):
        return np.empty((0, 2), dtype=np.int32)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (N, 2), got {array.shape}")
    if not np.isfinite(array).all() or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain finite integer coordinates")
    if np.any(array < 0) or np.any(array >= np.asarray(shape)):
        raise ValueError(f"{name} contains coordinates outside raster shape {shape}")
    return array.astype(np.int32)


def elevation_normalization(elevation: np.ndarray, elev_range: tuple[float, float] | None = None) -> Normalize:
    """Use one explicit linear range, or the finite scene extrema."""
    if elevation.ndim != 2 or not all(elevation.shape) or not np.isfinite(elevation).all():
        raise ValueError("elevation must be a nonempty finite two-dimensional raster")
    if elev_range is None:
        low, high = float(elevation.min()), float(elevation.max())
    else:
        try:
            low, high = map(float, elev_range)
        except (TypeError, ValueError) as exc:
            raise ValueError("elev_range must contain two finite increasing bounds") from exc
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError("elev_range must contain two finite increasing bounds")
    return Normalize(vmin=low, vmax=high, clip=True)


def decode_scene(row: Mapping[str, Any], *, tensor: np.ndarray | None = None) -> tuple:
    """Decode a dataset row to elevation metres, obstacle mask, start and goal.

    ``row['tensor']`` may be a decoded HWC array or zstd float32 bytes. A supplied
    ``tensor`` is the normalized HWC array, never an already calibrated raster.
    Recorded calibration is required; normalized values are not labelled metres.
    """
    metadata = row["metadata"]
    try:
        calibration = metadata["calibration"]
        low, high = float(calibration["min"]), float(calibration["max"])
        if not np.isfinite([low, high]).all() or high <= low:
            raise ValueError
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("finite increasing calibration min/max are required to decode elevation metres") from exc
    if tensor is None:
        raw = row["tensor"]
        if isinstance(raw, (bytes, bytearray, memoryview)):
            import zstandard

            resolution = metadata["resolution"]
            shape = (resolution, resolution) if np.isscalar(resolution) else tuple(resolution)
            if len(shape) != 2 or not all(isinstance(n, int) and n > 0 for n in shape):
                raise ValueError("metadata resolution must identify a positive raster shape")
            byte_count = int(np.prod(shape)) * 3 * 4
            data = zstandard.ZstdDecompressor().decompress(raw, max_output_size=byte_count)
            if len(data) != byte_count:
                raise ValueError("tensor byte count does not match metadata resolution")
            tensor = np.frombuffer(data, dtype="<f4").reshape(*shape, 3)
        else:
            tensor = np.asarray(raw)
    tensor = np.asarray(tensor)
    if tensor.ndim != 3 or tensor.shape[2] not in (2, 3) or not np.isfinite(tensor).all():
        raise ValueError("tensor must be a finite HWC array with two or three channels")
    shape = tensor.shape[:2]
    markers = metadata["markers"]
    start, goal = [
        validate_coordinates([[markers[key]["i"], markers[key]["j"]]], shape, key)[0]
        for key in ("start_point", "end_point")
    ]
    elevation = tensor[..., 0].astype(np.float64) * (high - low) + low
    obstacles = tensor[..., 2] > 0.5 if tensor.shape[2] == 3 else np.zeros(shape, dtype=bool)
    return elevation, obstacles, start, goal


def _legend_handles(labels: Sequence[str], colors: Sequence[str]) -> list:
    handles = [Line2D([], [], color=color, linestyle="-", lw=1.8, label=label) for label, color in zip(labels, colors)]
    handles.extend(
        [
            Patch(facecolor=OBSTACLE_COLOR, edgecolor="#e4e7e2", hatch="////", label="Obstacle"),
            Line2D(
                [],
                [],
                marker="o",
                color="none",
                markerfacecolor="white",
                markeredgecolor=MARKER_EDGE,
                markersize=6,
                label="Start",
            ),
            Line2D(
                [],
                [],
                marker="*",
                color="none",
                markerfacecolor="white",
                markeredgecolor=MARKER_EDGE,
                markersize=9,
                label="Goal",
            ),
        ]
    )
    return handles


def _draw_scene(
    ax: Axes,
    elevation: np.ndarray,
    obstacles: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray,
    paths: Mapping[str, np.ndarray],
    colors: Sequence[str],
    norm: Normalize,
):
    h, w = elevation.shape
    display_h, display_w = min(h, MAX_DISPLAY_CELLS), min(w, MAX_DISPLAY_CELLS)
    extent = [-0.5, w - 0.5, h - 0.5, -0.5]
    reduced = (display_h, display_w) != (h, w)
    if reduced:
        # Keep source coordinates and extent. Select actual elevation cells at
        # each display bin's centre; do not average/smooth elevation values.
        row_edges = np.arange(display_h + 1) * h / display_h
        col_edges = np.arange(display_w + 1) * w / display_w
        rows = np.floor((np.arange(display_h) + 0.5) * h / display_h).astype(int)
        cols = np.floor((np.arange(display_w) + 0.5) * w / display_w).astype(int)
        display_elevation = elevation[np.ix_(rows, cols)]
        # Any occupied source cell marks its display bin. This conservative
        # display can enlarge visible obstacles; it never changes route data.
        row_starts = np.floor(row_edges[:-1]).astype(int)
        col_starts = np.floor(col_edges[:-1]).astype(int)
        # Include the final cell intersected by a fractional display-bin edge.
        # A cell straddling two bins is conservatively occupied in both.
        row_ends = np.ceil(row_edges[1:]).astype(int) - 1
        col_ends = np.ceil(col_edges[1:]).astype(int) - 1
        pooled_rows = np.logical_or.reduceat(obstacles, row_starts, axis=0) | obstacles[row_ends, :]
        display_obstacles = np.logical_or.reduceat(pooled_rows, col_starts, axis=1) | pooled_rows[:, col_ends]
    else:
        display_elevation, display_obstacles = elevation, obstacles
    base = ax.imshow(
        display_elevation, cmap=terrain_colormap(), norm=norm, interpolation="nearest", origin="upper", extent=extent
    )
    if display_obstacles.any():
        # RGBA cell raster avoids an unbounded collection of vector polygons.
        # At <=512 cells every original obstacle footprint is retained exactly.
        overlay = np.zeros((*display_obstacles.shape, 4), dtype=np.float32)
        overlay[display_obstacles, :3] = to_rgb(OBSTACLE_COLOR)
        overlay[display_obstacles, 3] = 0.8
        rows, cols = np.nonzero(display_obstacles)
        hatch = (rows + cols) % 7 < 1
        overlay[rows[hatch], cols[hatch], :3] = to_rgb("#e4e7e2")
        ax.imshow(overlay, interpolation="nearest", origin="upper", extent=extent, zorder=3)
    for (label, path), color in zip(paths.items(), colors):
        (line,) = ax.plot(
            path[:, 1], path[:, 0], color=color, linestyle="-", lw=1.65, label=label, solid_capstyle="round", zorder=5
        )
        line.set_path_effects([patheffects.Stroke(linewidth=3.25, foreground="white"), patheffects.Normal()])
    ax.scatter(start[1], start[0], marker="o", s=38, facecolor="white", edgecolor=MARKER_EDGE, lw=1, zorder=8)
    ax.scatter(goal[1], goal[0], marker="*", s=75, facecolor="white", edgecolor=MARKER_EDGE, lw=0.9, zorder=8)
    ax.set_xlim(-3, w + 2)
    ax.set_ylim(h + 2, -3)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#b3b9b2")
        spine.set_linewidth(0.65)
    if reduced:
        ax.text(
            0.5,
            0.99,
            f"Display {display_h}×{display_w}: nearest elevation, any-occupied obstacles",
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=6,
            color="#5c6765",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
            zorder=10,
        )
    return base


def render_scene(
    elevation: np.ndarray,
    obstacles: np.ndarray | None,
    start: Sequence[int],
    goal: Sequence[int],
    paths: Mapping[str, Sequence[Sequence[int]]],
    *,
    route_labels: Mapping[str, str] | None = None,
    color_by: str = "cost objective",
    elevation_units: str = "m",
    elev_range: tuple[float, float] | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    legend: bool = True,
    colorbar: bool = True,
) -> Figure:
    """Render a scene; supplied elevation must already be in ``elevation_units``.

    ``color_by`` declares whether route colors denote algorithms or objectives.
    Up to three routes use distinct blue/orange/magenta colors. Empty routes are
    retained as empty lines; invalid coordinates raise rather than being clipped.
    For raw dataset rows use :func:`decode_scene` or :func:`render_cost_comparison`.
    Large display rasters are bounded to 512 cells per dimension, using nearest
    source elevation values and any-occupied obstacle bins; this display-only
    reduction is labelled, and routes/markers retain original cell coordinates.
    """
    elevation = np.asarray(elevation)
    norm = elevation_normalization(elevation, elev_range)
    shape = elevation.shape
    obstacles = np.zeros(shape, dtype=bool) if obstacles is None else np.asarray(obstacles) > 0.5
    if obstacles.shape != shape:
        raise ValueError("obstacles must have the same shape as elevation")
    if len(paths) > len(ROUTE_COLORS):
        raise ValueError("at most three routes can be distinguished in one scene")
    start = validate_coordinates([start], shape, "start")[0]
    goal = validate_coordinates([goal], shape, "goal")[0]
    paths = {name: validate_coordinates(path, shape, name) for name, path in paths.items()}
    if ax is None:
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        fig.subplots_adjust(left=0.06, right=0.83, bottom=0.18, top=0.92)
    else:
        fig = ax.figure
    base = _draw_scene(ax, elevation, obstacles, start, goal, paths, ROUTE_COLORS, norm)
    if title:
        ax.set_title(title, fontsize=11)
    if colorbar:
        fig.colorbar(base, ax=ax, fraction=0.046, pad=0.03).set_label(f"Elevation ({elevation_units})")
    if legend:
        labels = [(route_labels or {}).get(name, name) for name in paths]
        fig.legend(
            handles=_legend_handles(labels, ROUTE_COLORS),
            loc="lower center",
            ncol=min(6, len(paths) + 3),
            frameon=False,
            fontsize=8,
            title=f"Route color: {color_by}",
            title_fontsize=8,
        )
    return fig


def render_cost_comparison(
    row: Mapping[str, Any],
    *,
    tensor: np.ndarray | None = None,
    title: str | None = None,
    route_labels: Mapping[str, str] | None = None,
) -> Figure:
    """Render six stored A* routes on one calibrated terrain scale in a 2×3 grid.

    ``route_labels`` overrides the ``elevation``, ``slope``, and ``energy`` labels.
    The first row hides the obstacle mask; the second shows it. No solvers run.
    """
    elevation, obstacles, start, goal = decode_scene(row, tensor=tensor)
    norm = elevation_normalization(elevation)
    paths = {
        f"path_astar_{cost}_{regime}": validate_coordinates(
            row[f"path_astar_{cost}_{regime}"], elevation.shape, f"{cost}_{regime} route"
        )
        for regime in REGIMES
        for cost in COSTS
    }
    labels = {**COST_LABELS, **(route_labels or {})}
    with plt.rc_context(
        {"font.family": "DejaVu Sans", "font.size": 9, "hatch.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42}
    ):
        fig, axes = plt.subplots(2, 3, figsize=(8.6, 6.35))
        fig.subplots_adjust(left=0.105, right=0.925, bottom=0.16, top=0.895, wspace=0.055, hspace=0.06)
        for ri, regime in enumerate(REGIMES):
            for ci, cost in enumerate(COSTS):
                ax = axes[ri, ci]
                path = paths[f"path_astar_{cost}_{regime}"]
                mask = obstacles if regime == "obstacles" else np.zeros_like(obstacles)
                base = _draw_scene(ax, elevation, mask, start, goal, {cost: path}, [ROUTE_COLORS[ci]], norm)
                if ri == 0:
                    ax.set_title(labels[cost], pad=7, color=ROUTE_COLORS[ci], fontsize=11, fontweight="semibold")
                ax.text(
                    0.025,
                    0.035,
                    f"{len(path)} vertices",
                    transform=ax.transAxes,
                    fontsize=7.5,
                    color=OBSTACLE_COLOR,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 2},
                )
            axes[ri, 0].set_ylabel(("Without obstacles", "With obstacles")[ri], fontsize=10, labelpad=13)
        cax = fig.add_axes([0.944, 0.18, 0.012, 0.68])
        bar = fig.colorbar(base, cax=cax)
        bar.set_label("Elevation (m)", fontsize=9, labelpad=8)
        bar.ax.tick_params(labelsize=8, width=0.5, length=2)
        bar.outline.set_linewidth(0.5)
        fig.legend(
            handles=_legend_handles([labels[cost] for cost in COSTS], ROUTE_COLORS),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncol=6,
            frameon=False,
            columnspacing=1.2,
            handlelength=2.3,
            fontsize=8.5,
            title="Route color: cost objective",
            title_fontsize=8,
        )
        fig.text(
            0.105,
            0.975,
            title or "Same terrain, different routing costs",
            fontsize=14,
            weight="semibold",
            color="#243537",
        )
        h, w = elevation.shape
        fig.text(
            0.105, 0.941, f"{h} × {w} • sample {row.get('id', '')} • stored A* routes", fontsize=9, color="#5c6765"
        )
        fig.text(
            0.5,
            0.024,
            "One elevation scale across all panels; route optimality was not revalidated.",
            ha="center",
            fontsize=8,
            color="#5c6765",
        )
    return fig
