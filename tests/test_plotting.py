import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest
import zstandard


def scene_row():
    tensor = np.zeros((5, 7, 3), dtype=np.float32)
    tensor[..., 0] = np.linspace(0, 1, 35).reshape(5, 7)
    tensor[..., 1] = -1
    tensor[1, 1, 1] = 0
    tensor[3, 5, 1] = 1
    tensor[0, 6, 2] = 1
    row = {
        "id": "example",
        "metadata": {
            "resolution": [5, 7],
            "calibration": {"min": 100, "max": 300},
            "markers": {"start_point": {"i": 1, "j": 1}, "end_point": {"i": 3, "j": 5}},
        },
        "tensor": zstandard.ZstdCompressor().compress(tensor.tobytes()),
    }
    for cost in ("elevation", "slope", "energy"):
        for regime in ("free", "obstacles"):
            row[f"path_astar_{cost}_{regime}"] = [[1, 1], [1, 5], [3, 5]]
    return row, tensor


def test_scene_preserves_row_col_order_solid_routes_and_uninterpolated_terrain():
    from passage.plotting import render_scene

    path = np.array([[1, 1], [1, 5], [3, 5]])
    fig = render_scene(
        np.arange(35).reshape(5, 7),
        np.zeros((5, 7)),
        (1, 1),
        (3, 5),
        {"reference": path},
        color_by="algorithm",
        route_labels={"reference": "A*"},
    )
    ax = fig.axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 5, 5])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1, 1, 3])
    assert ax.lines[0].get_linestyle() == "-"
    assert len(ax.lines[0].get_path_effects()) == 2
    assert ax.images[0].get_interpolation() == "nearest"
    assert ax.images[0].origin == "upper"
    assert ax.get_ylim()[0] > ax.get_ylim()[1]
    assert fig.axes[-1].get_ylabel() == "Elevation (m)"
    assert fig.legends[0].get_title().get_text() == "Route color: algorithm"
    assert fig.legends[0].get_texts()[0].get_text() == "A*"
    plt.close(fig)


def test_decode_row_uses_calibration_and_same_scene_scale_in_six_panels():
    from passage.plotting import decode_scene, render_cost_comparison

    row, tensor = scene_row()
    elevation, obstacles, start, goal = decode_scene(row)
    np.testing.assert_allclose(elevation, tensor[..., 0] * 200 + 100)
    assert obstacles[0, 6]
    assert tuple(start) == (1, 1)
    assert tuple(goal) == (3, 5)
    fig = render_cost_comparison(row)
    assert len(fig.axes) == 7
    for ax in fig.axes[:6]:
        assert ax.images[0].norm.vmin == 100
        assert ax.images[0].norm.vmax == 300
        assert ax.lines[0].get_linestyle() == "-"
    assert len({ax.lines[0].get_color() for ax in fig.axes[:3]}) == 3
    assert len(fig.legends) == 1
    plt.close(fig)


@pytest.mark.parametrize("path", [[[-1, 1]], [[1.5, 1]], [[1, 7]], [[1, float("nan")]]])
def test_scene_rejects_malformed_routes(path):
    from passage.plotting import render_scene

    with pytest.raises(ValueError, match="route"):
        render_scene(np.zeros((5, 7)), None, (1, 1), (3, 5), {"route": path})


def test_missing_calibration_cannot_be_labelled_metres():
    from passage.plotting import decode_scene

    row, tensor = scene_row()
    del row["metadata"]["calibration"]
    with pytest.raises(ValueError, match="calibration"):
        decode_scene(row, tensor=tensor)


def test_large_display_is_bounded_but_routes_keep_full_resolution_coordinates():
    from passage.plotting import render_scene

    elevation = np.arange(1024 * 1536).reshape(1024, 1536)
    obstacles = np.zeros_like(elevation, dtype=bool)
    obstacles[0, 0] = True  # must survive conservative occupancy display bins
    path = [[0, 0], [1023, 1535]]
    fig = render_scene(elevation, obstacles, path[0], path[-1], {"route": path})
    ax = fig.axes[0]
    assert max(ax.images[0].get_array().shape) <= 512
    assert ax.images[0].get_extent() == [-0.5, 1535.5, 1023.5, -0.5]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [0, 1535])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [0, 1023])
    assert ax.images[1].get_array()[0, 0, 3] > 0
    assert any("display" in text.get_text().lower() for text in ax.texts)
    plt.close(fig)


def test_corner_markers_and_routes_use_original_cell_centres():
    from passage.plotting import render_scene

    fig = render_scene(np.zeros((3, 5)), None, (0, 0), (2, 4), {"route": [[0, 0], [2, 4]]})
    ax = fig.axes[0]
    assert ax.images[0].get_extent() == [-0.5, 4.5, 2.5, -0.5]
    np.testing.assert_array_equal(ax.collections[0].get_offsets(), [[0, 0]])
    np.testing.assert_array_equal(ax.collections[1].get_offsets(), [[4, 2]])
    assert ax.get_xlim()[0] < 0 and ax.get_xlim()[1] > 4
    assert ax.get_ylim()[0] > 2 and ax.get_ylim()[1] < 0
    plt.close(fig)


def test_noninteger_display_ratio_samples_nearest_centres_and_covers_partial_obstacle_cells():
    from passage.plotting import render_scene

    elevation = np.repeat(np.arange(768)[:, None], 2, axis=1)
    obstacles = np.zeros_like(elevation, dtype=bool)
    obstacles[1, 0] = True  # footprint straddles the first two display bins
    fig = render_scene(elevation, obstacles, (0, 0), (767, 1), {})
    ax = fig.axes[0]
    assert ax.images[0].get_array()[1, 0] == 2  # bin centre y=1.75
    assert ax.images[1].get_array()[0, 0, 3] > 0
    assert ax.images[1].get_array()[1, 0, 3] > 0
    plt.close(fig)
