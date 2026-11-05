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

"""Presentation changes must preserve every supplied terrain and route value."""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/trajectory_style.py"
spec = importlib.util.spec_from_file_location("trajectory_style_tested", PATH)
style = importlib.util.module_from_spec(spec)
spec.loader.exec_module(style)


def test_reference_and_prediction_match_paper_and_preserve_vertices():
    elevation = np.arange(36, dtype=float).reshape(6, 6)
    original = elevation.copy()
    reference = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    prediction = np.array([[0, 0], [0, 1], [1, 2], [2, 3], [3, 3]])
    fig, ax = plt.subplots()
    try:
        style.draw_comparison(elevation, np.zeros((6, 6), dtype=bool), [0, 0], [3, 3], reference, prediction, ax=ax)
        assert ax.lines[0].get_color() == "#238b45"
        assert ax.lines[1].get_color() == "#1767ac"
        assert ax.lines[0].get_linestyle() == "-"
        assert ax.lines[1].get_linestyle() == "--"
        for line, path in zip(ax.lines, [reference, prediction]):
            np.testing.assert_array_equal(line.get_xdata(), path[:, 1])
            np.testing.assert_array_equal(line.get_ydata(), path[:, 0])
        np.testing.assert_array_equal(elevation, original)
        np.testing.assert_array_equal(ax.images[0].get_array(), elevation)
    finally:
        plt.close(fig)


def test_failed_prediction_remains_empty_instead_of_inventing_a_route():
    fig, ax = plt.subplots()
    try:
        style.draw_comparison(np.zeros((4, 4)), None, [0, 0], [3, 3], [[0, 0], [1, 1], [2, 2], [3, 3]], [], ax=ax)
        assert len(ax.lines) == 2
        assert len(ax.lines[1].get_xdata()) == 0
    finally:
        plt.close(fig)


def test_cost_routes_keep_color_meaning_and_only_obstacles_are_dashed():
    fig, ax = plt.subplots()
    try:
        paths = {"elevation_free": [[0, 0], [1, 1]], "elevation_obstacles": [[0, 0], [1, 0]]}
        for points in paths.values():
            values = np.asarray(points)
            ax.plot(values[:, 1], values[:, 0], color="orange")
        style.style_routes(ax, paths, free_color=style.FREE_TERRAIN_COLOR, obstacle_color=style.OBSTACLE_AWARE_COLOR)
        assert [line.get_linestyle() for line in ax.lines] == ["-", "--"]
        assert [line.get_color() for line in ax.lines] == ["#238b45", "#1767ac"]
    finally:
        plt.close(fig)
