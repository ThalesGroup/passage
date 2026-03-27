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

import passage.generate as gen


def test_compute_edge_weights_and_cost_matrix():
    # Small grid
    rows, cols = 4, 4
    tpl = gen._get_graph_template(rows, cols, "always")

    # Use signed elevations so elevation and absolute_elevation differ.
    elev = np.linspace(-8.0, 7.0, rows * cols, dtype=np.float64).reshape((rows, cols))

    # distance model should return template distances unchanged
    dist = gen._compute_edge_weights(tpl, elev, weight=1.0, cost_model="distance")
    assert np.array_equal(dist, tpl.distance_m)

    elev_w = gen._compute_edge_weights(tpl, elev, weight=2.5, cost_model="elevation")

    # absolute_elevation uses abs_norm of destination
    abs_w = gen._compute_edge_weights(tpl, elev, weight=2.5, cost_model="absolute_elevation")
    normalized, abs_norm, elev_range = gen._normalize_elevation_arrays(elev)
    expected = tpl.distance_m * (1.0 + 2.5 * abs_norm[tpl.dst_i, tpl.dst_j])
    assert np.allclose(abs_w, expected)
    assert not np.allclose(abs_w, elev_w)

    # slope_uphill penalizes uphill (max(0, dst - src)/range)
    slope = gen._compute_edge_weights(tpl, elev, weight=10.0, cost_model="slope_uphill")
    delta = elev[tpl.dst_i, tpl.dst_j] - elev[tpl.src_i, tpl.src_j]
    slope_norm = np.maximum(0.0, delta) / (elev.max() - elev.min())
    expected_slope = tpl.distance_m * (1.0 + 10.0 * slope_norm)
    assert np.allclose(slope, expected_slope)

    # energy: distance + weight * max(0, delta)
    energy = gen._compute_edge_weights(tpl, elev, weight=5.0, cost_model="energy")
    expected_energy = tpl.distance_m + 5.0 * np.maximum(0.0, delta)
    assert np.allclose(energy, expected_energy)


def test_build_pathfinding_cost_matrix_variants():
    elev = np.linspace(-5.0, 5.0, 9).reshape((3, 3)).astype(np.float64)

    # slope_uphill and energy should return None
    assert gen._build_pathfinding_cost_matrix(elev, 1.0, "slope_uphill") is None
    assert gen._build_pathfinding_cost_matrix(elev, 1.0, "energy") is None

    # distance returns ones
    dmat = gen._build_pathfinding_cost_matrix(elev, 1.0, "distance")
    assert np.allclose(dmat, np.ones_like(elev))

    # absolute_elevation and elevation return >0 matrices
    m_abs = gen._build_pathfinding_cost_matrix(elev, 2.0, "absolute_elevation")
    assert m_abs.shape == elev.shape
    assert np.all(m_abs >= 1e-6)

    m_elev = gen._build_pathfinding_cost_matrix(elev, 3.0, "elevation")
    assert m_elev.shape == elev.shape
    assert np.all(m_elev >= 1e-6)
    assert not np.allclose(m_abs, m_elev)


def test_pathfinding_path_to_coords_variants():
    # empty path
    assert gen._pathfinding_path_to_coords([]) == []

    # tuple/list path flips (i,j)
    inp = [(0, 1), (2, 3)]
    out = gen._pathfinding_path_to_coords(inp)
    assert out == [(1, 0), (3, 2)]

    # object with x,y attributes
    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    npath = [Node(5, 6), Node(7, 8)]
    out2 = gen._pathfinding_path_to_coords(npath)
    assert out2 == [(6, 5), (8, 7)]
