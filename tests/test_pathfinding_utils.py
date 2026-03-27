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

import math

import networkx as nx
import numpy as np
import pytest

import passage.pathfinding_utils as pf


def _py_func(func):
    return getattr(func, "py_func", func)


@pytest.mark.unit
def test_graph_template_cache_and_single_cell_graph():
    template = pf._get_graph_template(1, 1, "always")
    cached = pf._get_graph_template(1, 1, "always")

    assert cached is template
    assert template.distance_m.size == 0

    weights = pf._compute_edge_weights(template, np.array([[42.0]], dtype=np.float64), weight=3.0, cost_model="bogus")
    assert weights.size == 0

    graph = pf.build_graph_from_elevation(np.array([[42.0]], dtype=np.float64), weight=1.0, cost_model="bogus")
    assert graph.number_of_nodes() == 1
    assert graph.number_of_edges() == 0


@pytest.mark.unit
def test_normalization_and_cost_helpers_cover_edge_cases():
    flat = np.full((2, 2), -5.0, dtype=np.float64)
    normalized, abs_norm, elev_range = pf._normalize_elevation_arrays(flat)
    assert elev_range == 0.0
    assert np.allclose(normalized, 0.0)
    assert np.allclose(abs_norm, 1.0)

    invalid = pf._build_pathfinding_cost_matrix(flat, 2.0, "not-a-model")
    assert invalid.shape == flat.shape
    assert np.all(invalid >= 1e-6)
    assert pf._build_pathfinding_cost_matrix(flat, 2.0, "energy") is None

    assert pf._pathfinding_path_to_coords([object()]) == []


@pytest.mark.unit
def test_find_path_pathfinding_variants_and_unsupported_cost_model():
    crop = np.arange(9, dtype=np.float64).reshape(3, 3)

    for solver_name in ["astar", "dijkstra", "best_first"]:
        path, prep_time, solve_time = pf._find_path_pathfinding(
            crop=crop,
            start=(0, 0),
            end=(2, 2),
            solver_name=solver_name,
            diagonal_movement="always",
            weight=1.0,
            heuristic_name="manhattan",
            cost_model="elevation",
        )
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)
        assert prep_time >= 0.0
        assert solve_time >= 0.0

    path, prep_time, solve_time = pf._find_path_pathfinding(
        crop=crop,
        start=(0, 0),
        end=(2, 2),
        solver_name="astar",
        diagonal_movement="never",
        weight=1.0,
        heuristic_name="euclidean",
        cost_model="slope_uphill",
    )
    assert path == []
    assert prep_time == 0.0
    assert solve_time == 0.0


@pytest.mark.unit
def test_path_cost_and_collision_fallbacks():
    graph = nx.DiGraph()
    graph.add_edge((0, 0), (0, 1), weight=12.5)
    graph.add_edge((0, 1), (1, 1))

    cost = pf._path_cost_from_graph(graph, [(0, 0), (0, 1), (1, 1)])
    assert cost > 12.5

    class BrokenGraph:
        def get_edge_data(self, *_args, **_kwargs):
            raise RuntimeError("boom")

    broken_cost = pf._path_cost_from_graph(BrokenGraph(), [(0, 0), (1, 1)])
    assert broken_cost == pytest.approx(math.hypot(1, 1) * pf.PIXEL_SIZE_M)

    obstacles = np.zeros((3, 3), dtype=np.uint8)
    obstacles[1, 1] = 1

    assert pf.check_path_collision([(0, 0), (1, 1)], obstacles) is True

    class Node:
        def __init__(self, x, y):
            self.x = x
            self.y = y

    assert pf.check_path_collision([Node(1, 1)], obstacles) is True
    assert pf.check_path_collision([("bad", "value")], obstacles) is False
    assert pf.check_path_collision([], obstacles) is False


@pytest.mark.unit
def test_numba_helper_pyfuncs_cover_heuristics_and_heap_operations():
    heuristic_euclidean = _py_func(pf._numba_heuristic_euclidean)
    heuristic_manhattan = _py_func(pf._numba_heuristic_manhattan)
    heuristic_chebyshev = _py_func(pf._numba_heuristic_chebyshev)
    compute_step_cost = _py_func(pf._numba_compute_step_cost)
    heap_push = _py_func(pf._numba_heap_push)
    heap_pop = _py_func(pf._numba_heap_pop)

    assert heuristic_euclidean(0, 0, 3, 4, 2.0) == pytest.approx(10.0)
    assert heuristic_manhattan(0, 0, 3, 4, 2.0) == pytest.approx(14.0)
    assert heuristic_chebyshev(0, 0, 3, 4, 2.0) == pytest.approx(8.0)

    assert compute_step_cost(pf._COST_DISTANCE, 3.0, 2.0, 5.0, 1.2, 1.0, 4.0) == pytest.approx(3.0)
    assert compute_step_cost(pf._COST_ELEVATION, 3.0, 2.0, 5.0, 1.2, 1.0, 4.0) == pytest.approx(3.6)
    assert compute_step_cost(pf._COST_ABSOLUTE_ELEVATION, 3.0, 2.0, 5.0, 1.2, 1.0, 4.0) == pytest.approx(3.6)
    assert compute_step_cost(pf._COST_SLOPE_UPHILL, 3.0, 2.0, 5.0, 1.0, 1.0, 4.0) == pytest.approx(6.6)
    assert compute_step_cost(pf._COST_ENERGY, 3.0, 2.0, 5.0, 1.0, 1.0, 4.0) == pytest.approx(9.0)
    assert compute_step_cost(999, 3.0, 2.0, 5.0, 1.2, 1.0, 4.0) == pytest.approx(3.6)

    heap = np.zeros((8, 4), dtype=np.float64)
    heap_size = heap_push(heap, 0, 5.0, 5.0, 1, 1)
    heap_size = heap_push(heap, heap_size, 2.0, 2.0, 0, 0)
    heap_size = heap_push(heap, heap_size, 4.0, 4.0, 2, 2)

    f, g, i, j, heap_size = heap_pop(heap, heap_size)
    assert (f, g, i, j) == pytest.approx((2.0, 2.0, 0, 0))
    assert heap_size == 2


@pytest.mark.unit
def test_numba_core_pyfuncs_and_reconstruct_path_cover_solver_internals():
    astar_core = _py_func(pf._numba_astar_core)
    dijkstra_core = _py_func(pf._numba_dijkstra_core)
    reconstruct_numba = _py_func(pf._numba_reconstruct_path)

    crop = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)
    dest_factor = np.ones_like(crop, dtype=np.float64)
    neighbors = pf._get_neighbors_array("always")

    parents_euclidean = astar_core(
        crop, dest_factor, 0, 0, 1, 1, 1.0, 2.0, pf._COST_DISTANCE, neighbors, 0, pf.PIXEL_SIZE_M
    )
    parents_manhattan = astar_core(
        crop, dest_factor, 0, 0, 1, 1, 1.0, 2.0, pf._COST_DISTANCE, neighbors, 1, pf.PIXEL_SIZE_M
    )
    parents_chebyshev = astar_core(
        crop, dest_factor, 0, 0, 1, 1, 1.0, 2.0, pf._COST_DISTANCE, neighbors, 2, pf.PIXEL_SIZE_M
    )
    parents_dijkstra = dijkstra_core(crop, dest_factor, 0, 0, 1, 1, 1.0, 2.0, pf._COST_DISTANCE, neighbors)

    for parents in [parents_euclidean, parents_manhattan, parents_chebyshev, parents_dijkstra]:
        assert tuple(parents[1, 1]) != (-1, -1)
        path = reconstruct_numba(parents, 0, 0, 1, 1)
        assert path.shape[1] == 2
        assert tuple(path[0]) == (0, 0)
        assert tuple(path[-1]) == (1, 1)

    invalid_parents = np.full((2, 2, 2), -1, dtype=np.int32)
    assert reconstruct_numba(invalid_parents, 0, 0, 1, 1).size == 0


@pytest.mark.unit
def test_neighbor_helpers_and_numba_wrappers(monkeypatch):
    always = pf._get_neighbors_array("always")
    never = pf._get_neighbors_array("never")
    assert always.shape == (8, 3)
    assert never.shape == (4, 3)
    assert len(pf._grid_solvers_neighbors("always")) == 8
    assert len(pf._grid_solvers_neighbors("never")) == 4

    crop = np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float64)

    assert pf._grid_astar_numba(crop, (0, 0), (0, 0), 1.0, "always", "euclidean", "distance") == [(0, 0)]
    assert pf._grid_dijkstra_numba(crop, (0, 0), (0, 0), 1.0, "always", "distance") == [(0, 0)]

    def no_path_core(*_args, **_kwargs):
        return np.full((2, 2, 2), -1, dtype=np.int32)

    monkeypatch.setattr(pf, "_numba_astar_core", no_path_core)
    assert pf._grid_astar_numba(crop, (0, 0), (1, 1), 1.0, "always", "euclidean", "distance") == []

    def astar_core_with_parent(*_args, **_kwargs):
        parents = np.full((2, 2, 2), -1, dtype=np.int32)
        parents[1, 1] = (0, 0)
        return parents

    monkeypatch.setattr(pf, "_numba_astar_core", astar_core_with_parent)
    monkeypatch.setattr(pf, "_numba_reconstruct_path", lambda *_args, **_kwargs: np.empty((0, 2), dtype=np.int32))
    assert pf._grid_astar_numba(crop, (0, 0), (1, 1), 1.0, "always", "euclidean", "distance") == []

    monkeypatch.setattr(pf, "_numba_dijkstra_core", no_path_core)
    assert pf._grid_dijkstra_numba(crop, (0, 0), (1, 1), 1.0, "always", "distance") == []

    def dijkstra_core_with_parent(*_args, **_kwargs):
        parents = np.full((2, 2, 2), -1, dtype=np.int32)
        parents[1, 1] = (0, 0)
        return parents

    monkeypatch.setattr(pf, "_numba_dijkstra_core", dijkstra_core_with_parent)
    assert pf._grid_dijkstra_numba(crop, (0, 0), (1, 1), 1.0, "always", "distance") == []


@pytest.mark.unit
def test_python_grid_solver_fallbacks_cover_multiple_cost_models(monkeypatch):
    monkeypatch.setattr(pf, "NUMBA_AVAILABLE", False)

    crop = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float64,
    )

    assert pf._grid_astar(crop, (1, 1), (1, 1), 1.0, "always", "euclidean", "distance") == [(1, 1)]

    for cost_model in ["bogus", "distance", "absolute_elevation", "slope_uphill", "energy", "elevation"]:
        path = pf._grid_astar(crop, (0, 0), (2, 2), 1.0, "always", "manhattan", cost_model)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    monkeypatch.setattr(pf, "_grid_solvers_neighbors", lambda *_args, **_kwargs: [])
    assert pf._grid_astar(crop, (0, 0), (2, 2), 1.0, "always", "euclidean", "distance") == []


@pytest.mark.unit
def test_python_grid_dijkstra_fallbacks_cover_multiple_cost_models(monkeypatch):
    monkeypatch.setattr(pf, "NUMBA_AVAILABLE", False)

    crop = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=np.float64,
    )

    assert pf._grid_dijkstra(crop, (1, 1), (1, 1), 1.0, "always", "distance") == [(1, 1)]

    for cost_model in ["bogus", "distance", "absolute_elevation", "slope_uphill", "energy", "elevation"]:
        path = pf._grid_dijkstra(crop, (0, 0), (2, 2), 1.0, "always", cost_model)
        assert path[0] == (0, 0)
        assert path[-1] == (2, 2)

    monkeypatch.setattr(pf, "_grid_solvers_neighbors", lambda *_args, **_kwargs: [])
    assert pf._grid_dijkstra(crop, (0, 0), (2, 2), 1.0, "always", "distance") == []
