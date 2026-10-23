"""Analytic and independent checks for the canonical directed grid reference."""

import math

import numpy as np
import pytest

from passage_eval.metrics.cost import path_cost
from passage_eval.protocol.contracts import GraphSpec


def test_float32_differences_are_promoted_before_energy_arithmetic():
    # Subtraction in float32 rounds 16,777,217 down to 16,777,216.
    terrain = np.array([[-1, 16_777_216]], dtype=np.float32)
    assert path_cost(terrain, [(0, 0), (0, 1)], weight=1, cost_model="energy") == 16_777_247.0


def test_engine_module_available():
    import importlib.util

    assert importlib.util.find_spec("passage.reference") is not None


@pytest.mark.parametrize("model,want", [("elevation", 150.0), ("slope", 120.0), ("energy", 80.0)])
def test_directed_reverse_field_uses_predecessor_to_current_cost(model, want):
    from passage.reference import reverse_goal_field, solve_reference

    terrain = np.array([[-5, 0, 5]], dtype=np.float32)
    mask = np.zeros_like(terrain, dtype=np.uint8)
    graph = GraphSpec(cost_model=model, cost_weight=2)
    forward = solve_reference(terrain, mask, (0, 0), (0, 2), graph=graph)
    reverse = solve_reference(terrain, mask, (0, 2), (0, 0), graph=graph)
    field = reverse_goal_field(terrain, mask, (0, 2), graph=graph)
    assert forward.status == field.status == "exact"
    assert forward.cost == want
    assert field.distances[0, 0] == want
    assert field.route((0, 0)).path == forward.path
    assert reverse.cost != want


def test_destination_only_diagonal_and_illegal_obstacle_recost():
    from passage.reference import recost_legal_path, solve_reference

    terrain = np.zeros((2, 2), dtype=np.float32)
    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    result = solve_reference(terrain, mask, (0, 0), (1, 1))
    assert result.status == "exact"
    assert result.cost == 30 * math.sqrt(2)
    assert recost_legal_path(terrain, result.path, obstacles=mask, start=(0, 0), goal=(1, 1)) == result.cost
    assert math.isnan(recost_legal_path(terrain, [(0, 0), (0, 1), (1, 1)], obstacles=mask))


@pytest.mark.parametrize("height", [0, -15, 20])
def test_plateaus_and_ties_are_deterministic_with_exact_distance(height):
    from passage.reference import solve_reference

    terrain = np.full((4, 4), height, dtype=np.float32)
    mask = np.zeros_like(terrain, dtype=np.uint8)
    results = [solve_reference(terrain, mask, (0, 0), (3, 2)) for _ in range(3)]
    assert all(item.status == "exact" and item.path == results[0].path for item in results)
    assert results[0].cost == pytest.approx(30 + 60 * math.sqrt(2), rel=1e-14)
    assert solve_reference(terrain, mask, (0, 0), (3, 2), algorithm="dijkstra").cost == results[0].cost


def test_censoring_never_becomes_no_path_or_a_partial_optimum():
    from passage.reference import reverse_goal_field, solve_reference

    terrain = np.zeros((3, 4), dtype=np.float32)
    mask = np.zeros_like(terrain, dtype=np.uint8)
    result = solve_reference(terrain, mask, (0, 0), (2, 3), max_expansions=0)
    assert result.status == "censored" and result.cost is None and not result.path
    assert result.expansions == 0
    field = reverse_goal_field(terrain, mask, (2, 3), max_expansions=1)
    assert field.status == "censored"
    assert np.isfinite(field.distances).sum() == field.settled.sum() == 1
    assert field.route((0, 0)).status == "censored"
    mask[:, 2] = 1
    result = solve_reference(terrain, mask, (0, 0), (0, 3))
    assert result.status == "no_path" and result.cost is None and not result.path
    assert solve_reference(terrain, mask, (0, 0), (0, 2)).status == "no_path"


def test_roundtrip_decoded_float32_metres_and_independent_graph_optima(tmp_path):
    import networkx as nx
    from passage.reference import recost_legal_path, reverse_goal_field, solve_reference

    terrain = np.random.default_rng(28).uniform(-80, 200, (5, 7)).astype(np.float32)
    np.save(tmp_path / "terrain.npy", terrain)
    decoded = np.load(tmp_path / "terrain.npy")
    obstacles = np.zeros(terrain.shape, dtype=np.uint8)
    obstacles[1:4, 3] = 1
    for model in ("elevation", "slope_uphill", "energy", "absolute_elevation", "distance"):
        # Independent small explicit graph, with Python double scalar arithmetic.
        graph = nx.DiGraph()
        lo, hi = float(decoded.min()), float(decoded.max())
        for r, c in np.ndindex(terrain.shape):
            if obstacles[r, c]:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if not (dr or dc) or not (0 <= rr < 5 and 0 <= cc < 7) or obstacles[rr, cc]:
                        continue
                    d = 30 * math.hypot(dr, dc)
                    dst = float(decoded[rr, cc])
                    gain = max(0, dst - float(decoded[r, c]))
                    if model == "elevation":
                        cost = d * (1 + 2 * ((dst - lo) / (hi - lo)))
                    elif model == "slope_uphill":
                        cost = d * (1 + 2 * (gain / (hi - lo)))
                    elif model == "energy":
                        cost = d + 2 * gain
                    elif model == "absolute_elevation":
                        cost = d * (1 + 2 * abs(dst) / max(abs(lo), abs(hi)))
                    else:
                        cost = d
                    graph.add_edge((r, c), (rr, cc), weight=cost)
        spec = GraphSpec(cost_model=model, cost_weight=2)
        field = reverse_goal_field(decoded, obstacles, (4, 6), graph=spec)
        for start in ((0, 0), (0, 6), (4, 0)):
            expected = nx.shortest_path_length(graph, start, (4, 6), weight="weight")
            result = solve_reference(decoded, obstacles, start, (4, 6), graph=spec)
            assert result.cost == pytest.approx(expected, rel=1e-13)
            assert field.distances[start] == pytest.approx(expected, rel=1e-13)
            assert recost_legal_path(decoded, result.path, obstacles=obstacles, graph=spec) == result.cost


def test_invalid_inputs_fail_before_search():
    from passage.reference import solve_reference

    terrain = np.zeros((2, 2))
    for kwargs in ({"max_expansions": -1}, {"max_expansions": True}, {"algorithm": "weighted"}):
        with pytest.raises(ValueError):
            solve_reference(terrain, np.zeros_like(terrain), (0, 0), (1, 1), **kwargs)
    with pytest.raises(ValueError):
        solve_reference(terrain, np.zeros_like(terrain), (0.2, 0), (1, 1))


@pytest.mark.parametrize(
    "model,values",
    [
        ("elevation", [0, 1]),
        ("slope_uphill", [0, 1]),
        ("absolute_elevation", [0, 1]),
        # Both energy edges are finite (1e308), but their path total is not.
        ("energy", [0, 1, 2]),
    ],
)
def test_float64_overflow_is_an_arithmetic_error_not_proved_unreachable(model, values):
    from passage.reference import recost_legal_path, reverse_goal_field, solve_reference

    terrain = np.array([values], dtype=np.float32)
    obstacles = np.zeros_like(terrain, dtype=np.uint8)
    start, goal = (0, 0), (0, len(values) - 1)
    spec = GraphSpec(cost_model=model, cost_weight=1e308)
    for algorithm in ("astar", "dijkstra"):
        with pytest.raises(ValueError, match="float64.*(cost|arithmetic)|arithmetic.*float64"):
            solve_reference(terrain, obstacles, start, goal, graph=spec, algorithm=algorithm)
    with pytest.raises(ValueError, match="float64.*(cost|arithmetic)|arithmetic.*float64"):
        reverse_goal_field(terrain, obstacles, goal, graph=spec)
    with pytest.raises(ValueError, match="float64.*(cost|arithmetic)|arithmetic.*float64"):
        recost_legal_path(terrain, [(0, col) for col in range(len(values))], graph=spec)


def test_endpoints_connected_matches_solver_proofs_on_random_masks():
    from passage.reference import endpoints_connected, solve_reference

    rng = np.random.default_rng(7)
    seen = set()
    for _ in range(300):
        shape = tuple(rng.integers(2, 12, size=2))
        mask = (rng.random(shape) < rng.uniform(0.2, 0.7)).astype(np.uint8)
        terrain = rng.normal(size=shape).astype(np.float32)
        start, goal = (tuple(int(rng.integers(0, n)) for n in shape) for _ in range(2))
        connected = endpoints_connected(mask, start, goal)
        seen.add(connected)
        assert connected == (solve_reference(terrain, mask, start, goal).status == "exact")
    assert seen == {True, False}


def test_endpoints_connected_uses_destination_only_diagonals():
    from passage.reference import endpoints_connected

    mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert endpoints_connected(mask, (0, 0), (1, 1))
    assert not endpoints_connected(mask, (0, 0), (0, 1))
    with pytest.raises(ValueError, match="binary"):
        endpoints_connected(np.full((2, 2), 2), (0, 0), (1, 1))
