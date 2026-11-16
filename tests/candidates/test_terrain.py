import numpy as np


def test_reverse_targets_preserve_directed_destination_cost():
    from passage_eval.candidates.terrain import reverse_targets

    terrain = np.array([[0.0, 1.0]])
    obstacles = np.zeros((1, 2), bool)
    values, path = reverse_targets(terrain, obstacles, (0, 0), (0, 1))
    assert values[0, 0] == 1902.0
    assert path.tolist() == [[0, 0], [0, 1]]
    reversed_values, _ = reverse_targets(terrain, obstacles, (0, 1), (0, 0))
    assert reversed_values[0, 1] == 30.0


def test_diagonal_can_cross_blocked_flanks_and_extrema_include_blocked():
    from passage_eval.candidates.terrain import reverse_targets

    terrain = np.array([[0.0, 100.0], [100.0, 1.0]])
    obstacles = np.array([[False, True], [True, False]])
    values, path = reverse_targets(terrain, obstacles, (0, 0), (1, 1))
    assert np.isclose(values[0, 0], 30 * np.sqrt(2) * (1 + 62.4 / 100))
    assert path.tolist() == [[0, 0], [1, 1]]


def test_unreachable_target_is_not_replaced_by_fake_route():
    from passage_eval.candidates.terrain import reverse_targets

    terrain = np.zeros((3, 3))
    obstacles = np.zeros((3, 3), bool)
    obstacles[1, :] = True
    values, path = reverse_targets(terrain, obstacles, (0, 0), (2, 2))
    assert np.isinf(values[0, 0])
    assert path.shape == (0, 2)


def test_every_cost_to_goal_matches_forward_heap_reference_and_canonical_recost():
    from passage_eval.candidates.terrain import reverse_targets
    from passage_eval.metrics.cost import path_cost
    from passage_eval.protocol.second_reference import dijkstra8_cost

    terrain = np.array([[3.0, 8.0, 1.0, 5.0], [2.0, 100.0, 4.0, 7.0], [9.0, 6.0, 0.0, 11.0], [4.0, 2.0, 3.0, 8.0]])
    obstacles = np.zeros_like(terrain, dtype=bool)
    obstacles[1, 1] = True
    goal = (3, 3)
    for point in zip(*np.where(~obstacles)):
        values, path = reverse_targets(terrain, obstacles, point, goal)
        expected = dijkstra8_cost(terrain, obstacles, point, goal, cost_model="elevation", weight=62.4)
        assert np.isclose(values[point], expected, rtol=1e-12)
        if point != goal:
            assert np.isclose(path_cost(terrain, path, cost_model="elevation", weight=62.4), expected, rtol=1e-12)
        assert np.isinf(values[1, 1])
