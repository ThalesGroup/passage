# MIT License
#
# Copyright (c) 2026 THALES

import numpy as np

from passage_eval import (
    bellman_residual_reachable,
    boundary_f_score,
    descent_direction_agreement,
    field_ssi_mae,
    local_ordinal_concordance,
    mask_connected_components_total,
    mask_endpoints_in_same_component,
)


def test_mask_connected_components_total():
    m = np.zeros((10, 10), dtype=bool)
    m[1:4, 1:4] = True
    m[6:9, 6:9] = True
    assert mask_connected_components_total(m) == 2


def test_mask_endpoints_in_same_component_true_for_bridged_endpoints():
    m = np.zeros((10, 10), dtype=bool)
    m[0:5, 5] = True
    m[4:10, 5] = True
    assert mask_endpoints_in_same_component(m, (0, 5), (9, 5)) is True


def test_mask_endpoints_in_same_component_false_when_cut():
    m = np.zeros((10, 10), dtype=bool)
    m[0:3, 5] = True
    m[6:10, 5] = True
    assert mask_endpoints_in_same_component(m, (0, 5), (9, 5)) is False


def test_boundary_f_score_perfect_for_identical_masks():
    m = np.zeros((10, 10), dtype=bool)
    m[3:7, 3:7] = True
    f = boundary_f_score(m, m)
    assert abs(f - 1.0) < 1e-9


def test_descent_direction_agreement_one_for_identical_fields():
    rng = np.random.default_rng(0)
    field = rng.standard_normal((8, 8))
    out = descent_direction_agreement(field, field)
    assert out == 1.0


def test_descent_direction_agreement_invariant_to_affine_shift():
    rng = np.random.default_rng(1)
    field = rng.standard_normal((8, 8))
    out = descent_direction_agreement(2.0 * field + 100.0, field)
    assert out == 1.0


def test_local_ordinal_concordance_one_under_monotone_transform():
    rng = np.random.default_rng(2)
    field = rng.standard_normal((8, 8))
    out = local_ordinal_concordance(np.exp(field), field)
    assert out == 1.0


def test_field_ssi_mae_zero_under_affine_transform():
    rng = np.random.default_rng(3)
    field = rng.standard_normal((8, 8))
    out = field_ssi_mae(2.0 * field + 100.0, field)
    assert out < 1e-9


def test_bellman_residual_reachable_finite_on_smooth_field():
    h = w = 8
    field = np.fromfunction(lambda i, j: (h - i) + (w - j), (h, w), dtype=np.float64)
    cost = np.ones((h, w), dtype=np.float64)
    out = bellman_residual_reachable(field, cost, goal=(h - 1, w - 1))
    assert np.isfinite(out)
