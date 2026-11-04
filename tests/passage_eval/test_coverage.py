"""Coverage is joint descriptive occupancy, without independence or causal claims."""

from passage_eval.diagnostics.coverage import joint_odd_coverage


def test_joint_cells_expose_missing_common_support_that_marginals_hide():
    a = [{"resolution": 64, "cost_model": "distance"}, {"resolution": 128, "cost_model": "elevation"}]
    b = [{"resolution": 64, "cost_model": "elevation"}, {"resolution": 128, "cost_model": "distance"}]
    result = joint_odd_coverage({"a": a, "b": b}, axes=["resolution", "cost_model"])
    assert result["n_common_cells"] == 0
    assert result["common_support_status"] == "absent"
    assert len(result["cells"]) == 4
    assert result["interpretation"] == "descriptive_only_no_independence_or_causal_claim"


def test_coverage_retains_unknown_axis_values_and_declares_incomplete_metadata():
    result = joint_odd_coverage({"a": [{"resolution": 64}, {}]}, axes=["resolution", "task_variant"])
    assert result["status"] == "incomplete_metadata"
    assert result["missing_axis_counts"]["a"] == {"resolution": 1, "task_variant": 2}
    assert sum(cell["populations"]["a"]["count"] for cell in result["cells"]) == 2


def test_descriptive_cells_include_legal_denominator_and_failure_regret():
    rows = [
        {"resolution": 64, "legal": True, "cost_regret": 0.2, "latency_ms": 4},
        {"resolution": 64, "legal": False, "cost_regret": 999999, "latency_ms": 8},
    ]
    result = joint_odd_coverage({"a": rows}, axes=["resolution"])
    population = result["cells"][0]["populations"]["a"]
    assert population["count"] == 2
    assert population["legal_count"] == 1
    assert population["mean_cost_regret"] == (999999 + 0.2) / 2
    assert population["median_latency_ms"] == 6
    assert population["latency_censored_count"] is None  # the rows do not say which latencies are censored
    rows[1]["latency_censored"] = True
    rows[0]["latency_censored"] = False
    flagged = joint_odd_coverage({"a": rows}, axes=["resolution"])["cells"][0]["populations"]["a"]
    assert flagged["latency_censored_count"] == 1 and flagged["median_latency_ms"] == 6


def test_numeric_strata_use_predeclared_edges_and_keep_unknown_outside_values():
    from passage_eval.diagnostics.coverage import stratify_numeric

    assert stratify_numeric([0, 0.1, 0.5, 1, -1, 2, None, float("nan")], [0, 0.1, 0.5, 1]) == [
        0,
        1,
        2,
        2,
        None,
        None,
        None,
        None,
    ]


def test_invalid_or_outcome_derived_bin_boundaries_are_not_accepted():
    import pytest
    from passage_eval.diagnostics.coverage import stratify_numeric

    for edges in ([0], [0, 0, 1], [1, 0], [0, float("inf")]):
        with pytest.raises(ValueError, match="edges"):
            stratify_numeric([0.5], edges)
