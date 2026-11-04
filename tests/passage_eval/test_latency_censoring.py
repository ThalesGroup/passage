"""The proposed (not sealed) censoring-aware latency rule, checked against hand-computed order statistics."""

import math

import numpy as np
import pytest

from passage_eval.metrics.latency import (
    MIN_WARM_REQUESTS,
    RULE_STATUS,
    censored_latency_quantiles,
    latency_censoring,
)


def _hand_rank(n, q):
    """1-based rank of the inverted-CDF order statistic: the smallest k with k / n >= q."""
    return max(1, math.ceil(n * q - 1e-12))


def test_rule_is_labelled_proposed_and_uses_order_statistics_without_interpolation():
    block = censored_latency_quantiles([10.0, 20.0], status=["ok", "ok"], sample_id=["a", "b"])
    assert block["status"] == RULE_STATUS == "proposed_not_sealed"
    assert block["method"] == "inverted_cdf" and block["interpolation"] is False
    assert block["quantiles"]["p50"]["value_ms"] == 10.0  # np.quantile(..., 0.5) would interpolate to 15.0
    assert block["quantiles"]["p95"]["value_ms"] == 20.0


@pytest.mark.parametrize("n", [1, 2, 3, 7, 20, 99, 100, 101, 250])
def test_ranks_match_the_hand_definition(n):
    values = np.arange(1, n + 1, dtype=float) * 3.0
    rng = np.random.default_rng(n)
    order = rng.permutation(n)
    block = censored_latency_quantiles(values[order], status=["ok"] * n, sample_id=[f"s{i}" for i in range(n)])
    for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
        rank = _hand_rank(n, q)
        assert block["quantiles"][name]["rank"] == rank
        assert block["quantiles"][name]["value_ms"] == 3.0 * rank
        assert block["quantiles"][name]["censored"] is False


def test_timeouts_are_right_censored_and_report_a_lower_bound():
    # 10 requests: 7 fast successes, one fast failure, two timeouts recorded at the budget plus grace.
    latency = [5, 6, 7, 8, 9, 10, 11, 2, 1010, 1012]
    status = ["ok"] * 7 + ["runtime_error", "timeout", "timeout"]
    block = censored_latency_quantiles(latency, status=status, sample_id=[f"s{i}" for i in range(10)])
    assert (block["n_requests"], block["n_ok"], block["n_censored"], block["n_timeout"]) == (10, 7, 2, 2)
    assert block["n_failed_non_timeout"] == 1 and block["n_charged_not_measured"] == 0
    assert block["n_failed_non_timeout_measured"] == 1
    assert block["n_ok"] + block["n_timeout"] + block["n_failed_non_timeout"] == block["n_requests"]
    # order with censored at +inf: 2, 5, 6, 7, 8, 9, 10, 11, +inf, +inf
    assert block["quantiles"]["p50"] == {
        "q": 0.5,
        "rank": 5,
        "value_ms": 8.0,
        "censored": False,
        "lower_bound_ms": 8.0,
    }
    p95 = block["quantiles"]["p95"]
    assert p95["rank"] == 10 and p95["censored"] is True and p95["value_ms"] is None
    assert p95["lower_bound_ms"] == 1012.0  # 10th smallest recorded lower bound
    assert block["quantiles"]["p99"]["censored"] is True
    # the legacy interpolated value would have been finite and below the budget-plus-grace values
    assert np.quantile(latency, 0.95) < 1012.0
    secondary = block["success_only"]
    assert secondary["secondary"] is True and secondary["n"] == 7
    assert secondary["quantiles"]["p95"]["value_ms"] == 11.0


def test_charged_latencies_of_unexecuted_tasks_are_censored_too():
    status = ["ok", "runtime_error", "runtime_error"]
    charged = [False, True, False]
    flags = latency_censoring(status, charged)
    assert flags.tolist() == [False, True, False]
    block = censored_latency_quantiles([4.0, 120000.0, 3.0], status=status, sample_id=["a", "b", "c"], censored=flags)
    assert block["n_censored"] == 1 and block["n_charged_not_measured"] == 1
    # Both runtime errors are non-timeout failures: one charged (censored), one fast (measured).
    assert block["n_failed_non_timeout"] == 2 and block["n_failed_non_timeout_measured"] == 1
    assert block["n_ok"] + block["n_timeout"] + block["n_failed_non_timeout"] == block["n_requests"] == 3
    assert block["quantiles"]["p50"]["value_ms"] == 4.0
    assert block["quantiles"]["p99"]["censored"] and block["quantiles"]["p99"]["lower_bound_ms"] == 120000.0


WARMED = {"declared": True, "performed": True}


def test_scene_and_repeat_counts_and_the_warm_request_flag():
    ids = [f"s{i % 25}" for i in range(100)]  # 25 scenes x 4 repeats
    block = censored_latency_quantiles(np.linspace(1, 2, 100), status=["ok"] * 100, sample_id=ids, warmup=WARMED)
    assert (block["n_distinct_scenes"], block["n_repeats"]) == (25, 4)
    assert block["repeats_per_scene"] == {"min": 4, "max": 4}
    assert block["min_warm_requests"] == MIN_WARM_REQUESTS == 100 and block["meets_min_warm_requests"] is True
    assert block["n_warm_requests"] == 100 and block["meets_min_warm_requests_reason"] is None
    short = censored_latency_quantiles(
        [1.0] * 99, status=["ok"] * 99, sample_id=[str(i) for i in range(99)], warmup=WARMED
    )
    assert short["meets_min_warm_requests"] is False  # a flag, not an exception
    assert short["meets_min_warm_requests_reason"] == "fewer_warm_requests_than_the_minimum"
    cold = censored_latency_quantiles(
        np.ones(100), status=["ok"] * 100, sample_id=[str(i) for i in range(100)], series="cold", warmup=WARMED
    )
    assert cold["meets_min_warm_requests"] is False and cold["meets_min_warm_requests_reason"] == "not_a_warm_series"
    uneven = censored_latency_quantiles([1.0, 1.0, 1.0], status=["ok"] * 3, sample_id=["a", "a", "b"])
    assert uneven["n_repeats"] is None and uneven["repeats_per_scene"] == {"min": 1, "max": 2}


def test_a_series_named_warm_is_not_warm_without_a_declared_and_performed_warmup():
    """The series name alone never qualifies 100 cold-start requests as a warm latency row."""
    kwargs = dict(status=["ok"] * 100, sample_id=[str(i) for i in range(100)])
    unknown = censored_latency_quantiles(np.ones(100), **kwargs)
    assert unknown["meets_min_warm_requests"] is None and unknown["n_warm_requests"] is None
    assert unknown["meets_min_warm_requests_reason"] == "warmup_evidence_not_given"
    undeclared = censored_latency_quantiles(np.ones(100), warmup={"declared": False, "performed": False}, **kwargs)
    assert undeclared["meets_min_warm_requests"] is False and undeclared["n_warm_requests"] == 0
    assert undeclared["meets_min_warm_requests_reason"] == "no_declared_warmup"
    skipped = censored_latency_quantiles(np.ones(100), warmup={"declared": True, "performed": False}, **kwargs)
    assert skipped["meets_min_warm_requests"] is False
    assert skipped["meets_min_warm_requests_reason"] == "declared_warmup_not_performed_in_every_pass"
    # per-request states: a request without a warm-up (None) or with an incomplete one (False) is not warm
    states = [True] * 98 + [None, False]
    partial = censored_latency_quantiles(np.ones(100), warmup=WARMED, warmup_complete=states, **kwargs)
    assert partial["n_requests_without_complete_warmup"] == 2 and partial["n_warm_requests"] == 98
    assert partial["meets_min_warm_requests"] is False
    assert partial["meets_min_warm_requests_reason"] == "fewer_warm_requests_than_the_minimum"
    unplanned = censored_latency_quantiles(
        np.ones(100), warmup={"declared": False, "performed": False}, warmup_complete=[None] * 100, **kwargs
    )
    assert unplanned["n_requests_without_complete_warmup"] == 100 and unplanned["meets_min_warm_requests"] is False


def test_rewarm_incomplete_requests_remain_counted_but_are_excluded_from_hot_latency_quantiles():
    states = [True] * 99 + [False]
    block = censored_latency_quantiles(
        list(range(1, 101)),
        status=["ok"] * 100,
        sample_id=[str(i) for i in range(100)],
        warmup=WARMED,
        warmup_complete=states,
    )
    assert block["n_requests"] == 100
    assert block["n_rewarm_incomplete_requests"] == 1
    assert block["n_hot_latency_requests"] == 99
    assert block["quantiles"]["p99"]["value_ms"] == 99.0
    assert block["hot_latency_final"] is True
    over_limit = censored_latency_quantiles(
        list(range(1, 101)),
        status=["ok"] * 100,
        sample_id=[str(i) for i in range(100)],
        warmup=WARMED,
        warmup_complete=[True] * 98 + [False, False],
    )
    assert over_limit["n_rewarm_incomplete_requests"] == 2
    assert over_limit["hot_latency_final"] is False


def test_all_failures_leave_the_secondary_block_empty_and_warmup_states_are_counted():
    block = censored_latency_quantiles(
        [1000.0, 1001.0, 1002.0],
        status=["timeout", "timeout", "timeout"],
        sample_id=["a", "b", "c"],
        warmup_complete=[True, False, None],
    )
    assert block["success_only"]["quantiles"] is None and block["success_only"]["n"] == 0
    assert all(row["censored"] for row in block["quantiles"].values())
    assert block["n_requests_without_complete_warmup"] == 2
    assert (
        censored_latency_quantiles([1.0], status=["ok"], sample_id=["a"])["n_requests_without_complete_warmup"] is None
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(latency_ms=[], status=[], sample_id=[]), "nonempty"),
        (dict(latency_ms=[1.0, 2.0], status=["ok"], sample_id=["a", "b"]), "align"),
        (dict(latency_ms=[float("nan")], status=["ok"], sample_id=["a"]), "finite"),
        (dict(latency_ms=[-1.0], status=["ok"], sample_id=["a"]), "finite"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], censored=[1]), "booleans"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], censored=np.array([True])), "successful"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], min_warm_requests=0), "positive"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], quantiles=(("p100", 1.0),)), "strictly"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], warmup_complete=["yes"]), "warm-up"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], warmup={"declared": "yes", "performed": True}), "warm"),
        (dict(latency_ms=[1.0], status=["ok"], sample_id=["a"], warmup=[True]), "warm-up evidence"),
    ],
)
def test_invalid_inputs_fail_closed(kwargs, match):
    latency = kwargs.pop("latency_ms")
    with pytest.raises(ValueError, match=match):
        censored_latency_quantiles(latency, **kwargs)


def test_scorer_keeps_legacy_keys_and_adds_the_censored_block(metre_scene):
    from dataclasses import replace

    from passage_eval.execution import run_requests, score_run_artifact
    from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult

    calls = {"n": 0}

    def planner(request):
        calls["n"] += 1
        return PlanResult([(0, 0), (1, 1), (2, 2)], status="timeout" if calls["n"] == 2 else "ok")

    request = replace(PlanRequest.from_scene(metre_scene), budget=BudgetSpec(wall_time_ms=60000.0))
    run = run_requests(planner, [request], repeats=3)
    metrics, arrays = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    assert set(metrics["latency_all_repeats_ms"]) == {"p50", "p95", "p99"}  # legacy interpolated keys kept
    assert arrays["latency_censored"].tolist() == [False, True, False]
    block = metrics["latency_all_repeats_censored"]
    assert block["n_requests"] == 3 and block["n_censored"] == 1 and block["series"] == "warm"
    assert block["timing_profile"] == "supervised_total_v3" and block["latency_field"] == "total_wall_ms"
    assert block["quantiles"]["p99"]["censored"] is True


@pytest.mark.parametrize(
    "bench_spec,meets,reason",
    [
        (None, False, "no_declared_warmup"),  # run_requests alone records no warm-up
        ({"measure_latency": False, "latency_warmup": 5}, False, "no_declared_warmup"),  # the launcher's setting
        ({"measure_latency": True, "latency_warmup": 0}, False, "no_declared_warmup"),
        ({"measure_latency": True, "latency_warmup": 2}, True, None),
    ],
)
def test_scorer_qualifies_100_warm_requests_only_after_a_declared_warmup(metre_scene, bench_spec, meets, reason):
    from passage_eval.execution import run_requests, score_run_artifact
    from passage_eval.protocol.contracts import PlanRequest, PlanResult

    def planner(request):
        return PlanResult([(0, 0), (1, 1), (2, 2)])

    manifest = None if bench_spec is None else {"bench_spec": bench_spec}
    run = run_requests(planner, [PlanRequest.from_scene(metre_scene)], repeats=100, manifest=manifest)
    assert run.manifest["series"] == "warm"
    metrics, _ = score_run_artifact(run, [metre_scene], require_qualified_reference=False)
    block = metrics["latency_all_repeats_censored"]
    assert block["n_requests"] == 100 and block["meets_min_warm_requests"] is meets
    assert block["meets_min_warm_requests_reason"] == reason
    assert block["warmup"]["declared"] is meets and block["warmup"]["performed"] is meets


def _supervised_manifest(series="warm", latency_warmup=8, policy="disjoint_inputs", warmups=(8, 8)):
    passes = [
        {
            "series": series,
            "repeat": index,
            "warmup": None if requests is None else {"requests": requests, "requested": 8, "complete": requests == 8},
        }
        for index, requests in enumerate(warmups)
    ]
    return {
        "series": series,
        "bench_spec": {"measure_latency": True, "latency_warmup": latency_warmup},
        "supervision": {
            "timing_policy": {"latency_warmup": latency_warmup, "warmup": {"policy": policy}},
            "passes": passes + [{"series": "cold", "repeat": 0, "warmup": None}],
        },
    }


def test_supervised_warmup_evidence_reads_the_timing_policy_and_every_pass():
    from passage_eval.execution.timing import manifest_warmup

    full = manifest_warmup(_supervised_manifest())
    assert full["source"] == "supervised_timing_policy_and_pass_log"
    assert (full["declared"], full["performed"], full["passes"], full["passes_with_warmup"]) == (True, True, 2, 2)
    assert full["passes_with_complete_warmup"] == 2 and full["per_pass"] == 8
    partial = manifest_warmup(_supervised_manifest(warmups=(8, 3)))
    assert partial["performed"] is True and partial["passes_with_complete_warmup"] == 1
    missing = manifest_warmup(_supervised_manifest(warmups=(8, None)))
    assert missing["declared"] is True and missing["performed"] is False and missing["passes_with_warmup"] == 1
    for manifest in (
        _supervised_manifest(latency_warmup=0, policy="none", warmups=(None, None)),  # measure_latency false
        _supervised_manifest(policy="none"),
        _supervised_manifest(series="cold", warmups=(None,)),
    ):
        evidence = manifest_warmup(manifest)
        assert evidence["declared"] is False and evidence["performed"] is False
    legacy = _supervised_manifest()
    for entry in legacy["supervision"]["passes"]:
        if entry["warmup"] is not None:
            del entry["warmup"]["complete"]  # summaries written before completeness was recorded
    assert manifest_warmup(legacy)["performed"] is True
    assert manifest_warmup(legacy)["passes_with_complete_warmup"] is None
    assert manifest_warmup({"series": "warm"})["source"] == "none"


@pytest.mark.parametrize("seed", range(5))
def test_status_counts_partition_every_series(seed):
    rng = np.random.default_rng(seed)
    n = 60
    status = rng.choice(["ok", "timeout", "runtime_error", "invalid_output"], size=n).tolist()
    charged = [state != "ok" and bool(rng.integers(0, 2)) for state in status]
    flags = latency_censoring(status, charged)
    block = censored_latency_quantiles(
        rng.uniform(1, 100, size=n), status=status, sample_id=[f"s{i % 7}" for i in range(n)], censored=flags
    )
    assert block["n_ok"] + block["n_timeout"] + block["n_failed_non_timeout"] == block["n_requests"] == n
    assert block["n_censored"] == block["n_timeout"] + block["n_charged_not_measured"]
    assert block["n_failed_non_timeout"] == block["n_failed_non_timeout_measured"] + block["n_charged_not_measured"]
