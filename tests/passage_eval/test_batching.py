"""Worst-case wall time of a supervised batch and the plan check against a lease or batch timeout."""

import sys
from dataclasses import replace

import pytest

from passage_eval.cli import BenchSpec
from passage_eval.protocol.batching import (
    BatchPlanError,
    SupervisedBatchPlan,
    check_batch_plan,
    largest_batch_size,
    worst_case_batch_wall_ms,
)
from passage_eval.supervision.session import session_setup_reserve_ms
from passage_eval.supervision import WORKER_KILL_WAIT_S, sandbox
from passage_eval.supervision.supervisor import SupervisorConfig


def plan(**changes):
    settings = dict(
        batch_size=32,
        wall_time_ms=120_000.0,
        deadline_grace_ms=10.0,
        hard_timeout_ms=600_000.0,
        startup_timeout_s=60.0,
        max_restarts=100,
        fixed_overhead_ms=30_000.0,
        per_request_overhead_ms=50.0,
        seeds=1,
        per_seed_overhead_ms=0.0,
    )
    settings.update(changes)
    return SupervisedBatchPlan(**settings)


def test_seeds_and_the_per_seed_overhead_are_required():
    with pytest.raises(TypeError, match="seeds"):
        SupervisedBatchPlan(
            batch_size=1,
            wall_time_ms=1.0,
            deadline_grace_ms=0.0,
            hard_timeout_ms=1.0,
            startup_timeout_s=1.0,
            max_restarts=0,
            fixed_overhead_ms=0.0,
            per_request_overhead_ms=0.0,
        )


def test_simple_case_reduces_to_batch_times_budget_grace_and_restart_plus_overhead():
    bound = worst_case_batch_wall_ms(plan(batch_size=4, max_restarts=10))
    kill = WORKER_KILL_WAIT_S * 1000.0
    request = 120_000.0 + 10.0 + kill + 50.0
    start = 60_000.0 + kill
    assert bound["request_bound_ms"] == request and bound["worker_start_bound_ms"] == start
    assert bound["measured_requests"] == 4 and bound["counted_restarts_bound"] == 4
    assert bound["planned_worker_starts"] == 2  # the initial start and the pass's fresh worker
    assert bound["worst_case_wall_ms"] == pytest.approx(4 * (request + start) + 30_000.0 + 2 * start)


def test_session_admission_reserves_the_fresh_start_and_first_dynamic_warmup_before_workers_start():
    """Later warm-ups are separately admitted, so their count is not a static preflight reservation."""
    start = 120_000.0 + WORKER_KILL_WAIT_S * 1000.0
    request = 5_000.0 + 20.0 + WORKER_KILL_WAIT_S * 1000.0 + 1_000.0
    assert (
        session_setup_reserve_ms(
            roles=2,
            wall_time_ms=5_000.0,
            deadline_grace_ms=20.0,
            hard_timeout_ms=60_000.0,
            startup_timeout_s=120.0,
            persistence_ms=1_000.0,
        )
        == 2 * start + request
    )


def test_the_restart_cap_bounds_the_restart_term():
    capped = worst_case_batch_wall_ms(plan(max_restarts=2))
    assert capped["counted_restarts_bound"] == 2
    uncapped = worst_case_batch_wall_ms(plan(max_restarts=1000))
    assert uncapped["counted_restarts_bound"] == 32
    assert uncapped["worst_case_wall_ms"] - capped["worst_case_wall_ms"] == pytest.approx(
        30 * capped["worker_start_bound_ms"]
    )


def test_warmup_cold_pass_split_roles_and_rewarm_enlarge_the_bound():
    base = worst_case_batch_wall_ms(plan(max_restarts=1000))["worst_case_wall_ms"]
    latency = worst_case_batch_wall_ms(plan(max_restarts=1000, cold_passes=1, warmup_requests=8))
    assert latency["measured_requests"] == 64 and latency["warmup_requests"] == 8
    assert latency["rewarm_requests_bound"] == 0 and latency["worst_case_wall_ms"] > base
    rewarm = worst_case_batch_wall_ms(
        plan(max_restarts=1000, cold_passes=1, warmup_requests=8, rewarm_after_restart=True)
    )
    assert rewarm["rewarms_bound"] == 32 and rewarm["rewarm_requests_bound"] == 8 * 32
    assert rewarm["rewarm_restarts_outside_cap_bound"] == 1  # one failed re-warm per warm pass, one role
    assert rewarm["worst_case_wall_ms"] > latency["worst_case_wall_ms"]
    capped = worst_case_batch_wall_ms(plan(max_restarts=3, warmup_requests=8, rewarm_after_restart=True))
    assert capped["rewarms_bound"] == 3 and capped["rewarm_requests_bound"] == 24
    assert capped["counted_restarts_bound"] == 3 and capped["rewarm_restarts_outside_cap_bound"] == 1
    # without a counted restart there is no re-warm, hence no restart outside the cap
    no_cap = worst_case_batch_wall_ms(
        plan(max_restarts=0, warmup_requests=8, warm_passes=3, roles=2, rewarm_after_restart=True)
    )
    assert no_cap["rewarms_bound"] == 0 and no_cap["rewarm_restarts_outside_cap_bound"] == 0
    split_rewarm = worst_case_batch_wall_ms(
        plan(max_restarts=1000, warmup_requests=8, warm_passes=3, roles=2, rewarm_after_restart=True)
    )
    assert split_rewarm["rewarm_restarts_outside_cap_bound"] == 6
    without_warmup = worst_case_batch_wall_ms(plan(max_restarts=1000, rewarm_after_restart=True))
    assert without_warmup["rewarms_bound"] == 0 and without_warmup["rewarm_restarts_outside_cap_bound"] == 0
    split = worst_case_batch_wall_ms(plan(max_restarts=1000, roles=2))
    assert split["planned_worker_starts"] == 4 and split["counted_restarts_bound"] == 64


def test_every_seed_repeats_the_starts_the_restart_cap_and_the_requests():
    one = worst_case_batch_wall_ms(plan(batch_size=4, max_restarts=2, per_seed_overhead_ms=500.0))
    three = worst_case_batch_wall_ms(plan(batch_size=4, max_restarts=2, per_seed_overhead_ms=500.0, seeds=3))
    assert three["seeds"] == 3 and three["per_seed_wall_ms"] == one["per_seed_wall_ms"]
    # the fixed overhead is paid once per run, everything else once per seed
    assert three["worst_case_wall_ms"] - 30_000.0 == pytest.approx(3 * (one["worst_case_wall_ms"] - 30_000.0))
    assert one["per_seed_wall_ms"] == pytest.approx(
        500.0 + (2 + 2) * one["worker_start_bound_ms"] + 4 * one["request_bound_ms"]
    )


def test_three_seeds_refuse_a_plan_that_one_seed_would_fit():
    """The review case: BenchSpec's three default seeds under a 1200 s lease, split mode, no restarts."""
    spec = BenchSpec(
        "plan",
        "classical",
        "fixture",
        budget={"wall_time_ms": 120_000, "max_expansions": 65_537},
        measure_latency=False,
    )
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="process"),
        startup_timeout_s=60.0,
        max_restarts=0,
    )
    settings = dict(batch_size=4, split=True, fixed_overhead_ms=30_000.0, per_request_overhead_ms=50.0)
    derived = SupervisedBatchPlan.from_run_settings(spec, config, per_seed_overhead_ms=0.0, **settings)
    assert derived.seeds == len(spec.seeds) == 3
    bound = worst_case_batch_wall_ms(derived)
    request = 120_000.0 + config.deadline_grace_ms + WORKER_KILL_WAIT_S * 1000.0 + 50.0
    start = 60_000.0 + WORKER_KILL_WAIT_S * 1000.0
    assert bound["per_seed_wall_ms"] == pytest.approx(4 * start + 4 * request)  # 2 roles x (initial + pass)
    assert bound["worst_case_wall_ms"] == pytest.approx(30_000.0 + 3 * (4 * start + 4 * request))
    with pytest.raises(BatchPlanError, match="3 seed"):
        check_batch_plan(derived, 1_200_000.0)
    single = SupervisedBatchPlan.from_run_settings(
        replace(spec, seeds=[2026]), config, per_seed_overhead_ms=0.0, **settings
    )
    assert check_batch_plan(single, 1_200_000.0)["admissible"] is True
    with pytest.raises(ValueError, match="seeds"):
        SupervisedBatchPlan.from_run_settings(replace(spec, seeds=[]), config, per_seed_overhead_ms=0.0, **settings)


def test_no_wall_budget_uses_the_hard_timeout():
    bound = worst_case_batch_wall_ms(plan(wall_time_ms=None, hard_timeout_ms=200_000.0))
    assert bound["request_bound_ms"] == 200_000.0 + WORKER_KILL_WAIT_S * 1000.0 + 50.0


def test_check_refuses_a_plan_that_a_1200_s_batch_timeout_cannot_cover():
    """The gap analysis case: 32 tasks of 120 s cannot fit a 1200 s batch timeout."""
    with pytest.raises(BatchPlanError, match="exceeds the bound 1200000 ms"):
        check_batch_plan(plan(), 1_200_000.0)
    fits = largest_batch_size(plan(), 1_200_000.0)
    assert fits >= 1
    accepted = check_batch_plan(plan(batch_size=fits), 1_200_000.0)
    assert accepted["admissible"] is True and accepted["margin_ms"] >= 0
    with pytest.raises(BatchPlanError):
        check_batch_plan(plan(batch_size=fits + 1), 1_200_000.0)
    assert largest_batch_size(plan(), 1.0) == 0


def test_from_run_settings_mirrors_the_supervisor_series():
    spec = BenchSpec(
        "plan",
        "classical",
        "fixture",
        budget={"wall_time_ms": 120_000, "max_expansions": 65_537},
        latency_repeats=2,
        latency_warmup=8,
    )
    config = SupervisorConfig(
        worker_python=sys.executable,
        isolation=sandbox.IsolationConfig(level="process"),
        startup_timeout_s=90.0,
        max_restarts=5,
        timing_profile="supervised_service_rewarm_v1",
    )
    derived = SupervisedBatchPlan.from_run_settings(
        spec,
        config,
        batch_size=16,
        split=True,
        fixed_overhead_ms=1000.0,
        per_request_overhead_ms=5.0,
        per_seed_overhead_ms=200.0,
    )
    assert (derived.cold_passes, derived.warm_passes, derived.warmup_requests, derived.roles) == (1, 2, 8, 2)
    assert derived.wall_time_ms == 120_000 and derived.startup_timeout_s == 90.0 and derived.max_restarts == 5
    assert derived.rewarm_after_restart is True
    assert derived.seeds == len(spec.seeds) == 3 and derived.per_seed_overhead_ms == 200.0
    unmeasured = SupervisedBatchPlan.from_run_settings(
        BenchSpec("plan", "classical", "fixture", measure_latency=False, latency_warmup=8, seeds=[7]),
        SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig(level="process")),
        batch_size=16,
        split=False,
        fixed_overhead_ms=0.0,
        per_request_overhead_ms=0.0,
        per_seed_overhead_ms=0.0,
    )
    assert (unmeasured.cold_passes, unmeasured.warm_passes, unmeasured.warmup_requests) == (0, 1, 0)
    assert unmeasured.wall_time_ms is None and unmeasured.rewarm_after_restart is False and unmeasured.seeds == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"batch_size": -1},
        {"batch_size": 1.5},
        {"wall_time_ms": 0.0},
        {"wall_time_ms": float("inf")},
        {"deadline_grace_ms": -1.0},
        {"startup_timeout_s": 0.0},
        {"max_restarts": True},
        {"roles": 0},
        {"cold_passes": 0, "warm_passes": 0},
        {"fixed_overhead_ms": float("nan")},
        {"rewarm_after_restart": "yes"},
        {"seeds": 0},
        {"seeds": 2.0},
        {"per_seed_overhead_ms": -1.0},
    ],
)
def test_invalid_plans_are_refused(changes):
    with pytest.raises(ValueError):
        plan(**changes)
    with pytest.raises(ValueError):
        check_batch_plan(plan(), 0.0)


# --- dynamic request admission (R7): the campaign guard -----------------------------------------------------------


def sizing(**changes):
    from passage_eval.protocol.batching import admission_sizing

    settings = dict(
        lease_ms=1_190_000.0,
        batch_size=1024,
        warmup_requests=16,
        procedures={
            "seg": {"roles": 2, "wall_time_ms": 120_000.0, "observed_request_ms": 25.0},
            "reg": {"roles": 2, "wall_time_ms": 120_000.0, "observed_request_ms": 23.0},
        },
        fixed_overhead_ms=30_000.0,
        per_seed_overhead_ms=10_000.0,
        per_request_overhead_ms=50.0,
        persistence_ms=5_000.0,
        safety_margin=3.0,
        deadline_grace_ms=10.0,
        hard_timeout_ms=600_000.0,
        startup_timeout_s=120.0,
    )
    settings.update(changes)
    return admission_sizing(**settings)


def test_dynamic_admission_accepts_observed_warmups_without_reserving_every_warmup_at_the_wall_budget():
    """Every warm-up is admitted dynamically; only the following measured request needs the strict reserve."""
    result = sizing()
    seg = result["per_procedure"]["seg"]
    assert seg["request_reserve_ms"] == 120_000 + 10 + 15_000 + 5_000
    assert seg["expected_session_ms"] == 30_000 + 10_000 + (16 + 1024) * 75.0 + 5_000
    assert seg["sanity_floor_ms"] == 30_000 + 10_000 + 2 * 135_000 + 16 * 75.0 + 140_010
    assert result["admissible"] and result["window_share"].startswith("equal split")
    worst = worst_case_batch_wall_ms(
        plan(batch_size=1024, roles=2, warmup_requests=16, max_restarts=1024, rewarm_after_restart=True)
    )
    assert worst["worst_case_wall_ms"] > 1_190_000.0


def test_a_lot_of_32_is_admitted_when_16_warmups_are_observed_fast():
    result = sizing(batch_size=32)
    assert result["admissible"] is True


@pytest.mark.parametrize(
    "changes, flag",
    [
        ({"safety_margin": 2.99}, "margin_ok"),
        ({"lease_ms": 600_000.0}, "margin_ok"),
        ({"lease_ms": 900_000.0, "startup_timeout_s": 200.0}, "sanity_ok"),
    ],
)
def test_the_margin_and_the_sanity_floor_are_each_enforced(changes, flag):
    result = sizing(**changes)
    assert result[flag] is False and result["admissible"] is False


def test_without_a_wall_budget_the_request_reserve_uses_the_hard_timeout():
    result = sizing(procedures={"astar": {"roles": 1, "wall_time_ms": None, "observed_request_ms": 30.0}})
    assert result["per_procedure"]["astar"]["request_reserve_ms"] == 600_000 + 15_000 + 5_000


@pytest.mark.parametrize(
    "changes",
    [{"lease_ms": 0}, {"batch_size": 0}, {"safety_margin": 0}, {"procedures": {}}, {"persistence_ms": -1.0}],
)
def test_invalid_sizing_inputs_are_refused(changes):
    with pytest.raises(ValueError):
        sizing(**changes)


def test_a_campaign_session_has_no_cold_pass_in_its_worst_case():
    from types import SimpleNamespace

    spec = SimpleNamespace(
        seeds=[2026], budget={"wall_time_ms": 1000.0}, measure_latency=True, latency_repeats=1, latency_warmup=16
    )
    config = SimpleNamespace(
        deadline_grace_ms=10.0,
        hard_timeout_ms=1.0,
        startup_timeout_s=1.0,
        max_restarts=4,
        timing_profile="supervised_service_rewarm_v1",
    )
    common = dict(batch_size=4, split=True, fixed_overhead_ms=0, per_request_overhead_ms=0, per_seed_overhead_ms=0)
    assert SupervisedBatchPlan.from_run_settings(spec, config, **common).cold_passes == 1
    assert SupervisedBatchPlan.from_run_settings(spec, config, session=True, **common).cold_passes == 0
