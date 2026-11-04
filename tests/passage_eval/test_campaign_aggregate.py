"""Campaign aggregation: concatenated per-task rows, parent blocks, population counts, penalties and latency.

The equivalence test scores the same synthetic tasks once as one canonical run and once as three batches; the
aggregation of the batches must equal the canonical scorer on the same rows. In-process runs on synthetic 3x3
scenes only.
"""

from dataclasses import replace

import numpy as np
import pytest
from passage_eval.campaign.aggregate import SIX_METRICS, aggregate_scored_batches
from passage_eval.metrics.comparison import FAILURE_REGRET, load_scored
from passage_eval.metrics.latency import censored_latency_quantiles
from passage_eval.metrics.summary import aggregate_primary_metrics

DETOUR = [(0, 0), (0, 1), (1, 2), (2, 2)]
#: Per task: (status, route); a detour is legal but costs more than the diagonal reference.
PLAN = {
    "s0": ("ok", None),
    "s1": ("ok", DETOUR),
    "s2": ("timeout", None),
    "s3": ("ok", DETOUR),
    "s4": ("ok", None),
    "s5": ("ok", DETOUR),
}
BATCHES = [["s0", "s1"], ["s2", "s3"], ["s4", "s5"]]
BLOCKS = {"s0": "geo5:0:0", "s1": "geo5:0:0", "s2": "geo5:0:1", "s3": "geo5:1:1", "s4": "geo5:1:1", "s5": "geo5:2:2"}
POPULATION = {"n_source": 7, "n_excluded": 1, "n_admissible": 6}


def scored_run(directory, scenes):
    """One in-process run of ``scenes``, scored canonically and loaded with the public loader."""
    import hashlib
    import json

    from passage.audit import qualified_reference_for_scene
    from passage_eval.execution import run_requests, score_run_artifact, write_scored_artifact
    from passage_eval.protocol.contracts import BudgetSpec, PlanRequest, PlanResult

    references = {scene.sample_id: qualified_reference_for_scene(scene) for scene in scenes}
    by_id = {scene.sample_id: scene for scene in scenes}

    class Planner:
        def plan(self, request):
            status, route = PLAN[request.sample_id]
            return PlanResult(route or by_id[request.sample_id].oracle_path, status=status)

    manifest = {
        "data_identity": {"source": "fixture", "revision": "frozen", "split": "validation"},
        "reference_identity": hashlib.sha256(json.dumps(references, sort_keys=True).encode()).hexdigest(),
        "seed": 2026,
    }
    requests = [PlanRequest.from_scene(scene, budget=BudgetSpec()) for scene in scenes]
    run = run_requests(Planner(), requests, output_dir=directory, manifest=manifest, repeats=1, series="warm")
    metrics, arrays = score_run_artifact(run, scenes, reference_store=references)
    write_scored_artifact(run, metrics, arrays)
    return load_scored(directory), metrics


@pytest.fixture
def scenes(metre_scene):
    return {sample_id: replace(metre_scene, sample_id=sample_id) for sample_id in PLAN}


@pytest.fixture
def batched(tmp_path, scenes):
    return [(scored_run(tmp_path / f"batch-{i}", [scenes[s] for s in ids])[0], ids) for i, ids in enumerate(BATCHES)]


def aggregate(batches, **overrides):
    values = dict(parent_order=list(PLAN), parent_blocks=BLOCKS, population=POPULATION)
    values.update(overrides)
    return aggregate_scored_batches(batches, **values)


def test_the_aggregation_of_batches_equals_the_canonical_scorer_on_the_same_rows(tmp_path, scenes, batched):
    full, full_metrics = scored_run(tmp_path / "full", [scenes[s] for s in PLAN])
    result = aggregate(batched)
    for name in SIX_METRICS:
        expected = full_metrics[name]
        if isinstance(expected, float) and np.isnan(expected):
            assert result["metrics"][name] is None
        else:
            assert result["metrics"][name] == pytest.approx(expected, rel=1e-12, abs=0)
    assert result["canonical_equivalence"]["equal"] is True
    assert result["canonical_equivalence"]["max_abs_difference"] == 0.0
    assert [row["n_tasks"] for row in result["canonical_equivalence"]["batches"]] == [2, 2, 2]
    # the latency block is the censored rule applied to every request of every batch, concatenated
    runs = [run for run, _ in batched]
    expected = censored_latency_quantiles(
        np.concatenate([run["arrays"]["latency_ms"] for run in runs]),
        status=np.concatenate([run["arrays"]["status"] for run in runs]),
        sample_id=np.concatenate([run["arrays"]["sample_id"] for run in runs]),
        censored=np.concatenate([run["arrays"]["latency_censored"] for run in runs]),
        warmup={"declared": False, "performed": False, "batches": 3},
    )
    assert result["latency"] == expected
    block = result["latency"]
    assert (block["n_requests"], block["n_distinct_scenes"], block["n_repeats"], block["n_censored"]) == (6, 6, 1, 1)
    assert block["meets_min_warm_requests"] is False and block["meets_min_warm_requests_reason"] == "no_declared_warmup"
    assert set(block["quantiles"]) == {"p50", "p95", "p99"}


def test_failures_stay_in_the_denominator_with_the_canonical_penalty(batched):
    result = aggregate(batched)
    assert result["failures"]["n_failed"] == 1 and result["failures"]["statuses"] == {"ok": 5, "timeout": 1}
    assert result["metrics"]["success_rate"] == pytest.approx(5 / 6)
    assert result["metrics"]["cost_regret_p90"] > 1e5  # the timeout's penalty, not a dropped row
    assert result["population"] == {**POPULATION, "n_scored": 6}
    regrets = np.concatenate([run["arrays"]["regret"] for run, _ in batched])
    assert np.sum(np.isclose(regrets, FAILURE_REGRET)) == 1


def test_quantiles_come_from_the_concatenated_rows_never_from_batch_quantiles(batched):
    result = aggregate(batched)
    rows = {
        name: np.concatenate(
            [run["arrays"][name][[run["indices"][(s, 0, "warm")] for s in ids]] for run, ids in batched]
        )
        for name in ("regret", "assd", "h95", "success", "obstacle_violation", "valid_path", "obstacle_cells")
    }
    direct = aggregate_primary_metrics(rows)
    batch_medians = [
        aggregate_primary_metrics({k: run["arrays"][k] for k in rows})["cost_regret_median"] for run, _ in batched
    ]
    assert result["metrics"]["cost_regret_median"] == direct["cost_regret_median"]
    assert result["metrics"]["cost_regret_median"] != pytest.approx(float(np.mean(batch_medians)))
    assert result["quantile_rule"].startswith("every quantile is computed on the concatenated per-task rows")


def test_parent_blocks_are_kept_and_a_null_block_is_refused(batched):
    result = aggregate(batched)
    assert result["geographic_blocks"]["tasks"] == BLOCKS and result["geographic_blocks"]["number_of_blocks"] == 4
    with pytest.raises(ValueError, match="null parent geographic block"):
        aggregate(batched, parent_blocks=dict(BLOCKS, s3=None))


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda b: b.pop(), "cover the admissible roster"),
        (lambda b: b.append(b[0]), "more than one batch"),
        (lambda b: b.reverse(), "parent roster order"),
        (lambda b: b.__setitem__(0, (b[0][0], ["s1", "s0"])), "scored roster differs"),
    ],
)
def test_missing_duplicated_misordered_or_rebound_batches_are_refused(batched, change, message):
    batches = list(batched)
    change(batches)
    with pytest.raises(ValueError, match=message):
        aggregate(batches)


@pytest.mark.parametrize(
    "population, message",
    [
        ({"n_source": 7, "n_excluded": 0, "n_admissible": 6}, "n_source must equal"),
        ({"n_source": 6, "n_excluded": 1, "n_admissible": 5}, "number n_admissible"),
        ({"n_source": 6, "n_admissible": 6}, "needs n_source"),
    ],
)
def test_the_source_excluded_and_admissible_counts_must_be_consistent(batched, population, message):
    with pytest.raises(ValueError, match=message):
        aggregate(batched, population=population)


def test_a_batch_whose_scorer_metrics_differ_is_not_equivalent(batched):
    import copy

    batches = [(dict(run), ids) for run, ids in batched]
    stored = copy.deepcopy(batches[1][0].get("metrics") or load_scored_metrics(batches[1][0]))
    stored["cost_regret_median"] = (stored["cost_regret_median"] or 0.0) + 0.5
    batches[1][0]["metrics"] = stored
    result = aggregate(batches)
    assert result["canonical_equivalence"]["equal"] is False
    rows = result["canonical_equivalence"]["batches"]
    assert [row["equal"] for row in rows] == [True, False, True]
    assert rows[1]["max_abs_difference"] == pytest.approx(0.5)


def load_scored_metrics(run):
    from passage_eval.campaign.aggregate import json_load_metrics

    return json_load_metrics(run["path"])


def test_a_failed_task_without_the_canonical_penalty_is_refused(batched):
    run, ids = batched[1]  # s2 timed out
    arrays = dict(run["arrays"])
    regret = np.array(arrays["regret"], dtype=float)
    regret[run["indices"][("s2", 0, "warm")]] = 10.0
    arrays["regret"] = regret
    batches = list(batched)
    batches[1] = (dict(run, arrays=arrays), ids)
    with pytest.raises(ValueError, match="lost its canonical penalty"):
        aggregate(batches)


def _seed_result(batched, **overrides):
    result = aggregate(batched)
    result.update(overrides)
    return result


def test_one_seed_is_repeated_at_the_top_level_and_several_seeds_are_not_reduced(batched):
    from passage_eval.campaign.aggregate import SEED_RESULT_FIELDS, combine_seeds

    one = _seed_result(batched)
    single = combine_seeds({"2026": one}, [2026])
    assert single["seed"] == 2026 and single["seeds"] == [2026] and single["per_seed"]["2026"] == one
    assert all(single[name] == one[name] for name in SEED_RESULT_FIELDS)
    assert single["canonical_equivalence"]["equal"] is True and single["population"] == one["population"]
    other = _seed_result(batched)
    other["canonical_equivalence"] = dict(other["canonical_equivalence"], equal=False, max_abs_difference=0.25)
    several = combine_seeds({"1234": one, "2026": one, "3407": other}, [1234, 2026, 3407])
    assert several["seed"] is None and all(several[name] is None for name in SEED_RESULT_FIELDS)
    assert several["seed_reduction"].startswith("none")
    assert several["canonical_equivalence"]["equal"] is False
    assert several["canonical_equivalence"]["per_seed"] == {"1234": True, "2026": True, "3407": False}
    assert several["canonical_equivalence"]["max_abs_difference"] == 0.25
    with pytest.raises(ValueError, match="one aggregate per seed"):
        combine_seeds({"2026": one}, [2026, 3407])


def test_seed_2026_is_explicitly_descriptive_without_a_future_analysis_commitment(batched):
    from passage_eval.campaign.aggregate import combine_seeds

    combined = combine_seeds({"2026": _seed_result(batched)}, [2026])
    interpretation = combined["seed_interpretation"]
    assert interpretation["status"] == "descriptive_single_seed_test_exposed_non_confirmatory"
    assert interpretation["seed"] == 2026
    assert "future_seed_plan" not in interpretation
    assert "bootstrap" not in str(interpretation).lower()
    assert "holm" not in str(interpretation).lower()


def test_a_real_campaign_aggregate_carries_no_marker_the_admission_gate_refuses(batched):
    """G2 scans rehearsal aggregates for synthetic and unsealed markers; the aggregator's own text must pass."""
    from passage_eval.campaign.aggregate import LIMITATIONS, combine_seeds
    from passage_eval.protocol import admission

    combined = combine_seeds({"2026": _seed_result(batched)}, [2026])
    combined["limitations"] = list(LIMITATIONS)
    assert combined["latency"]["status"] == "proposed_not_sealed"
    assert admission.non_evidence_markers(combined) == []
    assert admission.unsealed_markers(admission._without_installed_rule_status(combined)) == []


def _imputed(sample_id, charged=20000.0):
    from passage_eval.campaign import IMPUTED_SCHEMA

    return {
        "schema": IMPUTED_SCHEMA,
        "sample_id": sample_id,
        "unit": "000001.0",
        "charged_latency_ms": charged,
        "sha256": "e" * 64,
    }


def test_an_r7_candidate_failure_keeps_the_task_in_the_denominator_with_the_canonical_penalty(batched):
    """A single task that killed its batch twice: no scorer output, but every quality and latency statistic counts it."""
    (first, ids0), _, _ = batched
    parts = [(first, ids0), _imputed("s2")]  # s2 is a candidate failure imputed by R7 after the first batch
    order = ["s0", "s1", "s2"]
    result = aggregate(
        parts,
        parent_order=order,
        parent_blocks={s: BLOCKS[s] for s in order},
        population={"n_source": 4, "n_excluded": 1, "n_admissible": 3},
    )
    assert result["imputed_failures"]["n"] == 1 and result["imputed_failures"]["tasks"][0]["sample_id"] == "s2"
    assert result["failures"]["n_failed"] == 1 and result["failures"]["statuses"] == {"ok": 2, "runtime_error": 1}
    assert result["metrics"]["success_rate"] == pytest.approx(2 / 3)
    assert result["metrics"]["cost_regret_p90"] > 1e5  # the canonical penalty of the imputed task
    block = result["latency"]
    assert block["n_requests"] == 3 and block["n_censored"] == 1 and block["n_charged_not_measured"] == 1
    # the imputed task has no scorer output: canonical equivalence covers the scored batch only
    assert [row["n_tasks"] for row in result["canonical_equivalence"]["batches"]] == [2]
    with pytest.raises(ValueError, match="more than one batch"):
        aggregate(
            [(first, ids0), _imputed("s1")],
            parent_order=order[:2],
            parent_blocks={s: BLOCKS[s] for s in order[:2]},
            population={"n_source": 2, "n_excluded": 0, "n_admissible": 2},
        )


def test_the_cold_start_is_summarised_per_session_and_never_enters_the_warm_row():
    from passage_eval.campaign.aggregate import cold_start_summary, restart_summary

    sessions = [
        {
            "cold_start": {
                "worker_load_ms": {"predictor": load, "decoder": load / 2},
                "first_warmup_request": {"latency_ms": first, "input_sha256": "a" * 64},
            }
        }
        for load, first in ((100.0, 6100.0), (300.0, 2500.0), (200.0, 4000.0))
    ]
    summary = cold_start_summary(sessions)
    assert summary["n_sessions"] == 3 and summary["worker_load_ms"]["predictor"] == {
        "n": 3,
        "median": 200.0,
        "max": 300.0,
    }
    assert summary["first_warmup_request_ms"] == {"n": 3, "median": 4000.0, "max": 6100.0}
    assert summary["n_distinct_first_warmup_inputs"] == 1 and summary["repetitions_of_the_first_warmup_input"] == 3
    manifests = [
        {
            "supervision": {
                "restart_cap": {"unplanned_restarts_used": 2, "rewarm_restarts_outside_cap": 1},
                "passes": [{"rewarms": [{"complete": True}, {"complete": False}]}],
            }
        },
        {"supervision": {"restart_cap": {"unplanned_restarts_used": 0}, "passes": [{"rewarms": []}]}},
    ]
    assert restart_summary(manifests) == {
        "n_sessions": 2,
        "counted_restarts": 2,
        "restarts_outside_cap": 1,
        "rewarms": 2,
        "failed_rewarms": 1,
    }
