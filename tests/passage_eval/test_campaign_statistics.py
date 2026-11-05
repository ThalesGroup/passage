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

"""Campaign statistics: per-seed block intervals, the confirmatory family and session-bootstrap latency intervals.

Synthetic aggregates only, built here with the aggregator's own metric and latency functions (and, in the last
test, by ``aggregate_scored_batches`` from in-process scored runs of synthetic 3x3 scenes).
"""

import copy
import json
import re
from dataclasses import replace

import numpy as np
import pytest
from passage_eval.campaign import AGGREGATE_SCHEMA
from passage_eval.campaign import statistics as S
from passage_eval.execution.artifacts import json_safe
from passage_eval.metrics.comparison import FAILURE_REGRET
from passage_eval.metrics.latency import censored_latency_quantiles
from passage_eval.metrics.summary import aggregate_primary_metrics
from passage_eval.protocol.block_census import PROPOSED_CRITERION

SIX = [
    "cost_regret_median",
    "cost_regret_p90",
    "path_assd",
    "path_hausdorff_p95",
    "success_rate",
    "obstacle_violation_rate",
]


def criterion(**changes):
    return {**PROPOSED_CRITERION, "status": "sealed for this test", **changes}


def rows(n, rng, *, offset, failures=0):
    regret = rng.uniform(0.0, 0.05, size=n) + offset
    valid = np.ones(n, dtype=bool)
    valid[:failures] = False
    regret[~valid] = FAILURE_REGRET
    return {
        "regret": regret,
        "assd": np.where(valid, 1.0 + offset * 10 + rng.uniform(0, 0.1, size=n), np.nan),
        "h95": np.where(valid, 2.0 + offset * 10 + rng.uniform(0, 0.1, size=n), np.nan),
        "success": valid.copy(),
        "obstacle_violation": np.zeros(n, dtype=bool),
        "valid_path": valid.copy(),
        "obstacle_cells": np.zeros(n, dtype=np.int64),
    }


def sessions_for(ids, rng, *, size, scale, timeouts=()):
    sessions = []
    for number, start in enumerate(range(0, len(ids), size)):
        members = ids[start : start + size]
        latency = rng.uniform(scale, 2 * scale, size=len(members)) + number
        status = ["timeout" if sample_id in timeouts else "ok" for sample_id in members]
        sessions.append(
            {
                "session": number,
                "sample_ids": list(members),
                "latency_ms": latency.tolist(),
                "censored": [value == "timeout" for value in status],
                "status": status,
                "warmup_complete": None,
            }
        )
    return sessions


def aggregate(procedure, ids, blocks, seeds, *, offset, seed=0, failures=0, size=4, scale=10.0, timeouts=()):
    rng = np.random.default_rng([seed, {"seg": 1, "reg": 2}[procedure]])
    per_seed = {}
    for training_seed in seeds:
        values = rows(len(ids), rng, offset=offset, failures=failures)
        sessions = sessions_for(ids, rng, size=size, scale=scale, timeouts=timeouts)
        metrics = aggregate_primary_metrics(values)
        latency = censored_latency_quantiles(
            np.concatenate([s["latency_ms"] for s in sessions]),
            status=np.concatenate([s["status"] for s in sessions]),
            sample_id=np.concatenate([s["sample_ids"] for s in sessions]),
            censored=np.concatenate([s["censored"] for s in sessions]).astype(bool),
            warmup={"declared": True, "performed": True, "batches": len(sessions)},
        )
        task_rows = {"task_id": list(ids), **{key: value.tolist() for key, value in values.items()}}
        task_rows["status"] = ["ok" if valid else "runtime_error" for valid in values["valid_path"]]
        per_seed[str(training_seed)] = json_safe(
            {
                "metrics": {name: metrics[name] for name in SIX},
                "geographic_blocks": {"tasks": dict(zip(ids, blocks))},
                "completeness": {"complete": True, "n_records": len(ids), "n_admissible": len(ids)},
                "task_rows": task_rows,
                "latency_sessions": sessions,
                "latency": latency,
            }
        )
    population = {"n_source": len(ids), "n_excluded": 0, "n_admissible": len(ids), "n_scored": len(ids)}
    return {
        "schema": AGGREGATE_SCHEMA,
        "procedure": procedure,
        "campaign_id": "synthetic-campaign",
        "target": "host",
        "split": "validation",
        "plan_sha256": "a" * 64,
        "admissible_roster_sha256": "b" * 64,
        "reference_admission_chain_sha256": "c" * 64,
        "seeds": list(seeds),
        "population": population,
        "budget": {"wall_time_ms": 120000},
        "timing_profile": "supervised_service_rewarm_v1",
        "harness": {"commit": "d" * 40, "source_sha256": "e" * 64},
        "per_seed": per_seed,
    }


def world(n_blocks=32, per_block=2, seeds=(2026,), *, reverse=False, **kwargs):
    """Seg and Reg aggregates; ``reverse`` gives a parent order that is not the lexicographic order of the ids."""
    ids = [f"t{i:03d}" for i in range(n_blocks * per_block)]
    if reverse:
        ids.reverse()
    blocks = [f"geo5:{i // per_block:02d}:00" for i in range(len(ids))]
    seg = aggregate(
        "seg", ids, blocks, seeds, offset=0.0, **{k: v for k, v in kwargs.items() if not k.startswith("reg_")}
    )
    reg_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("reg_")}
    reg = aggregate("reg", ids, blocks, seeds, offset=0.05, **reg_kwargs)
    return seg, reg


def stats(aggregates, **kwargs):
    values = dict(bootstrap_seed=11, permutation_seed=12, latency_bootstrap_seed=13, draws=200, permutations=200)
    values.update(kwargs)
    return S.campaign_statistics(aggregates, **values)


# ----------------------------------------------------------------------------------------------- one seed (2026)


def test_a_single_seed_gives_descriptive_block_intervals_and_no_p_value_or_holm():
    result = stats(world(), criterion=criterion())
    quality = result["quality"]
    assert result["status"] == "descriptive" and result["label"] == S.SINGLE_SEED_LABEL
    assert S.SINGLE_SEED_LABEL == "descriptive, single seed, exposed public test, not confirmatory"
    assert quality["mode"] == "descriptive_single_seed"
    assert quality["p_values"] is None and quality["holm"] is None and quality["decision"] is None
    intervals = quality["intervals"]["2026"]
    assert list(intervals) == SIX
    regret = intervals["cost_regret_median"]
    assert regret["delta"] < 0 and regret["ci_high"] < 0 and regret["n_blocks"] == 32 and regret["n_bootstrap"] == 200
    assert intervals["success_rate"]["delta"] == 0.0
    assert result["blocks"]["status"] == "meets_criterion"  # adequate blocks still give no p-value alone
    assert result["resampling"]["bootstrap_seed"] == 11 and result["quantile_methods"]["quality"] == "linear"
    assert result["quantile_methods"]["latency"] == "inverted_cdf"
    again = stats(world(), criterion=criterion())
    assert again == result  # the sealed seeds make the record reproducible
    assert stats(world(), criterion=criterion(), bootstrap_seed=99)["quality"] != quality


def test_the_resampling_seeds_have_no_default_and_are_validated():
    with pytest.raises(TypeError):
        S.campaign_statistics(world(), permutation_seed=1, latency_bootstrap_seed=2)
    for name in ("bootstrap_seed", "permutation_seed", "latency_bootstrap_seed"):
        with pytest.raises(S.StatisticsInputError, match=name):
            stats(world(), **{name: -1})
        with pytest.raises(S.StatisticsInputError, match=name):
            stats(world(), **{name: True})
    with pytest.raises(S.StatisticsInputError, match="draws"):
        stats(world(), draws=0)
    for alpha in (0, 1, True, "0.05"):
        with pytest.raises(S.StatisticsInputError, match="alpha"):
            stats(world(), alpha=alpha)


# --------------------------------------------------------------------------------------------- three seeds


#: The sealed decisions of the synthetic protocol records (values for unit tests only, sealed by no author); the
#: resampling equals the ``stats`` defaults.
SEALED_ANALYSIS = {
    "a": "seg",
    "b": "reg",
    "target": "host",
    "seeds": [1234, 2026, 3407],
    "alpha": 0.05,
    "draws": 200,
    "permutations": 200,
    "bootstrap_seed": 11,
    "permutation_seed": 12,
    "latency_bootstrap_seed": 13,
}
SEALED_BLOCKS = {
    "rule": "source_window_union",
    "block_degrees": 5,
    "missing_footprint": "unplaced_block",
    "min_blocks": 30,
    "max_block_share": 0.10,
    "max_unplaced_fraction": 0.005,
}


def protocol_record(seed, harness, *, analysis=None, blocks=None, approve=True, mutate=None):
    """A sealed and approved ``passage-test-admission/v1`` record of one seed's campaign (decisions other than the
    seeds, the block analysis and the confirmatory analysis are sealed with null values: the statistics read only
    those three)."""
    from passage_eval.protocol import admission as A

    record = A.template_record(f"synthetic-record-{seed}")
    record.update(status="sealed", harness=dict(harness), notes="synthetic record of a unit test")
    record["protocol"]["version"] = "1.0"
    decisions = record["protocol"]["decisions"]
    for entry in decisions.values():
        entry["status"] = "sealed"
    decisions["seeds"]["value"] = [seed]
    decisions["block_analysis"]["value"] = {**SEALED_BLOCKS, **(blocks or {})}
    decisions["confirmatory_analysis"]["value"] = {**SEALED_ANALYSIS, **(analysis or {})}
    if mutate is not None:
        mutate(record)
    if approve:
        record["approval"] = {
            "approver": "Synthetic Approver",
            "approved_at": "2026-11-01T00:00:00+00:00",
            "approved_record_sha256": A.record_core_sha256(record),
        }
    return record


def sealed_family(aggregates=None, *, split="test", per_seed=None, **record_options):
    """One test campaign per seed, each naming its sealed protocol record; ``per_seed`` maps a seed to the record
    options of that seed alone. Returns the aggregates and the records keyed by digest."""
    from passage_eval.io import pinned

    campaigns = per_seed_campaigns(aggregates or world(seeds=(1234, 2026, 3407)))
    records = {}
    for seed in sorted({a["seeds"][0] for a in campaigns}):
        members = [a for a in campaigns if a["seeds"] == [seed]]
        options = dict(record_options, **((per_seed or {}).get(seed) or {}))
        record = protocol_record(seed, members[0]["harness"], **options)
        digest = pinned.canonical_sha256(record)
        for aggregate in members:
            aggregate.update(protocol_record_sha256=digest, split=split)
        records[digest] = record
    return campaigns, records


def test_three_sealed_seeds_with_adequate_blocks_give_the_confirmatory_family_with_holm():
    campaigns, records = sealed_family()
    result = stats(campaigns, protocol_records=records)
    quality = result["quality"]
    assert result["status"] == "confirmatory" and quality["mode"] == "confirmatory"
    sealing = result["sealing"]
    assert sealing["confirmatory_allowed"] is True and sealing["reasons"] == []
    assert sealing["protocol_records"] == {
        str(a["seeds"][0]): a["protocol_record_sha256"] for a in campaigns if a["procedure"] == "seg"
    }
    assert sealing["confirmatory_analysis"] == SEALED_ANALYSIS and sealing["block_analysis"] == SEALED_BLOCKS
    assert result["blocks"]["criterion"]["source"] == "block_analysis of the sealed protocol records"
    assert result["blocks"]["criterion_sealed"] is True and result["blocks"]["status"] == "meets_criterion"
    assert set(quality["p_values"]) == set(SIX) and set(quality["holm"]) == set(SIX)
    test = quality["paired_test"]
    assert test["schema"] == "passage-paired-block-randomization/v1" and test["method"] == "monte_carlo"
    assert test["random_seed"] == 12 and test["permutations"] == 200 and test["number_of_blocks"] == 32
    assert quality["p_values"]["cost_regret_median"] == pytest.approx(1 / 201)
    assert quality["holm"]["cost_regret_median"]["rejected"] is True
    # no success difference in any draw: every swap is as extreme as the observed zero
    assert quality["p_values"]["success_rate"] == 1.0 and quality["holm"]["success_rate"]["rejected"] is False
    assert quality["votes"]["cost_regret_median"] == "win_a" and quality["votes"]["success_rate"] == "tie"
    assert quality["decision"] == "partial_evidence"
    assert set(quality["intervals"]["path_assd"]["per_seed"]) == {"1234", "2026", "3407"}
    assert result["family"] == {"a": "seg", "b": "reg", "contrast": "a - b", "metrics": SIX, "alpha": 0.05}


def test_sixteen_blocks_or_fewer_use_the_exact_test():
    campaigns, records = sealed_family(
        world(n_blocks=10, per_block=4, seeds=(1234, 2026, 3407)), blocks={"min_blocks": 8}
    )
    result = stats(campaigns, protocol_records=records)
    assert result["quality"]["paired_test"]["method"] == "exact"
    assert result["quality"]["paired_test"]["permutations"] == 2**10
    assert result["quality"]["p_values"]["cost_regret_median"] == pytest.approx(2 / 2**10)


def test_inadequate_blocks_keep_three_sealed_seeds_descriptive():
    campaigns, records = sealed_family(world(n_blocks=10, per_block=4, seeds=(1234, 2026, 3407)))
    result = stats(campaigns, protocol_records=records)
    assert result["sealing"]["confirmatory_allowed"] is True
    assert result["blocks"]["status"] == "fails_criterion" and result["blocks"]["adequacy"]["meets_min_blocks"] is False
    quality = result["quality"]
    assert quality["mode"] == "descriptive_blocks_inadequate" and result["status"] == "descriptive"
    assert quality["p_values"] is None and quality["holm"] is None and quality["decision"] is None
    assert quality["intervals"]["cost_regret_median"]["ci_high"] < 0
    campaigns, records = sealed_family(blocks={"max_block_share": 0.03})
    share = stats(campaigns, protocol_records=records)
    assert share["blocks"]["adequacy"]["meets_max_block_share"] is False and share["quality"]["p_values"] is None


@pytest.mark.parametrize(
    "given",
    [
        None,  # the proposed criterion, status proposed_not_sealed
        criterion(status="draft, not sealed"),
        criterion(),  # a status string alone seals nothing: no sealed record binds it
    ],
)
def test_without_sealed_records_three_seeds_give_no_p_value_holm_or_decision(given):
    """A proposed, draft or self-declared criterion, and resampling seeds bound to no sealed record, never give the
    confirmatory family (findings on unsealed confirmatory output)."""
    for aggregates in (world(seeds=(1234, 2026, 3407)), per_seed_campaigns(world(seeds=(1234, 2026, 3407)))):
        result = stats(aggregates, criterion=given)
        quality = result["quality"]
        assert result["status"] == "descriptive" and quality["mode"] == "descriptive_unsealed"
        assert quality["p_values"] is None and quality["holm"] is None and quality["decision"] is None
        assert "paired_test" not in quality and "votes" not in quality
        assert quality["intervals"]["cost_regret_median"]["ci_high"] < 0  # intervals stay, descriptive
        assert result["sealing"]["confirmatory_allowed"] is False
        assert "no protocol record is given" in result["sealing"]["reasons"][0]
        assert result["blocks"]["criterion_sealed"] is False
    proposed = stats(world(seeds=(1234, 2026, 3407)))["blocks"]
    assert proposed["criterion_status"] == "proposed_not_sealed" and proposed["criterion"]["unsealed_markers"]


def _drop_record(family, seed):
    campaigns, records = family
    records.pop(next(a["protocol_record_sha256"] for a in campaigns if a["seeds"] == [seed]))
    return campaigns, records


def _unname(family, seed):
    campaigns, records = family
    named = next(a["protocol_record_sha256"] for a in campaigns if a["seeds"] == [seed])
    for aggregate in campaigns:
        if aggregate["seeds"] == [seed]:
            aggregate["protocol_record_sha256"] = None
    records.pop(named)
    return campaigns, records


def _changed_after_approval(family):
    campaigns, records = family
    next(iter(records.values()))["record_id"] = "changed after the approval"
    return campaigns, records


def _decision_status(name, status):
    return lambda record: record["protocol"]["decisions"][name].update(status=status)


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: sealed_family(mutate=lambda r: r.update(status="draft")), "'draft', not sealed"),
        (
            lambda: sealed_family(mutate=_decision_status("confirmatory_analysis", "proposed_not_sealed")),
            r"decisions not sealed: \['confirmatory_analysis'\]",
        ),
        (
            lambda: sealed_family(mutate=_decision_status("block_analysis", "required")),
            r"decisions not sealed: \['block_analysis'\]",
        ),
        (
            lambda: sealed_family(
                mutate=lambda r: r["protocol"]["decisions"]["cold_start_policy"].update(value="draft wording")
            ),
            r"decisions not sealed: \['cold_start_policy'\]",
        ),
        (lambda: sealed_family(approve=False), "no approval of its exact content"),
        (lambda: _changed_after_approval(sealed_family()), "no approval of its exact content"),
        (lambda: sealed_family(split="validation"), "admits the test split"),
        (lambda: sealed_family(mutate=lambda r: r["harness"].update(commit="0" * 40)), "another harness"),
        (
            lambda: sealed_family(mutate=lambda r: r["protocol"]["decisions"]["seeds"].update(value=[9999])),
            "does not seal training seed",
        ),
        (lambda: sealed_family(analysis={"alpha": 2.0}), "confirmatory_analysis: alpha"),
        (lambda: sealed_family(analysis={"a": "reg", "b": "seg"}), "confirmatory_analysis: the family"),
        (lambda: sealed_family(blocks={"min_blocks": 1}), "block_analysis: min_blocks"),
        (lambda: sealed_family(analysis={"bootstrap_seed": 99}), r"confirmatory_analysis in \['bootstrap_seed'\]"),
        (lambda: sealed_family(analysis={"permutation_seed": 99}), r"in \['permutation_seed'\]"),
        (lambda: sealed_family(analysis={"latency_bootstrap_seed": 99}), r"in \['latency_bootstrap_seed'\]"),
        (lambda: sealed_family(analysis={"draws": 10000}), r"in \['draws'\]"),
        (lambda: sealed_family(analysis={"permutations": 10000}), r"in \['permutations'\]"),
        (lambda: sealed_family(analysis={"alpha": 0.01}), r"in \['alpha'\]"),
        (lambda: sealed_family(analysis={"target": "thor"}), r"in \['target'\]"),  # edge pairs stay descriptive
        (lambda: sealed_family(analysis={"seeds": [1234, 2026]}), r"in \['seeds'\]"),
        (
            lambda: sealed_family(per_seed={3407: {"blocks": {"max_block_share": 0.2}}}),
            "seal different block or confirmatory analyses",
        ),
        (
            lambda: sealed_family(per_seed={1234: {"analysis": {"bootstrap_seed": 7}}}),
            "seal different block or confirmatory analyses",
        ),
        (lambda: _drop_record(sealed_family(), 3407), "seed 3407: its protocol record .* is not given"),
        (lambda: _unname(sealed_family(), 2026), "seed 2026: the aggregates name no protocol record"),
    ],
)
def test_three_seeds_without_every_sealed_binding_stay_descriptive(build, match):
    campaigns, records = build()
    result = stats(campaigns, protocol_records=records)
    quality = result["quality"]
    assert result["status"] == "descriptive" and quality["mode"] == "descriptive_unsealed"
    assert quality["p_values"] is None and quality["holm"] is None and quality["decision"] is None
    assert result["sealing"]["confirmatory_allowed"] is False
    assert any(re.search(match, reason) for reason in result["sealing"]["reasons"]), result["sealing"]


def test_a_criterion_beside_sealed_records_must_be_sealed_and_equal_to_their_block_analysis():
    campaigns, records = sealed_family()
    same = stats(campaigns, protocol_records=records, criterion=criterion())  # equal limits, no unsealed marker
    assert same["status"] == "confirmatory"
    with pytest.raises(S.StatisticsInputError, match=r"differs from the sealed block_analysis in \['min_blocks'\]"):
        stats(campaigns, protocol_records=records, criterion=criterion(min_blocks=8))
    with pytest.raises(S.StatisticsInputError, match="is not sealed"):
        stats(campaigns, protocol_records=records, criterion=PROPOSED_CRITERION)
    with pytest.raises(S.StatisticsInputError, match="that no aggregate names"):
        stats(campaigns, protocol_records={**records, "e" * 64: next(iter(records.values()))})


def test_the_2026_pass_takes_the_sealed_criterion_and_stays_descriptive():
    seg, reg = world()
    campaigns, records = sealed_family([seg, reg])
    result = stats(campaigns, protocol_records=records)
    assert result["quality"]["mode"] == "descriptive_single_seed" and result["quality"]["p_values"] is None
    assert result["blocks"]["criterion"]["source"] == "block_analysis of the sealed protocol records"
    assert result["sealing"]["confirmatory_allowed"] is False
    assert any("in ['seeds']" in reason for reason in result["sealing"]["reasons"])


def _native_descriptive_record(record):
    record["protocol"]["profile"] = "descriptive_native_2026"
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"] = {"mode": "descriptive_only"}
    record["evidence"]["references"]["historical_roster"] = None
    for candidate in record["candidates"].values():
        candidate["targets"] = {}


def test_native_descriptive_profile_reports_points_and_session_latency_without_geographic_intervals():
    # Adequate synthetic blocks still cannot turn the explicitly descriptive campaign into inference.
    campaigns, records = sealed_family(world(), mutate=_native_descriptive_record)
    result = stats(campaigns, protocol_records=records)
    assert result["status"] == "descriptive"
    assert result["quality"]["mode"] == "descriptive_native_2026"
    assert set(result["quality"]["point_estimates"]["2026"]) == set(SIX)
    assert result["quality"]["intervals"] is None
    assert result["quality"]["p_values"] is None and result["quality"]["holm"] is None
    assert result["quality"]["decision"] is None
    assert result["sealing"]["confirmatory_allowed"] is False
    assert result["latency"]


def test_native_descriptive_profile_cannot_be_upgraded_by_passing_three_seed_aggregates():
    campaigns, records = sealed_family(mutate=_native_descriptive_record)
    with pytest.raises(S.StatisticsInputError, match="2026"):
        stats(campaigns, protocol_records=records)


def test_native_descriptive_profile_computes_no_geographic_sensitivity_interval():
    campaigns, records = sealed_family(
        _with_unplaced(world(n_blocks=32, per_block=10), 1), mutate=_native_descriptive_record
    )
    result = stats(campaigns, protocol_records=records)
    assert result["blocks"]["adequacy"]["unplaced_within_cap"] is True
    assert result["quality"]["intervals"] is None
    assert result["quality_sensitivity_without_unplaced"] is None


@pytest.mark.parametrize("given", ["missing", "wrong_hash", "wrong_profile", "wrong_harness", "one_marker_missing"])
def test_marked_native_aggregates_require_their_matching_descriptive_protocol_records(given):
    campaigns, records = sealed_family(world(), mutate=_native_descriptive_record)
    for aggregate in campaigns:
        aggregate["qualification_profile"] = "descriptive_native_2026"
    if given == "missing":
        records = None
    elif given == "wrong_hash":
        records = {"e" * 64: next(iter(records.values()))}
    elif given == "wrong_profile":
        next(iter(records.values()))["protocol"].pop("profile")
    elif given == "wrong_harness":
        from passage_eval.protocol import admission

        record = next(iter(records.values()))
        record["harness"] = {**record["harness"], "commit": "d" * 40}
        record["approval"]["approved_record_sha256"] = admission.record_core_sha256(record)
    else:
        campaigns[0].pop("qualification_profile")
    with pytest.raises(S.StatisticsInputError, match="profile|record"):
        stats(campaigns, protocol_records=records)


def test_marked_native_aggregates_with_matching_records_keep_the_descriptive_scope():
    campaigns, records = sealed_family(world(), mutate=_native_descriptive_record)
    for aggregate in campaigns:
        aggregate["qualification_profile"] = "descriptive_native_2026"
    result = stats(campaigns, protocol_records=records)
    assert result["quality"]["intervals"] is None and result["quality"]["p_values"] is None
    from passage_eval.io import pinned

    for described, aggregate in zip(result["inputs"]["aggregates"], campaigns):
        assert described["qualification_profile"] == "descriptive_native_2026"
        assert described["aggregate_content_sha256"] == pinned.canonical_sha256(aggregate)


def test_persistent_native_latency_reports_quantiles_without_degenerate_single_session_intervals():
    aggregates = world()
    for aggregate in aggregates:
        aggregate["timing_profile"] = "native_raw_path_persistent_v1"
        entry = aggregate["per_seed"]["2026"]
        sessions = entry["latency_sessions"]
        entry["latency_sessions"] = [
            {
                "session": 0,
                **{
                    key: [value for session in sessions for value in session[key]]
                    for key in ("sample_ids", "latency_ms", "censored", "status")
                },
                "warmup_complete": None,
            }
        ]
    campaigns, records = sealed_family(aggregates, mutate=_native_descriptive_record)
    result = stats(campaigns, protocol_records=records)
    latency = result["latency"]["2026"]
    assert latency["uncertainty"] == "unavailable: one persistent process per candidate; no session bootstrap"
    assert latency["contrast_a_minus_b"] is None
    for procedure in ("seg", "reg"):
        assert latency["procedures"][procedure]["n_sessions"] == 1
        for quantile in latency["procedures"][procedure]["quantiles"].values():
            assert quantile["ci_low"] is None and quantile["ci_high"] is None


def _rule8_native_record(record):
    record["protocol"]["profile"] = "rule8_native_persistent"
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"]["draws"] = 10000
    record["evidence"]["replica_catalog"] = {"path": "replicas.json", "sha256": "a" * 64}
    record["evidence"]["references"]["historical_roster"] = None
    for candidate in record["candidates"].values():
        candidate["targets"] = {}


@pytest.mark.parametrize("n_blocks,expected", [(32, "confirmatory"), (18, "descriptive_blocks_inadequate")])
def test_rule8_direct_profile_uses_three_seed_quality_cis_without_inventing_latency_sessions(n_blocks, expected):
    aggregates = world(n_blocks=n_blocks, seeds=(1234, 2026, 3407))
    for aggregate in aggregates:
        aggregate["timing_profile"] = "native_raw_path_persistent_v1"
        aggregate["qualification_profile"] = "rule8_native_persistent"
        for entry in aggregate["per_seed"].values():
            sessions = entry["latency_sessions"]
            entry["latency_sessions"] = [
                {
                    "session": 0,
                    "warmup_complete": None,
                    **{
                        key: [value for session in sessions for value in session[key]]
                        for key in ("sample_ids", "latency_ms", "censored", "status")
                    },
                }
            ]
    campaigns, records = sealed_family(aggregates, mutate=_rule8_native_record)
    result = stats(campaigns, protocol_records=records, draws=10000)
    assert result["quality"]["mode"] == expected
    assert result["sealing"]["protocol_records"] == {
        str(aggregate["seeds"][0]): aggregate["protocol_record_sha256"]
        for aggregate in campaigns
        if aggregate["procedure"] == "seg"
    }
    assert result["completeness"]["seeds"] == [1234, 2026, 3407]
    assert result["quality"]["intervals"]["cost_regret_median"]["n_bootstrap"] == 10000
    for seed in (1234, 2026, 3407):
        assert result["latency"][str(seed)]["contrast_a_minus_b"] is None
        assert result["latency"][str(seed)]["procedures"]["seg"]["quantiles"]["p50"]["ci_low"] is None
    with pytest.raises(S.StatisticsInputError, match="matching protocol records"):
        stats(campaigns)


def test_rule8_three_seed_analysis_refuses_a_substituted_replica_commitment():
    def substitute(record):
        _rule8_native_record(record)
        if record["protocol"]["decisions"]["seeds"]["value"] == [1234]:
            record["evidence"]["replica_catalog"]["sha256"] = "b" * 64

    campaigns, records = sealed_family(mutate=substitute)
    result = stats(campaigns, protocol_records=records, draws=10000)
    assert result["sealing"]["confirmatory_allowed"] is False
    assert "different replica catalogue digests" in "; ".join(result["sealing"]["reasons"])
    assert result["quality"]["p_values"] is None and result["quality"]["holm"] is None
    assert result["quality"]["mode"] == "descriptive_unsealed"


@pytest.mark.parametrize("invalid", [None, "journal", "replay", "warmup", "prefix"])
def test_documented_native_resumes_are_descriptively_aggregated_without_replay_or_bootstrap(invalid):
    from passage_eval.io import pinned

    aggregates = world()
    for aggregate in aggregates:
        aggregate["timing_profile"] = "native_raw_path_persistent_v1"
        entry = aggregate["per_seed"]["2026"]
        previous = entry["latency_sessions"]
        sessions = []
        for index, fragments in enumerate((previous[:8], previous[8:])):
            session = {
                "session": index,
                "warmup_complete": None,
                **{
                    key: [value for fragment in fragments for value in fragment[key]]
                    for key in ("sample_ids", "latency_ms", "censored", "status")
                },
                "native_execution": {
                    "session_id": f"native-{index}",
                    "n_processes": 1,
                    "n_restarts": 0,
                    "warmup": {
                        "k": 1,
                        "repeats": 16,
                        "performed": True,
                        "disjoint_by_content": True,
                        "input_roster_sha256": "a" * 64,
                    },
                },
            }
            sessions.append(session)
        resume = {
            "previous_session_id": "native-0",
            "session_id": "native-1",
            "accepted_prefix_sha256": pinned.canonical_sha256(sessions[:1]),
            "journal_sha256": "b" * 64,
            "reason": "new process after the prior lease window ended; completed requests retained",
        }
        entry["latency_sessions"] = sessions
        entry["native_resumes"] = [resume]
        if aggregate["procedure"] == "seg":
            if invalid == "journal":
                resume.pop("journal_sha256")
            elif invalid == "replay":
                sessions[1]["sample_ids"][0] = sessions[0]["sample_ids"][-1]
            elif invalid == "warmup":
                sessions[1]["native_execution"]["warmup"]["performed"] = False
            elif invalid == "prefix":
                resume["accepted_prefix_sha256"] = "c" * 64
    campaigns, records = sealed_family(aggregates, mutate=_native_descriptive_record)
    if invalid:
        with pytest.raises(S.StatisticsInputError, match="native|warm-up"):
            stats(campaigns, protocol_records=records)
    else:
        result = stats(campaigns, protocol_records=records)
        assert result["latency"]["2026"]["procedures"]["seg"]["n_sessions"] == 2
        assert result["latency"]["2026"]["contrast_a_minus_b"] is None


def test_later_seed_is_descriptive_under_its_own_record():
    def later(record):
        _native_descriptive_record(record)
        record["protocol"]["profile"] = "descriptive_native_single_seed"

    campaigns, records = sealed_family(world(seeds=(1234,)), mutate=later)
    for aggregate in campaigns:
        aggregate["qualification_profile"] = "descriptive_native_single_seed"
    result = stats(campaigns, protocol_records=records)
    assert result["status"] == "descriptive"
    assert result["quality"]["mode"] == "descriptive_native_single_seed"
    assert result["quality"]["p_values"] is None


def per_seed_campaigns(aggregates):
    """One single-seed campaign per seed, as ``campaign.plan`` forces (``seeds == [checkpoint seed]``): distinct
    campaign identifiers, plans, protocol records and harness commits, one harness source."""
    split = []
    for aggregate in aggregates:
        for seed in aggregate["seeds"]:
            single = copy.deepcopy(aggregate)
            single.update(
                seeds=[seed],
                per_seed={str(seed): copy.deepcopy(aggregate["per_seed"][str(seed)])},
                campaign_id=f"campaign-{seed}",
                plan_sha256=f"{seed:064d}",
                protocol_record_sha256=f"{seed:064x}",
                harness=dict(aggregate["harness"], commit=f"{seed:040x}"),
            )
            split.append(single)
    return split


def test_one_campaign_per_seed_gives_the_same_statistics_as_one_three_seed_aggregate():
    three = world(seeds=(1234, 2026, 3407))
    campaigns = per_seed_campaigns(three)
    assert len(campaigns) == 6 and {a["campaign_id"] for a in campaigns} == {
        f"campaign-{s}" for s in (1234, 2026, 3407)
    }
    combined, separate = stats(three, criterion=criterion()), stats(campaigns[::-1], criterion=criterion())
    for key in ("quality", "latency", "p90_penalty_dominated", "blocks"):
        assert separate[key] == combined[key]
    assert separate["completeness"]["seeds"] == [1234, 2026, 3407]
    inputs = separate["inputs"]
    assert len(inputs["aggregates"]) == 6 and inputs["common"]["target"] == "host"
    assert inputs["per_seed"]["1234"]["seg"]["campaign_id"] == "campaign-1234"
    assert inputs["per_seed"]["3407"]["reg"]["plan_sha256"] == f"{3407:064d}"
    from passage_eval.io import pinned

    first = inputs["aggregates"][0]
    seed = str(first["seeds"][0])
    source = next(a for a in campaigns[::-1] if a["procedure"] == first["procedure"] and str(a["seeds"][0]) == seed)
    assert first["per_seed_sha256"] == {seed: pinned.canonical_sha256(source["per_seed"][seed])}


@pytest.mark.parametrize(
    "change, match",
    [
        # seed 2026 given twice for seg (a second campaign of the same seed)
        (lambda c: c.append(dict(copy.deepcopy(c[1]), campaign_id="again")), "two aggregates of 'seg' for seed 2026"),
        # seed 3407 without its reg aggregate
        (lambda c: c.pop(), "every seed needs both procedures"),
        # the reg aggregate of seed 1234 comes from another campaign than its seg aggregate
        (lambda c: c[3].update(plan_sha256="f" * 64), "seed 1234: .* not one campaign"),
        (lambda c: c[3].update(protocol_record_sha256="f" * 64), "seed 1234: .* not one campaign"),
        # another admissible roster, admission chain, target or harness source for one seed
        (lambda c: [a.update(admissible_roster_sha256="f" * 64) for a in c[2::3]], "one population, target"),
        (lambda c: [a.update(reference_admission_chain_sha256="f" * 64) for a in c[2::3]], "one population"),
        (lambda c: [a.update(target="orin") for a in c[2::3]], "one population, target"),
        (lambda c: [a["harness"].update(source_sha256="f" * 64) for a in c[2::3]], "harness.source_sha256"),
    ],
)
def test_per_seed_campaigns_must_measure_one_population_and_give_each_seed_once(change, match):
    campaigns = per_seed_campaigns(world(seeds=(1234, 2026, 3407)))
    assert [(a["procedure"], a["seeds"][0]) for a in campaigns] == [
        ("seg", 1234),
        ("seg", 2026),
        ("seg", 3407),
        ("reg", 1234),
        ("reg", 2026),
        ("reg", 3407),
    ]
    change(campaigns)
    with pytest.raises(S.StatisticsInputError, match=match):
        stats(campaigns, criterion=criterion())


def test_another_seed_set_is_reported_seed_by_seed():
    result = stats(world(seeds=(1234, 2026)), criterion=criterion())
    assert result["quality"]["mode"] == "descriptive_per_seed" and set(result["quality"]["intervals"]) == {
        "1234",
        "2026",
    }
    assert result["quality"]["p_values"] is None


# ------------------------------------------------------------------------------------------------- unplaced


def _with_unplaced(aggregates, count):
    changed = copy.deepcopy(aggregates)
    for aggregate in changed:
        for entry in aggregate["per_seed"].values():
            tasks = entry["geographic_blocks"]["tasks"]
            for task in list(tasks)[:count]:
                tasks[task] = "unplaced"
    return changed


def test_unplaced_tasks_stay_in_the_estimate_with_a_sensitivity_and_beyond_the_cap_stop_the_block_analysis():
    within = stats(_with_unplaced(world(), 1), criterion=criterion(max_unplaced_fraction=0.02))
    assert within["blocks"]["adequacy"]["n_unplaced"] == 1 and within["quality"]["intervals"]["2026"]
    assert within["quality"]["intervals"]["2026"]["cost_regret_median"]["n_tasks"] == 64
    sensitivity = within["quality_sensitivity_without_unplaced"]
    assert sensitivity["n_tasks"] == 63 and sensitivity["label"].startswith("sensitivity without the unplaced")
    halted = stats(_with_unplaced(world(), 1), criterion=criterion())  # 1/64 > 0.5 %
    assert halted["blocks"]["status"] == "halted_unplaced_cap" and halted["quality"]["mode"] == "block_analysis_halted"
    assert halted["quality"]["intervals"] is None and halted["quality_sensitivity_without_unplaced"] is None
    assert stats(world(), criterion=criterion())["quality_sensitivity_without_unplaced"] is None


def test_a_halted_block_analysis_still_reports_the_descriptive_point_estimates_and_latency():
    """Judge R5 stops the block analysis only: the regret and latency of the whole population stay reported."""
    seg, reg = _with_unplaced(world(), 1)
    halted = stats([seg, reg], criterion=criterion())
    quality = halted["quality"]
    assert quality["mode"] == "block_analysis_halted" and halted["status"] == "descriptive"
    assert quality["p_values"] is None and quality["holm"] is None and quality["decision"] is None
    points = quality["point_estimates"]["2026"]
    assert list(points) == SIX
    stored_a, stored_b = seg["per_seed"]["2026"]["metrics"], reg["per_seed"]["2026"]["metrics"]
    for name in SIX:
        assert points[name]["a"] == stored_a[name] and points[name]["b"] == stored_b[name]
        assert points[name]["delta"] == pytest.approx(stored_a[name] - stored_b[name])
    assert points["cost_regret_median"]["delta"] < 0 and points["cost_regret_median"]["n_tasks"] == 64
    p50 = halted["latency"]["2026"]["procedures"]["seg"]["quantiles"]["p50"]
    assert p50["ci_low"]["value_ms"] < p50["ci_high"]["value_ms"]  # the session bootstrap is not a block analysis
    assert halted["latency"]["2026"]["contrast_a_minus_b"]["p50"]["ci_low_ms"] is not None
    # the same point estimates are reported in every mode
    assert stats(world(), criterion=criterion())["quality"]["point_estimates"]["2026"]["success_rate"]["delta"] == 0


# --------------------------------------------------------------------------------------------- p90 penalty


def test_p90_is_flagged_as_penalty_dominated_from_ten_percent_failures():
    result = stats(world(n_blocks=20, per_block=2, failures=3, reg_failures=4), criterion=criterion(min_blocks=2))
    flags = result["p90_penalty_dominated"]
    assert flags["seg"]["2026"] == {"n_failed": 3, "failure_rate": 3 / 40, "p90_penalty_dominated": False}
    assert flags["reg"]["2026"] == {"n_failed": 4, "failure_rate": 0.1, "p90_penalty_dominated": True}


# ------------------------------------------------------------------------------------------------- latency


def test_latency_intervals_come_from_a_bootstrap_of_whole_sessions():
    seg, reg = world()
    result = stats([seg, reg], criterion=criterion())["latency"]["2026"]
    assert result["sessions_paired"] is True
    block = seg["per_seed"]["2026"]["latency"]["quantiles"]
    for label in ("p50", "p95", "p99"):
        entry = result["procedures"]["seg"]["quantiles"][label]
        assert entry["estimate"] == {"value_ms": block[label]["value_ms"], "censored": False}
        assert entry["ci_low"]["value_ms"] <= entry["estimate"]["value_ms"] <= entry["ci_high"]["value_ms"]
    assert result["procedures"]["seg"]["n_sessions"] == 16 and result["procedures"]["seg"]["n_requests"] == 64
    p50 = result["procedures"]["seg"]["quantiles"]["p50"]
    assert p50["ci_low"]["value_ms"] < p50["ci_high"]["value_ms"]  # the sessions are resampled
    contrast = result["contrast_a_minus_b"]["p50"]
    assert (
        contrast["n_undefined_draws"] == 0
        and contrast["ci_low_ms"] <= contrast["estimate_ms"] <= contrast["ci_high_ms"]
    )
    # one session: every session draw is that session, so the interval collapses (a request bootstrap would not)
    single = stats(list(world(n_blocks=2, per_block=4, size=8)), criterion=criterion(min_blocks=2, max_block_share=0.5))
    one = single["latency"]["2026"]["procedures"]["seg"]["quantiles"]["p50"]
    assert single["latency"]["2026"]["procedures"]["seg"]["n_sessions"] == 1
    assert one["ci_low"] == one["estimate"] == one["ci_high"]


def test_censored_draws_make_the_paired_contrast_unavailable():
    seg, reg = world(reg_timeouts=("t000", "t001", "t002"))  # the first session of reg is three timeouts
    result = stats([seg, reg], criterion=criterion())["latency"]["2026"]
    assert result["procedures"]["reg"]["n_censored"] == 3
    p99 = result["contrast_a_minus_b"]["p99"]
    assert p99["n_undefined_draws"] > 0 and p99["ci_low_ms"] is None and p99["ci_high_ms"] is None
    high = result["procedures"]["reg"]["quantiles"]["p99"]["ci_high"]
    assert high == {"value_ms": None, "censored": True}


def test_procedures_that_do_not_share_sessions_get_no_paired_contrast():
    seg, reg = world(size=4, reg_size=8)
    result = stats([seg, reg], criterion=criterion())["latency"]["2026"]
    assert result["sessions_paired"] is False and result["contrast_a_minus_b"] is None
    assert "do not share the same sessions" in result["contrast_unavailable_reason"]


# -------------------------------------------------------------------------------------------- input checks


def _edit(aggregates, procedure, change):
    changed = copy.deepcopy(aggregates)
    target = next(a for a in changed if a["procedure"] == procedure)
    change(target)
    return changed


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda a: a.update(plan_sha256="f" * 64), "not one campaign"),
        (lambda a: a.update(target="orin"), "one population, target and protocol"),
        (lambda a: a.update(campaign_id="another"), "not one campaign"),
        (lambda a: a["per_seed"]["2026"]["task_rows"]["regret"].__setitem__(0, 0.5), "recomputed from the rows"),
        (lambda a: a["per_seed"]["2026"]["task_rows"]["task_id"].pop(), "exactly one accepted record"),
        (lambda a: a["per_seed"]["2026"]["completeness"].update(complete=False), "not complete"),
        (lambda a: a["per_seed"]["2026"]["geographic_blocks"]["tasks"].update(t000="geo5:99:00"), "blocks differ"),
        (lambda a: a["per_seed"]["2026"]["latency"]["quantiles"]["p50"].update(value_ms=-1.0), "reproduce"),
        (lambda a: a["per_seed"]["2026"].pop("latency_sessions"), "no latency sessions"),
        (lambda a: a["per_seed"]["2026"].pop("task_rows"), "no task rows"),
        (lambda a: a["per_seed"].update({"1234": a["per_seed"]["2026"]}), "exactly one aggregate per seed"),
    ],
)
def test_inputs_that_break_completeness_identity_or_integrity_are_refused(change, match):
    with pytest.raises(S.StatisticsInputError, match=match):
        stats(_edit(world(), "reg", change), criterion=criterion())


def test_exactly_one_aggregate_of_seg_and_of_reg_is_required():
    seg, reg = world()
    with pytest.raises(S.StatisticsInputError, match="two aggregates of 'seg'"):
        stats([seg, seg, reg])
    with pytest.raises(S.StatisticsInputError, match="exactly 'seg' and 'reg'"):
        stats([seg, dict(reg, procedure="astar")])
    with pytest.raises(S.StatisticsInputError, match="aggregate is required"):
        stats([seg, dict(reg, schema="other")])


def test_the_command_reads_pinned_aggregates(tmp_path, capsys):
    from passage_eval.campaign.__main__ import main
    from passage_eval.io import pinned

    paths, shas = [], []
    for aggregate in world():
        path = tmp_path / f"{aggregate['procedure']}.json"
        path.write_text(json.dumps(aggregate))
        paths.append(path)
        shas.append(pinned.sha256_bytes(path.read_bytes()))
    criterion_path = tmp_path / "criterion.json"
    criterion_path.write_text(json.dumps(criterion()))
    args = ["statistics"]
    for path, sha in zip(paths, shas):
        args += ["--aggregate", str(path), "--aggregate-sha256", sha]
    args += ["--bootstrap-seed", "1", "--permutation-seed", "2", "--latency-bootstrap-seed", "3", "--draws", "50"]
    args += ["--block-criterion", str(criterion_path), "--output", str(tmp_path / "stats.json")]
    assert main(args) == 0
    written = json.loads((tmp_path / "stats.json").read_text())
    assert [entry["file_sha256"] for entry in written["inputs"]["aggregates"]] == shas
    assert [entry["procedure"] for entry in written["inputs"]["aggregates"]] == ["seg", "reg"]
    assert written["blocks"]["criterion_status"] == "sealed for this test" and written["resampling"]["draws"] == 50
    wrong = list(args)
    wrong[wrong.index(shas[0])] = "0" * 64
    wrong[wrong.index("--output") + 1] = str(tmp_path / "other.json")
    with pytest.raises(Exception, match="SHA-256|sha256|digest"):
        main(wrong)
    with pytest.raises(SystemExit):
        main([*args[:1], "--aggregate", str(paths[0]), "--aggregate-sha256", shas[0]])


def test_the_command_reads_aggregates_written_by_the_aggregate_command_in_any_parent_order(tmp_path):
    """``campaign aggregate --output`` writes with sorted keys: the block map then comes back in id order, while the
    task rows keep the parent order (ascending salted selection key). The order must come from the rows."""
    from passage_eval.campaign.__main__ import main
    from passage_eval.io import pinned

    seg, reg = world(reverse=True)
    parent_order = seg["per_seed"]["2026"]["task_rows"]["task_id"]
    assert parent_order != sorted(parent_order)
    args = ["statistics"]
    for aggregate in (seg, reg):
        path = tmp_path / f"{aggregate['procedure']}.json"
        args += ["--aggregate", str(path), "--aggregate-sha256", pinned.write_json_exclusive(path, aggregate)]
        written = json.loads(path.read_text())["per_seed"]["2026"]
        assert list(written["geographic_blocks"]["tasks"]) == sorted(parent_order)  # as the aggregator writes it
        assert written["task_rows"]["task_id"] == parent_order
    criterion_path = tmp_path / "criterion.json"
    criterion_path.write_text(json.dumps(criterion()))
    args += ["--bootstrap-seed", "1", "--permutation-seed", "2", "--latency-bootstrap-seed", "3", "--draws", "50"]
    args += ["--block-criterion", str(criterion_path), "--output", str(tmp_path / "stats.json")]
    assert main(args) == 0
    result = json.loads((tmp_path / "stats.json").read_text())
    assert result["quality"]["mode"] == "descriptive_single_seed"
    intervals = result["quality"]["intervals"]["2026"]
    assert intervals["cost_regret_median"]["n_blocks"] == 32 and intervals["cost_regret_median"]["n_tasks"] == 64
    # the blocks are bound to the tasks by id, not by position: the record equals the in-memory one
    assert result["quality"] == stats([seg, reg], criterion=criterion(), draws=50, bootstrap_seed=1)["quality"]


def test_the_command_binds_pinned_protocol_records(tmp_path):
    from passage_eval.campaign.__main__ import main
    from passage_eval.io import pinned

    campaigns, records = sealed_family()
    file_digest, record_args = {}, []
    for number, (digest, record) in enumerate(records.items()):
        path = tmp_path / f"record-{number}.json"
        file_digest[digest] = pinned.write_json_exclusive(path, record)
        record_args += ["--protocol-record", str(path), "--protocol-record-sha256", file_digest[digest]]
    args = ["statistics"]
    for number, aggregate in enumerate(campaigns):
        aggregate["protocol_record_sha256"] = file_digest[aggregate["protocol_record_sha256"]]  # the plan's pin
        path = tmp_path / f"aggregate-{number}.json"
        args += ["--aggregate", str(path), "--aggregate-sha256", pinned.write_json_exclusive(path, aggregate)]
    args += ["--bootstrap-seed", "11", "--permutation-seed", "12", "--latency-bootstrap-seed", "13"]
    args += ["--draws", "200", "--permutations", "200"]
    assert main([*args, *record_args, "--output", str(tmp_path / "confirmatory.json")]) == 0
    written = json.loads((tmp_path / "confirmatory.json").read_text())
    assert written["status"] == "confirmatory" and written["quality"]["decision"] == "partial_evidence"
    assert written["sealing"]["protocol_records"] == {
        str(a["seeds"][0]): a["protocol_record_sha256"] for a in campaigns if a["procedure"] == "seg"
    }
    assert main([*args, "--output", str(tmp_path / "unsealed.json")]) == 0  # the same aggregates without records
    unsealed = json.loads((tmp_path / "unsealed.json").read_text())
    assert unsealed["status"] == "descriptive" and unsealed["quality"]["mode"] == "descriptive_unsealed"
    assert unsealed["quality"]["p_values"] is None and unsealed["quality"]["decision"] is None
    assert main([*args, *record_args, "--alpha", "0.01", "--output", str(tmp_path / "alpha.json")]) == 0
    assert json.loads((tmp_path / "alpha.json").read_text())["quality"]["p_values"] is None  # not the sealed alpha
    wrong = list(record_args)
    wrong[3] = "0" * 64
    with pytest.raises(Exception, match="SHA-256|sha256|digest"):
        main([*args, *wrong, "--output", str(tmp_path / "other.json")])
    with pytest.raises(SystemExit):
        main([*args, *record_args[:2], "--output", str(tmp_path / "other2.json")])


def test_task_rows_in_different_orders_or_a_block_map_of_other_tasks_are_refused():
    seg, reg = world()
    shuffled = copy.deepcopy(reg)
    rows = shuffled["per_seed"]["2026"]["task_rows"]
    order = list(range(len(rows["task_id"])))
    order[0], order[1] = 1, 0
    for key, values in rows.items():
        rows[key] = [values[i] for i in order]
    with pytest.raises(S.StatisticsInputError, match="one task order"):
        stats([seg, shuffled], criterion=criterion())
    renamed = copy.deepcopy([seg, reg])
    for aggregate in renamed:
        tasks = aggregate["per_seed"]["2026"]["geographic_blocks"]["tasks"]
        tasks["other"] = tasks.pop("t000")
    with pytest.raises(S.StatisticsInputError, match="name exactly the admissible tasks"):
        stats(renamed, criterion=criterion())


# --------------------------------------------------------------------------------- the aggregator's own rows


def test_real_aggregates_carry_the_rows_and_sessions_that_the_statistics_read(tmp_path, metre_scene):
    from passage_eval.campaign.aggregate import aggregate_scored_batches

    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "aggregate_tests", Path(__file__).parent / "test_campaign_aggregate.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scenes = {sample_id: replace(metre_scene, sample_id=sample_id) for sample_id in module.PLAN}
    produced = []
    for procedure in ("seg", "reg"):
        batches = [
            (module.scored_run(tmp_path / procedure / f"batch-{i}", [scenes[s] for s in ids])[0], ids)
            for i, ids in enumerate(module.BATCHES)
        ]
        result = aggregate_scored_batches(
            batches, parent_order=list(module.PLAN), parent_blocks=module.BLOCKS, population=module.POPULATION
        )
        assert result["completeness"] == {
            "rule": "exactly one accepted record per admissible task, procedure, target and seed",
            "n_admissible": 6,
            "n_records": 6,
            "complete": True,
        }
        assert result["task_rows"]["task_id"] == list(module.PLAN) and len(result["latency_sessions"]) == 3
        assert result["geographic_blocks"]["n_unplaced"] == 0
        common = dict(
            schema=AGGREGATE_SCHEMA,
            procedure=procedure,
            seeds=[2026],
            per_seed={"2026": result},
            **{field: "x" for field in ("campaign_id", "target", "split", "plan_sha256")},
            admissible_roster_sha256="b" * 64,
            reference_admission_chain_sha256="c" * 64,
            population=result["population"],
            budget={},
            timing_profile="t",
            harness={},
        )
        produced.append(json.loads(json.dumps(common)))
    record = stats(produced, criterion=criterion(min_blocks=2, max_block_share=0.5, max_unplaced_fraction=0.0))
    assert record["quality"]["mode"] == "descriptive_single_seed"
    assert record["quality"]["intervals"]["2026"]["cost_regret_median"]["delta"] == 0.0  # same plan for both
    assert record["latency"]["2026"]["sessions_paired"] is True


def test_a_single_block_gives_no_block_interval():
    result = stats(world(n_blocks=1, per_block=8), criterion=criterion(min_blocks=2, max_block_share=1.0))
    assert result["quality"]["mode"] == "no_block_interval" and result["quality"]["intervals"] is None


def test_the_paired_latency_contrast_uses_the_same_session_draws_for_both_procedures():
    seg, reg = world()
    twin = copy.deepcopy(reg)
    for key in ("latency_sessions", "latency"):
        twin["per_seed"]["2026"][key] = copy.deepcopy(seg["per_seed"]["2026"][key])
    contrast = stats([seg, twin], criterion=criterion())["latency"]["2026"]["contrast_a_minus_b"]
    for label in ("p50", "p95", "p99"):  # identical sessions under shared draws: every draw differs by zero
        assert contrast[label] == {"estimate_ms": 0.0, "ci_low_ms": 0.0, "ci_high_ms": 0.0, "n_undefined_draws": 0}
