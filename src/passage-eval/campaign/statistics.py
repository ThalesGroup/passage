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

"""Statistics of a campaign: block intervals, the confirmatory family and session-bootstrap latency intervals.

``campaign_statistics`` links the per-procedure campaign aggregates (``passage-campaign-aggregate/v1``, with their
``task_rows`` and ``latency_sessions``) to the protocol's statistical tools:

* **Quality, one seed** (for example the 2026 pass): per-seed geographic block-bootstrap intervals of the Seg minus
  Reg contrast of the six primary metrics (``discipline.multiseed_paired_bootstrap`` on that seed, the statistics of
  ``compare_primary_metrics``), labelled *descriptive, single seed, exposed public test, not confirmatory*. No
  p-value and no Holm decision are computed (judge R8).
* **Quality, the three declared seeds** (1234, 2026, 3407): the confirmatory family A = Seg, B = Reg, six metrics in
  the declared order, ``draws`` shared block draws (seed mean of the contrasts in every draw,
  ``discipline.compare_primary_metrics``), p-values of ``passage-paired-block-randomization/v1``
  (``block_randomization_p_values``: exact enumeration up to 16 blocks, otherwise Monte-Carlo with the identity
  added) and Holm at ``alpha``. The family is confirmatory only when (1) every seed's aggregates name a
  ``passage-test-admission/v1`` protocol record that is given, sealed in every decision and approved on its exact
  content, for the aggregates' split, harness and training seed; (2) these records seal one ``block_analysis`` (the
  block criterion used: a criterion file given beside them must carry no unsealed marker and equal it) and one
  ``confirmatory_analysis`` equal to the computed one (family, target, seeds, alpha, draws, permutations and the
  three resampling seeds); and (3) the blocks meet that sealed criterion. Otherwise the intervals are descriptive
  (modes ``descriptive_unsealed`` and ``descriptive_blocks_inadequate``) and no p-value, Holm adjustment or decision
  is computed; ``sealing.reasons`` says why. A proposed criterion (``proposed_not_sealed``) never yields a
  confirmatory record. With any other set of seeds, each seed is reported alone, descriptively.
* **Unplaced tasks.** The sealed ``unplaced`` block stays in every estimate. When it exceeds the criterion's
  ``max_unplaced_fraction``, the block analysis stops for diagnosis (judge R5): no block interval, no p-value; the
  point estimates and the latency intervals are still reported. Otherwise a descriptive sensitivity without the
  unplaced tasks is reported.
* **Point estimates.** In every mode, per seed, the six metrics of each procedure on every admissible task and their
  difference, without interval (``quality.point_estimates``).
* **Latency.** Per procedure and seed, intervals of p50, p95 and p99 under ``passage-censored-latency/v1``
  (``inverted_cdf`` order statistics, censored requests ordered at +inf) by a bootstrap of *sessions*: the batches
  are the clusters, drawn with replacement ``draws`` times with ``latency_bootstrap_seed``; requests are never
  resampled one by one. The paired Seg minus Reg contrast uses the same session draws for both procedures; a draw in
  which either quantile is censored is undefined, and any undefined draw makes the contrast interval unavailable.
* **p90 flag.** ``p90_penalty_dominated`` is set for a procedure and seed whose share of penalised failures is at
  least 10 %: the p90 regret is then the penalty, not a route quality (manuscript, median and p90 robust below 10 %).

Quantile methods are declared: ``linear`` for the quality metrics (``numpy`` default, as the scorer),
``inverted_cdf`` for latency. Bootstrap interval endpoints are the 2.5 % and 97.5 % quantiles of the draws
(``linear`` for quality as in ``discipline``, ``inverted_cdf`` for latency so that a censored draw stays an order
statistic).

Inputs are checked: exactly one Seg and one Reg aggregate per seed. A campaign plan holds the single training seed of
its frozen checkpoint and every seed has its own test pass (rule 3, judge R8), so the seeds may come from separate
campaigns; a seed is never given twice. Every aggregate measures the same target, split, admissible roster,
admission chain, population, budget, timing profile and harness source hash; the Seg and Reg aggregates of one seed
come from one campaign (campaign identifier, plan, harness commit and protocol record). Per seed, exactly one row per
admissible task in one task order for both procedures and every seed, the same blocks, and the six metrics
recomputed from the rows equal to the aggregate's. The record lists every input with its file digest and the
canonical digest of each per-seed entry. The resampling seeds have no default: the protocol author seals them
before the pass (decision ``confirmatory_analysis``).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from passage_eval.campaign import AGGREGATE_SCHEMA
from passage_eval.execution.artifacts import json_safe
from passage_eval.io import pinned
from passage_eval.metrics.latency import DEFAULT_QUANTILES, METHOD as LATENCY_METHOD, RULE_ID, _rank_index
from passage_eval.metrics.summary import aggregate_primary_metrics
from passage_eval.protocol.block_census import PROPOSED_CRITERION, adequacy, validate_criterion
from passage_eval.protocol.block_randomization import ASSUMPTIONS, block_randomization_p_values
from passage_eval.protocol.discipline import (
    DEFAULT_SEEDS,
    PRIMARY_METRICS,
    MetricFamily,
    compare_primary_metrics,
    multiseed_paired_bootstrap,
)
from passage_eval.protocol.geographic_blocks import UNPLACED_BLOCK

STATISTICS_SCHEMA = "passage-campaign-statistics/v1"
PROCEDURE_A, PROCEDURE_B = "seg", "reg"
QUALITY_QUANTILE_METHOD = "linear"
LATENCY_QUANTILE_METHOD = LATENCY_METHOD
PENALTY_DOMINATED_FAILURE_RATE = 0.10
EXACT_BLOCK_LIMIT = 16
DEFAULT_DRAWS = 10_000
SINGLE_SEED_LABEL = "descriptive, single seed, exposed public test, not confirmatory"
_ROW_KEYS = ("task_id", "regret", "assd", "h95", "success", "obstacle_violation", "valid_path", "obstacle_cells")
#: Fields that every aggregate shares, whatever its seed and campaign: one population, target and request protocol.
#: The harness source hash (``harness.source_sha256``, Python sources only) is shared too; the harness commit may
#: differ between seeds, since each seed's frozen checkpoint is declared at the commit of its own campaign.
_COMMON_FIELDS = (
    "target",
    "split",
    "admissible_roster_sha256",
    "reference_admission_chain_sha256",
    "population",
    "budget",
    "timing_profile",
)
#: Fields that the Seg and Reg aggregates of one seed share: one campaign, plan, harness and protocol record (the
#: two procedures run in the same batches, so their latency sessions pair).
_CAMPAIGN_FIELDS = ("campaign_id", "plan_sha256", "harness", "protocol_record_sha256")
_STATISTICS = {"cost_regret_median": np.median, "cost_regret_p90": lambda array: np.quantile(array, 0.9)}
_COLUMNS = {
    "cost_regret_median": "regret",
    "cost_regret_p90": "regret",
    "path_assd": "assd",
    "path_hausdorff_p95": "h95",
    "success_rate": "success",
    "obstacle_violation_rate": "obstacle_violation",
}


class StatisticsInputError(ValueError):
    """The aggregates cannot be compared: completeness, identity or integrity fails."""


def _fail(message: str) -> None:
    raise StatisticsInputError(message)


def _same(a: Any, b: Any) -> bool:
    def norm(value):
        return None if value is None or (isinstance(value, float) and math.isnan(value)) else value

    return norm(a) == norm(b)


# ------------------------------------------------------------------------------------------------------ inputs


def _source_sha256(aggregate: Mapping[str, Any]) -> Any:
    harness = aggregate.get("harness")
    return harness.get("source_sha256") if isinstance(harness, Mapping) else None


def _inputs(
    aggregates: Sequence[Mapping[str, Any]], file_sha256: Sequence[str] | None
) -> tuple[dict[tuple[str, int], dict[str, Any]], list[int], list[dict[str, Any]]]:
    """Exactly one Seg and one Reg aggregate per seed, from one or several campaigns.

    Every training seed has its own test pass (rule 3; judge R8): a campaign plan holds the single seed of its frozen
    checkpoint, so the three confirmatory seeds come from three campaigns. A multi-seed aggregate is split into its
    seeds. Returns the entries keyed by ``(procedure, seed)``, the sorted seeds and a description of every input.
    """
    if not isinstance(aggregates, Sequence) or not aggregates:
        _fail(f"{PROCEDURE_A!r} and {PROCEDURE_B!r} aggregates are required")
    if file_sha256 is not None and len(file_sha256) != len(aggregates):
        _fail("give one file digest per aggregate")
    by_key: dict[tuple[str, int], dict[str, Any]] = {}
    described = []
    for index, aggregate in enumerate(aggregates):
        if not isinstance(aggregate, Mapping) or aggregate.get("schema") != AGGREGATE_SCHEMA:
            _fail(f"a {AGGREGATE_SCHEMA} aggregate is required")
        procedure = aggregate.get("procedure")
        if procedure not in (PROCEDURE_A, PROCEDURE_B):
            _fail(f"the family compares exactly {PROCEDURE_A!r} and {PROCEDURE_B!r}, got {procedure!r}")
        seeds = aggregate.get("seeds")
        if (
            not isinstance(seeds, list)
            or not seeds
            or not all(isinstance(seed, int) and not isinstance(seed, bool) for seed in seeds)
            or len(set(seeds)) != len(seeds)
        ):
            _fail(f"{procedure}: the aggregate names no unique integer seeds")
        per_seed = aggregate.get("per_seed")
        if not isinstance(per_seed, Mapping) or sorted(per_seed) != sorted(str(seed) for seed in seeds):
            _fail(f"{procedure}: per_seed must hold exactly one aggregate per seed")
        for seed in seeds:
            if (procedure, seed) in by_key:
                _fail(f"two aggregates of {procedure!r} for seed {seed}: exactly one per procedure, target and seed")
            by_key[(procedure, seed)] = {"aggregate": aggregate, "entry": per_seed[str(seed)], "index": index}
        described.append(
            {
                "index": index,
                "procedure": procedure,
                "seeds": list(seeds),
                "file_sha256": None if file_sha256 is None else file_sha256[index],
                "per_seed_sha256": {
                    str(seed): pinned.canonical_sha256(json_safe(per_seed[str(seed)])) for seed in seeds
                },
                **{field: copy.deepcopy(aggregate.get(field)) for field in _CAMPAIGN_FIELDS},
            }
        )
        if aggregate.get("qualification_profile") is not None:
            described[-1]["qualification_profile"] = aggregate["qualification_profile"]
            described[-1]["aggregate_content_sha256"] = pinned.canonical_sha256(json_safe(aggregate))
    seeds_by_procedure = {
        name: sorted(seed for procedure, seed in by_key if procedure == name) for name in (PROCEDURE_A, PROCEDURE_B)
    }
    if not all(seeds_by_procedure.values()):
        present = sorted({procedure for procedure, _ in by_key})
        _fail(f"the family compares exactly {PROCEDURE_A!r} and {PROCEDURE_B!r}, got {present}")
    if seeds_by_procedure[PROCEDURE_A] != seeds_by_procedure[PROCEDURE_B]:
        _fail(
            f"the {PROCEDURE_A} seeds {seeds_by_procedure[PROCEDURE_A]} and the {PROCEDURE_B} seeds "
            f"{seeds_by_procedure[PROCEDURE_B]} differ: every seed needs both procedures"
        )
    first = aggregates[0]
    differing = [field for field in _COMMON_FIELDS if any(item.get(field) != first.get(field) for item in aggregates)]
    if any(_source_sha256(item) != _source_sha256(first) for item in aggregates):
        differing.append("harness.source_sha256")
    if differing:
        _fail(f"the aggregates differ in {differing}: they do not measure one population, target and protocol")
    seeds = seeds_by_procedure[PROCEDURE_A]
    for seed in seeds:
        a, b = by_key[(PROCEDURE_A, seed)]["aggregate"], by_key[(PROCEDURE_B, seed)]["aggregate"]
        differing = [field for field in _CAMPAIGN_FIELDS if a.get(field) != b.get(field)]
        if differing:
            _fail(
                f"seed {seed}: the {PROCEDURE_A} and {PROCEDURE_B} aggregates differ in {differing}: not one campaign"
            )
    return by_key, seeds, described


#: The decisions of a ``passage-test-admission/v1`` record that the statistics read: the block rule and adequacy
#: criterion, and the confirmatory family with its resampling.
SEALED_DECISIONS = ("block_analysis", "confirmatory_analysis")
_CRITERION_LIMITS = ("min_blocks", "max_block_share", "max_unplaced_fraction", "block_degrees")


def _record_problems(record: Any, seed: int, aggregate: Mapping[str, Any]) -> list[str]:
    """Why a seed's protocol record does not bind sealed decisions to its aggregates (empty: it does).

    The record must be a structurally valid, ``sealed`` record whose every decision is ``sealed`` without an unsealed
    marker, approved on its exact content, for the aggregates' split and harness, sealing this training seed and
    valid ``block_analysis`` and ``confirmatory_analysis`` decisions. The campaign plan already refused a test run
    without an all-GREEN record; this check binds the statistics to that record's decisions.
    """
    from passage_eval.protocol import admission

    try:
        record = admission.validate_record_structure(record)
    except admission.AdmissionRecordError as exc:
        return [f"the protocol record is malformed: {exc}"]
    problems = []
    if record["status"] != admission.SEALED:
        problems.append(f"the protocol record is {record['status']!r}, not sealed")
    decisions = record["protocol"]["decisions"]
    unsealed = sorted(
        name
        for name, entry in decisions.items()
        if entry["status"] != admission.SEALED or admission.unsealed_markers(entry["value"])
    )
    if unsealed:
        problems.append(f"decisions not sealed: {unsealed}")
    if admission.approval_problems(record):
        problems.append("the protocol record carries no approval of its exact content")
    if record["split"] != aggregate.get("split"):
        problems.append(
            f"the record admits the {record['split']} split; the aggregates are of {aggregate.get('split')!r}"
        )
    if record["harness"] != aggregate.get("harness"):
        problems.append("the protocol record names another harness than the aggregates")
    sealed_seeds = decisions["seeds"]["value"]
    if not isinstance(sealed_seeds, list) or seed not in sealed_seeds:
        problems.append(f"the protocol record does not seal training seed {seed}")
    for name in SEALED_DECISIONS:
        problem = admission.decision_problem(
            name,
            decisions[name]["value"],
            decisions,
            profile=record["protocol"].get("profile", admission.STRICT_PROFILE),
        )
        if problem:
            problems.append(f"{name}: {problem}")
    return problems


def _sealing(
    entries: Mapping[tuple[str, int], Mapping[str, Any]],
    seeds: Sequence[int],
    protocol_records: Mapping[str, Any] | None,
    *,
    resampling: Mapping[str, Any],
    alpha: float,
    target: Any,
) -> dict[str, Any]:
    """Whether every seed's sealed protocol record binds the block criterion, the family and the resampling.

    ``protocol_records`` maps the SHA-256 of each record file to its parsed content; each seed's aggregates name
    theirs (``protocol_record_sha256``). The confirmatory family is allowed only when every seed has a sealed,
    approved record, the records seal one ``block_analysis`` and one ``confirmatory_analysis``, and that analysis is
    the one computed here (family, target, seeds, alpha, draws, permutations and the three resampling seeds).
    """
    result: dict[str, Any] = {
        "confirmatory_allowed": False,
        "reasons": [],
        "protocol_records": {},
        "block_analysis": None,
        "confirmatory_analysis": None,
    }
    reasons = result["reasons"]
    if not protocol_records:
        reasons.append(
            "no protocol record is given: the block criterion, the family and the resampling seeds are bound to no "
            "sealed decision"
        )
        return result
    if not isinstance(protocol_records, Mapping):
        _fail("protocol_records maps the SHA-256 of each record file to its content")
    bound: dict[str, Any] = {}
    named = set()
    replica_catalog_sha256 = None
    for seed in seeds:
        aggregate = entries[(PROCEDURE_A, seed)]["aggregate"]
        digest = aggregate.get("protocol_record_sha256")
        if not pinned.is_hex64(digest):
            reasons.append(f"seed {seed}: the aggregates name no protocol record")
            continue
        named.add(digest)
        if digest not in protocol_records:
            reasons.append(f"seed {seed}: its protocol record {digest} is not given")
            continue
        record = protocol_records[digest]
        from passage_eval.protocol.admission import DESCRIPTIVE_PROFILES, RULE8_NATIVE_PROFILE

        profile = (record.get("protocol") or {}).get("profile") if isinstance(record, Mapping) else None
        if profile in DESCRIPTIVE_PROFILES:
            result["descriptive_only"] = True
            result["qualification_profile"] = profile
        elif profile == RULE8_NATIVE_PROFILE:
            evidence = record.get("evidence")
            catalogue = evidence.get("replica_catalog") if isinstance(evidence, Mapping) else None
            catalog_digest = catalogue.get("sha256") if isinstance(catalogue, Mapping) else None
            if not pinned.is_hex64(catalog_digest):
                reasons.append(f"seed {seed}: the complete replica catalogue has no file digest")
            elif replica_catalog_sha256 is None:
                replica_catalog_sha256 = catalog_digest
            elif catalog_digest != replica_catalog_sha256:
                reasons.append("the seed records seal different replica catalogue digests")
        problems = _record_problems(record, seed, aggregate)
        reasons.extend(f"seed {seed}: {problem}" for problem in problems)
        if not problems:
            decisions = record["protocol"]["decisions"]
            bound[str(seed)] = {name: copy.deepcopy(decisions[name]["value"]) for name in SEALED_DECISIONS}
            result["protocol_records"][str(seed)] = digest
    if replica_catalog_sha256 is not None:
        result["replica_catalog_sha256"] = replica_catalog_sha256
    unnamed = sorted(set(protocol_records) - named)
    if unnamed:
        _fail(f"protocol records that no aggregate names: {unnamed}")
    values = list(bound.values())
    if any(value != values[0] for value in values):
        reasons.append("the protocol records of the seeds seal different block or confirmatory analyses")
        return result
    if len(bound) != len(seeds):
        return result
    sealed = values[0]
    result["block_analysis"] = sealed["block_analysis"]
    result["confirmatory_analysis"] = analysis = sealed["confirmatory_analysis"]
    if result.get("descriptive_only"):
        reasons.append("the native 2026 profile permits descriptive point estimates and session latency only")
        return result
    computed = {"a": PROCEDURE_A, "b": PROCEDURE_B, "target": target, "seeds": sorted(seeds), "alpha": alpha}
    computed.update(resampling)
    # Both sides are validated numbers (no booleans): the sealed decision by its G1 validator, the computed values above.
    differing = sorted(
        key for key, value in computed.items() if (sorted(analysis[key]) if key == "seeds" else analysis[key]) != value
    )
    if differing:
        reasons.append(f"the statistics differ from the sealed confirmatory_analysis in {differing}")
    result["confirmatory_allowed"] = not reasons
    return result


def _criterion(criterion: Mapping[str, Any] | None, sealing: Mapping[str, Any]) -> dict[str, Any]:
    """The block criterion in use and its source: the sealed ``block_analysis`` when the records bind one."""
    from passage_eval.protocol.admission import unsealed_markers

    sealed = sealing["block_analysis"]
    if sealed is None:
        used = validate_criterion(PROPOSED_CRITERION if criterion is None else criterion)
        used["source"] = "the proposed criterion" if criterion is None else "the given criterion file"
        markers = unsealed_markers({key: value for key, value in used.items() if key != "source"})
        used["sealed"] = False
        used["unsealed_markers"] = markers
        return used
    if criterion is not None:
        given = validate_criterion(criterion)
        markers = unsealed_markers(given)
        if markers:
            _fail(f"the block criterion is not sealed ({markers}) while the protocol records seal block_analysis")
        differing = [key for key in _CRITERION_LIMITS if given[key] != sealed[key]]
        if differing:
            _fail(f"the block criterion differs from the sealed block_analysis in {differing}")
    return {
        "source": "block_analysis of the sealed protocol records",
        "status": "sealed",
        "sealed": True,
        "unsealed_markers": [],
        "rule": sealed["rule"],
        **{key: sealed[key] for key in _CRITERION_LIMITS},
    }


def _rows(
    entry: Mapping[str, Any], procedure: str, seed: str, n_admissible: int
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """The task rows of one procedure and seed as arrays, and the six metrics recomputed from them."""
    rows = entry.get("task_rows")
    if not isinstance(rows, Mapping) or not set(_ROW_KEYS) <= set(rows):
        _fail(f"{procedure} seed {seed}: the aggregate carries no task rows")
    ids = list(rows["task_id"])
    if len(ids) != n_admissible or len(set(ids)) != len(ids):
        _fail(
            f"{procedure} seed {seed}: {len(ids)} rows for {n_admissible} admissible tasks; exactly one accepted "
            "record per admissible task, procedure, target and seed is required"
        )
    completeness = entry.get("completeness") or {}
    if completeness.get("complete") is not True or completeness.get("n_records") != n_admissible:
        _fail(f"{procedure} seed {seed}: the aggregate is not complete")
    arrays = {"task_id": np.asarray(ids, dtype=object)}
    for key in _ROW_KEYS[1:]:
        values = [np.nan if value is None else value for value in rows[key]]
        if len(values) != len(ids):
            _fail(f"{procedure} seed {seed}: row column {key} does not align with the tasks")
        arrays[key] = np.asarray(values, dtype=np.float64)
    for key in ("success", "obstacle_violation", "valid_path"):
        arrays[key] = arrays[key].astype(bool)
    arrays["obstacle_cells"] = arrays["obstacle_cells"].astype(np.int64)
    recomputed = aggregate_primary_metrics({key: value for key, value in arrays.items() if key != "task_id"})
    stored = entry.get("metrics") or {}
    differing = [name for name, _, _ in PRIMARY_METRICS if not _same(recomputed[name], stored.get(name))]
    if differing:
        _fail(f"{procedure} seed {seed}: the metrics recomputed from the rows differ in {differing}")
    return arrays, {name: recomputed[name] for name, _, _ in PRIMARY_METRICS}


def _point_estimates(metrics_a, metrics_b, seeds, n_admissible) -> dict[str, Any]:
    """Per seed, the six metrics of each procedure on every admissible task and their difference (no interval)."""
    result = {}
    for seed in seeds:
        rows = {}
        for name, _, higher_is_better in PRIMARY_METRICS:
            a, b = metrics_a[seed][name], metrics_b[seed][name]
            defined = not any(value is None or (isinstance(value, float) and math.isnan(value)) for value in (a, b))
            rows[name] = {
                "a": a,
                "b": b,
                "delta": float(a - b) if defined else None,
                "n_tasks": n_admissible,
                "higher_is_better": higher_is_better,
            }
        result[str(seed)] = rows
    return result


def _compare_columns(arrays: Mapping[str, np.ndarray], keep: np.ndarray | None = None) -> dict[str, np.ndarray]:
    columns = {key: arrays[key] for key in ("task_id", "regret", "assd", "h95")}
    columns.update({key: arrays[key].astype(np.float64) for key in ("success", "obstacle_violation", "valid_path")})
    if keep is not None:
        columns = {key: value[keep] for key, value in columns.items()}
    return columns


# ----------------------------------------------------------------------------------------------------- quality


def _interval_record(record) -> dict[str, Any]:
    return {
        "delta": record.delta,
        "ci_low": record.ci_low,
        "ci_high": record.ci_high,
        "n_tasks": record.n,
        "n_a": record.n_a,
        "n_b": record.n_b,
        "n_blocks": record.n_blocks,
        "n_bootstrap": record.n_bootstrap,
        "higher_is_better": record.higher_is_better,
        "excludes_zero": record.excludes_zero,
    }


def _block_intervals(a_by_seed, b_by_seed, seeds, blocks, draws, bootstrap_seed) -> dict[str, Any]:
    """The statistics of ``compare_primary_metrics`` for any nonempty seed set (shared block draws, seed mean)."""
    results = {}
    for name, family, higher_is_better in PRIMARY_METRICS:
        column = _COLUMNS[name]
        record = multiseed_paired_bootstrap(
            {seed: a_by_seed[seed][column] for seed in seeds},
            {seed: b_by_seed[seed][column] for seed in seeds},
            metric=name,
            higher_is_better=higher_is_better,
            statistic=_STATISTICS.get(name, np.mean),
            n_bootstrap_per_seed=draws,
            seeds=tuple(seeds),
            seed=bootstrap_seed,
            blocks=blocks,
            nan_policy="omit" if family is MetricFamily.GEOMETRIC else "raise",
        )
        results[name] = _interval_record(record)
        if len(seeds) > 1:
            results[name]["per_seed"] = {str(s): _interval_record(r) for s, r in record.per_seed.items()}
    return results


def _geometry_masked(columns: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    masked = dict(columns)
    valid = columns["valid_path"].astype(bool)
    for key in ("assd", "h95"):
        masked[key] = np.where(valid, columns[key], np.nan)
    return masked


def _quality(
    a_by_seed,
    b_by_seed,
    seeds,
    blocks,
    status,
    sealing,
    draws,
    bootstrap_seed,
    permutation_seed,
    permutations,
    alpha,
):
    names = [name for name, _, _ in PRIMARY_METRICS]
    if status == "halted_unplaced_cap":
        return {
            "mode": "block_analysis_halted",
            "reason": "the unplaced tasks exceed the sealed cap: the block analysis stops for diagnosis",
            "intervals": None,
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    if len(set(blocks)) < 2:
        return {
            "mode": "no_block_interval",
            "reason": "fewer than two geographic blocks: no block interval can be estimated",
            "intervals": None,
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    masked_a = {seed: _geometry_masked(a_by_seed[seed]) for seed in seeds}
    masked_b = {seed: _geometry_masked(b_by_seed[seed]) for seed in seeds}
    confirmatory_seeds = len(seeds) == len(DEFAULT_SEEDS) and set(seeds) == set(DEFAULT_SEEDS)
    if not confirmatory_seeds:
        single = len(seeds) == 1
        return {
            "mode": "descriptive_single_seed" if single else "descriptive_per_seed",
            "label": SINGLE_SEED_LABEL if single else "descriptive per seed, not confirmatory",
            "reason": (
                "a single seed gives per-seed block intervals only: no p-value and no Holm decision (judge R8)"
                if single
                else f"the confirmatory family needs exactly the declared seeds {list(DEFAULT_SEEDS)}"
            ),
            "intervals": {
                str(seed): _block_intervals(masked_a, masked_b, [seed], blocks, draws, bootstrap_seed) for seed in seeds
            },
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    if not sealing["confirmatory_allowed"]:
        comparison = compare_primary_metrics(
            a_by_seed, b_by_seed, blocks=blocks, n_bootstrap=draws, seed=bootstrap_seed
        )
        return {
            "mode": "descriptive_unsealed",
            "label": "descriptive, not confirmatory: the criterion, family or resampling are not sealed",
            "reason": "the confirmatory family needs the block criterion, the family and the resampling sealed in "
            "the protocol record of every seed: " + "; ".join(sealing["reasons"][:6]),
            "intervals": {name: _interval_record(comparison.deltas[name]) for name in names},
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    if status != "meets_criterion":
        comparison = compare_primary_metrics(
            a_by_seed, b_by_seed, blocks=blocks, n_bootstrap=draws, seed=bootstrap_seed
        )
        return {
            "mode": "descriptive_blocks_inadequate",
            "label": "descriptive, not confirmatory: the blocks fail the sealed adequacy criterion",
            "reason": "the blocks do not meet the sealed adequacy criterion: intervals are descriptive, no decision",
            "intervals": {name: _interval_record(comparison.deltas[name]) for name in names},
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    groups = len(set(blocks))
    test = block_randomization_p_values(
        a_by_seed,
        b_by_seed,
        blocks=blocks,
        method="exact" if groups <= EXACT_BLOCK_LIMIT else "monte_carlo",
        permutations=permutations,
        seed=permutation_seed,
    )
    comparison = compare_primary_metrics(
        a_by_seed,
        b_by_seed,
        blocks=blocks,
        n_bootstrap=draws,
        alpha=alpha,
        seed=bootstrap_seed,
        p_values=test["p_values"],
        paired_test=test["schema"],
        test_assumptions=ASSUMPTIONS,
    )
    intervals = {}
    for name in names:
        intervals[name] = _interval_record(comparison.deltas[name])
        intervals[name]["per_seed"] = {str(s): _interval_record(r) for s, r in comparison.deltas[name].per_seed.items()}
    return {
        "mode": "confirmatory",
        "label": "confirmatory family: A = seg, B = reg, six primary metrics, Holm at alpha",
        "intervals": intervals,
        "paired_test": {key: test[key] for key in ("schema", "method", "permutations", "identity_added", "random_seed")}
        | {"number_of_blocks": test["number_of_blocks"], "status": test["status"], "assumptions": ASSUMPTIONS},
        "p_values": test["p_values"],
        "holm": {
            name: {"p_value": r.p_value, "p_adjusted": r.p_adjusted, "rejected": r.rejected}
            for name, r in comparison.holm.items()
        },
        "decision": comparison.decision.value,
        "votes": dict(comparison.votes),
        "inference_status": comparison.inference_status,
    }


# ----------------------------------------------------------------------------------------------------- latency


def _session_arrays(sessions: Any, procedure: str, seed: str) -> tuple[list[np.ndarray], list[np.ndarray], list]:
    if not isinstance(sessions, list) or not sessions:
        _fail(f"{procedure} seed {seed}: the aggregate carries no latency sessions")
    values, censored, members = [], [], []
    for session in sessions:
        latency = np.asarray(session.get("latency_ms"), dtype=np.float64)
        flags = np.asarray(session.get("censored"), dtype=bool)
        ids = list(session.get("sample_ids") or [])
        if latency.ndim != 1 or not latency.size or flags.shape != latency.shape or len(ids) != latency.size:
            _fail(f"{procedure} seed {seed}: a latency session is malformed or empty")
        if not np.isfinite(latency).all() or np.any(latency < 0):
            _fail(f"{procedure} seed {seed}: recorded latencies must be finite and nonnegative")
        values.append(latency)
        censored.append(flags)
        members.append(sorted(ids))
    return values, censored, members


class _SessionQuantiles:
    """Censored ``inverted_cdf`` quantiles of any multiset of whole sessions, from session counts."""

    def __init__(self, values: Sequence[np.ndarray], censored: Sequence[np.ndarray]):
        keys = np.concatenate([np.where(flags, np.inf, latency) for latency, flags in zip(values, censored)])
        owner = np.concatenate([np.full(len(latency), index) for index, latency in enumerate(values)])
        order = np.argsort(keys, kind="stable")
        self.keys, self.owner, self.n_sessions = keys[order], owner[order], len(values)

    def quantiles(self, counts: np.ndarray) -> np.ndarray:
        cumulative = np.cumsum(counts[self.owner])
        total = int(cumulative[-1])
        ranks = [_rank_index(total, q) for _, q in DEFAULT_QUANTILES]
        return np.asarray([self.keys[np.searchsorted(cumulative, rank + 1, side="left")] for rank in ranks])


def _value(value: float) -> dict[str, Any]:
    return {"value_ms": None if math.isinf(value) else float(value), "censored": bool(math.isinf(value))}


def _interval(draws: np.ndarray, alpha: float) -> dict[str, Any]:
    low, high = np.quantile(draws, [alpha / 2, 1 - alpha / 2], method=LATENCY_QUANTILE_METHOD)
    return {"ci_low": _value(low), "ci_high": _value(high)}


def _latency(a_entry, b_entry, seed, draws, latency_seed, alpha, *, bootstrap=True) -> dict[str, Any]:
    procedures = {PROCEDURE_A: a_entry, PROCEDURE_B: b_entry}
    loaded = {name: _session_arrays(entry.get("latency_sessions"), name, seed) for name, entry in procedures.items()}
    paired = loaded[PROCEDURE_A][2] == loaded[PROCEDURE_B][2]
    engines = {name: _SessionQuantiles(values, censored) for name, (values, censored, _) in loaded.items()}
    n_sessions = engines[PROCEDURE_A].n_sessions
    point = {name: engine.quantiles(np.ones(engine.n_sessions, dtype=np.int64)) for name, engine in engines.items()}
    for name, entry in procedures.items():
        block = entry.get("latency") or {}
        if block.get("rule") != RULE_ID:
            _fail(f"{name} seed {seed}: the aggregate latency block is not {RULE_ID}")
        for (label, _), estimate in zip(DEFAULT_QUANTILES, point[name]):
            stored = (block.get("quantiles") or {}).get(label) or {}
            if stored.get("censored") != bool(math.isinf(estimate)) or (
                not math.isinf(estimate) and stored.get("value_ms") != float(estimate)
            ):
                _fail(f"{name} seed {seed}: the sessions do not reproduce the aggregate {label}")
    if not bootstrap:
        from passage_eval.execution import timing

        for name, entry in procedures.items():
            sessions = entry["latency_sessions"]
            all_ids = [sample_id for session in sessions for sample_id in session["sample_ids"]]
            if all_ids != entry["task_rows"]["task_id"] or len(set(all_ids)) != len(all_ids):
                _fail("native session requests must cover each scored task exactly once, without replay")
            if len(sessions) > 1:
                resumes = entry.get("native_resumes")
                if not isinstance(resumes, list) or len(resumes) != len(sessions) - 1:
                    _fail("multiple native sessions require documented resume evidence")
                session_ids = []
                for session in sessions:
                    execution = session.get("native_execution")
                    warmup = timing.manifest_warmup(
                        {"series": "warm", "timing_profile": timing.NATIVE_PERSISTENT, "native_execution": execution}
                    )
                    if not warmup["performed"]:
                        _fail("each resumed process requires its own initial warm-up")
                    session_ids.append(execution["session_id"])
                if len(set(session_ids)) != len(session_ids):
                    _fail("resumed native sessions must identify distinct process lifetimes")
                for index, resume in enumerate(resumes, 1):
                    if not isinstance(resume, Mapping) or (
                        resume.get("previous_session_id") != session_ids[index - 1]
                        or resume.get("session_id") != session_ids[index]
                        or resume.get("accepted_prefix_sha256") != pinned.canonical_sha256(json_safe(sessions[:index]))
                        or not pinned.is_hex64(resume.get("journal_sha256"))
                        or not isinstance(resume.get("reason"), str)
                        or not resume["reason"].strip()
                    ):
                        _fail("native resume must bind its prior accepted prefix, journal and process identities")
        return {
            "sessions_paired": False,
            "pairing": None,
            "procedures": {
                name: {
                    "n_sessions": engines[name].n_sessions,
                    "n_requests": int(sum(len(values) for values in loaded[name][0])),
                    "n_censored": int(sum(int(flags.sum()) for flags in loaded[name][1])),
                    "quantiles": {
                        label: {"estimate": _value(point[name][column]), "ci_low": None, "ci_high": None}
                        for column, (label, _) in enumerate(DEFAULT_QUANTILES)
                    },
                }
                for name in engines
            },
            "contrast_a_minus_b": None,
            "uncertainty": "unavailable: one persistent process per candidate; no session bootstrap"
            if all(engine.n_sessions == 1 for engine in engines.values())
            else "unavailable: documented native resumes; no session bootstrap",
        }
    rng = np.random.default_rng(latency_seed)
    sample = {name: np.empty((draws, len(DEFAULT_QUANTILES))) for name in engines}
    for index in range(draws):
        if paired:
            counts = np.bincount(rng.integers(0, n_sessions, size=n_sessions), minlength=n_sessions)
            for name, engine in engines.items():
                sample[name][index] = engine.quantiles(counts)
        else:
            for name, engine in engines.items():
                counts = np.bincount(
                    rng.integers(0, engine.n_sessions, size=engine.n_sessions), minlength=engine.n_sessions
                )
                sample[name][index] = engine.quantiles(counts)
    result: dict[str, Any] = {}
    for name, engine in engines.items():
        values, censored, _ = loaded[name]
        result[name] = {
            "n_sessions": engine.n_sessions,
            "n_requests": int(sum(len(v) for v in values)),
            "n_censored": int(sum(int(c.sum()) for c in censored)),
            "quantiles": {
                label: {"estimate": _value(point[name][column]), **_interval(sample[name][:, column], alpha)}
                for column, (label, _) in enumerate(DEFAULT_QUANTILES)
            },
        }
    contrast: dict[str, Any] | None = None
    if paired:
        contrast = {}
        for column, (label, _) in enumerate(DEFAULT_QUANTILES):
            a, b = sample[PROCEDURE_A][:, column], sample[PROCEDURE_B][:, column]
            defined = np.isfinite(a) & np.isfinite(b)
            estimate_a, estimate_b = point[PROCEDURE_A][column], point[PROCEDURE_B][column]
            estimate = None if math.isinf(estimate_a) or math.isinf(estimate_b) else float(estimate_a - estimate_b)
            low = high = None
            if defined.all():
                low, high = (float(v) for v in np.quantile(a - b, [alpha / 2, 1 - alpha / 2], method=LATENCY_METHOD))
            contrast[label] = {
                "estimate_ms": estimate,
                "ci_low_ms": low,
                "ci_high_ms": high,
                "n_undefined_draws": int((~defined).sum()),
            }
    return {
        "sessions_paired": paired,
        "pairing": "the same session draws for both procedures (same batches and tasks)" if paired else None,
        "procedures": result,
        "contrast_a_minus_b": contrast,
        "contrast_unavailable_reason": None
        if paired
        else "the two procedures do not share the same sessions of tasks: no paired contrast",
    }


# ------------------------------------------------------------------------------------------------------- entry


def campaign_statistics(
    aggregates: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    permutation_seed: int,
    latency_bootstrap_seed: int,
    draws: int = DEFAULT_DRAWS,
    permutations: int = DEFAULT_DRAWS,
    criterion: Mapping[str, Any] | None = None,
    alpha: float = 0.05,
    aggregate_sha256: Sequence[str] | None = None,
    protocol_records: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """The ``passage-campaign-statistics/v1`` record of Seg and Reg aggregates (see the module docstring).

    ``aggregates`` holds one Seg and one Reg aggregate per seed, from one campaign or from one campaign per seed;
    ``aggregate_sha256`` optionally lists the file digest of each aggregate, in the same order, for the record.
    ``protocol_records`` maps the SHA-256 of each ``passage-test-admission/v1`` record file that the aggregates
    name to its content; without them, nothing is confirmatory. ``criterion`` is used only when no sealed record
    binds a ``block_analysis``; given with one, it must carry no unsealed marker and equal it.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha < 1:
        _fail("alpha must lie in (0, 1)")
    for name, value in (
        ("bootstrap_seed", bootstrap_seed),
        ("permutation_seed", permutation_seed),
        ("latency_bootstrap_seed", latency_bootstrap_seed),
    ):
        if type(value) is not int or value < 0:
            _fail(f"{name} must be a nonnegative integer sealed before the pass (it has no default)")
    for name, value in (("draws", draws), ("permutations", permutations)):
        if type(value) is not int or value < 1:
            _fail(f"{name} must be a positive integer")
    entries, seeds, described = _inputs(aggregates, aggregate_sha256)
    first = aggregates[0]
    profiles = [aggregate.get("qualification_profile") for aggregate in aggregates]
    if any(profile is not None for profile in profiles):
        from passage_eval.protocol.admission import NATIVE_PROFILES

        if profiles[0] not in NATIVE_PROFILES or any(profile != profiles[0] for profile in profiles):
            _fail("every native descriptive or Rule8 aggregate must carry the same qualification profile")
        for aggregate in aggregates:
            digest = aggregate.get("protocol_record_sha256")
            if (
                not pinned.is_hex64(digest)
                or not isinstance(protocol_records, Mapping)
                or digest not in protocol_records
            ):
                _fail("native descriptive or Rule8 aggregates require their matching protocol records")
            record = protocol_records[digest]
            protocol = record.get("protocol") if isinstance(record, Mapping) else None
            if not isinstance(protocol, Mapping) or protocol.get("profile") != aggregate["qualification_profile"]:
                _fail("the aggregate's matching protocol record does not seal its native descriptive or Rule8 profile")
            for seed in aggregate["seeds"]:
                problems = _record_problems(record, seed, aggregate)
                if problems:
                    _fail("the native descriptive protocol record does not bind its aggregate: " + "; ".join(problems))
    resampling = {
        "bootstrap_seed": bootstrap_seed,
        "permutation_seed": permutation_seed,
        "latency_bootstrap_seed": latency_bootstrap_seed,
        "draws": draws,
        "permutations": permutations,
    }
    sealing = _sealing(entries, seeds, protocol_records, resampling=resampling, alpha=alpha, target=first.get("target"))
    if sealing.get("descriptive_only"):
        from passage_eval.protocol.admission import DESCRIPTIVE_NATIVE_PROFILE, PLANNED_SEEDS

        if (
            len(seeds) != 1
            or seeds[0] not in PLANNED_SEEDS
            or (sealing.get("qualification_profile") == DESCRIPTIVE_NATIVE_PROFILE and seeds != [2026])
        ):
            _fail("one sealed seed per descriptive record is required (legacy profile: 2026 only)")
    criterion = _criterion(criterion, sealing)
    population = first.get("population")
    if not isinstance(population, Mapping) or not isinstance(population.get("n_admissible"), int):
        _fail("the aggregates state no admissible population")
    n_admissible = population["n_admissible"]

    def entry(procedure: str, seed: int) -> Mapping[str, Any]:
        return entries[(procedure, seed)]["entry"]

    a_by_seed, b_by_seed, metrics_a, metrics_b = {}, {}, {}, {}
    for seed in seeds:
        key = str(seed)
        a_by_seed[seed], metrics_a[seed] = _rows(entry(PROCEDURE_A, seed), PROCEDURE_A, key, n_admissible)
        b_by_seed[seed], metrics_b[seed] = _rows(entry(PROCEDURE_B, seed), PROCEDURE_B, key, n_admissible)
    # The task rows keep the parent order of the admissible roster (the aggregator refuses any other order). The
    # block map is only a mapping: a JSON file written with sorted keys returns it in lexicographic id order.
    order = list(a_by_seed[seeds[0]]["task_id"])
    block_map = (entry(PROCEDURE_A, seeds[0]).get("geographic_blocks") or {}).get("tasks")
    if not isinstance(block_map, Mapping) or set(block_map) != set(order):
        _fail("the geographic blocks do not name exactly the admissible tasks of the task rows")
    for seed in seeds:
        key = str(seed)
        for procedure in (PROCEDURE_A, PROCEDURE_B):
            if dict((entry(procedure, seed).get("geographic_blocks") or {}).get("tasks") or {}) != dict(block_map):
                _fail(f"{procedure} seed {key}: the geographic blocks differ between the aggregates")
        for rows in (a_by_seed[seed], b_by_seed[seed]):
            if list(rows["task_id"]) != order:
                _fail(f"seed {key}: the task rows of every procedure and seed must follow one task order")
    blocks = [block_map[task] for task in order]
    if any(not isinstance(block, str) or not block for block in blocks):
        _fail("every task needs its geographic block")
    limits = {key: criterion[key] for key in ("min_blocks", "max_block_share", "max_unplaced_fraction")}
    census = adequacy(dict(zip(order, blocks)), **limits)
    if not census["unplaced_within_cap"]:
        status = "halted_unplaced_cap"
    elif census["meets_criterion"]:
        status = "meets_criterion"
    else:
        status = "fails_criterion"
    columns_a = {seed: _compare_columns(a_by_seed[seed]) for seed in seeds}
    columns_b = {seed: _compare_columns(b_by_seed[seed]) for seed in seeds}
    if sealing.get("descriptive_only"):
        quality = {
            "mode": sealing.get("qualification_profile", "descriptive_native_2026"),
            "label": SINGLE_SEED_LABEL,
            "reason": "the native descriptive profile does not perform geographic block inference",
            "intervals": None,
            "p_values": None,
            "holm": None,
            "decision": None,
        }
    else:
        quality = _quality(
            columns_a,
            columns_b,
            seeds,
            blocks,
            status,
            sealing,
            draws,
            bootstrap_seed,
            permutation_seed,
            permutations,
            alpha,
        )
    # Reported in every mode, also when the block analysis is halted (judge R5 stops the block analysis only).
    quality["point_estimates"] = _point_estimates(metrics_a, metrics_b, seeds, n_admissible)
    quality["point_estimates_label"] = (
        "descriptive point estimates: the six metrics of each procedure on every admissible task and a - b; no interval"
    )
    sensitivity = None
    unplaced = np.asarray([block == UNPLACED_BLOCK for block in blocks])
    if (
        not sealing.get("descriptive_only")
        and unplaced.any()
        and status != "halted_unplaced_cap"
        and (~unplaced).sum()
        and len(set(blocks)) > 2
    ):
        keep = ~unplaced
        kept_blocks = [block for block, flag in zip(blocks, keep) if flag]
        sensitivity = {
            "label": "sensitivity without the unplaced tasks: descriptive, never a decision",
            "n_tasks": int(keep.sum()),
            "intervals": {
                str(seed): _block_intervals(
                    {seed: _geometry_masked(_compare_columns(a_by_seed[seed], keep))},
                    {seed: _geometry_masked(_compare_columns(b_by_seed[seed], keep))},
                    [seed],
                    kept_blocks,
                    draws,
                    bootstrap_seed,
                )
                for seed in seeds
            },
        }
    p90 = {}
    for name, by_seed in ((PROCEDURE_A, a_by_seed), (PROCEDURE_B, b_by_seed)):
        p90[name] = {}
        for seed in seeds:
            failed = int((~by_seed[seed]["valid_path"]).sum())
            rate = failed / n_admissible
            p90[name][str(seed)] = {
                "n_failed": failed,
                "failure_rate": rate,
                "p90_penalty_dominated": rate >= PENALTY_DOMINATED_FAILURE_RATE,
            }
    from passage_eval.execution import timing

    native_persistent = first.get("timing_profile") == timing.NATIVE_PERSISTENT
    latency = {
        str(seed): _latency(
            entry(PROCEDURE_A, seed),
            entry(PROCEDURE_B, seed),
            str(seed),
            draws,
            latency_bootstrap_seed,
            alpha,
            bootstrap=not native_persistent,
        )
        for seed in seeds
    }
    single = len(seeds) == 1
    record = {
        "schema": STATISTICS_SCHEMA,
        "status": "confirmatory" if quality["mode"] == "confirmatory" else "descriptive",
        "label": SINGLE_SEED_LABEL if single else quality.get("label"),
        "inputs": {
            "common": {field: copy.deepcopy(first.get(field)) for field in _COMMON_FIELDS}
            | {"harness_source_sha256": _source_sha256(first)},
            "aggregates": described,
            "per_seed": {
                str(seed): {
                    procedure: {
                        "input": entries[(procedure, seed)]["index"],
                        **{field: entries[(procedure, seed)]["aggregate"].get(field) for field in _CAMPAIGN_FIELDS},
                    }
                    for procedure in (PROCEDURE_A, PROCEDURE_B)
                }
                for seed in seeds
            },
        },
        "family": {
            "a": PROCEDURE_A,
            "b": PROCEDURE_B,
            "contrast": "a - b",
            "metrics": [name for name, _, _ in PRIMARY_METRICS],
            "alpha": alpha,
        },
        "resampling": {
            "bootstrap_seed": bootstrap_seed,
            "permutation_seed": permutation_seed,
            "latency_bootstrap_seed": latency_bootstrap_seed,
            "draws": draws,
            "permutations": permutations,
            "quality_unit": "whole geographic blocks, the same draw for both procedures, every seed and metric",
            "latency_unit": "descriptive quantiles in one persistent process, no bootstrap"
            if native_persistent
            else "whole sessions (batches), the same draw for both procedures",
        },
        "quantile_methods": {
            "quality": QUALITY_QUANTILE_METHOD,
            "latency": LATENCY_QUANTILE_METHOD,
            "quality_interval_endpoints": "linear quantiles of the bootstrap draws",
            "latency_interval_endpoints": "inverted_cdf quantiles of the bootstrap draws",
            "latency_rule": RULE_ID,
        },
        "completeness": {
            "rule": "exactly one accepted record per admissible task, procedure, target and seed",
            "n_admissible": n_admissible,
            "procedures": [PROCEDURE_A, PROCEDURE_B],
            "seeds": seeds,
            "complete": True,
        },
        "sealing": sealing,
        "blocks": {
            "criterion": criterion,
            "criterion_status": criterion["status"],
            "criterion_sealed": criterion["sealed"],
            "status": status,
            "adequacy": census,
        },
        "quality": quality,
        "quality_sensitivity_without_unplaced": sensitivity,
        "p90_penalty_dominated": p90,
        "latency": latency,
        "limitations": [
            "conditional on the frozen checkpoints; the public test split has been exposed before",
            "blocks are not proven independent or exchangeable; the randomization test states its assumptions",
            "latency requests whose warm-up was incomplete stay in the series, as in the aggregate's point estimate",
        ],
    }
    return json_safe(record)
