# MIT License
#
# Copyright (c) 2026 THALES

"""The sealed campaign plan ``passage-campaign-plan/v1``: read, check and bind it before any batch runs.

``read_plan(path, sha256)`` refuses a plan whose bytes differ from the pin, an unknown or missing key at any level,
and every inconsistency below. It reads only control files (the plan, the pinned specs, rosters, chain and partition
manifest); it opens no scene file and lists no directory.

What the plan seals, and what is checked:

* **Scope.** ``split`` (``validation``; ``test`` only with ``allow_test`` and an all-GREEN
  ``passage-test-admission/v1`` record named by ``admission``, whose sealed decisions and candidate identities the
  plan must repeat),
  ``target`` (``host``; an edge target raises ``EdgeTransportNotQualified`` unless ``edge_transport_evidence`` names
  an admission record whose gate G3 is GREEN for it, and the runner still has no remote transport), ``purpose``
  (``verified-candidate-run`` required on test) and ``protocol_record_sha256``.
* **Population.** The admissible roster (file and ``roster_sha256``), the source roster digest and the reference
  admission chain (file, ``chain_sha256``, policy, cohort, search limit and counts), the partition (directory,
  manifest digest, batch size; re-derived from the admissible roster).
* **Request limits.** The complete ``budget``; the ``timing`` profile (a re-warm profile), its boundary identifier,
  ``deadline_grace_ms``, ``hard_timeout_ms``, ``startup_timeout_s``, ``max_restarts`` (at least the batch size, so
  that the restart cap never decides the status of a task) and the worker ``rlimits``.
* **Warm-up and order.** The validation warm-up roster (file, digest, salt, ``k``, requests per warm pass), built
  against the admissible roster on the same pinned dataset root; the paired order (schedule seed, ``per_batch``
  granularity, procedures).
* **Batch window and dynamic admission.** ``lease_or_timeout_ms``, the measured overheads (``fixed_overhead_ms``,
  ``per_seed_overhead_ms``, ``per_request_overhead_ms``, ``persistence_ms``), the observed warm request time of every
  procedure (``observed_request_ms``) and ``safety_margin`` (at least 3). Every session admits its requests
  dynamically within the window (:mod:`passage_eval.supervision.session`); the plan is refused unless the window
  holds the observed batch ``safety_margin`` times and each procedure's share of a fresh window can engage one
  request (``protocol.batching.admission_sizing``). The worst case is reported, not enforced. One seed per plan.
* **Procedures.** For each of ``astar``, ``seg``, ``reg``: the pinned spec (whose cell, seeds, budget, reference
  policy, latency settings and split must equal the plan), the decoder entry (one of the frozen entries for Seg and
  Reg), the decoder or planner tree pins, the environment pins, the split procedure pins, the candidate checkpoint
  (equal to ``docs/benchmark/final-candidates.json`` as committed at the plan's harness commit, whose single
  training seed must be the plan's only seed) and bundle digests. The runner refuses a scored run whose loaded
  checkpoint or bundle files are not the plan's (``runner.verify_scored``). An admitted test plan also repeats the
  record's predictor, decoder, checkpoint, target bundle and decoder tree (``decoder_allow_sha256`` is exactly that
  tree); its ``astar`` procedure must be a harness planner entry (``admission.CANONICAL_PLANNER_ENTRIES``) pinned by
  the same tree.
* **Harness.** Interpreters, commit and source hash (``cli._source_hash``); the running harness must have that
  commit and source hash, and every scored manifest must record the source hash.
* **Aggregation.** The censoring rule identifier (the installed one) and the minimum of warm requests; the seeds; the
  infrastructure relaunch policy (an author decision recorded with every relaunch).

The paths of a plan are absolute. The output directory must not overlap the dataset root, the partition, the
evidence or any candidate tree. Outside an admitted test plan, any path component named ``test`` is refused.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passage_eval.campaign import EDGE_VALIDATION_RELAY_SCHEMA, PLAN_SCHEMA, EdgeTransportNotQualified
from passage_eval.campaign import partition as partition_io
from passage_eval.io import pinned
from passage_eval.io import roster as roster_io
from passage_eval.protocol.guards import TestSplitRefused, refuse_test_roster, refuse_test_split

PROCEDURE_NAMES = ("astar", "seg", "reg")
SPLIT_PROCEDURES = ("seg", "reg")
PLANNER_PROCEDURES = ("astar",)
_PLAN_KEYS = {
    "schema",
    "campaign_id",
    "split",
    "target",
    "purpose",
    "protocol_record_sha256",
    "admission",
    "edge_transport_evidence",
    "dataset_root",
    "output_dir",
    "rosters",
    "reference",
    "partition",
    "budget",
    "timing",
    "warmup",
    "order",
    "batching",
    "procedures",
    "harness",
    "aggregation",
    "seeds",
    "infrastructure_relaunch_policy",
    "r7_policy",
}
_SUBKEYS = {
    "rosters": {"admissible", "source_roster_sha256", "reference_admission_chain"},
    "reference": {"policy", "cohort_id", "max_expansions"},
    "partition": {"dir", "manifest_sha256", "batch_size"},
    "budget": {"wall_time_ms", "max_expansions", "max_queue_operations", "max_live_states"},
    "timing": {
        "profile",
        "boundary_id",
        "deadline_grace_ms",
        "hard_timeout_ms",
        "startup_timeout_s",
        "max_restarts",
        "rlimits",
    },
    "warmup": {"roster", "salt", "k", "requests_per_warm_pass"},
    "order": {"schedule_seed", "granularity", "procedures"},
    "batching": {
        "lease_or_timeout_ms",
        "fixed_overhead_ms",
        "per_seed_overhead_ms",
        "per_request_overhead_ms",
        "persistence_ms",
        "observed_request_ms",
        "safety_margin",
    },
    "harness": {"python", "worker_python", "commit", "source_sha256"},
    "aggregation": {"censoring_rule_id", "min_warm_requests"},
}
_ROSTER_FILE_KEYS = {"path", "file_sha256", "roster_sha256"}
_CHAIN_KEYS = {"path", "file_sha256", "chain_sha256"}
_RLIMIT_KEYS = {"nofile", "fsize_mb", "as_mb", "cpu_s"}
_ADMISSION_KEYS = {"record", "record_sha256", "evidence_dir"}
_PROCEDURE_KEYS = {
    "spec",
    "mode",
    "predictor",
    "decoder",
    "decoder_kwargs",
    "precision",
    "candidate_paths",
    "runtime_binds",
    "decoder_allow_sha256",
    "planner_allow_sha256",
    "environment_allow_sha256",
    "split_procedure_allow_sha256",
    "checkpoint_sha256",
    "bundle_sha256",
}
PURPOSES = ("diagnostic", "verified-candidate-run")
R7_POLICY_KEYS = {
    "max_ambiguous_attempts",
    "single_task_candidate_failures",
    "relaunch_stop_fraction",
    "candidate_attributable",
}


class PlanError(ValueError):
    """A campaign plan that does not satisfy its schema, its pins or the protocol rules."""


@dataclass(frozen=True)
class CampaignPlan:
    """A checked plan with the control files it binds."""

    plan: dict[str, Any]
    sha256: str
    admissible: dict[str, Any]
    warmup_roster: dict[str, Any]
    chain: dict[str, Any]
    partition: dict[str, Any]
    specs: dict[str, dict[str, Any]]
    admission: Any = None

    @property
    def allow_test(self) -> bool:
        return self.admission is not None

    @property
    def procedures(self) -> tuple[str, ...]:
        return tuple(sorted(self.plan["procedures"]))

    def seed_spec(self, procedure: str, seed: int) -> dict[str, Any]:
        """The ``bench_spec`` a supervised run of ``seed`` records (``cli._seed_spec`` applied to the pinned spec)."""
        spec = copy.deepcopy(self.specs[procedure])
        by_seed = spec.get("inference_kwargs_by_seed", {})
        seed_kwargs = by_seed.get(str(seed), by_seed.get(seed, {}))
        spec["inference_kwargs"] = {**spec["inference_kwargs"], **seed_kwargs}
        spec["seeds"] = [int(seed)]
        return spec


@dataclass(frozen=True)
class ValidationEdgeRelayRequest:
    """The only pre-G3 edge entry: two sealed validation packages and host callbacks, never a roster or a plan.

    This is deliberately not a ``CampaignPlan`` variant.  In particular, it holds no roster, partition, dataset,
    admission, or path-to-plan field, so it cannot become an alternate way to read roster S before G3 is green.
    """

    target: str
    packages: Mapping[str, Any]
    candidates: Mapping[str, Mapping[str, Any]]
    native_relay: Callable[[str, Any], str | Path]
    scorer: Callable[[str, dict[str, Any]], Any]
    split: str = "validation"
    procedures: tuple[str, ...] = SPLIT_PROCEDURES
    seed: int = 2026
    schema: str = EDGE_VALIDATION_RELAY_SCHEMA
    replica_catalog: str | Path | None = None
    replica_catalog_sha256: str | None = None


def check_validation_edge_relay_request(request: ValidationEdgeRelayRequest) -> None:
    """Refuse every scope expansion before opening a sealed package or calling native code."""
    if not isinstance(request, ValidationEdgeRelayRequest):
        raise PlanError("validation edge relay needs its explicit request record")
    _require(request.schema == EDGE_VALIDATION_RELAY_SCHEMA, "validation edge relay schema differs")
    _require(request.split == "validation", "validation edge relay accepts the validation split only")
    from passage_eval.campaign.edge import EDGE_TARGETS

    _require(request.target in EDGE_TARGETS, "validation edge relay target must be Thor, Orin, or Tiger")
    _require(tuple(request.procedures) == SPLIT_PROCEDURES, "validation edge relay runs exactly Seg and Reg")
    _require(
        type(request.seed) is int and request.seed in (1234, 2026, 3407),
        "validation edge relay needs a declared training seed",
    )
    if request.seed != 2026:
        from passage_eval.protocol.replicas import load_replica_catalog

        _require(
            request.replica_catalog is not None and pinned.is_hex64(request.replica_catalog_sha256),
            "replica validation requires its pinned catalogue",
        )
        load_replica_catalog(request.replica_catalog, request.replica_catalog_sha256)
    _require(
        isinstance(request.candidates, Mapping) and set(request.candidates) == set(SPLIT_PROCEDURES),
        "validation edge relay needs exactly the frozen Seg and Reg candidates",
    )
    _require(
        isinstance(request.packages, Mapping) and set(request.packages) == set(SPLIT_PROCEDURES),
        "validation edge relay needs exactly the Seg and Reg packages",
    )
    _require(callable(request.native_relay), "validation edge relay needs a supplied native relay")
    _require(callable(request.scorer), "validation edge relay needs a host scorer")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _keys(value: Any, expected: set[str], where: str) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{where} must be an object")
    unknown, missing = sorted(set(value) - expected), sorted(expected - set(value))
    _require(not unknown and not missing, f"{where}: unknown keys {unknown}, missing keys {missing}")
    return dict(value)


def _hex_list(value: Any, where: str, *, required: bool) -> list[str]:
    _require(isinstance(value, list) and all(pinned.is_hex64(v) for v in value), f"{where} must list SHA-256 values")
    _require(not required or bool(value), f"{where} needs at least one pin")
    _require(len(set(value)) == len(value), f"{where} repeats a pin")
    return value


def _absolute(value: Any, where: str, *, allow_test_component: bool, must_exist: bool = True) -> Path:
    _require(isinstance(value, str) and value.startswith("/"), f"{where} must be an absolute path")
    path = Path(value)
    if not allow_test_component:
        _require(
            not any(part.lower() == "test" for part in path.parts),
            f"{where}: a path component named test is refused outside an admitted test plan",
        )
    if must_exist:
        _require(path.exists(), f"{where} does not exist")
        resolved = path.resolve(strict=True)
        if not allow_test_component:
            _require(not any(p.lower() == "test" for p in resolved.parts), f"{where} resolves through a test directory")
    return path


def _final_candidates(commit: str) -> dict[str, Any]:
    """``final-candidates.json`` as committed at the plan's harness commit (a working-tree edit is refused)."""
    from passage_eval.protocol.admission import committed_final_candidates

    try:
        return committed_final_candidates(commit)[0]
    except ValueError as exc:
        raise PlanError(str(exc)) from exc


def _check_scope(plan: dict[str, Any], allow_test: bool) -> Any:
    """Split and admission: return the grant of an admitted test plan (``None`` otherwise)."""
    split = refuse_test_split(plan["split"], context="the campaign planner") if plan["split"] != "test" else "test"
    if split == "test":
        if not isinstance(plan["admission"], Mapping):
            raise TestSplitRefused(
                "the campaign planner refuses the test split: a test plan needs an admission record whose six "
                "gates are GREEN"
            )
        admission = _keys(plan["admission"], _ADMISSION_KEYS, "admission")
        if allow_test is not True:
            raise TestSplitRefused(
                "the campaign planner refuses the test split: it needs --allow-test-split and an all-GREEN "
                "test admission record"
            )
        from passage_eval.protocol.admission import grant_test_admission

        record = _absolute(admission["record"], "admission.record", allow_test_component=True)
        evidence = _absolute(admission["evidence_dir"], "admission.evidence_dir", allow_test_component=True)
        _require(pinned.is_hex64(admission["record_sha256"]), "admission.record_sha256 must be a SHA-256")
        if pinned.sha256_bytes(pinned.read_bounded(record)) != admission["record_sha256"]:
            raise TestSplitRefused(
                "the campaign planner refuses the test split: the admission record differs from its pin"
            )
        grant = grant_test_admission(record, evidence, refuse=TestSplitRefused)
        refuse_test_split("test", allow_test=True, admission=grant, context="the campaign planner")
        return grant
    _require(split == "validation", f"campaigns run on the validation split or an admitted test split, not {split!r}")
    _require(plan["admission"] is None, "an admission record applies to a test plan only")
    return None


def _check_target(plan: dict[str, Any]) -> None:
    target = plan["target"]
    from passage_eval.protocol.admission import EDGE_TARGETS, check_admission

    if target == "host":
        _require(plan["edge_transport_evidence"] is None, "edge transport evidence applies to an edge target only")
        return
    _require(target in EDGE_TARGETS, f"unknown target {target!r}")
    evidence = plan["edge_transport_evidence"]
    if evidence is None:
        raise EdgeTransportNotQualified(
            f"edge transport not qualified: target {target!r} needs lease and transport evidence that the admission "
            "gate G3 validates; no such evidence is supplied"
        )
    evidence = _keys(evidence, _ADMISSION_KEYS, "edge_transport_evidence")
    record = _absolute(evidence["record"], "edge_transport_evidence.record", allow_test_component=True)
    if pinned.sha256_bytes(pinned.read_bounded(record)) != evidence["record_sha256"]:
        raise EdgeTransportNotQualified("edge transport not qualified: the evidence record differs from its pin")
    report = check_admission(record, evidence["evidence_dir"])
    g3 = report.gate("G3")
    targets = report.admitted.get("targets")
    if g3.status != "GREEN" or (targets is not None and target not in targets):
        codes = ", ".join(reason["code"] for reason in g3.reasons[:5]) or "target not sealed"
        raise EdgeTransportNotQualified(f"edge transport not qualified for {target!r}: gate G3 is RED ({codes})")


def _check_paths(plan: dict[str, Any], allow_test: bool) -> None:
    output = _absolute(plan["output_dir"], "output_dir", allow_test_component=allow_test, must_exist=False)
    protected = [
        _absolute(plan["dataset_root"], "dataset_root", allow_test_component=allow_test),
        _absolute(plan["partition"]["dir"], "partition.dir", allow_test_component=allow_test),
    ]
    for name, procedure in plan["procedures"].items():
        for index, candidate in enumerate(procedure["candidate_paths"]):
            protected.append(_absolute(candidate, f"{name}.candidate_paths[{index}]", allow_test_component=allow_test))
    if plan["admission"] is not None:
        protected.append(Path(plan["admission"]["evidence_dir"]))
    resolved_output = output.resolve(strict=False)
    for root in protected:
        root = root.resolve(strict=True)
        _require(
            not (
                resolved_output == root or resolved_output.is_relative_to(root) or root.is_relative_to(resolved_output)
            ),
            "output_dir must be separate from the dataset, partition, evidence and candidate trees",
        )
    for key in ("python", "worker_python"):
        _absolute(plan["harness"][key], f"harness.{key}", allow_test_component=allow_test)


def check_source_paths(plan: Mapping[str, Any], roster: Mapping[str, Any]) -> None:
    """Refuse a split directory reached through a symbolic link and a source file outside it (no listing)."""
    root = Path(plan["dataset_root"])
    split_dir = root / roster["dataset"]["config"] / roster["dataset"]["split"]
    for part in (root / roster["dataset"]["config"], split_dir):
        _require(not part.is_symlink(), f"{part.name} must not be a symbolic link")
    resolved = split_dir.resolve(strict=True)
    _require(resolved.is_relative_to(root.resolve(strict=True)), "the split directory escapes the dataset root")
    for record in roster["source_files"]:
        source = root / record["path"]
        _require(not source.is_symlink(), f"{record['path']} must not be a symbolic link")
        _require(source.resolve(strict=True).is_relative_to(resolved), f"{record['path']} resolves outside its split")


def _read_roster_file(entry: Any, where: str, allow_test: bool) -> dict[str, Any]:
    entry = _keys(entry, _ROSTER_FILE_KEYS, where)
    path = _absolute(entry["path"], f"{where}.path", allow_test_component=allow_test)
    payload = pinned.read_pinned_json(path, entry["file_sha256"])
    return roster_io.validate_roster(payload, expected_sha256=entry["roster_sha256"])


def _check_chain(plan: dict[str, Any], admissible: dict[str, Any], allow_test: bool) -> dict[str, Any]:
    entry = _keys(plan["rosters"]["reference_admission_chain"], _CHAIN_KEYS, "rosters.reference_admission_chain")
    path = _absolute(entry["path"], "rosters.reference_admission_chain.path", allow_test_component=allow_test)
    chain = pinned.read_pinned_json(path, entry["file_sha256"])
    from passage_eval.protocol.reference_admission import CHAIN_SCHEMA

    _require(isinstance(chain, Mapping) and chain.get("schema") == CHAIN_SCHEMA, f"a {CHAIN_SCHEMA} record is required")
    digest = pinned.canonical_sha256({k: v for k, v in chain.items() if k != "chain_sha256"})
    _require(digest == chain.get("chain_sha256") == entry["chain_sha256"], "the admission chain digest differs")
    _require(
        chain["admissible_roster_sha256"] == admissible["roster_sha256"], "the chain names another admissible roster"
    )
    _require(chain["source_roster_sha256"] == plan["rosters"]["source_roster_sha256"], "the chain names another source")
    reference = plan["reference"]
    _require(
        (chain["reference_policy"], chain["reference_cohort_id"], chain["reference_max_expansions"])
        == (reference["policy"], reference["cohort_id"], reference["max_expansions"]),
        "the plan's reference policy, cohort or search limit differs from the admission chain",
    )
    _require(chain["source_file_hashes"] == "verified", "the admission inventory did not verify its source files")
    derivation = admissible["selection"].get("derivation") or {}
    _require(
        derivation.get("inventory_sha256") == chain["inventory_sha256"]
        and derivation.get("policy_sha256") == chain["policy_sha256"]
        and derivation.get("parent_roster_sha256") == chain["source_roster_sha256"],
        "the admissible roster was not derived by this admission chain",
    )
    counts = chain["counts"]
    _require(
        (derivation.get("n_source"), derivation.get("n_excluded"), derivation.get("n_admissible"))
        == (counts["n_source"], counts["n_excluded"], counts["n_admissible"])
        and counts["n_admissible"] == len(admissible["tasks"]),
        "the source, excluded and admissible counts differ between the chain and the admissible roster",
    )
    return chain


def _check_limits(plan: dict[str, Any]) -> None:
    from passage_eval.execution import timing
    from passage_eval.protocol.contracts import BudgetSpec

    budget = _keys(plan["budget"], _SUBKEYS["budget"], "budget")
    try:
        canonical = BudgetSpec(**budget).to_dict()
    except (TypeError, ValueError) as exc:
        raise PlanError(f"invalid budget: {exc}") from exc
    _require(canonical == budget, "the budget must be written in its canonical form")
    _require(_is_number(budget["wall_time_ms"]) and budget["wall_time_ms"] > 0, "a wall budget is required")
    t = _keys(plan["timing"], _SUBKEYS["timing"], "timing")
    try:
        profile = timing.get_profile(t["profile"])
    except ValueError as exc:
        raise PlanError(str(exc)) from exc
    _require(profile.rewarm_after_restart, "a campaign measures warm latency under a re-warm timing profile")
    _require(t["boundary_id"] == profile.boundary, "timing.boundary_id differs from the profile's boundary")
    for key in ("deadline_grace_ms", "hard_timeout_ms", "startup_timeout_s"):
        _require(_is_number(t[key]) and t[key] > 0, f"timing.{key} must be positive")
    _require(_is_int(t["max_restarts"]) and t["max_restarts"] >= 0, "timing.max_restarts must be a nonnegative integer")
    rlimits = _keys(t["rlimits"], _RLIMIT_KEYS, "timing.rlimits")
    for key, value in rlimits.items():
        _require(value is None or (_is_int(value) and value > 0), f"timing.rlimits.{key} must be positive or null")
    _require(_is_int(rlimits["nofile"]) and _is_int(rlimits["fsize_mb"]), "nofile and fsize_mb are required")


def _check_warmup(plan: dict[str, Any], admissible: dict[str, Any], allow_test: bool) -> dict[str, Any]:
    from passage_eval.protocol import warmup as warmup_io

    w = _keys(plan["warmup"], _SUBKEYS["warmup"], "warmup")
    roster = _read_roster_file(w["roster"], "warmup.roster", allow_test)
    refuse_test_roster(roster, context="the campaign warm-up")  # warm-up inputs come from validation only
    _require(roster["dataset"]["split"] == warmup_io.WARMUP_SPLIT, "the warm-up roster must be a validation roster")
    _require(roster["schema"] == roster_io.SCHEMA_V2, "the warm-up roster must be passage-roster-v2")
    _require(isinstance(w["salt"], str) and w["salt"] == roster["selection"]["salt"], "warm-up salt differs")
    _require(_is_int(w["k"]) and w["k"] == len(roster["tasks"]), "warm-up k differs from the warm-up roster")
    _require(w["requests_per_warm_pass"] == w["k"], "every warm pass replays the k warm-up inputs")
    derivation = roster["selection"].get("warmup") or {}
    _require(
        derivation.get("evaluated_roster_sha256") == admissible["roster_sha256"]
        and derivation.get("disjointness_checked_by") == warmup_io.CONTENT_CHECK,
        "the warm-up roster was not built disjoint (by content) from the admissible roster",
    )
    _require(
        roster["dataset_root_identity"] == admissible["dataset_root_identity"]
        and roster["dataset"]["commit"] == admissible["dataset"]["commit"],
        "the warm-up roster must pin the same dataset root as the admissible roster (--warmup-roster reads it)",
    )
    _require(roster["cell"] == admissible["cell"], "warm-up inputs must come from the evaluated cell")
    return roster


def _check_order_and_aggregation(plan: dict[str, Any]) -> None:
    from passage_eval.metrics import latency

    order = _keys(plan["order"], _SUBKEYS["order"], "order")
    _require(order["granularity"] == "per_batch", "only the per_batch paired order is implemented")
    seed = order["schedule_seed"]
    _require(_is_int(seed) or (isinstance(seed, str) and seed.strip()), "order.schedule_seed is required")
    _require(sorted(order["procedures"]) == sorted(plan["procedures"]), "order.procedures must name every procedure")
    aggregation = _keys(plan["aggregation"], _SUBKEYS["aggregation"], "aggregation")
    _require(aggregation["censoring_rule_id"] == latency.RULE_ID, f"the installed censoring rule is {latency.RULE_ID}")
    _require(_is_int(aggregation["min_warm_requests"]) and aggregation["min_warm_requests"] >= 1, "min_warm_requests")
    seeds = plan["seeds"]
    _require(
        isinstance(seeds, list) and seeds and all(_is_int(s) for s in seeds) and len(set(seeds)) == len(seeds),
        "seeds must be unique integers",
    )
    _require(
        isinstance(plan["infrastructure_relaunch_policy"], str) and plan["infrastructure_relaunch_policy"].strip(),
        "the infrastructure relaunch policy must be stated",
    )
    policy = plan["r7_policy"]
    _require(isinstance(policy, Mapping) and set(policy) == R7_POLICY_KEYS, "r7_policy has unknown or missing fields")
    _require(
        _is_int(policy["max_ambiguous_attempts"]) and policy["max_ambiguous_attempts"] >= 1,
        "r7_policy.max_ambiguous_attempts must be a positive integer",
    )
    _require(
        _is_int(policy["single_task_candidate_failures"]) and policy["single_task_candidate_failures"] >= 1,
        "r7_policy.single_task_candidate_failures must be a positive integer",
    )
    _require(
        _is_number(policy["relaunch_stop_fraction"]) and 0 <= policy["relaunch_stop_fraction"] <= 1,
        "r7_policy.relaunch_stop_fraction must be between zero and one",
    )
    _require(
        policy["candidate_attributable"] == ["killed_by_window", "invalid_session"],
        "r7_policy.candidate_attributable must name the two executable candidate-attributable classes",
    )


def _check_procedures(
    plan: dict[str, Any], admissible: dict[str, Any], allow_test: bool, admission=None
) -> dict[str, dict]:
    from passage_eval.cli import BenchSpec
    from passage_eval.protocol.admission import FROZEN_DECODER_ENTRIES
    from passage_eval.protocol.contracts import BudgetSpec

    procedures = plan["procedures"]
    _require(isinstance(procedures, Mapping) and procedures, "procedures must be a nonempty object")
    _require(set(procedures) <= set(PROCEDURE_NAMES), f"procedures are among {list(PROCEDURE_NAMES)}")
    frozen = _final_candidates(plan["harness"]["commit"]) if set(procedures) & set(SPLIT_PROCEDURES) else {}
    replicas = {}
    if admission is not None:
        admission.ensure_unchanged()
        record = pinned.read_pinned_json(admission.record_path, admission.record_sha256)
        from passage_eval.protocol.admission import REPLICA_PROFILES

        if record["protocol"].get("profile") in REPLICA_PROFILES:
            seeds = record["protocol"]["decisions"]["seeds"]["value"]
            replicas = {
                name: {"seed": seeds[0], "sha256": candidate["checkpoint_sha256"]}
                for name, candidate in record["candidates"].items()
            }
    elif plan.get("replica_catalog") is not None:
        from passage_eval.protocol.replicas import load_replica_catalog, replica_identity

        ref = _keys(plan["replica_catalog"], {"path", "sha256"}, "replica_catalog")
        path = _absolute(ref["path"], "replica_catalog.path", allow_test_component=allow_test)
        catalog = load_replica_catalog(path, ref["sha256"], harness_commit=plan["harness"]["commit"])
        _require(len(plan["seeds"]) == 1, "one replica seed is required")
        replicas = {
            name: {
                "seed": plan["seeds"][0],
                "sha256": replica_identity(catalog, name, plan["seeds"][0])["checkpoint_sha256"],
            }
            for name in SPLIT_PROCEDURES
        }
    specs = {}
    for name, procedure in procedures.items():
        p = _keys(procedure, _PROCEDURE_KEYS, f"procedures.{name}")
        _require(isinstance(p["candidate_paths"], list) and p["candidate_paths"], f"{name}: candidate paths required")
        _require(isinstance(p["runtime_binds"], list), f"{name}: runtime_binds must be a list")
        for index, bind in enumerate(p["runtime_binds"]):
            _absolute(bind, f"{name}.runtime_binds[{index}]", allow_test_component=allow_test)
        _hex_list(p["environment_allow_sha256"], f"{name}.environment_allow_sha256", required=True)
        if name in SPLIT_PROCEDURES:
            _require(p["mode"] == "split", f"{name} runs in split mode")
            _require(p["decoder"] in FROZEN_DECODER_ENTRIES[name], f"{name}: the frozen decoder entry is required")
            _require(
                isinstance(p["predictor"], str) and ":" in p["predictor"], f"{name}: a predictor entry is required"
            )
            _require(p["decoder_kwargs"] == {}, f"{name}: the frozen decoder takes no keyword arguments")
            _require(isinstance(p["precision"], str) and p["precision"].strip(), f"{name}: precision is required")
            _hex_list(p["decoder_allow_sha256"], f"{name}.decoder_allow_sha256", required=True)
            _hex_list(p["split_procedure_allow_sha256"], f"{name}.split_procedure_allow_sha256", required=True)
            _require(p["planner_allow_sha256"] == [], f"{name}: planner pins apply to planner mode only")
            checkpoint = (
                replicas.get(name) or ((frozen.get("candidates") or {}).get(name) or {}).get("checkpoint_256") or {}
            )
            _require(
                p["checkpoint_sha256"] == checkpoint.get("sha256"),
                f"{name}: checkpoint differs from final-candidates.json",
            )
            # One frozen checkpoint per candidate: the plan's seeds must be its training seed, or the results of other
            # weights would be attributed to it (final-candidates.json freezes no per-seed checkpoint map).
            _require(
                _is_int(checkpoint.get("seed")) and plan["seeds"] == [checkpoint["seed"]],
                f"{name}: final-candidates.json freezes one checkpoint (training seed {checkpoint.get('seed')!r}); "
                f"the plan's seeds {plan['seeds']} need a frozen checkpoint per seed",
            )
            _require(pinned.is_hex64(p["bundle_sha256"]), f"{name}: the bundle digest is required")
        else:
            _require(p["mode"] == "planner", f"{name} runs in planner mode")
            _require(
                p["predictor"] is None
                and p["decoder"] is None
                and p["decoder_kwargs"] == {}
                and p["precision"] is None,
                f"{name}: planner mode takes no predictor, decoder or precision",
            )
            _hex_list(p["planner_allow_sha256"], f"{name}.planner_allow_sha256", required=True)
            _require(p["decoder_allow_sha256"] == [] and p["split_procedure_allow_sha256"] == [], f"{name}: pins")
            _require(p["checkpoint_sha256"] is None and p["bundle_sha256"] is None, f"{name}: no checkpoint or bundle")
        spec_ref = _keys(p["spec"], {"path", "sha256"}, f"{name}.spec")
        spec_path = _absolute(spec_ref["path"], f"{name}.spec.path", allow_test_component=allow_test)
        data = pinned.read_pinned_json(spec_path, spec_ref["sha256"])
        _require(isinstance(data, Mapping), f"{name}: the spec must be a JSON object")
        try:
            spec = BenchSpec(**data).to_dict()
        except TypeError as exc:
            raise PlanError(f"{name}: invalid spec: {exc}") from exc
        dataset, cell = admissible["dataset"], admissible["cell"]
        budget = BudgetSpec(**spec["budget"]).to_dict() if isinstance(spec["budget"], Mapping) else None
        reference = plan["reference"]
        problems = [
            key
            for key, ok in (
                ("split", spec["split"] == plan["split"] == dataset["split"]),
                ("resolution", spec["resolution"] == dataset["resolution"]),
                ("cost_model", spec["cost_model"] == cell["cost_model"]),
                ("cost_weight", spec["cost_weight"] == cell["cost_weight"]),
                ("task_variant", spec["task_variant"] == cell["task_variant"]),
                ("seeds", list(spec["seeds"]) == plan["seeds"]),
                ("measure_latency", spec["measure_latency"] is True),
                ("latency_warmup", spec["latency_warmup"] == plan["warmup"]["requests_per_warm_pass"]),
                ("latency_repeats", spec["latency_repeats"] in (None, 1)),
                ("quality_repeat_index", spec["quality_repeat_index"] == 0),
                ("budget", budget == plan["budget"]),
                ("reference_policy", spec["reference_policy"] == reference["policy"]),
                ("reference_cohort_id", spec["reference_cohort_id"] == reference["cohort_id"]),
                ("reference_max_expansions", spec["reference_max_expansions"] == reference["max_expansions"]),
                ("dataset_path", spec["dataset_path"] is None),
                ("batch_size", spec["batch_size"] == 1),
            )
            if not ok
        ]
        _require(not problems, f"{name}: the spec differs from the plan in {problems}")
        specs[name] = spec
    return specs


def batch_bounds(plan: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The worst-case wall time of one batch (information only): the sum of every procedure's session bound."""
    from types import SimpleNamespace

    from passage_eval.cli import BenchSpec
    from passage_eval.protocol.batching import BatchPlanError, SupervisedBatchPlan, worst_case_batch_wall_ms

    batching, t = plan["batching"], plan["timing"]
    config = SimpleNamespace(
        deadline_grace_ms=t["deadline_grace_ms"],
        hard_timeout_ms=t["hard_timeout_ms"],
        startup_timeout_s=t["startup_timeout_s"],
        max_restarts=t["max_restarts"],
        timing_profile=t["profile"],
    )
    bounds = {}
    for name in sorted(specs):
        try:
            batch = SupervisedBatchPlan.from_run_settings(
                BenchSpec(**specs[name]),
                config,
                batch_size=plan["partition"]["batch_size"],
                split=plan["procedures"][name]["mode"] == "split",
                fixed_overhead_ms=batching["fixed_overhead_ms"],
                per_request_overhead_ms=batching["per_request_overhead_ms"],
                per_seed_overhead_ms=batching["per_seed_overhead_ms"],
                session=True,
            )
        except (TypeError, ValueError, BatchPlanError) as exc:
            raise PlanError(f"{name}: batch bound: {exc}") from exc
        bounds[name] = worst_case_batch_wall_ms(batch)
    total = sum(bound["worst_case_wall_ms"] for bound in bounds.values())
    limit = batching["lease_or_timeout_ms"]
    return {"per_procedure": bounds, "worst_case_wall_ms": total, "limit_ms": limit, "margin_ms": limit - total}


def admission_sizing(plan: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The dynamic admission sizing of one batch window (``protocol.batching.admission_sizing``)."""
    from passage_eval.protocol.batching import admission_sizing as sizing
    from passage_eval.protocol.contracts import BudgetSpec

    b, t = plan["batching"], plan["timing"]
    procedures = {
        name: {
            "roles": 2 if plan["procedures"][name]["mode"] == "split" else 1,
            "wall_time_ms": BudgetSpec(**specs[name]["budget"]).wall_time_ms,
            "observed_request_ms": b["observed_request_ms"][name],
        }
        for name in specs
    }
    try:
        return sizing(
            lease_ms=b["lease_or_timeout_ms"],
            batch_size=plan["partition"]["batch_size"],
            warmup_requests=plan["warmup"]["requests_per_warm_pass"],
            procedures=procedures,
            fixed_overhead_ms=b["fixed_overhead_ms"],
            per_seed_overhead_ms=b["per_seed_overhead_ms"],
            per_request_overhead_ms=b["per_request_overhead_ms"],
            persistence_ms=b["persistence_ms"],
            safety_margin=b["safety_margin"],
            deadline_grace_ms=t["deadline_grace_ms"],
            hard_timeout_ms=t["hard_timeout_ms"],
            startup_timeout_s=t["startup_timeout_s"],
        )
    except (TypeError, ValueError) as exc:
        raise PlanError(f"batching: {exc}") from exc


def _check_batching(plan: dict[str, Any], specs: dict[str, dict]) -> dict[str, Any]:
    from passage_eval.protocol.batching import MIN_SAFETY_MARGIN

    batching = _keys(plan["batching"], _SUBKEYS["batching"], "batching")
    _require(_is_number(batching["lease_or_timeout_ms"]) and batching["lease_or_timeout_ms"] > 0, "a batch bound")
    for key in ("fixed_overhead_ms", "per_seed_overhead_ms", "per_request_overhead_ms", "persistence_ms"):
        _require(_is_number(batching[key]) and batching[key] >= 0, f"batching.{key} must be nonnegative")
    observed = batching["observed_request_ms"]
    _require(
        isinstance(observed, Mapping)
        and sorted(observed) == sorted(plan["procedures"])
        and all(_is_number(value) and value > 0 for value in observed.values()),
        "batching.observed_request_ms must give a positive observed warm request time for every procedure",
    )
    _require(
        _is_number(batching["safety_margin"]) and batching["safety_margin"] >= MIN_SAFETY_MARGIN,
        f"batching.safety_margin must be at least {MIN_SAFETY_MARGIN:g} (R7)",
    )
    _require(
        _is_int(plan["timing"]["max_restarts"]) and plan["timing"]["max_restarts"] >= plan["partition"]["batch_size"],
        "timing.max_restarts must be at least the batch size, so that the restart cap never decides a task status",
    )
    _require(len(plan["seeds"]) == 1, "a campaign plan runs one seed (one campaign per seed)")
    sizing = admission_sizing(plan, specs)
    _require(
        sizing["margin_ok"],
        f"the window {sizing['lease_ms']:.0f} ms does not hold {sizing['safety_margin']:g} times the observed batch "
        f"time {sizing['expected_batch_ms']:.0f} ms; use smaller batches or a longer window",
    )
    _require(
        sizing["sanity_ok"],
        "a procedure's share of a fresh window cannot engage one request after a full start of its workers "
        f"({ {name: round(row['sanity_floor_ms']) for name, row in sizing['per_procedure'].items()} } ms needed, "
        f"{round(sizing['lease_ms'] / len(specs))} ms per procedure); the batch could be deferred forever",
    )
    return sizing


def _check_harness(plan: dict[str, Any]) -> None:
    from passage_eval import cli

    harness = _keys(plan["harness"], _SUBKEYS["harness"], "harness")
    _require(pinned.is_hex40(harness["commit"]), "harness.commit must be a 40-hex commit")
    _require(pinned.is_hex64(harness["source_sha256"]), "harness.source_sha256 must be a SHA-256")
    _require(cli._source_hash() == harness["source_sha256"], "the running harness source differs from the plan's pin")
    _require(cli._checkout_commit() == harness["commit"], "the running harness commit differs from the plan's pin")


def _check_against_admission(
    plan: dict[str, Any], grant: Any, chain: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]
) -> None:
    """An admitted test plan repeats the sealed decisions and candidate identities of its admission record."""
    record = pinned.read_pinned_json(grant.record_path, grant.record_sha256)
    decisions = {name: entry["value"] for name, entry in record["protocol"]["decisions"].items()}
    t, w, order, batching = plan["timing"], plan["warmup"], plan["order"], plan["batching"]
    # The future public-test authority is deliberately narrow: roster S is
    # evaluated only as the descriptive 2026 Seg/Reg comparison on an edge
    # target.  Keep legacy/synthetic smaller rosters testable without making
    # them an authority to broaden S.
    from passage_eval.protocol.admission import EDGE_TARGETS, S_ROSTER_SIZE

    selection = decisions.get("roster_selection")
    if isinstance(selection, Mapping) and selection.get("n") == S_ROSTER_SIZE:
        _require(set(plan["procedures"]) == {"seg", "reg"}, "roster S admits exactly the Seg and Reg procedures")
        _require(sorted(order["procedures"]) == ["reg", "seg"], "roster S orders exactly the Seg and Reg procedures")
        from passage_eval.protocol.admission import REPLICA_PROFILES

        if record["protocol"].get("profile") in REPLICA_PROFILES:
            _require(
                plan["seeds"] == decisions["seeds"] and len(plan["seeds"]) == 1,
                "roster S runs only the one seed of its admitted replica record",
            )
        else:
            _require(plan["seeds"] == [2026], "roster S is descriptive for training seed 2026 only")
        _require(plan["target"] in EDGE_TARGETS, "roster S runs on Thor, Orin or Tiger only")
    checks = {
        "protocol_record_sha256": plan["protocol_record_sha256"] == grant.record_sha256,
        "purpose": plan["purpose"] == "verified-candidate-run",
        "harness": {"commit": plan["harness"]["commit"], "source_sha256": plan["harness"]["source_sha256"]}
        == record["harness"],
        "admissible_roster": plan["rosters"]["admissible"]["roster_sha256"] == grant.admissible_roster_sha256,
        "source_roster": plan["rosters"]["source_roster_sha256"] == grant.source_roster_sha256,
        "chain": chain["chain_sha256"] == grant.report.admitted.get("chain_sha256"),
        "budget": plan["budget"] == decisions["budget"],
        "timing_profile": t["profile"] == decisions["timing_profile"],
        "timing_boundary_id": t["boundary_id"] == decisions["timing_boundary_id"],
        "deadline_grace_ms": t["deadline_grace_ms"] == decisions["deadline_grace_ms"],
        "hard_timeout_ms": t["hard_timeout_ms"] == decisions["hard_timeout_ms"],
        "startup_timeout_s": t["startup_timeout_s"] == decisions["startup_timeout_s"],
        "max_restarts": t["max_restarts"] == decisions["max_restarts"],
        "worker_rlimits": t["rlimits"] == decisions["worker_rlimits"],
        "reference": (plan["reference"]["policy"], plan["reference"]["cohort_id"], plan["reference"]["max_expansions"])
        == (decisions["reference_policy"], decisions["reference_cohort_id"], decisions["reference_max_expansions"]),
        "warmup": (w["salt"], w["k"]) == (decisions["warmup"]["salt"], decisions["warmup"]["k"]),
        "paired_order": (order["granularity"], order["schedule_seed"], sorted(order["procedures"]))
        == (
            decisions["paired_order"]["granularity"],
            decisions["paired_order"]["schedule_seed"],
            sorted(decisions["paired_order"]["procedures"]),
        ),
        "batching": {"batch_size": plan["partition"]["batch_size"], **batching} == decisions["batching"],
        "seeds": plan["seeds"] == decisions["seeds"],
        "censoring_rule": plan["aggregation"]["censoring_rule_id"] == decisions["censoring_rule"]["id"],
        "min_warm_requests": plan["aggregation"]["min_warm_requests"] == decisions["min_warm_requests"],
        "targets": plan["target"] in decisions["targets"],
        "infrastructure_relaunch_policy": plan["infrastructure_relaunch_policy"]
        == decisions["infrastructure_relaunch_policy"],
        "r7_policy": plan["r7_policy"] == decisions["r7_policy"],
    }
    checks.update(_candidate_bindings(plan, record, specs))
    differing = sorted(name for name, ok in checks.items() if not ok)
    _require(not differing, f"the test plan differs from its sealed admission record in {differing}")


def _candidate_bindings(
    plan: Mapping[str, Any], record: Mapping[str, Any], specs: Mapping[str, Mapping[str, Any]]
) -> dict[str, bool]:
    """Every identity of an admitted procedure equals what gate G6 verified for the record.

    Seg and Reg: predictor, decoder entry, checkpoint, the bundle digest of the plan's target, and the decoder tree
    pin, which must be exactly the record's tree. A*: one of the harness planner entries (the package of the frozen
    decoders), pinned by exactly that same tree, with no checkpoint or bundle.
    """
    from passage_eval.protocol.admission import CANONICAL_PLANNER_ENTRIES

    checks: dict[str, bool] = {}
    for name in SPLIT_PROCEDURES:
        if name not in plan["procedures"]:
            continue
        p, candidate = plan["procedures"][name], record["candidates"][name]
        target = candidate["targets"].get(plan["target"]) or {}
        checks.update(
            {
                f"candidates.{name}.checkpoint": p["checkpoint_sha256"] == candidate["checkpoint_sha256"],
                f"candidates.{name}.predictor": p["predictor"] == candidate["predictor"],
                f"candidates.{name}.decoder": p["decoder"] == candidate["decoder"],
                f"candidates.{name}.decoder_tree": p["decoder_allow_sha256"] == [candidate["decoder_tree_sha256"]],
                f"candidates.{name}.bundle": pinned.is_hex64(target.get("bundle_sha256"))
                and p["bundle_sha256"] == target["bundle_sha256"],
            }
        )
    if "astar" in plan["procedures"]:
        p = plan["procedures"]["astar"]
        tree = record["candidates"]["seg"]["decoder_tree_sha256"]
        same_package = {entry.split(".")[0] for entry in CANONICAL_PLANNER_ENTRIES} == {
            record["candidates"][name]["decoder"].split(".")[0] for name in SPLIT_PROCEDURES
        }
        checks.update(
            {
                "astar.planner": specs["astar"]["inference"] in CANONICAL_PLANNER_ENTRIES and same_package,
                "astar.planner_tree": p["planner_allow_sha256"] == [tree]
                and record["candidates"]["reg"]["decoder_tree_sha256"] == tree,
            }
        )
    return checks


def read_plan(path: str | Path, expected_sha256: str, *, allow_test: bool = False) -> CampaignPlan:
    """Check a pinned campaign plan and every control file it binds; raise on the first problem.

    The test split needs ``allow_test=True`` and an all-GREEN admission record named by the plan; the split and the
    target are checked before any roster, spec or partition file is read.
    """
    raw = pinned.read_pinned(path, expected_sha256)
    plan = pinned.strict_json_loads(raw, name=Path(path).name)
    optional = {"replica_catalog"} if isinstance(plan, Mapping) and "replica_catalog" in plan else set()
    plan = _keys(plan, _PLAN_KEYS | optional, "plan")
    _require(plan["schema"] == PLAN_SCHEMA, f"schema must be {PLAN_SCHEMA!r}")
    _require(isinstance(plan["campaign_id"], str) and plan["campaign_id"].strip(), "campaign_id is required")
    _require(pinned.is_hex64(plan["protocol_record_sha256"]), "protocol_record_sha256 must be a SHA-256")
    _require(plan["purpose"] in PURPOSES, f"purpose must be one of {PURPOSES}")
    for key, expected in _SUBKEYS.items():
        if key not in {"budget", "timing", "warmup", "order", "batching", "aggregation", "harness"}:
            plan[key] = _keys(plan[key], expected, key)
    _require(isinstance(plan["procedures"], Mapping) and plan["procedures"], "procedures must be a nonempty object")
    for name, procedure in plan["procedures"].items():
        _keys(procedure, _PROCEDURE_KEYS, f"procedures.{name}")
    grant = _check_scope(plan, allow_test)  # before any roster or partition file is read
    admitted = grant is not None
    _check_target(plan)
    _check_harness(plan)
    _check_paths(plan, admitted)
    admissible = _read_roster_file(plan["rosters"]["admissible"], "rosters.admissible", admitted)
    refuse_test_roster(
        admissible, allow_test=admitted, admission=grant, context="the campaign planner", require_admitted_roster=True
    )
    _require(admissible["dataset"]["split"] == plan["split"], "the admissible roster's split differs from the plan")
    _require(pinned.is_hex64(plan["rosters"]["source_roster_sha256"]), "rosters.source_roster_sha256 is required")
    check_source_paths(plan, admissible)
    chain = _check_chain(plan, admissible, admitted)
    _check_limits(plan)
    warmup_roster = _check_warmup(plan, admissible, admitted)
    _check_order_and_aggregation(plan)
    specs = _check_procedures(plan, admissible, admitted, grant)
    p = plan["partition"]
    _require(_is_int(p["batch_size"]) and 1 <= p["batch_size"] <= partition_io.MAX_BATCH, "partition.batch_size")
    _check_batching(plan, specs)
    admissible_entry = plan["rosters"]["admissible"]
    manifest = partition_io.verify_partition(
        admissible_entry["path"],
        admissible_entry["file_sha256"],
        admissible_entry["roster_sha256"],
        p["dir"],
        p["manifest_sha256"],
        allow_test=admitted,
        admission=grant,
    )
    _require(manifest["batch_size"] == p["batch_size"], "the partition's batch size differs from the plan")
    if admitted:
        _check_against_admission(plan, grant, chain, specs)
    elif plan["purpose"] != "verified-candidate-run":
        _require(plan["split"] == "validation", "a diagnostic campaign runs on validation only")
    return CampaignPlan(plan, expected_sha256, admissible, warmup_roster, chain, manifest, specs, grant)
