"""Reference admission inventory ``passage-reference-admission/v1``.

Before any candidate runs, the reference of every task of a sealed source roster
is qualified under one explicit policy, and each task is put in one category:

* ``admissible``: a qualified exact reference exists; the task enters the
  admissible roster, which every candidate runs with the same limits;
* ``excluded``: the reference cannot be qualified; the task stays in the source
  denominator and is listed with its identifier and reason; no other scene
  replaces it;
* ``halt``: an input inconsistency, a duplicate identifier, a scene that cannot
  be read or matched, a reference that cannot be determined (search exhausted), a
  reason the category table does not know, a disagreement of the second
  implementation, the no-path exclusion of a task whose published label is a
  legal path, or (released-label policy only) a qualified reference whose row the
  execution loader refuses; the preparation stops for diagnosis and no admissible
  roster is derived.

``load_admission_scenes`` reads the pinned rows like the execution loader, but a
row whose stored route or cost the execution loader refuses (``scene_from_row``)
is still decoded, with its label unvalidated, so that the qualifiers turn the
defect into a per-task reason (``stored_path_illegal``,
``stored_cost_missing_or_invalid``, ``stored_path_optimum_unassessable``) instead
of stopping the whole inventory.

Independent analytic policy (judge R1, R6). The reference of every task is the
analytic A* optimum under one search limit; the published labels are never the
reference. Their defects (``source_label_reasons``, and a label the validating
loader refuses) are dataset findings (``label_findings``, published with the
graph tolerance) that neither exclude nor halt: the supervisor and the scorer load
such rows with the tolerant execution loader (``io.roster.load_roster_scenes``
with ``tolerant_labels``), which the policy records as ``execution_loader_labels:
tolerant``. Only a task without analytic path is excluded. The policy must carry
exclusion caps and must map an exhausted search to ``halt``. Under the
released-label policy a qualified reference whose row the validating execution
loader refuses halts with ``execution_loader_refuses_label``.

Second implementation. Every task whose A* search ended (exact optimum or no
path) is recomputed by an independent 8-connected Dijkstra search
(``protocol.second_reference``); a cost outside the graph tolerance, or a path
where the A* found none (or the reverse), halts with
``second_reference_disagreement``.

Exclusion caps (``exclusion_caps``, proposed 0.5 % of the source tasks in total
and 5 % of a geographic block when the block has at least two exclusions): the
inventory reports the excluded counts by block, and a breach makes its outcome
``halt``. A null cap (released-label diagnostics only) applies none.

The records form a SHA-256 chain: the policy (reference policy, cohort, search
limit and category table, bound to the source roster digest) is hashed first;
the inventory contains the policy and its digest; the admissible roster records
the inventory and policy digests under its own ``roster_sha256``.

What is not decided here. The mapping from qualification reasons to
``excluded`` or ``halt`` belongs to the protocol author.
``PROPOSED_CATEGORY_TABLE`` is a proposed default, not a sealed decision; its
status is part of the hashed policy. Candidate outputs are never an input: a
candidate failure on an admissible task stays in the admissible denominator
with the canonical penalty of the scorer. The category table accepts only the
reasons the reference qualifiers emit, so no candidate outcome can be mapped to
an exclusion; ``validate_inventory`` re-derives every task's category from its
recorded reasons and the policy's table, and the exclusion and halt tables from
the task records.

The test split is refused (``protocol.guards``) by ``admission_policy``,
``build_reference_inventory`` and the command, before any scene is loaded,
unless they get ``allow_test=True`` and an ``admission`` grant: a
``passage-test-admission/v1`` record whose six gates are GREEN
(``protocol.admission``). The command obtains the grant only from
``guards.test_split_allowed`` (``--allow-test-split`` with a
``--test-admission-record`` that checks all GREEN). The record checks
(``validate_inventory``, ``admissible_roster``, ``admission_chain``) load no
scene.

``--skip-file-hashes`` (``verify_file_hashes=False``) is recorded in the
inventory as ``source_file_hashes: skipped``: the stored labels that decide
exclusions are then unverified, so ``admissible_roster`` and
``admission_chain`` refuse such an inventory and the command derives no
admissible roster from it (exit code 2). The command also exits with code 2,
without a traceback, when no task is admissible.

Command line (validation rosters; the test split needs an all-GREEN test
admission record, or a preparation approval that names the
``reference_inventory`` reader)::

    python -I -m passage_eval.protocol.reference_admission \\
      --roster ROSTER.json --roster-sha256 HEX --dataset-root DIR \\
      --reference-policy released_labels --reference-max-expansions 65536 \\
      --output-dir private/reference-admission
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

from passage_eval.io import roster as roster_io
from passage_eval.io.dataset import scene_from_row, scene_from_row_unvalidated_label
from passage_eval.protocol.contracts import GraphSpec
from passage_eval.protocol.guards import add_test_admission_arguments, refuse_test_roster, test_split_allowed
from passage_eval.protocol.planner import Scene
from passage_eval.protocol.second_reference import IMPLEMENTATION as SECOND_IMPLEMENTATION
from passage_eval.protocol.second_reference import second_reference_check

SCHEMA = "passage-reference-admission/v1"
POLICY_SCHEMA = "passage-reference-admission-policy/v1"
CATEGORY_TABLE_SCHEMA = "passage-reference-category-table/v1"
CHAIN_SCHEMA = "passage-reference-admission-chain/v1"
REFERENCE_POLICIES = ("released_labels", "independent_analytic_reference")
CATEGORIES = ("admissible", "excluded", "halt")
PROPOSED_STATUS = "proposed_not_sealed"
# Every reason the qualifiers of passage.audit emit for a reference that is not qualified. A category table maps
# exactly these, so no other outcome (in particular no candidate outcome) can create an exclusion.
QUALIFICATION_REASONS = (
    "stored_path_illegal",
    "stored_cost_disagreement",
    "stored_cost_missing_or_invalid",
    "stored_path_suboptimal",
    "stored_path_optimum_unassessable",
    "reference_no_path",
    "reference_budget_exhausted",
    "independent_reference_no_path",
    "independent_reference_budget_exhausted",
)
# Raised by this module itself. Input inconsistencies always halt; they are not configurable.
# ``execution_loader_refuses_label``: the reference qualifies but ``scene_from_row`` refuses the row's stored label,
# so the supervisor could not load the task (independent analytic policy only; see the module docstring).
HALT_REASONS = (
    "scene_missing",
    "scene_unreadable",
    "duplicate_sample_id",
    "input_identity_mismatch",
    "qualification_error",
    "inconsistent_qualification_record",
    "execution_loader_refuses_label",
    "second_reference_disagreement",
    "no_path_with_legal_label",
)
NO_PATH_REASONS = ("reference_no_path", "independent_reference_no_path")
CAPS_SCHEMA = "passage-reference-exclusion-caps/v1"
_CAP_KEYS = ("max_total_fraction", "max_block_fraction", "min_block_exclusions")
# PROPOSED, NOT SEALED: the judge's caps (R1). The protocol author seals caps with these keys before the inventory.
PROPOSED_EXCLUSION_CAPS: dict[str, Any] = {
    "schema": CAPS_SCHEMA,
    "status": PROPOSED_STATUS,
    "max_total_fraction": 0.005,
    "max_block_fraction": 0.05,
    "min_block_exclusions": 2,
}
#: The graph tolerance of the cell (``GraphSpec``): the second implementation and the label findings use it.
_GRAPH_TOLERANCE = {"rtol": GraphSpec().rtol, "atol": GraphSpec().atol}
_SECOND_REFERENCE = {
    "implementation": SECOND_IMPLEMENTATION,
    "coverage": "every task whose A* search ended with an exact optimum or no path",
    **_GRAPH_TOLERANCE,
}
# Appended to the reasons of a task when a qualifier emitted a reason that the category table does not know.
UNKNOWN_REASON = "unknown_reason"
# Whether the inventory's loader re-hashed the source files. Only a "verified" inventory can be sealed.
SOURCE_FILE_HASHES = ("verified", "skipped")
_POLICY_DATASET_FIELDS = ("repo", "commit", "config", "split", "resolution")
# PROPOSED, NOT SEALED. A reference that cannot be qualified excludes its task; a reference that cannot be
# determined (search exhausted, which a limit of at least N*N expansions makes impossible) halts. The protocol
# author seals this table, or replaces it, before the inventory.
PROPOSED_CATEGORY_TABLE: dict[str, Any] = {
    "schema": CATEGORY_TABLE_SCHEMA,
    "status": PROPOSED_STATUS,
    "note": (
        "Proposed default, not sealed: a non-qualifiable reference excludes its task (listed with its reason, "
        "counted in the source denominator, never replaced); an undeterminable reference halts. Input "
        "inconsistencies, duplicate identifiers, unreadable scenes and unknown reasons always halt."
    ),
    "reasons": {
        "stored_path_illegal": "excluded",
        "stored_cost_disagreement": "excluded",
        "stored_cost_missing_or_invalid": "excluded",
        "stored_path_suboptimal": "excluded",
        "stored_path_optimum_unassessable": "excluded",
        "reference_no_path": "excluded",
        "reference_budget_exhausted": "halt",
        "independent_reference_no_path": "excluded",
        "independent_reference_budget_exhausted": "halt",
    },
}


@dataclass(frozen=True)
class AdmissionScene:
    """One source row as read for reference admission.

    ``loader_error`` is ``None`` when the execution loader (``scene_from_row``) accepts the row, and ``scene`` is
    then its scene. Otherwise ``loader_error`` says why it refuses the row, and ``scene`` holds the same inputs with
    the stored label unvalidated (``scene_from_row_unvalidated_label``), or ``None`` when the inputs themselves
    cannot be decoded. ``source_files_verified`` is false when the loader skipped re-hashing the source files
    (``verify_file_hashes=False``): the stored labels that decide exclusions are then unverified.
    """

    sample_id: str
    scene: Scene | None
    loader_error: str | None = None
    source_files_verified: bool = True


SceneLoader = Callable[[Mapping[str, Any]], Iterable["Scene | AdmissionScene"]]


class ReferenceAdmissionHalt(ValueError):
    """The preparation stops for diagnosis; no admissible roster may be derived."""


def _digest(record: Mapping[str, Any], field: str) -> str:
    return roster_io.sha256_hex(roster_io.canonical_json({key: value for key, value in record.items() if key != field}))


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _is_hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_category_table(table: Any) -> dict[str, Any]:
    """An explicit mapping of every qualification reason to ``excluded`` or ``halt``, with its sealing status."""
    if not isinstance(table, Mapping) or table.get("schema") != CATEGORY_TABLE_SCHEMA:
        raise ValueError(f"a {CATEGORY_TABLE_SCHEMA} category table is required")
    if not set(table) <= {"schema", "status", "note", "reasons"} or not {"schema", "status", "reasons"} <= set(table):
        raise ValueError("a category table has exactly schema, status, reasons and an optional note")
    if not isinstance(table["status"], str) or not table["status"].strip():
        raise ValueError("a category table needs a sealing status")
    if "note" in table and not isinstance(table["note"], str):
        raise ValueError("the category table note must be text")
    reasons = table["reasons"]
    if not isinstance(reasons, Mapping) or set(reasons) != set(QUALIFICATION_REASONS):
        raise ValueError(f"the category table must map exactly the qualification reasons {list(QUALIFICATION_REASONS)}")
    bad = sorted(reason for reason, category in reasons.items() if category not in {"excluded", "halt"})
    if bad:
        raise ValueError(f"a reference that is not qualified can only be excluded or halt: {bad}")
    return copy.deepcopy(dict(table))


def validate_exclusion_caps(caps: Any) -> dict[str, Any]:
    """Exclusion caps: a total fraction and a per-block fraction in [0, 1), a minimum count per block, a status."""
    if not isinstance(caps, Mapping) or caps.get("schema") != CAPS_SCHEMA:
        raise ValueError(f"a {CAPS_SCHEMA} record is required")
    if set(caps) != {"schema", "status", *_CAP_KEYS}:
        raise ValueError(f"exclusion caps have exactly schema, status and {list(_CAP_KEYS)}")
    if not isinstance(caps["status"], str) or not caps["status"].strip():
        raise ValueError("exclusion caps need a sealing status")
    for key in _CAP_KEYS[:2]:
        value = caps[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < 1:
            raise ValueError(f"{key} must lie in [0, 1)")
    if not _is_int(caps["min_block_exclusions"]) or caps["min_block_exclusions"] < 1:
        raise ValueError("min_block_exclusions must be a positive integer")
    return copy.deepcopy(dict(caps))


def admission_policy(
    source_roster: Mapping[str, Any],
    *,
    reference_policy: str,
    reference_max_expansions: int,
    reference_cohort_id: str | None = None,
    category_table: Mapping[str, Any] | None = None,
    exclusion_caps: Mapping[str, Any] | None = None,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """The hashed admission policy for one source roster. There is no default reference policy.

    ``reference_max_expansions`` must be at least ``N*N`` for the roster's resolution, so the exact search cannot
    be exhausted. The independent analytic policy needs a named cohort, exclusion caps (``PROPOSED_EXCLUSION_CAPS``
    is the proposal) and a table that halts on an exhausted search; the released-label policy refuses a cohort and
    applies caps only when given. Without ``category_table`` the proposed default is used and recorded as
    ``proposed_not_sealed``. A roster of the test split raises ``TestSplitRefused`` unless ``allow_test`` is true
    and ``admission`` is an all-GREEN test admission grant or a preparation grant that names the
    ``reference_inventory`` reader (``guards.test_split_allowed``).
    """
    roster = roster_io.validate_roster(source_roster)
    refuse_test_roster(
        roster, allow_test=allow_test, admission=admission, context="reference admission", reader="reference_inventory"
    )
    if reference_policy not in REFERENCE_POLICIES:
        raise ValueError(f"reference_policy must be one of {list(REFERENCE_POLICIES)}; there is no default")
    if reference_policy == "independent_analytic_reference":
        if not isinstance(reference_cohort_id, str) or not reference_cohort_id.strip():
            raise ValueError("the independent analytic reference policy needs an explicit reference_cohort_id")
    elif reference_cohort_id is not None:
        raise ValueError("released_labels does not use a reference cohort; pass reference_cohort_id=None")
    resolution = roster["dataset"]["resolution"]
    minimum = resolution * resolution
    if not _is_int(reference_max_expansions) or reference_max_expansions < minimum:
        raise ValueError(
            f"reference_max_expansions must be an integer of at least N*N = {minimum} at resolution {resolution}, "
            "so that the exact reference search cannot be exhausted"
        )
    table = validate_category_table(PROPOSED_CATEGORY_TABLE if category_table is None else category_table)
    independent = reference_policy == "independent_analytic_reference"
    if independent and table["reasons"]["independent_reference_budget_exhausted"] != "halt":
        raise ValueError("under the independent analytic policy an exhausted search halts the preparation")
    if independent and exclusion_caps is None:
        raise ValueError("the independent analytic policy needs exclusion caps (PROPOSED_EXCLUSION_CAPS is proposed)")
    caps = None if exclusion_caps is None else validate_exclusion_caps(exclusion_caps)
    policy = {
        "schema": POLICY_SCHEMA,
        "source_roster_sha256": roster["roster_sha256"],
        "benchmark_id": roster["benchmark_id"],
        "dataset": {key: roster["dataset"][key] for key in _POLICY_DATASET_FIELDS},
        "cell": dict(roster["cell"]),
        "reference_policy": reference_policy,
        "reference_cohort_id": reference_cohort_id,
        "reference_max_expansions": int(reference_max_expansions),
        "minimum_reference_max_expansions": minimum,
        "category_table": table,
        "category_table_status": table["status"],
        "input_error_category": "halt",
        "unknown_reason_category": "halt",
        "candidate_outputs_used": False,
        "exclusion_caps": caps,
        "second_reference": dict(_SECOND_REFERENCE),
        "label_tolerance": dict(_GRAPH_TOLERANCE),
        "execution_loader_labels": "tolerant" if independent else "validated",
    }
    policy["policy_sha256"] = _digest(policy, "policy_sha256")
    return policy


def validate_policy(policy: Any) -> dict[str, Any]:
    if not isinstance(policy, Mapping) or policy.get("schema") != POLICY_SCHEMA:
        raise ValueError(f"a {POLICY_SCHEMA} record is required")
    policy = copy.deepcopy(dict(policy))
    if policy.get("policy_sha256") != _digest(policy, "policy_sha256"):
        raise ValueError("policy_sha256 differs from the recomputed digest")
    if policy.get("reference_policy") not in REFERENCE_POLICIES:
        raise ValueError("unknown reference policy")
    cohort = policy.get("reference_cohort_id")
    if (policy["reference_policy"] == "independent_analytic_reference") != (
        isinstance(cohort, str) and bool(cohort.strip())
    ) or (policy["reference_policy"] == "released_labels" and cohort is not None):
        raise ValueError("the reference cohort does not match the reference policy")
    table = validate_category_table(policy.get("category_table"))
    independent = policy["reference_policy"] == "independent_analytic_reference"
    caps = policy.get("exclusion_caps")
    if caps is not None:
        validate_exclusion_caps(caps)
    if (
        policy.get("category_table_status") != table["status"]
        or policy.get("input_error_category") != "halt"
        or policy.get("unknown_reason_category") != "halt"
        or policy.get("candidate_outputs_used") is not False
        or not _is_int(policy.get("reference_max_expansions"))
        or policy["reference_max_expansions"] < policy.get("minimum_reference_max_expansions", float("inf"))
        or policy.get("second_reference") != _SECOND_REFERENCE
        or policy.get("label_tolerance") != _GRAPH_TOLERANCE
        or policy.get("execution_loader_labels") != ("tolerant" if independent else "validated")
        or (independent and (caps is None or table["reasons"]["independent_reference_budget_exhausted"] != "halt"))
    ):
        raise ValueError("the admission policy fields are inconsistent")
    return policy


def _qualify(scene: Scene, policy: Mapping[str, Any]) -> dict[str, Any]:
    """The same qualifiers, with the same arguments, as the evaluator's pre-candidate qualification."""
    from passage.audit import independently_qualified_reference_for_scene, qualified_reference_for_scene

    if policy["reference_policy"] == "independent_analytic_reference":
        return independently_qualified_reference_for_scene(
            scene, cohort_id=policy["reference_cohort_id"], max_expansions=policy["reference_max_expansions"]
        )
    return qualified_reference_for_scene(scene, max_expansions=policy["reference_max_expansions"])


def _categorize_reasons(reasons: Sequence[str], table: Mapping[str, Any]) -> tuple[str, str, list[str]]:
    """Category, determining reason and recorded reasons of a reference that is not qualified.

    The category is ``halt`` if any reason maps to ``halt`` or is unknown to the table, else ``excluded``; the
    determining reason is the first reason of that category. An unknown reason adds ``unknown_reason`` at the end.
    """
    categories = [table["reasons"].get(reason, "halt") for reason in reasons]
    category = "halt" if "halt" in categories else "excluded"
    reason = next(reason for reason, chosen in zip(reasons, categories) if chosen == category)
    unknown = any(reason not in table["reasons"] for reason in reasons)
    return category, reason, [*reasons, UNKNOWN_REASON] if unknown else list(reasons)


def _categorize(record: Mapping[str, Any], table: Mapping[str, Any]) -> tuple[str, str | None, list[str]]:
    reasons = record.get("qualification_reasons")
    reasons = [] if reasons is None else reasons
    inconsistent = ("halt", "inconsistent_qualification_record", ["inconsistent_qualification_record"])
    if not isinstance(reasons, list):
        return inconsistent
    if record.get("qualified") is True:
        if reasons or record.get("reference_status") != "exact" or not _is_hex64(record.get("reference_id")):
            return inconsistent
        return "admissible", None, []
    # A qualifier reason that reuses a name of this module could not be told apart from it in the record.
    if not reasons or any(not isinstance(r, str) or r in HALT_REASONS or r == UNKNOWN_REASON for r in reasons):
        return inconsistent
    return _categorize_reasons(reasons, table)


def _detail(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {str(exc)[:300]}"


def load_admission_scenes(
    roster: Mapping[str, Any],
    dataset_path: str | Path,
    *,
    require_root_receipt: bool = True,
    verify_file_hashes: bool = True,
) -> list[AdmissionScene]:
    """Read the roster's pinned rows for reference admission, one ``AdmissionScene`` per task in roster order.

    The roster, the dataset root and the files are checked as by ``load_roster_scenes`` (any failure raises before
    a row is decoded). Each row is decoded by the execution loader ``scene_from_row``; a row it refuses is decoded
    again with its stored label unvalidated, so that the reference qualifiers classify a label defect per task.
    Input identities are checked by ``build_reference_inventory``, per task. ``verify_file_hashes=False`` skips
    re-hashing the source files; every returned item then says so, and the inventory records it.
    """
    roster, rows = roster_io.open_roster_rows(
        roster, dataset_path, require_root_receipt=require_root_receipt, verify_file_hashes=verify_file_hashes
    )
    dataset, cell = roster["dataset"], roster["cell"]
    options = {
        "resolution": dataset["resolution"],
        "cost_model": cell["cost_model"],
        "split": dataset["split"],
        "cost_weight": cell["cost_weight"],
        "task_variant": cell["task_variant"],
    }
    loaded = []
    with rows as pairs:
        for task, row in pairs:
            try:
                loaded.append(
                    AdmissionScene(
                        task["sample_id"], scene_from_row(row, **options), source_files_verified=verify_file_hashes
                    )
                )
                continue
            except ValueError as exc:
                error = _detail(exc)
            try:
                scene = scene_from_row_unvalidated_label(row, **options)
            except ValueError:
                scene = None
            loaded.append(AdmissionScene(task["sample_id"], scene, error, source_files_verified=verify_file_hashes))
    return loaded


def _check_policy_roster(policy: Mapping[str, Any], roster: Mapping[str, Any]) -> None:
    """The policy names this roster, its cell and the ``N*N`` minimum of its resolution."""
    resolution = roster["dataset"]["resolution"]
    if (
        policy["source_roster_sha256"] != roster["roster_sha256"]
        or policy.get("benchmark_id") != roster["benchmark_id"]
        or policy.get("dataset") != {key: roster["dataset"][key] for key in _POLICY_DATASET_FIELDS}
        or policy.get("cell") != dict(roster["cell"])
        or policy.get("minimum_reference_max_expansions") != resolution * resolution
    ):
        raise ValueError("the admission policy was sealed for another source roster")


def _listing(records: Sequence[Mapping[str, Any]], category: str) -> list[dict[str, Any]]:
    keys = ("rank", "sample_id", "input_identity_sha256", "geographic_block", "status", "reason", "reasons")
    return [{key: record.get(key) for key in keys} for record in records if record.get("category") == category]


def _excluded_reason_counts(records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(r.get("reason") for r in records if r.get("category") == "excluded").items()))


def exclusion_cap_check(
    records: Sequence[Mapping[str, Any]], caps: Mapping[str, Any] | None, block_sizes: Mapping[str, int]
) -> dict[str, Any] | None:
    """The exclusion counts in total and by geographic block, and the breaches of ``caps`` (``None``: no caps).

    A total breach is more excluded tasks than ``max_total_fraction`` of the source tasks. A block breach is a block
    with at least ``min_block_exclusions`` excluded tasks and more than ``max_block_fraction`` of its tasks. An
    excluded task without block (incomplete block table) is a breach: its block share cannot be bounded.
    """
    if caps is None:
        return None
    excluded = [record for record in records if record.get("category") == "excluded"]
    n_source, n_excluded = len(records), len(excluded)
    by_block = Counter(record.get("geographic_block") for record in excluded)
    breaches = []
    if n_excluded > caps["max_total_fraction"] * n_source:
        breaches.append({"scope": "total", "n_excluded": n_excluded, "n_tasks": n_source})
    for block in sorted(by_block, key=lambda value: (value is None, str(value))):
        count = by_block[block]
        if block is None:
            breaches.append({"scope": "block", "block": None, "n_excluded": count, "n_tasks": None})
        elif count >= caps["min_block_exclusions"] and count > caps["max_block_fraction"] * block_sizes[block]:
            breaches.append({"scope": "block", "block": block, "n_excluded": count, "n_tasks": block_sizes[block]})
    return {
        "caps": {key: caps[key] for key in ("status", *_CAP_KEYS)},
        "n_source": n_source,
        "n_excluded": n_excluded,
        "excluded_fraction": n_excluded / n_source if n_source else 0.0,
        "excluded_by_block": {
            str(block): {"n_excluded": by_block[block], "n_tasks": block_sizes.get(block) if block else None}
            for block in sorted(by_block, key=lambda value: (value is None, str(value)))
        },
        "breaches": breaches,
        "within_caps": not breaches,
    }


def label_findings(records: Sequence[Mapping[str, Any]], policy: Mapping[str, Any]) -> dict[str, Any]:
    """Defects of the published labels that are dataset findings, not exclusions (independent analytic policy).

    A finding is a qualifier reason about the stored label (``source_label_reasons``) or a label that the validating
    execution loader refuses (``execution_loader_error``), on a task that is not excluded for it.
    """
    tasks = []
    for record in records:
        reasons = list(record.get("source_label_reasons") or [])
        if record.get("execution_loader_error") is not None and record.get("category") != "halt":
            reasons.append("execution_loader_refuses_label")
        if reasons and record.get("category") == "admissible":
            tasks.append(
                {
                    "sample_id": record["sample_id"],
                    "geographic_block": record.get("geographic_block"),
                    "reasons": reasons,
                }
            )
    return {
        "tolerance": dict(policy["label_tolerance"]),
        "n_tasks_with_findings": len(tasks),
        "reason_counts": dict(sorted(Counter(reason for task in tasks for reason in task["reasons"]).items())),
        "tasks": tasks,
        "publication": (
            "published as dataset quality findings with the tolerance; they change neither the reference nor the "
            "population"
        ),
    }


def _block_sizes(table: Mapping[str, Any]) -> dict[str, int]:
    return dict(Counter(block for block in table["blocks"].values() if block is not None))


def build_reference_inventory(
    source_roster: Mapping[str, Any],
    load_scenes: SceneLoader,
    policy: Mapping[str, Any],
    *,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """Qualify the reference of every source task under ``policy`` and categorize it.

    ``load_scenes(roster)`` returns one item per task: an ``AdmissionScene`` (for a pinned dataset,
    ``functools.partial(load_admission_scenes, dataset_path=root)``), or a ``Scene`` that the execution loader
    accepted. A loader failure raises ``ReferenceAdmissionHalt``: nothing can be qualified. Every other problem is a
    per-task ``halt`` record and makes the inventory's outcome ``halt``. Candidate outputs are never read. A roster
    of the test split raises ``TestSplitRefused`` before ``load_scenes`` is called, unless ``allow_test`` is true and
    ``admission`` is an all-GREEN test admission grant or a preparation grant that names the
    ``reference_inventory`` reader. The inventory records ``source_file_hashes``: ``skipped`` when any loaded item
    says its source files were not re-hashed. Every task whose search ended is recomputed by the second
    implementation; the exclusion caps of the policy are checked on the source block table.
    """
    roster = roster_io.validate_roster(source_roster)
    refuse_test_roster(
        roster, allow_test=allow_test, admission=admission, context="reference admission", reader="reference_inventory"
    )
    policy = validate_policy(policy)
    _check_policy_roster(policy, roster)
    try:
        items = list(load_scenes(roster))
    except Exception as exc:
        raise ReferenceAdmissionHalt(f"scene loading failed before any qualification: {_detail(exc)}") from exc
    table = policy["category_table"]
    independent = policy["reference_policy"] == "independent_analytic_reference"
    blocks = roster_io.roster_block_table(roster)
    by_id: dict[str, AdmissionScene] = {}
    duplicates: set[str] = set()
    files_verified = True
    for item in items:
        if isinstance(item, Scene):
            item = AdmissionScene(item.sample_id, item)
        elif not isinstance(item, AdmissionScene):
            raise ReferenceAdmissionHalt(f"the scene loader returned a {type(item).__name__}, not a scene")
        files_verified = files_verified and item.source_files_verified is True
        if item.sample_id in by_id:
            duplicates.add(item.sample_id)
        by_id[item.sample_id] = item
    roster_ids = [task["sample_id"] for task in roster["tasks"]]
    foreign = sorted(set(by_id) - set(roster_ids))
    records = []
    for task in roster["tasks"]:
        sample_id = task["sample_id"]
        record: dict[str, Any] = {
            "rank": task["rank"],
            "sample_id": sample_id,
            "input_identity_sha256": task["input_identity_sha256"],
            "geographic_block": blocks["blocks"][sample_id],
            "status": "not_run",
            "qualified": False,
            "reference_id": None,
            "reasons": [],
        }
        item = by_id.get(sample_id)
        if item is not None and item.loader_error is not None:
            record["execution_loader_error"] = item.loader_error
        halt = None
        if sample_id in duplicates:
            halt = "duplicate_sample_id"
        elif item is None:
            halt = "scene_missing"
        elif item.scene is None:
            halt = "scene_unreadable"
        else:
            try:
                identity = roster_io.identity_digest(roster_io.scene_input_identity(item.scene))
            except Exception as exc:  # inputs that do not even form a valid request
                identity = None
                record["detail"] = _detail(exc)
            if identity != task["input_identity_sha256"]:
                halt = "input_identity_mismatch"
        if halt is not None:
            record.update(category="halt", reason=halt, reasons=[halt])
            records.append(record)
            continue
        try:
            qualified = _qualify(item.scene, policy)
        except Exception as exc:
            record.update(
                category="halt", reason="qualification_error", reasons=["qualification_error"], detail=_detail(exc)
            )
            records.append(record)
            continue
        category, reason, reasons = _categorize(qualified, table)
        if category == "admissible" and item.loader_error is not None and not independent:
            # Qualified, but the validating execution loader refuses the stored label: the supervisor could not load
            # the task. The independent policy loads it with the tolerant loader and keeps the defect as a finding.
            category, reason, reasons = "halt", "execution_loader_refuses_label", ["execution_loader_refuses_label"]
        try:
            second = second_reference_check(item.scene, qualified, **_GRAPH_TOLERANCE)
        except Exception as exc:
            second = {"implementation": SECOND_IMPLEMENTATION, "status": "error", "detail": _detail(exc)}
        if second["status"] in ("disagrees", "error"):
            category, reason, reasons = "halt", "second_reference_disagreement", ["second_reference_disagreement"]
        elif category == "excluded" and set(reasons) & set(NO_PATH_REASONS) and qualified.get("legal") is True:
            # A legal published route proves that a path exists: the analytic search contradicts it.
            category, reason, reasons = "halt", "no_path_with_legal_label", ["no_path_with_legal_label"]
        record.update(
            status=qualified.get("reference_status", "unknown"),
            qualified=category == "admissible",
            reference_id=qualified.get("reference_id") if category == "admissible" else None,
            reference_expansions=qualified.get("reference_expansions"),
            category=category,
            reason=reason,
            reasons=reasons,
            stored_label_legal=qualified.get("legal"),
            second_reference=second,
        )
        if reason == "inconsistent_qualification_record":
            raw = qualified.get("qualification_reasons")
            record["qualifier_reasons"] = (
                [r if isinstance(r, str) else repr(r) for r in raw] if isinstance(raw, list) else repr(raw)
            )
        if "source_label_qualification_reasons" in qualified:  # label findings of the independent policy
            record["source_label_reasons"] = list(qualified["source_label_qualification_reasons"])
        records.append(record)
    counts = Counter(record["category"] for record in records)
    caps_check = exclusion_cap_check(records, policy.get("exclusion_caps"), _block_sizes(blocks))
    breached = caps_check is not None and not caps_check["within_caps"]
    outcome = "halt" if counts["halt"] or foreign or breached else "complete"
    inventory = {
        "schema": SCHEMA,
        "policy": policy,
        "policy_sha256": policy["policy_sha256"],
        "source_roster_sha256": roster["roster_sha256"],
        "source_block_table_sha256": blocks["block_table_sha256"],
        "benchmark_id": roster["benchmark_id"],
        "dataset": dict(policy["dataset"]),
        "cell": dict(policy["cell"]),
        "category_table_status": policy["category_table_status"],
        # "skipped": the source files were not re-hashed, so the stored labels are unverified (diagnostic only).
        "source_file_hashes": "verified" if files_verified else "skipped",
        "outcome": outcome,
        "counts": {
            "n_source": len(records),
            "n_admissible": counts["admissible"],
            "n_excluded": counts["excluded"],
            "n_halt": counts["halt"],
        },
        "excluded_reason_counts": _excluded_reason_counts(records),
        "exclusions": _listing(records, "excluded"),
        "halts": _listing(records, "halt"),
        "exclusion_caps_check": caps_check,
        "label_findings": label_findings(records, policy),
        "foreign_scene_ids": foreign,
        "tasks": records,
        "substitution": "none: excluded tasks are listed with their reason and never replaced",
        "candidate_outputs_read": False,
        "denominators": (
            "n_source counts every source task; n_admissible is the denominator every candidate runs and is scored "
            "on; a candidate failure on an admissible task stays in it with the canonical penalty"
        ),
    }
    inventory["inventory_sha256"] = _digest(inventory, "inventory_sha256")
    return inventory


def _task_follows(task: Mapping[str, Any], table: Mapping[str, Any]) -> bool:
    """Whether a task record's category follows from its recorded reasons under the category table."""
    category, reason, reasons = task.get("category"), task.get("reason"), task.get("reasons")
    if not isinstance(reasons, list):
        return False
    if category == "admissible":
        return (
            reason is None and reasons == [] and task.get("qualified") is True and _is_hex64(task.get("reference_id"))
        )
    if task.get("qualified") is not False or task.get("reference_id") is not None:
        return False
    if category == "halt" and reason in HALT_REASONS:
        return reasons == [reason]
    raw = reasons[:-1] if reasons and reasons[-1] == UNKNOWN_REASON else reasons
    if not raw or any(not isinstance(r, str) or r in HALT_REASONS or r == UNKNOWN_REASON for r in raw):
        return False
    return _categorize_reasons(raw, table) == (category, reason, reasons)


def validate_inventory(inventory: Any, source_roster: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute the inventory digest and check it against its policy and the source roster (no re-qualification).

    Every task must carry the source roster's rank, input identity and block; its category must follow from its
    recorded reasons under the policy's category table (so a reason the qualifiers never emit, such as a candidate
    outcome, cannot exclude a task), and an admissible task must carry a reference identity. The counts, the outcome
    and the exclusion, halt and reason-count tables are recomputed from the task records. This proves consistency,
    not that the qualifiers were run: rebuilding the inventory and comparing digests does that.
    """
    if not isinstance(inventory, Mapping) or inventory.get("schema") != SCHEMA:
        raise ValueError(f"a {SCHEMA} inventory is required")
    inventory = copy.deepcopy(dict(inventory))
    if inventory.get("inventory_sha256") != _digest(inventory, "inventory_sha256"):
        raise ValueError("inventory_sha256 differs from the recomputed digest")
    roster = roster_io.validate_roster(source_roster)
    policy = validate_policy(inventory.get("policy"))
    _check_policy_roster(policy, roster)
    blocks = roster_io.roster_block_table(roster)
    tasks = inventory.get("tasks")
    roster_ids = [task["sample_id"] for task in roster["tasks"]]
    foreign = inventory.get("foreign_scene_ids")
    if (
        inventory.get("policy_sha256") != policy["policy_sha256"]
        or inventory.get("source_roster_sha256") != roster["roster_sha256"]
        or inventory.get("source_block_table_sha256") != blocks["block_table_sha256"]
        or inventory.get("benchmark_id") != policy["benchmark_id"]
        or inventory.get("dataset") != policy["dataset"]
        or inventory.get("cell") != policy["cell"]
        or inventory.get("category_table_status") != policy["category_table_status"]
        or inventory.get("source_file_hashes") not in SOURCE_FILE_HASHES
        or inventory.get("candidate_outputs_read") is not False
        or not isinstance(tasks, list)
        or any(not isinstance(task, Mapping) for task in tasks)
        or [task.get("sample_id") for task in tasks] != roster_ids
        or any(task.get("category") not in CATEGORIES for task in tasks)
        or not isinstance(foreign, list)
        or any(not isinstance(sample_id, str) for sample_id in foreign)
        or foreign != sorted(set(foreign) - set(roster_ids))
    ):
        raise ValueError("the inventory does not belong to this policy and source roster")
    table = policy["category_table"]
    for task, source in zip(tasks, roster["tasks"]):
        sample_id = source["sample_id"]
        if (task.get("rank"), task.get("input_identity_sha256"), task.get("geographic_block")) != (
            source["rank"],
            source["input_identity_sha256"],
            blocks["blocks"][sample_id],
        ):
            raise ValueError(f"inventory task {sample_id} differs from the source roster in rank, identity or block")
        if not _task_follows(task, table):
            raise ValueError(
                f"inventory task {sample_id}: category {task.get('category')!r} with reason {task.get('reason')!r} "
                "does not follow from its reasons under the policy's category table"
            )
        second = task.get("second_reference")
        ended = task.get("status") in ("exact", "no_path")
        if task.get("reason") == "second_reference_disagreement":
            if not isinstance(second, Mapping) or second.get("status") not in ("disagrees", "error"):
                raise ValueError(f"inventory task {sample_id}: a second-reference halt without a disagreement")
        elif ended and (not isinstance(second, Mapping) or second.get("status") != "agrees"):
            raise ValueError(f"inventory task {sample_id}: the second implementation does not confirm its reference")
        if (
            task.get("category") == "excluded"
            and set(task.get("reasons") or []) & set(NO_PATH_REASONS)
            and task.get("stored_label_legal") is not False
            and task.get("stored_label_legal") is not None
        ):
            raise ValueError(f"inventory task {sample_id}: a no-path exclusion of a task with a legal published label")
    counts = Counter(task["category"] for task in tasks)
    expected = {
        "n_source": len(tasks),
        "n_admissible": counts["admissible"],
        "n_excluded": counts["excluded"],
        "n_halt": counts["halt"],
    }
    caps_check = exclusion_cap_check(tasks, policy.get("exclusion_caps"), _block_sizes(blocks))
    breached = caps_check is not None and not caps_check["within_caps"]
    outcome = "halt" if counts["halt"] or foreign or breached else "complete"
    if inventory.get("counts") != expected or inventory.get("outcome") != outcome:
        raise ValueError("inventory counts or outcome differ from its task records")
    if inventory.get("exclusion_caps_check") != caps_check:
        raise ValueError("the exclusion cap check differs from the task records")
    if inventory.get("label_findings") != label_findings(tasks, policy):
        raise ValueError("the label findings differ from the task records")
    if (
        inventory.get("exclusions") != _listing(tasks, "excluded")
        or inventory.get("halts") != _listing(tasks, "halt")
        or inventory.get("excluded_reason_counts") != _excluded_reason_counts(tasks)
    ):
        raise ValueError("the exclusion, halt or excluded-reason tables differ from the task records")
    return inventory


ADMISSIBLE_RULE = (
    "Source roster tasks whose reference qualified under the sealed admission policy, in source order. Excluded "
    "tasks are listed in the inventory and never replaced; no fresh sampling."
)


def admissible_roster(source_roster: Mapping[str, Any], inventory: Mapping[str, Any]) -> dict[str, Any]:
    """The admissible roster: the source tasks of category ``admissible``, with the parent's ranks and blocks."""
    roster = roster_io.validate_roster(source_roster)
    inventory = validate_inventory(inventory, roster)
    if inventory["source_file_hashes"] != "verified":
        raise ReferenceAdmissionHalt(
            "the inventory was built without re-hashing the source files (--skip-file-hashes); its stored labels are "
            "unverified, so it is diagnostic only and no admissible roster is derived"
        )
    if inventory["outcome"] != "complete":
        breaches = (inventory.get("exclusion_caps_check") or {}).get("breaches") or []
        raise ReferenceAdmissionHalt(
            f"the inventory halted ({inventory['counts']['n_halt']} halt record(s), "
            f"{len(inventory['foreign_scene_ids'])} foreign scene(s), {len(breaches)} exclusion cap breach(es)); "
            "no admissible roster is derived"
        )
    admitted = [task["sample_id"] for task in inventory["tasks"] if task["category"] == "admissible"]
    if not admitted:
        raise ReferenceAdmissionHalt("no source task has a qualified reference; no admissible roster is derived")
    return roster_io.derive_roster(
        roster,
        admitted,
        rule=ADMISSIBLE_RULE,
        derivation={
            "kind": "reference_admission",
            "admission_schema": SCHEMA,
            "policy_sha256": inventory["policy_sha256"],
            "inventory_sha256": inventory["inventory_sha256"],
            "category_table_status": inventory["category_table_status"],
            "source_file_hashes": inventory["source_file_hashes"],
            "n_source": inventory["counts"]["n_source"],
            "n_excluded": inventory["counts"]["n_excluded"],
            "n_admissible": inventory["counts"]["n_admissible"],
        },
    )


def admission_chain(
    source_roster: Mapping[str, Any],
    policy: Mapping[str, Any],
    inventory: Mapping[str, Any],
    admissible: Mapping[str, Any],
) -> dict[str, Any]:
    """Verify policy -> inventory -> admissible roster and return the chain of digests.

    The admissible roster must be exactly the one ``admissible_roster`` derives from the source roster and the
    inventory, and must keep the source's ranks and blocks.
    """
    roster = roster_io.validate_roster(source_roster)
    policy = validate_policy(policy)
    inventory = validate_inventory(inventory, roster)
    if inventory["policy"] != policy:
        raise ValueError("the inventory was made under another admission policy")
    expected = admissible_roster(roster, inventory)
    admissible = roster_io.validate_roster(admissible)
    if admissible["roster_sha256"] != expected["roster_sha256"]:
        raise ValueError("the admissible roster is not the one derived from this inventory")
    blocks = roster_io.verify_child_blocks(roster, admissible)
    chain = {
        "schema": CHAIN_SCHEMA,
        "source_roster_sha256": roster["roster_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "admissible_roster_sha256": admissible["roster_sha256"],
        "admissible_block_table_sha256": blocks["child_block_table_sha256"],
        "reference_policy": policy["reference_policy"],
        "reference_cohort_id": policy["reference_cohort_id"],
        "reference_max_expansions": policy["reference_max_expansions"],
        "category_table_status": policy["category_table_status"],
        "exclusion_caps": policy["exclusion_caps"],
        "second_reference": policy["second_reference"],
        "execution_loader_labels": policy["execution_loader_labels"],
        "source_file_hashes": inventory["source_file_hashes"],
        "counts": dict(inventory["counts"]),
        "n_label_findings": inventory["label_findings"]["n_tasks_with_findings"],
        "order": "policy, then inventory (contains the policy digest), then admissible roster (contains both)",
    }
    chain["chain_sha256"] = _digest(chain, "chain_sha256")
    return chain


class ReferenceIdentityMismatch(ValueError):
    """A run's qualified references are not those of the sealed reference inventory."""


def inventory_reference_ids(inventory: Mapping[str, Any]) -> dict[str, str]:
    """The ``reference_id`` of every admissible task of an inventory."""
    return {
        task["sample_id"]: task["reference_id"]
        for task in inventory.get("tasks", [])
        if task.get("category") == "admissible"
    }


def check_reference_ids(
    qualification: Mapping[str, Mapping[str, Any]], expected: Mapping[str, str], *, context: str
) -> int:
    """Refuse a run whose qualified references differ from the sealed inventory; return the number checked.

    Every task of ``qualification`` (``sample_id`` -> qualified reference record, as the supervisor and the roster
    scorer compute it) must be an admissible task of the inventory with the same ``reference_id``. The identity
    covers the search limit, the graph and the scene content, so one limit and one inventory bind run and scorer.
    """
    if not expected:
        raise ReferenceIdentityMismatch(f"{context}: the sealed inventory lists no admissible reference")
    missing = sorted(sample_id for sample_id in qualification if sample_id not in expected)
    differing = sorted(
        sample_id
        for sample_id, record in qualification.items()
        if sample_id in expected and record.get("reference_id") != expected[sample_id]
    )
    if missing or differing:
        raise ReferenceIdentityMismatch(
            f"{context}: {len(missing)} task(s) outside the sealed inventory (e.g. {missing[:3]}), {len(differing)} "
            f"reference_id(s) differ from it (e.g. {differing[:3]}); the run and the inventory must use one search "
            "limit and one reference policy"
        )
    return len(qualification)


# --------------------------------------------------------------------------------------------------------------
# command line
# --------------------------------------------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path.name} repeats the key {key!r}")
            result[key] = value
        return result

    return json.loads(path.read_bytes().decode("utf-8"), object_pairs_hook=no_duplicates)


def _write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=1, sort_keys=True, allow_nan=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -I -m passage_eval.protocol.reference_admission",
        description="Qualify the references of a sealed roster and derive its admissible roster.",
    )
    parser.add_argument("--roster", required=True, type=Path, help="sealed source roster (passage-roster-v1/v2)")
    parser.add_argument("--roster-sha256", required=True, metavar="HEX", help="pinned roster_sha256 of --roster")
    parser.add_argument("--dataset-root", required=True, type=Path, help="pinned dataset root holding PINNED.json")
    parser.add_argument("--reference-policy", required=True, choices=REFERENCE_POLICIES)
    parser.add_argument("--reference-max-expansions", required=True, type=int, help="at least N*N")
    parser.add_argument("--reference-cohort-id", help="required for independent_analytic_reference")
    parser.add_argument("--category-table", type=Path, help=f"{CATEGORY_TABLE_SCHEMA} JSON; default: the proposed one")
    parser.add_argument(
        "--exclusion-caps",
        type=Path,
        help=f"{CAPS_SCHEMA} JSON (required by the independent policy); default: no cap under released_labels",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="new private directory")
    add_test_admission_arguments(parser, what="reference admission", preparation=True)
    parser.add_argument(
        "--skip-file-hashes",
        action="store_true",
        help="do not re-hash the source files: the inventory is recorded as unverified and no admissible roster is derived",
    )
    args = parser.parse_args(argv)

    admission = test_split_allowed(
        args.allow_test_split,
        args.test_admission_record,
        evidence_dir=args.test_admission_evidence_dir,
        preparation_approval=args.test_preparation_approval,
    )
    roster = roster_io.load_roster(args.roster, expected_sha256=args.roster_sha256)
    allow_test = admission is not None
    split = refuse_test_roster(
        roster, allow_test=allow_test, admission=admission, context="reference admission", reader="reference_inventory"
    )
    if split != "validation" and not (split == "test" and allow_test):
        parser.error(f"reference admission runs on validation rosters (or an admitted test roster), not {split!r}")
    policy = admission_policy(
        roster,
        reference_policy=args.reference_policy,
        reference_max_expansions=args.reference_max_expansions,
        reference_cohort_id=args.reference_cohort_id,
        category_table=_read_json(args.category_table) if args.category_table else None,
        exclusion_caps=_read_json(args.exclusion_caps) if args.exclusion_caps else None,
        allow_test=allow_test,
        admission=admission,
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    _write_json(out / "reference-admission-policy.json", policy)

    def load(payload: Mapping[str, Any]) -> list[AdmissionScene]:
        return load_admission_scenes(payload, args.dataset_root, verify_file_hashes=not args.skip_file_hashes)

    try:
        inventory = build_reference_inventory(roster, load, policy, allow_test=allow_test, admission=admission)
    except ReferenceAdmissionHalt as exc:
        print(f"halt: {exc}", file=sys.stderr)
        return 2
    _write_json(out / "reference-admission-inventory.json", inventory)
    summary = {
        "outcome": inventory["outcome"],
        "counts": inventory["counts"],
        "excluded_reason_counts": inventory["excluded_reason_counts"],
        "exclusion_caps_check": inventory["exclusion_caps_check"],
        "n_label_findings": inventory["label_findings"]["n_tasks_with_findings"],
        "policy_sha256": policy["policy_sha256"],
        "inventory_sha256": inventory["inventory_sha256"],
        "category_table_status": policy["category_table_status"],
        "source_file_hashes": inventory["source_file_hashes"],
    }
    if inventory["outcome"] != "complete":
        print(json.dumps(summary, indent=1, sort_keys=True))
        print("halt: the inventory has halt records; no admissible roster was derived", file=sys.stderr)
        return 2
    try:
        admissible = admissible_roster(roster, inventory)
    except ReferenceAdmissionHalt as exc:  # every task excluded, or unverified source files
        print(json.dumps(summary, indent=1, sort_keys=True))
        print(f"halt: {exc}", file=sys.stderr)
        return 2
    _write_json(out / "admissible-roster.json", admissible)
    (out / "admissible-roster.sha256").write_text(admissible["roster_sha256"] + "\n", encoding="ascii")
    chain = admission_chain(roster, policy, inventory, admissible)
    _write_json(out / "reference-admission-chain.json", chain)
    print(json.dumps(dict(summary, admissible_roster_sha256=admissible["roster_sha256"]), indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
