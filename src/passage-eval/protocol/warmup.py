"""Warm-up inputs and the paired procedure order of a timed pass.

``select_warmup_scenes`` ranks validation sample ids by ``sha256(utf-8(salt + id))``
(the roster selection key, independent of any metric or outcome) and keeps the
first ``k`` that are disjoint from the evaluated tasks, by sample id and, when
identities are given, by input content. The content identity is the input
identity without its ordinary-metadata digest (``roster.CONTENT_IDENTITY_FIELDS``):
that metadata carries the split name and the row index, so a row copied into
another split keeps its terrain, obstacles, tensor, endpoints and objective but
not its input identity. The record holds the salt and the identities of the
selected inputs, so that both can be sealed.

``build_warmup_roster`` turns that selection into a ``passage-roster-v2`` of the
validation split built by the existing label-free roster builder, which the
supervisor accepts as ``--warmup-roster``.

``paired_order`` orders the procedures of one batch by
``sha256(canonical JSON [schema, schedule seed, batch index, procedure])``.

Proposed, not sealed. The protocol proposes eight warm-up scenes per worker and
an order fixed per batch; ``PROPOSED_WARMUP_COUNT`` and
``PROPOSED_ORDER_GRANULARITY`` implement that proposal and every record labels
it ``proposed_not_sealed``. The salt, the dataset root, the schedule seed and the
procedure set have no default: the protocol author fixes them.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

from passage_eval.io import roster as roster_io
from passage_eval.protocol.guards import refuse_test_roster, refuse_test_split

SELECTION_SCHEMA = "passage-warmup-selection/v1"
ORDER_SCHEMA = "passage-paired-order/v1"
WARMUP_SPLIT = "validation"
PROPOSED_STATUS = "proposed_not_sealed"
PROPOSED_WARMUP_COUNT = 8  # proposed, not sealed
PROPOSED_ORDER_GRANULARITY = "per_batch"  # proposed, not sealed
SELECTION_RULE = (
    "Rank every candidate validation sample id by sha256(utf-8(salt + sample_id)) hexdigest ascending; skip ids of "
    "the evaluated tasks and, when identities are known, inputs whose content identity (the input identity without "
    "the ordinary-metadata digest, which carries split and index) equals an evaluated task's; keep the first k. No "
    "metric or outcome enters the ranking."
)
CONTENT_CHECK = "sample_id_and_input_content_identity"


def _is_int(value: Any) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


def _digest(record: Mapping[str, Any], field: str) -> str:
    return roster_io.sha256_hex(roster_io.canonical_json({key: value for key, value in record.items() if key != field}))


def _ids_sha256(ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sorted(ids):
        digest.update(sample_id.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _rank(
    ids: Sequence[str],
    salt: str,
    k: int,
    evaluated_ids: set[str],
    digests: Mapping[str, tuple[str, str]] | None,
    evaluated_contents: set[str] | None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """``digests`` maps a candidate id to its (input identity, content identity) digests."""
    keys = {sample_id: roster_io.selection_key(salt, sample_id) for sample_id in ids}
    if len(set(keys.values())) != len(keys):
        raise ValueError("selection keys collide; choose another salt")
    selected, skipped_ids, skipped_inputs = [], [], []
    for position, sample_id in enumerate(sorted(ids, key=keys.__getitem__)):
        if len(selected) == k:
            break
        if sample_id in evaluated_ids:
            skipped_ids.append(sample_id)
            continue
        identity, content = digests[sample_id] if digests is not None else (None, None)
        if evaluated_contents is not None and content in evaluated_contents:
            skipped_inputs.append(sample_id)
            continue
        selected.append(
            {
                "rank": len(selected),
                "candidate_rank": position,
                "sample_id": sample_id,
                "selection_key": keys[sample_id],
                "input_identity_sha256": identity,
                "input_content_sha256": content,
            }
        )
    return selected, skipped_ids, skipped_inputs


def _identity_digests(identity: Any) -> tuple[str, str]:
    if not isinstance(identity, Mapping):
        raise ValueError("an input identity must be a record of identity fields (a roster task's input_identity)")
    return roster_io.identity_digest(identity), roster_io.content_identity_digest(identity)


def _check_inputs(ids, salt, k, split, evaluated_ids, identities, evaluated_identities):
    canonical = refuse_test_split(split, context="warm-up selection")
    if canonical != WARMUP_SPLIT:
        raise ValueError(f"warm-up scenes come from the validation split only, not {split!r}")
    ids = list(ids)
    if any(not isinstance(sample_id, str) or not sample_id.strip() for sample_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("candidate warm-up ids must be unique nonempty strings")
    if not isinstance(salt, str) or not salt:
        raise ValueError("a nonempty salt is required; it has no default")
    if not _is_int(k) or k < 1:
        raise ValueError("k must be a positive integer")
    if isinstance(evaluated_ids, (str, bytes)) or not isinstance(evaluated_ids, Iterable):
        raise ValueError("evaluated_ids must be a collection of sample ids")
    evaluated = set(evaluated_ids)
    digests = evaluated_contents = None
    if identities is not None:
        if not isinstance(identities, Mapping) or set(identities) != set(ids):
            raise ValueError("identities, when given, must cover every candidate id")
        if evaluated_identities is None:
            raise ValueError("identities need the evaluated identities to check disjointness")
        digests = {sample_id: _identity_digests(identity) for sample_id, identity in identities.items()}
    if evaluated_identities is not None:
        if identities is None:
            raise ValueError("evaluated identities need the candidates' identities")
        if isinstance(evaluated_identities, Mapping) or not isinstance(evaluated_identities, Iterable):
            raise ValueError("evaluated_identities must be a collection of input identity records")
        evaluated_contents = {_identity_digests(identity)[1] for identity in evaluated_identities}
    return ids, evaluated, digests, evaluated_contents


def select_warmup_scenes(
    ids: Iterable[str],
    salt: str,
    k: int = PROPOSED_WARMUP_COUNT,
    *,
    split: str,
    evaluated_ids: Iterable[str],
    identities: Mapping[str, Mapping[str, Any]] | None = None,
    evaluated_identities: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Select ``k`` validation warm-up inputs by salted SHA-256 rank, disjoint from the evaluated tasks.

    ``ids`` are candidate ids of the validation split; ``identities`` maps each of them to its input identity
    record (a roster task's ``input_identity``) and ``evaluated_identities`` holds the records of the evaluated
    tasks. A candidate whose content identity (``roster.content_identity_digest``) equals an evaluated task's is
    skipped. Raises ``ValueError`` when fewer than ``k`` disjoint candidates exist, and ``TestSplitRefused`` for
    the test split. ``k`` defaults to the proposed, unsealed eight.
    """
    ids, evaluated, digests, evaluated_contents = _check_inputs(
        ids, salt, k, split, evaluated_ids, identities, evaluated_identities
    )
    selected, skipped_ids, skipped_inputs = _rank(ids, salt, k, evaluated, digests, evaluated_contents)
    if len(selected) < k:
        raise ValueError(
            f"only {len(selected)} of {len(ids)} candidates are disjoint from the evaluated tasks; need {k}"
        )
    record = {
        "schema": SELECTION_SCHEMA,
        "decision_status": PROPOSED_STATUS,
        "split": WARMUP_SPLIT,
        "salt": salt,
        "k": k,
        "rule": SELECTION_RULE,
        "key": "sha256(utf-8(salt + sample_id)) hexdigest, ascending lexicographic",
        "n_candidates": len(ids),
        "candidate_ids_sha256": _ids_sha256(ids),
        "n_evaluated_ids": len(evaluated),
        "evaluated_ids_sha256": _ids_sha256(evaluated),
        "disjointness_checked_by": "sample_id_only" if digests is None else CONTENT_CHECK,
        "content_identity_fields": None if digests is None else list(roster_io.CONTENT_IDENTITY_FIELDS),
        "skipped_evaluated_ids": skipped_ids,
        "skipped_evaluated_inputs": skipped_inputs,
        "selected": selected,
        "selected_ids": [item["sample_id"] for item in selected],
        "selected_input_identity_sha256": None
        if digests is None
        else [item["input_identity_sha256"] for item in selected],
        "selected_input_content_sha256": None
        if digests is None
        else [item["input_content_sha256"] for item in selected],
    }
    record["selection_sha256"] = _digest(record, "selection_sha256")
    return record


def _pool_check(pool: Mapping[str, Any], salt: str, evaluated: Mapping[str, Any], n: int) -> dict[str, Any]:
    pool = roster_io.validate_roster(pool)
    refuse_test_roster(pool, context="warm-up roster")
    if pool["dataset"]["split"] != WARMUP_SPLIT:
        raise ValueError("the warm-up pool must be a validation roster")
    if pool["schema"] != roster_io.SCHEMA_V2:
        raise ValueError("the warm-up pool must be built by the current roster builder (passage-roster-v2)")
    if pool["selection"]["salt"] != salt or len(pool["tasks"]) != n:
        raise ValueError("the warm-up pool was not built with this salt and size")
    if pool["cell"] != evaluated["cell"] or pool["dataset"]["resolution"] != evaluated["dataset"]["resolution"]:
        raise ValueError("warm-up inputs must come from the evaluated cell (resolution, cost model, weight, variant)")
    return pool


def build_warmup_roster(
    build_pool: Callable[[int], Mapping[str, Any]],
    *,
    salt: str,
    evaluated_roster: Mapping[str, Any],
    benchmark_id: str,
    k: int = PROPOSED_WARMUP_COUNT,
    allow_test: bool = False,
    admission: Any = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(warm-up roster, selection record)`` for ``k`` validation inputs disjoint from ``evaluated_roster``.

    ``build_pool(n)`` must return the label-free roster of the first ``n`` validation rows by
    ``selection_key(salt, id)`` over the whole split (the roster builder with ``n_expected=n``). The pool grows
    until ``k`` disjoint inputs are found, so the result equals the selection over the whole population. An
    evaluated roster of the test split (its task identities only are read) needs ``allow_test`` and an all-GREEN
    admission grant or a preparation grant that names the ``warmup_roster`` reader.
    """
    evaluated = roster_io.validate_roster(evaluated_roster)
    refuse_test_roster(
        evaluated,
        allow_test=allow_test,
        admission=admission,
        context="the warm-up roster builder",
        reader="warmup_roster",
    )
    if not isinstance(benchmark_id, str) or not benchmark_id.strip():
        raise ValueError("a warm-up benchmark id is required")
    if not isinstance(salt, str) or not salt:
        raise ValueError("a nonempty salt is required; it has no default")
    if not _is_int(k) or k < 1:
        raise ValueError("k must be a positive integer")
    evaluated_ids = {task["sample_id"] for task in evaluated["tasks"]}
    evaluated_identities = [task["input_identity"] for task in evaluated["tasks"]]
    n = k
    while True:
        pool = _pool_check(build_pool(n), salt, evaluated, n)
        population = pool["selection"].get("population", {}).get("n_rows")
        ids = [task["sample_id"] for task in pool["tasks"]]
        identities = {task["sample_id"]: task["input_identity"] for task in pool["tasks"]}
        ids, evaluated_set, digests, evaluated_contents = _check_inputs(
            ids, salt, k, pool["dataset"]["split"], evaluated_ids, identities, evaluated_identities
        )
        selected, _, _ = _rank(ids, salt, k, evaluated_set, digests, evaluated_contents)
        if len(selected) == k:
            break
        if not _is_int(population) or n >= population:
            raise ValueError(f"the validation population holds fewer than {k} inputs disjoint from the evaluated tasks")
        n = min(population, n + k - len(selected))
    selection = select_warmup_scenes(
        ids,
        salt,
        k,
        split=pool["dataset"]["split"],
        evaluated_ids=evaluated_ids,
        identities=identities,
        evaluated_identities=evaluated_identities,
    )
    by_id = {task["sample_id"]: task for task in pool["tasks"]}
    tasks = []
    for rank, sample_id in enumerate(selection["selected_ids"]):
        task = copy.deepcopy(by_id[sample_id])
        task["rank"] = rank
        task["geographic_block"] = None
        tasks.append(task)
    records = [
        {"sample_id": t["sample_id"], "source_windows": t["source_windows"]} for t in tasks if t["source_windows"]
    ]
    blocks, assignments = roster_io.geographic_block_record(
        records,
        n_tasks=len(tasks),
        block_degrees=pool["geographic_blocks"]["block_degrees"],
        footprints_required=False,
        source=pool["geographic_blocks"].get("source", "pool roster source windows"),
    )
    for task in tasks:
        if task["source_windows"]:
            task["geographic_block"] = assignments[task["sample_id"]]
    dataset = dict(pool["dataset"])
    dataset["purpose"] = "untimed warm-up inputs of a timed pass; never scored; validation split"
    roster = {
        "schema": roster_io.SCHEMA_V2,
        "benchmark_id": benchmark_id,
        "dataset": dataset,
        "cell": dict(pool["cell"]),
        "selection": {
            "rule": SELECTION_RULE,
            "salt": salt,
            "key": "sha256(utf-8(salt + id)) hexdigest, ascending lexicographic",
            "n_expected": k,
            "n_selected": k,
            "population": copy.deepcopy(pool["selection"].get("population")),
            "warmup": {
                "decision_status": PROPOSED_STATUS,
                "selection_sha256": selection["selection_sha256"],
                "evaluated_roster_sha256": evaluated["roster_sha256"],
                "evaluated_benchmark_id": evaluated["benchmark_id"],
                "disjointness_checked_by": selection["disjointness_checked_by"],
                "skipped_evaluated_ids": selection["skipped_evaluated_ids"],
                "skipped_evaluated_inputs": selection["skipped_evaluated_inputs"],
                "pool_size": n,
            },
        },
        "dataset_root_identity": copy.deepcopy(pool["dataset_root_identity"]),
        "input_identity_spec": copy.deepcopy(pool["input_identity_spec"]),
        "source_files": copy.deepcopy(pool["source_files"]),
        "geographic_blocks": blocks,
        "tasks": tasks,
    }
    roster["roster_sha256"] = roster_io.compute_roster_sha256(roster)
    return roster_io.validate_roster(roster), selection


def _check_order_inputs(schedule_seed: Any, procedures: Sequence[str]) -> list[str]:
    if not (_is_int(schedule_seed) or (isinstance(schedule_seed, str) and schedule_seed.strip())):
        raise ValueError("schedule_seed must be an integer or a nonempty string; it has no default")
    if isinstance(procedures, (str, bytes)) or not isinstance(procedures, Iterable):
        raise ValueError("procedures must be a collection of procedure ids")
    procedures = list(procedures)
    if (
        not procedures
        or any(not isinstance(item, str) or not item.strip() for item in procedures)
        or len(set(procedures)) != len(procedures)
    ):
        raise ValueError("procedures must be unique nonempty strings")
    return procedures


def _order_key(schedule_seed: int | str, batch_index: int, procedure: str) -> str:
    return roster_io.sha256_hex(roster_io.canonical_json([ORDER_SCHEMA, schedule_seed, batch_index, procedure]))


def paired_order(schedule_seed: int | str, batch_index: int, procedures: Sequence[str]) -> list[str]:
    """The procedures of one batch, ordered by SHA-256 of (schema, seed, batch index, procedure).

    The order depends on the seed, the batch index and the procedure ids only, not on the order of ``procedures``.
    """
    procedures = _check_order_inputs(schedule_seed, procedures)
    if not _is_int(batch_index) or batch_index < 0:
        raise ValueError("batch_index must be a nonnegative integer")
    return sorted(procedures, key=lambda procedure: (_order_key(schedule_seed, batch_index, procedure), procedure))


def paired_schedule(schedule_seed: int | str, procedures: Sequence[str], n_batches: int) -> dict[str, Any]:
    """The per-batch paired order of every batch, as a hashed record (granularity proposed, not sealed)."""
    procedures = _check_order_inputs(schedule_seed, procedures)
    if not _is_int(n_batches) or n_batches < 1:
        raise ValueError("n_batches must be a positive integer")
    record = {
        "schema": ORDER_SCHEMA,
        "granularity": PROPOSED_ORDER_GRANULARITY,
        "decision_status": PROPOSED_STATUS,
        "schedule_seed": schedule_seed,
        "procedures": sorted(procedures),
        "n_batches": n_batches,
        "key": "sha256 of the canonical JSON [schema, schedule_seed, batch_index, procedure], ascending",
        "orders": [paired_order(schedule_seed, index, procedures) for index in range(n_batches)],
    }
    record["schedule_sha256"] = _digest(record, "schedule_sha256")
    return record
