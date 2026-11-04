# MIT License
#
# Copyright (c) 2026 THALES

"""Label-free census of geographic block adequacy ``passage-block-census/v1``.

Before any reference or candidate output exists, the census reads only the footprints a sealed roster records for
each task (``source_windows``: the used one-degree tiles; ``source_tile``: the main tile) and measures, for the
block rules of ``protocol.geographic_blocks``, the number of blocks ``G``, the block size distribution and the
share of the largest block. It never opens a dataset file and never reads a label.

Order of the rules (judge tightening 1, proposed):

1. **Rule A**, ``source_window_union`` at ``block_degrees`` (5): the grid cells of the used tiles, merged by
   transitive union across tasks. It is retained when it meets the criterion.
2. **Rule B**, ``main_tile_cell`` at ``block_degrees``: the cell of the main tile, without union. It is retained
   when rule A fails and rule B meets the criterion. Two sensitivities are reported, never decisive: rule B without
   the tasks whose used tiles cross a cell boundary, and rule B at ``sensitivity_degrees`` (10).
3. Otherwise the block analysis is **descriptive only**: no confirmatory claim follows from it.

The criterion (``PROPOSED_CRITERION``: at least 30 blocks and no block above 10 % of the tasks) is
``proposed_not_sealed`` until the protocol author seals a criterion record with the same fields; the census records
the status it was given. Tasks without the footprint a rule needs are counted in the one sealed ``unplaced`` block
(never in invented singleton blocks), which counts as a block. When they exceed ``max_unplaced_fraction`` (proposed:
0.5 %) of the tasks, the block analysis of the population stops for diagnosis (decision ``halt_unplaced_cap``).
The census is computed on the roster it is given: the source roster before the reference inventory, and again on
the admissible roster (gate G4 applies the sealed cap to the admissible tasks).

Nothing here proves that blocks are independent or exchangeable.

Command (a test roster needs ``--allow-test-split`` with an all-GREEN admission record, or a preparation approval
that names the ``block_census`` reader)::

    python -I -m passage_eval.protocol.block_census --roster ROSTER.json --roster-sha256 HEX [--criterion FILE] \\
        [--output CENSUS.json]
"""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path
from typing import Any

from passage_eval.io import roster as roster_io
from passage_eval.protocol.geographic_blocks import (
    MAIN_TILE_CELL,
    SOURCE_WINDOW_UNION,
    UNPLACED_BLOCK,
    compute_blocks,
    task_footprint,
)
from passage_eval.protocol.guards import add_test_admission_arguments, refuse_test_roster, test_split_allowed

CENSUS_SCHEMA = "passage-block-census/v1"
CRITERION_SCHEMA = "passage-block-adequacy-criterion/v1"
PROPOSED_STATUS = "proposed_not_sealed"
#: PROPOSED, NOT SEALED: the judge's criterion and fallback order; the protocol author seals a record with these keys.
PROPOSED_CRITERION: dict[str, Any] = {
    "schema": CRITERION_SCHEMA,
    "status": PROPOSED_STATUS,
    "min_blocks": 30,
    "max_block_share": 0.10,
    "max_unplaced_fraction": 0.005,
    "block_degrees": 5,
    "sensitivity_degrees": 10,
}
DECISIONS = ("rule_a", "rule_b", "descriptive_only", "halt_unplaced_cap")


def _number(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and value == value and abs(value) != float("inf")


def _degrees(value: Any) -> bool:
    return type(value) is int and value >= 1 and not 180 % value and not 360 % value


def validate_criterion(criterion: Any) -> dict[str, Any]:
    """A block adequacy criterion record: exactly the keys of ``PROPOSED_CRITERION`` with valid values."""
    if not isinstance(criterion, Mapping) or criterion.get("schema") != CRITERION_SCHEMA:
        raise ValueError(f"a {CRITERION_SCHEMA} criterion is required")
    if set(criterion) != set(PROPOSED_CRITERION):
        raise ValueError(f"a criterion has exactly the keys {sorted(PROPOSED_CRITERION)}")
    if not isinstance(criterion["status"], str) or not criterion["status"].strip():
        raise ValueError("a criterion needs a sealing status")
    if not isinstance(criterion["min_blocks"], Integral) or isinstance(criterion["min_blocks"], bool):
        raise ValueError("min_blocks must be an integer")
    if criterion["min_blocks"] < 2:
        raise ValueError("min_blocks must be at least 2")
    if not _number(criterion["max_block_share"]) or not 0 < criterion["max_block_share"] <= 1:
        raise ValueError("max_block_share must lie in (0, 1]")
    if not _number(criterion["max_unplaced_fraction"]) or not 0 <= criterion["max_unplaced_fraction"] < 1:
        raise ValueError("max_unplaced_fraction must lie in [0, 1)")
    if not _degrees(criterion["block_degrees"]) or not _degrees(criterion["sensitivity_degrees"]):
        raise ValueError("block and sensitivity degrees must divide 180 and 360")
    return copy.deepcopy(dict(criterion))


def _distribution(sizes: Counter) -> dict[str, Any]:
    ordered = sorted(sizes.values())
    return {
        "min": ordered[0] if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "max": ordered[-1] if ordered else None,
        "singleton_blocks": sum(1 for size in ordered if size == 1),
        "histogram": {str(size): count for size, count in sorted(Counter(ordered).items())},
        "largest_blocks": dict(sorted(sizes.items(), key=lambda item: (-item[1], item[0]))[:10]),
    }


def adequacy(
    blocks: Mapping[str, str | None],
    *,
    min_blocks: int,
    max_block_share: float,
    max_unplaced_fraction: float,
) -> dict[str, Any]:
    """Number of blocks, size distribution, largest share and the criterion for one block assignment.

    ``blocks`` maps every analysed task to its block; the sealed ``unplaced`` block counts as one block. A task
    mapped to ``None`` (no block, no unplaced block) leaves the assignment incomplete and fails the criterion.
    """
    n = len(blocks)
    if not n:
        raise ValueError("block adequacy needs at least one task")
    sizes = Counter(block for block in blocks.values() if block is not None)
    without = sum(block is None for block in blocks.values())
    n_unplaced = sizes.get(UNPLACED_BLOCK, 0)
    largest = max(sizes.items(), key=lambda item: (item[1], item[0]), default=(None, 0))
    share = largest[1] / n
    unplaced_fraction = n_unplaced / n
    result = {
        "n_tasks": n,
        "number_of_blocks": len(sizes),
        "n_unplaced": n_unplaced,
        "unplaced_fraction": unplaced_fraction,
        "n_tasks_without_block": without,
        "largest_block": largest[0],
        "largest_block_tasks": largest[1],
        "largest_block_share": share,
        "block_size_distribution": _distribution(sizes),
        "complete": without == 0,
        "meets_min_blocks": len(sizes) >= min_blocks,
        "meets_max_block_share": share <= max_block_share,
        "unplaced_within_cap": unplaced_fraction <= max_unplaced_fraction,
    }
    result["meets_criterion"] = all(
        result[key] for key in ("complete", "meets_min_blocks", "meets_max_block_share", "unplaced_within_cap")
    )
    return result


def rule_assignment(
    tasks: Sequence[Mapping[str, Any]], *, rule: str, block_degrees: int, exclude_crossing: bool = False
) -> dict[str, Any]:
    """The blocks of ``tasks`` under one rule: placed tasks in their blocks, the others in the unplaced block.

    ``exclude_crossing`` (rule B only) leaves out the tasks whose used tiles fall in another cell than their main
    tile; removing them changes no other block, since rule B has no union.
    """
    if exclude_crossing and rule != MAIN_TILE_CELL:
        raise ValueError("only rule B can leave out the crossing tasks without changing the other blocks")
    footprints = {task["sample_id"]: task_footprint(task, rule) for task in tasks}
    records = [footprint for footprint in footprints.values() if footprint is not None]
    computed = compute_blocks(records, rule=rule, block_degrees=block_degrees) if records else {}
    placed = computed.get("task_blocks", {})
    crossing = set(computed.get("crossing_task_ids", ()))
    kept = [sample_id for sample_id in footprints if not (exclude_crossing and sample_id in crossing)]
    return {
        "rule": rule,
        "block_degrees": block_degrees,
        "blocks": {sample_id: placed.get(sample_id, UNPLACED_BLOCK) for sample_id in kept},
        "unplaced_task_ids": [sample_id for sample_id in kept if footprints[sample_id] is None],
        "crossing_task_count": computed.get("crossing_task_count", 0),
        "excluded_crossing_tasks": len(footprints) - len(kept),
    }


def _analysis(tasks, criterion, *, rule, degrees, exclude_crossing=False, decisive=True) -> dict[str, Any]:
    assignment = rule_assignment(tasks, rule=rule, block_degrees=degrees, exclude_crossing=exclude_crossing)
    limits = {key: criterion[key] for key in ("min_blocks", "max_block_share", "max_unplaced_fraction")}
    result = {key: value for key, value in assignment.items() if key != "blocks"}
    result.update(decisive=decisive, adequacy=adequacy(assignment["blocks"], **limits))
    return result


def block_census(
    roster: Mapping[str, Any],
    *,
    criterion: Mapping[str, Any] | None = None,
    allow_test: bool = False,
    admission: Any = None,
) -> dict[str, Any]:
    """The ``passage-block-census/v1`` record of a sealed roster (see the module docstring); reads no dataset file.

    A roster of the test split raises ``TestSplitRefused`` unless ``allow_test`` is true and ``admission`` is an
    all-GREEN admission grant or a preparation grant that names the ``block_census`` reader.
    """
    roster = roster_io.validate_roster(roster)
    refuse_test_roster(
        roster, allow_test=allow_test, admission=admission, context="the block census", reader="block_census"
    )
    criterion = validate_criterion(PROPOSED_CRITERION if criterion is None else criterion)
    tasks = roster["tasks"]
    degrees, sensitivity = criterion["block_degrees"], criterion["sensitivity_degrees"]
    rules = {
        "rule_a": _analysis(tasks, criterion, rule=SOURCE_WINDOW_UNION, degrees=degrees),
        "rule_b": _analysis(tasks, criterion, rule=MAIN_TILE_CELL, degrees=degrees),
        "rule_b_without_crossing_tasks": _analysis(
            tasks, criterion, rule=MAIN_TILE_CELL, degrees=degrees, exclude_crossing=True, decisive=False
        ),
        "rule_b_sensitivity_degrees": _analysis(
            tasks, criterion, rule=MAIN_TILE_CELL, degrees=sensitivity, decisive=False
        ),
    }
    a, b = rules["rule_a"]["adequacy"], rules["rule_b"]["adequacy"]
    if not a["unplaced_within_cap"]:
        decision, reason = "halt_unplaced_cap", "rule A: tasks without footprint exceed the unplaced cap"
    elif a["meets_criterion"]:
        decision, reason = "rule_a", "rule A meets the criterion"
    elif not b["unplaced_within_cap"]:
        decision, reason = (
            "halt_unplaced_cap",
            "rule A fails the criterion; rule B: tasks without footprint exceed the cap",
        )
    elif b["meets_criterion"]:
        decision, reason = "rule_b", "rule A fails the criterion; rule B meets it"
    else:
        decision, reason = "descriptive_only", "neither rule meets the criterion: block analysis is descriptive only"
    census = {
        "schema": CENSUS_SCHEMA,
        "roster_sha256": roster["roster_sha256"],
        "benchmark_id": roster["benchmark_id"],
        "dataset": {key: roster["dataset"][key] for key in ("repo", "commit", "config", "split", "resolution")},
        "n_tasks": len(tasks),
        "roster_block_rule": roster_io.block_rule(roster),
        "inputs": "the roster's task footprints only (source_windows, source_tile); no dataset file, no label",
        "labels_read": False,
        "candidate_outputs_read": False,
        "criterion": criterion,
        "criterion_status": criterion["status"],
        "rules": rules,
        "decision": decision,
        "decision_reason": reason,
        "confirmatory_block_analysis_possible": decision in ("rule_a", "rule_b"),
        "statistical_independence_proven": False,
    }
    census["census_sha256"] = roster_io.sha256_hex(roster_io.canonical_json(census))
    return census


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -I -m passage_eval.protocol.block_census",
        description="Label-free census of geographic block adequacy for a sealed roster.",
    )
    parser.add_argument("--roster", required=True, type=Path, help="sealed roster (passage-roster-v1/v2)")
    parser.add_argument("--roster-sha256", required=True, metavar="HEX", help="pinned roster_sha256 of --roster")
    parser.add_argument("--criterion", type=Path, help=f"{CRITERION_SCHEMA} JSON; default: the proposed criterion")
    parser.add_argument("--output", type=Path, help="new JSON file (default: print)")
    add_test_admission_arguments(parser, what="the block census", preparation=True)
    args = parser.parse_args(argv)
    admission = test_split_allowed(
        args.allow_test_split,
        args.test_admission_record,
        evidence_dir=args.test_admission_evidence_dir,
        preparation_approval=args.test_preparation_approval,
    )
    roster = roster_io.load_roster(args.roster, expected_sha256=args.roster_sha256)
    criterion = None
    if args.criterion is not None:
        criterion = json.loads(args.criterion.read_bytes().decode("utf-8"))
    census = block_census(roster, criterion=criterion, allow_test=admission is not None, admission=admission)
    text = json.dumps(census, indent=1, sort_keys=True, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as handle:
            handle.write(text)
    summary = {
        "decision": census["decision"],
        "criterion_status": census["criterion_status"],
        "census_sha256": census["census_sha256"],
        **{
            name: {
                key: entry["adequacy"][key] for key in ("number_of_blocks", "largest_block_share", "meets_criterion")
            }
            for name, entry in census["rules"].items()
        },
    }
    print(json.dumps(summary, indent=1, sort_keys=True) if args.output is not None else text, end="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
