# MIT License
#
# Copyright (c) 2026 THALES

"""``python -I -m passage_eval.campaign COMMAND``.

``partition``         split a pinned admissible roster into child rosters (new directory).
``verify-partition``  re-derive and compare an existing partition.
``check-plan``        check a pinned plan and print its dynamic admission sizing, its worst case (information only) and
                      the memory of one lot; runs nothing.
``run``               execute or resume the batches of a pinned plan, with automatic R7 relaunches; ``--stop-after-batches``
                      pauses at a batch boundary, ``--deliberate-kill`` exercises R7 in a validation rehearsal.
``relaunch-batch``    archive one attempt stopped by a harness error, with operator and reason, so that ``run`` retries it.
``aggregate``         aggregate one procedure of a completed campaign.
``rehearsal-receipt`` build the ``passage-native-chain-rehearsal/v1`` receipt of a completed validation rehearsal.
``statistics``        analyse already-pinned aggregates; it is not part of the descriptive single-seed pass.

The test split needs ``--allow-test-split``; the plan names its ``passage-test-admission/v1`` record, and
``partition`` takes ``--test-admission-record``. Every record must check all GREEN (``passage-eval admission-check``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from passage_eval.protocol import guards


def _plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plan", required=True, type=Path, help="passage-campaign-plan/v1 JSON")
    parser.add_argument("--plan-sha256", required=True, metavar="HEX", help="SHA-256 pin of the plan file")
    parser.add_argument(
        "--allow-test-split", action="store_true", help="run an admitted test plan (its admission record must be GREEN)"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -I -m passage_eval.campaign", allow_abbrev=False)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("partition", "verify-partition"):
        sub = commands.add_parser(name, allow_abbrev=False)
        sub.add_argument("--parent", required=True, type=Path, help="pinned admissible roster file")
        sub.add_argument("--parent-file-sha256", required=True, metavar="HEX")
        sub.add_argument("--parent-roster-sha256", required=True, metavar="HEX")
        sub.add_argument("--out-dir", required=True, type=Path)
        if name == "partition":
            sub.add_argument("--batch-size", required=True, type=int)
        else:
            sub.add_argument("--manifest-sha256", required=True, metavar="HEX")
        guards.add_test_admission_arguments(sub, what="a partition")
    check = commands.add_parser("check-plan", allow_abbrev=False)
    _plan_arguments(check)
    run = commands.add_parser("run", allow_abbrev=False)
    _plan_arguments(run)
    run.add_argument(
        "--stop-after-batches",
        type=int,
        metavar="N",
        help="stop cleanly after N batches completed by this invocation (a resume point)",
    )
    run.add_argument(
        "--deliberate-kill",
        action="append",
        default=[],
        metavar="BATCH:PROCEDURE:SECONDS",
        help="validation rehearsal only: kill the first attempt of this batch's procedure session after SECONDS to "
        "exercise the R7 relaunch on purpose; repeatable",
    )
    relaunch = commands.add_parser("relaunch-batch", allow_abbrev=False)
    _plan_arguments(relaunch)
    relaunch.add_argument("--batch", required=True, type=int)
    relaunch.add_argument("--unit", help="an R7 half of the batch, such as 000003.1 (default: the batch itself)")
    relaunch.add_argument("--operator", required=True)
    relaunch.add_argument("--reason", required=True)
    relaunch.add_argument("--reason-class", required=True, choices=["infrastructure", "candidate"])
    aggregate = commands.add_parser("aggregate", allow_abbrev=False)
    _plan_arguments(aggregate)
    aggregate.add_argument("--procedure", required=True)
    aggregate.add_argument("--output", type=Path, help="new JSON file (default: print)")
    stats = commands.add_parser("statistics", allow_abbrev=False)
    stats.add_argument(
        "--aggregate",
        required=True,
        action="append",
        type=Path,
        help="aggregate JSON; one Seg and one Reg per seed, from one campaign or one campaign per seed",
    )
    stats.add_argument("--aggregate-sha256", required=True, action="append", metavar="HEX", help="one per --aggregate")
    for name in ("--bootstrap-seed", "--permutation-seed", "--latency-bootstrap-seed"):
        stats.add_argument(name, required=True, type=int, help="sealed resampling seed (no default)")
    stats.add_argument("--draws", type=int, default=10_000, help="shared bootstrap draws")
    stats.add_argument("--permutations", type=int, default=10_000, help="Monte-Carlo block swaps above 16 blocks")
    stats.add_argument("--alpha", type=float, default=0.05, help="family-wise level of the Holm adjustment")
    stats.add_argument(
        "--block-criterion",
        type=Path,
        help="passage-block-adequacy-criterion/v1 JSON used without a sealed record; otherwise it must equal the "
        "record's block analysis",
    )
    stats.add_argument(
        "--protocol-record",
        action="append",
        default=[],
        type=Path,
        help="passage-test-admission/v1 record named by an aggregate; one per seed campaign",
    )
    stats.add_argument(
        "--protocol-record-sha256", action="append", default=[], metavar="HEX", help="one per --protocol-record"
    )
    stats.add_argument("--output", type=Path, help="new JSON file (default: print)")
    receipt = commands.add_parser("rehearsal-receipt", allow_abbrev=False)
    receipt.add_argument("--plan", required=True, type=Path, help="the validation rehearsal plan")
    receipt.add_argument("--plan-sha256", required=True, metavar="HEX")
    receipt.add_argument("--evidence-dir", required=True, type=Path, help="evidence directory of the admission record")
    receipt.add_argument(
        "--aggregate",
        action="append",
        default=[],
        required=True,
        metavar="PROCEDURE=RELATIVE_PATH",
        help="aggregate file of one procedure, relative to the evidence directory; one per procedure",
    )
    receipt.add_argument("--output", required=True, help="new receipt file, relative to the evidence directory")
    return parser


def _kills(values: list[str]) -> list[tuple[int, str, float]]:
    kills = []
    for value in values:
        index, _, rest = value.partition(":")
        procedure, _, seconds = rest.partition(":")
        try:
            kills.append((int(index), procedure, float(seconds)))
        except ValueError:
            raise SystemExit(f"--deliberate-kill expects BATCH:PROCEDURE:SECONDS, got {value!r}") from None
    return kills


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in ("partition", "verify-partition"):
        from passage_eval.campaign import partition as partition_io

        admission = guards.test_split_allowed(
            args.allow_test_split, args.test_admission_record, evidence_dir=args.test_admission_evidence_dir
        )
        common = (args.parent, args.parent_file_sha256, args.parent_roster_sha256, args.out_dir)
        if args.command == "partition":
            manifest = partition_io.partition(
                *common, args.batch_size, allow_test=admission is not None, admission=admission
            )
        else:
            manifest = partition_io.verify_partition(
                *common, args.manifest_sha256, allow_test=admission is not None, admission=admission
            )
        print(json.dumps({"n_expected": manifest["n_expected"], "n_batches": len(manifest["batches"])}, sort_keys=True))
        return 0
    if args.command == "check-plan":
        from passage_eval.campaign import partition as partition_io
        from passage_eval.campaign.plan import admission_sizing, batch_bounds, read_plan

        campaign = read_plan(args.plan, args.plan_sha256, allow_test=args.allow_test_split)
        bound = batch_bounds(campaign.plan, campaign.specs)
        summary = {
            "plan_sha256": campaign.sha256,
            "split": campaign.plan["split"],
            "target": campaign.plan["target"],
            "procedures": list(campaign.procedures),
            "n_batches": len(campaign.partition["batches"]),
            "n_expected": campaign.partition["n_expected"],
            "dynamic_admission": admission_sizing(campaign.plan, campaign.specs),
            "worst_case_batch_wall_ms_information_only": bound["worst_case_wall_ms"],
            "batch_window_ms": bound["limit_ms"],
            "lot_memory": partition_io.lot_memory_bytes(
                campaign.admissible["dataset"]["resolution"], campaign.plan["partition"]["batch_size"]
            ),
        }
        print(json.dumps(summary, indent=1, sort_keys=True))
        return 0
    if args.command == "run":
        from passage_eval.campaign.runner import run_campaign

        status = run_campaign(
            args.plan,
            args.plan_sha256,
            allow_test=args.allow_test_split,
            deliberate_kills=_kills(args.deliberate_kill),
            stop_after_batches=args.stop_after_batches,
        )
        print(json.dumps(status, sort_keys=True))
        return 0
    if args.command == "relaunch-batch":
        from passage_eval.campaign.runner import relaunch_batch

        entry = relaunch_batch(
            args.plan,
            args.plan_sha256,
            args.batch,
            operator=args.operator,
            reason=args.reason,
            reason_class=args.reason_class,
            allow_test=args.allow_test_split,
            unit=args.unit,
        )
        print(json.dumps(entry, sort_keys=True))
        return 0
    if args.command == "rehearsal-receipt":
        from passage_eval.campaign.rehearsal import build_rehearsal_receipt
        from passage_eval.io import pinned

        aggregates = {}
        for value in args.aggregate:
            name, separator, path = value.partition("=")
            if not separator or not name or name in aggregates:
                raise SystemExit(f"--aggregate expects one PROCEDURE=RELATIVE_PATH per procedure, got {value!r}")
            aggregates[name] = path
        receipt = build_rehearsal_receipt(
            args.plan, args.plan_sha256, evidence_dir=args.evidence_dir, aggregates=aggregates
        )
        target = args.evidence_dir / pinned.safe_relative(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = pinned.write_json_exclusive(target, receipt)
        print(json.dumps({"path": str(pinned.safe_relative(args.output)), "sha256": digest}, sort_keys=True))
        return 0
    if args.command == "statistics":
        return _statistics(args)
    from passage_eval.campaign.aggregate import aggregate_campaign
    from passage_eval.io import pinned

    result = aggregate_campaign(args.plan, args.plan_sha256, procedure=args.procedure, allow_test=args.allow_test_split)
    if args.output is not None:
        print(pinned.write_json_exclusive(args.output, result))
    else:
        print(json.dumps(result, indent=1, sort_keys=True, allow_nan=False))
    return 0


def _statistics(args: argparse.Namespace) -> int:
    from passage_eval.campaign.statistics import campaign_statistics
    from passage_eval.io import pinned

    if len(args.aggregate) != len(args.aggregate_sha256):
        raise SystemExit("give one --aggregate-sha256 per --aggregate")
    if len(args.protocol_record) != len(args.protocol_record_sha256):
        raise SystemExit("give one --protocol-record-sha256 per --protocol-record")
    aggregates = [pinned.read_pinned_json(path, digest) for path, digest in zip(args.aggregate, args.aggregate_sha256)]
    digests = list(args.aggregate_sha256)
    records = {
        digest: pinned.read_pinned_json(path, digest)
        for path, digest in zip(args.protocol_record, args.protocol_record_sha256)
    }
    criterion = pinned.strict_json_loads(pinned.read_bounded(args.block_criterion)) if args.block_criterion else None
    result = campaign_statistics(
        aggregates,
        bootstrap_seed=args.bootstrap_seed,
        permutation_seed=args.permutation_seed,
        latency_bootstrap_seed=args.latency_bootstrap_seed,
        draws=args.draws,
        permutations=args.permutations,
        alpha=args.alpha,
        criterion=criterion,
        aggregate_sha256=digests,
        protocol_records=records or None,
    )
    if args.output is not None:
        print(pinned.write_json_exclusive(args.output, result))
    else:
        print(json.dumps(result, indent=1, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
