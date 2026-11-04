# MIT License
#
# Copyright (c) 2026 THALES

"""``python -I -m passage_eval.supervision {run,stage-score,score-roster,tree-hash}``.

``run``          qualify references in this (trusted) process, then execute the candidate
                 in supervised workers and write standard run artifacts. The artifacts
                 contain no references and no dataset location (but candidate-controlled
                 text: treat them as private until reviewed).
``stage-score``  scorer side: copy a public run directory to a private location and attach
                 the references so that ``passage-eval score`` can be run on it (runs made
                 from a spec; a roster run is scored by ``score-roster``).
``score-roster`` scorer side, own process: score a roster run against its pinned roster and
                 dataset root into a private directory (the layout ``cmd_score`` produces).
``tree-hash``    print the deterministic tree digest of a file or directory (the value an
                 operator pins with ``--decoder-allow-sha256`` or ``--planner-allow-sha256``;
                 with ``--environment`` the digest of an interpreter-environment tree, as
                 ``--environment-allow-sha256`` expects it).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from passage_eval.execution import timing
from passage_eval.protocol import guards
from passage_eval.supervision import DEFAULT_FORBIDDEN_MODULES, PURPOSE_DIAGNOSTIC, PURPOSES, sandbox, treehash
from passage_eval.supervision.protocol import FrameLimits
from passage_eval.supervision.supervisor import RosterInput, SplitSpec, SupervisorConfig, WarmupInput, run_supervised


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m passage_eval.supervision")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="supervised candidate execution")
    run.add_argument("--spec", required=True, type=Path, help="BenchSpec JSON/YAML")
    run.add_argument("--output", required=True, type=Path)
    run.add_argument("--worker-python", required=True, help="interpreter that runs the workers")
    run.add_argument("--isolation", required=True, choices=["bwrap", "process"])
    run.add_argument(
        "--purpose",
        default=PURPOSE_DIAGNOSTIC,
        choices=list(PURPOSES),
        help="diagnostic (default) or verified-candidate-run (needs bwrap, a pinned roster, an "
        "interpreter-environment allow-list and either split mode with a decoder allow-list or planner "
        "mode with a planner allow-list)",
    )
    run.add_argument(
        "--decoder-allow-sha256",
        action="append",
        default=[],
        metavar="HEX",
        help="operator allow-list entry: tree digest of the decoder's top-level package "
        "(see the tree-hash command); repeatable. A matching decoder runs with only that tree",
    )
    run.add_argument(
        "--planner-allow-sha256",
        action="append",
        default=[],
        metavar="HEX",
        help="operator allow-list entry: tree digest of the planner's top-level package (planner mode, "
        "e.g. a classical baseline); repeatable. A matching planner runs with only that tree",
    )
    run.add_argument(
        "--environment-allow-sha256",
        action="append",
        default=[],
        metavar="HEX",
        help="operator allow-list entry: tree digest of an interpreter-environment tree the sandbox binds "
        "(see tree-hash --environment); a verified run needs every bound environment tree listed",
    )
    run.add_argument(
        "--split-procedure-allow-sha256",
        action="append",
        default=[],
        metavar="HEX",
        help="operator-reviewed split procedure digest, including entries, kwargs and bound trees; "
        "if supplied, a mismatch fails closed",
    )
    run.add_argument(
        "--require-split-procedure-pin",
        action="store_true",
        help="require a matching operator-approved split procedure digest",
    )
    run.add_argument(
        "--attest-file",
        action="append",
        default=[],
        metavar="ROLE=/ABSOLUTE/PATH",
        help="hash a specific mounted predictor or decoder file from the staged view and bind it to "
        "the split procedure; repeatable",
    )
    run.add_argument("--roster", type=Path, help="pinned roster JSON (passage-roster-v2 or v1) naming the exact tasks")
    run.add_argument("--roster-sha256", metavar="HEX", help="pinned roster_sha256 (required with --roster)")
    run.add_argument("--roster-id", help="identifier recorded in the manifest (default: the roster's benchmark_id)")
    run.add_argument("--dataset-root", type=Path, help="pinned dataset root with PINNED.json (required with --roster)")
    run.add_argument(
        "--warmup-split", help="dataset split that provides the untimed warm-up inputs (disjoint from the roster)"
    )
    run.add_argument("--warmup-roster", type=Path, help="second pinned roster that provides the warm-up inputs")
    run.add_argument("--warmup-roster-sha256", metavar="HEX")
    guards.add_test_admission_arguments(
        run,
        what="the evaluated --roster",
        extra_flags=(("--allow-test-warmup", "a warm-up source"),),
    )
    run.add_argument(
        "--forbid-module",
        action="append",
        default=[],
        metavar="NAME",
        help=f"top-level module that must not appear in a worker (default: {', '.join(DEFAULT_FORBIDDEN_MODULES)})",
    )
    run.add_argument(
        "--no-default-forbidden-modules",
        action="store_true",
        help="do not forbid the default modules (refused for --purpose verified-candidate-run)",
    )
    run.add_argument(
        "--runtime-bind",
        action="append",
        default=[],
        metavar="PATH",
        help="extra interpreter-environment tree, bound read-only and hashed in place",
    )
    run.add_argument(
        "--tree-hash-cache",
        metavar="DIR|off",
        help="cache of environment tree digests keyed by stat signature (default: user cache for "
        "diagnostic runs; refused for verified runs, which recompute every digest)",
    )
    run.add_argument("--staging-dir", help="parent directory of the private staging area (default: system temp)")
    run.add_argument("--max-restarts", type=int, default=100)
    run.add_argument("--split-predictor", help="module:attr of the field predictor (split mode)")
    run.add_argument("--split-decoder", help="module:attr of the ordinary-input decoder (split mode)")
    run.add_argument("--split-decoder-kwargs", default="{}", help="JSON kwargs for the decoder")
    run.add_argument("--split-precision", default="unspecified")
    run.add_argument(
        "--candidate-path", action="append", default=[], help="directory added to the worker sys.path (read-only bind)"
    )
    run.add_argument(
        "--ro-bind",
        action="append",
        default=[],
        help="extra read-only path declared as a candidate artifact (staged and hashed)",
    )
    run.add_argument(
        "--dev-bind",
        action="append",
        default=[],
        help="character-device node to expose (bwrap only), e.g. /dev/nvidia0; directories, links and "
        "other file types are refused",
    )
    run.add_argument("--protect", action="append", default=[], help="path that no bind may expose")
    run.add_argument(
        "--env", action="append", default=[], metavar="NAME=VALUE", help="explicit worker environment variable"
    )
    run.add_argument("--env-passthrough", action="append", default=[], metavar="NAME")
    run.add_argument(
        "--metadata-allow",
        action="append",
        default=[],
        metavar="KEY",
        help="ordinary metadata key a candidate may see (default: none)",
    )
    run.add_argument("--rlimit-nofile", type=int, default=1024)
    run.add_argument("--rlimit-fsize-mb", type=int, default=1024)
    run.add_argument("--rlimit-as-mb", type=int)
    run.add_argument("--rlimit-cpu-s", type=int)
    run.add_argument(
        "--timing-profile",
        choices=list(timing.PROFILES),
        default=timing.LEGACY,
        help=f"what the wall budget and the scorer measure (default {timing.LEGACY}: service time plus the "
        f"supervisor's validation and recost; {timing.SERVICE}: service time, validation and recost recorded "
        f"separately; {timing.SERVICE_REWARM}: as {timing.SERVICE}, and a worker restarted inside a warm pass is "
        "warmed up again before the next measured request)",
    )
    run.add_argument(
        "--campaign-session",
        action="store_true",
        help="campaign session mode: one seed, warm requests only (no cold series), dynamic request admission "
        "within the window, session invalidation instead of task failures when a worker cannot start, and a "
        "session.json receipt (see passage_eval.supervision.session)",
    )
    run.add_argument(
        "--session-deadline-monotonic",
        type=float,
        metavar="SECONDS",
        help="end of the session's window as a time.monotonic() value of this host (campaign session only)",
    )
    run.add_argument(
        "--session-persistence-ms",
        type=float,
        default=0.0,
        help="reserve kept after the last request to persist the session (campaign session only)",
    )
    run.add_argument("--deadline-grace-ms", type=float, default=10.0)
    run.add_argument("--hard-timeout-ms", type=float, default=600_000.0)
    run.add_argument("--startup-timeout-s", type=float, default=600.0)
    run.add_argument("--ipc-baseline-pings", type=int, default=50)
    run.add_argument("--max-payload-mb", type=int, default=256)
    run.add_argument("--bwrap", default="bwrap")

    stage = commands.add_parser("stage-score", help="attach withheld references to a private copy of a run")
    stage.add_argument("--run", required=True, type=Path, help="public run directory written by `run`")
    stage.add_argument("--spec", required=True, type=Path, help="the private BenchSpec (with dataset_path)")
    stage.add_argument("--output", required=True, type=Path, help="private staging directory (must not exist)")
    stage.add_argument("--score", action="store_true", help="also run cmd_score on the staged copy")
    stage.add_argument(
        "--roster", type=Path, help="roster of a roster run (cmd_score cannot score such a run: use score-roster)"
    )
    stage.add_argument("--roster-sha256", metavar="HEX")
    stage.add_argument("--dataset-root", type=Path)

    score = commands.add_parser("score-roster", help="score a supervised roster run against its pinned roster")
    score.add_argument("--run", required=True, type=Path, help="public run directory written by `run --roster`")
    score.add_argument(
        "--roster", required=True, type=Path, help="the roster file (passage-roster-v2 or v1) the run was made with"
    )
    score.add_argument("--roster-sha256", required=True, metavar="HEX", help="the operator's pin of roster_sha256")
    score.add_argument("--dataset-root", required=True, type=Path, help="pinned dataset root with PINNED.json (labels)")
    score.add_argument("--output", required=True, type=Path, help="private output directory (must not exist)")
    score.add_argument("--roster-id", help="roster id recorded in the run (default: the roster's benchmark_id)")
    guards.add_test_admission_arguments(score, what="score-roster")

    tree = commands.add_parser("tree-hash", help="print the tree digest of a file or directory")
    tree.add_argument("path", type=Path)
    tree.add_argument("--json", action="store_true", help="print the full record")
    tree.add_argument(
        "--environment",
        action="store_true",
        help="digest as recorded for an interpreter-environment tree: nothing excluded, symbolic links "
        "recorded by target (what --environment-allow-sha256 expects)",
    )
    return parser


def _config(args: argparse.Namespace) -> SupervisorConfig:
    env_set = {}
    for item in args.env:
        name, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--env expects NAME=VALUE, got {item!r}")
        env_set[name] = value
    rlimits = sandbox.RlimitConfig(
        nofile=args.rlimit_nofile,
        fsize_bytes=args.rlimit_fsize_mb << 20,
        address_space_bytes=None if args.rlimit_as_mb is None else args.rlimit_as_mb << 20,
        cpu_seconds=args.rlimit_cpu_s,
    )
    isolation = sandbox.IsolationConfig(
        level=args.isolation,
        ro_binds=tuple(args.ro_bind),
        runtime_binds=tuple(args.runtime_bind),
        dev_binds=tuple(args.dev_bind),
        protected_paths=tuple(args.protect),
        env_set=env_set,
        env_passthrough=tuple(args.env_passthrough),
        bwrap_path=args.bwrap,
        rlimits=rlimits,
    )
    forbidden = (() if args.no_default_forbidden_modules else DEFAULT_FORBIDDEN_MODULES) + tuple(args.forbid_module)
    return SupervisorConfig(
        worker_python=os.path.abspath(args.worker_python),
        isolation=isolation,
        candidate_paths=tuple(str(Path(p).resolve()) for p in args.candidate_path),
        metadata_allow=tuple(args.metadata_allow),
        limits=FrameLimits(max_payload_bytes=args.max_payload_mb << 20),
        timing_profile=args.timing_profile,
        deadline_grace_ms=args.deadline_grace_ms,
        hard_timeout_ms=args.hard_timeout_ms,
        startup_timeout_s=args.startup_timeout_s,
        ipc_baseline_pings=args.ipc_baseline_pings,
        max_restarts=args.max_restarts,
        purpose=args.purpose,
        decoder_allow_sha256=tuple(args.decoder_allow_sha256),
        planner_allow_sha256=tuple(args.planner_allow_sha256),
        environment_allow_sha256=tuple(args.environment_allow_sha256),
        split_procedure_allow_sha256=tuple(args.split_procedure_allow_sha256),
        require_split_procedure_pin=args.require_split_procedure_pin,
        attest_files=tuple(args.attest_file),
        forbidden_modules=forbidden,
        tree_hash_cache=args.tree_hash_cache,
        staging_dir=args.staging_dir,
    )


def _roster_input(parser, roster, sha, dataset_root, roster_id=None) -> RosterInput | None:
    if roster is None:
        if sha or dataset_root:
            parser.error("--roster-sha256 and --dataset-root need --roster")
        return None
    if not sha or dataset_root is None:
        parser.error("--roster needs --roster-sha256 and --dataset-root")
    return RosterInput(str(roster), sha, str(dataset_root), roster_id)


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "tree-hash":
        if args.environment:
            record = treehash.hash_tree(args.path, role="environment", symlinks="record")
        else:
            record = treehash.hash_tree(
                args.path,
                role="decoder_entry",
                symlinks="refuse",
                strict=True,
                exclude_dirs=treehash.CANDIDATE_EXCLUDES,
            )
        print(json.dumps(record.to_manifest(), indent=2, sort_keys=True) if args.json else record.sha256)
        return 0
    if args.command == "score-roster":
        from passage_eval.supervision.roster_scoring import score_roster_run

        admission = guards.test_split_allowed(
            args.allow_test_split, args.test_admission_record, evidence_dir=args.test_admission_evidence_dir
        )
        for path in score_roster_run(
            args.run,
            args.roster,
            args.roster_sha256,
            args.dataset_root,
            args.output,
            roster_id=args.roster_id,
            allow_test=admission is not None,
            admission=admission,
        ):
            print(path)
        return 0
    from passage_eval.cli import BenchSpec

    spec = BenchSpec.from_file(args.spec)
    if args.command == "stage-score":
        from passage_eval.supervision.staging import stage_for_scoring

        roster = _roster_input(parser, args.roster, args.roster_sha256, args.dataset_root)
        if roster is not None and args.score:
            parser.error(
                "a roster run cannot be scored by the unchanged cmd_score, which reloads scenes from the "
                "spec, not from the roster; stage without --score, or score it with the score-roster command"
            )
        staged = stage_for_scoring(args.run, spec, args.output, roster=roster)
        if args.score:
            from passage_eval.cli import cmd_score

            cmd_score(args.output)
        for path in staged:
            print(path)
        return 0
    if bool(args.split_predictor) != bool(args.split_decoder):
        parser.error("--split-predictor and --split-decoder must be given together")
    split = None
    if args.split_predictor:
        split = SplitSpec(
            args.split_predictor, args.split_decoder, json.loads(args.split_decoder_kwargs), args.split_precision
        )
    roster = _roster_input(parser, args.roster, args.roster_sha256, args.dataset_root, args.roster_id)
    warmup = None
    if args.warmup_split and args.warmup_roster:
        parser.error("give --warmup-split or --warmup-roster, not both")
    wanted = args.allow_test_split or args.allow_test_warmup
    admission = guards.test_split_allowed(
        wanted,
        args.test_admission_record,
        evidence_dir=args.test_admission_evidence_dir,
        flag_name="--allow-test-split or --allow-test-warmup",
    )
    warmup_admission = admission if args.allow_test_warmup else None
    if args.warmup_split:
        warmup = WarmupInput(
            split=args.warmup_split, allow_test=warmup_admission is not None, admission=warmup_admission
        )
    elif args.warmup_roster:
        if not args.warmup_roster_sha256 or args.dataset_root is None:
            parser.error("--warmup-roster needs --warmup-roster-sha256 and --dataset-root")
        warmup = WarmupInput(
            roster=RosterInput(str(args.warmup_roster), args.warmup_roster_sha256, str(args.dataset_root)),
            allow_test=warmup_admission is not None,
            admission=warmup_admission,
        )
    run_admission = admission if args.allow_test_split else None
    session = None
    if args.campaign_session:
        from passage_eval.supervision.session import SessionPolicy

        session = SessionPolicy(args.session_deadline_monotonic, args.session_persistence_ms)
    elif args.session_deadline_monotonic is not None or args.session_persistence_ms:
        parser.error("--session-deadline-monotonic and --session-persistence-ms need --campaign-session")
    for path in run_supervised(
        spec,
        args.output,
        _config(args),
        split=split,
        roster=roster,
        warmup=warmup,
        allow_test=run_admission is not None,
        admission=run_admission,
        session=session,
    ):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
