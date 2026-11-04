# MIT License
#
# Copyright (c) 2026 THALES

"""Execute and resume the batches of a sealed campaign plan on the host: windows, dynamic admission and R7 relaunches.

Units, windows and sessions. Every batch of the partition is a *unit* (``batch-000003``); a unit that R7 splits has two
halves (``batch-000003.0``, ``batch-000003.1``). An *attempt* of a unit runs in one or more *windows* of
``lease_or_timeout_ms``. In a window, the procedures that still have tasks run in the paired order of their batch
(``protocol.warmup.paired_order(schedule_seed, batch_index, procedures)``); each is one supervised *session*
(``python -I -B -m passage_eval.supervision run --campaign-session``: warm requests only, the warm-up roster, the
re-warm profile, every limit of the plan) whose admission deadline is its equal share of what remains of the window.
The supervisor engages a request only while the window holds its budget, grace, kill wait and persistence
(:mod:`passage_eval.supervision.session`); otherwise it closes the session cleanly and the runner carries the
deferred tasks, in order, into the procedure's next window, through a continuation roster derived from the admissible
roster. The kill bound of a session is the end of the window; the scoring (``score-roster``) of the window's sessions
runs after the window, on the host, with its own bound.

Receipts. ``campaign.json`` binds the output directory to one plan; a lock forbids two runners. An attempt writes
``intent.json`` (plan, unit, attempt, effect key, paired order) before any work and ``completion.json`` after every
session's score receipts are verified. ``invocations.jsonl`` records every invocation.

R7 relaunch (automatic). An attempt is *ambiguous* when a session is killed by the window bound, its transport or lease
is lost (``TransportLost``), it is invalid (a worker cannot start, the initial warm-up failed), or the runner was
interrupted. The attempt is then archived intact under ``relaunched/`` with the SHA-256 of every file
(``attempt-archive.json``), never read for results and never scored. After the drain proof and the lease
verification of the infrastructure (``HostInfrastructure`` here; pluggable) and with a new effect key, the unit is
relaunched identically, at most twice. After a third ambiguous attempt it is split deterministically in two halves.
A single task whose session was killed or invalid twice with green infrastructure checks becomes a candidate failure
for that procedure (``runtime_error``, canonical penalty, latency charged), so a candidate that cannot start never
leaves the denominator. A transport or lease loss is never imputed to the candidate. When more than 5% of the
partition's batches have been relaunched (deliberate kills excluded), the campaign stops for diagnosis. Only the
first complete attempt of a unit is scored. Every decision is appended to ``relaunch-log.jsonl``.

A harness error (a supervisor that exits nonzero, a refused score receipt) writes ``failure.json`` and stops the
campaign; only ``relaunch-batch`` (an operator, a reason, the ``infrastructure`` reason class) retries such an attempt.
A deliberate kill (``deliberate_kills``, validation only) exercises the R7 rule on purpose in a rehearsal; it is
logged as deliberate and excluded from the 5% rule and from candidate failures.

An edge target raises ``EdgeTransportNotQualified``: no remote transport exists in this harness.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import json
import math
import os
import secrets
import signal
import subprocess
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from passage_eval.campaign import (
    COMPLETION_SCHEMA,
    FAILURE_SCHEMA,
    IMPUTED_SCHEMA,
    INTENT_SCHEMA,
    PLAN_SCHEMA,
    RELAUNCH_SCHEMA,
    SPLIT_SCHEMA,
    STATUS_SCHEMA,
    EdgeTransportNotQualified,
)
from passage_eval.campaign import partition as partition_io
from passage_eval.campaign.plan import (
    CampaignPlan,
    ValidationEdgeRelayRequest,
    check_validation_edge_relay_request,
    read_plan,
)
from passage_eval.io import pinned
from passage_eval.io import roster as roster_io

IDENTITY_FILE = "campaign.json"
INVOCATIONS_FILE = "invocations.jsonl"
RELAUNCH_LOG = "relaunch-log.jsonl"
SPLIT_LOG = "split-log.jsonl"
RELAUNCHED_DIR = "relaunched"
SPLITS_DIR = "splits"
IMPUTED_DIR = "imputed"
ARCHIVE_MANIFEST = "attempt-archive.json"
SCORE_FILES = (
    ("run_manifest_sha256", "run_manifest.json"),
    ("per_sample_sha256", "per_sample.npz"),
    ("metrics_sha256", "metrics.json"),
    ("reference_inputs_sha256", "score_reference_inputs.json"),
)
#: Ambiguity classes (R7).
KILLED, INVALID_SESSION, TRANSPORT, INTERRUPTED, DELIBERATE = (
    "killed_by_window",
    "invalid_session",
    "transport_or_lease_loss",
    "interrupted",
    "deliberate_kill",
)
AMBIGUOUS = (KILLED, INVALID_SESSION, TRANSPORT, INTERRUPTED, DELIBERATE)
#: Stops that a resume does not lift by itself: an operator ``relaunch-batch`` must decide.
OPERATOR_STOPS = ("stop_single_task_not_attributable",)


class IncompleteBatch(RuntimeError):
    """An attempt stopped on a harness error; it needs inspection and an explicit ``relaunch-batch``."""


class CampaignStopped(RuntimeError):
    """The campaign stops for diagnosis (R7 relaunch fraction, failed drain proof, no progress, unsplittable task)."""


class TransportLost(RuntimeError):
    """The transport or the lease of a session was lost: an ambiguous attempt, never imputed to the candidate."""


@dataclass(frozen=True)
class ValidationEdgeRelayResult:
    """Host-only scores of the fixed pre-G3 validation rehearsal."""

    target: str
    seed: int
    plan_sha256: str
    scores: dict[str, Any]


def run_validation_edge_relay(request: ValidationEdgeRelayRequest) -> ValidationEdgeRelayResult:
    """Collect and score the two sealed validation packages without consulting a campaign plan or roster.

    The ordinary ``read_plan`` route must remain the only route for a test campaign.  This entry exists solely to
    establish G3 evidence without circularly requiring G3 first.
    """
    from passage_eval.campaign.edge.collect import score_collected
    from passage_eval.campaign.edge.package import InputPackage, PackageView, verify_package
    from passage_eval.campaign.edge.targets import frozen_candidate_identity
    from passage_eval.campaign.edge.validation_control import (
        CONTROL_SCHEMA,
        ValidationRelayControl,
        validation_relay_control,
    )
    from passage_eval.io import roster as roster_io
    from passage_eval.protocol.guards import refuse_test_roster

    check_validation_edge_relay_request(request)
    control = validation_relay_control()
    if not isinstance(control, ValidationRelayControl) or control.schema != CONTROL_SCHEMA:
        raise ValueError("validation edge relay is unconfigured: no reviewed validation control is installed")
    try:
        raw_roster = pinned.read_pinned(control.roster_path, control.roster_file_sha256)
        validation_roster = roster_io.validate_roster(
            pinned.strict_json_loads(raw_roster, name=control.roster_path.name),
            expected_sha256=control.roster_sha256,
        )
        refuse_test_roster(validation_roster, context="the validation edge relay")
        if validation_roster["dataset"]["split"] != "validation":
            raise ValueError("validation edge relay roster split differs")
        if validation_roster["dataset_root_identity"]["sha256"] != control.pinned_receipt_sha256:
            raise ValueError("validation edge relay control root receipt differs from its roster")
        from passage_eval.campaign.plan import check_source_paths

        check_source_paths({"dataset_root": str(control.dataset_root)}, validation_roster)
        scenes = roster_io.load_roster_scenes(validation_roster, control.dataset_root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"validation edge relay roster is invalid: {exc}") from exc
    expected_inputs = {
        scene.sample_id: {
            "terrain": scene.elevation.tolist(),
            "obstacles": scene.obstacles.tolist(),
            "starts": list(scene.start),
            "goals": list(scene.goal),
            "input_tensor": scene.input_tensor.tolist(),
        }
        for scene in scenes
    }
    views: dict[str, PackageView] = {}
    for procedure in request.procedures:
        package = request.packages[procedure]
        if not isinstance(package, (InputPackage, PackageView)):
            raise ValueError(f"{procedure}: validation edge relay needs a sealed input package")
        view = verify_package(package.root, package.sha256)
        if view.manifest["target"] != request.target or view.manifest["procedure"] != procedure:
            raise ValueError(f"{procedure}: sealed package differs from the validation relay scope")
        binding = view.manifest.get("validation")
        if not isinstance(binding, Mapping) or binding.get("roster_sha256") != validation_roster["roster_sha256"]:
            raise ValueError(f"{procedure}: package is not bound to the trusted validation roster")
        if binding.get("sample_ids_sha256") != pinned.canonical_sha256(view.inputs["sample_ids"]) or not set(
            view.inputs["sample_ids"]
        ) <= set(expected_inputs):
            raise ValueError(f"{procedure}: package inputs are not bound to the trusted validation roster")
        for index, sample_id in enumerate(view.inputs["sample_ids"]):
            packaged = {
                name: view.inputs[name][index] for name in ("terrain", "obstacles", "starts", "goals", "input_tensor")
            }
            if pinned.canonical_json(packaged) != pinned.canonical_json(expected_inputs[sample_id]):
                raise ValueError(f"{procedure}: package ordinary inputs differ from the trusted validation roster")
        candidate = view.manifest.get("candidate")
        expected_candidate = frozen_candidate_identity(procedure) if request.seed == 2026 else frozen_candidate_identity(
            procedure,seed=request.seed,replica_catalog=request.replica_catalog,replica_catalog_sha256=request.replica_catalog_sha256)
        expected_binding = {key: expected_candidate[key] for key in ("checkpoint_sha256", "seed")}
        if request.candidates[procedure] != expected_binding:
            raise ValueError(f"{procedure}: request candidate differs from the frozen candidate")
        if candidate != expected_binding:
            raise ValueError(f"{procedure}: package candidate differs from the frozen candidate")
        views[procedure] = view
    plan_sha256s = {view.manifest["plan_sha256"] for view in views.values()}
    if len(plan_sha256s) != 1:
        raise ValueError("validation edge relay packages do not bind one plan")

    scores: dict[str, Any] = {}
    for procedure in request.procedures:
        package = request.packages[procedure]
        result_root = request.native_relay(procedure, package)
        if not isinstance(result_root, (str, Path)):
            raise ValueError(f"{procedure}: native relay did not return a collected result directory")
        scores[procedure] = score_collected(
            result_root,
            package=package,
            scorer=lambda outputs, procedure=procedure: request.scorer(procedure, outputs),
        )
    return ValidationEdgeRelayResult(
        target=request.target,
        seed=request.seed,
        plan_sha256=plan_sha256s.pop(),
        scores=scores,
    )


def run_admitted_edge_relay(
    plan_path: str | Path,
    plan_sha256: str,
    *,
    allow_test: bool = False,
    native_relay: Callable[[CampaignPlan], Any],
) -> Any:
    """Dispatch an edge relay only after the normal plan reader has admitted its exact target.

    ``read_plan`` performs the all-green test admission and per-target G3 checks before it opens a roster or a
    partition.  The injected relay is the later leased-driver boundary (Task 3), not a pre-G3 escape hatch.
    """
    if not callable(native_relay):
        raise ValueError("an admitted edge relay needs a supplied native relay")
    campaign = read_plan(plan_path, plan_sha256, allow_test=allow_test)
    if campaign.plan["target"] == "host":
        raise EdgeTransportNotQualified("an admitted edge relay requires Thor, Orin, or Tiger")
    return native_relay(campaign)


class _Ambiguous(Exception):
    def __init__(self, classification: str, procedure: str | None, detail: str = ""):
        super().__init__(f"{classification} ({procedure}): {detail}")
        self.classification, self.procedure, self.detail = classification, procedure, detail


class HostInfrastructure:
    """Infrastructure checks of the host target, where the window is a local timeout and there is no lease.

    ``drain_proof`` returns ``ok`` when no process still names the attempt directory in its command line (the
    supervisor names its output there; its workers die with it through their parent-death signal). An edge transport
    replaces this class with its lease probe, scope and accelerator checks (not implemented in this harness).
    """

    kind = "host"

    def drain_proof(self, attempt_dir: Path) -> dict[str, Any]:
        survivors = _processes_naming(attempt_dir)
        return {
            "ok": not survivors,
            "kind": "host_process_scan",
            "processes_naming_the_attempt": len(survivors),
            "accelerator": "not checked: CPU host",
        }

    def verify_lease(self) -> dict[str, Any]:
        return {"ok": True, "kind": "host_timeout_window", "lease": None}

    def new_effect_key(self) -> str:
        return secrets.token_hex(16)


def _processes_naming(path: Path) -> list[int]:
    needle = os.fsencode(str(path))
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            cmdline = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if needle in cmdline:
            found.append(int(entry.name))
    return found


@dataclass(frozen=True)
class Unit:
    """A batch of the partition, or one of the deterministic halves R7 splits it into."""

    unit_id: str
    batch_index: int
    start: int
    stop: int
    roster: dict[str, str]  # path, file_sha256, roster_sha256
    sample_ids: tuple[str, ...]

    @property
    def directory(self) -> str:
        return f"batch-{self.unit_id}"


def _now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    line = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600)
    try:
        os.write(fd, line)
        os.fsync(fd)
    finally:
        os.close(fd)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in pinned.read_bounded(path).decode("utf-8").splitlines() if line.strip()]


def run_command(argv: Sequence[str], timeout_seconds: float, log: Path) -> None:
    """Run one bounded local process group; its log stays private. A timeout or a survivor is an error."""
    with log.open("xb") as output:
        process = subprocess.Popen(
            list(argv), stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                pass
            # The leader may exit on SIGTERM while a descendant ignores it: always kill the rest of the group.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            if isinstance(error, subprocess.TimeoutExpired):
                raise TimeoutError("batch subprocess timed out; inspect before any relaunch") from error
            raise
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            # A launcher that exits while group members survive cannot establish a completed batch.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            raise RuntimeError("batch subprocess left child processes; inspect before any relaunch")
    if returncode != 0:
        raise RuntimeError(f"batch subprocess exited {returncode}; inspect before any relaunch")


def _admission_args(campaign: CampaignPlan) -> list[str]:
    if campaign.admission is None:
        return []
    admission = campaign.plan["admission"]
    return [
        "--allow-test-split",
        "--test-admission-record",
        admission["record"],
        "--test-admission-evidence-dir",
        admission["evidence_dir"],
    ]


def build_commands(
    campaign: CampaignPlan,
    roster: Mapping[str, str],
    procedure_dir: Path,
    procedure: str,
    *,
    deadline_monotonic: float | None = None,
):
    """The supervised session and the roster score of one procedure over one roster (a batch or a continuation)."""
    plan = campaign.plan
    if plan["target"] != "host":
        raise EdgeTransportNotQualified(
            f"edge transport not qualified: no remote transport to {plan['target']!r} is implemented in this harness"
        )
    p, t, w = plan["procedures"][procedure], plan["timing"], plan["warmup"]
    common = [
        "--roster",
        roster["path"],
        "--roster-sha256",
        roster["roster_sha256"],
        "--dataset-root",
        plan["dataset_root"],
    ]
    prefix = [plan["harness"]["python"], "-I", "-B", "-m", "passage_eval.supervision"]
    run = prefix + [
        "run",
        "--spec",
        p["spec"]["path"],
        "--output",
        str(procedure_dir / "run"),
        "--worker-python",
        plan["harness"]["worker_python"],
        "--isolation",
        "bwrap",
        "--purpose",
        plan["purpose"],
        "--campaign-session",
        "--session-persistence-ms",
        repr(float(plan["batching"]["persistence_ms"])),
        "--timing-profile",
        t["profile"],
        "--deadline-grace-ms",
        repr(float(t["deadline_grace_ms"])),
        "--hard-timeout-ms",
        repr(float(t["hard_timeout_ms"])),
        "--startup-timeout-s",
        repr(float(t["startup_timeout_s"])),
        "--max-restarts",
        str(t["max_restarts"]),
        "--rlimit-nofile",
        str(t["rlimits"]["nofile"]),
        "--rlimit-fsize-mb",
        str(t["rlimits"]["fsize_mb"]),
        "--warmup-roster",
        w["roster"]["path"],
        "--warmup-roster-sha256",
        w["roster"]["roster_sha256"],
        *common,
    ]
    if deadline_monotonic is not None:
        run += ["--session-deadline-monotonic", repr(float(deadline_monotonic))]
    if t["rlimits"]["as_mb"] is not None:
        run += ["--rlimit-as-mb", str(t["rlimits"]["as_mb"])]
    if t["rlimits"]["cpu_s"] is not None:
        run += ["--rlimit-cpu-s", str(t["rlimits"]["cpu_s"])]
    if p["mode"] == "split":
        run += [
            "--split-predictor",
            p["predictor"],
            "--split-decoder",
            p["decoder"],
            "--split-decoder-kwargs",
            json.dumps(p["decoder_kwargs"], sort_keys=True),
            "--split-precision",
            p["precision"],
            "--require-split-procedure-pin",
        ]
        for digest in p["decoder_allow_sha256"]:
            run += ["--decoder-allow-sha256", digest]
        for digest in p["split_procedure_allow_sha256"]:
            run += ["--split-procedure-allow-sha256", digest]
    else:
        for digest in p["planner_allow_sha256"]:
            run += ["--planner-allow-sha256", digest]
    for digest in p["environment_allow_sha256"]:
        run += ["--environment-allow-sha256", digest]
    for bind in p["runtime_binds"]:
        run += ["--runtime-bind", bind]
    for candidate in p["candidate_paths"]:
        run += ["--candidate-path", candidate]
    run += _admission_args(campaign)
    score = prefix + [
        "score-roster",
        "--run",
        str(procedure_dir / "run"),
        "--output",
        str(procedure_dir / "scored"),
        *common,
        *_admission_args(campaign),
    ]
    return run, score


def _expected(campaign: CampaignPlan, roster: Mapping[str, str], procedure: str, executed: Sequence[str]):
    """What the score receipts of one session must show: the session roster, its measured prefix, every limit."""
    from passage_eval.supervision.session import SESSION_POLICY_ID

    plan = campaign.plan
    p = plan["procedures"][procedure]
    return {
        "procedure": procedure,
        # The digests the supervisor may record for the checkpoint and bundle files that the worker loads
        # (``checkpoint_hashes``): the plan's frozen checkpoint or bundle for Seg and Reg, none for a planner.
        "executed_artifacts": sorted({p["checkpoint_sha256"], p["bundle_sha256"]}) if p["mode"] == "split" else [],
        "split": plan["split"],
        "seeds": list(plan["seeds"]),
        "child_roster_sha256": roster["roster_sha256"],
        "sample_ids": list(executed),
        "bench_specs": {seed: campaign.seed_spec(procedure, seed) for seed in plan["seeds"]},
        "timing": plan["timing"],
        "budget": plan["budget"],
        "warmup_source": f"roster:{plan['warmup']['roster']['roster_sha256']}",
        "source_sha256": plan["harness"]["source_sha256"],
        "purpose": plan["purpose"],
        "session_policy": SESSION_POLICY_ID,
    }


def _limits_match(limits: Any, expected: Mapping[str, Any]) -> bool:
    if not isinstance(limits, Mapping):
        return False
    t = expected["timing"]
    rlimits = limits.get("worker_rlimits") or {}
    return (
        limits.get("budget") == expected["budget"]
        and limits.get("timing_profile") == t["profile"]
        and limits.get("timing_boundary") == t["boundary_id"]
        and limits.get("deadline_grace_ms") == t["deadline_grace_ms"]
        and limits.get("hard_timeout_ms") == t["hard_timeout_ms"]
        and limits.get("startup_timeout_s") == t["startup_timeout_s"]
        and limits.get("max_restarts") == t["max_restarts"]
        and rlimits.get("nofile") == t["rlimits"]["nofile"]
        and rlimits.get("fsize_bytes") == t["rlimits"]["fsize_mb"] << 20
        and rlimits.get("address_space_bytes")
        == (None if t["rlimits"]["as_mb"] is None else t["rlimits"]["as_mb"] << 20)
        and rlimits.get("cpu_seconds") == t["rlimits"]["cpu_s"]
    )


def verify_scored(procedure_dir: Path, expected: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Check the native score receipts of one session and bind them to the plan, the session roster and its prefix.

    One warm scored series per seed is required: a campaign session measures warm requests only (no cold series). The
    manifest must come from the campaign session mode and declare the measured prefix. The checkpoint and bundle files
    that the worker loaded must be the plan's frozen ones (``_executed_artifacts``); their digests are kept in the
    completion receipt.
    """
    from passage_eval.execution import read_run_artifact

    root = procedure_dir / "scored"
    receipts = sorted(root.rglob("score_receipt.json"))
    wanted = {(seed, "warm") for seed in expected["seeds"]}
    if len(receipts) != len(wanted):
        raise ValueError(f"{expected['procedure']}: expected {len(wanted)} score receipts, found {len(receipts)}")
    records, seen = [], set()
    for receipt_path in receipts:
        directory = receipt_path.parent
        receipt = pinned.read_json(receipt_path)
        manifest = pinned.read_json(directory / "run_manifest.json")
        artifact = read_run_artifact(directory)
        artifact.validate()
        seed, series = manifest.get("seed"), manifest.get("series")
        supervision = manifest.get("supervision") if isinstance(manifest.get("supervision"), Mapping) else {}
        procedure_block = (manifest.get("procedure") or {}).get("supervision") or {}
        session = supervision.get("session") if isinstance(supervision.get("session"), Mapping) else {}
        if (
            receipt.get("schema") != "passage-offline-score-v1"
            or receipt.get("qualified_reference") is not True
            or receipt.get("metric_contract") != "passage-qualified-complete-v1"
            or receipt.get("scored_records") != len(artifact.records)
            or receipt.get("scored_repeats") != manifest.get("repeats")
            or (seed, series) not in wanted
            or (seed, series) in seen
            or manifest.get("repeats") != 1
            or manifest.get("quality_repeat_index") != 0
            or (manifest.get("dataset") or {}).get("split") != expected["split"]
            or manifest.get("bench_spec") != expected["bench_specs"][seed]
            or (supervision.get("roster") or {}).get("roster_sha256") != expected["child_roster_sha256"]
            or manifest.get("roster") != expected["sample_ids"]
            or session.get("executed_sample_ids") != expected["sample_ids"]
            or procedure_block.get("session_policy") != expected["session_policy"]
            or manifest.get("timing_profile") != expected["timing"]["profile"]
            or manifest.get("timing_boundary") != expected["timing"]["boundary_id"]
            or manifest.get("runtime_source_hash") != expected["source_sha256"]
            or (supervision.get("purpose") or {}).get("name") != expected["purpose"]
            or not _limits_match(procedure_block.get("limits"), expected)
            or (procedure_block.get("latency_protocol") or {}).get("warmup_source") != expected["warmup_source"]
        ):
            raise ValueError(
                f"{expected['procedure']}: scored artifact identity, limits, reference or coverage mismatch"
            )
        seen.add((seed, series))
        for field, name in SCORE_FILES:
            if receipt.get(field) != pinned.sha256_bytes(pinned.read_bounded(directory / name)):
                raise ValueError(f"{expected['procedure']}: scored artifact hash mismatch: {name}")
        procedure_id = manifest.get("procedure_id")
        if not pinned.is_hex64(procedure_id):
            raise ValueError(f"{expected['procedure']}: missing procedure identity")
        executed = _executed_artifacts(manifest, expected)
        records.append(
            {
                "path": str(receipt_path.relative_to(procedure_dir)),
                "sha256": pinned.sha256_bytes(pinned.read_bounded(receipt_path)),
                "seed": seed,
                "series": series,
                "procedure_id": procedure_id,
                "executed_artifact_sha256": executed,
            }
        )
    return records


def _executed_artifacts(manifest: Mapping[str, Any], expected: Mapping[str, Any]) -> list[str]:
    """The checkpoint and bundle digests the run loaded, refused unless they are the plan's frozen ones.

    The supervisor hashes every file that a ``checkpoint`` or ``bundle`` keyword of the worker names and records
    the digests in ``checkpoint_hashes`` (top level and procedure block, which ``procedure_id`` covers). A split
    procedure must have loaded at least one file, and only the plan's checkpoint or bundle; a planner none.
    """
    hashes = manifest.get("checkpoint_hashes")
    procedure_hashes = (manifest.get("procedure") or {}).get("checkpoint_hashes")
    allowed = list(expected["executed_artifacts"])
    if not isinstance(hashes, Mapping) or hashes != procedure_hashes:
        raise ValueError(f"{expected['procedure']}: the run manifest records no consistent checkpoint_hashes")
    executed = sorted(set(map(str, hashes.values())))
    if (allowed and not executed) or not set(executed) <= set(allowed):
        raise ValueError(
            f"{expected['procedure']}: the executed checkpoint or bundle files {executed} are not the plan's frozen "
            f"candidate artifacts {allowed}"
        )
    return executed


# ------------------------------------------------------------------------------------------------ units and rosters


def root_unit(campaign: CampaignPlan, item: Mapping[str, Any]) -> Unit:
    return Unit(
        unit_id=f"{item['index']:06d}",
        batch_index=item["index"],
        start=item["start"],
        stop=item["stop"],
        roster={
            "path": str(Path(campaign.plan["partition"]["dir"]) / item["file"]),
            "file_sha256": item["file_sha256"],
            "roster_sha256": item["roster_sha256"],
        },
        sample_ids=tuple(row["sample_id"] for row in item["parent_mapping"]),
    )


def _write_derived(path: Path, child: Mapping[str, Any]) -> dict[str, str]:
    """Write a derived roster once (later calls must find identical bytes) and return its pin."""
    raw = roster_io.canonical_json(child)
    if path.exists():
        if pinned.read_bounded(path) != raw:
            raise ValueError(f"{path.name} differs from the roster derived again from the admissible roster")
    else:
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        with path.open("xb") as target:
            target.write(raw)
    return {"path": str(path), "file_sha256": pinned.sha256_bytes(raw), "roster_sha256": child["roster_sha256"]}


def _split(campaign: CampaignPlan, output: Path, unit: Unit) -> tuple[Unit, Unit]:
    """The two deterministic halves of a unit: parent ranks ``[start, middle)`` and ``[middle, stop)``."""
    middle = unit.start + (unit.stop - unit.start + 1) // 2
    halves = []
    for suffix, (start, stop) in enumerate(((unit.start, middle), (middle, unit.stop))):
        unit_id = f"{unit.unit_id}.{suffix}"
        ids = [task["sample_id"] for task in campaign.admissible["tasks"][start:stop]]
        child = roster_io.derive_roster(
            campaign.admissible,
            ids,
            rule=f"R7 split: contiguous parent ranks [{start}, {stop}) of the pinned admissible roster.",
            derivation={"kind": "campaign_r7_split", "batch_index": unit.batch_index, "unit": unit_id},
        )
        pin = _write_derived(output / SPLITS_DIR / f"batch-{unit_id}.json", child)
        halves.append(Unit(unit_id, unit.batch_index, start, stop, pin, tuple(ids)))
    return halves[0], halves[1]


def _continuation(campaign: CampaignPlan, path: Path, unit: Unit, procedure: str, window: int, ids) -> dict:
    child = roster_io.derive_roster(
        campaign.admissible,
        list(ids),
        rule="Tasks deferred by dynamic admission, carried in order into the next window of their unit.",
        derivation={
            "kind": "campaign_continuation",
            "batch_index": unit.batch_index,
            "unit": unit.unit_id,
            "procedure": procedure,
            "window": window,
        },
    )
    return _write_derived(path, child)


def _children(output: Path, unit_id: str) -> list[str] | None:
    for row in _read_jsonl(output / SPLIT_LOG):
        if row.get("unit") == unit_id:
            return list(row["children"])
    return None


def unit_leaves(campaign: CampaignPlan, output: Path, item: Mapping[str, Any]) -> list[Unit]:
    """The units of one partition batch in parent order, following its recorded R7 splits."""
    leaves: list[Unit] = []

    def walk(unit: Unit) -> None:
        children = _children(output, unit.unit_id)
        if children is None:
            leaves.append(unit)
            return
        for child in _split(campaign, output, unit):
            walk(child)

    walk(root_unit(campaign, item))
    return leaves


# ------------------------------------------------------------------------------------------------- receipts


def _batch_metadata(campaign: CampaignPlan, unit: Unit, order: list[str]) -> dict[str, Any]:
    plan = campaign.plan
    return {
        "plan_sha256": campaign.sha256,
        "campaign_id": plan["campaign_id"],
        "split": plan["split"],
        "target": plan["target"],
        "verified": False,
        "batch_index": unit.batch_index,
        "unit": unit.unit_id,
        "parent_ranks": [unit.start, unit.stop],
        "child_roster_sha256": unit.roster["roster_sha256"],
        "child_file_sha256": unit.roster["file_sha256"],
        "admissible_roster_sha256": plan["rosters"]["admissible"]["roster_sha256"],
        "partition_manifest_sha256": plan["partition"]["manifest_sha256"],
        "paired_order": order,
        "seeds": list(plan["seeds"]),
    }


def _order(campaign: CampaignPlan, index: int) -> list[str]:
    from passage_eval.protocol.warmup import paired_order

    return paired_order(campaign.plan["order"]["schedule_seed"], index, list(campaign.procedures))


def _identity(campaign: CampaignPlan) -> dict[str, Any]:
    plan = campaign.plan
    return {
        "schema": PLAN_SCHEMA,
        "campaign_id": plan["campaign_id"],
        "plan_sha256": campaign.sha256,
        "split": plan["split"],
        "target": plan["target"],
        "verified": False,
    }


def _unit_log(output: Path, unit_id: str) -> list[dict[str, Any]]:
    return [row for row in _read_jsonl(output / RELAUNCH_LOG) if row.get("unit") == unit_id]


def _attempt(output: Path, unit_id: str) -> int:
    return len(_unit_log(output, unit_id)) + 1


def _check_not_stopped(output: Path, unit: Unit) -> None:
    log = _unit_log(output, unit.unit_id)
    if log and log[-1].get("decision") in OPERATOR_STOPS:
        raise CampaignStopped(
            f"unit {unit.unit_id} stopped for diagnosis ({log[-1]['decision']}); an operator relaunch-batch is required"
        )


def _imputed(output: Path, unit: Unit) -> dict[str, dict[str, Any]]:
    found = {}
    directory = output / IMPUTED_DIR
    if directory.is_dir():
        for path in sorted(directory.glob(f"{unit.directory}--*.json")):
            row = pinned.read_json(path)
            body = {key: value for key, value in row.items() if key != "sha256"}
            if row.get("schema") != IMPUTED_SCHEMA or row.get("sha256") != pinned.canonical_sha256(body):
                raise ValueError(f"{path.name}: the imputed failure record changed")
            found[row["procedure"]] = row
    return found


def _relaunched_batches(output: Path) -> set[int]:
    """Partition batches with at least one R7 relaunch decision that was not a deliberate kill."""
    return {
        row["batch_index"]
        for row in _read_jsonl(output / RELAUNCH_LOG)
        if row.get("kind") == "r7" and row.get("classification") != DELIBERATE
    }


def _r7_policy(campaign: CampaignPlan) -> Mapping[str, Any]:
    return campaign.plan["r7_policy"]


def _check_relaunch_fraction(output: Path, n_batches: int, policy: Mapping[str, Any]) -> None:
    relaunched = _relaunched_batches(output)
    if len(relaunched) > policy["relaunch_stop_fraction"] * n_batches:
        raise CampaignStopped(
            f"{len(relaunched)} of {n_batches} batches were relaunched (more than "
            f"{policy['relaunch_stop_fraction']:.0%}): the campaign of this target stops for diagnosis"
        )


def _archive(output: Path, unit: Unit, attempt: int) -> dict[str, Any]:
    """Move an ambiguous attempt intact under ``relaunched/`` and hash every file it holds (never read for results)."""
    batch_dir = output / unit.directory
    archive_root = output / RELAUNCHED_DIR
    archive_root.mkdir(mode=0o700, exist_ok=True)
    archived = archive_root / f"{unit.directory}-attempt-{attempt:02d}"
    if archived.exists():
        raise FileExistsError(archived)
    os.rename(batch_dir, archived)
    files = {
        str(path.relative_to(archived)): pinned.sha256_bytes(path.read_bytes())
        for path in sorted(archived.rglob("*"))
        if path.is_file() and not path.is_symlink()
    }
    manifest = {"unit": unit.unit_id, "attempt": attempt, "scored": False, "files": files}
    pinned.write_json_exclusive(archived / ARCHIVE_MANIFEST, manifest)
    return {
        "archived_to": str(archived.relative_to(output)),
        "archive_sha256": pinned.canonical_sha256(manifest),
        "n_files": len(files),
    }


# ------------------------------------------------------------------------------------------------ one attempt


def _session_receipt(run_dir: Path, queue: Sequence[str], procedure: str) -> dict[str, Any]:
    from passage_eval.supervision import session as session_mod

    path = run_dir / session_mod.SESSION_RECEIPT
    if not path.is_file():
        raise ValueError(f"{procedure}: the session wrote no receipt")
    receipt = pinned.read_json(path)
    executed, deferred = receipt.get("executed_sample_ids"), receipt.get("deferred_sample_ids")
    status = receipt.get("status")
    if receipt.get("schema") != session_mod.SESSION_RECEIPT_SCHEMA or status not in session_mod.STATUSES:
        raise ValueError(f"{procedure}: invalid session receipt")
    if status == session_mod.INVALID:
        return receipt
    if (
        not isinstance(executed, list)
        or not isinstance(deferred, list)
        or list(queue) != executed + deferred
        or (status == session_mod.COMPLETE and deferred)
        or (status == session_mod.CLOSED and not deferred)
        or receipt.get("run_artifact") != (f"seed-{receipt.get('seed')}" if executed else None)
    ):
        raise ValueError(f"{procedure}: the session receipt does not split its roster into a measured prefix")
    return receipt


def _execute_attempt(
    campaign: CampaignPlan,
    output: Path,
    unit: Unit,
    attempt: int,
    effect_key: str,
    procedures: list[str],
    imputed: Mapping[str, Mapping[str, Any]],
    *,
    command: Callable[[Sequence[str], float, Path], None],
    infrastructure: Any,
    deliberate: Mapping[str, float],
) -> dict[str, Any]:
    """Run one attempt of a unit, window after window, until every procedure has measured every task."""
    from passage_eval.supervision import session as session_mod

    plan = campaign.plan
    lease_s = plan["batching"]["lease_or_timeout_ms"] / 1000.0
    batch_dir = output / unit.directory
    queues = {p: list(unit.sample_ids) for p in procedures}
    windows = []
    window = 0
    while any(queues.values()):
        window += 1
        if window > len(unit.sample_ids) + 1:
            raise CampaignStopped(f"unit {unit.unit_id}: more windows than tasks; the plan's window is too short")
        lease = infrastructure.verify_lease()
        if not lease.get("ok"):
            raise _Ambiguous(TRANSPORT, None, f"lease not verified at window {window}")
        window_dir = batch_dir / f"window-{window:02d}"
        window_dir.mkdir(mode=0o700)
        end = time.monotonic() + lease_s
        pending = [p for p in procedures if queues[p]]
        ran: list[tuple[str, Path, dict, dict, list]] = []
        executed_in_window = 0
        for position, procedure in enumerate(pending):
            now = time.monotonic()
            left = end - now
            if left < 1:
                continue  # the window is spent: this procedure starts in the next window
            deadline = now + left / (len(pending) - position)  # equal share of what remains of the window
            procedure_dir = window_dir / procedure
            procedure_dir.mkdir(mode=0o700)
            queue = queues[procedure]
            if queue == list(unit.sample_ids):
                roster = dict(unit.roster)
            else:
                roster = _continuation(
                    campaign, procedure_dir / "continuation-roster.json", unit, procedure, window, queue
                )
            run, score = build_commands(campaign, roster, procedure_dir, procedure, deadline_monotonic=deadline)
            bound = left
            kill_after = deliberate.get(procedure) if window == 1 else None
            if kill_after is not None:
                bound = min(left, kill_after)
            try:
                command(run, bound, procedure_dir / "run.log")
            except TimeoutError:
                classification = DELIBERATE if kill_after is not None and bound < left else KILLED
                raise _Ambiguous(classification, procedure, "the session was killed at its bound") from None
            except TransportLost as error:
                raise _Ambiguous(TRANSPORT, procedure, str(error)) from None
            receipt = _session_receipt(procedure_dir / "run", queue, procedure)
            if receipt["status"] == session_mod.INVALID:
                raise _Ambiguous(INVALID_SESSION, procedure, json.dumps(receipt.get("invalid"), sort_keys=True))
            executed = receipt["executed_sample_ids"]
            executed_in_window += len(executed)
            queues[procedure] = receipt["deferred_sample_ids"]
            ran.append((procedure, procedure_dir, roster, receipt, score))
        if not executed_in_window:
            raise CampaignStopped(
                f"unit {unit.unit_id}: window {window} measured no task; the window cannot hold a session"
            )
        # Scoring is host work outside the window: each session with measured tasks is scored and verified.
        window_rows = []
        for procedure, procedure_dir, roster, receipt, score in ran:
            executed = receipt["executed_sample_ids"]
            records = []
            if executed:
                command(score, lease_s, procedure_dir / "score.log")
                records = verify_scored(procedure_dir, _expected(campaign, roster, procedure, executed))
            row = {
                "window": window,
                "procedure": procedure,
                "roster": {
                    "path": str(Path(roster["path"]).relative_to(output))
                    if Path(roster["path"]).is_relative_to(output)
                    else roster["path"],
                    "file_sha256": roster["file_sha256"],
                    "roster_sha256": roster["roster_sha256"],
                },
                "status": receipt["status"],
                "executed_sample_ids": executed,
                "deferred_sample_ids": receipt["deferred_sample_ids"],
                "closure": receipt.get("closure"),
                "cold_start": receipt.get("cold_start"),
                "scored_receipts": records,
            }
            window_rows.append(row)
        windows.append({"window": window, "lease": lease, "sessions": window_rows})
    metadata = _batch_metadata(campaign, unit, _order(campaign, unit.batch_index))
    completion = {
        "schema": COMPLETION_SCHEMA,
        **metadata,
        "attempt": attempt,
        "effect_key": effect_key,
        "procedures": procedures,
        "imputed_failures": {p: imputed[p]["sha256"] for p in sorted(imputed)},
        "windows": windows,
    }
    pinned.write_json_exclusive(batch_dir / "completion.json", completion)
    return completion


def _verify_completed(campaign: CampaignPlan, output: Path, unit: Unit) -> dict[str, Any]:
    """Re-verify a completed unit: metadata, every session's score receipts and continuation rosters, imputations."""
    batch_dir = output / unit.directory
    saved = pinned.read_json(batch_dir / "completion.json")
    metadata = _batch_metadata(campaign, unit, _order(campaign, unit.batch_index))
    imputed = _imputed(output, unit)
    if (
        saved.get("schema") != COMPLETION_SCHEMA
        or {key: saved.get(key) for key in metadata} != metadata
        or saved.get("imputed_failures") != {p: imputed[p]["sha256"] for p in sorted(imputed)}
        or sorted([*saved.get("procedures", []), *imputed]) != sorted(campaign.procedures)
    ):
        raise ValueError(f"completed unit {unit.unit_id} changed or belongs to another plan")
    measured: dict[str, list[str]] = {p: [] for p in saved["procedures"]}
    for window in saved["windows"]:
        for row in window["sessions"]:
            procedure, roster = row["procedure"], dict(row["roster"])
            if not Path(roster["path"]).is_absolute():
                roster["path"] = str(output / roster["path"])
            pinned.read_pinned(Path(roster["path"]), roster["file_sha256"])
            procedure_dir = batch_dir / f"window-{window['window']:02d}" / procedure
            records = (
                verify_scored(procedure_dir, _expected(campaign, roster, procedure, row["executed_sample_ids"]))
                if row["executed_sample_ids"]
                else []
            )
            if records != row["scored_receipts"]:
                raise ValueError(f"completed unit {unit.unit_id}: a scored session changed")
            measured[procedure] += row["executed_sample_ids"]
    if any(ids != list(unit.sample_ids) for ids in measured.values()):
        raise ValueError(f"completed unit {unit.unit_id} does not measure every task once, in order")
    return saved


def _impute(output: Path, unit: Unit, procedure: str, campaign: CampaignPlan, history: list) -> dict[str, Any]:
    """R7: a single task that killed its batch twice with green infrastructure is a candidate failure."""
    plan = campaign.plan
    row = {
        "schema": IMPUTED_SCHEMA,
        "plan_sha256": campaign.sha256,
        "unit": unit.unit_id,
        "batch_index": unit.batch_index,
        "procedure": procedure,
        "sample_id": unit.sample_ids[0],
        "status": "runtime_error",
        "reason": "r7_single_task_killed_its_batch_twice_with_green_infrastructure",
        "penalty": "canonical: regret 1e6 - 1, no valid path; latency charged at the wall budget and censored",
        "charged_latency_ms": plan["budget"]["wall_time_ms"],
        "attempts": [
            {key: entry.get(key) for key in ("attempt", "classification", "archive_sha256")}
            for entry in history
            if entry.get("procedure") == procedure
        ],
        "verified": False,
    }
    row["sha256"] = pinned.canonical_sha256(row)
    directory = output / IMPUTED_DIR
    directory.mkdir(mode=0o700, exist_ok=True)
    pinned.write_json_exclusive(directory / f"{unit.directory}--{procedure}.json", row)
    return row


def _run_unit(
    campaign: CampaignPlan,
    output: Path,
    unit: Unit,
    status: dict[str, Any],
    *,
    command,
    infrastructure,
    deliberate: Mapping[tuple[int, str], float],
) -> list[Unit]:
    """Run (or resume) one unit to completion; return the halves when R7 splits it, else []."""
    batch_dir = output / unit.directory
    if (batch_dir / "completion.json").is_file():
        _verify_completed(campaign, output, unit)
        status["resumed_units"].append(unit.unit_id)
        return []
    order = _order(campaign, unit.batch_index)
    metadata = _batch_metadata(campaign, unit, order)
    if batch_dir.exists():
        failure_path = batch_dir / "failure.json"
        failure = pinned.read_json(failure_path) if failure_path.is_file() else {}
        if failure.get("classification") == "harness_error":
            raise IncompleteBatch(
                f"incomplete unit {unit.unit_id}: a harness error stopped it; inspect it and relaunch it with "
                "`relaunch-batch`"
            )
        # An attempt left by a stopped runner, or by a drain not yet proven: ambiguous, relaunched under R7, never read.
        classification = failure.get("classification") if failure.get("classification") in AMBIGUOUS else INTERRUPTED
        ambiguous = _Ambiguous(classification, failure.get("procedure"), "attempt found incomplete on resume")
        if (
            _decide(campaign, output, unit, _attempt(output, unit.unit_id), ambiguous, status, infrastructure)
            == "split"
        ):
            return list(_split(campaign, output, unit))
    else:
        _check_not_stopped(output, unit)
    while True:
        attempt = _attempt(output, unit.unit_id)
        imputed = _imputed(output, unit)
        procedures = [p for p in order if p not in imputed]
        effect_key = infrastructure.new_effect_key()
        partition_io.load_child(campaign.plan["partition"]["dir"], campaign.partition["batches"][unit.batch_index])
        batch_dir.mkdir(mode=0o700)
        kills = {
            procedure: seconds
            for (index, procedure), seconds in deliberate.items()
            if index == unit.batch_index and unit.unit_id == f"{index:06d}" and attempt == 1
        }
        pinned.write_json_exclusive(
            batch_dir / "intent.json",
            {
                "schema": INTENT_SCHEMA,
                **metadata,
                "attempt": attempt,
                "effect_key": effect_key,
                "procedures": procedures,
                "deliberate_kill": kills or None,
                "at": _now(),
            },
        )
        try:
            _execute_attempt(
                campaign,
                output,
                unit,
                attempt,
                effect_key,
                procedures,
                imputed,
                command=command,
                infrastructure=infrastructure,
                deliberate=kills,
            )
        except _Ambiguous as ambiguous:
            pinned.write_json_exclusive(
                batch_dir / "failure.json",
                {
                    "schema": FAILURE_SCHEMA,
                    **metadata,
                    "attempt": attempt,
                    "classification": ambiguous.classification,
                    "procedure": ambiguous.procedure,
                    "message": ambiguous.detail[:500],
                    "at": _now(),
                },
            )
            if _decide(campaign, output, unit, attempt, ambiguous, status, infrastructure) == "split":
                return list(_split(campaign, output, unit))
            continue
        except BaseException as error:
            interrupted = isinstance(error, (KeyboardInterrupt, SystemExit))
            pinned.write_json_exclusive(
                batch_dir / "failure.json",
                {
                    "schema": FAILURE_SCHEMA,
                    **metadata,
                    "attempt": attempt,
                    "classification": INTERRUPTED if interrupted else "harness_error",
                    "type": type(error).__name__,
                    "message": str(error)[:500],
                    "at": _now(),
                },
            )
            raise
        status["completed_units"].append(unit.unit_id)
        return []


def _decide(
    campaign: CampaignPlan,
    output: Path,
    unit: Unit,
    attempt: int,
    ambiguous: _Ambiguous,
    status: dict[str, Any],
    infrastructure: Any,
) -> str:
    """Prove the drain, archive the ambiguous attempt, verify the lease and apply R7: relaunch, impute, split or stop.

    A drain that is not proven stops the campaign before anything moves: the attempt stays in place, and the next
    ``run`` decides again once the drain is proven.
    """
    drain = infrastructure.drain_proof(output / unit.directory)
    if not drain.get("ok"):
        raise CampaignStopped(
            f"unit {unit.unit_id}: the drain of the ambiguous attempt is not proven ({drain}); nothing was archived"
        )
    archive = _archive(output, unit, attempt)
    lease = infrastructure.verify_lease()
    green = bool(drain.get("ok")) and bool(lease.get("ok")) and ambiguous.classification != TRANSPORT
    history = [row for row in _unit_log(output, unit.unit_id) if row.get("kind") == "r7"]
    policy = _r7_policy(campaign)
    deliberate = ambiguous.classification == DELIBERATE
    counted = [row for row in history if row.get("classification") != DELIBERATE]
    ambiguous_attempts = len(counted) + (0 if deliberate else 1)
    attributable = [
        row
        for row in counted
        if row.get("green")
        and row.get("classification") in policy["candidate_attributable"]
        and row.get("procedure") == ambiguous.procedure
    ]
    kills = len(attributable) + (
        1 if green and ambiguous.classification in policy["candidate_attributable"] and not deliberate else 0
    )
    if not lease.get("ok"):
        decision = "stop_lease_not_verified"
    elif (
        len(unit.sample_ids) == 1
        and kills >= policy["single_task_candidate_failures"]
        and ambiguous.procedure is not None
    ):
        decision = "impute_candidate_failure"
    elif ambiguous_attempts >= policy["max_ambiguous_attempts"]:
        decision = "split" if len(unit.sample_ids) > 1 else "stop_single_task_not_attributable"
    else:
        decision = "relaunch"
    n_batches = len(campaign.partition["batches"])
    if decision in ("relaunch", "split") and not deliberate:
        relaunched = _relaunched_batches(output) | {unit.batch_index}
        if len(relaunched) > policy["relaunch_stop_fraction"] * n_batches:
            decision = "stop_relaunch_fraction"
    entry = {
        "schema": RELAUNCH_SCHEMA,
        "kind": "r7",
        "plan_sha256": campaign.sha256,
        "batch_index": unit.batch_index,
        "unit": unit.unit_id,
        "attempt": attempt,
        "classification": ambiguous.classification,
        "procedure": ambiguous.procedure,
        "detail": ambiguous.detail[:500],
        "green": green,
        "drain_proof": drain,
        "lease": lease,
        "decision": decision,
        "policy": campaign.plan["infrastructure_relaunch_policy"],
        **archive,
        "at": _now(),
    }
    _append_jsonl(output / RELAUNCH_LOG, entry)
    if not deliberate:
        status["r7_relaunches"] += 1
    else:
        status["deliberate_kills"] += 1
    if decision.startswith("stop"):
        raise CampaignStopped(f"unit {unit.unit_id}: {decision} after {ambiguous.classification}")
    if decision == "impute_candidate_failure":
        _impute(output, unit, ambiguous.procedure, campaign, [*history, entry])
    if decision == "split":
        _append_jsonl(
            output / SPLIT_LOG,
            {
                "schema": SPLIT_SCHEMA,
                "plan_sha256": campaign.sha256,
                "unit": unit.unit_id,
                "children": [f"{unit.unit_id}.0", f"{unit.unit_id}.1"],
                "at": _now(),
            },
        )
    return decision


def _parse_kills(campaign: CampaignPlan, kills: Sequence[tuple[int, str, float]]) -> dict[tuple[int, str], float]:
    if kills and campaign.plan["split"] != "validation":
        raise ValueError("a deliberate kill exercises R7 in a validation rehearsal only")
    parsed = {}
    for index, procedure, seconds in kills:
        if procedure not in campaign.procedures or not 0 <= index < len(campaign.partition["batches"]):
            raise ValueError(f"deliberate kill {index}:{procedure} names no batch procedure of this plan")
        if not isinstance(seconds, (int, float)) or isinstance(seconds, bool) or not seconds > 0:
            raise ValueError("a deliberate kill needs a positive delay in seconds")
        parsed[(int(index), procedure)] = float(seconds)
    return parsed


def run_campaign(
    plan_path: str | Path,
    plan_sha256: str,
    *,
    allow_test: bool = False,
    command: Callable[[Sequence[str], float, Path], None] | None = None,
    infrastructure: Any = None,
    deliberate_kills: Sequence[tuple[int, str, float]] = (),
    stop_after_batches: int | None = None,
) -> dict[str, Any]:
    """Execute or resume every batch of a pinned plan with R7 relaunches; stop on a harness error or for diagnosis.

    ``command`` replaces ``run_command`` (tests); ``infrastructure`` replaces ``HostInfrastructure`` (drain proof,
    lease verification, effect keys). ``deliberate_kills`` lists ``(batch index, procedure, seconds)`` kills of the
    first attempt (validation only). ``stop_after_batches`` stops cleanly after that many batches completed in this
    invocation (a resume point for a rehearsal). Returns the campaign status.
    """
    command = command or run_command
    infrastructure = infrastructure or HostInfrastructure()
    campaign = read_plan(plan_path, plan_sha256, allow_test=allow_test)
    plan = campaign.plan
    if plan["target"] != "host":
        raise EdgeTransportNotQualified(
            f"edge transport not qualified: no remote transport to {plan['target']!r} is implemented in this harness"
        )
    kills = _parse_kills(campaign, deliberate_kills)
    if stop_after_batches is not None and (isinstance(stop_after_batches, bool) or stop_after_batches < 1):
        raise ValueError("stop_after_batches must be a positive integer")
    output = Path(plan["output_dir"])
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    with (output / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("campaign already running") from error
        identity_path = output / IDENTITY_FILE
        if identity_path.exists():
            if pinned.read_json(identity_path) != _identity(campaign):
                raise ValueError("the output directory belongs to another campaign")
        else:
            pinned.write_json_exclusive(identity_path, _identity(campaign))
        n_batches = len(campaign.partition["batches"])
        status = {
            "schema": STATUS_SCHEMA,
            "plan_sha256": plan_sha256,
            "campaign_id": plan["campaign_id"],
            "split": plan["split"],
            "target": plan["target"],
            "verified": False,
            "started_at": _now(),
            "n_batches": n_batches,
            "n_expected": campaign.partition["n_expected"],
            "resumed_batches": [],
            "completed_batches": [],
            "resumed_units": [],
            "completed_units": [],
            "r7_relaunches": 0,
            "deliberate_kills": 0,
        }
        try:
            _check_relaunch_fraction(output, n_batches, _r7_policy(campaign))
            done_here = 0
            for item in campaign.partition["batches"]:
                pending = [root_unit(campaign, item)]
                resumed = True
                while pending:
                    unit = pending.pop(0)
                    children = _children(output, unit.unit_id)
                    if children is not None:
                        pending = list(_split(campaign, output, unit)) + pending
                        continue
                    before = len(status["resumed_units"])
                    halves = _run_unit(
                        campaign, output, unit, status, command=command, infrastructure=infrastructure, deliberate=kills
                    )
                    resumed = resumed and len(status["resumed_units"]) > before
                    pending = list(halves) + pending
                (status["resumed_batches"] if resumed else status["completed_batches"]).append(item["index"])
                if not resumed:
                    done_here += 1
                    if stop_after_batches is not None and done_here >= stop_after_batches:
                        status["outcome"] = "paused"
                        _append_jsonl(output / INVOCATIONS_FILE, status)
                        return status
        except BaseException as error:
            _append_jsonl(output / INVOCATIONS_FILE, {**status, "outcome": "stopped", "error": type(error).__name__})
            raise
        status["outcome"] = "complete"
        _append_jsonl(output / INVOCATIONS_FILE, status)
        return status


def relaunch_batch(
    plan_path: str | Path,
    plan_sha256: str,
    batch_index: int,
    *,
    operator: str,
    reason: str,
    reason_class: str,
    allow_test: bool = False,
    unit: str | None = None,
) -> dict[str, Any]:
    """Archive one attempt stopped by a harness error so that the next ``run_campaign`` runs it again; trace it.

    ``unit`` names a split half (default: the batch itself). Refuses a completed unit, a missing unit, a reason class
    other than ``infrastructure`` and an empty operator or reason. Runs nothing itself.
    """
    campaign = read_plan(plan_path, plan_sha256, allow_test=allow_test)
    if reason_class != "infrastructure":
        raise ValueError("only a batch failed by infrastructure may be relaunched (reason class 'infrastructure')")
    if not isinstance(operator, str) or not operator.strip() or not isinstance(reason, str) or not reason.strip():
        raise ValueError("a relaunch needs a named operator and a stated reason")
    indices = {item["index"] for item in campaign.partition["batches"]}
    if isinstance(batch_index, bool) or batch_index not in indices:
        raise ValueError(f"batch {batch_index!r} is not a batch of this plan")
    unit_id = unit if unit is not None else f"{batch_index:06d}"
    if unit_id.split(".")[0] != f"{batch_index:06d}" or not all(part in "01" for part in unit_id.split(".")[1:]):
        raise ValueError(f"unit {unit_id!r} is not a unit of batch {batch_index}")
    output = Path(campaign.plan["output_dir"])
    with (output / ".lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("campaign already running") from error
        if pinned.read_json(output / IDENTITY_FILE) != _identity(campaign):
            raise ValueError("the output directory belongs to another campaign")
        batch_dir = output / f"batch-{unit_id}"
        log = _unit_log(output, unit_id)
        stopped = bool(log) and log[-1].get("decision") in OPERATOR_STOPS
        if not batch_dir.is_dir() and not stopped:
            raise ValueError(f"batch {unit_id} was never started")
        if (batch_dir / "completion.json").exists():
            raise ValueError(f"batch {unit_id} is complete and cannot be relaunched")
        attempt = _attempt(output, unit_id)

        def digest(name: str) -> str | None:
            path = batch_dir / name
            return pinned.sha256_bytes(pinned.read_bounded(path)) if path.is_file() else None

        intent, failure = digest("intent.json"), digest("failure.json")
        target = Unit(unit_id, batch_index, 0, 0, {}, ())
        archive = (
            _archive(output, target, attempt)
            if batch_dir.is_dir()
            else {"archived_to": None, "archive_sha256": None, "n_files": 0}
        )
        entry = {
            "schema": RELAUNCH_SCHEMA,
            "kind": "operator",
            "plan_sha256": plan_sha256,
            "batch_index": batch_index,
            "unit": unit_id,
            "attempt": attempt,
            "operator": operator.strip(),
            "reason": reason.strip(),
            "reason_class": reason_class,
            "policy": campaign.plan["infrastructure_relaunch_policy"],
            "intent_sha256": intent,
            "failure_sha256": failure,
            **archive,
            "at": _now(),
        }
        _append_jsonl(output / RELAUNCH_LOG, entry)
        return entry


def relaunch_summary(output: Path, n_batches: int, policy: Mapping[str, Any]) -> dict[str, Any]:
    """Counts of the relaunch log of one campaign (target), as published with its results."""
    rows = _read_jsonl(output / RELAUNCH_LOG)
    r7 = [row for row in rows if row.get("kind") == "r7"]
    planned = [row for row in r7 if row["classification"] != DELIBERATE]
    relaunched = sorted({row["batch_index"] for row in planned})
    return {
        "n_batches": n_batches,
        "n_r7_ambiguous_attempts": len(planned),
        "by_classification": dict(sorted(Counter(row["classification"] for row in r7).items())),
        "by_decision": dict(sorted(Counter(row["decision"] for row in r7).items())),
        "n_deliberate_kills": sum(row["classification"] == DELIBERATE for row in r7),
        "n_operator_relaunches": sum(row.get("kind") == "operator" for row in rows),
        "relaunched_batches": relaunched,
        "relaunched_fraction": len(relaunched) / n_batches if n_batches else math.nan,
        "stop_fraction": policy["relaunch_stop_fraction"],
        "archived_attempts": [
            {key: row.get(key) for key in ("unit", "attempt", "classification", "archive_sha256")} for row in rows
        ],
        "scored": "only the first complete attempt of every unit; archived attempts are never scored",
    }
