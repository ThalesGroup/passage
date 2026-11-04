"""Host-only validation and scoring of a completed direct-session journal."""

from __future__ import annotations
import json
import hashlib
from dataclasses import replace
from pathlib import Path
from passage_eval.execution import run_requests, score_run_artifact
from passage_eval.protocol.contracts import PlanRequest, PlanResult
from passage_eval.campaign.direct.direct_session import inspect_journal
from passage_eval.campaign.direct.raw_identity import raw_digest


class _SavedPaths:
    def __init__(self, records):
        self.records = iter(records)

    def plan(self, request):
        row = next(self.records)
        if row["sample_id"] != request.sample_id:
            raise ValueError("direct journal order differs from host scenes")
        work = row.get("work", {})
        expansions = work.get("expansions", work.get("descent_steps", 0) + work.get("fallback_expansions", 0))
        fallback = row.get("fallback") is True
        return PlanResult(
            row["path"],
            status=row["status"],
            stage="fallback" if fallback else "primary",
            work={"expansions": expansions},
            stage_work={"fallback": {"expansions": work.get("fallback_expansions", 0)}} if fallback else {},
        )


def check_journal_input_identities(rows, requests):
    if len(rows) != len(requests):
        raise ValueError("journal input identity count differs")
    for row, request in zip(rows, requests):
        if row["sample_id"] != request.sample_id or row.get("raw_input_sha256") != raw_digest(request):
            raise ValueError("journal input identity differs from host")


def host_score_direct(journal_path, scenes, budget, reference_store, *, reference_cohort_id, warmup_input_sha256):
    """Validate path legality and score on host; direct elapsed time is preserved without host recost."""
    audit = inspect_journal(journal_path)
    if not audit["complete"] or not audit["runtime_released"]:
        raise ValueError("incomplete, ambiguous, or unreleased direct session cannot be scored")
    from passage_eval.execution import timing

    if not hasattr(timing, "NATIVE_PERSISTENT"):
        raise ValueError("native persistent timing profile is not installed")
    if len(warmup_input_sha256) != 64 or set(warmup_input_sha256) - set("0123456789abcdef"):
        raise ValueError("warmup input identity differs")
    events = [json.loads(line) for line in Path(journal_path).read_text().splitlines()]
    rows = [event for event in events if event["kind"] == "request"]
    # Host validation is not bounded by or charged to the target wall deadline.
    validation_budget = replace(budget, wall_time_ms=None)
    requests = [PlanRequest.from_scene(scene, budget=validation_budget) for scene in scenes]
    check_journal_input_identities(rows, requests)
    artifact = run_requests(
        _SavedPaths(rows),
        requests,
        series="warm",
        manifest={"reference_policy": "independent_analytic_reference", "reference_cohort_id": reference_cohort_id},
    )
    artifact.manifest["schema"] = timing.NATIVE_SCHEMA
    artifact.manifest["timing_profile"] = timing.NATIVE_PERSISTENT
    artifact.manifest["timing_boundary"] = timing.NATIVE_BOUNDARY
    artifact.manifest["procedure"] = {"execution": {"timing_profile": timing.NATIVE_PERSISTENT}}
    artifact.manifest["native_execution"] = {
        "session_id": hashlib.sha256(Path(journal_path).read_bytes()).hexdigest(),
        "n_processes": 1,
        "n_restarts": 0,
        "warmup": {
            "k": 1,
            "repeats": 16,
            "performed": True,
            "input_roster_sha256": warmup_input_sha256,
            "disjoint_by_content": True,
        },
    }
    artifact.manifest["timing_note"] = "direct target time; host validation and recost excluded"
    artifact.manifest["reference_identity"] = hashlib.sha256(
        json.dumps(reference_store, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    artifact.manifest["qualification_status"] = "exploratory_unadmitted"
    for request in requests:
        artifact.manifest["request_specs"][request.sample_id]["budget"] = budget.to_dict()
    for record, row in zip(artifact.records, rows):
        elapsed = row["elapsed_ns"] / 1_000_000
        validation_ms = record["total_wall_ms"]
        deadline = budget.wall_time_ms
        record["deadline_ms"] = deadline
        late = deadline is not None and elapsed > deadline
        if late and row["status"] != "timeout":
            raise ValueError("journal deadline status differs")
        record["status"] = row["status"] if row["status"] != "ok" else record["status"]
        record["native_wall_ms"] = elapsed
        record["validation_recost_ms"] = validation_ms
        record["total_wall_ms"] = elapsed + validation_ms
        record["timing_boundary"] = timing.NATIVE_BOUNDARY
        record["candidate_wall_ms"] = elapsed
        record["late_completion"] = late
        record["failed_attempt_ms"] = min(elapsed, deadline) if late else elapsed if record["status"] != "ok" else 0.0
        record["drain_ms"] = max(0.0, elapsed - deadline) if late else 0.0
        record["diagnostics"]["direct_elapsed_ns"] = row["elapsed_ns"]
        record["diagnostics"]["direct_error"] = row.get("error")
        if "status_before_deadline" in row:
            record["diagnostics"]["status_before_deadline"] = row["status_before_deadline"]
    artifact.validate()
    metrics, arrays = score_run_artifact(artifact, scenes, reference_store=reference_store)
    return metrics, arrays, artifact


def verify_drain(receipt, *, target, procedure, job_id, scope):
    """Check session-specific process exit; do not claim driver-context introspection."""
    wanted = {
        "schema": "passage-direct-session-drain/v1",
        "target": target,
        "procedure": procedure,
        "job_id": job_id,
        "scope": scope,
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    if not isinstance(receipt, dict) or any(receipt.get(k) != v for k, v in wanted.items()):
        raise ValueError("session drain evidence differs")
    return True


def verify_target_receipt(root, *, config_sha256, index_sha256, native_pins):
    """Check receipt byte pin and journal binding before any host-side score."""
    from passage_eval.campaign.edge.target import _sha256_regular_file
    from passage_eval.io import pinned

    root = Path(root)
    files = [root / "target-receipt.json", root / "target-receipt.sha256", root / "journal.jsonl"]
    if root.is_symlink() or not root.is_dir() or any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("target receipt files are not regular")
    expected = (root / "target-receipt.sha256").read_text().strip()
    receipt = pinned.strict_json_loads(
        pinned.read_pinned(root / "target-receipt.json", expected), name="direct receipt"
    )
    if (
        receipt.get("schema") != "passage-direct-target-validation-result-prototype/v1"
        or receipt.get("configuration_sha256") != config_sha256
        or receipt.get("input_index_sha256") != index_sha256
        or receipt.get("native_artifact_hashes") != native_pins
        or receipt.get("qualification_status") != "exploratory_unadmitted"
        or receipt.get("scope_gpu_drain_verified") is not False
    ):
        raise ValueError("target receipt identity differs")
    if _sha256_regular_file(root / "journal.jsonl") != receipt.get("journal_sha256"):
        raise ValueError("target journal digest differs")
    return receipt


def _check_rule8_journal(journal, config, count, expected_sha):
    raw = Path(journal).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("Rule8 journal changed after verification")
    events = [json.loads(line) for line in raw.splitlines()]
    kinds = [event.get("kind") for event in events]
    if (
        not kinds
        or kinds[0] != "session_start"
        or kinds[-1] != "runtime_released"
        or any(
            kinds.count(kind) != 1
            for kind in ("session_start", "warmup_complete", "session_complete", "runtime_released")
        )
        or any(kind in kinds for kind in ("session_invalid", "session_incomplete", "runtime_release_failed"))
        or kinds.index("warmup_complete")
        >= next((i for i, k in enumerate(kinds) if k in ("reservation", "request_started", "request")), len(kinds))
        or kinds.index("session_complete") <= kinds.index("warmup_complete")
    ):
        raise ValueError("Rule8 journal session lifecycle differs")
    start = events[0]
    warm = events[kinds.index("warmup_complete")]
    if (
        start.get("procedure") != config["procedure"]
        or start.get("n_expected") != count
        or start.get("warmup_required") != 16
        or warm.get("count") != 16
        or warm.get("distinct_inputs") != 1
    ):
        raise ValueError("Rule8 warmup or procedure evidence differs")
    audit = inspect_journal(journal)
    if not audit["complete"] or not audit["runtime_released"]:
        raise ValueError("Rule8 journal incomplete or runtime unreleased")


def _check_scored_journal(artifact, journal):
    rows = [json.loads(line) for line in Path(journal).read_bytes().splitlines()]
    rows = [row for row in rows if row.get("kind") == "request"]
    if len(rows) != len(artifact.records) or len(rows) != len(artifact.paths):
        raise ValueError("scored records differ from the verified journal")
    for row, record, path in zip(rows, artifact.records, artifact.paths):
        diagnostics = record.get("diagnostics", {})
        if (
            row.get("sample_id") != record.get("sample_id")
            or path.tolist() != row.get("path")
            or diagnostics.get("direct_elapsed_ns") != row.get("elapsed_ns")
            or diagnostics.get("direct_error") != row.get("error")
            or diagnostics.get("status_before_deadline") != row.get("status_before_deadline")
            or record.get("fallback") != (row.get("fallback") is True)
            or record.get("native_wall_ms") != row.get("elapsed_ns") / 1_000_000
            or (row.get("status") != "ok" and record.get("status") != row.get("status"))
        ):
            raise ValueError("scored output, error or elapsed time differs from the verified journal")


def _roster_dataset_identity(roster):
    dataset = roster["dataset"]
    return (
        {**dataset, "revision": dataset["commit"], "task_variant": roster["cell"]["task_variant"]},
        {
            "roster_sha256": roster["roster_sha256"],
            "dataset_root_identity": roster["dataset_root_identity"],
            "dataset": dataset,
            "cell": roster["cell"],
        },
    )


def _check_admitted_references(grant, qualification, task_ids, cohort):
    """Bind all and only the scored references to the unchanged G4 inventory."""
    from collections.abc import Mapping
    from passage_eval.protocol.reference_admission import check_reference_ids

    grant.ensure_unchanged()
    policy = grant.reference_policy()
    if (
        policy.get("reference_policy") != "independent_analytic_reference"
        or not isinstance(cohort, str)
        or not cohort.strip()
        or cohort != policy.get("reference_cohort_id")
    ):
        raise ValueError("reference policy or cohort differs from admitted G4 policy")
    ids = grant.reference_ids()
    if (
        not isinstance(qualification, Mapping)
        or set(qualification) != set(task_ids)
        or len(task_ids) != len(set(task_ids))
        or set(ids) != set(task_ids)
    ):
        raise ValueError("reference store does not cover the admitted roster exactly")
    check_reference_ids(qualification, ids, context="Rule8 host scoring")
    grant.ensure_unchanged()


def _check_admitted_population(grant, record, roster, parent_order, parent_blocks, population):
    """Bind caller-supplied analysis population and blocks to the pinned G4 chain."""
    from passage_eval.io import pinned, roster as roster_io

    grant.ensure_unchanged()
    path, sha = pinned.file_ref(record["evidence"]["references"]["chain"])
    chain = pinned.read_pinned_json(pinned.resolve_inside(grant.evidence_dir, path), sha)
    blocks = roster_io.roster_block_table(roster)
    derivation = roster["selection"]["derivation"]
    expected_population = {key: derivation[key] for key in ("n_source", "n_excluded", "n_admissible")}
    if (
        chain.get("chain_sha256") != grant.report.admitted.get("chain_sha256")
        or chain.get("admissible_roster_sha256") != roster["roster_sha256"]
        or not blocks["complete"]
        or chain.get("admissible_block_table_sha256") != blocks["block_table_sha256"]
    ):
        raise ValueError("G4 admission chain or block table differs from the admitted roster")
    if list(parent_order) != [task["sample_id"] for task in roster["tasks"]]:
        raise ValueError("aggregation order differs from the admitted roster")
    if dict(parent_blocks) != blocks["blocks"]:
        raise ValueError("aggregation blocks differ from the admitted roster")
    counts = chain.get("counts", {})
    if (
        dict(population) != expected_population
        or any(counts.get(key) != value for key, value in expected_population.items())
        or expected_population["n_admissible"] != len(parent_order)
    ):
        raise ValueError("aggregation population differs from the admitted G4 chain")
    grant.ensure_unchanged()
    return chain["chain_sha256"]


def score_rule8_completed_session(
    session,
    target_root,
    image_start_path,
    image_start_sha256,
    scenes,
    reference_store,
    *,
    container_inspect,
    drain,
    job_id,
    reference_cohort_id,
):
    """Score a verified Rule8 session on the host, retaining its actual execution evidence.

    The caller supplies a host-observed OCI inspection and the host-held qualified
    references; neither is sent to the target. A verified receipt is necessary but
    does not establish publication, rebuildability, or campaign chronology.
    """
    from passage_eval.campaign.direct.admitted_stream import verify_rule8_completed_session
    from passage_eval.io import pinned, roster as roster_io
    from passage_eval.protocol.admission import native_input_projection

    observed = verify_rule8_completed_session(
        session,
        target_root,
        image_start_path,
        image_start_sha256,
        container_inspect=container_inspect,
        drain=drain,
        job_id=job_id,
    )
    if (
        type(observed.get("n_processes")) is not int
        or observed["n_processes"] != 1
        or type(observed.get("n_restarts")) is not int
        or observed["n_restarts"] != 0
    ):
        raise ValueError("Rule8 process count and restart evidence unavailable or invalid")
    config = pinned.strict_json_loads(session.config_bytes, name="Rule8 target config")
    record = session.rule8_scope.check_record(session.grant)
    measured = roster_io.load_roster(session.roster_path, expected_sha256=session.grant.admissible_roster_sha256)
    warm_path, warm_sha = pinned.file_ref(record["evidence"]["warmup"]["roster"])
    warm = pinned.read_pinned_json(pinned.resolve_inside(session.grant.evidence_dir, warm_path), warm_sha)
    measured_projection = native_input_projection(measured)
    warm_projection = native_input_projection(warm)
    measured_tasks, warm_tasks = measured_projection["tasks"], warm_projection["tasks"]
    if (
        len(warm_tasks) != 1
        or {t["sample_id"] for t in measured_tasks} & {t["sample_id"] for t in warm_tasks}
        or {t["content_identity_sha256"] for t in measured_tasks} & {t["content_identity_sha256"] for t in warm_tasks}
    ):
        raise ValueError("Rule8 warmup must be disjoint by ID and content")
    if warm["roster_sha256"] != session.warmup_roster_sha256:
        raise ValueError("Rule8 warmup roster differs from session")
    materialized = list(scenes)
    if len(materialized) != len(measured["tasks"]):
        raise ValueError("Rule8 scene count differs from admitted roster")
    for scene, task in zip(materialized, measured["tasks"]):
        req = PlanRequest.from_scene(scene, budget=session.budget)
        if scene.sample_id != task["sample_id"] or roster_io.request_input_identity(req) != task["input_identity"]:
            raise ValueError("Rule8 scoring scene differs from admitted roster")
    _check_admitted_references(
        session.grant,
        reference_store,
        [task["sample_id"] for task in measured["tasks"]],
        reference_cohort_id,
    )
    journal = Path(target_root) / "journal.jsonl"
    _check_rule8_journal(journal, config, len(measured_tasks), observed["journal_sha256"])
    warm_digest = pinned.canonical_sha256(warm_projection)
    metrics, arrays, artifact = host_score_direct(
        journal,
        materialized,
        session.budget,
        reference_store,
        reference_cohort_id=reference_cohort_id,
        warmup_input_sha256=warm_digest,
    )
    if hashlib.sha256(journal.read_bytes()).hexdigest() != observed["journal_sha256"]:
        raise ValueError("Rule8 journal changed during scoring")
    execution = {
        "session_id": observed["journal_sha256"],
        "n_processes": observed["n_processes"],
        "n_restarts": observed["n_restarts"],
        "input_roster_sha256": pinned.canonical_sha256(measured_projection),
        "artifacts": {
            "checkpoint_sha256": config["artifact_pins"]["checkpoint"],
            "engine_sha256": config["artifact_pins"]["engine"],
            "final_decoders_sha256": config["artifact_pins"]["decoder"],
        },
        "warmup": {
            "k": 1,
            "repeats": 16,
            "performed": True,
            "input_roster_sha256": warm_digest,
            "disjoint_by_content": True,
        },
        "container_digest": observed["container_digest"],
    }
    _, catalog_sha = pinned.file_ref(record["evidence"]["replica_catalog"])
    dataset, data_identity = _roster_dataset_identity(measured)
    artifact.manifest.update(
        dataset=dataset,
        data_identity=data_identity,
        seed=config["seed"],
        native_execution=execution,
        rule8_session_evidence={
            **observed,
            "image_start_sha256": image_start_sha256,
            "replica_catalog_sha256": catalog_sha,
            "global_scope_sha256": session.rule8_scope.sha256,
        },
        qualification_status="rule8_session_evidence_verified_host_scoring",
    )
    session.grant.ensure_unchanged()
    return metrics, arrays, artifact


def aggregate_rule8_scored_sessions(entries, *, parent_order, parent_blocks, population):
    """Combine three separately verified sessions, retaining each seed's evidence.

    Each entry maps a seed to the score directory and the arguments needed to
    independently reverify its completed session on the trusted host.
    """
    from passage_eval.campaign.aggregate import aggregate_scored_batches, combine_seeds
    from passage_eval.campaign.direct.admitted_stream import verify_rule8_completed_session
    from passage_eval.execution.artifacts import sha256_file
    from passage_eval.execution.timing import NATIVE_PERSISTENT
    from passage_eval.io import pinned, roster as roster_io
    from passage_eval.metrics.comparison import load_scored
    from passage_eval.protocol.admission import native_input_projection

    if set(entries) != {1234, 2026, 3407}:
        raise ValueError("Rule8 aggregate requires exactly the three declared seeds")
    per_seed, session_ids, catalog_hashes, references = {}, set(), set(), None
    identity = None
    for seed in (1234, 2026, 3407):
        entry = entries[seed]
        session = entry["session"]
        observed = verify_rule8_completed_session(
            session,
            entry["target_root"],
            entry["image_start_path"],
            entry["image_start_sha256"],
            container_inspect=entry["container_inspect"],
            drain=entry["drain"],
            job_id=entry["job_id"],
        )
        if (
            type(observed.get("n_processes")) is not int
            or observed["n_processes"] != 1
            or type(observed.get("n_restarts")) is not int
            or observed["n_restarts"] != 0
        ):
            raise ValueError("Rule8 process count and restart evidence unavailable or invalid")
        config = pinned.strict_json_loads(session.config_bytes, name="Rule8 target config")
        record = session.rule8_scope.check_record(session.grant)
        _, catalog = pinned.file_ref(record["evidence"]["replica_catalog"])
        roster = roster_io.load_roster(session.roster_path, expected_sha256=session.grant.admissible_roster_sha256)
        chain_sha = _check_admitted_population(
            session.grant,
            record,
            roster,
            parent_order,
            parent_blocks,
            population,
        )
        warm_path, warm_sha = pinned.file_ref(record["evidence"]["warmup"]["roster"])
        warm = pinned.read_pinned_json(pinned.resolve_inside(session.grant.evidence_dir, warm_path), warm_sha)
        warm_projection = native_input_projection(warm)
        measured_projection = native_input_projection(roster)
        warm_tasks, measured_tasks = warm_projection["tasks"], measured_projection["tasks"]
        if (
            len(warm_tasks) != 1
            or warm["roster_sha256"] != session.warmup_roster_sha256
            or {t["sample_id"] for t in warm_tasks} & {t["sample_id"] for t in measured_tasks}
            or {t["content_identity_sha256"] for t in warm_tasks}
            & {t["content_identity_sha256"] for t in measured_tasks}
        ):
            raise ValueError("Rule8 warmup must be disjoint by ID and content")
        expected_dataset, expected_data_identity = _roster_dataset_identity(roster)
        expected_execution = {
            "session_id": observed["journal_sha256"],
            "n_processes": observed["n_processes"],
            "n_restarts": observed["n_restarts"],
            "input_roster_sha256": pinned.canonical_sha256(native_input_projection(roster)),
            "artifacts": {
                "checkpoint_sha256": config["artifact_pins"]["checkpoint"],
                "engine_sha256": config["artifact_pins"]["engine"],
                "final_decoders_sha256": config["artifact_pins"]["decoder"],
            },
            "warmup": {
                "k": 1,
                "repeats": 16,
                "performed": True,
                "input_roster_sha256": pinned.canonical_sha256(warm_projection),
                "disjoint_by_content": True,
            },
            "container_digest": observed["container_digest"],
        }
        _check_rule8_journal(
            Path(entry["target_root"]) / "journal.jsonl", config, len(roster["tasks"]), observed["journal_sha256"]
        )
        run = load_scored(entry["scored_dir"])
        _check_scored_journal(run["artifact"], Path(entry["target_root"]) / "journal.jsonl")
        manifest = run["manifest"]
        execution = manifest.get("native_execution")
        evidence = manifest.get("rule8_session_evidence")
        if (
            config["seed"] != seed
            or manifest.get("seed") != seed
            or manifest.get("qualification_status") != "rule8_session_evidence_verified_host_scoring"
            or not isinstance(evidence, dict)
            or not isinstance(execution, dict)
            or any(evidence.get(k) != value for k, value in observed.items())
            or evidence.get("image_start_sha256") != entry["image_start_sha256"]
            or evidence.get("replica_catalog_sha256") != catalog
            or evidence.get("global_scope_sha256") != session.rule8_scope.sha256
            or execution != expected_execution
            or manifest.get("roster") != list(parent_order)
            or manifest.get("dataset") != expected_dataset
            or manifest.get("data_identity") != expected_data_identity
        ):
            raise ValueError(f"seed {seed}: scored artifact and verified session differ")
        candidate = (
            config["target"],
            config["procedure"],
            pinned.canonical_sha256(config["budget"]),
            session.rule8_scope.sha256,
            roster["roster_sha256"],
            chain_sha,
        )
        if identity is not None and identity != candidate:
            raise ValueError("Rule8 sessions do not share target, procedure, budget, scope and roster")
        identity = candidate
        if execution["session_id"] in session_ids:
            raise ValueError("Rule8 process session is reused")
        session_ids.add(execution["session_id"])
        catalog_hashes.add(catalog)
        arrays = run["arrays"]
        scored_references = {
            str(sample_id): {"reference_id": str(reference_id)}
            for sample_id, reference_id in zip(arrays["sample_id"], arrays["reference_id"])
        }
        _check_admitted_references(
            session.grant, scored_references, list(parent_order), manifest.get("reference_cohort_id")
        )
        if manifest.get("reference_policy") != session.grant.reference_policy().get("reference_policy"):
            raise ValueError("scored reference policy differs from admitted G4 policy")
        reference_rows = [
            (str(a), str(b), float(c))
            for a, b, c in zip(arrays["sample_id"], arrays["reference_id"], arrays["oracle_costs"])
        ]
        if references is not None and references != reference_rows:
            raise ValueError("Rule8 seeds use different reference identities or costs")
        references = reference_rows
        result = aggregate_scored_batches(
            [(run, parent_order)],
            parent_order=parent_order,
            parent_blocks=parent_blocks,
            population=population,
        )
        result["native_execution"] = execution
        result["rule8_session_evidence"] = evidence
        result["latency_sessions"][0]["native_execution"] = execution
        result["run_manifest_sha256"] = sha256_file(Path(entry["scored_dir"]) / "run_manifest.json")
        result["replica_catalog_sha256"] = catalog
        result["reference_admission_chain_sha256"] = chain_sha
        result["qualification_profile"] = record["protocol"]["profile"]
        result["harness"] = record["harness"]
        result["protocol_record_sha256"] = session.grant.record_sha256
        per_seed[str(seed)] = result
    if len(catalog_hashes) != 1:
        raise ValueError("Rule8 sessions use different replica catalogues")
    combined = combine_seeds(per_seed, [1234, 2026, 3407])
    statistics_inputs = rule8_statistics_inputs(
        per_seed,
        target=identity[0],
        procedure=identity[1],
        budget=config["budget"],
        global_scope_sha256=identity[3],
        admissible_roster_sha256=identity[4],
        split=roster["dataset"]["split"],
    )
    combined.update(
        statistics_inputs=statistics_inputs,
        replica_catalog_sha256=next(iter(catalog_hashes)),
        target=identity[0],
        procedure=identity[1],
        budget=config["budget"],
        global_scope_sha256=identity[3],
        admissible_roster_sha256=identity[4],
        split=roster["dataset"]["split"],
        timing_profile=NATIVE_PERSISTENT,
        reference_admission_chain_sha256=chain_sha,
    )
    return combined


def rule8_statistics_inputs(
    per_seed, *, target, procedure, budget, global_scope_sha256, admissible_roster_sha256, split
):
    """Project verified per-seed rows into the existing campaign-statistics interface."""
    from passage_eval.campaign.aggregate import combine_seeds
    from passage_eval.execution.timing import NATIVE_PERSISTENT

    if set(per_seed) != {"1234", "2026", "3407"}:
        raise ValueError("Rule8 statistics require exactly the three declared seeds")
    projected = []
    for seed in (1234, 2026, 3407):
        row = per_seed[str(seed)]
        single = combine_seeds({str(seed): row}, [seed])
        single["seed_interpretation"] = {"status": "per_seed_reporting_only"}
        single.update(
            qualification_profile=row["qualification_profile"],
            harness=row["harness"],
            protocol_record_sha256=row["protocol_record_sha256"],
            reference_admission_chain_sha256=row["reference_admission_chain_sha256"],
            replica_catalog_sha256=row["replica_catalog_sha256"],
            target=target,
            procedure=procedure,
            budget=budget,
            plan_sha256=global_scope_sha256,
            admissible_roster_sha256=admissible_roster_sha256,
            split=split,
            timing_profile=NATIVE_PERSISTENT,
        )
        projected.append(single)
    return projected
