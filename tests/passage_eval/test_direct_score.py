from dataclasses import replace
import importlib.util
from pathlib import Path
import numpy as np
from passage_eval import cli
from passage_eval.cli import BenchSpec
from passage_eval.io import roster as rio
from passage_eval.protocol.contracts import BudgetSpec
from passage_eval.campaign.direct.direct_session import execute_direct_candidate
from passage_eval.campaign.direct.direct_host_scoring import host_score_direct


class Runtime:
    def predict(self, tensor):
        return np.zeros((1, 1, 3, 3), np.float32)

    def synchronize(self):
        pass

    def close(self):
        pass


class Decoders:
    def __init__(self):
        self.count = 0

    def build_input(self, *args):
        return np.zeros((1, 3, 3, 3), np.float32)

    def reg_bellman_descent(self, field, obstacles, start, goal):
        self.count += 1
        return {
            "status": "no_path" if self.count == 19 else "ok",
            "path": [] if self.count == 19 else [start, (1, 1), goal],
            "fallback": False,
            "descent_steps": 2,
            "fallback_expansions": 0,
        }


def test_direct_journal_scores_failures_with_canonical_scorer(tmp_path, monkeypatch):
    p = Path(__file__).with_name("test_roster.py")
    spec = importlib.util.spec_from_file_location("direct_fixture_roster", p)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    root = fixture.write_dataset(tmp_path / "dataset", split="validation")
    roster, _ = fixture.build(root, split="validation")
    full, _ = fixture.build(root, split="validation", n_expected=12)
    # Fixture has only 12 distinct scenes; synthesize 16 disjoint ordinary warmups separately.
    budget = BudgetSpec(wall_time_ms=5)
    scenes = rio.load_roster_scenes(roster, root)
    from passage_eval.protocol.contracts import PlanRequest

    requests = [
        replace(PlanRequest.from_scene(scene, budget=budget), metadata={}, input_tensor=None) for scene in scenes
    ]
    warm = [
        PlanRequest(
            "warm", np.full((3, 3), 1000.0), np.zeros((3, 3), bool), (0, 0), (2, 2), "elevation", 62.4, budget=budget
        )
    ]
    from passage_eval.campaign.direct.raw_identity import raw_digest

    index = [{"sample_id": r.sample_id, "raw_input_sha256": raw_digest(r)} for r in requests]
    ticks = iter(
        [x for _ in range(16) for x in (0, 1_000_000)]
        + [0, 2_000_000, 0, 8_000_000]
        + [x for _ in range(4) for x in (0, 2_000_000)]
    )
    journal = tmp_path / "journal.jsonl"
    execute_direct_candidate(
        Runtime, Decoders(), "reg", warm, [requests[:2], requests[2:]], index, journal, clock_ns=lambda: next(ticks)
    )
    scenes = rio.load_roster_scenes(roster, root)
    bench = BenchSpec(
        "fixture",
        "classical",
        "fixture",
        split="validation",
        resolution=3,
        reference_policy="independent_analytic_reference",
        reference_cohort_id="fixture",
    )
    refs = cli._qualification(scenes, bench)
    metrics, arrays, artifact = host_score_direct(
        journal, scenes, budget, refs, reference_cohort_id="fixture", warmup_input_sha256="a" * 64
    )
    assert metrics["success_rate"] == 4 / 6
    assert sorted(r["status"] for r in artifact.records) == ["no_path", "ok", "ok", "ok", "ok", "timeout"]
    assert sum(np.isclose(arrays["regret"], 999999.0)) == 2
    assert artifact.manifest["timing_boundary"] == "passage-timing-boundary/v6-native-raw-arrays-to-path"
    from passage_eval.campaign.direct.direct_host_scoring import _check_scored_journal
    import pytest

    _check_scored_journal(artifact, journal)
    # Persist and reload a Rule8 score using actual synthetic scenes and the real scorer.
    import hashlib
    import json
    from types import SimpleNamespace
    from passage_eval.campaign.direct import admitted_stream, direct_host_scoring as scoring
    from passage_eval.execution import write_run_artifact, write_scored_artifact
    from passage_eval.io import pinned, roster as roster_io
    from passage_eval.metrics.comparison import load_scored
    from passage_eval.protocol import admission

    warm_roster = {"roster_sha256": "warm", "tasks": [{"sample_id": "warm", "content_identity_sha256": "warm"}]}
    config = {"seed": 2026, "procedure": "reg", "artifact_pins": {"checkpoint": "c", "engine": "e", "decoder": "d"}}
    record = {"evidence": {"warmup": {"roster": {"path": "w", "sha256": "a" * 64}},
                           "replica_catalog": {"path": "c", "sha256": "b" * 64}}}
    grant = SimpleNamespace(admissible_roster_sha256=roster["roster_sha256"], evidence_dir=tmp_path,
                            ensure_unchanged=lambda: None, reference_ids=lambda: {k: v["reference_id"] for k, v in refs.items()},
                            reference_policy=lambda: {"reference_policy": "independent_analytic_reference",
                                                      "reference_cohort_id": "fixture", "reference_max_expansions": 9})
    scope = SimpleNamespace(sha256="f" * 64, check_record=lambda grant: record)
    session = SimpleNamespace(config_bytes=json.dumps(config).encode(), rule8_scope=scope, grant=grant,
                              roster_path=tmp_path / "r", warmup_roster_sha256="warm", budget=budget)
    observed = {"container_digest": "sha256:" + "1" * 64, "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
                "n_processes": 1, "n_restarts": 0}
    monkeypatch.setattr(admitted_stream, "verify_rule8_completed_session", lambda *a, **k: observed)
    monkeypatch.setattr(roster_io, "load_roster", lambda *a, **k: roster)
    monkeypatch.setattr(pinned, "read_pinned_json", lambda *a, **k: warm_roster)
    monkeypatch.setattr(pinned, "resolve_inside", lambda root, path: root / path)
    real_projection = admission.native_input_projection
    monkeypatch.setattr(admission, "native_input_projection", lambda r: warm_roster if r is warm_roster else real_projection(r))
    metrics2, arrays2, scored = scoring.score_rule8_completed_session(
        session, tmp_path, tmp_path / "image", "4" * 64, scenes, refs,
        container_inspect={}, drain={}, job_id="job", reference_cohort_id="fixture",
    )
    write_run_artifact(scored, tmp_path / "score")
    write_scored_artifact(scored, metrics2, arrays2)
    reloaded = load_scored(tmp_path / "score")
    assert reloaded["manifest"]["data_identity"]["roster_sha256"] == roster["roster_sha256"]
    assert reloaded["manifest"]["dataset"]["split"] == roster["dataset"]["split"]
    assert len(reloaded["artifact"].records) == len(scenes)
    from passage_eval.campaign.aggregate import aggregate_scored_batches
    ids = [scene.sample_id for scene in scenes]
    result = aggregate_scored_batches([(reloaded, ids)], parent_order=ids,
        parent_blocks={sid: "synthetic-block" for sid in ids},
        population={"n_source": len(ids), "n_excluded": 0, "n_admissible": len(ids)})
    assert result["population"]["n_scored"] == len(ids)
    artifact.records[0]["diagnostics"]["direct_elapsed_ns"] += 1
    with pytest.raises(ValueError, match="elapsed time"):
        _check_scored_journal(artifact, journal)


def test_host_rejects_journal_with_forged_input_identity(tmp_path):
    from passage_eval.campaign.direct.direct_host_scoring import check_journal_input_identities
    from passage_eval.protocol.contracts import PlanRequest

    req = PlanRequest("a", np.zeros((3, 3)), np.zeros((3, 3), bool), (0, 0), (2, 2), "elevation", 62.4)
    import pytest

    with pytest.raises(ValueError, match="input identity"):
        check_journal_input_identities([{"sample_id": "a", "raw_input_sha256": "0" * 64}], [req])


def test_rule8_score_uses_verified_receipt_and_rosters(monkeypatch, tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    import pytest
    from passage_eval.campaign.direct import admitted_stream, direct_host_scoring as scoring
    from passage_eval.io import pinned, roster as roster_io
    from passage_eval.protocol import admission

    journal = tmp_path / "journal.jsonl"
    events = [
        {"kind": "session_start", "procedure": "reg", "n_expected": 1, "warmup_required": 16},
        {"kind": "warmup_complete", "count": 16, "distinct_inputs": 1},
        {"kind": "reservation", "start": 0, "end": 1},
        {"kind": "request", "sequence": 0},
        {"kind": "checkpoint", "n_completed": 1},
        {"kind": "session_complete", "n_completed": 1},
        {"kind": "runtime_released"},
    ]
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    journal_sha = hashlib.sha256(journal.read_bytes()).hexdigest()
    measured = {"tasks": [{"sample_id": "s", "input_identity": {"input": "s"}}], "roster_sha256": "m",
                "dataset": {"commit": "revision"}, "dataset_root_identity": {"sha256": "root"},
                "cell": {"task_variant": "fixture"}}
    warm = {"tasks": [{"sample_id": "w", "input_identity": {"input": "w"}}], "roster_sha256": "w"}
    projections = {
        "m": {"tasks": [{"sample_id": "s", "content_identity_sha256": "ms"}]},
        "w": {"tasks": [{"sample_id": "w", "content_identity_sha256": "ws"}]},
    }
    config = {"seed": 1234, "procedure": "reg", "artifact_pins": {"checkpoint": "c", "engine": "e", "decoder": "d"}}
    record = {"evidence": {"warmup": {"roster": {"path": "w", "sha256": "a" * 64}},
                           "replica_catalog": {"path": "c", "sha256": "b" * 64}}}
    grant = SimpleNamespace(admissible_roster_sha256="m", evidence_dir=tmp_path, ensure_unchanged=lambda: None,
                            reference_ids=lambda: {"s": "ref-sealed"},
                            reference_policy=lambda: {"reference_policy": "independent_analytic_reference",
                                                      "reference_cohort_id": "fixture", "reference_max_expansions": 9})
    scope = SimpleNamespace(sha256="f" * 64, check_record=lambda grant: record)
    session = SimpleNamespace(config_bytes=json.dumps(config).encode(), rule8_scope=scope, grant=grant,
                              roster_path=tmp_path / "roster", warmup_roster_sha256="w", budget=object())
    observed = {"container_digest": "sha256:" + "1" * 64, "journal_sha256": journal_sha,
                "configuration_sha256": "2" * 64, "input_index_sha256": "3" * 64,
                "scope": "scope", "job_id": "job", "n_processes": 1, "n_restarts": 0}
    monkeypatch.setattr(admitted_stream, "verify_rule8_completed_session", lambda *a, **k: observed, raising=False)
    monkeypatch.setattr(roster_io, "load_roster", lambda *a, **k: measured)
    monkeypatch.setattr(pinned, "read_pinned_json", lambda *a, **k: warm)
    monkeypatch.setattr(pinned, "resolve_inside", lambda root, path: root / path)
    monkeypatch.setattr(admission, "native_input_projection", lambda roster: projections[roster["roster_sha256"]])
    monkeypatch.setattr(roster_io, "request_input_identity", lambda req: {"input": "s"})
    monkeypatch.setattr(scoring.PlanRequest, "from_scene", lambda scene, budget: object())
    artifact = SimpleNamespace(manifest={})
    monkeypatch.setattr(scoring, "host_score_direct", lambda *a, **k: ({}, {}, artifact))
    scene = SimpleNamespace(sample_id="s")
    scoring.score_rule8_completed_session(session, tmp_path, tmp_path / "image", "4" * 64, [scene], {"s": {"reference_id": "ref-sealed"}},
        container_inspect={}, drain={}, job_id="job", reference_cohort_id="fixture")
    assert artifact.manifest["seed"] == 1234
    assert artifact.manifest["native_execution"]["container_digest"] == observed["container_digest"]
    assert artifact.manifest["native_execution"]["artifacts"] == {
        "checkpoint_sha256": "c", "engine_sha256": "e", "final_decoders_sha256": "d"}
    assert artifact.manifest["rule8_session_evidence"]["replica_catalog_sha256"] == "b" * 64
    with pytest.raises(ValueError, match="reference_id"):
        scoring.score_rule8_completed_session(session, tmp_path, tmp_path / "image", "4" * 64, [scene],
            {"s": {"reference_id": "substituted"}}, container_inspect={}, drain={}, job_id="job",
            reference_cohort_id="fixture")
    with pytest.raises(ValueError, match="cohort"):
        scoring.score_rule8_completed_session(session, tmp_path, tmp_path / "image", "4" * 64, [scene],
            {"s": {"reference_id": "ref-sealed"}}, container_inspect={}, drain={}, job_id="job",
            reference_cohort_id="other")
    del observed["n_processes"]
    with pytest.raises(ValueError, match="process count"):
        scoring.score_rule8_completed_session(session, tmp_path, tmp_path / "image", "4" * 64, [scene], {"s": {"reference_id": "ref-sealed"}},
            container_inspect={}, drain={}, job_id="job", reference_cohort_id="fixture")
    observed["n_processes"] = 1
    events[1]["count"] = 15
    journal.write_text("".join(json.dumps(e) + "\n" for e in events))
    observed["journal_sha256"] = hashlib.sha256(journal.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="warmup"):
        scoring.score_rule8_completed_session(session, tmp_path, tmp_path / "image", "4" * 64, [scene], {"s": {"reference_id": "ref-sealed"}},
            container_inspect={}, drain={}, job_id="job", reference_cohort_id="fixture")


def test_rule8_aggregate_requires_each_seed_and_revalidates_sessions(monkeypatch, tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    import pytest
    from passage_eval.campaign import aggregate
    from passage_eval.campaign.direct import admitted_stream, direct_host_scoring as scoring
    from passage_eval.execution import artifacts
    from passage_eval.io import pinned, roster as roster_io
    from passage_eval.metrics import comparison
    from passage_eval.protocol import admission

    events = [
        {"kind": "session_start", "procedure": "reg", "n_expected": 1, "warmup_required": 16},
        {"kind": "warmup_complete", "count": 16, "distinct_inputs": 1},
        {"kind": "reservation", "start": 0, "end": 1},
        {"kind": "request", "sequence": 0},
        {"kind": "checkpoint", "n_completed": 1},
        {"kind": "session_complete", "n_completed": 1}, {"kind": "runtime_released"},
    ]
    projection = {"tasks": [{"sample_id": "s", "content_identity_sha256": "i"}]}
    warm_projection = {"tasks": [{"sample_id": "w", "content_identity_sha256": "w"}]}
    roster = {"roster_sha256": "r", "tasks": [{"sample_id": "s"}], "dataset": {"split": "test", "commit": "revision"}, "dataset_root_identity": {"sha256": "root"},
              "cell": {"task_variant": "fixture"},
              "selection": {"derivation": {"n_source": 2, "n_excluded": 1, "n_admissible": 1}}}
    warm = {"roster_sha256": "w"}
    record = {"evidence": {"warmup": {"roster": {"path": "w", "sha256": "a" * 64}},
                           "replica_catalog": {"path": "c", "sha256": "b" * 64},
                           "references": {"chain": {"path": "chain", "sha256": "c" * 64}}},
              "protocol": {"profile": "rule8_native_persistent"}, "harness": {"source_sha256": "h"}}
    grant = SimpleNamespace(record_sha256="f" * 64, admissible_roster_sha256="r", evidence_dir=tmp_path, ensure_unchanged=lambda: None,
                            report=SimpleNamespace(admitted={"chain_sha256": "d" * 64}),
                            reference_ids=lambda: {"s": "ref"},
                            reference_policy=lambda: {"reference_policy": "independent_analytic_reference",
                                                      "reference_cohort_id": "fixture", "reference_max_expansions": 9})
    scope = SimpleNamespace(sha256="f" * 64, check_record=lambda g: record)
    entries, runs, observed = {}, {}, {}
    for seed in (1234, 2026, 3407):
        root = tmp_path / str(seed)
        root.mkdir()
        ev = [dict(e) for e in events]
        ev[0]["seed_fixture"] = seed
        (root / "journal.jsonl").write_text("".join(json.dumps(e) + "\n" for e in ev))
        digest = hashlib.sha256((root / "journal.jsonl").read_bytes()).hexdigest()
        config = {"seed": seed, "target": "orin", "procedure": "reg", "budget": {},
                  "artifact_pins": {"checkpoint": str(seed), "engine": "e", "decoder": "d"}}
        seed_grant = SimpleNamespace(**{**vars(grant), "record_sha256": f"{seed:064x}"})
        session = SimpleNamespace(config_bytes=json.dumps(config).encode(), rule8_scope=scope, grant=seed_grant, roster_path="r", warmup_roster_sha256="w")
        obs = {"container_digest": "sha256:" + "1" * 64, "journal_sha256": digest, "scope": "scope",
               "job_id": str(seed), "n_processes": 1, "n_restarts": 0}
        observed[seed] = obs
        execution = {"session_id": digest, "n_processes": 1, "n_restarts": 0,
                     "input_roster_sha256": pinned.canonical_sha256(projection),
                     "artifacts": {"checkpoint_sha256": str(seed), "engine_sha256": "e", "final_decoders_sha256": "d"},
                     "warmup": {"k": 1, "repeats": 16, "performed": True,
                                "input_roster_sha256": pinned.canonical_sha256(warm_projection), "disjoint_by_content": True},
                     "container_digest": obs["container_digest"]}
        evidence = {**obs, "image_start_sha256": "4" * 64, "replica_catalog_sha256": "b" * 64, "global_scope_sha256": "f" * 64}
        manifest = {"seed": seed, "native_execution": execution, "rule8_session_evidence": evidence,
                    "qualification_status": "rule8_session_evidence_verified_host_scoring", "roster": ["s"],
                    "reference_policy": "independent_analytic_reference", "reference_cohort_id": "fixture",
                    "dataset": {**roster["dataset"], "revision": "revision", "task_variant": "fixture"},
                    "data_identity": {"roster_sha256": "r", "dataset_root_identity": roster["dataset_root_identity"],
                                      "dataset": roster["dataset"], "cell": roster["cell"]}}
        runs[seed] = {"artifact": object(), "manifest": manifest, "arrays": {"sample_id": ["s"], "reference_id": ["ref"], "oracle_costs": [1.0]}}
        entries[seed] = {"session": session, "target_root": root, "image_start_path": root / "start",
                         "image_start_sha256": "4" * 64, "container_inspect": {}, "drain": {},
                         "job_id": str(seed), "scored_dir": root}
    monkeypatch.setattr(admitted_stream, "verify_rule8_completed_session",
                        lambda session, *a, **k: observed[json.loads(session.config_bytes)["seed"]], raising=False)
    monkeypatch.setattr(roster_io, "load_roster", lambda *a, **k: roster)
    monkeypatch.setattr(pinned, "resolve_inside", lambda root, path: root / path)
    chain = {"chain_sha256": "d" * 64, "admissible_roster_sha256": "r",
             "admissible_block_table_sha256": "e" * 64,
             "counts": {"n_source": 2, "n_excluded": 1, "n_admissible": 1}}
    monkeypatch.setattr(pinned, "read_pinned_json", lambda path, *a, **k: chain if path.name == "chain" else warm)
    monkeypatch.setattr(roster_io, "roster_block_table", lambda r: {"blocks": {"s": "b"}, "complete": True,
                                                                   "block_table_sha256": "e" * 64})
    monkeypatch.setattr(admission, "native_input_projection", lambda r: warm_projection if r["roster_sha256"] == "w" else projection)
    monkeypatch.setattr(comparison, "load_scored", lambda p: runs[int(p.name)])
    monkeypatch.setattr(scoring, "_check_scored_journal", lambda *a: None)
    monkeypatch.setattr(artifacts, "sha256_file", lambda p: "9" * 64)
    base = {"schema": aggregate.AGGREGATE_SCHEMA, "series": "warm", "population": {"n_source": 2, "n_excluded": 1, "n_admissible": 1, "n_scored": 1},
            "geographic_blocks": {}, "quantile_rule": "q", "latency_sessions": [{}],
            "canonical_equivalence": {"equal": True, "max_abs_difference": 0.0},
            **{field: {} for field in aggregate.SEED_RESULT_FIELDS}}
    import copy
    monkeypatch.setattr(aggregate, "aggregate_scored_batches", lambda *a, **k: copy.deepcopy(base))
    result = scoring.aggregate_rule8_scored_sessions(entries, parent_order=["s"], parent_blocks={"s": "b"}, population={"n_source": 2, "n_excluded": 1, "n_admissible": 1})
    assert result["seeds"] == [1234, 2026, 3407]
    assert [a["seeds"] for a in result["statistics_inputs"]] == [[1234], [2026], [3407]]
    assert [a["protocol_record_sha256"] for a in result["statistics_inputs"]] == [f"{seed:064x}" for seed in (1234, 2026, 3407)]
    from passage_eval.campaign.statistics import _inputs
    reg = result["statistics_inputs"]
    seg = [{**a, "procedure": "seg"} for a in reg]
    _, seeds, _ = _inputs([*seg, *reg], None)
    assert seeds == [1234, 2026, 3407]
    assert result["per_seed"]["3407"]["latency_sessions"][0]["native_execution"]["session_id"] == observed[3407]["journal_sha256"]
    with pytest.raises(ValueError, match="block"):
        scoring.aggregate_rule8_scored_sessions(entries, parent_order=["s"], parent_blocks={"s": "other"},
            population={"n_source": 2, "n_excluded": 1, "n_admissible": 1})
    with pytest.raises(ValueError, match="population"):
        scoring.aggregate_rule8_scored_sessions(entries, parent_order=["s"], parent_blocks={"s": "b"},
            population={"n_source": 1, "n_excluded": 0, "n_admissible": 1})
    for run in runs.values():
        run["arrays"]["reference_id"] = ["substituted"]
    with pytest.raises(ValueError, match="reference_id"):
        scoring.aggregate_rule8_scored_sessions(entries, parent_order=["s"], parent_blocks={"s": "b"}, population={"n_source": 2, "n_excluded": 1, "n_admissible": 1})
    for run in runs.values():
        run["arrays"]["reference_id"] = ["ref"]
    runs[2026]["manifest"]["native_execution"]["container_digest"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="verified session"):
        scoring.aggregate_rule8_scored_sessions(entries, parent_order=["s"], parent_blocks={"s": "b"}, population={"n_source": 2, "n_excluded": 1, "n_admissible": 1})
    with pytest.raises(ValueError, match="three declared seeds"):
        scoring.aggregate_rule8_scored_sessions({1234: entries[1234]}, parent_order=["s"], parent_blocks={"s": "b"}, population={"n_source": 2, "n_excluded": 1, "n_admissible": 1})


def test_scored_journal_rejects_changed_output_error_or_time(tmp_path):
    import json
    from types import SimpleNamespace
    import pytest
    from passage_eval.campaign.direct.direct_host_scoring import _check_scored_journal

    row = {"kind": "request", "sample_id": "one", "path": [], "status": "runtime_error",
           "elapsed_ns": 7_000_000, "error": "ValueError: failed"}
    journal = tmp_path / "journal.jsonl"
    journal.write_text(json.dumps(row) + "\n")
    record = {"sample_id": "one", "status": "runtime_error", "native_wall_ms": 7.0, "fallback": False,
              "diagnostics": {"direct_elapsed_ns": 7_000_000, "direct_error": row["error"]}}
    artifact = SimpleNamespace(records=[record], paths=[np.array([])])
    _check_scored_journal(artifact, journal)
    record["diagnostics"]["direct_error"] = "other"
    with pytest.raises(ValueError, match="output, error or elapsed"):
        _check_scored_journal(artifact, journal)


def test_emitted_statistics_inputs_run_complete_three_seed_analysis(tmp_path):
    """Only synthetic rows and synthetic sealed records; no target or real test data."""
    import copy
    from passage_eval.campaign.direct.direct_host_scoring import rule8_statistics_inputs
    from passage_eval.campaign.statistics import campaign_statistics
    from passage_eval.io import pinned

    p = Path(__file__).with_name("test_campaign_statistics.py")
    spec = importlib.util.spec_from_file_location("direct_stats_fixture", p)
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    world = fixture.world(n_blocks=32, per_block=2, seeds=(1234, 2026, 3407))
    for aggregate in world:
        aggregate.update(target="orin", timing_profile="native_raw_path_persistent_v1",
                         qualification_profile="rule8_native_persistent")
        for row in aggregate["per_seed"].values():
            sessions = row["latency_sessions"]
            row["latency_sessions"] = [{"session": 0, "warmup_complete": None, **{
                key: [value for session in sessions for value in session[key]]
                for key in ("sample_ids", "latency_ms", "censored", "status")}}]
            row.update(schema=aggregate["schema"], series="warm", population=aggregate["population"],
                       quantile_rule="synthetic fixture", canonical_equivalence={"equal": True, "max_abs_difference": 0.0},
                       metrics_extra={}, failures={}, imputed_failures={})
    family, records = fixture.sealed_family(world, mutate=fixture._rule8_native_record, analysis={"target": "orin"})
    record_paths, digests = {}, {}
    for old, record in records.items():
        path = tmp_path / f"record-{old}.json"
        digest = pinned.write_json_exclusive(path, record)
        digests[old], record_paths[digest] = digest, path
    inputs = []
    for source in world:
        procedure, per_seed = source["procedure"], {}
        for seed in (1234, 2026, 3407):
            row = copy.deepcopy(source["per_seed"][str(seed)])
            member = next(a for a in family if a["procedure"] == procedure and a["seeds"] == [seed])
            row.update(qualification_profile="rule8_native_persistent", harness=member["harness"],
                       protocol_record_sha256=digests[member["protocol_record_sha256"]],
                       reference_admission_chain_sha256=source["reference_admission_chain_sha256"],
                       replica_catalog_sha256="a" * 64)
            per_seed[str(seed)] = row
        inputs.extend(rule8_statistics_inputs(per_seed, target="orin", procedure=procedure,
            budget=source["budget"], global_scope_sha256="a" * 64,
            admissible_roster_sha256=source["admissible_roster_sha256"], split="test"))
    paths = []
    for aggregate in inputs:
        path = tmp_path / f"{aggregate['procedure']}-{aggregate['seeds'][0]}.json"
        pinned.write_json_exclusive(path, aggregate)
        paths.append(path)
    loaded = [pinned.read_json(path) for path in paths]
    loaded_records = {digest: pinned.read_json(path) for digest, path in record_paths.items()}
    result = campaign_statistics(loaded, bootstrap_seed=11, permutation_seed=12, latency_bootstrap_seed=13,
                                 draws=10000, permutations=200, protocol_records=loaded_records)
    assert result["completeness"]["seeds"] == [1234, 2026, 3407]
    assert result["sealing"]["protocol_records"] == {
        str(a["seeds"][0]): a["protocol_record_sha256"] for a in loaded if a["procedure"] == "seg"}
    assert result["quality"]["mode"] == "confirmatory"
    pinned.write_json_exclusive(tmp_path / "statistics.json", result)
