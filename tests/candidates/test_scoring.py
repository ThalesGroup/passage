import json
import numpy as np
import pytest


def score_fixture(tmp_path, status="ok"):
    from passage_eval.candidates.terrain import reverse_targets
    from passage_eval.protocol.candidate_package import file_sha256

    prepared = tmp_path / "prepared"
    (prepared / "scenes").mkdir(parents=True)
    z = np.zeros((256, 256))
    o = np.zeros_like(z, bool)
    values, path = reverse_targets(z, o, (0, 0), (1, 1))
    np.savez(
        prepared / "scenes/a.npz",
        terrain=z,
        obstacles=o,
        start=[0, 0],
        goal=[1, 1],
        expert_path=path,
        cost_to_go=values,
        reachable=np.isfinite(values),
    )
    index = {
        "schema": "passage-training-scenes/v1",
        "split": "validation",
        "dataset": {
            "id": "thalesgroup/passage",
            "revision": "c" * 40,
            "config": "256x256",
            "cost_model": "elevation",
            "cost_weight": 62.4,
        },
        "target_scale": 100.0,
        "scenes": [{"sample_id": "a", "file": "scenes/a.npz", "sha256": file_sha256(prepared / "scenes/a.npz")}],
    }
    (prepared / "index.json").write_text(json.dumps(index))
    run = tmp_path / "run"
    run.mkdir()
    from passage_eval.protocol.contracts import PlanRequest
    from passage_eval.campaign.candidate.transport import input_digest

    req = PlanRequest("a", z, o, (0, 0), (1, 1), "elevation", 62.4)
    candidate = {
        "candidate_id": "test-gnn",
        "family": "graph",
        "version": "1",
        "seed": 2026,
        "dataset": index["dataset"],
    }
    from hashlib import sha256
    from passage_eval.protocol.candidate_package import canonical_bytes

    manifest_hash = sha256(canonical_bytes(candidate)).hexdigest()
    rows = [
        {"kind": "session_start", "schema": "passage-candidate-session/v1", "manifest_sha256": manifest_hash},
        {
            "kind": "result",
            "sample_id": "a",
            "raw_input_sha256": input_digest(req),
            "path": path.tolist(),
            "status": status,
            "elapsed_ns": 1000000,
            "latency_censored": False,
        },
        {"kind": "session_complete", "n_completed": 1, "status_counts": {status: 1}},
    ]
    journal = "\n".join(json.dumps(r) for r in rows) + "\n"
    (run / "journal.jsonl").write_text(journal)
    doc = {
        "schema": "passage-candidate-run/v1",
        "manifest_sha256": manifest_hash,
        "journal_sha256": file_sha256(run / "journal.jsonl"),
        "n_completed": 1,
        "device": "cpu",
        "synchronization_complete": True,
        "runtime_close_complete": True,
        "timing_boundary": "raw_in_memory_arrays_to_emitted_path_non_supervised",
        "candidate": candidate,
    }
    (run / "run.json").write_text(json.dumps(doc))
    from .test_binding import receipt_fixture
    from passage_eval.campaign.candidate.binding import bind_container_evidence

    receipt_fixture(run)
    bind_container_evidence(run)
    return run, prepared / "index.json"


def test_host_recosts_exact_route_and_binds_input_identity(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path)
    row = score_candidate_run(run, index, target="cpu-station")
    assert row["metrics"]["success_rate"] == 1 and row["metrics"]["cost_regret_p90"] == 0
    assert row["latency"]["p95_ms"] == 1
    assert row["seed"] == 2026 and row["verification"] is None


def test_timeout_with_legal_route_is_still_penalized(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path, "timeout")
    row = score_candidate_run(run, index, target="cpu-station")
    assert row["metrics"]["success_rate"] == 0
    assert row["metrics"]["cost_regret_median"] == 999999
    assert row["metrics"]["path_assd"] is None


def test_tampered_journal_is_rejected(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path)
    with (run / "journal.jsonl").open("a") as f:
        f.write("{}\n")
    with pytest.raises(ValueError, match="journal"):
        score_candidate_run(run, index, target="cpu-station")


def test_wrong_executed_input_is_rejected_even_with_valid_journal_hash(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run
    from passage_eval.protocol.candidate_package import file_sha256

    run, index = score_fixture(tmp_path)
    rows = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines()]
    rows[1]["raw_input_sha256"] = "0" * 64
    (run / "journal.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    doc = json.loads((run / "run.json").read_text())
    doc["journal_sha256"] = file_sha256(run / "journal.jsonl")
    (run / "run.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="input identity"):
        score_candidate_run(run, index, target="cpu-station")


def test_failed_route_obstacle_violation_is_not_hidden(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run
    from passage_eval.protocol.candidate_package import file_sha256
    from passage_eval.campaign.candidate.transport import input_digest
    from passage_eval.protocol.contracts import PlanRequest

    run, index = score_fixture(tmp_path, "invalid_path")
    data_path = index.parent / "scenes/a.npz"
    with np.load(data_path) as d:
        data = {k: d[k] for k in d.files}
    data["obstacles"][0, 1] = True
    np.savez(data_path, **data)
    doc = json.loads(index.read_text())
    doc["scenes"][0]["sha256"] = file_sha256(data_path)
    index.write_text(json.dumps(doc))
    rows = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines()]
    rows[1]["path"] = [[0, 0], [0, 1], [1, 1]]
    rows[1]["raw_input_sha256"] = input_digest(
        PlanRequest("a", data["terrain"], data["obstacles"], (0, 0), (1, 1), "elevation", 62.4)
    )
    (run / "journal.jsonl").write_text("\n".join(json.dumps(x) for x in rows) + "\n")
    doc = json.loads((run / "run.json").read_text())
    doc["journal_sha256"] = file_sha256(run / "journal.jsonl")
    (run / "run.json").write_text(json.dumps(doc))
    row = score_candidate_run(run, index, target="cpu-station")
    assert row["metrics"]["success_rate"] == 0
    assert row["metrics"]["obstacle_violation_rate"] == 1


def test_candidate_identity_cannot_be_substituted_after_execution(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path)
    doc = json.loads((run / "run.json").read_text())
    doc["candidate"]["candidate_id"] = "other-candidate"
    (run / "run.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="manifest"):
        score_candidate_run(run, index, target="cpu-station")


def test_pilot_cannot_skip_binding_of_a_failed_oci_run(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path)
    (run / "oci-run.json").write_text(
        json.dumps({"returncode": 1, "timed_out": True, "child_namespace_drained": False})
    )
    with pytest.raises(ValueError, match="container|OCI"):
        score_candidate_run(run, index, target="cpu-station")


def test_private_score_preserves_terminal_stage_and_precision_attribution(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run
    from passage_eval.protocol.candidate_package import file_sha256

    run, index = score_fixture(tmp_path)
    rows = [json.loads(x) for x in (run / "journal.jsonl").read_text().splitlines()]
    rows[1].update(stage="fallback", precision="fp32", resource_peaks={"live_states": 20})
    (run / "journal.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    doc = json.loads((run / "run.json").read_text())
    doc["journal_sha256"] = file_sha256(run / "journal.jsonl")
    (run / "run.json").write_text(json.dumps(doc))
    score_candidate_run(run, index, target="cpu-station")
    score = json.loads((run / "score.json").read_text())
    assert score["terminal_stage_counts"] == {"fallback": 1}
    assert score["precision_counts"] == {"fp32": 1}
    assert score["reported_resource_peaks"]["live_states"] == 20


def test_unknown_historical_training_cost_is_explicit_not_zero(tmp_path):
    from passage_eval.campaign.candidate.scoring import score_candidate_run

    run, index = score_fixture(tmp_path)
    row = score_candidate_run(run, index, target="cpu-station", gpu_seconds=None)
    assert row["training"]["gpu_seconds"] is None
    with pytest.raises(ValueError, match="training.*cost"):
        score_candidate_run(run, index, target="cpu-station", gpu_seconds=-1)
