"""Synthetic checks for Tiger's image/rootfs/scoped-process observation."""

import pytest
from passage_eval.campaign.direct.rule8_bwrap import verify_running_bwrap

JOB = "job:0123456789abcdef"
SCOPE = "passage-edge-tiger-0123456789abcdef.scope"
DIGEST = "sha256:" + "a" * 64
CONFIG = "b" * 64
INDEX = "c" * 64


def observation():
    return {
        "job_id": JOB,
        "comparison": {
            "manifest_digest": DIGEST,
            "config_digest": "sha256:" + "d" * 64,
            "rootfs_archive_sha256": "e" * 64,
            "exceptions": {},
            "export_paths": 3,
        },
        "tree": {"rootfs_archive_sha256": "e" * 64, "file_count": 3, "rootfs_path": "/rootfs"},
        "process": {
            "target": "tiger",
            "rootfs": "/rootfs",
            "job_id": JOB,
            "scope": SCOPE,
            "runner": {
                "pid": 42,
                "namespace_pid": 2,
                "root_mount_options": "ro,nosuid",
                "argv": [
                    "/opt/venv/bin/python3",
                    "-I",
                    "-B",
                    "/opt/passage-direct/direct_target.py",
                    "--config",
                    "/run/config",
                    "--config-sha256",
                    CONFIG,
                    "--output",
                    "/run/result",
                ],
            },
        },
    }


def check(proof, process_in_scope=lambda pid, scope: pid == 42 and scope == SCOPE):
    return verify_running_bwrap(
        proof, DIGEST, config_sha256=CONFIG, index_sha256=INDEX, process_in_scope=process_in_scope
    )


def test_scoped_bwrap_start_is_bound_to_image_and_rootfs():
    result = check(observation())
    assert result["namespace_pid"] == 2 and result["scope"] == SCOPE


@pytest.mark.parametrize(
    "kind", ["image", "rootfs", "pid", "scope", "writable", "runner", "config", "tree_path", "exception"]
)
def test_rejects_substitution(kind):
    p = observation()
    if kind == "image":
        p["comparison"]["manifest_digest"] = "sha256:" + "f" * 64
    elif kind == "rootfs":
        p["tree"]["rootfs_archive_sha256"] = "f" * 64
    elif kind == "pid":
        p["process"]["runner"]["pid"] = 43
    elif kind == "scope":
        p["process"]["scope"] = "other.scope"
    elif kind == "writable":
        p["process"]["runner"]["root_mount_options"] = "rw"
    elif kind == "runner":
        p["process"]["runner"]["argv"][3] = "/tmp/other.py"
    elif kind == "config":
        p["process"]["runner"]["argv"][-3] = "f" * 64
    elif kind == "tree_path":
        p["tree"]["rootfs_path"] = "/different"
    else:
        p["comparison"]["exceptions"] = {
            "opt/passage-direct/direct_session.py": {"oci": None, "export": ["file", 1, "a" * 64, True]}
        }
    with pytest.raises(ValueError):
        check(p)


def test_refuses_lease_scope_mismatch():
    with pytest.raises(ValueError):
        check(observation(), process_in_scope=lambda *_: False)


def test_admitted_stream_persists_bwrap_evidence_before_psi1(monkeypatch, tmp_path):
    import hashlib
    import io
    import json
    from passage_eval.campaign.direct import admitted_stream

    config = {"target": "tiger", "procedure": "reg", "seed": 2026}
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    proof = observation()
    proof["process"]["runner"]["argv"][-3] = digest
    record = {"candidates": {"reg": {"targets": {"tiger": {"container_digest": DIGEST}}}}}

    class Grant:
        def ensure_unchanged(self):
            pass

    class Scope:
        def check_record(self, grant):
            return record

    session = admitted_stream.AdmittedSession(
        "record",
        "evidence",
        "roster",
        "dataset",
        None,
        Grant(),
        b"index",
        hashlib.sha256(b"index").hexdigest(),
        (),
        None,
        (),
        "e" * 64,
        config_bytes=raw,
        rule8_scope=Scope(),
    )
    monkeypatch.setattr(admitted_stream, "bind_target_config", lambda current, config: current)
    monkeypatch.setattr(admitted_stream, "admitted_requests", lambda *a, **kw: iter(()))
    monkeypatch.setattr(admitted_stream, "_frame", lambda *a, **kw: None)
    evidence = tmp_path / "start.json"
    output = io.BytesIO()
    admitted_stream.stream_admitted_session(
        output,
        session,
        bwrap_inspect=lambda: proof,
        process_in_scope=lambda pid, scope: evidence.exists() is False and (pid, scope) == (42, SCOPE),
        image_evidence_path=evidence,
    )
    assert evidence.is_file() and output.getvalue().startswith(b"PSI1")
    assert json.loads(evidence.read_bytes())["mechanism"] == "bwrap"

    changed = {"value": False}

    def ensure():
        if changed["value"]:
            raise RuntimeError("grant changed")

    monkeypatch.setattr(session.grant, "ensure_unchanged", ensure)

    def changed_after_observation(*_):
        changed["value"] = True
        return True

    blocked = io.BytesIO()
    with pytest.raises(RuntimeError, match="grant changed"):
        admitted_stream.stream_admitted_session(
            blocked,
            session,
            bwrap_inspect=lambda: proof,
            process_in_scope=changed_after_observation,
            image_evidence_path=tmp_path / "second.json",
        )
    assert blocked.getvalue() == b""


def test_bwrap_completion_reaches_scorer_without_docker_inspection(monkeypatch, tmp_path):
    import hashlib
    import json
    from types import SimpleNamespace
    from passage_eval.campaign.direct import admitted_stream, direct_host_scoring
    from passage_eval.io import roster as roster_io

    config = {"target": "tiger", "procedure": "reg", "seed": 2026}
    raw = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    job = JOB
    record = {"candidates": {"reg": {"targets": {"tiger": {"container_digest": DIGEST}}}}}

    class Grant:
        admissible_roster_sha256 = "a" * 64

        def ensure_unchanged(self):
            pass

    class Scope:
        def check_record(self, grant):
            return record

    session = SimpleNamespace(
        rule8_scope=Scope(), config_bytes=raw, grant=Grant(), index_sha256=INDEX, roster_path="unused"
    )
    proof = observation()
    proof["process"]["runner"]["argv"][-3] = hashlib.sha256(raw).hexdigest()
    start = verify_running_bwrap(
        proof,
        DIGEST,
        config_sha256=hashlib.sha256(raw).hexdigest(),
        index_sha256=INDEX,
        process_in_scope=lambda *_: True,
    )
    start.update(
        schema="passage-rule8-host-image-start/v1",
        mechanism="bwrap",
        target="tiger",
        procedure="reg",
        seed=2026,
        bwrap_proof=proof,
    )
    evidence = tmp_path / "start.json"
    evidence.write_text(json.dumps(start))
    start_sha = hashlib.sha256(evidence.read_bytes()).hexdigest()
    root = tmp_path / "target"
    root.mkdir()
    sid = "a" * 32
    (root / "journal.jsonl").write_text(
        json.dumps({"kind": "session_start", "session_id": sid, "namespace_pid": 2})
        + "\n"
        + json.dumps({"kind": "session_complete", "session_id": sid, "namespace_pid": 2})
        + "\n"
    )
    journal_sha = hashlib.sha256((root / "journal.jsonl").read_bytes()).hexdigest()
    monkeypatch.setattr(admitted_stream, "verify_admitted_target_receipt", lambda *_: {"journal_sha256": journal_sha})
    drain = {
        "schema": "passage-direct-session-drain/v1",
        "target": "tiger",
        "procedure": "reg",
        "job_id": job,
        "scope": SCOPE,
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    result = admitted_stream.verify_rule8_completed_session(
        session, root, evidence, start_sha, container_inspect=None, drain=drain, job_id=job
    )
    assert result["runner_pid"] == 42 and result["journal_sha256"] == journal_sha
    poisoned = json.loads(evidence.read_text())
    poisoned["bwrap_proof"]["tree"]["rootfs_path"] = "/another-root"
    evidence.write_text(json.dumps(poisoned))
    with pytest.raises(ValueError):
        admitted_stream.verify_rule8_completed_session(
            session,
            root,
            evidence,
            hashlib.sha256(evidence.read_bytes()).hexdigest(),
            container_inspect=None,
            drain=drain,
            job_id=job,
        )
    evidence.write_text(json.dumps(start))

    class ReachedHostScorer(Exception):
        pass

    def stop_at_roster(*a, **kw):
        raise ReachedHostScorer()

    monkeypatch.setattr(roster_io, "load_roster", stop_at_roster)
    with pytest.raises(ReachedHostScorer):
        direct_host_scoring.score_rule8_completed_session(
            session,
            root,
            evidence,
            start_sha,
            [],
            {},
            container_inspect=None,
            drain=drain,
            job_id=job,
            reference_cohort_id="synthetic",
        )
