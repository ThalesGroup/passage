# MIT License
#
# Copyright (c) 2026 THALES
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Synthetic causal lease and drainage evidence; no target or test data."""

import copy
import pytest
from passage_eval.protocol import admission as A


def valid():
    job = "job:0123456789abcdef"
    scope = "/scope"
    obs = {
        "schema": "passage-cavops-job-observation/v1",
        "target": "thor",
        "procedure": "reg",
        "job_id": job,
        "ledger_id": "synthetic-ledger",
        "activity": {"id": job, "host": "edge-thor", "state": "done", "rc": 0},
        "events": [
            {
                "seq": 10,
                "activity_id": job,
                "type": "job.started",
                "data": {"host": "edge-thor", "resources": ["device:edge-thor"]},
            },
            {
                "seq": 12,
                "activity_id": job,
                "type": "job.ended",
                "data": {"rc": 0, "state": "done", "detected_by": "supervisor"},
            },
        ],
    }
    start = {
        "target": "thor",
        "procedure": "reg",
        "job_id": job,
        "scope": scope,
        "pid": 11,
        "namespace_pid": 1,
        "container_digest": "sha256:" + "a" * 64,
    }
    finish = {
        "scope": scope,
        "runner_pid": 11,
        "exit_code": 0,
        "journal_markers": "matched",
        "n_processes": 1,
        "n_restarts": 0,
        "container_digest": "sha256:" + "a" * 64,
    }
    drain = {
        "schema": "passage-direct-session-drain/v1",
        "target": "thor",
        "procedure": "reg",
        "job_id": job,
        "scope": scope,
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_processes": [],
        "handle_evidence": "process_exit_closes_session_descriptors",
        "runtime_synchronize_close_recorded": True,
        "driver_context_introspection": False,
    }
    return obs, start, finish, drain


def test_realistic_causal_receipts_do_not_need_invented_timestamps():
    assert A.native_causal_lease(*valid(), target="thor", procedure="reg") == (10, 12)


@pytest.mark.parametrize(
    "kind",
    [
        "different_job",
        "wrong_event_job",
        "wrong_resource",
        "resource_string",
        "not_drained",
        "failed",
        "reversed",
        "extra_process",
    ],
)
def test_causal_receipt_refuses_unproven_process_or_lease(kind):
    obs, start, finish, drain = copy.deepcopy(valid())
    if kind == "different_job":
        drain["job_id"] = "job:1111111111111111"
    if kind == "wrong_event_job":
        obs["events"][0]["activity_id"] = "job:1111111111111111"
    if kind == "wrong_resource":
        obs["events"][0]["data"]["resources"] = ["device:edge-orin"]
    if kind == "resource_string":
        obs["events"][0]["data"]["resources"] = "device:edge-thor"
    if kind == "not_drained":
        drain["cgroup_empty"] = False
    if kind == "failed":
        obs["events"][1]["data"]["rc"] = 1
    if kind == "reversed":
        obs["events"][1]["seq"] = 9
    if kind == "extra_process":
        finish["n_processes"] = 2
    with pytest.raises(ValueError):
        A.native_causal_lease(obs, start, finish, drain, target="thor", procedure="reg")


def test_g3_accepts_proven_causal_order_and_rejects_overlap(monkeypatch, tmp_path):
    import datetime as dt

    def entry(proc, first, last, job):
        obs, start, finish, drain = copy.deepcopy(valid())
        obs.update(procedure=proc, job_id=job)
        obs["activity"]["id"] = job
        obs["events"][0]["seq"] = first
        obs["events"][1]["seq"] = last
        for event in obs["events"]:
            event["activity_id"] = job
        start.update(procedure=proc, job_id=job)
        drain.update(procedure=proc, job_id=job)
        return {
            "cavops_observation": obs,
            "image_start": start,
            "image_finish": finish,
            "drain": drain,
            "native_execution": {"session_id": job},
        }

    import json
    import hashlib

    receipt = {
        "procedures": {
            "reg": entry("reg", 10, 12, "job:0123456789abcdef"),
            "seg": entry("seg", 13, 15, "job:1111111111111111"),
        }
    }
    record = {
        "protocol": {
            "decisions": {"targets": {"value": ["thor"]}, "paired_order": {"value": {"procedures": ["reg", "seg"]}}}
        }
    }
    from passage_eval.campaign.direct.raw_identity import task_from_full

    identity = {
        "graph_id": "a",
        "cost_model": "elevation",
        "cost_weight": 62.4,
        "start": [0, 0],
        "goal": [1, 1],
        "shape": [2, 2],
        "elevation_hash": "b",
        "obstacles_hash": "c",
    }
    index = json.dumps(task_from_full("sample", identity), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    index_sha = hashlib.sha256(index).hexdigest()
    record["candidates"] = {}
    for name, item in receipt["procedures"].items():
        sid = ("a" if name == "reg" else "b") * 32
        journal = b"\n".join(
            json.dumps(x).encode()
            for x in [
                {"kind": "session_start", "session_id": sid, "namespace_pid": 1},
                {"kind": "session_complete", "session_id": sid, "namespace_pid": 1},
                {"kind": "runtime_released"},
            ]
        )
        item["journal"] = journal
        item["native_execution"]["session_id"] = hashlib.sha256(journal).hexdigest()
        item["image_finish"]["runner_session_uuid"] = sid
        candidate = {
            "checkpoint_sha256": "b" * 64,
            "final_decoders_sha256": "d" * 64,
            "targets": {
                "thor": {"container_digest": "sha256:" + "a" * 64, "bundle_sha256": "e" * 64, "engine_sha256": "c" * 64}
            },
        }
        record["candidates"][name] = candidate
        item["native_execution"].update(
            container_digest="sha256:" + "a" * 64,
            artifacts={"checkpoint_sha256": "b" * 64, "engine_sha256": "c" * 64, "final_decoders_sha256": "d" * 64},
        )
        config = {
            "target": "thor",
            "procedure": name,
            "seed": 2026,
            "dataset_split": "validation",
            "budget": None,
            "index_sha256": index_sha,
            "artifact_pins": {"bundle": "e" * 64, "checkpoint": "b" * 64, "engine": "c" * 64, "decoder": "d" * 64},
        }
        item["configuration"] = json.dumps(config, sort_keys=True).encode()
        config_sha = hashlib.sha256(item["configuration"]).hexdigest()
        item["image_start"].update(configuration_sha256=config_sha, input_index_sha256=index_sha)
        item["target_receipt"] = {
            "configuration_sha256": config_sha,
            "input_index_sha256": index_sha,
            "journal_sha256": hashlib.sha256(journal).hexdigest(),
            "native_artifact_hashes": {
                "bundle": "e" * 64,
                "checkpoint": "b" * 64,
                "engine": "c" * 64,
                "decoder": "d" * 64,
            },
        }
        item["run_manifest"] = {
            "native_execution": item["native_execution"],
            "roster": ["sample"],
            "request_specs": {"sample": identity},
            "seed": 2026,
        }
    checker = A._Checker(record, tmp_path, tmp_path / "unused", dt.datetime.now(dt.timezone.utc))
    monkeypatch.setattr(checker, "_native_receipt", lambda *_: receipt)
    monkeypatch.setattr(checker, "load", lambda ref, *_, **__: ref)
    assert checker._native_g3().status == "GREEN"
    receipt["procedures"]["seg"]["cavops_observation"]["events"][0]["seq"] = 11
    assert "native_candidates_overlap" in [x["code"] for x in checker._native_g3().reasons]


def test_g3_binding_rejects_substituted_image_configuration_and_index():
    import hashlib
    import json
    from passage_eval.campaign.direct.raw_identity import task_from_full

    identity = {
        "graph_id": "a",
        "cost_model": "elevation",
        "cost_weight": 62.4,
        "start": [0, 0],
        "goal": [1, 1],
        "shape": [2, 2],
        "elevation_hash": "b",
        "obstacles_hash": "c",
    }
    index = json.dumps(task_from_full("sample", identity), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    index_sha = hashlib.sha256(index).hexdigest()
    journal = b"journal"
    journal_sha = hashlib.sha256(journal).hexdigest()
    execution = {
        "session_id": journal_sha,
        "container_digest": "sha256:" + "a" * 64,
        "artifacts": {"checkpoint_sha256": "b" * 64, "engine_sha256": "c" * 64, "final_decoders_sha256": "d" * 64},
    }
    candidate = {
        "checkpoint_sha256": "b" * 64,
        "final_decoders_sha256": "d" * 64,
        "targets": {
            "thor": {"container_digest": "sha256:" + "a" * 64, "bundle_sha256": "e" * 64, "engine_sha256": "c" * 64}
        },
    }
    start = {
        "container_digest": "sha256:" + "a" * 64,
        "configuration_sha256": "f" * 64,
        "input_index_sha256": index_sha,
    }
    receipt = {
        "configuration_sha256": "f" * 64,
        "input_index_sha256": index_sha,
        "journal_sha256": journal_sha,
        "native_artifact_hashes": {"bundle": "e" * 64, "checkpoint": "b" * 64, "engine": "c" * 64, "decoder": "d" * 64},
    }
    manifest = {
        "native_execution": execution,
        "roster": ["sample"],
        "request_specs": {"sample": identity},
        "seed": 2026,
    }
    config = {
        "target": "thor",
        "procedure": "reg",
        "seed": 2026,
        "dataset_split": "validation",
        "budget": None,
        "index_sha256": index_sha,
        "artifact_pins": receipt["native_artifact_hashes"],
    }
    raw = json.dumps(config, sort_keys=True).encode()
    start["configuration_sha256"] = receipt["configuration_sha256"] = hashlib.sha256(raw).hexdigest()
    A.native_causal_g2_binding(
        start, receipt, manifest, execution, candidate, journal, configuration_raw=raw, target="thor", procedure="reg"
    )
    for key in ("container_digest", "configuration_sha256", "input_index_sha256"):
        altered = copy.deepcopy(start)
        altered[key] = "0" * 64
        with pytest.raises(ValueError):
            A.native_causal_g2_binding(
                altered,
                receipt,
                manifest,
                execution,
                candidate,
                journal,
                configuration_raw=raw,
                target="thor",
                procedure="reg",
            )


def test_g3_rejects_both_observers_naming_substituted_configuration():
    import hashlib
    import json
    from passage_eval.campaign.direct.raw_identity import task_from_full

    ident = {
        "graph_id": "a",
        "cost_model": "elevation",
        "cost_weight": 62.4,
        "start": [0, 0],
        "goal": [1, 1],
        "shape": [2, 2],
        "elevation_hash": "b",
        "obstacles_hash": "c",
        "budget": {"wall_time_ms": 5000},
    }
    index = json.dumps(task_from_full("x", ident), sort_keys=True, separators=(",", ":")).encode() + b"\n"
    idx = hashlib.sha256(index).hexdigest()
    journal = b"journal"
    jsha = hashlib.sha256(journal).hexdigest()
    pins = {"bundle": "e" * 64, "checkpoint": "b" * 64, "engine": "c" * 64, "decoder": "d" * 64}
    config = {
        "target": "thor",
        "procedure": "reg",
        "seed": 2026,
        "dataset_split": "validation",
        "budget": {"wall_time_ms": 5000},
        "index_sha256": idx,
        "artifact_pins": pins,
    }
    raw = json.dumps(config, sort_keys=True).encode()
    config_sha = hashlib.sha256(raw).hexdigest()
    execution = {
        "session_id": jsha,
        "container_digest": "sha256:" + "a" * 64,
        "artifacts": {"checkpoint_sha256": "b" * 64, "engine_sha256": "c" * 64, "final_decoders_sha256": "d" * 64},
    }
    candidate = {
        "checkpoint_sha256": "b" * 64,
        "final_decoders_sha256": "d" * 64,
        "targets": {
            "thor": {"container_digest": "sha256:" + "a" * 64, "bundle_sha256": "e" * 64, "engine_sha256": "c" * 64}
        },
    }
    start = {"container_digest": "sha256:" + "a" * 64, "configuration_sha256": config_sha, "input_index_sha256": idx}
    target = {
        "configuration_sha256": config_sha,
        "input_index_sha256": idx,
        "journal_sha256": jsha,
        "native_artifact_hashes": pins,
    }
    manifest = {"native_execution": execution, "roster": ["x"], "request_specs": {"x": ident}, "seed": 2026}
    A.native_causal_g2_binding(
        start, target, manifest, execution, candidate, journal, configuration_raw=raw, target="thor", procedure="reg"
    )
    start["configuration_sha256"] = target["configuration_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        A.native_causal_g2_binding(
            start,
            target,
            manifest,
            execution,
            candidate,
            journal,
            configuration_raw=raw,
            target="thor",
            procedure="reg",
        )
    for field, changed in (("seed", 3407), ("budget", {"wall_time_ms": 1}), ("index_sha256", "0" * 64)):
        changed_config = copy.deepcopy(config)
        changed_config[field] = changed
        changed_raw = json.dumps(changed_config, sort_keys=True).encode()
        start["configuration_sha256"] = target["configuration_sha256"] = hashlib.sha256(changed_raw).hexdigest()
        with pytest.raises(ValueError):
            A.native_causal_g2_binding(
                start,
                target,
                manifest,
                execution,
                candidate,
                journal,
                configuration_raw=changed_raw,
                target="thor",
                procedure="reg",
            )
