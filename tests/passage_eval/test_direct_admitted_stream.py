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

"""Full synthetic stream, with all admission checks replaced by explicit fakes."""

import io
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
from passage_eval.campaign.direct import admitted_stream
from passage_eval.campaign.direct.direct_target import ADMITTED_SCHEMA, execute_stream
from passage_eval.execution.runner import request_identity
from passage_eval.protocol.contracts import BudgetSpec, PlanRequest


class Grant:
    report = SimpleNamespace(admitted={"profile": "descriptive_native_2026"})
    admissible_roster_sha256 = "a" * 64

    def ensure_unchanged(self):
        pass


def request(i):
    return PlanRequest(
        str(i),
        np.full((3, 3), float(i)),
        np.zeros((3, 3), bool),
        (0, 0),
        (2, 2),
        "elevation",
        62.4,
        budget=BudgetSpec(wall_time_ms=1000),
    )


def task(r):
    return {"sample_id": r.sample_id, "input_identity": request_identity(r)}


class Runtime:
    def __init__(self):
        self.calls = 0
        self.closed = False

    def predict(self, tensor):
        self.calls += 1
        return np.zeros((1, 1, 3, 3), dtype=np.float32)

    def synchronize(self):
        pass

    def close(self):
        self.closed = True


class Decoder:
    def build_input(self, *args):
        return np.zeros((1, 3, 3, 3), dtype=np.float32)

    def reg_bellman_descent(self, field, obstacles, start, goal):
        return {"status": "ok", "path": [start, (1, 1), goal], "descent_steps": 2}


def setup(monkeypatch, tmp_path):
    measured = [request(i) for i in range(1, 1025)]
    warm = request(0)
    roster = {"dataset": {"split": "test"}, "tasks": [task(r) for r in measured]}
    warm_roster = {"dataset": {"split": "validation"}, "tasks": [task(warm)], "roster_sha256": "e" * 64}

    def write_ref(name, value):
        raw = json.dumps(value).encode()
        (tmp_path / name).write_bytes(raw)
        return {"path": name, "sha256": hashlib.sha256(raw).hexdigest()}

    warm_ref = write_ref("warm.json", warm_roster)
    env_ref = write_ref(
        "env.json",
        {
            "stack": {"precision": "FP16 mixed", "runtime": "TensorRT"},
            "runtime_expectation": {"sdk": "tensorrt", "version": "10.16.2.10"},
        },
    )
    pins = {"bundle": "1" * 64, "engine": "2" * 64, "checkpoint": "3" * 64, "decoder": "4" * 64}
    record = {
        "protocol": {
            "profile": "descriptive_native_2026",
            "decisions": {
                "timing_profile": {"value": "native_raw_path_persistent_v1"},
                "targets": {"value": ["orin"]},
                "seeds": {"value": [2026]},
                "budget": {"value": warm.budget.to_dict()},
            },
        },
        "evidence": {"warmup": {"roster": warm_ref}},
        "candidates": {
            "reg": {
                "checkpoint_sha256": pins["checkpoint"],
                "final_decoders_sha256": pins["decoder"],
                "targets": {
                    "orin": {
                        "bundle_sha256": pins["bundle"],
                        "engine_sha256": pins["engine"],
                        "native_environment": env_ref,
                    }
                },
            }
        },
    }
    ref = write_ref("record.json", record)
    grant = Grant()
    grant.record_path = tmp_path / "record.json"
    grant.record_sha256 = ref["sha256"]
    grant.evidence_dir = tmp_path
    monkeypatch.setattr(admitted_stream, "grant_test_admission", lambda *a: grant)
    monkeypatch.setattr(admitted_stream.roster_io, "load_roster", lambda *a, **k: roster)
    monkeypatch.setattr(admitted_stream.roster_io, "validate_roster", lambda r: r)
    monkeypatch.setattr(admitted_stream, "refuse_test_roster", lambda *a, **k: None)
    monkeypatch.setattr(admitted_stream, "admitted_requests", lambda *a: iter(measured))
    session = admitted_stream.prepare_admitted_session(
        "record", "evidence", "roster", "dataset", warm.budget, warm, warm_roster
    )
    cfg = {
        "schema": ADMITTED_SCHEMA,
        "dataset_split": "test",
        "target": "orin",
        "procedure": "reg",
        "seed": 2026,
        "artifact_paths": {},
        "artifact_pins": pins,
        "compiled_manifest": "/x",
        "compiled_sha256": "a" * 64,
        "exploratory_unadmitted_precision": False,
        "budget": warm.budget.to_dict(),
        "warmup_index": list(session.warmup_index),
        "index_sha256": session.index_sha256,
        "checkpoint_size": 32,
        "grn_evidence": None,
        "native_stack": {"precision": "FP16 mixed", "runtime": "TensorRT"},
    }
    return admitted_stream.bind_target_config(session, cfg), measured, cfg


def test_admitted_binary_stream_drives_exactly_one_persistent_session(monkeypatch, tmp_path):
    session, _, cfg = setup(monkeypatch, tmp_path)
    stream = io.BytesIO()
    admitted_stream.stream_admitted_session(stream, session, fragment_size=32)
    stream.seek(0)
    runtime = Runtime()
    result = execute_stream(stream, cfg, lambda: runtime, Decoder(), tmp_path / "result")
    assert result["n_completed"] == 1024 and result["warmups_completed"] == 16
    assert runtime.calls == 1040 and runtime.closed
    rows = [json.loads(x) for x in (tmp_path / "result" / "journal.jsonl").read_text().splitlines()]
    assert sum(x["kind"] == "request" for x in rows) == 1024


def test_producer_refuses_changed_input_before_sending_fragment(monkeypatch, tmp_path):
    session, requests, _ = setup(monkeypatch, tmp_path)
    requests[0] = request(2048)
    with pytest.raises(ValueError, match="differs"):
        admitted_stream.stream_admitted_session(io.BytesIO(), session, fragment_size=32)


def test_admitted_endpoint_does_not_accept_validation_schema_as_test():
    from passage_eval.campaign.direct.direct_target import validate_config, SCHEMA

    cfg = {
        "schema": SCHEMA,
        "dataset_split": "test",
        "target": "orin",
        "procedure": "reg",
        "seed": 2026,
        "artifact_paths": {},
        "artifact_pins": {},
        "compiled_manifest": "/x",
        "compiled_sha256": "a" * 64,
        "exploratory_unadmitted_precision": True,
        "budget": {},
        "warmup_index": [],
        "index_sha256": "b" * 64,
        "checkpoint_size": 32,
        "grn_evidence": None,
    }
    with pytest.raises(ValueError, match="split"):
        validate_config(cfg)


def test_config_artifact_swap_is_refused(monkeypatch, tmp_path):
    session, _, cfg = setup(monkeypatch, tmp_path)
    cfg["artifact_pins"] = {**cfg["artifact_pins"], "engine": "9" * 64}
    with pytest.raises(ValueError, match="artifact pins"):
        admitted_stream.bind_target_config(session, cfg)


def test_descriptive_precision_is_a_distinct_profile_and_strict_stays_refused(monkeypatch):
    import sys
    import types
    from passage_eval.campaign.direct import native_preflight as n

    pins = {"bundle": "b" * 64, "engine": "e" * 64, "checkpoint": n.CHECKPOINTS["reg"], "decoder": n.DECODER}
    bundle = types.SimpleNamespace(
        manifest_sha256=pins["bundle"], manifest={"lineage": {"checkpoint_sha256": pins["checkpoint"]}}
    )
    monkeypatch.setattr(n, "native_config", lambda *a, **kw: {})
    monkeypatch.setattr(n, "verify_files", lambda *a, **kw: pins)
    monkeypatch.setattr(n, "load_candidate_bundle", lambda *a: bundle)
    monkeypatch.setattr(n, "_sha256_regular_file", lambda path: "c" * 64 if path == "manifest" else pins["engine"])
    engine = types.ModuleType("passage_inference.engine")
    engine.checked_engine = lambda *a: ("engine", {"precision": "fp16", "precision_admission": {"status": "not_run"}})
    package = types.ModuleType("passage_inference")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "passage_inference", package)
    monkeypatch.setitem(sys.modules, "passage_inference.engine", engine)
    with pytest.raises(ValueError, match="no admitted precision"):
        n.validate_native_files("reg", "orin", {"bundle": "bundle"}, pins, "manifest", "c" * 64)
    facts = n.validate_native_files(
        "reg", "orin", {"bundle": "bundle"}, pins, "manifest", "c" * 64, precision_profile="native_descriptive"
    )
    assert facts["precision_profile"] == "native_descriptive"
    assert "not proof" in facts["precision_note"]
    with pytest.raises(ValueError, match="distinct"):
        n.validate_native_files(
            "reg",
            "orin",
            {"bundle": "bundle"},
            pins,
            "manifest",
            "c" * 64,
            exploratory=True,
            precision_profile="native_descriptive",
        )


def test_admitted_receipt_is_bound_to_config_index_and_engine(monkeypatch, tmp_path):
    session, _, cfg = setup(monkeypatch, tmp_path)
    output = tmp_path / "received"
    output.mkdir()
    journal = output / "journal.jsonl"
    journal.write_text("completed synthetic journal\n")
    receipt = {
        "schema": "passage-direct-target-admitted-s-result/v1",
        "configuration_sha256": hashlib.sha256(session.config_bytes).hexdigest(),
        "input_index_sha256": session.index_sha256,
        "native_artifact_hashes": cfg["artifact_pins"],
        "qualification_status": "host_admission_verification_required",
        "scope_gpu_drain_verified": False,
        "precision_profile": "native_descriptive",
        "native_stack": cfg["native_stack"],
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "runtime_identity": {
            "implementation": "passage_inference.backends.TensorRTRuntime",
            "options": {"device_id": 0, "engine_sha256": cfg["artifact_pins"]["engine"]},
            "versions": {
                "tensorrt": {"distributions": {"tensorrt": "10.16.2.10"}, "loaded_module_version": "10.16.2.10"}
            },
        },
    }
    raw = json.dumps(receipt).encode()
    (output / "target-receipt.json").write_bytes(raw)
    (output / "target-receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())
    assert admitted_stream.verify_admitted_target_receipt(output, session) == receipt
    bad = json.loads(json.dumps(receipt))
    bad["runtime_identity"]["versions"]["tensorrt"]["loaded_module_version"] = "10.9"
    bad_raw = json.dumps(bad).encode()
    (output / "target-receipt.json").write_bytes(bad_raw)
    (output / "target-receipt.sha256").write_text(hashlib.sha256(bad_raw).hexdigest())
    with pytest.raises(ValueError, match="loaded SDK"):
        admitted_stream.verify_admitted_target_receipt(output, session)
    (output / "target-receipt.json").write_bytes(raw)
    (output / "target-receipt.sha256").write_text(hashlib.sha256(raw).hexdigest())
    journal.write_text("changed")
    with pytest.raises(ValueError, match="journal digest"):
        admitted_stream.verify_admitted_target_receipt(output, session)


@pytest.mark.parametrize("field", ["missing", "sdk", "engine", "device"])
def test_runtime_observation_rejects_missing_or_mismatched_facts(field):
    ident = {
        "implementation": "passage_inference.backends.TensorRTRuntime",
        "options": {"device_id": 0, "engine_sha256": "a" * 64},
        "versions": {"tensorrt": {"distributions": {"tensorrt": "10.16.2.10"}, "loaded_module_version": "10.16.2.10"}},
    }
    expected = {"sdk": "tensorrt", "version": "10.16.2.10"}
    assert admitted_stream._verify_runtime_identity(ident, expected, "orin", "a" * 64)
    if field == "missing":
        ident = {}
    elif field == "sdk":
        ident["versions"]["tensorrt"]["loaded_module_version"] = "10.9"
    elif field == "engine":
        ident["options"]["engine_sha256"] = "b" * 64
    else:
        ident["options"]["device_id"] = 1
    with pytest.raises(ValueError):
        admitted_stream._verify_runtime_identity(ident, expected, "orin", "a" * 64)


def test_openvino_requires_observed_gpu_and_loaded_sdk():
    ident = {
        "implementation": "passage_inference.backends.OpenVinoRuntime",
        "options": {"device": "GPU", "execution_devices": ["GPU.0"]},
        "versions": {
            "openvino": {"distributions": {"openvino": "2026.4.1"}, "loaded_module_version": "2026.4.1-22982-release"}
        },
    }
    expected = {"sdk": "openvino", "version": "2026.4.1"}
    assert admitted_stream._verify_runtime_identity(ident, expected, "tiger", "a" * 64)
    ident["options"]["device"] = "GPU.0"
    assert admitted_stream._verify_runtime_identity(ident, expected, "tiger", "a" * 64)
    ident["options"]["execution_devices"] = ["CPU"]
    with pytest.raises(ValueError, match="accelerator"):
        admitted_stream._verify_runtime_identity(ident, expected, "tiger", "a" * 64)
