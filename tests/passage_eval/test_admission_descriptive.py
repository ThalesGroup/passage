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

"""Descriptive native admission on synthetic evidence only; no released split is read."""

import copy
from dataclasses import replace
import importlib.util
import sys
from pathlib import Path

import pytest
from passage_eval.io import pinned, roster as roster_io
from passage_eval.protocol import admission as A

HERE = Path(__file__).resolve().parent
NAME = "passage_eval_tests_campaign_world"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, HERE / "campaign_world.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
W = sys.modules[NAME]

PROFILE = "descriptive_native_2026"
DESCRIPTIVE = {"mode": "descriptive_only"}


def profile_record(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["profile"] = PROFILE
    decisions = record["protocol"]["decisions"]
    updates = {
        "dataset_revision": {
            "repo": "thalesgroup/passage",
            "commit": "c09cc5a21a55a93e7c09d47368d5f5bf75cb8fae",
            "config": "256x256",
            "resolution": 256,
        },
        "evaluation_cell": {"cost_model": "elevation", "cost_weight": 62.4, "task_variant": "obstacles"},
        "roster_selection": {
            "rule": "sha256(utf-8(salt + id)) hexdigest, ascending lexicographic",
            "salt": "passage-public-test-v1/r0256/roster-v1/",
            "n": 1024,
        },
        "population": {"n_rows": 1024, "ids_sha256": "a" * 64},
        "paired_order": {"granularity": "per_batch", "procedures": ["seg", "reg"], "schedule_seed": 2026},
        "targets": ["orin"],
        "seeds": [2026],
        "confirmatory_analysis": DESCRIPTIVE,
        "reference_policy": "independent_analytic_reference",
        "reference_cohort_id": "unit-test-reference-cohort",
        "reference_max_expansions": 65536,
    }
    for name, value in updates.items():
        decisions[name]["value"] = copy.deepcopy(value)
    record["evidence"]["references"]["historical_roster"] = None
    record["candidates"] = W.candidates_block(world.evidence, targets=("orin",))
    for candidate in record["candidates"].values():
        candidate["targets"]["orin"].update(container_digest=None, native_environment=None)
    return record


def persistent_record(world):
    record = profile_record(world)
    decisions = record["protocol"]["decisions"]
    for name, value in {
        "timing_profile": "native_raw_path_persistent_v1",
        "timing_boundary_id": "passage-timing-boundary/v6-native-raw-arrays-to-path",
        "max_restarts": 0,
        "r7_policy": {"mode": "abort_incomplete_session_no_automatic_replay"},
    }.items():
        decisions[name]["value"] = value
    decisions["paired_order"]["value"]["granularity"] = "per_candidate"
    decisions["warmup"]["value"].update(k=1, repeats=16)
    record["evidence"]["native_chain_rehearsal"] = None
    return record


def native_validation(world, record, monkeypatch):
    from passage_eval.protocol.warmup import build_warmup_roster

    data = W.green_world().world

    def pool(n):
        return data.TR.BUILD.build_roster(
            data.dataset,
            benchmark_id="validation-inputs",
            resolution=3,
            n_expected=n,
            salt=W.WARM_SALT,
            split="validation",
            require_footprints=False,
        )[0]

    measured = pool(2)
    warmup, _ = build_warmup_roster(
        pool, salt=W.WARM_SALT, evaluated_roster=measured, benchmark_id="validation-warmup", k=1
    )
    # Exercise the real input binding with small synthetic validation data, never the released split.
    revision = {key: measured["dataset"][key] for key in ("repo", "commit", "config", "resolution")}
    monkeypatch.setattr(A, "DESCRIPTIVE_DATASET", revision)
    record["protocol"]["decisions"]["dataset_revision"]["value"] = revision
    for label, roster in (("measured", measured), ("warmup", warmup)):
        W.write_json(world.evidence / f"native-{label}-roster.json", roster)
    receipt = {
        "schema": "passage-native-persistent-validation/v1",
        "target": "orin",
        "split": "validation",
        "harness": record["harness"],
        "plan_sha256": "a" * 64,
        "procedures": {},
    }
    for index, name in enumerate(("seg", "reg")):
        candidate = record["candidates"][name]
        aggregate = W.rehearsal_aggregate(record, name)
        aggregate.update(
            target="orin",
            plan_sha256="a" * 64,
            sessions={"n_sessions": 1},
            admissible_roster_sha256=measured["roster_sha256"],
            population={"n_source": 2, "n_excluded": 0, "n_admissible": 2, "n_scored": 2},
        )
        aggregate["per_seed"]["2026"]["population"] = aggregate["population"]
        for block in (aggregate["latency"], aggregate["per_seed"]["2026"]["latency"]):
            block.update(n_warm_requests=2, min_warm_requests=1, meets_min_warm_requests=True)
        path = world.evidence / f"native-validation-{name}-aggregate.json"
        W.write_json(path, aggregate)
        start = f"2026-11-01T00:0{index * 3}:00+00:00"
        finish = f"2026-11-01T00:0{index * 3 + 1}:00+00:00"
        drained = f"2026-11-01T00:0{index * 3 + 2}:00+00:00"
        receipt["procedures"][name] = {
            "aggregate": W.ref(world.evidence, path),
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "engine_sha256": candidate["targets"]["orin"]["engine_sha256"],
            "completed": True,
            "accelerator_used": True,
            "native_execution": {
                "session_id": f"process-{name}",
                "n_processes": 1,
                "n_restarts": 0,
                "warmup": {
                    "k": 1,
                    "repeats": 16,
                    "performed": True,
                    "input_roster_sha256": pinned.canonical_sha256(A.native_input_projection(warmup)),
                    "disjoint_by_content": True,
                },
            },
            "process": {
                "started_at": start,
                "finished_at": finish,
                "drained_at": drained,
                "process_exited": True,
                "gpu_context_released": True,
            },
            "lease": {
                "resource": "device:edge-orin",
                "holder_host": "edge-orin",
                "activity_id": f"job:{index + 1:016x}",
                "job_id": f"orin-job-{index}",
                "acquired_at": start,
                "released_at": drained,
            },
        }
        entry = receipt["procedures"][name]
        entry["roster"] = W.ref(world.evidence, world.evidence / "native-measured-roster.json")
        entry["warmup_roster"] = W.ref(world.evidence, world.evidence / "native-warmup-roster.json")
        execution = entry["native_execution"]
        execution["input_roster_sha256"] = pinned.canonical_sha256(A.native_input_projection(measured))
        execution["artifacts"] = {
            "checkpoint_sha256": candidate["checkpoint_sha256"],
            "engine_sha256": entry["engine_sha256"],
            "final_decoders_sha256": candidate["final_decoders_sha256"],
        }
        manifest_dir = world.evidence / f"native-run-{name}"
        manifest_dir.mkdir()
        for filename in ("requests.jsonl", "paths.npz"):
            (manifest_dir / filename).write_bytes(b"synthetic artifact bytes for verifier binding tests")
        manifest = {
            "schema": "passage-complete-request-native-v1",
            "seed": 2026,
            "timing_profile": "native_raw_path_persistent_v1",
            "timing_boundary": "passage-timing-boundary/v6-native-raw-arrays-to-path",
            "procedure": {"execution": {"timing_profile": "native_raw_path_persistent_v1"}},
            "native_execution": execution,
            "roster": [task["sample_id"] for task in measured["tasks"]],
            "request_specs": {
                task["sample_id"]: {
                    **task["input_identity"],
                    "budget": record["protocol"]["decisions"]["budget"]["value"],
                }
                for task in measured["tasks"]
            },
            "artifact_hashes": {
                filename: pinned.sha256_bytes((manifest_dir / filename).read_bytes())
                for filename in ("requests.jsonl", "paths.npz")
            },
        }
        W.write_json(manifest_dir / "run_manifest.json", manifest)
        entry["run_manifest"] = W.ref(world.evidence, manifest_dir / "run_manifest.json")
        aggregate["per_seed"]["2026"].update(
            native_execution=execution, run_manifest_sha256=entry["run_manifest"]["sha256"]
        )
        W.write_json(path, aggregate)
        entry["aggregate"] = W.ref(world.evidence, path)
    path = world.evidence / "native-validation-orin.json"
    W.write_json(path, receipt)
    record["evidence"]["edge"]["orin"] = W.ref(world.evidence, path)
    return path, receipt


@pytest.fixture
def world(tmp_path):
    return W.copy_green(tmp_path / "world")


def check(world, record):
    W.save_record(world, record)
    return A.check_admission(world.record_path, world.evidence)


def codes(report, gate):
    return [reason["code"] for reason in report.gate(gate).reasons]


def test_profile_seals_an_explicit_single_seed_scope_without_a_confirmatory_promise(world):
    report = check(world, profile_record(world))
    assert report.gate("G1").status == "GREEN", report.gate("G1").reasons
    scope = report.to_dict()["claim_scope"]
    assert scope == {"profile": PROFILE, "confirmatory_allowed": False, "reproducible_entry": False}


def test_persistent_native_profile_needs_no_supervised_host_rehearsal_or_fragment_rewarm(world, monkeypatch):
    record = persistent_record(world)
    native_validation(world, record, monkeypatch)
    report = check(world, record)
    assert report.gate("G1").status == "GREEN", report.gate("G1").reasons
    assert report.gate("G2").status == "GREEN", report.gate("G2").reasons
    assert report.gate("G3").status == "GREEN", report.gate("G3").reasons


@pytest.mark.parametrize(
    "change",
    [
        "overlap",
        "not_released",
        "second_process",
        "second_warmup",
        "wrong_candidate",
        "warmup_projection",
        "session_substitution",
        "emitted_engine",
    ],
)
def test_persistent_native_proof_keeps_identity_and_complete_candidate_release(world, change, monkeypatch):
    record = persistent_record(world)
    path, receipt = native_validation(world, record, monkeypatch)
    seg = receipt["procedures"]["seg"]
    if change == "overlap":
        receipt["procedures"]["reg"]["process"]["started_at"] = seg["process"]["finished_at"]
    elif change == "not_released":
        seg["process"]["gpu_context_released"] = False
    elif change == "second_process":
        seg["native_execution"]["n_processes"] = 2
    elif change == "second_warmup":
        seg["native_execution"]["warmup"]["repeats"] = 32
    elif change == "wrong_candidate":
        seg["checkpoint_sha256"] = "d" * 64
    elif change == "warmup_projection":
        seg["native_execution"]["warmup"]["input_roster_sha256"] = "d" * 64
    elif change == "session_substitution":
        seg["native_execution"]["session_id"] = "another-process"
    else:
        seg["native_execution"]["artifacts"]["engine_sha256"] = "d" * 64
    W.write_json(path, receipt)
    record["evidence"]["edge"]["orin"] = W.ref(world.evidence, path)
    report = check(world, record)
    assert report.gate("G2").status == "RED" or report.gate("G3").status == "RED"


def test_native_rehearsal_computes_disjointness_instead_of_trusting_the_receipt_boolean(world, monkeypatch):
    record = persistent_record(world)
    path, receipt = native_validation(world, record, monkeypatch)
    data = W.green_world().world
    overlapping = data.TR.BUILD.build_roster(
        data.dataset,
        benchmark_id="overlapping-validation",
        resolution=3,
        n_expected=1,
        salt=W.WARM_SALT,
        split="validation",
        require_footprints=False,
    )[0]
    overlap_path = world.evidence / "overlapping-warmup.json"
    W.write_json(overlap_path, overlapping)
    receipt["procedures"]["seg"]["warmup_roster"] = W.ref(world.evidence, overlap_path)
    W.write_json(path, receipt)
    record["evidence"]["edge"]["orin"] = W.ref(world.evidence, path)
    report = check(world, record)
    assert any(
        reason["code"] == "native_measurement_binding_invalid" and "overlaps" in reason["detail"]
        for reason in report.gate("G2").reasons
    )


def test_native_rehearsal_pins_the_actual_emitted_payload_files(world, monkeypatch):
    record = persistent_record(world)
    native_validation(world, record, monkeypatch)
    (world.evidence / "native-run-seg" / "paths.npz").write_bytes(b"changed after emission")
    assert check(world, record).gate("G2").status == "RED"


def test_g5_cannot_substitute_a_different_warmup_for_the_native_rehearsal(world, monkeypatch):
    record = persistent_record(world)
    native_validation(world, record, monkeypatch)
    assert "native_warmup_projection_differs" in codes(check(world, record), "G5")


def test_native_timing_is_not_an_escape_for_strict_admission(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["timing_profile"]["value"] = "native_raw_path_persistent_v1"
    record["protocol"]["decisions"]["timing_boundary_id"]["value"] = (
        "passage-timing-boundary/v6-native-raw-arrays-to-path"
    )
    assert check(world, record).gate("G1").status == "RED"


def test_persistent_native_scope_still_refuses_full_test_population(world):
    record = persistent_record(world)
    record["protocol"]["decisions"]["roster_selection"]["value"]["n"] = 40000
    record["protocol"]["decisions"]["population"]["value"]["n_rows"] = 40000
    assert check(world, record).gate("G1").status == "RED"


@pytest.mark.parametrize("seed", [1234, 2026, 3407])
def test_separate_descriptive_records_can_declare_one_planned_seed_without_relabelling_weights(world, seed):
    record = persistent_record(world)
    record["protocol"]["profile"] = "descriptive_native_single_seed"
    record["protocol"]["decisions"]["seeds"]["value"] = [seed]
    report = check(world, record)
    assert report.gate("G1").status == "GREEN", report.gate("G1").reasons
    assert report.to_dict()["claim_scope"]["reproducible_entry"] is False
    if seed != 2026:
        assert "seed_checkpoint_not_frozen" in codes(report, "G6")


@pytest.mark.parametrize("seeds", [[2026, 1234], [9999]])
def test_separate_seed_records_do_not_authorize_a_seed_set_or_unplanned_seed(world, seeds):
    record = persistent_record(world)
    record["protocol"]["profile"] = "descriptive_native_single_seed"
    record["protocol"]["decisions"]["seeds"]["value"] = seeds
    assert check(world, record).gate("G1").status == "RED"


@pytest.mark.parametrize("seed", [1234, 3407])
def test_g6_binds_each_replica_to_a_separate_pinned_catalogue(world, seed):
    from passage_eval.protocol import replicas

    record = persistent_record(world)
    record["protocol"]["profile"] = A.DESCRIPTIVE_SINGLE_SEED_PROFILE
    record["protocol"]["decisions"]["seeds"]["value"] = [seed]
    catalogue_path = Path(__file__).resolve().parents[2] / "docs/benchmark/replica-candidates-256.json"
    catalogue = pinned.read_json(catalogue_path)
    W.write_json(world.evidence / "replicas.json", catalogue)
    record["evidence"]["replica_catalog"] = W.ref(world.evidence, world.evidence / "replicas.json")
    for name in ("seg", "reg"):
        candidate = record["candidates"][name]
        candidate["checkpoint_sha256"] = replicas.replica_identity(catalogue, name, seed)["checkpoint_sha256"]
        bundle = W.write_bundle(world.evidence, name, "orin", candidate["checkpoint_sha256"])
        candidate["targets"]["orin"].update(bundle=bundle, bundle_sha256=bundle["sha256"])
        attest_environment(world, record, name)
    report = check(world, record)
    assert report.gate("G6").status == "GREEN", report.gate("G6").reasons
    catalogue["replicas"][str(seed)]["seg_checkpoint_sha256"] = "0" * 64
    W.write_json(world.evidence / "replicas.json", catalogue)
    record["evidence"]["replica_catalog"] = W.ref(world.evidence, world.evidence / "replicas.json")
    assert "checkpoint_differs_from_final_candidates" in codes(check(world, record), "G6")


@pytest.mark.parametrize(
    "decision, value",
    [
        ("seeds", [1234, 2026, 3407]),
        ("targets", ["host"]),
        ("targets", ["orin", "host"]),
        ("evaluation_cell", {"cost_model": "slope", "cost_weight": 62.4, "task_variant": "obstacles"}),
        (
            "dataset_revision",
            {"repo": "thalesgroup/passage", "commit": "d" * 40, "config": "256x256", "resolution": 256},
        ),
        ("roster_selection", {"rule": A.ROSTER_SELECTION_RULE, "salt": "another-selection/", "n": 1024}),
        (
            "roster_selection",
            {"rule": A.ROSTER_SELECTION_RULE, "salt": "passage-public-test-v1/r0256/roster-v1/", "n": 40000},
        ),
        ("population", {"n_rows": 40000, "ids_sha256": "a" * 64}),
        ("paired_order", {"granularity": "per_batch", "procedures": ["astar", "seg", "reg"], "schedule_seed": 2026}),
        ("paired_order", {"granularity": "per_batch", "procedures": 3, "schedule_seed": 2026}),
        ("paired_order", {"granularity": "per_batch", "procedures": ["seg", 3], "schedule_seed": 2026}),
        ("confirmatory_analysis", W.SEALED_CONFIRMATORY_ANALYSIS),
        ("min_warm_requests", 99),
        ("reference_policy", "released_labels"),
    ],
)
def test_profile_refuses_a_broader_population_procedure_seed_or_claim(world, decision, value):
    record = profile_record(world)
    record["protocol"]["decisions"][decision]["value"] = value
    report = check(world, record)
    assert report.gate("G1").status == "RED"
    assert "gate_check_error" not in codes(report, "G1")


def test_an_unknown_profile_is_not_a_way_around_the_strict_gate(world):
    record = profile_record(world)
    record["protocol"]["profile"] = "native-without-checks"
    assert "record_structure_invalid" in codes(check(world, record), "G1")


def attest_environment(world, record, name):
    candidate = record["candidates"][name]
    entry = candidate["targets"]["orin"]
    environment = {
        "schema": "passage-native-environment/v1",
        "target": "orin",
        "candidate": name,
        "harness": record["harness"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "bundle_sha256": entry["bundle_sha256"],
        "engine_sha256": entry["engine_sha256"],
        "predictor": candidate["predictor"],
        "decoder": candidate["decoder"],
        "decoder_tree_sha256": candidate["decoder_tree_sha256"],
        "final_decoders_sha256": candidate["final_decoders_sha256"],
        "operator": "Test Operator",
        "observed_at": "2026-11-01T00:00:00+00:00",
        "stack": {
            "python": "3.12.3",
            "platform": "Linux aarch64",
            "accelerator": "Orin GPU",
            "driver": "test-driver",
            "runtime": "TensorRT test-version",
            "precision": "FP16",
            "power_mode": "test-mode",
            "determinism": "algorithmically deterministic",
        },
        "isolation": {"level": "process", "security_boundary": False},
    }
    path = world.evidence / f"native-environment-{name}.json"
    W.write_json(path, environment)
    entry["native_environment"] = W.ref(world.evidence, path)
    return path, environment


def test_native_environment_replaces_only_the_container_requirement_and_stays_pinned(world):
    record = profile_record(world)
    for name in ("seg", "reg"):
        attest_environment(world, record, name)
    report = check(world, record)
    assert report.gate("G6").status == "GREEN", report.gate("G6").reasons
    path = world.evidence / "native-environment-seg.json"
    assert ("native-environment-seg.json", pinned.sha256_bytes(path.read_bytes())) in report.file_pins
    path.write_text("{}")
    assert A.check_admission(world.record_path, world.evidence).gate("G6").status == "RED"


@pytest.mark.parametrize(
    "field", ["engine_sha256", "checkpoint_sha256", "harness", "precision", "operator", "security_boundary"]
)
def test_native_attestation_cannot_name_a_different_artifact_or_hide_its_scope(world, field):
    record = profile_record(world)
    for name in ("seg", "reg"):
        path, environment = attest_environment(world, record, name)
        if name == "seg":
            if field == "precision":
                environment["stack"]["precision"] = ""
            elif field == "security_boundary":
                environment["isolation"]["security_boundary"] = True
            else:
                environment[field] = "different" if field != "operator" else ""
            W.write_json(path, environment)
            record["candidates"][name]["targets"]["orin"]["native_environment"] = W.ref(world.evidence, path)
    assert check(world, record).gate("G6").status == "RED"


def test_native_attestation_and_engine_are_required_without_a_container(world):
    record = profile_record(world)
    report = check(world, record)
    assert "native_environment_seg_orin_missing" in codes(report, "G6")
    for name in ("seg", "reg"):
        attest_environment(world, record, name)
    record["candidates"]["seg"]["targets"]["orin"]["engine_sha256"] = None
    assert "engine_digest_missing" in codes(check(world, record), "G6")


def test_native_profile_cannot_attach_an_unexecuted_container_identity(world):
    record = profile_record(world)
    for name in ("seg", "reg"):
        attest_environment(world, record, name)
    record["candidates"]["seg"]["targets"]["orin"]["container_digest"] = "sha256:" + "b" * 64
    assert "native_container_identity_not_applicable" in codes(check(world, record), "G6")


def test_strict_records_still_refuse_native_only_artifacts_and_descriptive_analysis(world):
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"] = DESCRIPTIVE
    assert check(world, record).gate("G1").status == "RED"
    record = copy.deepcopy(world.record)
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"] = W.SEALED_CONFIRMATORY_ANALYSIS
    record["protocol"]["decisions"]["targets"]["value"] = ["orin"]
    record["candidates"] = W.candidates_block(world.evidence, targets=("orin",))
    record["candidates"]["seg"]["targets"]["orin"]["container_digest"] = None
    assert "container_digest_missing" in codes(check(world, record), "G6")


@pytest.mark.parametrize("change", ["input", "full_inventory"])
def test_historical_roster_is_compared_by_ordered_ids_and_inputs_not_by_v2_file_digest(world, monkeypatch, change):
    record = profile_record(world)
    # Only G4 is exercised here on small synthetic rows. G1 separately enforces the real 256/1024 scope.
    source = pinned.read_json(world.evidence / "source-roster.json")
    for name in (
        "dataset_revision",
        "evaluation_cell",
        "roster_selection",
        "population",
        "reference_policy",
        "reference_cohort_id",
        "reference_max_expansions",
    ):
        record["protocol"]["decisions"][name] = world.record["protocol"]["decisions"][name]
    historical = copy.deepcopy(source)
    historical["schema"] = "passage-roster-v1"
    historical["roster_sha256"] = roster_io.compute_roster_sha256(historical)
    monkeypatch.setattr(A, "HISTORICAL_S_ROSTER_SHA256", historical["roster_sha256"])
    path = world.evidence / "historical-roster.json"
    W.write_json(path, historical)
    record["evidence"]["references"]["historical_roster"] = W.ref(world.evidence, path)
    assert check(world, record).gate("G4").status == "GREEN"
    if change == "input":
        source["tasks"][0]["input_identity"]["elevation_hash"] = "e" * 64
        source["tasks"][0]["input_identity_sha256"] = roster_io.identity_digest(source["tasks"][0]["input_identity"])
    else:
        source["selection"]["population"]["ids_sha256"] = "e" * 64
    source["roster_sha256"] = roster_io.compute_roster_sha256(source)
    source_path = world.evidence / "source-roster.json"
    W.write_json(source_path, source)
    record["evidence"]["references"]["source_roster"] = W.ref(world.evidence, source_path)
    assert "historical_roster_identity_differs" in codes(check(world, record), "G4")


@pytest.mark.parametrize("label", ["source", "historical"])
@pytest.mark.parametrize("value", [None, 42, {"n_rows": 40000}, "absent"])
def test_historical_comparison_refuses_a_missing_or_malformed_full_identifier_inventory(
    world, monkeypatch, label, value
):
    record = profile_record(world)
    for name in (
        "dataset_revision",
        "evaluation_cell",
        "roster_selection",
        "population",
        "reference_policy",
        "reference_cohort_id",
        "reference_max_expansions",
    ):
        record["protocol"]["decisions"][name] = world.record["protocol"]["decisions"][name]
    source = pinned.read_json(world.evidence / "source-roster.json")
    historical = copy.deepcopy(source)
    historical["schema"] = "passage-roster-v1"
    changed = source if label == "source" else historical
    if value == "absent":
        changed["selection"].pop("population")
    else:
        changed["selection"]["population"] = value
    for name, payload in (("source", source), ("historical", historical)):
        payload["roster_sha256"] = roster_io.compute_roster_sha256(payload)
        path = world.evidence / f"{name}-roster.json"
        W.write_json(path, payload)
        key = "source_roster" if name == "source" else "historical_roster"
        record["evidence"]["references"][key] = W.ref(world.evidence, path)
    monkeypatch.setattr(A, "HISTORICAL_S_ROSTER_SHA256", historical["roster_sha256"])
    report = check(world, record)
    assert "roster_identifier_inventory_invalid" in codes(report, "G4")
    assert "gate_check_error" not in codes(report, "G4")


def test_unplaced_cap_stops_only_block_inference_in_native_descriptive_mode(tmp_path, monkeypatch):
    world = W.build_green(
        tmp_path / "unplaced",
        unplaced_block=True,
        block_analysis=dict(W.SEALED_BLOCK_ANALYSIS, missing_footprint="unplaced_block"),
    )
    assert "unplaced_cap_exceeded" in codes(A.check_admission(world.record_path, world.evidence), "G4")
    record = profile_record(world)
    for name in (
        "dataset_revision",
        "evaluation_cell",
        "roster_selection",
        "population",
        "reference_policy",
        "reference_cohort_id",
        "reference_max_expansions",
    ):
        record["protocol"]["decisions"][name] = world.record["protocol"]["decisions"][name]
    historical = pinned.read_json(world.evidence / "source-roster.json")
    historical["schema"] = "passage-roster-v1"
    historical["geographic_blocks"].pop("unplaced_block")
    historical["roster_sha256"] = roster_io.compute_roster_sha256(historical)
    monkeypatch.setattr(A, "HISTORICAL_S_ROSTER_SHA256", historical["roster_sha256"])
    path = world.evidence / "historical-roster.json"
    W.write_json(path, historical)
    record["evidence"]["references"]["historical_roster"] = W.ref(world.evidence, path)
    report = check(world, record)
    assert report.gate("G4").status == "GREEN", report.gate("G4").reasons
    assert report.admitted["block_analysis"]["status"] == "halted_unplaced_cap"
    assert report.admitted["block_analysis"]["confirmatory_block_analysis_possible"] is False


def test_preparation_stays_scoped_and_never_grants_execution_while_evidence_is_missing(world):
    from passage_eval.protocol import guards

    spec = importlib.util.spec_from_file_location("descriptive_edge_helpers", HERE / "test_admission.py")
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    record = profile_record(world)
    record["protocol"]["decisions"]["population"] = {"value": None, "status": "required"}
    record["evidence"]["references"] = dict.fromkeys((*A._REFERENCE_KEYS, "historical_roster"))
    record["evidence"]["warmup"] = {"roster": None, "selection": None}
    record["evidence"]["edge"]["orin"] = helpers.write_edge_evidence(world, "orin")
    record["evidence"]["native_chain_rehearsal"] = W.write_rehearsal(world.evidence, record)
    for name in ("seg", "reg"):
        attest_environment(world, record, name)
    report = check(world, record)
    assert report.red_gates() == ["G1", "G4", "G5"]
    W.write_preparation_approval(world)
    grant = W.preparation_grant(world)
    grant.require_roster_selection(rule=A.ROSTER_SELECTION_RULE, salt="passage-public-test-v1/r0256/roster-v1/", n=1024)
    with pytest.raises(guards.TestSplitRefused):
        grant.require_roster_selection(
            rule=A.ROSTER_SELECTION_RULE, salt="passage-public-test-v1/r0256/roster-v1/", n=40000
        )
    with pytest.raises(guards.TestSplitRefused):
        guards.refuse_test_split("test", allow_test=True, admission=grant, reader="run_supervised")


@pytest.mark.parametrize("seeds", [[2026], [3407], [1234, 2026, 3407]])
@pytest.mark.parametrize("profile", [PROFILE, "rule8_native_persistent"])
def test_standalone_supervisor_refuses_native_grant_before_any_roster_or_scene_read(
    tmp_path, monkeypatch, seeds, profile
):
    from passage_eval.cli import BenchSpec
    from passage_eval.protocol.guards import TestSplitRefused
    from passage_eval.supervision import sandbox, supervisor

    # The already-checked report is a synthetic stand-in. The test exercises the execution boundary,
    # not qualification of a real 1024-scene native campaign.
    grant = copy.copy(W.green_grant())
    grant.report = replace(grant.report, admitted={**grant.report.admitted, "profile": profile})

    def forbidden_read(*args, **kwargs):
        pytest.fail("standalone supervisor read roster or data before refusing the native grant")

    monkeypatch.setattr(supervisor.roster_io, "load_roster", forbidden_read)
    monkeypatch.setattr(supervisor, "_load_roster_scenes", forbidden_read)
    spec = BenchSpec("standalone", "classical", "fixture", split="test", seeds=seeds)
    roster = supervisor.RosterInput(str(tmp_path / "not-opened.json"), "a" * 64, str(tmp_path / "not-listed"))
    config = supervisor.SupervisorConfig(worker_python=sys.executable, isolation=sandbox.IsolationConfig())
    with pytest.raises(TestSplitRefused, match="edge campaign"):
        supervisor.run_supervised(
            spec, tmp_path / "not-created", config, roster=roster, allow_test=True, admission=grant
        )
    assert not (tmp_path / "not-created").exists()
