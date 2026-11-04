"""Strict native qualification contracts on synthetic evidence; no released split or target is opened."""

import importlib.util
import sys
from pathlib import Path

import pytest
from passage_eval.io import pinned

NAME = "passage_eval_tests_admission_descriptive"
if NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(NAME, Path(__file__).with_name("test_admission_descriptive.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[NAME] = module
    spec.loader.exec_module(module)
D = sys.modules[NAME]


@pytest.fixture
def world(tmp_path):
    return D.W.copy_green(tmp_path / "green")


def record_for(world):
    record = D.persistent_record(world)
    record["protocol"]["profile"] = "rule8_native_persistent"
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"] = {
        **D.W.SEALED_CONFIRMATORY_ANALYSIS,"target":"orin","draws":10000}
    catalogue_path=Path(__file__).resolve().parents[2]/"docs/benchmark/replica-candidates-256.json"
    D.W.write_json(world.evidence/"replicas.json",pinned.read_json(catalogue_path))
    record["evidence"]["replica_catalog"]=D.W.ref(world.evidence,world.evidence/"replicas.json")
    for candidate in record["candidates"].values():
        entry = candidate["targets"]["orin"]
        entry.pop("native_environment")
        entry["container_digest"] = "sha256:" + "a" * 64
        entry["container_reconstruction"] = None
    return record


def test_rule8_native_accepts_direct_timing_with_strict_three_seed_analysis(world):
    record = record_for(world)
    report = D.check(world,record)
    assert report.gate("G1").status == "GREEN",report.gate("G1").reasons
    assert report.gate("G6").status == "RED"


@pytest.mark.parametrize("analysis", [D.DESCRIPTIVE,{**D.W.SEALED_CONFIRMATORY_ANALYSIS,"seeds":[2026,1234]},
    {**D.W.SEALED_CONFIRMATORY_ANALYSIS,"draws":200},{**D.W.SEALED_CONFIRMATORY_ANALYSIS,"draws":10000,"alpha":0.1}])
def test_rule8_native_refuses_missing_three_seed_analysis(world,analysis):
    record=record_for(world)
    record["protocol"]["decisions"]["confirmatory_analysis"]["value"]=analysis
    assert D.check(world,record).gate("G1").status == "RED"


@pytest.mark.parametrize("decision,value", [("population",{"n_rows":40000,"ids_sha256":"b"*64}),
    ("seeds",[1234,2026,3407]),("targets",["host"]),("max_restarts",1)])
def test_rule8_native_retains_authorised_s_and_sequential_process_scope(world,decision,value):
    record=record_for(world)
    record["protocol"]["decisions"][decision]["value"]=value
    assert D.check(world,record).gate("G1").status == "RED"


def test_rule8_native_still_requires_actual_container_identity_and_reconstruction(world):
    record=record_for(world)
    for candidate in record["candidates"].values():
        candidate["targets"]["orin"]["container_digest"]=None
    report=D.check(world,record)
    assert "container_digest_missing" in D.codes(report,"G6")
    assert "container_reconstruction_seg_orin_missing" in D.codes(report,"G6")


def test_rule8_native_pins_the_complete_replica_family_even_before_the_2026_pass(world):
    record=record_for(world)
    path=world.evidence/"replicas.json"
    catalogue=pinned.read_json(path)
    catalogue["replicas"]["3407"]={"status":"pending","seg_checkpoint_sha256":None,"reg_checkpoint_sha256":None}
    D.W.write_json(path,catalogue)
    record["evidence"]["replica_catalog"]=D.W.ref(world.evidence,path)
    assert "replica_catalog_incomplete" in D.codes(D.check(world,record),"G6")


def pin_reconstruction(world,record,name):
    for filename,contents in (("Dockerfile","FROM base@sha256:"+"b"*64+"\n"),("runtime.lock","runtime==1\n")):
        (world.evidence/filename).write_text(contents)
    entry=record["candidates"][name]["targets"]["orin"]
    value={
        "schema":"passage-container-reconstruction/v1","target":"orin","container_digest":entry["container_digest"],
        "harness":record["harness"],"source_repository":"https://gitlab.com/passage/unit-world",
        "build_command":["docker","build","--file","Dockerfile","."],
        "recipe":D.W.ref(world.evidence,world.evidence/"Dockerfile"),
        "dependencies":[D.W.ref(world.evidence,world.evidence/"runtime.lock")]}
    path=world.evidence/f"reconstruction-{name}.json"
    D.W.write_json(path,value)
    entry["container_reconstruction"]=D.W.ref(world.evidence,path)
    return path,value


def test_rule8_native_g6_pins_real_reconstruction_files_and_refuses_substitution(world):
    # Synthetic pinned files exercise the contract; this is not an image-build or publication test.
    record=record_for(world)
    for name in ("seg","reg"):
        pin_reconstruction(world,record,name)
    report=D.check(world,record)
    assert report.gate("G6").status == "GREEN",report.gate("G6").reasons
    (world.evidence/"runtime.lock").write_text("substituted")
    report=D.check(world,record)
    assert "container_reconstruction_invalid" in D.codes(report,"G6")


@pytest.mark.parametrize("seed", [1234,3407])
def test_rule8_native_binds_the_single_seed_to_the_separate_catalogue(world,seed):
    from passage_eval.protocol import replicas

    record=record_for(world)
    record["protocol"]["decisions"]["seeds"]["value"]=[seed]
    path=Path(__file__).resolve().parents[2]/"docs/benchmark/replica-candidates-256.json"
    catalogue=pinned.read_json(path)
    D.W.write_json(world.evidence/"replicas.json",catalogue)
    record["evidence"]["replica_catalog"]=D.W.ref(world.evidence,world.evidence/"replicas.json")
    for name in ("seg","reg"):
        candidate=record["candidates"][name]
        candidate["checkpoint_sha256"]=replicas.replica_identity(catalogue,name,seed)["checkpoint_sha256"]
        bundle=D.W.write_bundle(world.evidence,name,"orin",candidate["checkpoint_sha256"])
        candidate["targets"]["orin"].update(bundle=bundle,bundle_sha256=bundle["sha256"])
        pin_reconstruction(world,record,name)
    report=D.check(world,record)
    assert report.gate("G1").status == "GREEN",report.gate("G1").reasons
    assert report.gate("G6").status == "GREEN",report.gate("G6").reasons


@pytest.mark.parametrize("substitute", [None,"receipt","manifest","aggregate"])
def test_rule8_native_uses_real_native_validation_without_supervised_ipc(world,monkeypatch,substitute):
    record=record_for(world)
    D.native_validation(world,record,monkeypatch)
    report=D.check(world,record)
    assert "rehearsal_receipt_missing" not in D.codes(report,"G2")
    assert "native_container_execution_differs" in D.codes(report,"G2")
    path=world.evidence/record["evidence"]["edge"]["orin"]["path"]
    receipt=pinned.read_json(path)
    for name,entry in receipt["procedures"].items():
        digest=record["candidates"][name]["targets"]["orin"]["container_digest"]
        entry["native_execution"]["container_digest"]=digest
        manifest_path=world.evidence/entry["run_manifest"]["path"]
        manifest=pinned.read_json(manifest_path)
        manifest["native_execution"]["container_digest"]=digest
        D.W.write_json(manifest_path,manifest)
        entry["run_manifest"]=D.W.ref(world.evidence,manifest_path)
        aggregate_path=world.evidence/entry["aggregate"]["path"]
        aggregate=pinned.read_json(aggregate_path)
        per_seed=aggregate["per_seed"]["2026"]
        per_seed["native_execution"]["container_digest"]=digest
        per_seed["run_manifest_sha256"]=entry["run_manifest"]["sha256"]
        D.W.write_json(aggregate_path,aggregate)
        entry["aggregate"]=D.W.ref(world.evidence,aggregate_path)
    D.W.write_json(path,receipt)
    record["evidence"]["edge"]["orin"]=D.W.ref(world.evidence,path)
    report=D.check(world,record)
    assert report.gate("G2").status == "GREEN",report.gate("G2").reasons
    if substitute is None:
        return
    entry=receipt["procedures"]["seg"]
    manifest_path=world.evidence/entry["run_manifest"]["path"]
    manifest=pinned.read_json(manifest_path)
    aggregate_path=world.evidence/entry["aggregate"]["path"]
    aggregate=pinned.read_json(aggregate_path)
    bad="sha256:"+"b"*64
    if substitute == "receipt":
        entry["native_execution"]["container_digest"]=bad
    elif substitute == "manifest":
        manifest["native_execution"]["container_digest"]=bad
        D.W.write_json(manifest_path,manifest)
        entry["run_manifest"]=D.W.ref(world.evidence,manifest_path)
    else:
        aggregate["per_seed"]["2026"]["native_execution"]["container_digest"]=bad
    # Re-pin ordinary wrappers so this probes the identity link, rather than a stale-file digest.
    aggregate["per_seed"]["2026"]["run_manifest_sha256"]=entry["run_manifest"]["sha256"]
    D.W.write_json(aggregate_path,aggregate)
    entry["aggregate"]=D.W.ref(world.evidence,aggregate_path)
    D.W.write_json(path,receipt)
    record["evidence"]["edge"]["orin"]=D.W.ref(world.evidence,path)
    report=D.check(world,record)
    assert "native_measurement_binding_invalid" in D.codes(report,"G2")
