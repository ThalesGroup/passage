"""The reusable test-split refusal: declarations only, no data access, an all-GREEN admission grant to lift it."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
from passage_eval.protocol.guards import (
    TestSplitRefused,
    add_test_admission_arguments,
    refuse_test_roster,
    refuse_test_split,
    test_admission_allows,
    test_split_allowed,
)

HERE = Path(__file__).resolve().parent


def _world():
    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


@pytest.mark.parametrize("split", ["test", "Test", " test "])
def test_the_test_split_is_refused_without_allow_test_and_an_admission_grant(split):
    grant = _world().green_grant()
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        refuse_test_split(split)
    for truthy in ("yes", 1, "True"):
        with pytest.raises(TestSplitRefused):
            refuse_test_split(split, allow_test=truthy, admission=grant)
    with pytest.raises(TestSplitRefused, match="admission grant"):
        refuse_test_split(split, allow_test=True)
    with pytest.raises(TestSplitRefused, match="admission grant"):
        refuse_test_split(split, allow_test=True, admission={"all_green": True})  # not a grant object
    assert refuse_test_split(split, allow_test=True, admission=grant) == "test"


def test_other_splits_pass_with_their_canonical_name_and_a_missing_split_is_refused():
    assert refuse_test_split("val") == "validation" and refuse_test_split("calibration") == "calibration"
    for missing in (None, "", "  ", 3):
        with pytest.raises(TestSplitRefused, match="split name is required"):
            refuse_test_split(missing)
    assert issubclass(TestSplitRefused, PermissionError)


def test_a_grant_cannot_be_forged_in_process():
    from passage_eval.protocol.admission import AdmissionGrant

    report = _world().green_grant().report
    with pytest.raises(TypeError, match="only by grant_test_admission"):
        AdmissionGrant(object(), Path("x"), Path("."), report)


def roster(split="validation", paths=("3x3/validation/a.parquet",)):
    return {"dataset": {"split": split, "config": "3x3"}, "source_files": [{"path": path} for path in paths]}


def test_a_roster_is_judged_by_its_split_and_every_source_path():
    grant = _world().green_grant()
    assert refuse_test_roster(roster()) == "validation"
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        refuse_test_roster(roster("test", ("3x3/test/a.parquet",)))
    assert refuse_test_roster(roster("test", ("3x3/test/a.parquet",)), allow_test=True, admission=grant) == "test"
    with pytest.raises(TestSplitRefused, match="outside 3x3/validation/"):
        refuse_test_roster(roster(paths=("3x3/test/a.parquet",)))
    with pytest.raises(TestSplitRefused, match="under a test directory"):
        refuse_test_roster(roster(paths=("3x3/validation/test/a.parquet",)))
    with pytest.raises(TestSplitRefused, match="source files"):
        refuse_test_roster(roster(paths=()))
    with pytest.raises(TestSplitRefused, match="dataset block"):
        refuse_test_roster({})


def test_an_admission_admits_its_sealed_admissible_roster_and_its_derived_children_only():
    from passage_eval.io import roster as roster_io

    world = _world()
    grant = world.green_grant()
    admissible = world.green_world().world.admissible
    source = world.green_world().world.source
    assert refuse_test_roster(admissible, allow_test=True, admission=grant, require_admitted_roster=True) == "test"
    child = roster_io.derive_roster(admissible, [admissible["tasks"][1]["sample_id"]], rule="one task")
    assert refuse_test_roster(child, allow_test=True, admission=grant, require_admitted_roster=True) == "test"
    with pytest.raises(TestSplitRefused, match="not the admission's sealed admissible roster"):
        refuse_test_roster(source, allow_test=True, admission=grant, require_admitted_roster=True)


def test_test_admission_allows_reports_the_six_gates_without_raising(tmp_path):
    assert test_admission_allows(None) is False
    record = tmp_path / "admission.json"
    record.write_text('{"schema": "passage-test-admission/v1", "approved": true}')
    assert test_admission_allows(record) is False  # RED: a malformed record never raises
    world = _world().green_world()
    assert test_admission_allows(world.record_path, world.evidence) is True


def test_a_command_needs_both_the_flag_and_an_all_green_record(tmp_path):
    import argparse

    record = tmp_path / "admission.json"
    record.write_text('{"schema": "passage-test-admission/v1"}')
    assert test_split_allowed(False, None) is None
    with pytest.raises(TestSplitRefused, match="only together with --allow-test-split"):
        test_split_allowed(False, record)
    with pytest.raises(TestSplitRefused, match="only together with --allow-test-split"):
        test_split_allowed(False, None, evidence_dir=tmp_path)
    with pytest.raises(TestSplitRefused, match="needs a verified test admission record"):
        test_split_allowed(True, None)
    with pytest.raises(TestSplitRefused, match="not all GREEN.*G1: record_structure_invalid"):
        test_split_allowed(True, record)
    with pytest.raises(TestSplitRefused, match="boolean"):
        test_split_allowed("yes", record)
    parser = argparse.ArgumentParser()
    add_test_admission_arguments(parser, what="this command", extra_flags=(("--allow-other", "another source"),))
    args = parser.parse_args([])
    assert (args.allow_test_split, args.allow_other, args.test_admission_record) == (False, False, None)
    world = _world().green_world()
    args = parser.parse_args(
        ["--allow-test-split", "--test-admission-record", str(world.record_path)]
        + ["--test-admission-evidence-dir", str(world.evidence)]
    )
    grant = test_split_allowed(
        args.allow_test_split, args.test_admission_record, evidence_dir=args.test_admission_evidence_dir
    )
    assert refuse_test_split("test", allow_test=True, admission=grant) == "test"


def test_a_grant_is_refused_once_its_record_or_evidence_changes(tmp_path):
    from passage_eval.protocol.admission import grant_test_admission

    world = _world().copy_green(tmp_path / "copy")
    grant = grant_test_admission(world.record_path, world.evidence)
    assert refuse_test_split("test", allow_test=True, admission=grant) == "test"
    inventory = world.evidence / "reference-admission-inventory.json"
    original = inventory.read_bytes()
    inventory.write_bytes(original + b"\n")
    with pytest.raises(TestSplitRefused, match="SHA-256 mismatch"):
        refuse_test_split("test", allow_test=True, admission=grant)
    inventory.write_bytes(original)
    assert refuse_test_split("test", allow_test=True, admission=grant) == "test"
    record = json.loads(world.record_path.read_text())
    record["record_id"] = "edited after the check"
    world.record_path.write_text(json.dumps(record))
    with pytest.raises(TestSplitRefused, match="record changed after it was checked"):
        refuse_test_split("test", allow_test=True, admission=grant)
