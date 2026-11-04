"""Campaign partition: derived child rosters that keep the parent's ranks and blocks, without any scene read.

Ported from the external preparation tooling (``roster-partition``) and hardened: children are
``io.roster.derive_roster`` children, a null parent block is refused, the test split needs an all-GREEN admission
that admits the parent. Synthetic validation fixtures (``campaign_world.py``) only.
"""

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from passage_eval.campaign import partition as P
from passage_eval.io import pinned
from passage_eval.io import roster as R
from passage_eval.protocol.guards import TestSplitRefused

HERE = Path(__file__).resolve().parent


def _world():
    name = "passage_eval_tests_campaign_world"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "campaign_world.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


W = _world()


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    return W.reference_world(tmp_path_factory.mktemp("partition-world"), split="validation")


def pinned_parent(tmp_path, roster):
    path = tmp_path / "parent.json"
    path.write_text(json.dumps(roster, indent=1))
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), roster["roster_sha256"]


def test_children_cover_the_parent_keep_its_ranks_and_blocks_and_read_only_the_parent(reference, tmp_path, monkeypatch):
    parent = reference.admissible
    source, file_sha, roster_sha = pinned_parent(tmp_path, parent)
    opened = []
    original = pinned.os.open

    def checked_open(path, flags, *args, **kwargs):
        opened.append(Path(path))
        if Path(path) != source:
            raise AssertionError("a file other than the parent roster was opened")
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(pinned.os, "open", checked_open)
    manifest = P.partition(source, file_sha, roster_sha, tmp_path / "out", 2)
    monkeypatch.undo()
    assert opened == [source]
    n = len(parent["tasks"])
    assert manifest["n_expected"] == n and manifest["verified"] is False and manifest["admission_record_sha256"] is None
    assert [(b["start"], b["stop"]) for b in manifest["batches"]] == [(s, min(s + 2, n)) for s in range(0, n, 2)]
    parent_blocks = R.roster_block_table(parent)["blocks"]
    for batch in manifest["batches"]:
        child = P.load_child(tmp_path / "out", batch)
        assert child["selection"]["derivation"]["parent_roster_sha256"] == roster_sha
        assert child["selection"]["derivation"]["kind"] == "campaign_partition"
        R.verify_child_blocks(parent, child)
        for local, (task, mapping) in enumerate(zip(child["tasks"], batch["parent_mapping"])):
            origin = parent["tasks"][batch["start"] + local]
            assert task["sample_id"] == origin["sample_id"] == mapping["sample_id"]
            assert task["parent_rank"] == mapping["parent_index"] == batch["start"] + local
            assert mapping["parent_geographic_block"] == parent_blocks[task["sample_id"]] is not None
            assert R.roster_block_table(child)["blocks"][task["sample_id"]] == mapping["parent_geographic_block"]
    digest = (tmp_path / "out" / "partition-manifest.json.sha256").read_text().strip()
    assert P.verify_partition(source, file_sha, roster_sha, tmp_path / "out", digest) == manifest


def test_a_changed_parent_or_a_wrong_pin_creates_no_output(reference, tmp_path):
    source, file_sha, roster_sha = pinned_parent(tmp_path, reference.admissible)
    for raw, semantic in (("0" * 64, roster_sha), (file_sha, "0" * 64)):
        with pytest.raises(ValueError):
            P.partition(source, raw, semantic, tmp_path / "out", 2)
        assert not (tmp_path / "out").exists()


def test_a_parent_with_a_task_without_block_is_refused(tmp_path):
    TR = W.roster_fixtures()
    dataset = TR.write_dataset(tmp_path / "dataset", split="validation")
    roster, _ = TR.build(dataset, split="validation", n_expected=12)  # row 11 has no footprint
    source, file_sha, roster_sha = pinned_parent(tmp_path, roster)
    with pytest.raises(ValueError, match="without a geographic block"):
        P.partition(source, file_sha, roster_sha, tmp_path / "out", 4)
    assert not (tmp_path / "out").exists()


def test_the_test_split_is_refused_before_any_output_unless_admitted(tmp_path):
    green = W.green_world()
    admissible = green.world.admissible
    source, file_sha, roster_sha = pinned_parent(tmp_path, admissible)
    with pytest.raises(TestSplitRefused, match="refuses the test split"):
        P.partition(source, file_sha, roster_sha, tmp_path / "out", 4)
    with pytest.raises(TestSplitRefused, match="admission grant"):
        P.partition(source, file_sha, roster_sha, tmp_path / "out", 4, allow_test=True)
    assert not (tmp_path / "out").exists()
    grant = W.green_grant()
    manifest = P.partition(source, file_sha, roster_sha, tmp_path / "out", 4, allow_test=True, admission=grant)
    assert manifest["split"] == "test" and manifest["admission_record_sha256"] == grant.record_sha256
    digest = (tmp_path / "out" / "partition-manifest.json.sha256").read_text().strip()
    P.verify_partition(source, file_sha, roster_sha, tmp_path / "out", digest, allow_test=True, admission=grant)
    from passage_eval.protocol.admission import grant_test_admission

    copy_world = W.copy_green(tmp_path / "other-record")
    W.save_record(copy_world, dict(copy.deepcopy(copy_world.record), record_id="another sealing record"))
    other_grant = grant_test_admission(copy_world.record_path, copy_world.evidence)
    with pytest.raises(ValueError, match="another test admission record"):
        P.verify_partition(
            source, file_sha, roster_sha, tmp_path / "out", digest, allow_test=True, admission=other_grant
        )
    (tmp_path / "source").mkdir()
    other, other_file, other_sha = pinned_parent(tmp_path / "source", green.world.source)
    with pytest.raises(TestSplitRefused, match="not the admission's sealed admissible roster"):
        P.partition(other, other_file, other_sha, tmp_path / "out-source", 4, allow_test=True, admission=grant)


def test_no_overwrite_and_no_duplicate_json_key_or_symlinked_parent(reference, tmp_path):
    source, file_sha, roster_sha = pinned_parent(tmp_path, reference.admissible)
    out = tmp_path / "out"
    P.partition(source, file_sha, roster_sha, out, 2)
    before = (out / "partition-manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        P.partition(source, file_sha, roster_sha, out, 2)
    assert before == (out / "partition-manifest.json").read_bytes()
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(pinned.PinnedFileError):
        P.read_parent(alias, file_sha, roster_sha)
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_bytes(b'{"schema":"one","schema":"two"}')
    with pytest.raises(pinned.PinnedFileError, match="duplicate JSON key"):
        P.read_parent(duplicate, hashlib.sha256(duplicate.read_bytes()).hexdigest(), roster_sha)


@pytest.mark.parametrize("size", [True, 0, -1, 1025, 1.5])
def test_an_invalid_batch_size_is_refused_before_any_output(reference, tmp_path, size):
    source, file_sha, roster_sha = pinned_parent(tmp_path, reference.admissible)
    with pytest.raises(ValueError, match="batch_size"):
        P.partition(source, file_sha, roster_sha, tmp_path / "out", size)
    assert not (tmp_path / "out").exists()


def test_the_verifier_detects_child_tampering_truncation_and_a_falsified_parent_block(reference, tmp_path):
    source, file_sha, roster_sha = pinned_parent(tmp_path, reference.admissible)
    out = tmp_path / "out"
    manifest = P.partition(source, file_sha, roster_sha, out, 2)
    manifest_file = out / "partition-manifest.json"
    digest = hashlib.sha256(manifest_file.read_bytes()).hexdigest()
    child = out / manifest["batches"][0]["file"]
    original = child.read_bytes()
    child.write_bytes(original + b" ")
    with pytest.raises(pinned.PinnedFileError, match="SHA-256 mismatch"):
        P.verify_partition(source, file_sha, roster_sha, out, digest)
    child.write_bytes(original)
    for change, message in (
        (lambda m: m["batches"].pop(), "full parent roster"),
        (lambda m: m["batches"][0]["parent_mapping"][0].update(parent_geographic_block="geo5:0:0"), "parent mapping"),
        (lambda m: m.update(verified=True), "identity or policy"),
    ):
        altered = copy.deepcopy(manifest)
        change(altered)
        manifest_file.write_bytes(R.canonical_json(altered))
        with pytest.raises(ValueError, match=message):
            P.verify_partition(
                source, file_sha, roster_sha, out, hashlib.sha256(manifest_file.read_bytes()).hexdigest()
            )


def test_lots_of_up_to_1024_tasks_are_accepted_and_their_memory_is_estimated(reference, tmp_path):
    from passage_eval.campaign import partition as partition_io

    assert partition_io.MAX_BATCH == 1024 and partition_io._batch_size(1024) == 1024
    with pytest.raises(ValueError, match="batch_size"):
        partition_io._batch_size(1025)
    source, file_sha, roster_sha = pinned_parent(tmp_path, reference.admissible)
    manifest = partition_io.partition(source, file_sha, roster_sha, tmp_path / "lots", 1024)
    assert len(manifest["batches"]) == 1 and manifest["batch_size"] == 1024
    memory = partition_io.lot_memory_bytes(256, 1024)
    assert memory["per_task_bytes"] == 21 * 256 * 256
    assert memory["supervisor_peak_bytes"] == 2 * 21 * 256 * 256 * 1024  # about 2.8 GB for a lot of 1,024 at 256
    with pytest.raises(ValueError, match="positive integers"):
        partition_io.lot_memory_bytes(256, 0)


def test_the_per_task_memory_estimate_covers_every_array_a_request_copies(metre_scene):
    from passage_eval.campaign import partition as partition_io
    from passage_eval.protocol.contracts import PlanRequest

    request = PlanRequest.from_scene(metre_scene)
    n = request.elevation.shape[0]
    held = (
        request.elevation.nbytes
        + request.obstacles.nbytes
        + (request.input_tensor.nbytes if request.input_tensor is not None else 0)
    )
    assert held <= partition_io.lot_memory_bytes(n, 1)["per_task_bytes"]
