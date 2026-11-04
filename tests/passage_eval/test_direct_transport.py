import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest


def test_streams_input_only_fragments_with_pinned_order(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import build_manifest, stream_inputs

    paths = []
    for index in range(3):
        path = tmp_path / f"f{index}.json"
        path.write_text(
            json.dumps(
                {
                    "sample_ids": [f"id{index}"],
                    "terrain": [[[index]]],
                    "obstacles": [[[False]]],
                    "starts": [[0, 0]],
                    "goals": [[0, 0]],
                }
            )
        )
        paths.append(path)
    warm = tmp_path / "warm.json"
    warm.write_text(
        json.dumps(
            {
                "sample_ids": ["warm"],
                "terrain": [[[9]]],
                "obstacles": [[[False]]],
                "starts": [[0, 0]],
                "goals": [[0, 0]],
            }
        )
    )
    manifest = build_manifest("orin", "reg", paths, warm, expected_ids=["id0", "id1", "id2"], warmup_count=1)
    out = io.BytesIO()
    stream_inputs(out, manifest, paths, warm)
    out.seek(0)
    with tarfile.open(fileobj=out, mode="r:") as archive:
        assert archive.getnames() == [
            "manifest.json",
            "warmup.json",
            "inputs/00000.json",
            "inputs/00001.json",
            "inputs/00002.json",
        ]
        control = json.load(archive.extractfile("manifest.json"))
        assert control == manifest and [x["ids"] for x in control["fragments"]] == [["id0"], ["id1"], ["id2"]]
        assert control["warmup"]["sha256"] == hashlib.sha256(warm.read_bytes()).hexdigest()


def test_rejects_labels_duplicate_ids_and_mutation(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import build_manifest, stream_inputs

    inputs = {"sample_ids": ["x"], "terrain": [[[1]]], "obstacles": [[[False]]], "starts": [[0, 0]], "goals": [[0, 0]]}
    path = tmp_path / "input.json"
    path.write_text(json.dumps(inputs))
    warm = tmp_path / "warm.json"
    warm.write_text(json.dumps({**inputs, "sample_ids": ["w"]}))
    with pytest.raises(ValueError, match="order"):
        build_manifest("orin", "reg", [path, path], warm, expected_ids=["x", "x"], warmup_count=1)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({**inputs, "reference_cost": 12}))
    with pytest.raises(ValueError, match="input-only"):
        build_manifest("orin", "reg", [bad], warm, expected_ids=["x"], warmup_count=1)
    manifest = build_manifest("orin", "reg", [path], warm, expected_ids=["x"], warmup_count=1)
    path.write_text(json.dumps({**inputs, "terrain": [[[2]]]}))
    with pytest.raises(ValueError, match="changed"):
        stream_inputs(io.BytesIO(), manifest, [path], warm)


def test_rejects_hidden_object_in_input_arrays(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import build_manifest

    value = {
        "sample_ids": ["x"],
        "terrain": [[[{"reference_cost": 9}]]],
        "obstacles": [[[False]]],
        "starts": [[0, 0]],
        "goals": [[0, 0]],
    }
    source = tmp_path / "input.json"
    source.write_text(json.dumps(value))
    warm = tmp_path / "warm.json"
    warm.write_text(json.dumps({**value, "sample_ids": ["w"], "terrain": [[[1]]]}))
    with pytest.raises(ValueError, match="input-only"):
        build_manifest("orin", "reg", [source], warm, expected_ids=["x"], warmup_count=1)


def test_rejects_duplicate_json_keys(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import build_manifest

    path = tmp_path / "bad.json"
    path.write_text(
        '{"sample_ids":["x"],"sample_ids":["other"],"terrain":[[[1]]],"obstacles":[[[false]]],"starts":[[0,0]],"goals":[[0,0]]}'
    )
    warm = tmp_path / "warm.json"
    warm.write_text(
        json.dumps(
            {"sample_ids": ["w"], "terrain": [[[1]]], "obstacles": [[[False]]], "starts": [[0, 0]], "goals": [[0, 0]]}
        )
    )
    with pytest.raises(ValueError, match="duplicate"):
        build_manifest("orin", "reg", [path], warm, expected_ids=["other"], warmup_count=1)


def test_fragment_frames_roundtrip_and_refuse_truncation_and_tampering(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import write_frame, read_frame

    source = tmp_path / "fragment.json"
    value = {"sample_ids": ["x"], "terrain": [[[1]]], "obstacles": [[[False]]], "starts": [[0, 0]], "goals": [[0, 0]]}
    source.write_text(json.dumps(value))
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    stream = io.BytesIO()
    write_frame(stream, "fragment", 0, source, sha)
    stream.seek(0)
    header, received = read_frame(stream)
    assert header["kind"] == "fragment" and header["sequence"] == 0 and received == raw
    with pytest.raises(ValueError, match="truncated"):
        read_frame(io.BytesIO(stream.getvalue()[:-1]))
    damaged = bytearray(stream.getvalue())
    damaged[-1] ^= 1
    with pytest.raises(ValueError, match="digest"):
        read_frame(io.BytesIO(damaged))


def test_binary_fragment_preserves_types_and_refuses_extra_fields(tmp_path):
    import numpy as np
    from passage_eval.campaign.direct.persistent_transport import write_array_frame, read_array_frame

    path = tmp_path / "fragment.npz"
    arrays = {
        "sample_ids": np.array(["sample"]),
        "terrain": np.zeros((1, 2, 2), dtype=np.float64),
        "obstacles": np.zeros((1, 2, 2), dtype=bool),
        "starts": np.array([[0, 0]], dtype=np.int64),
        "goals": np.array([[1, 1]], dtype=np.int64),
    }
    np.savez(path, **arrays)
    stream = io.BytesIO()
    write_array_frame(stream, "fragment", 0, path, hashlib.sha256(path.read_bytes()).hexdigest())
    stream.seek(0)
    header, unpacked = read_array_frame(stream)
    assert header["ids"] == ["sample"]
    assert all(np.array_equal(arrays[k], unpacked[k]) and arrays[k].dtype == unpacked[k].dtype for k in arrays)
    np.savez(path, **arrays, reference_cost=np.array([5]))
    with pytest.raises(ValueError, match="fields"):
        write_array_frame(io.BytesIO(), "fragment", 0, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_binary_payload_exposes_raw_inputs_without_tensor():
    import numpy as np
    from passage_eval.campaign.direct.persistent_transport import as_raw_payload

    arrays = {
        "sample_ids": np.array(["x"]),
        "terrain": np.zeros((1, 2, 2), dtype=np.float64),
        "obstacles": np.zeros((1, 2, 2), dtype=bool),
        "starts": np.array([[0, 0]], dtype=np.int64),
        "goals": np.array([[1, 1]], dtype=np.int64),
    }
    value = as_raw_payload(arrays)
    assert value["sample_ids"] == ["x"] and type(value["starts"][0][0]) is int
    assert value["terrain"] is arrays["terrain"] and "input_tensor" not in value


def test_next_candidate_requires_session_drain_but_tolerates_system_gpu_processes():
    from passage_eval.campaign.direct.persistent_transport import require_session_drained

    proof = {
        "target": "orin",
        "candidate": "seg",
        "scope": "passage-edge-orin-0123456789abcdef.scope",
        "synchronization_complete": True,
        "runtime_close_complete": True,
        "unit_inactive": True,
        "cgroup_empty": True,
        "session_gpu_handles_observed": True,
        "session_gpu_handles": [],
        "other_gpu_processes": [42],
        "memory_bytes": 123,
        "load_percent": 2,
    }
    require_session_drained(proof, target="orin", candidate="seg", job="job:0123456789abcdef")
    for field in (
        "unit_inactive",
        "cgroup_empty",
        "session_gpu_handles_observed",
        "synchronization_complete",
        "runtime_close_complete",
    ):
        with pytest.raises(RuntimeError, match="drain"):
            require_session_drained({**proof, field: False}, target="orin", candidate="seg", job="job:0123456789abcdef")
    with pytest.raises(RuntimeError, match="drain"):
        require_session_drained(
            {**proof, "session_gpu_handles": [12]}, target="orin", candidate="seg", job="job:0123456789abcdef"
        )


def test_binary_rejects_out_of_range_endpoint(tmp_path):
    import numpy as np
    from passage_eval.campaign.direct.persistent_transport import write_array_frame

    path = tmp_path / "fragment.npz"
    np.savez(
        path,
        sample_ids=np.array(["x"]),
        terrain=np.zeros((1, 2, 2), dtype=np.float64),
        obstacles=np.zeros((1, 2, 2), dtype=bool),
        starts=np.array([[9, 0]], dtype=np.int64),
        goals=np.array([[1, 1]], dtype=np.int64),
    )
    with pytest.raises(ValueError, match="endpoint"):
        write_array_frame(io.BytesIO(), "fragment", 0, path, hashlib.sha256(path.read_bytes()).hexdigest())


def test_streamed_fragments_build_raw_requests_once_per_fragment(tmp_path):
    import numpy as np
    from passage_eval.protocol.contracts import BudgetSpec
    from passage_eval.campaign.direct.persistent_transport import (
        write_array_frame,
        write_end,
        read_warmup,
        blocks_from_stream,
    )

    stream = io.BytesIO()

    def frame(name, kind, seq, values):
        p = tmp_path / name
        n = len(values)
        terrains = np.stack([np.full((2, 2), v, dtype=np.float64) for v in values])
        np.savez(
            p,
            sample_ids=np.array([f"id-{v}" for v in values]),
            terrain=terrains,
            obstacles=np.zeros((n, 2, 2), dtype=bool),
            starts=np.zeros((n, 2), dtype=np.int64),
            goals=np.full((n, 2), 1, dtype=np.int64),
        )
        write_array_frame(stream, kind, seq, p, hashlib.sha256(p.read_bytes()).hexdigest())

    frame("w.npz", "warmup", 0, [0])
    frame("a.npz", "fragment", 0, [20, 21])
    frame("b.npz", "fragment", 1, [22])
    write_end(stream)
    stream.seek(0)
    budget = BudgetSpec(wall_time_ms=100)
    from passage_eval.campaign.direct.raw_identity import raw_digest
    from passage_eval.protocol.contracts import PlanRequest

    warm_index = []
    for value in range(1):
        r = PlanRequest(
            f"id-{value}",
            np.full((2, 2), value, dtype=np.float64),
            np.zeros((2, 2), dtype=bool),
            (0, 0),
            (1, 1),
            "elevation",
            62.4,
            budget=budget,
        )
        warm_index.append({"sample_id": r.sample_id, "raw_input_sha256": raw_digest(r)})
    warm = read_warmup(stream, budget, warm_index)
    assert len(warm) == 1 and all(r.input_tensor is None and not r.metadata for r in warm)
    index = []
    for value in (20, 21, 22):
        r = PlanRequest(
            f"id-{value}",
            np.full((2, 2), value, dtype=np.float64),
            np.zeros((2, 2), dtype=bool),
            (0, 0),
            (1, 1),
            "elevation",
            62.4,
            budget=budget,
        )
        index.append({"sample_id": r.sample_id, "raw_input_sha256": raw_digest(r)})
    blocks = list(blocks_from_stream(stream, index, budget))
    assert [[r.sample_id for r in block] for block in blocks] == [["id-20", "id-21"], ["id-22"]]
    assert all(r.input_tensor is None and not r.metadata for block in blocks for r in block)


def test_index_jsonl_is_pinned_bounded_and_incremental(tmp_path):
    from passage_eval.campaign.direct.persistent_transport import write_task_index, read_task_index

    path = tmp_path / "index.jsonl"
    records = [{"sample_id": f"id-{i}", "raw_input_sha256": ("a" if i == 0 else "b") * 64} for i in range(2)]
    path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    stream = io.BytesIO()
    write_task_index(stream, path, sha, max_tasks=2)
    stream.seek(0)
    assert read_task_index(stream, sha, max_tasks=2) == records
    stream.seek(0)
    with pytest.raises(ValueError, match="count"):
        read_task_index(stream, sha, max_tasks=1)
    altered = bytearray(stream.getvalue())
    altered[-3] ^= 1
    with pytest.raises(ValueError):
        read_task_index(io.BytesIO(altered), sha, max_tasks=2)


def test_binary_stream_drives_one_persistent_direct_session(tmp_path):
    import numpy as np
    from passage_eval.protocol.contracts import BudgetSpec
    from passage_eval.campaign.direct.persistent_transport import (
        write_task_index,
        read_task_index,
        write_array_frame,
        write_end,
        read_warmup,
        blocks_from_stream,
    )
    from passage_eval.campaign.direct.direct_session import run_direct_session
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from test_direct_session import req, task, Decoders, RecordingRuntime

    tasks = [task(req(i)) for i in range(16, 20)]
    warmindex = [task(req(0))]
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in tasks))
    stream = io.BytesIO()
    write_task_index(stream, index, hashlib.sha256(index.read_bytes()).hexdigest())

    def frame(name, kind, seq, nums):
        path = tmp_path / name
        n = len(nums)
        np.savez(
            path,
            sample_ids=np.array([str(x) for x in nums]),
            terrain=np.stack([np.full((3, 3), float(x), dtype=np.float64) for x in nums]),
            obstacles=np.zeros((n, 3, 3), dtype=bool),
            starts=np.zeros((n, 2), dtype=np.int64),
            goals=np.full((n, 2), 2, dtype=np.int64),
        )
        write_array_frame(stream, kind, seq, path, hashlib.sha256(path.read_bytes()).hexdigest())

    frame("warm.npz", "warmup", 0, [0])
    frame("first.npz", "fragment", 0, [16, 17])
    frame("second.npz", "fragment", 1, [18, 19])
    write_end(stream)
    stream.seek(0)
    index_tasks = read_task_index(stream, hashlib.sha256(index.read_bytes()).hexdigest())
    budget = BudgetSpec(wall_time_ms=5)
    warm = read_warmup(stream, budget, warmindex)
    ticks = iter(
        [x for _ in range(16) for x in (0, 1_000_000)] + [0, 2_000_000, 0, 8_000_000, 0, 2_000_000, 0, 2_000_000]
    )
    decoder = Decoders()
    runtime = RecordingRuntime(decoder)
    result = run_direct_session(
        runtime,
        decoder,
        "reg",
        warm,
        blocks_from_stream(stream, index_tasks, budget),
        index_tasks,
        tmp_path / "journal.jsonl",
        clock_ns=lambda: next(ticks),
        checkpoint_size=2,
    )
    rows = [json.loads(line) for line in (tmp_path / "journal.jsonl").read_text().splitlines()]
    assert runtime.calls == 20 and runtime.seen == [0.0] * 16 + [16.0, 17.0, 18.0, 19.0]
    assert [r["status"] for r in rows if r["kind"] == "request"] == ["ok", "timeout", "no_path", "ok"]
    assert result["warmups_completed"] == 16
