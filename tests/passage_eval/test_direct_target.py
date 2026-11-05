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

from pathlib import Path
import pytest
from passage_eval.campaign.direct.direct_target import validate_config


def test_config_rejects_test_scope():
    cfg = {
        "schema": "passage-direct-target-validation-prototype/v1",
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
    with pytest.raises(ValueError, match="validation"):
        validate_config(cfg)


def test_persistent_binary_stream_runs_once(tmp_path, monkeypatch):
    import io
    import sys
    import json
    import hashlib
    import numpy as np
    from passage_eval.campaign.direct.persistent_transport import write_task_index, write_array_frame, write_end

    sys.path.insert(0, str(Path(__file__).parent))
    from test_direct_session import req, task, Decoders, RecordingRuntime
    from passage_eval.campaign.direct.direct_target import execute_stream

    warm = req(0)
    measured = [req(16), req(17), req(18)]

    def save(path, requests):
        np.savez(
            path,
            sample_ids=np.array([r.sample_id for r in requests]),
            terrain=np.stack([r.elevation for r in requests]),
            obstacles=np.stack([r.obstacles for r in requests]),
            starts=np.array([r.start for r in requests], dtype=np.int64),
            goals=np.array([r.goal for r in requests], dtype=np.int64),
        )

    warm_file = tmp_path / "warm.npz"
    a = tmp_path / "a.npz"
    b = tmp_path / "b.npz"
    save(warm_file, [warm])
    save(a, measured[:1])
    save(b, measured[1:])
    index = tmp_path / "index.jsonl"
    index.write_text("".join(json.dumps(task(r), sort_keys=True) + "\n" for r in measured))

    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()

    wire = io.BytesIO()
    write_task_index(wire, index, sha(index))
    write_array_frame(wire, "warmup", 0, warm_file, sha(warm_file))
    write_array_frame(wire, "fragment", 0, a, sha(a))
    write_array_frame(wire, "fragment", 1, b, sha(b))
    write_end(wire)
    wire.seek(0)
    cfg = {
        "schema": "passage-direct-target-validation-prototype/v1",
        "dataset_split": "validation",
        "target": "orin",
        "procedure": "reg",
        "seed": 2026,
        "artifact_paths": {},
        "artifact_pins": {},
        "compiled_manifest": "/x",
        "compiled_sha256": "a" * 64,
        "exploratory_unadmitted_precision": True,
        "budget": {"wall_time_ms": 30000},
        "warmup_index": [task(warm)],
        "index_sha256": sha(index),
        "checkpoint_size": 2,
        "grn_evidence": None,
    }

    class Runtime(RecordingRuntime):
        def close(self):
            self.closed = True

    dec = Decoders()
    created = []

    def factory():
        runtime = Runtime(dec)
        created.append(runtime)
        return runtime

    result = execute_stream(wire, cfg, factory, dec, tmp_path / "out")
    assert result["n_completed"] == 3 and result["warmups_completed"] == 16
    assert len(created) == 1 and created[0].calls == 19 and created[0].closed
    rows = [json.loads(line) for line in (tmp_path / "out" / "journal.jsonl").read_text().splitlines()]
    assert len([r for r in rows if r["kind"] == "warmup_complete"]) == 1
