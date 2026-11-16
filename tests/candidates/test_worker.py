import json
import subprocess
import sys
import os
import numpy as np
from .test_package import package_fixture


def test_worker_persists_runtime_and_observed_memory_with_real_cpu_planner(tmp_path):
    from passage_eval.protocol.candidate_package import file_sha256

    path, doc = package_fixture(tmp_path)
    (tmp_path / "planner.py").write_text("""from passage_eval.protocol.contracts import PlanResult
class Planner:
    def __init__(self,device='cpu'): pass
    def plan(self,request): return PlanResult([request.start, request.goal])
""")
    doc["artifacts"][0]["sha256"] = file_sha256(tmp_path / "planner.py")
    path.write_text(json.dumps(doc))
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        sample_ids=np.asarray(["a"]),
        terrain=np.zeros((1, 256, 256), np.float64),
        obstacles=np.zeros((1, 256, 256), bool),
        starts=np.asarray([[0, 0]], np.int64),
        goals=np.asarray([[0, 1]], np.int64),
    )
    output = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "passage_eval.campaign.candidate.worker",
            "--manifest",
            str(path),
            "--inputs",
            str(inputs),
            "--output",
            str(output),
        ],
        check=True,
        env={
            "PATH": os.defpath,
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
    )
    run = json.loads((output / "run.json").read_text())
    assert run["resources"]["max_rss_bytes"] > 0
    assert run["runtime_versions"]["numpy"] == np.__version__
    assert run["n_completed"] == 1 and run["synchronization_complete"]


def test_worker_always_attempts_close_even_when_synchronization_raises(tmp_path):
    from passage_eval.protocol.candidate_package import file_sha256

    path, doc = package_fixture(tmp_path)
    marker = tmp_path / "closed"
    (tmp_path / "planner.py").write_text(f"""from pathlib import Path
class Planner:
    def __init__(self, device='cpu'): pass
    def synchronize(self): raise RuntimeError('synchronization failed')
    def close(self): Path({str(marker)!r}).write_text('closed')
""")
    doc["artifacts"][0]["sha256"] = file_sha256(tmp_path / "planner.py")
    path.write_text(json.dumps(doc))
    inputs = tmp_path / "inputs.npz"
    np.savez(
        inputs,
        sample_ids=np.asarray(["a"]),
        terrain=np.zeros((1, 256, 256), np.float64),
        obstacles=np.zeros((1, 256, 256), bool),
        starts=np.asarray([[0, 0]], np.int64),
        goals=np.asarray([[0, 1]], np.int64),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "passage_eval.campaign.candidate.worker",
            "--manifest",
            str(path),
            "--inputs",
            str(inputs),
            "--output",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert marker.read_text() == "closed"
    assert "synchronization failed" in result.stderr
    assert not (tmp_path / "out/run.json").exists()
