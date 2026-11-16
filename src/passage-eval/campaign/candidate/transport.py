"""Strict input-only NPZ transport. Targets stay on the scoring host."""

from __future__ import annotations
from hashlib import sha256
from pathlib import Path
import zipfile
import numpy as np

from passage_eval.protocol.contracts import PlanRequest, BudgetSpec
from passage_eval.protocol.candidate_package import CandidatePackage

KEYS = {"sample_ids", "terrain", "obstacles", "starts", "goals"}


def input_digest(request: PlanRequest) -> str:
    digest = sha256(request.sample_id.encode())
    for value, dtype in [
        (request.elevation, "<f8"),
        (request.obstacles, "u1"),
        (request.start, "<i8"),
        (request.goal, "<i8"),
    ]:
        arr = np.ascontiguousarray(value, dtype=dtype)
        digest.update(str(arr.shape).encode())
        digest.update(arr.tobytes())
    return digest.hexdigest()


def load_input_requests(path: str | Path, package: CandidatePackage):
    with zipfile.ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist()) > 2 * 1024**3:
            raise ValueError("input archive exceeds 2 GiB")
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != KEYS:
            raise ValueError("input archive must contain only permitted arrays")
        ids, terrain, obstacles, starts, goals = [
            data[k] for k in ("sample_ids", "terrain", "obstacles", "starts", "goals")
        ]
    shape = tuple(map(int, package.document["dataset"]["config"].split("x")))
    n = len(ids)
    if (
        ids.ndim != 1
        or ids.dtype.kind != "U"
        or not 1 <= n <= 40000
        or len(set(ids.tolist())) != n
        or terrain.shape != (n, *shape)
        or terrain.dtype != np.float64
        or not np.isfinite(terrain).all()
        or obstacles.shape != terrain.shape
        or obstacles.dtype != np.bool_
        or starts.shape != (n, 2)
        or goals.shape != (n, 2)
        or starts.dtype != np.int64
        or goals.dtype != np.int64
    ):
        raise ValueError("input arrays differ from the declared scene contract")
    objective = package.document["dataset"]
    budget = BudgetSpec(**package.document["budget"])
    for sid, z, o, s, g in zip(ids, terrain, obstacles, starts, goals):
        yield PlanRequest(
            str(sid),
            z,
            o,
            tuple(map(int, s)),
            tuple(map(int, g)),
            objective["cost_model"],
            objective["cost_weight"],
            budget=budget,
        )
