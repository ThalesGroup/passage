"""Check the installed frozen decoder and its worker adapters without data or GPU.

This build-time check does not establish hardware execution, numerical parity,
or the integrity and isolation of a derived candidate image.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

FROZEN_SHA256 = '2a25ad31fda6ac75f369e8546d9fa0239191fb04b1d7204a73045aa880cbafc4'
SCOPE = 'synthetic CPU only; hardware execution and submission isolation not verified'


def verify_source(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != FROZEN_SHA256:
        raise ValueError('frozen decoder SHA-256 mismatch')
    return digest


def run() -> dict:
    spec = importlib.util.find_spec('passage_eval.baselines.final_decoders')
    if spec is None or spec.origin is None:
        raise ValueError('installed frozen decoder unavailable')
    digest = verify_source(Path(spec.origin))

    import numpy as np
    from passage_eval.baselines.final_decoders import (
        REG_KAPPA,
        REG_TRAIN_SCALE,
        SEG_HEURISTIC_WEIGHT,
        SEG_STEP_WEIGHT,
    )
    from passage_eval.baselines.final_operator_adapters import RegFinalDecoder, SegFinalDecoder
    from passage_eval.protocol.contracts import BudgetSpec

    if (SEG_STEP_WEIGHT, SEG_HEURISTIC_WEIGHT, REG_KAPPA, REG_TRAIN_SCALE) != (0.25, 1.5, 8.0, 897803.25):
        raise ValueError('frozen decoder parameter mismatch')
    obstacles = np.zeros((8, 8), dtype=bool)
    start, goal = (0, 0), (6, 6)
    field = np.zeros((8, 8), dtype=np.float32)
    results = (
        SegFinalDecoder().decode(field, start, goal, obstacles, BudgetSpec()),
        RegFinalDecoder().decode(field, start, goal, obstacles, BudgetSpec()),
    )
    for result in results:
        route = np.asarray(result.path)
        if (result.status != 'ok' or route.ndim != 2 or route.shape[1] != 2
                or tuple(route[0]) != start or tuple(route[-1]) != goal):
            raise ValueError('synthetic decoder or adapter smoke failed')
        steps = np.abs(np.diff(route, axis=0))
        if (np.any(route < 0) or np.any(route >= obstacles.shape[0])
                or np.any(steps.max(axis=1) != 1)
                or np.any(obstacles[route[:, 0], route[:, 1]])):
            raise ValueError('synthetic decoder returned an invalid route')
    return {'schema': 'passage-frozen-decoder-smoke/v1', 'status': 'ok',
            'decoder_sha256': digest, 'scope': SCOPE}


if __name__ == '__main__':
    print(json.dumps(run(), sort_keys=True, allow_nan=False))
