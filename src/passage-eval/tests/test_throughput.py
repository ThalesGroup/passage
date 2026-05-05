# MIT License
#
# Copyright (c) 2026 THALES

import time

import numpy as np

from passage_eval import benchmark_decoder, benchmark_end_to_end, benchmark_forward


class _ToyModel:
    def __init__(self):
        self.n_calls = 0

    def __call__(self, x):
        self.n_calls += 1
        return np.sum(x)


def _slow_decoder(x):
    time.sleep(0.001)
    return float(x)


def test_benchmark_forward_runs_on_cpu_callable():
    model = _ToyModel()
    rec = benchmark_forward(
        model,
        input_shape=(1, 3, 16, 16),
        n_warmup=2,
        n_timed=5,
    )
    assert rec.n_timed == 5
    assert rec.batch_size == 1
    assert rec.error is None
    assert np.isfinite(rec.mean_latency_ms)


def test_benchmark_decoder_emits_percentile_breakdown():
    inputs = np.arange(20, dtype=float)
    rec = benchmark_decoder(_slow_decoder, list(inputs), n_warmup=1)
    assert rec.n_timed == 20
    assert rec.mean_latency_ms > 0.0
    assert rec.p50_ms > 0.0
    assert rec.p95_ms >= rec.p50_ms
    assert rec.p99_ms >= rec.p95_ms


def test_benchmark_end_to_end_chains_forward_and_decoder():
    model = _ToyModel()
    inputs = [np.zeros((1, 3, 8, 8), dtype=np.float32) for _ in range(10)]
    rec = benchmark_end_to_end(model, _slow_decoder, inputs, n_warmup=1)
    assert rec.n_timed == 10
    assert rec.mean_latency_ms > 0.0
    assert model.n_calls >= 10
