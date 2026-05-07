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
