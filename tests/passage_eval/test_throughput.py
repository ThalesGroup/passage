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

import sys
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

from passage_eval import benchmark_decoder, benchmark_end_to_end, benchmark_forward
from passage_eval.deployment import throughput


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


@pytest.mark.parametrize("device", [None, "cpu", "cuda:1"])
def test_torch_benchmarks_only_use_cuda_for_an_explicit_cuda_device(monkeypatch, device):
    torch = MagicMock()
    torch.device.side_effect = str
    torch.cuda.is_available.return_value = True
    torch.cuda.max_memory_allocated.return_value = 4 * 2**20
    torch.inference_mode.return_value = lambda fn: fn
    torch.Tensor = np.ndarray
    model = MagicMock()
    model.network = model
    model.to.return_value = model
    model.eval.return_value = model
    monkeypatch.setattr(throughput, "torch", torch)
    monkeypatch.setattr(throughput, "_TORCH_AVAILABLE", True)

    forward = benchmark_forward(model, input_shape=(1, 3, 4, 4), n_warmup=1, n_timed=1, device=device)
    end_to_end = benchmark_end_to_end(model, float, [1], n_warmup=1, device=device)
    decoded = benchmark_decoder(float, [1])

    assert forward.device == end_to_end.device == (device or "cpu")
    model.to.assert_any_call(device or "cpu")
    assert forward.error is None
    assert decoded.peak_gpu_mem_mib == 0
    torch.cuda.is_available.assert_not_called()
    if device == "cuda:1":
        assert forward.peak_gpu_mem_mib == end_to_end.peak_gpu_mem_mib == 4
        torch.cuda.synchronize.assert_called_with(device)
        torch.cuda.reset_peak_memory_stats.assert_called_with(device)
        torch.cuda.max_memory_allocated.assert_called_with(device)
        torch.cuda.empty_cache.assert_called_once_with()
    else:
        assert torch.cuda.mock_calls == []


def test_torch_benchmarks_reject_accelerator_requests_without_torch(monkeypatch):
    monkeypatch.setattr(throughput, "_TORCH_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="accelerator benchmarks require PyTorch"):
        benchmark_forward(_ToyModel(), input_shape=(1, 3, 4, 4), device="cuda:0")
    with pytest.raises(RuntimeError, match="accelerator benchmarks require PyTorch"):
        benchmark_end_to_end(_ToyModel(), float, [1], device="cuda:0")


def test_cli_only_synchronizes_an_initialized_torch_cuda_context(monkeypatch):
    from passage_eval.cli import _sync_runtime

    torch = MagicMock()
    torch.cuda.is_available.return_value = True
    monkeypatch.setitem(sys.modules, "torch", torch)

    torch.cuda.is_initialized.return_value = False
    _sync_runtime()
    torch.cuda.synchronize.assert_not_called()
    torch.cuda.is_available.assert_not_called()

    torch.cuda.is_initialized.return_value = True
    _sync_runtime()
    torch.cuda.synchronize.assert_called_once_with()
