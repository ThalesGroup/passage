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

"""Wall-clock inference benchmarking.

Three primitives:

    benchmark_forward       — bare neural network forward pass
                              (no data loader, no decoder).
    benchmark_decoder       — canonical decoder wall-clock
                              (A* recovery / constrained descent).
    benchmark_end_to_end    — forward + decoder, percentile latency.

All three accept a callable that runs one inference iteration and
return a ``BenchmarkRecord`` with the same fields used by
``inference_cost_matrix.csv`` in the paper:
    mean_latency_ms, p50_ms, p95_ms, p99_ms, throughput_sps,
    peak_gpu_mem_mib, params_M, n_warmup, n_timed.

The CUDA path is used iff ``torch`` is importable AND a CUDA device is
visible. CPU fallback is always available so that the package is usable
without a GPU.
"""

from __future__ import annotations

import gc
import time
from dataclasses import dataclass, asdict
from typing import Any, Callable, Sequence

import numpy as np

try:
    import torch  # type: ignore[import-not-found]

    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover - torch is an optional dependency
    torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


@dataclass(frozen=True)
class BenchmarkRecord:
    name: str
    n_warmup: int
    n_timed: int
    batch_size: int
    mean_latency_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    throughput_sps: float
    peak_gpu_mem_mib: float
    params_M: float | None
    determinism_flag: str
    device: str
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _is_cuda_available() -> bool:
    return _TORCH_AVAILABLE and torch.cuda.is_available()


def _sync() -> None:
    if _is_cuda_available():
        torch.cuda.synchronize()


def _reset_peak_memory() -> None:
    if _is_cuda_available():
        torch.cuda.reset_peak_memory_stats()


def _peak_gpu_mib() -> float:
    if _is_cuda_available():
        return float(torch.cuda.max_memory_allocated() / 2**20)
    return 0.0


def _percentile(samples: np.ndarray, p: float) -> float:
    if samples.size == 0:
        return float("nan")
    return float(np.percentile(samples, p))


def _params_in_millions(model: Any) -> float | None:
    if not _TORCH_AVAILABLE:
        return None
    try:
        net = getattr(model, "network", model)
        if not hasattr(net, "parameters"):
            return None
        n = sum(int(p.numel()) for p in net.parameters())
        return round(n / 1e6, 4)
    except Exception:
        return None


def _run_timing_loop(
    name: str,
    iter_fn: Callable[[], Any],
    *,
    n_warmup: int,
    n_timed: int,
    batch_size: int,
    determinism_flag: str,
    device: str,
    params_M: float | None,
) -> BenchmarkRecord:
    _reset_peak_memory()
    try:
        for _ in range(n_warmup):
            iter_fn()
        _sync()
        per_iter_ms = np.empty(n_timed, dtype=np.float64)
        for k in range(n_timed):
            t0 = time.perf_counter()
            iter_fn()
            _sync()
            per_iter_ms[k] = (time.perf_counter() - t0) * 1000.0
    except Exception as e:  # pragma: no cover - exercised in integration tests
        return BenchmarkRecord(
            name=name,
            n_warmup=n_warmup,
            n_timed=n_timed,
            batch_size=batch_size,
            mean_latency_ms=float("nan"),
            p50_ms=float("nan"),
            p95_ms=float("nan"),
            p99_ms=float("nan"),
            throughput_sps=float("nan"),
            peak_gpu_mem_mib=_peak_gpu_mib(),
            params_M=params_M,
            determinism_flag=determinism_flag,
            device=device,
            error=f"{type(e).__name__}: {str(e)[:200]}",
        )
    mean_ms = float(per_iter_ms.mean())
    throughput = float(batch_size * 1000.0 / mean_ms) if mean_ms > 0 else float("nan")
    return BenchmarkRecord(
        name=name,
        n_warmup=n_warmup,
        n_timed=n_timed,
        batch_size=batch_size,
        mean_latency_ms=round(mean_ms, 3),
        p50_ms=round(_percentile(per_iter_ms, 50.0), 3),
        p95_ms=round(_percentile(per_iter_ms, 95.0), 3),
        p99_ms=round(_percentile(per_iter_ms, 99.0), 3),
        throughput_sps=round(throughput, 3),
        peak_gpu_mem_mib=round(_peak_gpu_mib(), 1),
        params_M=params_M,
        determinism_flag=determinism_flag,
        device=device,
        error=None,
    )


def benchmark_forward(
    model: Any,
    *,
    input_shape: tuple[int, ...],
    name: str = "forward",
    n_warmup: int = 5,
    n_timed: int = 20,
    determinism_flag: str = "algorithmically_deterministic",
    device: str | None = None,
    dtype: str = "fp32",
) -> BenchmarkRecord:
    """Benchmark the bare forward pass of ``model`` on synthetic input.

    Parameters
    ----------
    model
        A torch ``nn.Module``-like callable (or a wrapper exposing
        ``network``). Plain Python callables also work — they will be
        treated as bare functions on ``numpy`` input when torch is not
        present.
    input_shape
        Shape of one synthetic input batch (e.g. ``(1, 3, 256, 256)``).
    dtype
        ``"fp32"`` (default), ``"fp16"`` or ``"bf16"``.
    """
    if not _TORCH_AVAILABLE:
        # No torch: degrade to plain timing of model(np.zeros(...)).
        input_array = np.zeros(input_shape, dtype=np.float32)

        def _iter() -> Any:
            return model(input_array)

        return _run_timing_loop(
            name=name,
            iter_fn=_iter,
            n_warmup=n_warmup,
            n_timed=n_timed,
            batch_size=int(input_shape[0]),
            determinism_flag=determinism_flag,
            device="cpu",
            params_M=None,
        )

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    network = getattr(model, "network", model)
    network = network.to(dev).eval()
    torch_dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}.get(dtype, torch.float32)
    input_tensor = torch.randn(*input_shape, device=dev, dtype=torch_dtype)
    if torch_dtype != torch.float32:
        network = network.to(torch_dtype)

    @torch.inference_mode()
    def _iter() -> Any:
        return network(input_tensor)

    rec = _run_timing_loop(
        name=name,
        iter_fn=_iter,
        n_warmup=n_warmup,
        n_timed=n_timed,
        batch_size=int(input_shape[0]),
        determinism_flag=determinism_flag,
        device=str(dev),
        params_M=_params_in_millions(model),
    )
    gc.collect()
    if _is_cuda_available():
        torch.cuda.empty_cache()
    return rec


def benchmark_decoder(
    decoder_fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    *,
    name: str = "decoder",
    n_warmup: int = 1,
    determinism_flag: str = "bit_exact",
) -> BenchmarkRecord:
    """Time the canonical decoder over a fixed batch of model outputs.

    The decoder is run once per element of ``inputs``. Latency is the
    mean per-sample wall-clock, and the percentile breakdown is computed
    over the per-sample times.
    """
    n = len(inputs)
    if n == 0:
        raise ValueError("benchmark_decoder requires at least one input.")
    # warmup
    for _ in range(n_warmup):
        for x in inputs[: min(n, 5)]:
            decoder_fn(x)
    per_sample_ms = np.empty(n, dtype=np.float64)
    for k, x in enumerate(inputs):
        t0 = time.perf_counter()
        decoder_fn(x)
        per_sample_ms[k] = (time.perf_counter() - t0) * 1000.0
    mean_ms = float(per_sample_ms.mean())
    return BenchmarkRecord(
        name=name,
        n_warmup=n_warmup,
        n_timed=n,
        batch_size=1,
        mean_latency_ms=round(mean_ms, 3),
        p50_ms=round(_percentile(per_sample_ms, 50.0), 3),
        p95_ms=round(_percentile(per_sample_ms, 95.0), 3),
        p99_ms=round(_percentile(per_sample_ms, 99.0), 3),
        throughput_sps=round(1000.0 / mean_ms, 3) if mean_ms > 0 else float("nan"),
        peak_gpu_mem_mib=_peak_gpu_mib(),
        params_M=None,
        determinism_flag=determinism_flag,
        device="cpu",
        error=None,
    )


def benchmark_end_to_end(
    model: Any,
    decoder_fn: Callable[[Any], Any],
    inputs: Sequence[Any],
    *,
    name: str = "end_to_end",
    n_warmup: int = 5,
    determinism_flag: str = "algorithmically_deterministic",
    device: str | None = None,
) -> BenchmarkRecord:
    """Forward + decoder per sample, returning percentile latency.

    Each ``inputs[i]`` is fed to ``model`` and the output to
    ``decoder_fn``. End-to-end time per sample = forward + decode. Use
    this to populate the ``Lat. (ms)`` column of the headline table.
    """
    n = len(inputs)
    if n == 0:
        raise ValueError("benchmark_end_to_end requires at least one input.")
    if _TORCH_AVAILABLE:
        dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        network = getattr(model, "network", model)
        network = network.to(dev).eval()
    else:
        network = model

    def _forward(x: Any) -> Any:
        if _TORCH_AVAILABLE and isinstance(x, torch.Tensor):
            with torch.inference_mode():
                return network(x.to(dev))  # noqa: F821 (dev defined when torch available)
        return network(x)

    # warmup
    for _ in range(n_warmup):
        for x in inputs[: min(n, 5)]:
            decoder_fn(_forward(x))
    _sync()
    _reset_peak_memory()
    per_sample_ms = np.empty(n, dtype=np.float64)
    for k, x in enumerate(inputs):
        t0 = time.perf_counter()
        out = _forward(x)
        decoder_fn(out)
        _sync()
        per_sample_ms[k] = (time.perf_counter() - t0) * 1000.0
    mean_ms = float(per_sample_ms.mean())
    return BenchmarkRecord(
        name=name,
        n_warmup=n_warmup,
        n_timed=n,
        batch_size=1,
        mean_latency_ms=round(mean_ms, 3),
        p50_ms=round(_percentile(per_sample_ms, 50.0), 3),
        p95_ms=round(_percentile(per_sample_ms, 95.0), 3),
        p99_ms=round(_percentile(per_sample_ms, 99.0), 3),
        throughput_sps=round(1000.0 / mean_ms, 3) if mean_ms > 0 else float("nan"),
        peak_gpu_mem_mib=round(_peak_gpu_mib(), 1),
        params_M=_params_in_millions(model),
        determinism_flag=determinism_flag,
        device=str(getattr(network, "device", "cpu")) if _TORCH_AVAILABLE else "cpu",
        error=None,
    )
